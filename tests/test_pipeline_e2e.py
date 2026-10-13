"""End to end: generate the demo site into a temp dir, run the pipeline, check the scenarios."""
import filecmp
import json
import os
import time

import pandas as pd
import yaml

from InSITE_Engine.contracts import ALL_SIGNALS, EVENT_COLUMNS, STATE_COLUMNS
from InSITE_Engine.demo.generate import GROUND_TRUTH_FILE, generate_demo_data
from InSITE_Engine.pipeline import run_pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "InSITE_Engine", "config", "config.yaml")


def _config(input_path: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["input_path"] = input_path
    return cfg


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_demo_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    truth = generate_demo_data(str(raw))
    assert (tmp_path / GROUND_TRUTH_FILE).exists()
    assert not (raw / GROUND_TRUTH_FILE).exists()
    with open(tmp_path / GROUND_TRUTH_FILE, encoding="utf-8") as fh:
        assert {s["id"] for s in json.load(fh)["scenarios"]} == {
            "pump_trip", "stuck_meter", "radio_outage", "sensor_drift", "leak", "glitch_spikes"}

    started = time.perf_counter()
    result = run_pipeline(_config(str(raw)))
    assert time.perf_counter() - started < 10.0

    datasets = result.datasets
    assert set(datasets) == {"TRIO/telemetry_site_a", "TCL/tank_pumps", "Meters/zone_flow", "AX/work_orders"}

    ax = datasets["AX/work_orders"]
    assert ax.time_column == "timestamp" and ax.signals == []
    assert {"asset", "category", "priority", "description"} <= set(ax.text_columns())
    assert len(ax.frame) == 2 + truth["routine_work_orders"]
    assert ax.frame["description"].str.contains("P2 tripped on overload").any()

    tcl = datasets["TCL/tank_pumps"]
    assert tcl.signals == ["tank_level_a_pct", "tank_level_b_pct", "p1_current_a", "p2_current_a"]
    assert str(tcl.frame["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert tcl.frame["timestamp"].is_monotonic_increasing and tcl.frame["timestamp"].is_unique
    assert tcl.frame["timestamp"].iloc[-1] == _ts("2026-10-07 23:55")  # day-first dates read correctly
    mess = truth["messiness"]["TCL/tank_pumps"]
    for time_text, column in mess["blank_cells"]:
        when = pd.to_datetime(time_text, dayfirst=True).tz_localize("UTC")
        row = tcl.frame[tcl.frame["timestamp"] == when]
        assert len(row) == 1
        assert pd.isna(row[column.lower().replace(" (%)", "_pct").replace(" (a)", "_a").replace(" ", "_")].iloc[0])
    assert (tcl.frame[tcl.signals] == 0).sum().sum() == 36  # only the real pump trip reads zero
    assert "p2_current_a__z" in tcl.frame.columns

    events = result.events
    assert list(events.columns) == EVENT_COLUMNS
    assert events["start"].is_monotonic_increasing
    assert events["event_id"].is_unique

    p2 = events[(events["dataset"] == "TCL/tank_pumps") & (events["signal"] == "p2_current_a")]
    flat = p2[p2["kind"] == "flatline"]
    assert len(flat) == 1
    assert flat["start"].iloc[0] == _ts("2026-10-03 14:00") and flat["end"].iloc[0] == _ts("2026-10-03 16:55")
    assert flat["peak_value"].iloc[0] == 0.0 and "stuck at 0.0" in flat["detail"].iloc[0]
    assert p2[p2["kind"] == "step_change"]["start"].tolist() == [_ts("2026-10-03 14:00"), _ts("2026-10-03 17:00")]

    outlet = events[(events["signal"] == "outlet_pressure_bar") & (events["kind"] == "step_change")]
    assert outlet["start"].tolist() == [_ts("2026-10-03 14:10"), _ts("2026-10-03 17:10")]
    flow_steps = events[(events["signal"] == "zone_flow_lps") & (events["kind"] == "step_change")]
    assert len(flow_steps) == 2
    assert abs(flow_steps["start"].iloc[0] - _ts("2026-10-03 14:15")) <= pd.Timedelta(minutes=10)

    gaps = events[(events["dataset"] == "TRIO/telemetry_site_a") & (events["kind"] == "gap")]
    assert len(gaps) == 1 and gaps["signal"].iloc[0] == ALL_SIGNALS
    gap = gaps.iloc[0]
    assert gap["start"] == _ts("2026-10-04 10:00") and gap["end"] == _ts("2026-10-04 11:55")
    assert 100 <= (gap["end"] - gap["start"]).total_seconds() / 60 + 5 <= 140
    assert gap["score"] == 24 and gap["severity"] == "high"

    states = result.states
    assert list(states.columns) == STATE_COLUMNS
    trio = datasets["TRIO/telemetry_site_a"]
    for sig in trio.signals:
        offline = states[(states["dataset"] == trio.id) & (states["signal"] == sig) & (states["state"] == "offline")]
        assert len(offline) == 1
        assert offline["start"].iloc[0] == _ts("2026-10-04 10:00") and offline["end"].iloc[0] == _ts("2026-10-04 12:00")
        assert offline["duration_minutes"].iloc[0] == 120.0

    bulk = events[(events["signal"] == "customer_bulk_m3h") & (events["kind"] == "flatline")]
    assert len(bulk) == 1
    assert bulk["start"].iloc[0] == _ts("2026-10-05 10:00") and bulk["end"].iloc[0] == _ts("2026-10-05 17:55")
    assert bulk["score"].iloc[0] == 96 and bulk["severity"].iloc[0] == "high"
    stuck_value = truth["scenarios"][1]["effects"][0]["stuck_value"]
    assert bulk["peak_value"].iloc[0] == stuck_value

    inlet = events[(events["signal"] == "inlet_pressure_bar") & (events["kind"].isin(["spike_high", "spike_low"]))]
    assert len(inlet) >= 1
    assert _ts("2026-10-02 09:35") in set(inlet["start"]) or _ts("2026-10-06 03:15") in set(inlet["start"])

    # Precision: the trends left for the Mind and the quiet signals must stay quiet.
    assert events[events["signal"] == "tank_level_b_pct"].empty  # drift only
    assert events[(events["signal"] == "zone_flow_lps") & (events["kind"] != "step_change")].empty  # leak is a trend
    assert events[events["kind"].isin(["step_change", "flatline", "gap"]) & ~events["signal"].isin(
        ["p2_current_a", "outlet_pressure_bar", "zone_flow_lps", "customer_bulk_m3h", ALL_SIGNALS])].empty
    assert len(events) <= 14

    # Determinism of ids across runs and a JSON-serialisable summary.
    again = run_pipeline(_config(str(raw)))
    assert again.events["event_id"].tolist() == events["event_id"].tolist()
    json.dumps(result.summary())
    written = result.save(str(tmp_path / "out"))
    assert all(os.path.exists(p) for p in written)


def test_demo_generation_is_deterministic(tmp_path):
    first = generate_demo_data(str(tmp_path / "a" / "raw"), days=2, seed=3)
    second = generate_demo_data(str(tmp_path / "b" / "raw"), days=2, seed=3)
    for ds_id in first["files"]:
        assert filecmp.cmp(first["files"][ds_id], second["files"][ds_id], shallow=False)
    other = generate_demo_data(str(tmp_path / "c" / "raw"), days=2, seed=4)
    assert not filecmp.cmp(first["files"]["TRIO/telemetry_site_a"], other["files"]["TRIO/telemetry_site_a"], shallow=False)
    assert {s["id"] for s in first["scenarios"]} == {"sensor_drift", "glitch_spikes"}  # only what fits in 2 days


def test_missing_input_path_yields_no_datasets(tmp_path):
    result = run_pipeline({"input_path": str(tmp_path / "nowhere"), "pipeline": {}})
    assert result.datasets == {}
    assert result.events.empty and list(result.events.columns) == EVENT_COLUMNS
    assert result.states.empty and list(result.states.columns) == STATE_COLUMNS
    json.dumps(result.summary())
