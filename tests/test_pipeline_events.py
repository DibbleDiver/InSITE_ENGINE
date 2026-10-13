"""Event detection on small synthetic frames: one test per kind, ids, precision on demand cycles."""
import numpy as np
import pandas as pd

from InSITE_Engine.contracts import ALL_SIGNALS, EVENT_COLUMNS, EVENT_KINDS, SEVERITIES, Dataset
from InSITE_Engine.events.events import detect_events
from InSITE_Engine.features.features import compute_features

DATASET = "TEST/synthetic"


def make_telemetry(days: int = 7, seed: int = 1) -> pd.DataFrame:
    """5-minute telemetry with daily demand cycles and AR(1) noise, no anomalies."""
    rng = np.random.default_rng(seed)
    n = days * 288
    t = pd.date_range("2026-10-01", periods=n, freq="5min", tz="UTC")
    hod = (t.hour + t.minute / 60.0).to_numpy()
    demand = 1.0 + 0.3 * np.cos(2 * np.pi * (hod - 13.5) / 24) + 0.2 * np.cos(4 * np.pi * (hod - 8) / 24)

    def ar1(sigma, phi=0.6):
        eps = rng.normal(0.0, sigma * np.sqrt(1 - phi * phi), n)
        out = np.empty(n)
        acc = 0.0
        for i in range(n):
            acc = phi * acc + eps[i]
            out[i] = acc
        return out

    return pd.DataFrame({
        "timestamp": t,
        "flow": 30.0 * demand + ar1(0.5),
        "pressure": 4.2 - 0.6 * (demand - 1.0) + ar1(0.012),
        "current": 7.0 + 1.4 * (demand - 1.0) + ar1(0.07),
    })


def events_for(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    return detect_events(compute_features(df, cfg=cfg), cfg, dataset_id=DATASET)


def test_spike_events_merge_consecutive_samples_and_report_the_peak():
    df = make_telemetry()
    t = df["timestamp"]
    df.loc[400, "flow"] += 8.0           # single-sample glitch
    df.loc[900:902, "flow"] -= 7.0       # three consecutive anomalous samples (during the overnight ramp)
    ev = events_for(df)
    assert list(ev.columns) == EVENT_COLUMNS
    assert set(ev["kind"]) <= set(EVENT_KINDS) and set(ev["severity"]) <= set(SEVERITIES)
    assert str(ev["start"].dtype) == "datetime64[ns, UTC]" and str(ev["end"].dtype) == "datetime64[ns, UTC]"

    high = ev[ev["kind"] == "spike_high"]
    assert len(high) == 1
    assert high["start"].iloc[0] == t[400] == high["end"].iloc[0]
    assert high["peak_value"].iloc[0] == df.loc[400, "flow"]
    assert high["score"].iloc[0] > 4.0 and "flow" in high["detail"].iloc[0]

    low = ev[ev["kind"] == "spike_low"]
    assert len(low) == 1
    assert low["start"].iloc[0] == t[900] and low["end"].iloc[0] == t[902]
    assert low["peak_value"].iloc[0] == df.loc[900:902, "flow"].min()
    assert ev[ev["signal"] != "flow"].empty, "the other signals carry no anomalies"


def test_step_change_yields_one_event_per_physical_step():
    df = make_telemetry()
    t = df["timestamp"]
    df.loc[1000:1039, "current"] = df.loc[1000:1039, "current"] - 3.0  # sustained shift down, then back up
    ev = events_for(df)
    steps = ev[(ev["kind"] == "step_change") & (ev["signal"] == "current")]
    assert steps["start"].tolist() == [t[1000], t[1040]]
    assert (steps["start"] == steps["end"]).all()  # point events
    assert "down" in steps["detail"].iloc[0] and "up" in steps["detail"].iloc[1]
    assert (steps["score"] > 3.0).all()
    assert ev[(ev["kind"] == "step_change") & (ev["signal"] != "current")].empty


def test_step_change_ignores_ramps_and_kinks():
    df = make_telemetry()
    n = len(df)
    level = 60.0 + np.random.default_rng(3).normal(0.0, 0.08, n)
    drain = np.interp(np.arange(n), [1000, 1036, 1108], [0.0, -15.0, 0.0])
    drain[:1000] = 0.0
    drain[1108:] = 0.0
    df["tank"] = level + drain  # continuous level, two abrupt changes of slope
    ev = events_for(df)
    assert ev[ev["signal"] == "tank"].empty, ev[ev["signal"] == "tank"]["detail"].tolist()
    assert ev[ev["kind"] == "step_change"].empty  # the demand cycles are ramps, not steps


def test_flatline_detection_rules():
    df = make_telemetry()
    t = df["timestamp"]
    df.loc[1500:1539, "current"] = 0.0       # 40 identical readings
    df.loc[300:304, "current"] = 5.0         # 5 identical readings: too short
    df["setpoint"] = 42.0                    # constant for the whole history
    ev = events_for(df)
    flat = ev[ev["kind"] == "flatline"]
    assert len(flat) == 1
    row = flat.iloc[0]
    assert row["signal"] == "current" and row["start"] == t[1500] and row["end"] == t[1539]
    assert row["peak_value"] == 0.0 and row["score"] == 40
    assert "stuck at 0.0" in row["detail"] and "3h20m" in row["detail"]
    assert ev[ev["signal"] == "setpoint"].empty


def test_gap_events_dataset_level_and_per_signal():
    df = make_telemetry()
    t = df["timestamp"]
    hole = (t >= "2026-10-04 10:00") & (t < "2026-10-04 12:00")
    df = df[~hole].reset_index(drop=True)
    df.loc[1700:1711, "flow"] = np.nan      # 12 missing readings = 1 hour
    df.loc[200:201, "pressure"] = np.nan    # 2 missing readings: below min_gap_minutes
    ev = events_for(df)
    gaps = ev[ev["kind"] == "gap"]
    whole = gaps[gaps["signal"] == ALL_SIGNALS]
    assert len(whole) == 1
    assert whole["start"].iloc[0] == pd.Timestamp("2026-10-04 10:00", tz="UTC")
    assert whole["end"].iloc[0] == pd.Timestamp("2026-10-04 11:55", tz="UTC")
    assert whole["score"].iloc[0] == 24 and whole["severity"].iloc[0] == "high"
    assert np.isnan(whole["peak_value"].iloc[0])
    per_signal = gaps[gaps["signal"] != ALL_SIGNALS]
    assert per_signal["signal"].tolist() == ["flow"]
    assert per_signal["start"].iloc[0] == df.loc[1700, "timestamp"] and per_signal["end"].iloc[0] == df.loc[1711, "timestamp"]
    assert per_signal["score"].iloc[0] == 12
    assert ev[ev["kind"] == "step_change"].empty, "no steps may be invented across the hole"


def test_event_ids_are_deterministic_and_unique():
    df = make_telemetry()
    df.loc[400, "flow"] += 8.0
    df.loc[1000:1039, "current"] = 0.0
    first = events_for(df)
    second = events_for(df.copy(deep=True))
    assert len(first) >= 3
    assert first["event_id"].tolist() == second["event_id"].tolist()
    assert first["event_id"].is_unique
    assert first["event_id"].str.len().eq(12).all()
    assert first["event_id"].iloc[0] == detect_events(compute_features(df), dataset_id=DATASET)["event_id"].iloc[0]


def test_text_and_irregular_datasets_produce_no_numeric_events():
    orders = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-10-01 09:00", "2026-10-01 09:00", "2026-10-02 15:30", "2026-10-04 08:10"], utc=True),
        "work_order": [1, 2, 3, 4],
        "description": ["a", "b", "c", "d"],
    })
    ds = Dataset(id="AX/orders", source="AX", name="orders", path="x", frame=orders, signals=[], time_column="timestamp")
    ev = detect_events(ds)
    assert ev.empty and list(ev.columns) == EVENT_COLUMNS

    irregular = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-10-01 09:00", "2026-10-01 09:01", "2026-10-01 12:30", "2026-10-02 08:10", "2026-10-03 23:59"], utc=True),
        "hours": [1.0, 20.0, 0.5, 30.0, 2.0],
    })
    assert detect_events(irregular).empty

    assert detect_events(pd.DataFrame({"a": [1.0, 2.0]})).empty  # no time axis at all


def test_daily_demand_cycles_do_not_produce_a_storm():
    ev = events_for(make_telemetry(seed=5))
    assert ev[ev["kind"].isin(["step_change", "flatline", "gap"])].empty
    assert len(ev) <= 1  # at most the Gaussian tail of a single low-severity spike
    if len(ev):
        assert ev["severity"].iloc[0] == "low"
