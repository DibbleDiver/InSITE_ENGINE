"""States: segments tile each dataset's span, precedence, events-only mode."""
import numpy as np
import pandas as pd

from InSITE_Engine.contracts import ALL_SIGNALS, STATE_COLUMNS, STATES, Dataset, empty_events
from InSITE_Engine.events.events import detect_events
from InSITE_Engine.features.features import compute_features
from InSITE_Engine.states.states import derive_states
from tests.test_pipeline_events import DATASET, make_telemetry


def _dataset(df: pd.DataFrame) -> Dataset:
    featured = compute_features(df)
    return Dataset(id=DATASET, source="TEST", name="synthetic", path="x", frame=featured,
                   signals=["flow", "pressure", "current"], time_column="timestamp")


def _assert_tiles_span(states: pd.DataFrame, t_min: pd.Timestamp, t_max: pd.Timestamp, signals: list[str]) -> None:
    assert list(states.columns) == STATE_COLUMNS
    assert set(states["state"]) <= set(STATES)
    assert sorted(states["signal"].unique()) == sorted(signals)
    for _, group in states.groupby(["dataset", "signal"]):
        group = group.sort_values("start")
        assert group["start"].iloc[0] == t_min
        assert group["end"].iloc[-1] == t_max
        assert (group["end"] > group["start"]).all()
        assert (group["start"].iloc[1:].to_numpy() == group["end"].iloc[:-1].to_numpy()).all()
        assert np.isclose(group["duration_minutes"].sum(), (t_max - t_min).total_seconds() / 60)
        assert group["duration_minutes"].dtype.kind == "f"


def test_states_tile_the_span_with_the_right_states():
    df = make_telemetry()
    t = df["timestamp"]
    df.loc[400, "flow"] += 8.0
    df.loc[1000:1039, "current"] = 0.0
    hole = (t >= "2026-10-04 10:00") & (t < "2026-10-04 12:00")
    df = df[~hole].reset_index(drop=True)
    ds = _dataset(df)
    events = detect_events(ds)
    states = derive_states(events, {ds.id: ds})
    t_min, t_max = ds.time_range()
    _assert_tiles_span(states, t_min, t_max, ds.signals)

    def state_at(signal: str, when: str) -> str:
        ts = pd.Timestamp(when, tz="UTC")
        rows = states[(states["signal"] == signal) & (states["start"] <= ts) & (states["end"] > ts)]
        assert len(rows) == 1, rows
        return rows["state"].iloc[0]

    for sig in ds.signals:  # the dataset-level gap takes every signal offline
        assert state_at(sig, "2026-10-04 11:00") == "offline"
        assert state_at(sig, "2026-10-04 09:55") != "offline"
    assert state_at("flow", "2026-10-02 09:20") == "anomalous"   # the spike sample (index 400)
    assert state_at("flow", "2026-10-02 09:25") == "normal"
    # The zeros at original indices 1000-1039 straddle the hole: the flat stretch runs 12:00-14:35, the step up is at 14:40.
    assert state_at("current", "2026-10-04 12:00") == "flatlined"
    assert state_at("current", "2026-10-04 13:30") == "flatlined"
    assert state_at("current", "2026-10-04 14:40") == "anomalous"  # a point event occupies one interval
    assert state_at("current", "2026-10-04 14:45") == "normal"
    assert state_at("pressure", "2026-10-02 12:00") == "normal"


def test_state_precedence_offline_over_flatlined_over_anomalous():
    t = pd.date_range("2026-10-01 00:00", periods=288, freq="5min", tz="UTC")
    frame = pd.DataFrame({"timestamp": t, "a": 1.0, "b": 2.0})
    ds = Dataset(id="D/x", source="D", name="x", path="x", frame=frame, signals=["a", "b"], time_column="timestamp")

    def ev(signal, kind, start, end):
        return {"event_id": f"{signal}{kind}{start}", "dataset": "D/x", "source": "D", "signal": signal, "kind": kind,
                "severity": "low", "start": pd.Timestamp(start, tz="UTC"), "end": pd.Timestamp(end, tz="UTC"),
                "peak_value": np.nan, "score": 1.0, "detail": ""}

    events = pd.DataFrame([
        ev(ALL_SIGNALS, "gap", "2026-10-01 10:00", "2026-10-01 11:55"),
        ev("a", "flatline", "2026-10-01 09:00", "2026-10-01 12:55"),
        ev("a", "spike_high", "2026-10-01 09:30", "2026-10-01 09:30"),
        ev("a", "step_change", "2026-10-01 15:00", "2026-10-01 15:00"),
    ])
    states = derive_states(events, {ds.id: ds})
    _assert_tiles_span(states, t[0], t[-1], ["a", "b"])
    a = states[states["signal"] == "a"].sort_values("start")
    expected = [
        ("normal", "2026-10-01 00:00", "2026-10-01 09:00"),
        ("flatlined", "2026-10-01 09:00", "2026-10-01 10:00"),   # the spike inside the flatline is outranked
        ("offline", "2026-10-01 10:00", "2026-10-01 12:00"),     # the gap outranks the flatline
        ("flatlined", "2026-10-01 12:00", "2026-10-01 13:00"),   # end + one interval
        ("normal", "2026-10-01 13:00", "2026-10-01 15:00"),
        ("anomalous", "2026-10-01 15:00", "2026-10-01 15:05"),   # a point event occupies one interval
        ("normal", "2026-10-01 15:05", "2026-10-01 23:55"),
    ]
    assert [(r.state, str(r.start)[:16], str(r.end)[:16]) for r in a.itertuples()] == expected
    b = states[states["signal"] == "b"].sort_values("start")
    assert b["state"].tolist() == ["normal", "offline", "normal"]


def test_derive_states_works_events_only_and_on_empty_input():
    df = make_telemetry()
    df.loc[1000:1039, "current"] = 0.0
    events = detect_events(compute_features(df), dataset_id=DATASET)
    states = derive_states(events)  # legacy main.py call shape: events only
    assert list(states.columns) == STATE_COLUMNS and not states.empty
    assert states["start"].min() == events["start"].min()
    assert states["end"].max() == events["end"].max()
    assert "flatlined" in set(states["state"])

    empty = derive_states(empty_events(), {})
    assert list(empty.columns) == STATE_COLUMNS and empty.empty
