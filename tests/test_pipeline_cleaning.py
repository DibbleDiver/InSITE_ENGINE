"""Cleaning and normalisation: headers, numeric coercion, day-first dates, time axis, signals."""
import numpy as np
import pandas as pd

from InSITE_Engine.cleaning.clean import basic_clean, normalise_header
from InSITE_Engine.normalisation.normalise import (
    TIME_COLUMN,
    normalise,
    parse_timestamps,
    pipeline_config,
    select_signals,
)


def test_header_normalisation_to_snake_case():
    assert normalise_header("Tank Level A (%)") == "tank_level_a_pct"
    assert normalise_header("P2 Current (A)") == "p2_current_a"
    assert normalise_header("  DateTime ") == "datetime"
    assert normalise_header("Flow (l/s)") == "flow_l_s"
    df = pd.DataFrame({"Tank Level A (%)": [1.0], "P2 Current (A)": [2.0], " Time ": ["x"], "Flow": [3.0], "flow": [4.0]})
    assert list(basic_clean(df).columns) == ["tank_level_a_pct", "p2_current_a", "time", "flow", "flow_2"]


def test_numeric_text_is_coerced_and_gaps_are_never_zero_filled():
    n = 20
    current = [f"{7 + 0.01 * i:.2f}" for i in range(n)]
    current[0], current[1], current[2], current[3], current[4] = "12.5", " 7 ", "", "ERR", "1,250.5"
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01", periods=n, freq="5min").strftime("%Y-%m-%d %H:%M"),
        "p2_current_a": current,
        "pct": [f"{12 + i}%" for i in range(n)],
        "note": ["a", "b", ""] + ["x"] * (n - 3),
    })
    out = basic_clean(df)
    assert out["p2_current_a"].dtype.kind == "f"
    assert out["p2_current_a"].tolist()[:2] == [12.5, 7.0]
    assert out["p2_current_a"].isna().tolist()[:5] == [False, False, True, True, False]
    assert out["p2_current_a"].iloc[4] == 1250.5
    assert out["pct"].tolist()[:3] == [12.0, 13.0, 14.0]
    assert (out["p2_current_a"] == 0).sum() == 0  # the old fillna(0) must be gone
    assert out["note"].isna().tolist()[:3] == [False, False, True]
    assert not pd.api.types.is_numeric_dtype(out["note"])


def test_day_first_dates_are_decided_from_the_data():
    parsed, info = parse_timestamps(pd.Series(["13/10/2026 14:05", "13/10/2026 14:10", "14/10/2026 09:00"]))
    assert info["dayfirst"] is True
    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2026-10-13 14:05", tz="UTC")

    parsed, info = parse_timestamps(pd.Series(["10/13/2026 14:05", "10/14/2026 14:05"]))
    assert info["dayfirst"] is False
    assert parsed.iloc[1] == pd.Timestamp("2026-10-14 14:05", tz="UTC")

    # Every field <= 12 (1-7 October): the reading that gives a 7-day span beats the one giving 6 months.
    hourly = pd.date_range("2026-10-01", periods=7 * 24, freq="h")
    parsed, info = parse_timestamps(pd.Series(hourly.strftime("%d/%m/%Y %H:%M")))
    assert info["dayfirst"] is True
    assert parsed.iloc[-1] == pd.Timestamp("2026-10-07 23:00", tz="UTC")
    assert parsed.is_monotonic_increasing


def test_normalise_sorts_drops_rowless_timestamps_and_keeps_last_duplicate():
    df = pd.DataFrame({
        "Time": ["01/10/2026 00:10", "01/10/2026 00:00", "01/10/2026 00:05", "01/10/2026 00:05", "", "01/10/2026 00:10"],
        "Level (%)": [3.0, 1.0, 2.0, 2.5, 9.0, 3.0],
    })
    out = normalise(basic_clean(df), {"pipeline": {"assume_timezone": "UTC"}})
    assert str(out[TIME_COLUMN].dtype) == "datetime64[ns, UTC]"
    assert out[TIME_COLUMN].tolist() == [pd.Timestamp(f"2026-10-01 00:{m:02d}", tz="UTC") for m in (0, 5, 10)]
    assert out["level_pct"].tolist() == [1.0, 2.5, 3.0]  # exact duplicate dropped, re-sent 00:05 keeps last


def test_naive_timestamps_use_assume_timezone_and_offsets_are_converted():
    naive = pd.DataFrame({"timestamp": ["2026-07-01 12:00:00", "2026-07-01 12:05:00"], "x": [1.0, 2.0]})
    out = normalise(basic_clean(naive), {"pipeline": {"assume_timezone": "Europe/London"}})
    assert out[TIME_COLUMN].iloc[0] == pd.Timestamp("2026-07-01 11:00", tz="UTC")  # BST -> UTC
    aware = pd.DataFrame({"timestamp": ["2026-07-01T12:00:00+02:00", "2026-07-01T12:05:00Z"], "x": [1.0, 2.0]})
    out = normalise(basic_clean(aware), {"pipeline": {"assume_timezone": "Europe/London"}})
    assert out[TIME_COLUMN].tolist() == [pd.Timestamp("2026-07-01 10:00", tz="UTC"), pd.Timestamp("2026-07-01 12:05", tz="UTC")]


def test_time_column_found_by_content_and_signals_exclude_ids_and_empty_columns():
    df = pd.DataFrame({
        "when": ["2026-10-01 00:00:00", "2026-10-01 00:05:00", "2026-10-01 00:10:00", "2026-10-01 00:15:00", "2026-10-01 00:20:00"],
        "work_order": [1001, 1002, 1003, 1004, 1005],
        "flow": [1.0, 2.0, 3.0, 4.0, 5.0],  # too short and small to be mistaken for a counter
        "empty": [np.nan] * 5,
        "label": list("abcde"),
    })
    out = normalise(basic_clean(df))
    assert TIME_COLUMN in out.columns and "when" not in out.columns
    assert select_signals(out) == ["flow"]
    counter = pd.DataFrame({
        "row": range(1, 41),                   # id-like name
        "seq": np.arange(100, 140),            # strictly increasing, compact: a counter
        "level": np.linspace(1.0, 2.0, 40),
        "count": np.arange(40) * 3,            # increasing but not compact: a measurement
    })
    assert select_signals(counter, None) == ["level", "count"]


def test_text_dataset_passes_through_with_its_rows_and_no_signals():
    df = pd.DataFrame({
        "created_at": ["2026-10-01 09:00:00", "2026-10-01 09:00:00", "2026-10-02 10:30:00"],
        "work_order": [100231, 100232, 100233],
        "asset": ["Pump P2", "Tank A", "Zone 3"],
        "description": ["P2 tripped on overload, reset by operator", "Routine inspection", "Customer complaint"],
    })
    out = normalise(basic_clean(df))
    assert len(out) == 3  # same-minute work orders are not collapsed
    assert TIME_COLUMN in out.columns
    assert select_signals(out) == []
    assert out["description"].iloc[0] == "P2 tripped on overload, reset by operator"


def test_pipeline_config_defaults_and_overrides():
    defaults = pipeline_config(None)
    assert defaults["spike_z"] == 4.0 and defaults["assume_timezone"] == "UTC"
    merged = pipeline_config({"pipeline": {"spike_z": 5.5, "min_gap_minutes": None}})
    assert merged["spike_z"] == 5.5 and merged["min_gap_minutes"] == 30
    assert pipeline_config({"spike_z": 6.0})["spike_z"] == 6.0
