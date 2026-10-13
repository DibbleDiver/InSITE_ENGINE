"""Data tools, input validation and evidence tagging, on the hand-built fixture."""
from __future__ import annotations

import json

import pytest

from InSITE_Engine.mind.evidence import EvidenceLedger
from InSITE_Engine.mind.memory import MemoryStore, StagedChanges
from InSITE_Engine.mind.tools import ToolRegistry, build_tool_definitions, tool_names
from InSITE_Engine.mind.tools.data import MAX_RESULT_CHARS, DataTools, ToolError
from tests.test_mind_fixtures import build_result


@pytest.fixture(scope="module")
def result():
    return build_result()


@pytest.fixture
def tools(result):
    return DataTools(result, new_event_ids={"ev_flow_dip"})


def test_list_datasets_inventory(tools):
    out = tools.list_datasets({})
    assert "4 datasets" in out
    assert "TRIO/pressure [TRIO] rows=2016" in out and "interval=5min" in out
    assert "Meters/site_flow" in out and "interval=15min" in out
    assert "AX/workorders [AX] rows=3 (no time axis)" in out
    assert "text columns: wo_id, created, asset, description, status" in out


def test_describe_signal_stats_trend_state_events(tools):
    out = tools.describe_signal({"dataset": "TRIO/pressure", "signal": "pressure"})
    assert "samples=2016 missing=0%" in out and "interval=5min" in out
    assert "min=3.88" in out and "max=5.25" in out
    assert "trend: slope=" in out and "over 7 days" in out
    assert "latest state: normal since 2026-01-03T10:10Z" in out
    assert "events on this signal: step_change=1" in out and "ev_pressure_step" in out
    # feature columns are reachable too
    assert "samples=" in tools.describe_signal({"dataset": "TRIO/pressure", "signal": "pressure__z"})


def test_describe_signal_unknown_signal_lists_options(tools):
    with pytest.raises(ToolError, match="signals: pressure, flow"):
        tools.describe_signal({"dataset": "TRIO/pressure", "signal": "nope"})
    with pytest.raises(ToolError, match="unknown dataset"):
        tools.describe_signal({"dataset": "TRIO/nope", "signal": "pressure"})


def test_get_series_window_and_cross_dataset_alignment(tools):
    out = tools.get_series({
        "signals": [{"dataset": "TCL/pumps", "signal": "pump1_current"}, {"dataset": "TRIO/pressure", "signal": "pressure"}],
        "start": "2026-01-03T03:50Z", "end": "2026-01-03T04:15Z",
    })
    lines = out.splitlines()
    assert lines[0].startswith("6 rows from 2026-01-03T03:50Z to 2026-01-03T04:15Z")
    assert lines[1] == "time | TCL/pumps:pump1_current | TRIO/pressure:pressure"
    row_0400 = next(line for line in lines if line.startswith("2026-01-03T04:00Z"))
    assert row_0400.split(" | ")[1] == "0"           # pump already tripped
    assert float(row_0400.split(" | ")[2]) > 5.0     # pressure not yet dropped
    row_0410 = next(line for line in lines if line.startswith("2026-01-03T04:10Z"))
    assert float(row_0410.split(" | ")[2]) < 4.1     # pressure dropped ten minutes later


def test_get_series_auto_downsamples_to_max_points(tools):
    out = tools.get_series({"signals": [{"dataset": "Meters/site_flow", "signal": "site_flow"}], "max_points": 20})
    header = out.splitlines()[0]
    assert "downsampled to 12h (mean) from 672 raw points" in header
    assert len(out.splitlines()) - 2 <= 20
    out = tools.get_series({"signals": [{"dataset": "Meters/site_flow", "signal": "site_flow"}], "resample": "1D", "agg": "min"})
    assert "resampled to 1D (min)" in out and len(out.splitlines()) == 2 + 7


def test_query_rows_on_text_dataset(tools):
    out = tools.query_rows({"dataset": "AX/workorders",
                            "filters": [{"column": "description", "op": "contains", "value": "PUMP 1"}]})
    assert "1 of 3 rows match" in out and "WO-1001" in out
    out = tools.query_rows({"dataset": "AX/workorders", "filters": [{"column": "priority", "op": "in", "value": [1, 2]}],
                            "sort_by": "priority", "descending": True, "columns": ["wo_id", "priority"]})
    rows = out.splitlines()[2:]
    assert rows == ["WO-1002 | 2", "WO-1001 | 1"]
    out = tools.query_rows({"dataset": "AX/workorders", "filters": [{"column": "status", "op": "==", "value": "OPEN"}],
                            "start": "2026-01-01T00:00Z"})
    assert "1 of 3 rows match" in out and "start/end ignored" in out


def test_query_rows_numeric_and_time_filters(tools):
    out = tools.query_rows({"dataset": "TRIO/pressure", "filters": [{"column": "pressure", "op": "<", "value": 4.0}],
                            "start": "2026-01-03T00:00Z", "end": "2026-01-03T05:00Z", "columns": ["timestamp", "pressure"], "limit": 3})
    lines = out.splitlines()
    assert lines[0].startswith("TRIO/pressure: ") and "showing 3" in lines[0]
    assert lines[2].startswith("2026-01-03T04:10Z")
    with pytest.raises(ToolError, match="unknown column"):
        tools.query_rows({"dataset": "TRIO/pressure", "filters": [{"column": "nope", "op": "==", "value": 1}]})
    with pytest.raises(ToolError, match="not a number"):
        tools.query_rows({"dataset": "TRIO/pressure", "filters": [{"column": "pressure", "op": ">", "value": "abc"}]})


def test_get_events_filters_sort_and_new_marker(tools):
    out = tools.get_events({"severity": "high"})
    assert "2 of 4 events match" in out and "ev_pump_trip" in out and "ev_pressure_step" in out
    out = tools.get_events({"sort": "score", "limit": 1})
    assert out.splitlines()[2].startswith("ev_pump_trip")
    out = tools.get_events({"dataset": "Meters/site_flow"})
    assert "ev_flow_dip | NEW" in out and "1 marked NEW" in out
    out = tools.get_events({"start": "2026-01-05T00:00Z"})
    assert "1 of 4 events match" in out and "ev_trio_gap" in out


def test_get_states_current_only(tools):
    out = tools.get_states({"current_only": True})
    assert "3 state intervals match" in out and "current state of each signal" in out
    out = tools.get_states({"state": "anomalous"})
    assert "2 state intervals match" in out


def test_correlate_cross_dataset_finds_known_lag(tools):
    out = tools.correlate({"a": {"dataset": "TCL/pumps", "signal": "pump1_current"},
                           "b": {"dataset": "TRIO/pressure", "signal": "pressure"}, "mode": "diff"})
    assert "grid=5min, mode=diff" in out
    assert "best lag = +2 steps (10 min): TCL/pumps:pump1_current LEADS TRIO/pressure:pressure" in out
    assert "strong positive linear association" in out
    # swapping the signals swaps the lead
    out = tools.correlate({"a": {"dataset": "TRIO/pressure", "signal": "pressure"},
                           "b": {"dataset": "TCL/pumps", "signal": "pump1_current"}, "mode": "diff"})
    assert "best lag = -2 steps (10 min): TCL/pumps:pump1_current LEADS TRIO/pressure:pressure" in out
    with pytest.raises(ToolError, match="constant"):
        tools.correlate({"a": {"dataset": "TCL/pumps", "signal": "pump1_current"},
                         "b": {"dataset": "TRIO/pressure", "signal": "pressure"},
                         "start": "2026-01-03T05:00Z", "end": "2026-01-03T09:00Z"})


def test_profile_night_minimum_per_day_shows_rise(tools):
    out = tools.profile({"dataset": "Meters/site_flow", "signal": "site_flow", "group_by": "day", "stat": "min",
                         "hours": {"from": 1, "to": 4}})
    lines = out.splitlines()
    assert "hours 01-04 UTC inclusive" in lines[0] and "7 groups" in lines[0]
    rows = [line.split(" | ") for line in lines[2:9]]
    assert [r[0] for r in rows] == [f"2026-01-0{d}" for d in range(1, 8)]
    assert all(r[2] == "16" for r in rows)  # 16 quarter-hour samples per night
    mins = [float(r[1]) for r in rows]
    assert mins[0] < mins[1] < mins[3] < mins[4] < mins[5] < mins[6]  # rising (day 3 dips because of the trip)
    assert "trend of the daily min:" in out
    out = tools.profile({"dataset": "TRIO/pressure", "signal": "pressure", "group_by": "hour", "stat": "mean"})
    assert "24 groups" in out and out.splitlines()[2].startswith("00 | ")
    out = tools.profile({"dataset": "TRIO/pressure", "signal": "pressure", "group_by": "weekday", "stat": "count",
                         "hours": {"from": 22, "to": 2}})
    assert "Mon" in out and "hours 22-02" in out


def test_compare_windows_quantifies_shift(tools):
    out = tools.compare_windows({"dataset": "TRIO/pressure", "signal": "pressure",
                                 "window_a": {"start": "2026-01-02T04:10Z", "end": "2026-01-02T10:10Z"},
                                 "window_b": {"start": "2026-01-03T04:10Z", "end": "2026-01-03T10:10Z"}})
    assert "window A" in out and "window B" in out and "n=73" in out
    assert "change in mean B-A = -1.1" in out and "large effect" in out


# ------------------------------------------------------------------ registry


@pytest.fixture
def registry(result, tmp_path):
    store = MemoryStore(str(tmp_path / "mind"))
    staged = StagedChanges(store.state, cycle=1)
    return ToolRegistry(result, staged, store, EvidenceLedger(), new_event_ids={"ev_flow_dip"})


def test_registry_rejects_invalid_input_with_is_error(registry):
    out = registry.execute("describe_signal", {"dataset": "TRIO/pressure"})  # missing signal
    assert out.is_error and out.evidence_id is None
    payload = json.loads(out.content)
    assert json.loads(payload["INVALID_JSON"]) == {"dataset": "TRIO/pressure"}
    assert "signal" in payload["error"]
    out = registry.execute("get_events", {"severity": "critical"})
    assert out.is_error and "severity" in json.loads(out.content)["error"]
    out = registry.execute("get_events", "not an object")
    assert out.is_error and "JSON object" in json.loads(out.content)["error"]
    out = registry.execute("no_such_tool", {})
    assert out.is_error and "unknown tool" in json.loads(out.content)["error"]
    assert len(registry.ledger) == 0


def test_registry_tags_data_results_with_evidence_ids(registry):
    a = registry.execute("list_datasets", {}, step=1)
    b = registry.execute("get_events", {"severity": "high"}, step=1)
    assert (a.evidence_id, b.evidence_id) == ("E1", "E2")
    assert a.content.startswith("E1 = list_datasets({})\n")
    assert b.content.startswith("E2 = get_events(")
    assert registry.ledger.items["E2"].tool == "get_events"
    assert registry.ledger.items["E2"].input == {"severity": "high"}
    bad = registry.execute("describe_signal", {"dataset": "TRIO/pressure", "signal": "nope"})
    assert bad.is_error and bad.content.startswith("error: unknown signal") and bad.evidence_id is None
    assert len(registry.ledger) == 2


def test_registry_caps_long_results_and_bounds_excerpts(registry):
    out = registry.execute("get_series", {"signals": [{"dataset": "TRIO/pressure", "signal": s} for s in
                                                      ("pressure", "flow", "pressure__mean", "flow__mean", "pressure__z", "flow__z")],
                                          "max_points": 500})
    assert not out.is_error and out.truncated
    assert "[output truncated: showing the first" in out.content
    assert len(out.content) < MAX_RESULT_CHARS + 400
    ev = registry.ledger.items[out.evidence_id]
    assert ev.truncated and len(ev.excerpt) <= 1500 + 30 and ev.chars > MAX_RESULT_CHARS


def test_memory_tools_require_known_evidence(registry):
    out = registry.execute("record_insight", {"title": "t", "finding": "f", "severity": "info", "evidence_ids": ["E9"]})
    assert out.is_error and "unknown evidence id(s) ['E9']" in out.content
    registry.execute("get_events", {})
    out = registry.execute("record_insight", {"title": "t", "finding": "f", "severity": "info", "evidence_ids": ["e1"]})
    assert not out.is_error and "staged insight C1" in out.content
    assert registry.staged.insights[0]["evidence_ids"] == ["E1"]
    out = registry.execute("propose_hypothesis", {"statement": "s", "evidence_ids": [], "direction": "supports", "strength": "weak"})
    assert out.is_error and "at least one evidence id" in out.content
    out = registry.execute("update_hypothesis", {"hypothesis_id": "H7", "direction": "supports", "strength": "weak", "evidence_ids": ["E1"]})
    assert out.is_error and "unknown hypothesis" in out.content


def test_tool_definitions_match_specs_and_eager_flag():
    defs = build_tool_definitions(eager_input_streaming=True)
    assert [d["name"] for d in defs] == tool_names()
    assert all(d["eager_input_streaming"] is True for d in defs)
    assert all(set(d) == {"name", "description", "input_schema", "eager_input_streaming"} for d in defs)
    assert all("eager_input_streaming" not in d for d in build_tool_definitions(False))
    assert {"conclude_cycle", "send_to_user", "correlate", "profile", "compare_windows", "save_lesson"} <= set(tool_names())
