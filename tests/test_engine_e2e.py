"""The whole engine on the demo site: demo data -> pipeline -> a scripted MYND cycle ->
critic -> memory and report. No model is called; the scripted model plays the
investigation a real cycle is expected to make."""
import copy
import os

import yaml

from InSITE_Engine.demo.generate import generate_demo_data
from InSITE_Engine.mind import Mind
from InSITE_Engine.mind.testing import ScriptedLLM, conclude, critic_message, message, thinking, tool_use

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIP = {"start": "2026-10-03T12:00Z", "end": "2026-10-03T19:00Z"}


def engine_cfg(tmp_path):
    with open(os.path.join(ROOT, "InSITE_Engine", "config", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["input_path"] = str(tmp_path / "raw")
    cfg["output_path"] = str(tmp_path / "out")
    cfg["mind"]["memory_dir"] = str(tmp_path / "out" / "mind")
    return cfg


def investigation_script():
    return [
        message(thinking("The day-3 cluster looks like one incident."),
                tool_use("get_events", {}),
                tool_use("query_rows", {"dataset": "AX/work_orders",
                                        "filters": [{"column": "description", "op": "contains", "value": "trip"}]})),
        message(tool_use("correlate", {"a": {"dataset": "TCL/tank_pumps", "signal": "p2_current_a"},
                                       "b": {"dataset": "TRIO/telemetry_site_a", "signal": "outlet_pressure_bar"},
                                       "mode": "diff", "max_lag": 6, **TRIP}),
                tool_use("profile", {"dataset": "Meters/zone_flow", "signal": "zone_flow_lps", "group_by": "day",
                                     "stat": "min", "hours": {"from": 1, "to": 4}}),
                tool_use("describe_signal", {"dataset": "TCL/tank_pumps", "signal": "tank_level_b_pct"})),
        message(tool_use("propose_hypothesis", {
                    "statement": "The 3 Oct trip of pump P2 caused the TRIO outlet pressure drop and the zone flow dip.",
                    "evidence_ids": ["E1", "E2", "E3"], "direction": "supports", "strength": "strong"}),
                tool_use("propose_hypothesis", {
                    "statement": "Minimum night flow rose about 20% from 4 Oct, consistent with a leak.",
                    "evidence_ids": ["E4"], "direction": "supports", "strength": "moderate"}),
                tool_use("record_insight", {
                    "title": "Pump P2 trip explains the 3 Oct pressure and flow dip",
                    "finding": "P2 current fell to 0 A at 14:00 UTC; outlet pressure followed 10 minutes later.",
                    "severity": "info", "evidence_ids": ["E1", "E2", "E3"], "hypothesis_id": "H1"}),
                tool_use("record_insight", {
                    "title": "Night flow up about 20% since 4 Oct",
                    "finding": "The 01-04 UTC daily minimum stepped from about 14.5 to 17.6 l/s on 4 Oct.",
                    "severity": "action", "evidence_ids": ["E4"], "hypothesis_id": "H2"}),
                tool_use("record_insight", {
                    "title": "Tank B may be drifting",
                    "finding": "Tank level B trends up about 0.3 points per day; tank A not yet compared.",
                    "severity": "watch", "evidence_ids": ["E5"]})),
        conclude("Pump P2 tripped on 3 October and explains the pressure and flow dip. Night flow has been about 20% "
                 "higher since 4 October, which points to a leak.", ["Compare tank B with tank A"]),
        critic_message([
            {"claim_id": "C1", "verdict": "supported", "reason": "E3 shows P2 leading pressure by 10 min, r=0.99."},
            {"claim_id": "C2", "verdict": "supported", "reason": "E4 daily minima step up on 4 Oct."},
            {"claim_id": "C3", "verdict": "supported", "reason": "E1-E3 agree on the timing."},
            {"claim_id": "C4", "verdict": "supported", "reason": "E4 shows the step."},
            {"claim_id": "C5", "verdict": "unsupported", "reason": "A 7-day slope alone does not show drift."},
        ], ["Does tank A show the same trend?"]),
    ]


def test_full_engine_cycle_on_demo_site(tmp_path):
    cfg = engine_cfg(tmp_path)
    generate_demo_data(cfg["input_path"])
    llm = ScriptedLLM(investigation_script())
    outcome = Mind(cfg, llm=llm).think()

    assert outcome.status == "completed"
    assert outcome.steps == 4
    assert outcome.perception["event_count"] == 11
    assert outcome.perception["datasets"] == 4

    # The tools found what the pipeline alone cannot: the cross-source chain and the night-flow rise.
    tool_results = [b for m in llm.requests[2]["messages"] if m["role"] == "user" for b in m["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_result"]
    joined = "\n".join(str(r["content"]) for r in tool_results)
    assert "P2 tripped on overload" in joined
    assert "p2_current_a LEADS TRIO/telemetry_site_a:outlet_pressure_bar" in joined
    assert "r at best lag = 0.9" in joined
    assert "trend of the daily min" in joined

    # Critic verdicts applied: two accepted hypotheses, the drift insight rejected.
    board = {h["id"]: h for h in outcome.hypothesis_board}
    assert board["H1"]["status"] == "supported" and board["H1"]["probability"] > 0.85
    assert 0.7 < board["H2"]["probability"] < 0.8
    assert [i["title"] for i in outcome.insights] == [
        "Pump P2 trip explains the 3 Oct pressure and flow dip", "Night flow up about 20% since 4 Oct"]
    assert [i["title"] for i in outcome.rejected_insights] == ["Tank B may be drifting"]
    assert any("tank A" in q["question"] for q in outcome.open_questions)

    report = open(outcome.report_path).read()
    assert report.startswith("# MŸND report - cycle 1 (completed)")
    assert "points to a leak" in report
    assert "Rejected" in report or "rejected" in report
    assert os.path.exists(os.path.join(cfg["mind"]["memory_dir"], "reports", "latest.md"))

    # Every request kept the conversation append-only with a frozen system prompt and tool list.
    agent_requests = llm.requests[:-1]
    for prev, cur in zip(agent_requests, agent_requests[1:]):
        assert cur["system"] == prev["system"] and cur["tools"] == prev["tools"]
        assert cur["messages"][: len(prev["messages"])] == prev["messages"]
