"""Confidence maths, critic verdicts, memory persistence, offline mode, credentials and the live loop."""
from __future__ import annotations

import json
import math
import os

import anthropic
import httpx2
import pytest

from InSITE_Engine.mind import Mind, MindUnavailable
from InSITE_Engine.mind import hypotheses as hyp
from InSITE_Engine.mind.config import MindConfig
from InSITE_Engine.mind.console import Console
from InSITE_Engine.mind.critic import CRITIC_SCHEMA
from InSITE_Engine.mind.testing import (
    FakeStopDetails,
    ScriptedLLM,
    conclude,
    critic_message,
    message,
    text,
    tool_use,
)
from tests.test_mind_fixtures import build_result, extend_result, make_cfg


def make_mind(tmp_path, script, results=None, **overrides):
    llm = ScriptedLLM(script)
    results = list(results or [build_result()])
    mind = Mind(make_cfg(tmp_path, **overrides), llm=llm, pipeline_runner=lambda cfg: results[0], console=Console(quiet=True))
    return mind, llm, results


def memory_json(mind):
    with open(os.path.join(mind.config.memory_dir, "memory.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ maths


def test_log_odds_update_uses_fixed_likelihood_ratios():
    assert hyp.apply_update(0.5, "supports", "weak") == pytest.approx(1.5 / 2.5, abs=1e-4)
    assert hyp.apply_update(0.5, "supports", "moderate") == pytest.approx(0.75, abs=1e-4)
    assert hyp.apply_update(0.5, "supports", "strong") == pytest.approx(8 / 9, abs=1e-4)
    assert hyp.apply_update(0.5, "contradicts", "strong") == pytest.approx(1 / 9, abs=1e-4)
    # updates compose in log-odds space: two moderates == odds x9
    p = hyp.apply_update(hyp.apply_update(0.5, "supports", "moderate"), "supports", "moderate")
    assert p == pytest.approx(0.9, abs=1e-4)
    # a half-weight update applies half the log-odds change (odds x sqrt(8))
    half = hyp.apply_update(0.5, "supports", "strong", weight=0.5)
    assert half == pytest.approx(math.sqrt(8) / (1 + math.sqrt(8)), abs=1e-4)
    # probabilities never reach certainty
    assert hyp.apply_update(0.99, "supports", "strong") == 0.99
    assert hyp.apply_update(0.01, "contradicts", "strong") == 0.01
    with pytest.raises(ValueError):
        hyp.apply_update(0.5, "maybe", "weak")


def test_status_thresholds_and_uncertainty():
    assert hyp.status_for(0.85) == "supported" and hyp.status_for(0.849) == "open"
    assert hyp.status_for(0.10) == "refuted" and hyp.status_for(0.11) == "open"
    assert hyp.uncertainty(0.5) == 0.25 and hyp.uncertainty(0.9) == pytest.approx(0.09)


def test_config_defaults_nullable_keys_and_pricing():
    cfg = MindConfig.from_cfg({"mind": {"effort": None, "budget_usd_total": None, "fallbacks": None, "max_steps": 7}})
    assert cfg.effort == "high" and cfg.budget_usd_total is None and cfg.fallbacks is None and cfg.max_steps == 7
    assert MindConfig.from_cfg({}).model == "claude-fable-5-1"
    assert cfg.prices_for("claude-fable-5-1")["cache_read"] == 0.25
    assert cfg.prices_for("claude-unknown") == cfg.prices_for("claude-fable-5-1")  # conservative default
    with pytest.raises(ValueError):
        MindConfig.from_cfg({"mind": {"effort": "extreme"}})


def test_critic_schema_is_structured_output_safe():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "$ref"):
                assert key not in node
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)
    check(CRITIC_SCHEMA)


# ------------------------------------------------------------- critic application


CLAIMS_TURN = message(
    tool_use("get_events", {}),
    tool_use("correlate", {"a": {"dataset": "TCL/pumps", "signal": "pump1_current"},
                           "b": {"dataset": "TRIO/pressure", "signal": "pressure"}, "mode": "diff"}),
)
STAGING_TURN = message(
    tool_use("propose_hypothesis", {"statement": "Pump 1 trip caused the pressure drop", "evidence_ids": ["E2"],
                                    "direction": "supports", "strength": "strong"}),            # C1
    tool_use("propose_hypothesis", {"statement": "Night flow is rising", "evidence_ids": ["E1"],
                                    "direction": "supports", "strength": "moderate"}),          # C2
    tool_use("update_hypothesis", {"hypothesis_id": "H2", "direction": "supports", "strength": "strong",
                                   "evidence_ids": ["E1"]}),                                   # C3
    tool_use("record_insight", {"title": "Trip explains drop", "finding": "pump leads pressure by 10 min",
                                "severity": "action", "evidence_ids": ["E2"], "hypothesis_id": "H1"}),  # C4
    tool_use("record_insight", {"title": "Tank is empty", "finding": "no", "severity": "action", "evidence_ids": ["E1"]}),  # C5
    tool_use("record_insight", {"title": "Night flow", "finding": "rising", "severity": "watch", "evidence_ids": ["E1"]}),  # C6
    tool_use("add_question", {"question": "Is pump 2 affected too?"}),
    tool_use("save_lesson", {"name": "night-window", "summary": "Night window is 01-04 UTC", "body": "Use hours 1-4."}),
)


def test_critic_verdicts_are_applied_deterministically(tmp_path):
    mind, llm, _ = make_mind(tmp_path, [
        CLAIMS_TURN, STAGING_TURN, conclude("Done.", ["recheck night flow"]),
        critic_message([
            {"claim_id": "C1", "verdict": "supported", "reason": "lag +2 steps"},
            {"claim_id": "C2", "verdict": "weak", "reason": "one week only"},
            {"claim_id": "C3", "verdict": "unsupported", "reason": "E1 lists events, not flows"},
            {"claim_id": "C4", "verdict": "supported", "reason": "matches E2"},
            {"claim_id": "C5", "verdict": "unsupported", "reason": "nothing about tanks in E1"},
            {"claim_id": "C6", "verdict": "weak", "reason": "short series"},
        ], ["Does the rise continue?", "does the rise continue", "Is the pump protection faulty?"]),
    ])
    outcome = mind.think()
    assert outcome.status == "completed"
    board = {h["id"]: h for h in outcome.hypothesis_board}
    assert board["H1"]["probability"] == pytest.approx(8 / 9, abs=1e-3) and board["H1"]["status"] == "supported"
    # H2: proposal counted half (weak), update reverted (unsupported)
    assert board["H2"]["probability"] == pytest.approx(hyp.apply_update(0.5, "supports", "moderate", 0.5), abs=1e-3)
    mem = memory_json(mind)
    log = mem["hypotheses"]["H2"]["evidence_log"]
    assert [e["verdict"] for e in log] == ["weak", "unsupported"] and log[1]["weight"] == 0.0
    assert log[1]["probability_before"] == log[1]["probability_after"]
    assert [i["title"] for i in outcome.insights] == ["Trip explains drop", "Night flow"]
    assert outcome.insights[1]["tentative"] is True and outcome.insights[0].get("tentative") is None
    assert [i["title"] for i in outcome.rejected_insights] == ["Tank is empty"]
    assert outcome.rejected_insights[0]["critic_reason"] == "nothing about tanks in E1"
    # follow-up questions join the agenda, deduplicated
    questions = {q["question"] for q in outcome.open_questions}
    assert questions == {"Is pump 2 affected too?", "Does the rise continue?", "Is the pump protection faulty?"}
    assert outcome.next_focus[0] == "recheck night flow" and any("Does the rise continue?" in x for x in outcome.next_focus)
    report = open(outcome.report_path).read()
    assert "## Rejected by the critic" in report and "Tank is empty" in report
    assert "## Tentative insights" in report and "Night flow" in report
    assert "update reverted" in report and "counted half" in report
    assert mind.store.list_lessons() == [{"name": "night-window", "summary": "Night window is 01-04 UTC"}]
    assert outcome.critic["counts"] == {"supported": 2, "weak": 2, "unsupported": 2, "unreviewed": 0}


@pytest.mark.parametrize("critic_response", [
    message(text("not json at all"), stop_reason="end_turn"),
    message(text(json.dumps({"verdicts": "nope", "follow_up_questions": []}))),
    message(stop_reason="refusal", stop_details=FakeStopDetails(category="reasoning_extraction")),
    message(text("{\"verdicts\": ["), stop_reason="max_tokens"),
])
def test_critic_failures_leave_claims_unreviewed(tmp_path, critic_response):
    mind, llm, _ = make_mind(tmp_path, [CLAIMS_TURN, STAGING_TURN, conclude("Done.", []), critic_response])
    outcome = mind.think()
    assert outcome.status == "completed"
    assert outcome.critic["status"] == "unreviewed" and outcome.critic["error"]
    assert outcome.critic["counts"]["unreviewed"] == 6
    assert [i["title"] for i in outcome.insights] == ["Trip explains drop", "Tank is empty", "Night flow"]
    assert all(i["verdict"] == "unreviewed" for i in outcome.insights)
    assert "unreviewed" in open(outcome.report_path).read()


def test_critic_api_error_does_not_lose_the_cycle(tmp_path):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mind, llm, _ = make_mind(tmp_path, [CLAIMS_TURN, STAGING_TURN, conclude("Done.", []),
                                        anthropic.APIConnectionError(message="critic down", request=request)])
    outcome = mind.think()
    assert outcome.status == "completed" and outcome.critic["status"] == "unreviewed"
    assert len(outcome.insights) == 3 and mind.store.state.fingerprint is not None


# ------------------------------------------------------------- persistence


def test_memory_persists_and_second_cycle_sees_only_new_events(tmp_path):
    r1 = build_result()
    r2 = extend_result(r1)
    mind, llm, results = make_mind(tmp_path, [
        CLAIMS_TURN, STAGING_TURN, conclude("Cycle one.", ["look at tank level"]),
        critic_message([{"claim_id": f"C{i}", "verdict": "supported", "reason": "ok"} for i in range(1, 7)]),
    ], results=[r1])
    first = mind.think()
    mem = memory_json(mind)
    assert mem["cycle"] == 1 and sorted(mem["seen_event_ids"]) == sorted(r1.events["event_id"].tolist())
    assert mem["fingerprint"]["datasets"]["TCL/pumps"]["rows"] == 2016
    assert first.perception["diff"]["first_cycle"] is True

    # a fresh Mind on the same memory dir, with new data
    results[0] = r2
    llm2 = ScriptedLLM([
        message(tool_use("get_events", {"dataset": "TCL/pumps"})),
        message(tool_use("update_hypothesis", {"hypothesis_id": "H2", "direction": "contradicts", "strength": "weak",
                                               "evidence_ids": ["E1"]})),
        conclude("Cycle two.", []),
        critic_message([{"claim_id": "C1", "verdict": "supported", "reason": "ok"}]),
    ])
    mind2 = Mind(make_cfg(tmp_path), llm=llm2, pipeline_runner=lambda cfg: results[0], console=Console(quiet=True))
    assert set(mind2.store.state.hypotheses) == {"H1", "H2"}
    second = mind2.think()
    diff = second.perception["diff"]
    assert diff["first_cycle"] is False and diff["new_events"] == ["ev_new_flatline"]
    assert diff["updated_datasets"]["TCL/pumps"]["rows_added"] == 288
    assert diff["stale_datasets"] == ["AX/workorders", "Meters/site_flow", "TRIO/pressure"]
    assert diff["state_transitions"] == [{"signal": "TCL/pumps:tank_level", "from": None, "to": "flatlined"}]
    brief = llm2.requests[0]["messages"][0]["content"][0]["text"]
    assert "ev_new_flatline NEW" in brief and "+288 rows" in brief and "look at tank level" in brief
    assert "Settled hypotheses: H1 supported p=0.89" in brief and "H2 supported p=0.96: Night flow is rising" in brief
    assert "night-window: Night window is 01-04 UTC" in brief
    assert "ev_new_flatline | NEW" in llm2.requests[1]["messages"][-1]["content"][0]["content"]
    mem = memory_json(mind2)
    assert mem["cycle"] == 2 and "ev_new_flatline" in mem["seen_event_ids"]
    h2 = mem["hypotheses"]["H2"]
    assert [h["cycle"] for h in h2["history"]] == [1, 1, 2] and h2["updated_cycle"] == 2
    assert len(h2["evidence_log"]) == 3 and h2["evidence_log"][-1]["direction"] == "contradicts"
    assert h2["evidence_log"][-1]["excerpts"][0]["tool"] == "get_events"
    assert "0.50 -> " in second.hypothesis_board[0]["trend"] or any("->" in h["trend"] for h in second.hypothesis_board)
    assert os.path.exists(os.path.join(mind2.config.memory_dir, "reports", "cycle_0002.md"))
    assert open(os.path.join(mind2.config.memory_dir, "reports", "latest.md")).read().startswith("# MŸND report - cycle 2")


def test_offline_cycle_writes_report_and_leaves_fingerprint_untouched(tmp_path):
    mind = Mind(make_cfg(tmp_path), llm=False, pipeline_runner=lambda cfg: build_result(), console=Console(quiet=True))
    outcome = mind.think(offline=True)
    assert outcome.status == "offline" and outcome.cost_usd == 0.0
    assert outcome.summary.startswith("Perception-only cycle (no model was called).")
    assert "4 events" in outcome.summary and "everything is new" in outcome.summary
    assert os.path.exists(outcome.report_path) and os.path.exists(outcome.journal_path)
    report = open(outcome.report_path).read()
    assert "(offline)" in report and "ev_pump_trip" in report
    mem = memory_json(mind)
    assert mem["fingerprint"] is None and mem["seen_event_ids"] == [] and mem["cycle"] == 1
    # a second offline run still sees everything as new, and numbering continues
    again = mind.think(offline=True)
    assert again.cycle == 2 and again.perception["diff"]["first_cycle"] is True
    # without a model client, a non-offline think() degrades to the offline cycle instead of failing
    assert mind.think().status == "offline"
    with pytest.raises(MindUnavailable):
        mind.ask("why?")


def test_mind_unavailable_without_credentials(tmp_path, monkeypatch):
    class NoCredentialsClient:
        api_key = None
        auth_token = None
        _token_cache = None

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: NoCredentialsClient())
    with pytest.raises(MindUnavailable, match="ANTHROPIC_API_KEY"):
        Mind(make_cfg(tmp_path), pipeline_runner=lambda cfg: build_result())


def test_mind_builds_real_client_when_credentials_exist_without_calling_api(tmp_path, monkeypatch):
    class KeyClient:
        api_key = "sk-ant-test"
        auth_token = None
        _token_cache = None

        def __getattr__(self, name):  # any request would blow up
            raise AssertionError(f"unexpected client access: {name}")

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: KeyClient())
    mind = Mind(make_cfg(tmp_path), pipeline_runner=lambda cfg: build_result())
    assert mind.llm.client.api_key == "sk-ant-test"
    assert mind.think(offline=True).status == "offline"


def test_ask_uses_the_question_as_goal(tmp_path):
    mind, llm, _ = make_mind(tmp_path, [conclude("Pressure dropped because pump 1 tripped.", [])])
    outcome = mind.ask("why did pressure drop on day 3?")
    assert outcome.goal == "why did pressure drop on day 3?"
    assert outcome.summary == "Pressure dropped because pump 1 tripped."
    brief = llm.requests[0]["messages"][0]["content"][0]["text"]
    assert "why did pressure drop on day 3?" in brief
    assert "Question: why did pressure drop on day 3?" in open(outcome.report_path).read()


def test_lessons_are_staged_until_commit_and_readable(tmp_path):
    mind, llm, _ = make_mind(tmp_path, [
        message(tool_use("save_lesson", {"name": "tcl-time", "summary": "TCL timestamps are local time", "body": "Shift by +1h."}),
                tool_use("read_lesson", {"name": "tcl-time"}),
                tool_use("read_lesson", {"name": "missing"}),
                tool_use("save_lesson", {"name": "Bad Name", "summary": "x", "body": "y"})),
        conclude("Done.", []),
    ])
    outcome = mind.think()
    results = llm.requests[1]["messages"][-1]["content"]
    assert "staged (new)" in results[0]["content"]
    assert results[1]["content"].startswith("# TCL timestamps are local time") and "staged this cycle" in results[1]["content"]
    assert results[2]["is_error"] and "no lesson named 'missing'" in results[2]["content"]
    assert results[3]["is_error"] and "slug" in results[3]["content"]
    path = os.path.join(mind.config.memory_dir, "lessons", "tcl-time.md")
    assert open(path).read() == "# TCL timestamps are local time\n\nShift by +1h.\n"
    assert outcome.status == "completed"
    # delete in a later cycle
    llm.extend([message(tool_use("delete_lesson", {"name": "tcl-time", "reason": "wrong"})), conclude("Done.", [])])
    mind.think()
    assert not os.path.exists(path) and mind.store.list_lessons() == []


# ------------------------------------------------------------- live loop


def test_live_runs_cycles_idles_then_waits_for_data(tmp_path):
    r1 = build_result()
    script = []
    for _ in range(5):
        script.append(conclude("ok", []))
    mind, llm, results = make_mind(tmp_path, script, results=[r1], max_idle_cycles=2)
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 6:
            raise KeyboardInterrupt
    mind._sleep = fake_sleep
    outcomes = mind.live(interval_seconds=7)
    # cycle 1 has new data, cycles 2 and 3 are idle, then the loop only waits
    assert [o.cycle for o in outcomes] == [1, 2, 3]
    assert [o.status for o in outcomes] == ["completed"] * 3
    assert len(llm.requests) == 3 and sleeps == [7] * 6
    briefs = [r["messages"][0]["content"][0]["text"] for r in llm.requests]
    assert "No new data since the last cycle" in briefs[1] and "No new data" not in briefs[0]


def test_live_stops_on_max_cycles_and_total_budget(tmp_path):
    mind, llm, _ = make_mind(tmp_path, [conclude("ok", []) for _ in range(4)])
    mind._sleep = lambda s: None
    outcomes = mind.live(interval_seconds=0, max_cycles=2)
    assert [o.cycle for o in outcomes] == [1, 2] and len(llm.requests) == 2
    # each cycle costs $0.02; a $0.03 total budget allows two cycles, the second capped to the remainder
    mind, llm, _ = make_mind(tmp_path / "b", [conclude("ok", []) for _ in range(4)], max_idle_cycles=10)
    mind._sleep = lambda s: None
    outcomes = mind.live(interval_seconds=0, budget_usd_total=0.03)
    assert len(outcomes) == 2 and sum(o.cost_usd for o in outcomes) == pytest.approx(0.04)


def test_live_backs_off_after_an_api_error_instead_of_dying(tmp_path):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mind, llm, _ = make_mind(tmp_path, [
        anthropic.APIConnectionError(message="down", request=request),
        conclude("recovered", []),
    ], max_idle_cycles=5)
    sleeps = []
    mind._sleep = sleeps.append
    outcomes = mind.live(interval_seconds=10, max_cycles=2)
    assert [o.status for o in outcomes] == ["error", "completed"]
    assert sleeps[0] == 30.0  # backoff: max(interval, 30s) after the first failure
    assert mind.store.state.fingerprint is not None  # the second cycle committed normally
