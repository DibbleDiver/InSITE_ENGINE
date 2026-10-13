"""The investigate loop: conversation invariants and stop-reason handling, all with a scripted model."""
from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from InSITE_Engine.mind import Mind
from InSITE_Engine.mind.config import MindConfig
from InSITE_Engine.mind.console import Console
from InSITE_Engine.mind.llm import AnthropicLLM
from InSITE_Engine.mind.prompts import CUT_OFF_NOTE, NO_TOOL_NOTE
from InSITE_Engine.mind.testing import (
    FakeIteration,
    FakeStopDetails,
    FakeUsage,
    ScriptedLLM,
    conclude,
    critic_message,
    fallback,
    message,
    text,
    thinking,
    tool_use,
)
from tests.test_mind_fixtures import build_result, make_cfg


def make_mind(tmp_path, script, **overrides):
    llm = ScriptedLLM(script)
    result = build_result()
    mind = Mind(make_cfg(tmp_path, **overrides), llm=llm, pipeline_runner=lambda cfg: result, console=Console(quiet=True))
    return mind, llm


def agent_requests(llm):
    """Requests of the investigation conversation (the critic's request carries no tools)."""
    return [r for r in llm.requests if r["tools"] is not None]


def last_user_content(request):
    msg = request["messages"][-1]
    assert msg["role"] == "user"
    return msg["content"]


def test_parallel_tool_calls_return_all_results_in_one_user_message(tmp_path):
    mind, llm = make_mind(tmp_path, [
        message(thinking("look at both"),
                tool_use("describe_signal", {"dataset": "TRIO/pressure", "signal": "pressure"}, id="t1"),
                tool_use("get_events", {"severity": "high"}, id="t2"),
                tool_use("describe_signal", {"dataset": "TRIO/pressure", "signal": "nope"}, id="t3")),
        conclude("Nothing to report.", ["x"]),
    ])
    outcome = mind.think()
    assert outcome.status == "completed" and outcome.summary == "Nothing to report."
    assert len(llm.requests) == 2  # no claims -> no critic call
    content = last_user_content(llm.requests[1])
    assert [b["type"] for b in content] == ["tool_result"] * 3
    assert [b["tool_use_id"] for b in content] == ["t1", "t2", "t3"]
    assert content[0]["content"].startswith("E1 = describe_signal(") and content[1]["content"].startswith("E2 = get_events(")
    assert content[2]["is_error"] is True and "unknown signal" in content[2]["content"]
    assert "is_error" not in content[0]


def test_conversation_is_append_only_with_frozen_system_and_tools(tmp_path):
    mind, llm = make_mind(tmp_path, [
        message(thinking(""), tool_use("get_events", {})),                      # empty thinking text is kept verbatim
        message(text("looking further"), tool_use("list_datasets", {})),
        message(tool_use("record_insight", {"title": "t", "finding": "f", "severity": "info", "evidence_ids": ["E1"]})),
        conclude("Done.", []),
        critic_message([{"claim_id": "C1", "verdict": "supported", "reason": "ok"}]),
    ])
    mind.think()
    reqs = agent_requests(llm)
    assert len(reqs) == 4
    for prev, nxt in zip(reqs, reqs[1:]):
        assert nxt["system"] == prev["system"]
        assert nxt["tools"] == prev["tools"]
        assert len(nxt["messages"]) > len(prev["messages"])
        assert nxt["messages"][: len(prev["messages"])] == prev["messages"]
    # the assistant turns are echoed as received, thinking blocks included
    first_assistant = reqs[1]["messages"][1]
    assert first_assistant["role"] == "assistant"
    assert [b["type"] for b in first_assistant["content"]] == ["thinking", "tool_use"]
    assert first_assistant["content"][0]["thinking"] == ""
    assert reqs[0]["messages"][0]["content"][0]["text"].startswith("# Situation brief for cycle 1")
    # the critic ran in a fresh conversation with structured output and no tools
    critic_req = [r for r in llm.requests if r["tools"] is None]
    assert len(critic_req) == 1 and critic_req[0]["output_format"]["type"] == "object"
    assert critic_req[0]["messages"][0]["role"] == "user" and len(critic_req[0]["messages"]) == 1


def test_wrap_up_note_on_budget_is_appended_after_tool_results(tmp_path):
    # each scripted turn costs $0.02 (1000 input, 200 output tokens at Fable prices)
    mind, llm = make_mind(tmp_path, [
        message(tool_use("get_events", {})),
        message(tool_use("get_events", {})),
        message(tool_use("get_events", {})),
        message(tool_use("get_events", {})),
    ], budget_usd_per_cycle=0.046)
    outcome = mind.think()
    assert outcome.status == "wrapped_up_on_budget"
    assert len(llm.requests) == 3  # hard stop once $0.06 >= $0.046
    second = last_user_content(llm.requests[1])
    assert [b["type"] for b in second] == ["tool_result"]  # below 80%: no note yet
    third = last_user_content(llm.requests[2])
    assert [b["type"] for b in third] == ["tool_result", "text"]
    assert "wrap up now" in third[1]["text"] and "$0.04 of the $0.05 cycle budget spent" in third[1]["text"]
    assert outcome.cost_usd == pytest.approx(0.06)
    assert "stopped on its cost budget" in outcome.summary


def test_wrap_up_note_three_steps_before_limit_and_earlier_notes_stay(tmp_path):
    mind, llm = make_mind(tmp_path, [message(tool_use("get_events", {})) for _ in range(6)], max_steps=5)
    outcome = mind.think()
    assert outcome.status == "step_limit" and outcome.steps == 5
    assert len(llm.requests) == 5
    assert [b["type"] for b in last_user_content(llm.requests[1])] == ["tool_result"]
    third = last_user_content(llm.requests[2])
    assert [b["type"] for b in third] == ["tool_result", "text"] and "3 steps left" in third[1]["text"]
    fourth = llm.requests[3]["messages"]
    assert fourth[4]["content"] == third  # the earlier note is still there, byte for byte
    assert "2 steps left" in last_user_content(llm.requests[3])[1]["text"]
    assert "1 step left" in last_user_content(llm.requests[4])[1]["text"]


def test_refusal_ends_cycle_without_running_tools(tmp_path):
    mind, llm = make_mind(tmp_path, [
        message(tool_use("get_events", {}), stop_reason="refusal", stop_details=FakeStopDetails(category="cyber")),
        conclude("never reached", []),
    ])
    outcome = mind.think()
    assert outcome.status == "refused" and len(llm.requests) == 1
    journal = json.load(open(outcome.journal_path))
    assert journal["turns"][0]["tool_calls"] == [] and journal["evidence"] == {}
    assert journal["refusal"]["stop_details"]["category"] == "cyber"
    assert mind.store.state.fingerprint is None and mind.store.state.seen_event_ids == []
    assert mind.store.state.cycle == 1 and mind.store.state.last_cycle_status == "refused"
    assert "(refused)" in open(outcome.report_path).read()


def test_max_tokens_with_tool_use_returns_cut_off_errors(tmp_path):
    mind, llm = make_mind(tmp_path, [
        message(tool_use("get_events", {}, id="cut"), stop_reason="max_tokens"),
        conclude("Done.", []),
    ])
    outcome = mind.think()
    assert outcome.status == "completed"
    content = last_user_content(llm.requests[1])
    assert content == [{"type": "tool_result", "tool_use_id": "cut", "is_error": True,
                        "content": content[0]["content"]}]
    assert "cut off" in content[0]["content"]
    journal = json.load(open(outcome.journal_path))
    assert journal["evidence"] == {} and journal["turns"][0]["tool_calls"][0]["skipped"]


def test_max_tokens_without_tool_use_asks_to_continue(tmp_path):
    mind, llm = make_mind(tmp_path, [message(text("partial"), stop_reason="max_tokens"), conclude("Done.", [])])
    mind.think()
    assert last_user_content(llm.requests[1]) == [{"type": "text", "text": CUT_OFF_NOTE}]


def test_fallback_block_runs_only_later_tools_and_filters_echo(tmp_path):
    usage = FakeUsage(iterations=[FakeIteration("message", "claude-fable-5-1"), FakeIteration("fallback_message", "claude-opus-4-8")])
    mind, llm = make_mind(tmp_path, [
        message(thinking("before"), tool_use("get_events", {}, id="before"), fallback("claude-fable-5-1", "claude-opus-4-8"),
                text("continuing on the fallback model"), tool_use("list_datasets", {}, id="after"),
                stop_reason="tool_use", usage=usage, model="claude-opus-4-8"),
        conclude("Done.", []),
    ])
    outcome = mind.think()
    assert outcome.status == "completed" and outcome.served_by_fallback is True
    echoed = llm.requests[1]["messages"][1]
    assert [b["type"] for b in echoed["content"]] == ["fallback", "text", "tool_use"]
    assert echoed["content"][2]["id"] == "after"
    results = last_user_content(llm.requests[1])
    assert [b["tool_use_id"] for b in results] == ["after"]
    journal = json.load(open(outcome.journal_path))
    assert journal["turns"][0]["served_by_fallback"] is True and journal["turns"][0]["model"] == "claude-opus-4-8"
    assert "fallback model" in open(outcome.report_path).read()


def test_end_turn_without_conclude_uses_final_text_as_summary(tmp_path):
    mind, llm = make_mind(tmp_path, [message(text("All quiet; nothing needs attention."), stop_reason="end_turn")])
    outcome = mind.think()
    assert outcome.status == "completed" and outcome.summary == "All quiet; nothing needs attention."
    assert len(llm.requests) == 1
    assert "without calling conclude_cycle" in open(outcome.report_path).read()


def test_empty_end_turn_is_nudged_once(tmp_path):
    mind, llm = make_mind(tmp_path, [message(thinking("hmm"), stop_reason="end_turn"), conclude("Done.", [])])
    outcome = mind.think()
    assert outcome.status == "completed" and outcome.summary == "Done."
    assert last_user_content(llm.requests[1]) == [{"type": "text", "text": NO_TOOL_NOTE}]


def test_pause_turn_is_appended_and_resent(tmp_path):
    mind, llm = make_mind(tmp_path, [message(text("working"), stop_reason="pause_turn"), conclude("Done.", [])])
    outcome = mind.think()
    assert outcome.status == "completed"
    msgs = llm.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == [{"type": "text", "text": "working"}]


def test_api_error_gives_error_status_and_commits_nothing(tmp_path):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    mind, llm = make_mind(tmp_path, [
        message(tool_use("get_events", {}), tool_use("record_insight", {"title": "t", "finding": "f", "severity": "info", "evidence_ids": ["E1"]})),
        anthropic.APIConnectionError(message="connection reset", request=request),
    ])
    outcome = mind.think()
    assert outcome.status == "error" and "connection reset" in outcome.error
    assert outcome.insights == [] and mind.store.state.fingerprint is None
    assert mind.store.state.hypotheses == {} and mind.store.state.recent_insights == []
    assert mind.store.state.last_cycle_status == "error"
    assert "failed with a model error" in outcome.summary


def test_send_to_user_is_shown_verbatim_and_journaled(tmp_path):
    llm = ScriptedLLM([message(tool_use("send_to_user", {"message": "Half way: checking the pumps."})), conclude("Done.", [])])
    console = Console(quiet=True)
    mind = Mind(make_cfg(tmp_path), llm=llm, pipeline_runner=lambda cfg: build_result(), console=console)
    outcome = mind.think()
    assert outcome.sent_messages == ["Half way: checking the pumps."]
    assert "[to operations team] Half way: checking the pumps." in console.lines
    assert json.load(open(outcome.journal_path))["send_to_user"] == ["Half way: checking the pumps."]
    assert "Half way: checking the pumps." in open(outcome.report_path).read()


# ------------------------------------------------- the real client, against a fake SDK client


class FakeStream:
    def __init__(self, final=None, events=(), error=None):
        self.final, self.events, self.error = final, list(events), error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if self.error is not None:
            raise self.error
        yield from self.events

    def get_final_message(self):
        return self.final


class FakeClient:
    api_key = "sk-ant-test"
    auth_token = None
    _token_cache = None

    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return self.streams.pop(0)


def test_anthropic_llm_builds_the_fable_request_shape():
    final = message(text("ok"))
    events = [SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
              SimpleNamespace(type="thinking", thinking="summary..."),
              SimpleNamespace(type="content_block_stop", content_block=SimpleNamespace(type="thinking")),
              SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", name="get_events")),
              SimpleNamespace(type="input_json", partial_json="{}")]
    client = FakeClient([FakeStream(final, events)])
    config = MindConfig.from_cfg(make_cfg(__import__("pathlib").Path("/tmp")))
    llm = AnthropicLLM(config, client=client)
    seen = []
    tools = [{"name": "get_events", "description": "d", "input_schema": {"type": "object"}, "eager_input_streaming": True}]
    out = llm.respond(system="SYS", tools=tools, messages=[{"role": "user", "content": "hi"}], effort="high",
                      on_event=lambda kind, payload: seen.append((kind, payload)))
    assert out is final
    call = client.calls[0]
    assert call["model"] == "claude-fable-5-1" and call["max_tokens"] == 64000
    assert call["system"] == "SYS" and call["tools"] is tools
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["output_config"] == {"effort": "high"}
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert "tool_choice" not in call and "temperature" not in call
    assert seen == [("thinking_start", ""), ("thinking", "summary..."), ("block_stop", "thinking"), ("tool_use_start", "get_events")]
    # critic shape: no tools key at all, structured output, critic model/effort, no fallbacks when disabled
    config.fallbacks = None
    client.streams.append(FakeStream(final))
    llm.respond(system="C", tools=None, messages=[{"role": "user", "content": "x"}], effort="medium",
                output_format={"type": "object"}, model="claude-opus-5-5")
    call = client.calls[1]
    assert "tools" not in call and "betas" not in call and "fallbacks" not in call
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_anthropic_llm_reissues_on_unparseable_tool_json_at_most_twice():
    final = message(text("ok"))
    err = ValueError("Unable to parse tool parameter JSON from model")
    client = FakeClient([FakeStream(error=err), FakeStream(error=err), FakeStream(final)])
    llm = AnthropicLLM(MindConfig(), client=client)
    assert llm.respond(system="S", tools=[], messages=[], effort="low") is final
    assert len(client.calls) == 3
    client = FakeClient([FakeStream(error=err)] * 3)
    llm = AnthropicLLM(MindConfig(), client=client)
    with pytest.raises(ValueError):
        llm.respond(system="S", tools=[], messages=[], effort="low")
    assert len(client.calls) == 3
    # API errors are not retried here (the SDK already retries 429/5xx) and propagate unchanged
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeClient([FakeStream(error=anthropic.APIConnectionError(message="down", request=request))])
    with pytest.raises(anthropic.APIConnectionError):
        AnthropicLLM(MindConfig(), client=client).respond(system="S", tools=[], messages=[], effort="low")


def test_journal_serialization_handles_sdk_pydantic_blocks_and_usage():
    from anthropic.types.beta import (
        BetaFallbackBlock,
        BetaMessage,
        BetaTextBlock,
        BetaThinkingBlock,
        BetaToolUseBlock,
        BetaUsage,
    )

    from InSITE_Engine.mind.llm import served_by_fallback, usage_dict
    from InSITE_Engine.mind.serialize import block_to_dict, messages_to_dicts, to_jsonable

    fb = BetaFallbackBlock.model_validate({"type": "fallback", "from": {"model": "claude-fable-5-1"},
                                           "to": {"model": "claude-opus-4-8"}, "trigger": {"type": "refusal", "category": "cyber"}})
    blocks = [BetaThinkingBlock(type="thinking", thinking="", signature="sig"),
              BetaToolUseBlock(type="tool_use", id="toolu_1", name="get_events", input={"severity": "high"}),
              fb, BetaTextBlock(type="text", text="hi", citations=None)]
    dicts = [block_to_dict(b) for b in blocks]
    assert dicts[0] == {"type": "thinking", "thinking": "", "signature": "sig"}
    assert dicts[1]["input"] == {"severity": "high"} and dicts[1]["name"] == "get_events"
    assert dicts[2]["from"] == {"model": "claude-fable-5-1"} and dicts[2]["to"] == {"model": "claude-opus-4-8"}
    assert dicts[3]["text"] == "hi"
    transcript = messages_to_dicts([{"role": "assistant", "content": blocks},
                                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "E1 ..."}]}])
    json.dumps(transcript)  # fully JSON-serialisable
    usage = BetaUsage.model_validate({
        "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": None, "cache_creation_input_tokens": 3,
        "iterations": [
            {"type": "message", "model": "claude-fable-5-1", "input_tokens": 1, "output_tokens": 1,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 9, "output_tokens": 4,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 3},
        ]})
    msg = BetaMessage.model_validate({
        "id": "m", "type": "message", "role": "assistant", "model": "claude-opus-4-8", "content": [], "stop_reason": "refusal",
        "stop_sequence": None, "usage": usage.model_dump(),
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": None, "fallback_credit_token": None,
                         "fallback_has_prefill_claim": False, "recommended_model": None}})
    assert served_by_fallback(msg) is True
    assert usage_dict(msg) == {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 3}
    assert to_jsonable(msg.stop_details)["category"] == "cyber"
    plain = message(text("x"))
    assert served_by_fallback(plain) is False and usage_dict(plain)["input_tokens"] == 1000


def test_console_streams_thinking_and_text_live():
    import io

    buf = io.StringIO()
    console = Console(stream=buf)
    for kind, payload in [("message_start", "claude-fable-5-1"), ("thinking_start", ""), ("thinking", "Checking the "),
                          ("thinking", "pump first."), ("block_stop", "thinking"), ("thinking_start", ""),
                          ("block_stop", "thinking"), ("text_start", ""), ("text", "Pump 1 tripped."), ("block_stop", "text"),
                          ("tool_use_start", "get_events"), ("fallback", "claude-fable-5-1 -> claude-opus-4-8")]:
        console.on_event(kind, payload)
    console.user_message("Half way.")
    console.tool_call("get_events", {"severity": "high"}, "E1", False, 120)
    out = buf.getvalue()
    assert "[thinking] Checking the pump first.\n" in out
    assert "[thinking] \n" not in out  # an empty (omitted) thinking block prints nothing
    assert "[model] Pump 1 tripped.\n" in out
    assert "[calling get_events]\n" in out and "[fallback] request re-run on a fallback model" in out
    assert "[to operations team] Half way." in out
    assert '[tool] get_events({"severity": "high"}) -> E1 (120 chars)' in out
    quiet = Console(stream=io.StringIO(), quiet=True)
    quiet.user_message("x")
    assert quiet.lines == ["[to operations team] x"] and quiet.stream.getvalue() == ""
