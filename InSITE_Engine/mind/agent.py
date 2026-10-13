"""The investigate step: an agent loop over the model with the Mind's tools.

Invariants the loop keeps for Claude Fable 5.1's preserved thinking:
- one conversation per cycle; `system` and `tools` are frozen for its duration;
- every assistant turn is appended exactly as received (after a mid-output
  fallback: thinking/redacted_thinking/tool_use blocks before the last fallback
  block are omitted, text blocks stay);
- harness notes are extra text blocks appended AFTER the tool results of the
  same user message; earlier notes are never edited or removed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import MindConfig
from .console import Console
from .llm import LLM, TokenUsage, cost_usd, served_by_fallback, usage_dict
from .prompts import CUT_OFF_NOTE, NO_TOOL_NOTE, WRAP_UP_NOTE
from .serialize import block_type, to_jsonable
from .tools import ToolRegistry

log = logging.getLogger("insite.mind.agent")

try:  # the SDK is optional at import time so tests with a scripted model never need it
    import anthropic as _anthropic

    API_ERRORS: tuple[type[BaseException], ...] = (_anthropic.APIStatusError, _anthropic.APIConnectionError)
except ImportError:  # pragma: no cover
    API_ERRORS = ()

WRAP_UP_BUDGET_FRACTION = 0.8
WRAP_UP_STEPS_BEFORE_LIMIT = 3
MAX_NO_TOOL_NUDGES = 1
CUT_OFF_RESULT = "Your tool input was cut off because the reply hit the output limit (max_tokens); the tool did not run. " \
                 "Call it again with a shorter input."


@dataclass
class TurnRecord:
    step: int
    stop_reason: str | None
    stop_details: Any
    model: str | None
    served_by_fallback: bool
    fallback_block: bool
    usage: dict
    cost_usd: float
    thinking: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "step": self.step, "stop_reason": self.stop_reason, "stop_details": to_jsonable(self.stop_details),
            "model": self.model, "served_by_fallback": self.served_by_fallback, "fallback_block": self.fallback_block,
            "usage": self.usage, "cost_usd": round(self.cost_usd, 6), "thinking_summaries": self.thinking,
            "text": self.text, "tool_calls": self.tool_calls, "harness_note": self.note,
        }


@dataclass
class AgentResult:
    status: str
    summary: str
    concluded: bool
    next_focus: list[str]
    turns: list[TurnRecord]
    messages: list[dict]
    usage: TokenUsage
    cost_usd: float
    steps: int
    final_text: str = ""
    refusal: dict | None = None
    error: str | None = None
    served_by_fallback: bool = False
    stop_note: str = ""


class InvestigationAgent:
    def __init__(self, llm: LLM, config: MindConfig, registry: ToolRegistry, console: Console,
                 system_prompt: str, tools: list[dict], *, effort: str | None = None,
                 budget_usd: float | None = None, max_steps: int | None = None):
        self.llm = llm
        self.config = config
        self.registry = registry
        self.console = console
        self.system_prompt = system_prompt
        self.tools = tools
        self.effort = effort or config.effort
        self.budget_usd = budget_usd if budget_usd is not None else config.budget_usd_per_cycle
        self.max_steps = max_steps or config.max_steps

    # ------------------------------------------------------------------ run
    def run(self, brief: str) -> AgentResult:
        messages: list[dict] = [{"role": "user", "content": [{"type": "text", "text": brief}]}]
        usage = TokenUsage()
        cost = 0.0
        turns: list[TurnRecord] = []
        step = 0
        status = ""
        summary = ""
        final_text = ""
        refusal = None
        error = None
        any_fallback = False
        nudges = 0
        stop_note = ""

        while True:
            if step >= self.max_steps:
                status, stop_note = "step_limit", f"step limit of {self.max_steps} reached"
                break
            if cost >= self.budget_usd:
                status, stop_note = "wrapped_up_on_budget", f"budget of ${self.budget_usd:.2f} reached (${cost:.2f} spent)"
                break
            step += 1
            try:
                msg = self.llm.respond(system=self.system_prompt, tools=self.tools, messages=messages,
                                       effort=self.effort, on_event=self.console.on_event)
            except API_ERRORS as e:
                error = f"{type(e).__name__}: {e}"
                log.error("model request failed at step %d: %s", step, error)
                status, stop_note = "error", error
                break

            turn_usage = usage_dict(msg)
            model = getattr(msg, "model", None) or self.config.model
            turn_cost = cost_usd(turn_usage, model, self.config)
            usage.add(turn_usage)
            cost += turn_cost
            content = list(getattr(msg, "content", None) or [])
            last_fb = max((i for i, b in enumerate(content) if block_type(b) == "fallback"), default=-1)
            fallback_ran = served_by_fallback(msg) or last_fb >= 0
            any_fallback = any_fallback or fallback_ran
            tool_blocks = [b for i, b in enumerate(content) if block_type(b) == "tool_use" and i > last_fb]
            text_blocks = [getattr(b, "text", "") for b in content if block_type(b) == "text"]
            thinking = [getattr(b, "thinking", "") for b in content if block_type(b) == "thinking" and getattr(b, "thinking", "")]
            stop_reason = getattr(msg, "stop_reason", None)
            turn = TurnRecord(step=step, stop_reason=stop_reason, stop_details=getattr(msg, "stop_details", None),
                              model=model, served_by_fallback=fallback_ran, fallback_block=last_fb >= 0,
                              usage=turn_usage, cost_usd=turn_cost, thinking=thinking, text=text_blocks)
            turns.append(turn)
            if fallback_ran:
                log.info("step %d served by a fallback model (%s)", step, model)
            log.info("step %d: stop_reason=%s tools=%d cost=$%.3f (cycle $%.3f)", step, stop_reason, len(tool_blocks),
                     turn_cost, cost)

            if stop_reason == "refusal":
                details = to_jsonable(getattr(msg, "stop_details", None))
                refusal = {"model": model, "stop_details": details, "step": step}
                log.warning("the model declined this cycle (refusal): %s", details)
                self.console.status(f"the model declined the request (refusal){'; details: ' + str(details) if details else ''}")
                status, stop_note = "refused", "the model's safety classifiers declined the request"
                break

            messages.append({"role": "assistant", "content": self._echo(content, last_fb)})

            if stop_reason == "pause_turn":
                continue

            if stop_reason == "max_tokens":
                if tool_blocks:
                    results = []
                    for b in tool_blocks:
                        results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": CUT_OFF_RESULT})
                        turn.tool_calls.append({"tool_use_id": b.id, "name": b.name, "input": to_jsonable(b.input),
                                                "skipped": "input cut off at max_tokens"})
                    messages.append({"role": "user", "content": results})
                else:
                    turn.note = CUT_OFF_NOTE
                    messages.append({"role": "user", "content": [{"type": "text", "text": CUT_OFF_NOTE}]})
                continue

            if not tool_blocks:
                final_text = "\n".join(t for t in text_blocks if t).strip()
                if not final_text and nudges < MAX_NO_TOOL_NUDGES and step < self.max_steps:
                    nudges += 1
                    turn.note = NO_TOOL_NOTE
                    messages.append({"role": "user", "content": [{"type": "text", "text": NO_TOOL_NOTE}]})
                    continue
                status = "completed"
                stop_note = "the model ended its turn without calling conclude_cycle; its final text is the summary"
                summary = final_text
                break

            results, concluded = self._run_tools(tool_blocks, step, turn)
            if concluded:
                messages.append({"role": "user", "content": results})
                status = "completed"
                summary = self.registry.staged.conclusion["summary"]
                break
            reason = self._wrap_up_reason(step, cost)
            if reason:
                note = WRAP_UP_NOTE.format(reason=reason)
                turn.note = note
                results.append({"type": "text", "text": note})
            messages.append({"role": "user", "content": results})

        if not summary:
            summary = self._fallback_summary(status, turns, final_text, stop_note)
        next_focus = list((self.registry.staged.conclusion or {}).get("next_focus") or [])
        return AgentResult(status=status, summary=summary, concluded=self.registry.concluded, next_focus=next_focus,
                           turns=turns, messages=messages, usage=usage, cost_usd=cost, steps=step,
                           final_text=final_text, refusal=refusal, error=error, served_by_fallback=any_fallback,
                           stop_note=stop_note)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _echo(content: list, last_fb: int) -> list:
        """The assistant turn to append: as received, or filtered after a mid-output fallback."""
        if last_fb < 0:
            return content
        kept = []
        for i, b in enumerate(content):
            if i >= last_fb or block_type(b) in ("text", "fallback"):
                kept.append(b)
        return kept

    def _run_tools(self, tool_blocks: list, step: int, turn: TurnRecord) -> tuple[list[dict], bool]:
        results = []
        concluded = False
        for b in tool_blocks:
            outcome = self.registry.execute(b.name, b.input, step)
            self.console.tool_call(b.name, b.input, outcome.evidence_id, outcome.is_error, len(outcome.content))
            if outcome.is_error:
                log.info("tool %s returned an error: %s", b.name, outcome.content[:200])
            results.append(outcome.tool_result(b.id))
            turn.tool_calls.append(outcome.as_dict(b.id))
            if b.name == "conclude_cycle" and not outcome.is_error:
                concluded = True
        return results, concluded

    def _wrap_up_reason(self, step: int, cost: float) -> str | None:
        remaining = self.max_steps - step
        reasons = []
        if remaining <= WRAP_UP_STEPS_BEFORE_LIMIT:
            reasons.append(f"{remaining} step{'s' if remaining != 1 else ''} left before the step limit")
        if cost >= WRAP_UP_BUDGET_FRACTION * self.budget_usd:
            reasons.append(f"${cost:.2f} of the ${self.budget_usd:.2f} cycle budget spent")
        return " and ".join(reasons) if reasons else None

    @staticmethod
    def _fallback_summary(status: str, turns: list[TurnRecord], final_text: str, stop_note: str) -> str:
        if final_text:
            return final_text
        last_text = ""
        for turn in reversed(turns):
            joined = "\n".join(t for t in turn.text if t).strip()
            if joined:
                last_text = joined
                break
        head = {
            "wrapped_up_on_budget": "The cycle stopped on its cost budget before the Mind concluded",
            "step_limit": "The cycle stopped at its step limit before the Mind concluded",
            "refused": "The model declined this cycle, so no investigation was run",
            "error": "The cycle failed with a model error",
        }.get(status, "The cycle ended without a conclusion")
        text = f"{head} ({stop_note})." if stop_note else f"{head}."
        if last_text:
            text += f" The model's last message was: {last_text}"
        return text
