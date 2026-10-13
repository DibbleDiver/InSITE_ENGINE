"""Offline test doubles: fake content blocks, fake messages and a scripted model.

`ScriptedLLM` replays a list of pre-built responses and records a deep copy of
every request it receives, so tests can assert on the exact conversation the
harness sent (system, tools, messages, effort, output format).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .serialize import messages_to_dicts


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ThinkingBlock:
    thinking: str = ""
    signature: str = "fake-signature"
    type: str = "thinking"


@dataclass
class RedactedThinkingBlock:
    data: str = "redacted"
    type: str = "redacted_thinking"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: Any
    type: str = "tool_use"


@dataclass
class FallbackInfo:
    model: str


@dataclass
class FallbackBlock:
    from_: FallbackInfo
    to: FallbackInfo
    type: str = "fallback"


@dataclass
class FakeIteration:
    type: str
    model: str


@dataclass
class FakeUsage:
    input_tokens: int = 1000
    output_tokens: int = 200
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    iterations: list | None = None


@dataclass
class FakeStopDetails:
    category: str | None = None
    explanation: str | None = None
    type: str = "refusal"


@dataclass
class FakeMessage:
    content: list
    stop_reason: str = "end_turn"
    stop_details: Any = None
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "claude-fable-5-1"
    id: str = "msg_fake"
    role: str = "assistant"
    type: str = "message"


_counter = {"n": 0}


def tool_use(name: str, input: Any, id: str | None = None) -> ToolUseBlock:
    if id is None:
        _counter["n"] += 1
        id = f"toolu_fake_{_counter['n']:04d}"
    return ToolUseBlock(id=id, name=name, input=input)


def text(s: str) -> TextBlock:
    return TextBlock(text=s)


def thinking(s: str = "") -> ThinkingBlock:
    return ThinkingBlock(thinking=s)


def fallback(from_model: str, to_model: str) -> FallbackBlock:
    return FallbackBlock(from_=FallbackInfo(from_model), to=FallbackInfo(to_model))


def message(*blocks: Any, stop_reason: str | None = None, usage: FakeUsage | None = None, **kwargs) -> FakeMessage:
    """Build a fake response; stop_reason defaults to tool_use when a tool call is present."""
    content = list(blocks)
    if stop_reason is None:
        stop_reason = "tool_use" if any(getattr(b, "type", None) == "tool_use" for b in content) else "end_turn"
    return FakeMessage(content=content, stop_reason=stop_reason, usage=usage or FakeUsage(), **kwargs)


def conclude(summary: str, next_focus: list[str] | None = None, *extra_blocks: Any) -> FakeMessage:
    return message(*extra_blocks, tool_use("conclude_cycle", {"summary": summary, "next_focus": next_focus or []}))


def critic_message(verdicts: list[dict], follow_up_questions: list[str] | None = None) -> FakeMessage:
    payload = {"verdicts": verdicts, "follow_up_questions": follow_up_questions or []}
    return message(thinking("reviewing"), text(json.dumps(payload)), stop_reason="end_turn")


class ScriptExhausted(RuntimeError):
    pass


class ScriptedLLM:
    """Replays scripted responses in order; records every request.

    Script items are FakeMessage objects, or callables taking the recorded request
    dict and returning a FakeMessage (for responses that depend on the request).
    """

    def __init__(self, script: list[Any] | None = None):
        self.script = list(script or [])
        self.requests: list[dict] = []
        self.responses: list[Any] = []

    def extend(self, items: list[Any]) -> None:
        self.script.extend(items)

    def respond(
        self,
        *,
        system: str,
        tools: list[dict] | None,
        messages: list[dict],
        effort: str,
        output_format: dict | None = None,
        on_event: Callable[[str, str], None] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        request = {
            "system": system,
            "tools": copy.deepcopy(tools),
            "messages": copy.deepcopy(messages_to_dicts(messages)),
            "effort": effort,
            "output_format": copy.deepcopy(output_format),
            "model": model,
            "max_tokens": max_tokens,
        }
        self.requests.append(request)
        if not self.script:
            raise ScriptExhausted(f"ScriptedLLM has no response left for request #{len(self.requests)}")
        item = self.script.pop(0)
        if callable(item):
            item = item(request)
        if isinstance(item, Exception):
            raise item
        if on_event is not None:
            self._replay_events(item, on_event)
        self.responses.append(item)
        return item

    @staticmethod
    def _replay_events(msg: Any, on_event: Callable[[str, str], None]) -> None:
        on_event("message_start", getattr(msg, "model", "") or "")
        for block in getattr(msg, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                on_event("thinking_start", "")
                if getattr(block, "thinking", ""):
                    on_event("thinking", block.thinking)
                on_event("block_stop", "thinking")
            elif btype == "text":
                on_event("text_start", "")
                on_event("text", block.text)
                on_event("block_stop", "text")
            elif btype == "tool_use":
                on_event("tool_use_start", block.name)
                on_event("block_stop", "tool_use")
            elif btype == "fallback":
                on_event("fallback", f"{block.from_.model} -> {block.to.model}")
