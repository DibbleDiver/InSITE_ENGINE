"""Model client boundary: a tiny interface, the real Anthropic implementation
and the token/cost bookkeeping shared by both.

Everything the rest of the Mind needs from a model is `respond(...)`, which
returns a message-like object with `.content` blocks (each with `.type` and,
depending on the type, `.text` / `.thinking` / `.id` / `.name` / `.input`),
`.stop_reason`, `.stop_details`, `.usage` and `.model`. `mind.testing.ScriptedLLM`
implements the same interface for offline tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .config import MindConfig

log = logging.getLogger("insite.mind.llm")

OnEvent = Callable[[str, str], None]


class MindUnavailable(RuntimeError):
    """Raised when the Mind cannot reach a model (no credentials, no SDK)."""


class LLM(Protocol):
    def respond(
        self,
        *,
        system: str,
        tools: list[dict] | None,
        messages: list[dict],
        effort: str,
        output_format: dict | None = None,
        on_event: OnEvent | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any: ...


@dataclass
class TokenUsage:
    """Accumulated token usage; the four counters the API reports plus a call count."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls: int = 0

    def add(self, usage: dict | None) -> None:
        if not usage:
            return
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.cache_read_input_tokens += int(usage.get("cache_read_input_tokens") or 0)
        self.cache_creation_input_tokens += int(usage.get("cache_creation_input_tokens") or 0)
        self.calls += 1

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "calls": self.calls,
        }


def usage_dict(message: Any) -> dict:
    """The four token counters of a response as plain ints (None -> 0)."""
    usage = getattr(message, "usage", None)
    out = {}
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, key, None) if usage is not None else None
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        out[key] = int(value or 0)
    return out


def cost_usd(usage: dict, model: str | None, config: MindConfig) -> float:
    prices = config.prices_for(model)
    return (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("output_tokens", 0) * prices["output"]
        + usage.get("cache_creation_input_tokens", 0) * prices["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * prices["cache_read"]
    ) / 1_000_000


def served_by_fallback(message: Any) -> bool:
    """True when a fallback model produced the response (per usage.iterations)."""
    usage = getattr(message, "usage", None)
    iterations = getattr(usage, "iterations", None) if usage is not None else None
    return any(getattr(it, "type", None) == "fallback_message" for it in (iterations or []))


class AnthropicLLM:
    """Streams every request through `client.beta.messages.stream` (Claude Fable 5.1 shape)."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"
    MAX_JSON_RETRIES = 2

    def __init__(self, config: MindConfig, client: Any = None):
        self.config = config
        try:
            import anthropic  # noqa: F401  (import here so the SDK is optional at import time)
        except ImportError as e:  # pragma: no cover - the SDK is a declared dependency
            raise MindUnavailable("the `anthropic` package is not installed (pip install anthropic)") from e
        self._anthropic = anthropic
        if client is None:
            try:
                client = anthropic.Anthropic()
            except Exception as e:
                raise MindUnavailable(f"could not create the Anthropic client: {e}") from e
        if not self.has_credentials(client):
            raise MindUnavailable(
                "no Anthropic credentials found. Set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN), "
                "or run `ant auth login`, then retry; `python main.py think --offline` runs a "
                "perception-only cycle without a model."
            )
        self.client = client

    @staticmethod
    def has_credentials(client: Any) -> bool:
        if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
            return True
        # Profiles from `ant auth login` / workload identity resolve into a token cache
        # that injects the Authorization header per request.
        return getattr(client, "_token_cache", None) is not None

    def respond(
        self,
        *,
        system: str,
        tools: list[dict] | None,
        messages: list[dict],
        effort: str,
        output_format: dict | None = None,
        on_event: OnEvent | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = dict(
            model=model or self.config.model,
            max_tokens=max_tokens or self.config.max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive", "display": self.config.thinking_display},
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},
        )
        if tools:
            kwargs["tools"] = tools
        if output_format is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": output_format}
        if self.config.fallbacks == "default":
            kwargs["betas"] = [self.FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        attempt = 0
        while True:
            try:
                with self.client.beta.messages.stream(**kwargs) as stream:
                    for event in stream:
                        if on_event is not None:
                            self._dispatch_event(event, on_event)
                    return stream.get_final_message()
            except ValueError as e:
                # The SDK's stream iterator raises ValueError for tool-input JSON it cannot
                # parse at all (eager input streaming). API errors are APIError subclasses,
                # not ValueError, and propagate to the cycle.
                attempt += 1
                if attempt > self.MAX_JSON_RETRIES:
                    raise
                log.warning("unparseable tool input from the model (%s); re-issuing request (%d/%d)",
                            e, attempt, self.MAX_JSON_RETRIES)

    @staticmethod
    def _dispatch_event(event: Any, on_event: OnEvent) -> None:
        etype = getattr(event, "type", None)
        if etype == "thinking":
            on_event("thinking", event.thinking)
        elif etype == "text":
            on_event("text", event.text)
        elif etype == "content_block_start":
            block = getattr(event, "content_block", None)
            btype = getattr(block, "type", None)
            if btype == "thinking":
                on_event("thinking_start", "")
            elif btype == "text":
                on_event("text_start", "")
            elif btype == "tool_use":
                on_event("tool_use_start", getattr(block, "name", "") or "")
            elif btype == "fallback":
                frm = getattr(getattr(block, "from_", None), "model", "?")
                to = getattr(getattr(block, "to", None), "model", "?")
                on_event("fallback", f"{frm} -> {to}")
        elif etype == "content_block_stop":
            block = getattr(event, "content_block", None)
            on_event("block_stop", getattr(block, "type", "") or "")
        elif etype == "message_start":
            msg = getattr(event, "message", None)
            on_event("message_start", getattr(msg, "model", "") or "")
