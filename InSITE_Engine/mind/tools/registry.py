"""Validate every tool call against its schema, dispatch it, and tag data
results with evidence ids."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import jsonschema

from ...contracts import PipelineResult
from ..evidence import EvidenceLedger
from ..memory import MemoryStore, StagedChanges
from ..serialize import dumps
from .data import MAX_RESULT_CHARS, DataTools, ToolError, truncate
from .memory_tools import MemoryTools
from .schemas import SPECS_BY_NAME, build_tool_definitions

log = logging.getLogger("insite.mind.tools")


@dataclass
class ToolOutcome:
    name: str
    input: Any
    content: str
    is_error: bool
    kind: str
    evidence_id: str | None = None
    elapsed_ms: float = 0.0
    truncated: bool = False

    def tool_result(self, tool_use_id: str) -> dict:
        block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": self.content}
        if self.is_error:
            block["is_error"] = True
        return block

    def as_dict(self, tool_use_id: str | None = None) -> dict:
        return {
            "tool_use_id": tool_use_id,
            "name": self.name,
            "input": self.input,
            "kind": self.kind,
            "is_error": self.is_error,
            "evidence_id": self.evidence_id,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "truncated": self.truncated,
            "result_excerpt": self.content[:600],
        }


def invalid_input_result(raw_input: Any, reason: str) -> str:
    try:
        encoded = json.dumps(raw_input, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        encoded = repr(raw_input)
    return json.dumps({"INVALID_JSON": encoded, "error": reason}, ensure_ascii=False)


class ToolRegistry:
    def __init__(
        self,
        result: PipelineResult,
        staged: StagedChanges,
        store: MemoryStore,
        ledger: EvidenceLedger,
        *,
        new_event_ids: set[str] | None = None,
        on_send_to_user: Callable[[str], None] | None = None,
    ):
        self.data = DataTools(result, new_event_ids)
        self.memory = MemoryTools(staged, store, ledger)
        self.staged = staged
        self.ledger = ledger
        self.on_send_to_user = on_send_to_user
        self.sent_messages: list[str] = []
        self.concluded = False
        self._validators = {
            name: jsonschema.Draft202012Validator(spec["input_schema"]) for name, spec in SPECS_BY_NAME.items()
        }

    @staticmethod
    def definitions(eager_input_streaming: bool) -> list[dict]:
        return build_tool_definitions(eager_input_streaming)

    def validate(self, name: str, raw_input: Any) -> str | None:
        """None when the input is valid, else the reason it is not."""
        validator = self._validators.get(name)
        if validator is None:
            return f"unknown tool {name!r}"
        if not isinstance(raw_input, dict):
            return f"tool input must be a JSON object, got {type(raw_input).__name__}"
        errors = sorted(validator.iter_errors(raw_input), key=lambda e: list(e.path))
        if not errors:
            return None
        parts = []
        for err in errors[:4]:
            where = "/".join(str(p) for p in err.path) or "(root)"
            parts.append(f"{where}: {err.message}")
        return "; ".join(parts)

    def execute(self, name: str, raw_input: Any, step: int = 0) -> ToolOutcome:
        spec = SPECS_BY_NAME.get(name)
        kind = spec["kind"] if spec else "unknown"
        started = time.perf_counter()
        reason = self.validate(name, raw_input)
        if reason is not None:
            log.warning("tool %s rejected: %s", name, reason)
            return ToolOutcome(name, raw_input, invalid_input_result(raw_input, reason), True, kind,
                               elapsed_ms=(time.perf_counter() - started) * 1000)
        args = dict(raw_input)
        try:
            if kind == "data":
                output = getattr(self.data, name)(args)
                evidence = self.ledger.add(name, args, output, step)
                shown, truncated = truncate(output, MAX_RESULT_CHARS)
                content = f"{evidence.label()}\n{shown}"
                return ToolOutcome(name, args, content, False, kind, evidence_id=evidence.id,
                                   elapsed_ms=(time.perf_counter() - started) * 1000, truncated=truncated)
            if kind == "memory":
                output = getattr(self.memory, name)(args)
            elif name == "send_to_user":
                output = self._send_to_user(args["message"])
            elif name == "conclude_cycle":
                output = self._conclude(args)
            else:  # pragma: no cover - schemas cover every spec
                raise ToolError(f"tool {name} has no implementation")
            shown, truncated = truncate(output, MAX_RESULT_CHARS)
            return ToolOutcome(name, args, shown, False, kind, elapsed_ms=(time.perf_counter() - started) * 1000,
                               truncated=truncated)
        except ToolError as e:
            return ToolOutcome(name, args, f"error: {e}", True, kind, elapsed_ms=(time.perf_counter() - started) * 1000)
        except Exception as e:  # a bug in a tool must not kill the cycle
            log.exception("tool %s failed on %s", name, dumps(args)[:300])
            return ToolOutcome(name, args, f"error: {name} failed internally ({type(e).__name__}: {e}); "
                               f"try a different query", True, kind, elapsed_ms=(time.perf_counter() - started) * 1000)

    def _send_to_user(self, message: str) -> str:
        self.sent_messages.append(message)
        if self.on_send_to_user is not None:
            self.on_send_to_user(message)
        return "delivered"

    def _conclude(self, args: dict) -> str:
        summary = (args.get("summary") or "").strip()
        if not summary:
            raise ToolError("summary must not be empty")
        self.staged.stage_conclusion(summary, [str(x) for x in (args.get("next_focus") or [])])
        self.concluded = True
        return "cycle concluded; the summary and next focus are recorded"
