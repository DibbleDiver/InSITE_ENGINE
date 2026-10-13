"""Readable console progress: thinking summaries and text as they stream,
tool calls as they run, send_to_user messages verbatim."""
from __future__ import annotations

import logging
import sys
from typing import Any, TextIO

log = logging.getLogger("insite.mind")


class Console:
    def __init__(self, stream: TextIO | None = None, quiet: bool = False, stream_thinking: bool = True):
        self.stream = stream or sys.stdout
        self.quiet = quiet
        self.stream_thinking = stream_thinking
        self._block: str | None = None
        self._prefix_pending = False
        self._line_open = False
        self.lines: list[str] = []

    def _write(self, text: str) -> None:
        if self.quiet:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):  # closed stream; never let printing kill a cycle
            pass

    def _prefix(self, text: str) -> None:
        if self._prefix_pending:
            self._write(text)
            self._prefix_pending = False

    def _end_line(self) -> None:
        if self._line_open:
            self._write("\n")
            self._line_open = False

    def line(self, text: str) -> None:
        self._end_line()
        self.lines.append(text)
        self._write(text + "\n")

    def status(self, text: str) -> None:
        log.info(text)
        self.line(f"[mind] {text}")

    def user_message(self, text: str) -> None:
        self._end_line()
        self.lines.append(f"[to operations team] {text}")
        self._write(f"\n[to operations team] {text}\n\n")

    def tool_call(self, name: str, args: Any, evidence_id: str | None, is_error: bool, chars: int) -> None:
        from .serialize import dumps

        rendered = dumps(args)
        if len(rendered) > 140:
            rendered = rendered[:137] + "..."
        tag = "ERROR" if is_error else (evidence_id or "ok")
        self.line(f"[tool] {name}({rendered}) -> {tag} ({chars} chars)")

    # streaming events from the model client ----------------------------------
    def on_event(self, kind: str, payload: str) -> None:
        if kind == "thinking_start":
            self._end_line()
            self._block, self._prefix_pending = "thinking", True
        elif kind == "thinking":
            # with thinking display "omitted" the block has no text: print nothing, not a bare prefix
            if self.stream_thinking and payload:
                self._prefix("[thinking] ")
                self._write(payload)
                self._line_open = True
        elif kind == "text_start":
            self._end_line()
            self._block, self._prefix_pending = "text", True
        elif kind == "text":
            if payload:
                self._prefix("[model] ")
                self._write(payload)
                self._line_open = True
        elif kind == "block_stop":
            self._end_line()
            self._block = None
        elif kind == "tool_use_start":
            self._end_line()
            self._write(f"[calling {payload}]\n")
        elif kind == "fallback":
            self.line(f"[fallback] request re-run on a fallback model: {payload}")
        elif kind == "message_start":
            self._end_line()
