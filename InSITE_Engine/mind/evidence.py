"""Per-cycle evidence ledger: every data-tool result gets an id (E1, E2, ...)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .serialize import dumps

EXCERPT_CHARS = 1500


@dataclass
class Evidence:
    id: str
    tool: str
    input: dict
    excerpt: str
    chars: int
    truncated: bool
    step: int

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "tool": self.tool,
            "input": self.input,
            "excerpt": self.excerpt,
            "chars": self.chars,
            "truncated": self.truncated,
            "step": self.step,
        }

    def label(self) -> str:
        args = dumps(self.input)
        if len(args) > 160:
            args = args[:157] + "..."
        return f"{self.id} = {self.tool}({args})"


@dataclass
class EvidenceLedger:
    max_excerpt: int = EXCERPT_CHARS
    items: dict[str, Evidence] = field(default_factory=dict)
    _count: int = 0

    def add(self, tool: str, input: dict, output: str, step: int) -> Evidence:
        self._count += 1
        eid = f"E{self._count}"
        truncated = len(output) > self.max_excerpt
        excerpt = output[: self.max_excerpt] + (" ...[excerpt truncated]" if truncated else "")
        ev = Evidence(id=eid, tool=tool, input=input, excerpt=excerpt, chars=len(output), truncated=truncated, step=step)
        self.items[eid] = ev
        return ev

    def resolve(self, ids: list[str]) -> tuple[list[Evidence], list[str]]:
        known, unknown = [], []
        seen = set()
        for raw in ids or []:
            eid = str(raw).strip().upper()
            if eid in seen:
                continue
            seen.add(eid)
            if eid in self.items:
                known.append(self.items[eid])
            else:
                unknown.append(str(raw))
        return known, unknown

    def excerpts_for(self, ids: list[str], chars: int = 300) -> list[dict]:
        known, _ = self.resolve(ids)
        return [{"id": e.id, "tool": e.tool, "excerpt": e.excerpt[:chars]} for e in known]

    def as_dict(self) -> dict:
        return {eid: ev.as_dict() for eid, ev in self.items.items()}

    def __len__(self) -> int:
        return len(self.items)
