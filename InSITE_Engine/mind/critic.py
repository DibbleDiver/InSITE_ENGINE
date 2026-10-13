"""Critique: a fresh-context, tool-less model call that grades every claim of
the cycle against the evidence excerpts it cites. Verdicts are applied
deterministically by MemoryStore.commit; this module only produces them."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field


import jsonschema

from .config import MindConfig
from .console import Console
from .evidence import EvidenceLedger
from .llm import LLM, cost_usd, usage_dict
from .memory import StagedChanges
from .prompts import CRITIC_SYSTEM_PROMPT
from .serialize import to_jsonable

log = logging.getLogger("insite.mind.critic")

try:
    import anthropic as _anthropic

    API_ERRORS: tuple[type[BaseException], ...] = (_anthropic.APIStatusError, _anthropic.APIConnectionError)
except ImportError:  # pragma: no cover
    API_ERRORS = ()

VERDICTS = ("supported", "weak", "unsupported")
MAX_FOLLOW_UPS = 5
EXCERPT_CHARS = 1200

# Structured-output schema: every object closed (additionalProperties false) with
# all properties required; no min/max or length constraints (not supported).
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "reason": {"type": "string"},
                },
                "required": ["claim_id", "verdict", "reason"],
                "additionalProperties": False,
            },
        },
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdicts", "follow_up_questions"],
    "additionalProperties": False,
}


@dataclass
class CriticResult:
    status: str  # reviewed | unreviewed | skipped
    verdicts: dict[str, dict] = field(default_factory=dict)
    follow_ups: list[str] = field(default_factory=list)
    raw: str | None = None
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    model: str | None = None
    error: str | None = None
    claims: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        out = {v: 0 for v in VERDICTS}
        out["unreviewed"] = 0
        for claim in self.claims:
            verdict = self.verdicts.get(claim["claim_id"], {}).get("verdict", "unreviewed")
            out[verdict if verdict in out else "unreviewed"] += 1
        return out

    def as_dict(self) -> dict:
        return {
            "status": self.status, "verdicts": self.verdicts, "follow_up_questions": self.follow_ups,
            "raw": self.raw, "usage": self.usage, "cost_usd": round(self.cost_usd, 6), "model": self.model,
            "error": self.error, "counts": self.counts(), "claims": self.claims,
        }


def build_claims(staged: StagedChanges, ledger: EvidenceLedger) -> list[dict]:
    claims = []
    for entry in staged.claims:
        info = staged.claim_text(entry["claim_id"])
        if info is None:
            continue
        claims.append({
            "claim_id": entry["claim_id"],
            "kind": info["kind"],
            "text": info["text"],
            "evidence": ledger.excerpts_for(info["evidence_ids"], chars=EXCERPT_CHARS),
        })
    return claims


def render_claims(claims: list[dict], goal: str | None) -> str:
    lines = ["Claims made by the Mind this cycle, each with excerpts of the evidence it cited."]
    if goal:
        lines.append(f"The cycle's goal was: {goal}")
    lines.append("Grade every claim id listed here.\n")
    for c in claims:
        lines.append(f"### Claim {c['claim_id']} ({c['kind']})")
        lines.append(c["text"])
        if c["evidence"]:
            lines.append("Cited evidence:")
            for e in c["evidence"]:
                lines.append(f"[{e['id']}] from tool {e['tool']}:\n{e['excerpt']}")
        else:
            lines.append("Cited evidence: none could be resolved.")
        lines.append("")
    return "\n".join(lines)


def run_critic(llm: LLM, config: MindConfig, staged: StagedChanges, ledger: EvidenceLedger,
               console: Console | None = None, goal: str | None = None) -> CriticResult:
    claims = build_claims(staged, ledger)
    result = CriticResult(status="skipped", claims=claims, model=config.critic_model)
    if not claims:
        return result
    prompt = render_claims(claims, goal)
    try:
        msg = llm.respond(system=CRITIC_SYSTEM_PROMPT, tools=None,
                          messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                          effort=config.critic_effort, output_format=CRITIC_SCHEMA, on_event=None,
                          model=config.critic_model)
    except API_ERRORS as e:
        result.status, result.error = "unreviewed", f"critic request failed: {type(e).__name__}: {e}"
        log.error(result.error)
        return result
    result.usage = usage_dict(msg)
    result.model = getattr(msg, "model", None) or config.critic_model
    result.cost_usd = cost_usd(result.usage, result.model, config)
    stop_reason = getattr(msg, "stop_reason", None)
    if stop_reason == "refusal":
        result.status, result.error = "unreviewed", f"critic declined (refusal): {to_jsonable(getattr(msg, 'stop_details', None))}"
        log.warning(result.error)
        return result
    if stop_reason == "max_tokens":
        result.status, result.error = "unreviewed", "critic output was cut off at max_tokens"
        log.warning(result.error)
        return result
    text = next((getattr(b, "text", "") for b in (getattr(msg, "content", None) or []) if getattr(b, "type", None) == "text"), "")
    result.raw = text
    try:
        data = json.loads(text)
        jsonschema.validate(data, CRITIC_SCHEMA)
    except (ValueError, jsonschema.ValidationError) as e:
        result.status, result.error = "unreviewed", f"critic returned invalid JSON: {e}"
        log.warning(result.error)
        return result
    known = {c["claim_id"] for c in claims}
    for v in data["verdicts"]:
        cid = str(v["claim_id"]).strip().upper()
        if cid in known and cid not in result.verdicts:
            result.verdicts[cid] = {"verdict": v["verdict"], "reason": v["reason"]}
    result.follow_ups = [q.strip() for q in data["follow_up_questions"] if q and q.strip()][:MAX_FOLLOW_UPS]
    result.status = "reviewed"
    missing = sorted(known - set(result.verdicts))
    if missing:
        log.info("critic gave no verdict for %s; treated as unreviewed", missing)
    if console is not None:
        counts = result.counts()
        console.status("critic: " + ", ".join(f"{n} {k}" for k, n in counts.items() if n))
    return result
