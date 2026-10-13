"""Durable memory of the Mind plus the per-cycle staging layer.

Layout under memory_dir:
    memory.json          hypotheses, questions, fingerprint, seen event ids, counters, agenda
    lessons/<name>.md    one durable lesson per file, one-line summary on the first line
    journal/cycle_NNNN.json
    reports/cycle_NNNN.md, reports/latest.md

Nothing the model does during a cycle touches disk directly: tools record
changes in a `StagedChanges` object, the critic grades the claims, and
`MemoryStore.commit` applies the graded changes in one go. Writes are atomic
(temp file + os.replace).
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

from . import hypotheses as hyp

log = logging.getLogger("insite.mind.memory")

LESSON_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,60}$")
MAX_OPEN_QUESTIONS = 20
MAX_CRITIC_QUESTIONS_PER_CYCLE = 5
MAX_AGENDA = 10
MAX_RECENT_INSIGHTS = 30

VERDICT_WEIGHTS = {"supported": 1.0, "unreviewed": 1.0, "weak": 0.5, "unsupported": 0.0}


def atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------- state


@dataclass
class Hypothesis:
    id: str
    statement: str
    probability: float = hyp.PRIOR
    status: str = "open"
    created_cycle: int = 0
    updated_cycle: int = 0
    retired_reason: str | None = None
    evidence_log: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    @property
    def uncertainty(self) -> float:
        return hyp.uncertainty(self.probability)

    def trend(self) -> str:
        points = [f"{h['probability']:.2f}" for h in self.history[-6:]]
        return " -> ".join(points) if points else f"{self.probability:.2f}"


@dataclass
class Question:
    id: str
    question: str
    context: str = ""
    status: str = "open"
    source: str = "mind"
    created_cycle: int = 0
    resolved_cycle: int | None = None
    answer: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    tentative: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class MemoryState:
    cycle: int = 0
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    questions: dict[str, Question] = field(default_factory=dict)
    fingerprint: dict | None = None
    seen_event_ids: list[str] = field(default_factory=list)
    agenda: list[str] = field(default_factory=list)
    recent_insights: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: {"hypothesis": 0, "question": 0})
    last_cycle_status: str | None = None
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "hypotheses": {k: asdict(v) for k, v in self.hypotheses.items()},
            "questions": {k: asdict(v) for k, v in self.questions.items()},
            "fingerprint": self.fingerprint,
            "seen_event_ids": list(self.seen_event_ids),
            "agenda": list(self.agenda),
            "recent_insights": list(self.recent_insights),
            "counters": dict(self.counters),
            "last_cycle_status": self.last_cycle_status,
            "total_cost_usd": self.total_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryState":
        state = cls()
        state.cycle = int(data.get("cycle", 0))
        state.hypotheses = {k: Hypothesis(**v) for k, v in (data.get("hypotheses") or {}).items()}
        state.questions = {k: Question(**v) for k, v in (data.get("questions") or {}).items()}
        state.fingerprint = data.get("fingerprint")
        state.seen_event_ids = list(data.get("seen_event_ids") or [])
        state.agenda = list(data.get("agenda") or [])
        state.recent_insights = list(data.get("recent_insights") or [])
        state.counters = {"hypothesis": 0, "question": 0, **(data.get("counters") or {})}
        state.last_cycle_status = data.get("last_cycle_status")
        state.total_cost_usd = float(data.get("total_cost_usd") or 0.0)
        return state


# --------------------------------------------------------------------------- staging


class StagedChanges:
    """Everything the model asked to remember this cycle, pending the critic."""

    def __init__(self, state: MemoryState, cycle: int):
        self.cycle = cycle
        self.counters = dict(state.counters)
        self._state = state
        self.new_hypotheses: dict[str, dict] = {}
        self.hypothesis_updates: list[dict] = []
        self.retirements: dict[str, str] = {}
        self.new_questions: dict[str, dict] = {}
        self.resolutions: list[dict] = []
        self.insights: list[dict] = []
        self.lessons_written: dict[str, tuple[str, str]] = {}
        self.lessons_deleted: set[str] = set()
        self.conclusion: dict | None = None
        self.claims: list[dict] = []
        self._claim_count = 0

    # ids -------------------------------------------------------------------
    def _next_id(self, kind: str, prefix: str) -> str:
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return f"{prefix}{self.counters[kind]}"

    def next_claim_id(self) -> str:
        self._claim_count += 1
        return f"C{self._claim_count}"

    # lookups ----------------------------------------------------------------
    def hypothesis_exists(self, hid: str) -> bool:
        return hid in self._state.hypotheses or hid in self.new_hypotheses

    def hypothesis_statement(self, hid: str) -> str:
        if hid in self.new_hypotheses:
            return self.new_hypotheses[hid]["statement"]
        return self._state.hypotheses[hid].statement

    def question_exists(self, qid: str) -> bool:
        return qid in self._state.questions or qid in self.new_questions

    def is_question_open(self, qid: str) -> bool:
        if qid in self.new_questions:
            return True
        q = self._state.questions.get(qid)
        return bool(q and q.status == "open")

    def provisional_probability(self, hid: str) -> float:
        """Probability as if every staged update were accepted in full."""
        base = self._state.hypotheses[hid].probability if hid in self._state.hypotheses else hyp.PRIOR
        p = base
        if hid in self.new_hypotheses:
            nh = self.new_hypotheses[hid]
            p = hyp.apply_update(p, nh["direction"], nh["strength"])
        for up in self.hypothesis_updates:
            if up["hypothesis_id"] == hid:
                p = hyp.apply_update(p, up["direction"], up["strength"])
        return p

    # staging ---------------------------------------------------------------
    def stage_hypothesis(self, statement: str, direction: str, strength: str, evidence_ids: list[str], note: str) -> dict:
        hid = self._next_id("hypothesis", "H")
        claim_id = self.next_claim_id()
        record = {
            "claim_id": claim_id, "id": hid, "statement": statement, "direction": direction,
            "strength": strength, "evidence_ids": list(evidence_ids), "note": note,
        }
        self.new_hypotheses[hid] = record
        self.claims.append({"claim_id": claim_id, "kind": "propose_hypothesis", "ref": hid})
        return record

    def stage_update(self, hid: str, direction: str, strength: str, evidence_ids: list[str], note: str) -> dict:
        claim_id = self.next_claim_id()
        record = {
            "claim_id": claim_id, "hypothesis_id": hid, "direction": direction, "strength": strength,
            "evidence_ids": list(evidence_ids), "note": note,
        }
        self.hypothesis_updates.append(record)
        self.claims.append({"claim_id": claim_id, "kind": "update_hypothesis", "ref": hid})
        return record

    def stage_retirement(self, hid: str, reason: str) -> None:
        self.retirements[hid] = reason

    def stage_question(self, question: str, context: str, source: str = "mind") -> dict:
        qid = self._next_id("question", "Q")
        record = {"id": qid, "question": question, "context": context, "source": source}
        self.new_questions[qid] = record
        return record

    def stage_resolution(self, qid: str, answer: str, evidence_ids: list[str]) -> dict:
        claim_id = self.next_claim_id()
        record = {"claim_id": claim_id, "question_id": qid, "answer": answer, "evidence_ids": list(evidence_ids)}
        self.resolutions.append(record)
        self.claims.append({"claim_id": claim_id, "kind": "resolve_question", "ref": qid})
        return record

    def stage_insight(self, title: str, finding: str, severity: str, evidence_ids: list[str],
                      recommended_action: str | None, hypothesis_id: str | None) -> dict:
        claim_id = self.next_claim_id()
        record = {
            "claim_id": claim_id, "title": title, "finding": finding, "severity": severity,
            "evidence_ids": list(evidence_ids), "recommended_action": recommended_action,
            "hypothesis_id": hypothesis_id,
        }
        self.insights.append(record)
        self.claims.append({"claim_id": claim_id, "kind": "record_insight", "ref": title})
        return record

    def stage_lesson(self, name: str, summary: str, body: str) -> None:
        self.lessons_written[name] = (summary, body)
        self.lessons_deleted.discard(name)

    def stage_lesson_deletion(self, name: str) -> None:
        self.lessons_deleted.add(name)
        self.lessons_written.pop(name, None)

    def stage_conclusion(self, summary: str, next_focus: list[str]) -> None:
        self.conclusion = {"summary": summary, "next_focus": list(next_focus)}

    def claim_text(self, claim_id: str) -> dict | None:
        """Human-readable description of a claim for the critic."""
        for nh in self.new_hypotheses.values():
            if nh["claim_id"] == claim_id:
                return {
                    "kind": "hypothesis proposal",
                    "text": f"Proposed hypothesis {nh['id']}: {nh['statement']} -- cited evidence "
                            f"{nh['direction']} it with {nh['strength']} strength. {nh['note'] or ''}".strip(),
                    "evidence_ids": nh["evidence_ids"],
                }
        for up in self.hypothesis_updates:
            if up["claim_id"] == claim_id:
                return {
                    "kind": "hypothesis update",
                    "text": f"Hypothesis {up['hypothesis_id']} ('{self.hypothesis_statement(up['hypothesis_id'])}'): "
                            f"cited evidence {up['direction']} it with {up['strength']} strength. {up['note'] or ''}".strip(),
                    "evidence_ids": up["evidence_ids"],
                }
        for r in self.resolutions:
            if r["claim_id"] == claim_id:
                qtext = self.new_questions.get(r["question_id"], {}).get("question") or (
                    self._state.questions[r["question_id"]].question if r["question_id"] in self._state.questions else "?")
                return {
                    "kind": "question resolution",
                    "text": f"Question {r['question_id']} ('{qtext}') resolved: {r['answer']}",
                    "evidence_ids": r["evidence_ids"],
                }
        for ins in self.insights:
            if ins["claim_id"] == claim_id:
                extra = f" Recommended action: {ins['recommended_action']}" if ins.get("recommended_action") else ""
                return {
                    "kind": f"insight ({ins['severity']})",
                    "text": f"{ins['title']}: {ins['finding']}{extra}",
                    "evidence_ids": ins["evidence_ids"],
                }
        return None

    def summary_counts(self) -> dict:
        return {
            "hypotheses_proposed": len(self.new_hypotheses),
            "hypothesis_updates": len(self.hypothesis_updates),
            "retirements": len(self.retirements),
            "questions_added": len(self.new_questions),
            "questions_resolved": len(self.resolutions),
            "insights": len(self.insights),
            "lessons_written": len(self.lessons_written),
            "lessons_deleted": len(self.lessons_deleted),
        }


# --------------------------------------------------------------------------- store


@dataclass
class CommitResult:
    insights_accepted: list[dict] = field(default_factory=list)
    insights_tentative: list[dict] = field(default_factory=list)
    insights_rejected: list[dict] = field(default_factory=list)
    hypothesis_changes: list[dict] = field(default_factory=list)
    questions_added: list[dict] = field(default_factory=list)
    questions_resolved: list[dict] = field(default_factory=list)
    questions_unresolved: list[dict] = field(default_factory=list)
    lessons_written: list[str] = field(default_factory=list)
    lessons_deleted: list[str] = field(default_factory=list)
    agenda: list[str] = field(default_factory=list)


class MemoryStore:
    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        self.path = os.path.join(memory_dir, "memory.json")
        self.lessons_dir = os.path.join(memory_dir, "lessons")
        self.journal_dir = os.path.join(memory_dir, "journal")
        self.reports_dir = os.path.join(memory_dir, "reports")
        for d in (memory_dir, self.lessons_dir, self.journal_dir, self.reports_dir):
            os.makedirs(d, exist_ok=True)
        self.state = self.load()

    # persistence -----------------------------------------------------------
    def load(self) -> MemoryState:
        if not os.path.exists(self.path):
            return MemoryState()
        try:
            with open(self.path, encoding="utf-8") as f:
                return MemoryState.from_dict(json.load(f))
        except (OSError, ValueError, TypeError) as e:
            log.error("memory.json is unreadable (%s); starting with empty memory", e)
            return MemoryState()

    def save(self) -> None:
        atomic_write_json(self.path, self.state.to_dict())

    def reload(self) -> MemoryState:
        self.state = self.load()
        return self.state

    # lessons ---------------------------------------------------------------
    @staticmethod
    def valid_lesson_name(name: str) -> bool:
        return bool(LESSON_NAME_RE.match(name or ""))

    def _lesson_path(self, name: str) -> str:
        if not self.valid_lesson_name(name):
            raise ValueError("lesson names are slugs: lowercase letters, digits, '-' and '_' (max 61 chars)")
        return os.path.join(self.lessons_dir, f"{name}.md")

    def list_lessons(self) -> list[dict]:
        out = []
        for fn in sorted(os.listdir(self.lessons_dir)):
            if not fn.endswith(".md"):
                continue
            name = fn[:-3]
            summary = ""
            try:
                with open(os.path.join(self.lessons_dir, fn), encoding="utf-8") as f:
                    summary = f.readline().strip().lstrip("#").strip()
            except OSError:
                continue
            out.append({"name": name, "summary": summary})
        return out

    def read_lesson(self, name: str) -> str | None:
        path = self._lesson_path(name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    def write_lesson(self, name: str, summary: str, body: str) -> str:
        path = self._lesson_path(name)
        text = f"# {summary.strip()}\n\n{body.strip()}\n"
        atomic_write_text(path, text)
        return path

    def delete_lesson(self, name: str) -> bool:
        path = self._lesson_path(name)
        if os.path.exists(path):
            os.unlink(path)
            return True
        return False

    # digest ----------------------------------------------------------------
    def digest(self, max_hypotheses: int = 8, max_questions: int = 8) -> dict:
        st = self.state
        open_h = [h for h in st.hypotheses.values() if h.status == "open"]
        open_h.sort(key=lambda h: (-h.uncertainty, h.updated_cycle, h.id))
        settled = [h for h in st.hypotheses.values() if h.status in ("supported", "refuted")]
        settled.sort(key=lambda h: (-h.updated_cycle, h.id))
        retired = [h for h in st.hypotheses.values() if h.status == "retired"]
        open_q = [q for q in st.questions.values() if q.status == "open"]
        open_q.sort(key=lambda q: (q.created_cycle, q.id))

        def h_row(h: Hypothesis) -> dict:
            last = h.evidence_log[-1] if h.evidence_log else None
            return {
                "id": h.id, "statement": h.statement, "probability": h.probability, "status": h.status,
                "uncertainty": round(h.uncertainty, 3), "updated_cycle": h.updated_cycle,
                "last_update": (f"cycle {last['cycle']}: {last['direction']} ({last['strength']}"
                                f"{', critic: ' + last['verdict'] if last.get('verdict') not in (None, 'supported', 'unreviewed') else ''})"
                                + (f" -- {last['note']}" if last.get("note") else "")) if last else "no evidence yet",
                "trend": h.trend(),
            }

        return {
            "cycle": st.cycle,
            "open_hypotheses": [h_row(h) for h in open_h[:max_hypotheses]],
            "open_hypotheses_total": len(open_h),
            "settled_hypotheses": [h_row(h) for h in settled[:max_hypotheses]],
            "retired_hypotheses": len(retired),
            "open_questions": [
                {"id": q.id, "question": q.question, "source": q.source, "created_cycle": q.created_cycle}
                for q in open_q[:max_questions]
            ],
            "open_questions_total": len(open_q),
            "lessons": self.list_lessons(),
            "agenda": list(st.agenda),
            "recent_insights": list(st.recent_insights[-5:]),
            "last_cycle_status": st.last_cycle_status,
        }

    # commit ----------------------------------------------------------------
    def commit(
        self,
        staged: StagedChanges,
        verdicts: dict[str, dict],
        *,
        fingerprint: dict | None,
        event_ids: list[str],
        status: str,
        cost_usd: float,
        critic_follow_ups: list[str] | None = None,
        evidence_excerpts: dict[str, list[dict]] | None = None,
    ) -> CommitResult:
        """Apply the graded staged changes to memory and persist. Deterministic."""
        st = self.state
        cycle = staged.cycle
        result = CommitResult()
        excerpts = evidence_excerpts or {}

        def verdict_of(claim_id: str) -> tuple[str, str, float]:
            v = verdicts.get(claim_id) or {}
            verdict = v.get("verdict", "unreviewed")
            if verdict not in VERDICT_WEIGHTS:
                verdict = "unreviewed"
            return verdict, v.get("reason", ""), VERDICT_WEIGHTS[verdict]

        # hypotheses: proposals then updates, in staging order
        for nh in staged.new_hypotheses.values():
            h = Hypothesis(id=nh["id"], statement=nh["statement"], created_cycle=cycle, updated_cycle=cycle)
            h.history.append({"cycle": cycle, "probability": hyp.PRIOR})
            st.hypotheses[h.id] = h
            self._apply_update(h, nh, cycle, *verdict_of(nh["claim_id"]), excerpts.get(nh["claim_id"], []), result, created=True)
        for up in staged.hypothesis_updates:
            h = st.hypotheses.get(up["hypothesis_id"])
            if h is None:
                continue
            self._apply_update(h, up, cycle, *verdict_of(up["claim_id"]), excerpts.get(up["claim_id"], []), result)
        for hid, reason in staged.retirements.items():
            h = st.hypotheses.get(hid)
            if h is None:
                continue
            h.status = "retired"
            h.retired_reason = reason
            h.updated_cycle = cycle
            result.hypothesis_changes.append({"id": hid, "change": "retired", "reason": reason})
        for h in st.hypotheses.values():
            if h.updated_cycle == cycle and (not h.history or h.history[-1]["cycle"] != cycle
                                             or h.history[-1]["probability"] != h.probability):
                h.history.append({"cycle": cycle, "probability": h.probability})

        # questions
        for nq in staged.new_questions.values():
            q = Question(id=nq["id"], question=nq["question"], context=nq.get("context", ""),
                         source=nq.get("source", "mind"), created_cycle=cycle)
            st.questions[q.id] = q
            result.questions_added.append({"id": q.id, "question": q.question, "source": q.source})
        for r in staged.resolutions:
            q = st.questions.get(r["question_id"])
            if q is None:
                continue
            verdict, reason, weight = verdict_of(r["claim_id"])
            if weight > 0:
                q.status = "resolved"
                q.resolved_cycle = cycle
                q.answer = r["answer"]
                q.evidence_ids = list(r["evidence_ids"])
                q.tentative = weight < 1.0
                result.questions_resolved.append({"id": q.id, "question": q.question, "answer": q.answer,
                                                  "tentative": q.tentative, "verdict": verdict})
            else:
                q.notes.append(f"cycle {cycle}: proposed answer rejected by critic ({reason}): {r['answer']}")
                result.questions_unresolved.append({"id": q.id, "question": q.question, "answer": r["answer"],
                                                    "reason": reason})

        # critic follow-up questions join the agenda as open questions (deduped, capped)
        added = 0
        existing = {self._norm(q.question) for q in st.questions.values() if q.status == "open"}
        open_total = len(existing)
        for text in critic_follow_ups or []:
            key = self._norm(text)
            if not key or key in existing:
                continue
            if added >= MAX_CRITIC_QUESTIONS_PER_CYCLE or open_total >= MAX_OPEN_QUESTIONS:
                break
            staged.counters["question"] = staged.counters.get("question", 0) + 1
            qid = f"Q{staged.counters['question']}"
            q = Question(id=qid, question=text.strip(), source="critic", created_cycle=cycle)
            st.questions[qid] = q
            existing.add(key)
            added += 1
            open_total += 1
            result.questions_added.append({"id": qid, "question": q.question, "source": "critic"})

        # insights
        for ins in staged.insights:
            verdict, reason, weight = verdict_of(ins["claim_id"])
            record = {**ins, "cycle": cycle, "verdict": verdict, "critic_reason": reason}
            if verdict == "unsupported":
                result.insights_rejected.append(record)
            elif verdict == "weak":
                record["tentative"] = True
                result.insights_tentative.append(record)
            else:
                result.insights_accepted.append(record)
        kept = result.insights_accepted + result.insights_tentative
        st.recent_insights = (st.recent_insights + [
            {"cycle": cycle, "title": i["title"], "severity": i["severity"], "tentative": bool(i.get("tentative"))}
            for i in kept
        ])[-MAX_RECENT_INSIGHTS:]

        # lessons
        for name in staged.lessons_deleted:
            if self.delete_lesson(name):
                result.lessons_deleted.append(name)
        for name, (summary, body) in staged.lessons_written.items():
            self.write_lesson(name, summary, body)
            result.lessons_written.append(name)

        # agenda: the model's next focus first, then unanswered critic questions
        agenda = list((staged.conclusion or {}).get("next_focus") or [])
        for q in result.questions_added:
            if q["source"] == "critic":
                agenda.append(f"Answer {q['id']}: {q['question']}")
        st.agenda = self._dedupe(agenda)[:MAX_AGENDA]
        result.agenda = list(st.agenda)

        # perception bookkeeping
        if fingerprint is not None:
            st.fingerprint = fingerprint
        if event_ids:
            st.seen_event_ids = sorted(set(st.seen_event_ids) | set(event_ids))
        st.counters = dict(staged.counters)
        st.cycle = cycle
        st.last_cycle_status = status
        st.total_cost_usd = round(st.total_cost_usd + cost_usd, 6)
        self.save()
        return result

    def record_cycle_without_thinking(self, cycle: int, status: str, cost_usd: float = 0.0) -> None:
        """Bookkeeping for offline/error/refused cycles: counter and status only.

        The fingerprint and seen-event set are deliberately left untouched so the
        next thinking cycle still sees everything as new.
        """
        self.state.cycle = cycle
        self.state.last_cycle_status = status
        self.state.total_cost_usd = round(self.state.total_cost_usd + cost_usd, 6)
        self.save()

    # helpers ---------------------------------------------------------------
    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen, out = set(), []
        for item in items:
            key = MemoryStore._norm(item)
            if key and key not in seen:
                seen.add(key)
                out.append(item.strip())
        return out

    @staticmethod
    def _apply_update(h: Hypothesis, record: dict, cycle: int, verdict: str, reason: str, weight: float,
                      excerpts: list[dict], result: CommitResult, created: bool = False) -> None:
        before = h.probability
        if h.status == "retired":
            weight = 0.0
            reason = reason or "hypothesis is retired"
        after = hyp.apply_update(before, record["direction"], record["strength"], weight) if weight > 0 else before
        h.probability = after
        h.status = h.status if h.status == "retired" else hyp.status_for(after)
        h.updated_cycle = cycle
        h.evidence_log.append({
            "cycle": cycle, "claim_id": record["claim_id"], "evidence_ids": list(record["evidence_ids"]),
            "direction": record["direction"], "strength": record["strength"], "weight": weight,
            "verdict": verdict, "critic_reason": reason, "note": record.get("note") or "",
            "probability_before": before, "probability_after": after,
            "excerpts": [{"id": e["id"], "tool": e["tool"], "excerpt": e["excerpt"][:200]} for e in excerpts],
        })
        change = "proposed" if created else "updated"
        if weight == 0.0:
            change = "proposed (initial evidence reverted)" if created else "update reverted"
        elif weight < 1.0:
            change += " (counted half)"
        result.hypothesis_changes.append({
            "id": h.id, "statement": h.statement, "change": change, "verdict": verdict, "reason": reason,
            "before": before, "after": after, "status": h.status, "direction": record["direction"],
            "strength": record["strength"], "evidence_ids": list(record["evidence_ids"]),
        })
