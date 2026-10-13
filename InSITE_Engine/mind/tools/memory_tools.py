"""Memory tools: every call stages a change; nothing reaches disk until the
critic has graded the cycle and MemoryStore.commit runs."""
from __future__ import annotations

from ..evidence import Evidence, EvidenceLedger
from ..memory import MemoryStore, StagedChanges
from .data import ToolError


class MemoryTools:
    def __init__(self, staged: StagedChanges, store: MemoryStore, ledger: EvidenceLedger):
        self.staged = staged
        self.store = store
        self.ledger = ledger

    def _require_evidence(self, ids: list[str]) -> list[Evidence]:
        known, unknown = self.ledger.resolve(ids)
        if unknown:
            raise ToolError(
                f"unknown evidence id(s) {unknown}: cite only ids printed at the top of data-tool results in this cycle"
                + (f" (known: {', '.join(self.ledger.items)})" if self.ledger.items else " (no data tool has been called yet)")
            )
        if not known:
            raise ToolError("at least one evidence id from this cycle is required")
        return known

    def propose_hypothesis(self, args: dict) -> str:
        evidence = self._require_evidence(args["evidence_ids"])
        statement = args["statement"].strip()
        if not statement:
            raise ToolError("statement must not be empty")
        dup = self._duplicate_statement(statement)
        if dup:
            raise ToolError(f"a hypothesis with this statement already exists: {dup}; use update_hypothesis")
        record = self.staged.stage_hypothesis(statement, args["direction"], args["strength"],
                                              [e.id for e in evidence], args.get("note") or "")
        p = self.staged.provisional_probability(record["id"])
        return (f"staged hypothesis {record['id']} (claim {record['claim_id']}): \"{statement}\"\n"
                f"provisional probability {p:.2f} after {args['direction']}/{args['strength']} evidence "
                f"{', '.join(e.id for e in evidence)}; committed after the critic review")

    def update_hypothesis(self, args: dict) -> str:
        hid = args["hypothesis_id"].strip().upper()
        if not self.staged.hypothesis_exists(hid):
            raise ToolError(f"unknown hypothesis {hid!r}; known ids: {', '.join(self._known_hypotheses()) or '(none)'}")
        existing = self.store.state.hypotheses.get(hid)
        if existing is not None and existing.status == "retired":
            raise ToolError(f"{hid} is retired ({existing.retired_reason}); propose a new hypothesis instead")
        if hid in self.staged.retirements:
            raise ToolError(f"{hid} was retired earlier this cycle")
        evidence = self._require_evidence(args["evidence_ids"])
        before = self.staged.provisional_probability(hid)
        record = self.staged.stage_update(hid, args["direction"], args["strength"], [e.id for e in evidence], args.get("note") or "")
        after = self.staged.provisional_probability(hid)
        return (f"staged update {record['claim_id']} for {hid}: {args['direction']}/{args['strength']} "
                f"({', '.join(e.id for e in evidence)}); provisional probability {before:.2f} -> {after:.2f}"
                f"; committed after the critic review")

    def retire_hypothesis(self, args: dict) -> str:
        hid = args["hypothesis_id"].strip().upper()
        if not self.staged.hypothesis_exists(hid):
            raise ToolError(f"unknown hypothesis {hid!r}; known ids: {', '.join(self._known_hypotheses()) or '(none)'}")
        self.staged.stage_retirement(hid, args["reason"])
        return f"{hid} will be retired: {args['reason']}"

    def add_question(self, args: dict) -> str:
        question = args["question"].strip()
        if not question:
            raise ToolError("question must not be empty")
        record = self.staged.stage_question(question, args.get("context") or "")
        return f"recorded open question {record['id']}: {question}"

    def resolve_question(self, args: dict) -> str:
        qid = args["question_id"].strip().upper()
        if not self.staged.question_exists(qid):
            raise ToolError(f"unknown question {qid!r}; open questions: {', '.join(self._open_questions()) or '(none)'}")
        if not self.staged.is_question_open(qid):
            raise ToolError(f"{qid} is already resolved")
        evidence = self._require_evidence(args["evidence_ids"])
        record = self.staged.stage_resolution(qid, args["answer"], [e.id for e in evidence])
        return f"staged resolution {record['claim_id']} of {qid}; committed after the critic review"

    def record_insight(self, args: dict) -> str:
        evidence = self._require_evidence(args["evidence_ids"])
        hid = (args.get("hypothesis_id") or "").strip().upper() or None
        if hid and not self.staged.hypothesis_exists(hid):
            raise ToolError(f"unknown hypothesis {hid!r} in hypothesis_id")
        record = self.staged.stage_insight(args["title"].strip(), args["finding"].strip(), args["severity"],
                                           [e.id for e in evidence], args.get("recommended_action"), hid)
        return (f"staged insight {record['claim_id']} [{args['severity']}] \"{record['title']}\" citing "
                f"{', '.join(e.id for e in evidence)}; the critic reviews it before it is reported")

    def save_lesson(self, args: dict) -> str:
        name = args["name"].strip()
        if not MemoryStore.valid_lesson_name(name):
            raise ToolError("name must be a slug: lowercase letters, digits, '-' or '_' (max 61 chars)")
        summary = " ".join(args["summary"].split())
        if not summary:
            raise ToolError("summary must not be empty")
        self.staged.stage_lesson(name, summary, args["body"])
        existing = self.store.read_lesson(name) is not None
        return f"lesson '{name}' staged ({'replaces the existing lesson' if existing else 'new'}); written after the critic review"

    def read_lesson(self, args: dict) -> str:
        name = args["name"].strip()
        if not MemoryStore.valid_lesson_name(name):
            raise ToolError("name must be a slug: lowercase letters, digits, '-' or '_'")
        if name in self.staged.lessons_written:
            summary, body = self.staged.lessons_written[name]
            return f"# {summary}\n\n{body}\n(staged this cycle, not yet written)"
        if name in self.staged.lessons_deleted:
            raise ToolError(f"lesson '{name}' was deleted earlier this cycle")
        text = self.store.read_lesson(name)
        if text is None:
            names = [lesson["name"] for lesson in self.store.list_lessons()]
            raise ToolError(f"no lesson named '{name}'; available: {', '.join(names) or '(none)'}")
        return text

    def delete_lesson(self, args: dict) -> str:
        name = args["name"].strip()
        if not MemoryStore.valid_lesson_name(name):
            raise ToolError("name must be a slug: lowercase letters, digits, '-' or '_'")
        if name not in self.staged.lessons_written and self.store.read_lesson(name) is None:
            raise ToolError(f"no lesson named '{name}'")
        self.staged.stage_lesson_deletion(name)
        return f"lesson '{name}' will be deleted after the critic review" + (f": {args['reason']}" if args.get("reason") else "")

    # helpers ---------------------------------------------------------------
    def _known_hypotheses(self) -> list[str]:
        ids = [h.id for h in self.store.state.hypotheses.values() if h.status != "retired"]
        ids += list(self.staged.new_hypotheses)
        return ids

    def _open_questions(self) -> list[str]:
        ids = [q.id for q in self.store.state.questions.values() if q.status == "open"]
        ids += list(self.staged.new_questions)
        return ids

    def _duplicate_statement(self, statement: str) -> str | None:
        key = MemoryStore._norm(statement)
        for h in self.store.state.hypotheses.values():
            if h.status != "retired" and MemoryStore._norm(h.statement) == key:
                return h.id
        for nh in self.staged.new_hypotheses.values():
            if MemoryStore._norm(nh["statement"]) == key:
                return nh["id"]
        return None
