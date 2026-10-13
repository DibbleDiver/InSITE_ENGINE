"""Journal (machine-readable, everything) and report (for people) writers."""
from __future__ import annotations

import os
from .memory import MemoryStore, atomic_write_json, atomic_write_text
from .serialize import to_jsonable
from .tools.data import fmt_ts


def cycle_tag(cycle: int) -> str:
    return f"cycle_{cycle:04d}"


def write_journal(store: MemoryStore, cycle: int, data: dict) -> str:
    path = os.path.join(store.journal_dir, f"{cycle_tag(cycle)}.json")
    atomic_write_json(path, to_jsonable(data))
    return path


def write_report(store: MemoryStore, cycle: int, text: str) -> tuple[str, str]:
    path = os.path.join(store.reports_dir, f"{cycle_tag(cycle)}.md")
    latest = os.path.join(store.reports_dir, "latest.md")
    atomic_write_text(path, text)
    atomic_write_text(latest, text)
    return path, latest


def _evidence_refs(ids: list[str], evidence: dict[str, dict]) -> str:
    parts = []
    for eid in ids or []:
        ev = evidence.get(eid)
        parts.append(f"{eid} ({ev['tool']})" if ev else eid)
    return ", ".join(parts) if parts else "none"


def _insight_block(ins: dict, evidence: dict[str, dict]) -> list[str]:
    lines = [f"### [{ins['severity'].upper()}] {ins['title']}"]
    lines.append(ins["finding"])
    if ins.get("recommended_action"):
        lines.append(f"Recommended action: {ins['recommended_action']}")
    extra = f"; hypothesis {ins['hypothesis_id']}" if ins.get("hypothesis_id") else ""
    lines.append(f"Evidence: {_evidence_refs(ins.get('evidence_ids'), evidence)}{extra}")
    if ins.get("verdict") == "weak" and ins.get("critic_reason"):
        lines.append(f"Critic: weak - {ins['critic_reason']}")
    elif ins.get("verdict") == "unreviewed":
        lines.append("Critic: not reviewed this cycle")
    lines.append("")
    return lines


def render_report(ctx: dict) -> str:
    """ctx keys: cycle, status, generated_at, goal, summary, insights (accepted), tentative, rejected,
    hypotheses (board rows), questions, lessons, diff, significant_events, evidence, critic, usage,
    cost_usd, total_cost_usd, model, steps, served_by_fallback, next_focus, sent_messages, stop_note,
    hypothesis_changes."""
    evidence = ctx.get("evidence") or {}
    lines = [f"# InSITE Mind report - cycle {ctx['cycle']} ({ctx['status']})", ""]
    lines.append(f"Generated {ctx.get('generated_at', '')}; model {ctx.get('model', '')}; "
                 f"{ctx.get('steps', 0)} model turns; cost ${ctx.get('cost_usd', 0.0):.2f}"
                 + (f"; pipeline output of {ctx['pipeline_generated_at']}" if ctx.get("pipeline_generated_at") else ""))
    if ctx.get("goal"):
        lines.append(f"Question: {ctx['goal']}")
    if ctx.get("stop_note"):
        lines.append(f"Note: {ctx['stop_note']}")
    if ctx.get("served_by_fallback"):
        lines.append("Note: at least one model turn was served by a fallback model after a safety-classifier decline.")
    lines += ["", "## Headline", "", ctx.get("summary") or "(no summary)", ""]

    if ctx.get("sent_messages"):
        lines += ["## Messages sent during the cycle", ""]
        for m in ctx["sent_messages"]:
            lines.append(f"- {m}")
        lines.append("")

    lines += ["## Insights", ""]
    if ctx.get("insights"):
        for ins in ctx["insights"]:
            lines += _insight_block(ins, evidence)
    else:
        lines += ["No insights were recorded this cycle.", ""]
    if ctx.get("tentative"):
        lines += ["## Tentative insights (the critic graded the evidence as weak)", ""]
        for ins in ctx["tentative"]:
            lines += _insight_block(ins, evidence)
    if ctx.get("rejected"):
        lines += ["## Rejected by the critic", ""]
        for ins in ctx["rejected"]:
            lines.append(f"- [{ins['severity']}] {ins['title']} - {ins.get('critic_reason') or 'unsupported by the cited evidence'}")
            lines.append(f"  Claimed: {ins['finding']}")
        lines.append("")

    lines += ["## Hypothesis board", ""]
    board = ctx.get("hypotheses") or []
    if board:
        lines.append("| id | hypothesis | p | status | confidence trend | updated |")
        lines.append("|---|---|---|---|---|---|")
        for h in board:
            lines.append(f"| {h['id']} | {h['statement']} | {h['probability']:.2f} | {h['status']} | {h['trend']} | cycle {h['updated_cycle']} |")
        lines.append("")
    else:
        lines += ["No hypotheses on the board.", ""]
    changes = ctx.get("hypothesis_changes") or []
    if changes:
        lines += ["Changes this cycle:", ""]
        for c in changes:
            if c.get("change") == "retired":
                lines.append(f"- {c['id']} retired: {c.get('reason', '')}")
                continue
            lines.append(f"- {c['id']} {c['change']}: {c['direction']}/{c['strength']} on "
                         f"{_evidence_refs(c.get('evidence_ids'), evidence)}; p {c['before']:.2f} -> {c['after']:.2f}"
                         + (f" (critic: {c['verdict']}{' - ' + c['reason'] if c.get('reason') else ''})"
                            if c.get("verdict") not in (None, "supported") else ""))
        lines.append("")

    questions = ctx.get("questions") or []
    lines += ["## Open questions", ""]
    if questions:
        for q in questions:
            lines.append(f"- {q['id']} ({q['source']}, cycle {q['created_cycle']}): {q['question']}")
    else:
        lines.append("None.")
    lines.append("")
    resolved = ctx.get("questions_resolved") or []
    if resolved:
        lines += ["Resolved this cycle:", ""]
        for q in resolved:
            lines.append(f"- {q['id']}: {q['question']} -> {q['answer']}{' (tentative)' if q.get('tentative') else ''}")
        lines.append("")

    if ctx.get("next_focus"):
        lines += ["## Next focus", ""]
        for item in ctx["next_focus"]:
            lines.append(f"- {item}")
        lines.append("")

    lines += ["## What changed since the last cycle", ""]
    diff = ctx.get("diff") or {}
    if diff.get("first_cycle"):
        lines.append("First cycle on this memory: all data was new.")
    elif not diff.get("has_new_data"):
        lines.append("No new data since the last cycle.")
    for ds_id, u in (diff.get("updated_datasets") or {}).items():
        lines.append(f"- {ds_id}: {u['rows_added']:+d} rows, last timestamp {fmt_ts(u['last_timestamp_before'])} -> {fmt_ts(u['last_timestamp_after'])}")
    if diff.get("stale_datasets"):
        lines.append(f"- no new data in: {', '.join(diff['stale_datasets'])}")
    if diff.get("new_datasets") and not diff.get("first_cycle"):
        lines.append(f"- new datasets: {', '.join(diff['new_datasets'])}")
    if diff.get("removed_datasets"):
        lines.append(f"- datasets no longer present: {', '.join(diff['removed_datasets'])}")
    if diff.get("new_events"):
        lines.append(f"- new events: {', '.join(diff['new_events'][:20])}{' ...' if len(diff['new_events']) > 20 else ''}")
    for t in diff.get("state_transitions") or []:
        lines.append(f"- state change {t['signal']}: {t['from'] or 'unknown'} -> {t['to']}")
    lines.append("")
    sig = ctx.get("significant_events") or []
    if sig:
        lines += ["Most significant events in the data:", ""]
        for e in sig[:10]:
            lines.append(f"- {e['event_id']}{' (new)' if e.get('new') else ''} {e['severity']} {e['kind']} on "
                         f"{e['dataset']}:{e['signal']} {e['start']}..{e['end']}: {e['detail']}")
        lines.append("")

    lessons = ctx.get("lessons") or []
    if lessons:
        lines += ["## Lessons on file", ""]
        for lesson in lessons:
            lines.append(f"- {lesson['name']}: {lesson['summary']}")
        lines.append("")

    critic = ctx.get("critic") or {}
    lines += ["## Review and cost", ""]
    if critic:
        counts = critic.get("counts") or {}
        shown = ", ".join(f"{n} {k}" for k, n in counts.items() if n) or "no claims"
        lines.append(f"Critic ({critic.get('model') or ''}): {critic.get('status')} - {shown}"
                     + (f"; {critic['error']}" if critic.get("error") else ""))
    usage = ctx.get("usage") or {}
    lines.append(f"Tokens: input {usage.get('input_tokens', 0)}, output {usage.get('output_tokens', 0)}, "
                 f"cache read {usage.get('cache_read_input_tokens', 0)}, cache write {usage.get('cache_creation_input_tokens', 0)}; "
                 f"{usage.get('calls', 0)} model calls")
    lines.append(f"Cost this cycle ${ctx.get('cost_usd', 0.0):.2f}; total on this memory ${ctx.get('total_cost_usd', 0.0):.2f}")
    if ctx.get("evidence"):
        lines.append(f"Evidence collected: {len(ctx['evidence'])} tool results (full excerpts in the journal)")
    lines.append("")
    return "\n".join(lines)


def board_rows(store: MemoryStore) -> list[dict]:
    rows = []
    for h in store.state.hypotheses.values():
        rows.append({"id": h.id, "statement": h.statement, "probability": h.probability, "status": h.status,
                     "trend": h.trend(), "updated_cycle": h.updated_cycle, "created_cycle": h.created_cycle})
    order = {"supported": 0, "open": 1, "refuted": 2, "retired": 3}
    rows.sort(key=lambda r: (order.get(r["status"], 9), -r["probability"], r["id"]))
    return rows


def open_question_rows(store: MemoryStore) -> list[dict]:
    rows = [{"id": q.id, "question": q.question, "source": q.source, "created_cycle": q.created_cycle}
            for q in store.state.questions.values() if q.status == "open"]
    rows.sort(key=lambda r: (r["created_cycle"], r["id"]))
    return rows
