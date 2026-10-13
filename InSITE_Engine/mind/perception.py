"""Perceive: fingerprint a PipelineResult, diff it against memory, and write
the situation brief that opens every cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..contracts import PipelineResult
from .tools.data import fmt_num, fmt_ts

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
MAX_BRIEF_EVENTS = 15
LAG_WARNING_HOURS = 24.0


def _iso(ts: Any) -> str | None:
    if ts is None:
        return None
    try:
        if pd.isna(ts):
            return None
    except (TypeError, ValueError):
        pass
    return pd.Timestamp(ts).isoformat()


def fingerprint(result: PipelineResult) -> dict:
    datasets = {}
    for ds_id, ds in sorted(result.datasets.items()):
        start, end = ds.time_range()
        datasets[ds_id] = {
            "rows": int(len(ds.frame)),
            "first_timestamp": _iso(start),
            "last_timestamp": _iso(end),
            "signals": list(ds.signals),
        }
    events = result.events
    event_ids = sorted(str(e) for e in events["event_id"].tolist()) if events is not None and not events.empty else []
    current = {}
    states = result.states
    if states is not None and not states.empty:
        latest = states.sort_values("end").groupby(["dataset", "signal"]).tail(1)
        for row in latest.itertuples(index=False):
            current[f"{row.dataset}:{row.signal}"] = str(row.state)
    return {
        "generated_at": _iso(result.generated_at),
        "datasets": datasets,
        "event_ids": event_ids,
        "current_states": current,
    }


def diff_fingerprints(previous: dict | None, current: dict, seen_event_ids: list[str] | None = None) -> dict:
    seen = set(seen_event_ids or [])
    cur_ds = current.get("datasets", {})
    if previous is None:
        return {
            "first_cycle": True,
            "new_datasets": sorted(cur_ds),
            "removed_datasets": [],
            "updated_datasets": {},
            "stale_datasets": [],
            "new_events": [e for e in current.get("event_ids", []) if e not in seen],
            "removed_events": [],
            "state_transitions": [],
            "has_new_data": True,
        }
    prev_ds = previous.get("datasets", {})
    new_datasets = sorted(set(cur_ds) - set(prev_ds))
    removed_datasets = sorted(set(prev_ds) - set(cur_ds))
    updated, stale = {}, []
    for ds_id in sorted(set(cur_ds) & set(prev_ds)):
        before, after = prev_ds[ds_id], cur_ds[ds_id]
        changed = after["rows"] != before["rows"] or after["last_timestamp"] != before["last_timestamp"]
        if changed:
            updated[ds_id] = {
                "rows_before": before["rows"],
                "rows_after": after["rows"],
                "rows_added": after["rows"] - before["rows"],
                "last_timestamp_before": before["last_timestamp"],
                "last_timestamp_after": after["last_timestamp"],
            }
        else:
            stale.append(ds_id)
    prev_events = set(previous.get("event_ids", []))
    cur_events = current.get("event_ids", [])
    new_events = [e for e in cur_events if e not in prev_events and e not in seen]
    removed_events = sorted(prev_events - set(cur_events))
    transitions = []
    prev_states = previous.get("current_states", {})
    for key, state in sorted(current.get("current_states", {}).items()):
        before = prev_states.get(key)
        if before != state:
            transitions.append({"signal": key, "from": before, "to": state})
    has_new_data = bool(new_datasets or updated or new_events or transitions)
    return {
        "first_cycle": False,
        "new_datasets": new_datasets,
        "removed_datasets": removed_datasets,
        "updated_datasets": updated,
        "stale_datasets": stale,
        "new_events": new_events,
        "removed_events": removed_events,
        "state_transitions": transitions,
        "has_new_data": has_new_data,
    }


def significant_events(result: PipelineResult, new_ids: set[str], limit: int = MAX_BRIEF_EVENTS) -> list[dict]:
    events = result.events
    if events is None or events.empty:
        return []
    df = events.assign(
        _sev=events["severity"].map(SEVERITY_RANK).fillna(3),
        _new=~events["event_id"].isin(new_ids),
        _score=pd.to_numeric(events["score"], errors="coerce").fillna(0.0),
    )
    df = df.sort_values(["_new", "_sev", "_score"], ascending=[True, True, False])
    out = []
    for r in df.head(limit).itertuples(index=False):
        out.append({
            "event_id": str(r.event_id), "new": r.event_id in new_ids, "dataset": str(r.dataset), "signal": str(r.signal),
            "kind": str(r.kind), "severity": str(r.severity), "start": fmt_ts(r.start), "end": fmt_ts(r.end),
            "score": fmt_num(r.score), "detail": str(r.detail)[:120],
        })
    return out


def feed_lag_hours(fp: dict) -> dict[str, float]:
    """Hours each dataset's last timestamp trails the newest feed."""
    lasts = {}
    for ds_id, info in fp.get("datasets", {}).items():
        if info.get("last_timestamp"):
            lasts[ds_id] = pd.Timestamp(info["last_timestamp"])
    if not lasts:
        return {}
    newest = max(lasts.values())
    return {ds_id: round((newest - ts).total_seconds() / 3600.0, 1) for ds_id, ts in lasts.items()}


@dataclass
class Perception:
    result: PipelineResult
    summary: dict
    fingerprint: dict
    diff: dict
    new_event_ids: set[str]
    significant: list[dict] = field(default_factory=list)
    brief: str = ""


def perceive(result: PipelineResult, previous_fingerprint: dict | None, seen_event_ids: list[str],
             digest: dict, cycle: int, goal: str | None) -> Perception:
    fp = fingerprint(result)
    diff = diff_fingerprints(previous_fingerprint, fp, seen_event_ids)
    new_ids = set(diff["new_events"])
    summary = result.summary()
    sig = significant_events(result, new_ids)
    perception = Perception(result=result, summary=summary, fingerprint=fp, diff=diff, new_event_ids=new_ids, significant=sig)
    perception.brief = build_brief(perception, digest, cycle, goal)
    return perception


def build_brief(p: Perception, digest: dict, cycle: int, goal: str | None) -> str:
    summary, diff, fp = p.summary, p.diff, p.fingerprint
    lines = [f"# Situation brief for cycle {cycle} (pipeline output generated {fmt_ts(fp.get('generated_at'))})"]
    if goal:
        lines.append(f"\nGoal for this cycle (a question from the operations team; answer it in the conclude_cycle "
                     f"summary): {goal}")
    else:
        lines.append("\nGoal for this cycle: autonomous. Find what matters most for the operations team right now - "
                     "developing problems, root causes that join evidence across systems, and slow drifts no threshold "
                     "catches - and advance the open hypotheses and agenda below.")

    lines.append("\n## Data")
    lags = feed_lag_hours(fp)
    for ds_id, info in summary["datasets"].items():
        lag = lags.get(ds_id)
        lines.append(
            f"- {ds_id} [{info['source']}] rows={info['rows']}"
            + (f" {fmt_ts(info['start'])}..{fmt_ts(info['end'])}" if info.get("start") else " (no time axis)")
            + f"; signals: {', '.join(info['signals']) or '(none)'}"
            + (f"; text: {', '.join(info['text_columns'][:6])}" if info.get("text_columns") else "")
            + (f"; LAGGING by {lag} h behind the newest feed" if lag is not None and lag >= LAG_WARNING_HOURS else "")
        )
    kinds = ", ".join(f"{k}={v}" for k, v in summary["events_by_kind"].items()) or "none"
    sev = ", ".join(f"{k}={v}" for k, v in summary["events_by_severity"].items()) or "none"
    lines.append(f"Events: {summary['event_count']} ({kinds}); severity: {sev}")
    abnormal = {k: v for k, v in summary["current_states"].items() if v != "normal"}
    if abnormal:
        lines.append("Current non-normal states: " + ", ".join(f"{k}={v}" for k, v in sorted(abnormal.items())))
    else:
        lines.append("Current states: all signals normal" if summary["current_states"] else "Current states: none reported")

    lines.append("\n## What changed since the last cycle")
    if diff["first_cycle"]:
        lines.append("This is the first cycle on this memory: everything is new.")
    else:
        if not diff["has_new_data"]:
            lines.append("No new data since the last cycle (same rows, events and states). Use the time to test open "
                         "hypotheses and the agenda, or conclude quickly if nothing is worth doing.")
        for ds_id, u in diff["updated_datasets"].items():
            lines.append(f"- {ds_id}: {u['rows_added']:+d} rows, last timestamp "
                         f"{fmt_ts(u['last_timestamp_before'])} -> {fmt_ts(u['last_timestamp_after'])}")
        if diff["stale_datasets"]:
            lines.append(f"- no new data in: {', '.join(diff['stale_datasets'])}")
        if diff["new_datasets"]:
            lines.append(f"- new datasets: {', '.join(diff['new_datasets'])}")
        if diff["removed_datasets"]:
            lines.append(f"- datasets no longer present: {', '.join(diff['removed_datasets'])}")
        if diff["new_events"]:
            lines.append(f"- {len(diff['new_events'])} new events (marked NEW below)")
        if diff["removed_events"]:
            lines.append(f"- {len(diff['removed_events'])} events from the previous run are gone (data was re-processed)")
        for t in diff["state_transitions"]:
            lines.append(f"- state change {t['signal']}: {t['from'] or 'unknown'} -> {t['to']}")

    if p.significant:
        lines.append("\n## Most significant events (new first, then by severity and score)")
        for e in p.significant:
            lines.append(f"- {e['event_id']}{' NEW' if e['new'] else ''} {e['severity']} {e['kind']} on "
                         f"{e['dataset']}:{e['signal']} {e['start']}..{e['end']} score={e['score']}: {e['detail']}")
        total = summary["event_count"]
        if total > len(p.significant):
            lines.append(f"({total - len(p.significant)} more via get_events)")

    lines.append("\n## Memory")
    open_h = digest.get("open_hypotheses") or []
    if open_h:
        lines.append("Open hypotheses, ranked by how much a decisive test would teach (uncertainty p(1-p)):")
        for h in open_h:
            lines.append(f"- {h['id']} p={h['probability']:.2f} (trend {h['trend']}): {h['statement']} "
                         f"[last: {h['last_update']}]")
        if digest.get("open_hypotheses_total", 0) > len(open_h):
            lines.append(f"  ({digest['open_hypotheses_total'] - len(open_h)} more open hypotheses not shown)")
    else:
        lines.append("Open hypotheses: none yet.")
    settled = digest.get("settled_hypotheses") or []
    if settled:
        lines.append("Settled hypotheses: " + "; ".join(f"{h['id']} {h['status']} p={h['probability']:.2f}: {h['statement']}"
                                                        for h in settled))
    open_q = digest.get("open_questions") or []
    if open_q:
        lines.append("Open questions:")
        for q in open_q:
            lines.append(f"- {q['id']} ({q['source']}, cycle {q['created_cycle']}): {q['question']}")
    lessons = digest.get("lessons") or []
    if lessons:
        lines.append("Lessons (read_lesson for the full text):")
        for lesson in lessons[:20]:
            lines.append(f"- {lesson['name']}: {lesson['summary']}")
    agenda = digest.get("agenda") or []
    if agenda:
        lines.append("Agenda carried over from the last cycle:")
        for item in agenda:
            lines.append(f"- {item}")
    recent = digest.get("recent_insights") or []
    if recent:
        lines.append("Recent insights already reported: " + "; ".join(
            f"cycle {i['cycle']} [{i['severity']}{', tentative' if i.get('tentative') else ''}] {i['title']}" for i in recent))
    if digest.get("last_cycle_status") and digest["last_cycle_status"] not in ("completed", None):
        lines.append(f"Note: the previous cycle ended with status '{digest['last_cycle_status']}'.")
    return "\n".join(lines)
