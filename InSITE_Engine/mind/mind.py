"""MŸND: the Mind of the InSITE Engine. Perceive -> investigate -> critique -> consolidate, once or in a loop."""
from __future__ import annotations

import datetime as _dt
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..contracts import PipelineResult
from .agent import InvestigationAgent
from .config import MindConfig
from .console import Console
from .critic import CriticResult, run_critic
from .evidence import EvidenceLedger
from .llm import AnthropicLLM, MindUnavailable
from .memory import CommitResult, MemoryStore, StagedChanges
from .perception import Perception, perceive
from .prompts import SYSTEM_PROMPT
from .reporting import board_rows, open_question_rows, render_report, write_journal, write_report
from .serialize import messages_to_dicts, to_jsonable
from .tools import ToolRegistry, build_tool_definitions, tool_names

log = logging.getLogger("insite.mind")

THINKING_STATUSES = ("completed", "wrapped_up_on_budget", "step_limit")
MAX_BACKOFF_SECONDS = 3600.0


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class CycleOutcome:
    cycle: int
    status: str
    summary: str
    report_path: str | None = None
    journal_path: str | None = None
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)
    insights: list[dict] = field(default_factory=list)
    rejected_insights: list[dict] = field(default_factory=list)
    hypothesis_board: list[dict] = field(default_factory=list)
    open_questions: list[dict] = field(default_factory=list)
    next_focus: list[str] = field(default_factory=list)
    critic: dict = field(default_factory=dict)
    perception: dict = field(default_factory=dict)
    sent_messages: list[str] = field(default_factory=list)
    steps: int = 0
    error: str | None = None
    served_by_fallback: bool = False
    goal: str | None = None

    def as_dict(self) -> dict:
        return to_jsonable(asdict(self))


class Mind:
    """Public API used by main.py: Mind(cfg).think() / .ask(q) / .live()."""

    def __init__(self, cfg: dict, llm: Any = None, pipeline_runner: Callable[[dict], PipelineResult] | None = None,
                 console: Console | None = None):
        self.cfg = cfg or {}
        self.config = MindConfig.from_cfg(self.cfg)
        self.console = console or Console()
        self.store = MemoryStore(self.config.memory_dir)
        self._pipeline_runner = pipeline_runner
        if llm is None:
            self.llm = AnthropicLLM(self.config)
        elif llm is False:
            self.llm = None
        else:
            self.llm = llm
        self.system_prompt = SYSTEM_PROMPT
        self.tools = build_tool_definitions(self.config.eager_input_streaming)
        self._sleep = time.sleep
        self.outcomes: list[CycleOutcome] = []

    # ------------------------------------------------------------ public API
    def think(self, goal: str | None = None, offline: bool = False) -> CycleOutcome:
        result = self._run_pipeline()
        cycle = self.store.state.cycle + 1
        perception = self._perceive(result, cycle, goal)
        if offline or self.llm is None:
            if not offline:
                log.warning("the Mind has no model client; running a perception-only cycle")
            return self._offline_cycle(perception, cycle, goal)
        return self._think_cycle(perception, cycle, goal, self.config.budget_usd_per_cycle)

    def ask(self, question: str) -> CycleOutcome:
        question = (question or "").strip()
        if not question:
            raise ValueError("ask() needs a question")
        if self.llm is None:
            raise MindUnavailable("ask() needs a model client; this Mind was created without one")
        return self.think(goal=question)

    def live(self, interval_seconds: float | None = None, max_cycles: int | None = None,
             budget_usd_total: float | None = None) -> list[CycleOutcome]:
        interval = float(self.config.loop_interval_seconds if interval_seconds is None else interval_seconds)
        budget_total = self.config.budget_usd_total if budget_usd_total is None else budget_usd_total
        outcomes: list[CycleOutcome] = []
        spent = 0.0
        idle = 0
        failures = 0
        self.console.status(
            f"live mode: every {interval:.0f}s" + (f", at most {max_cycles} cycles" if max_cycles else "")
            + (f", total budget ${budget_total:.2f}" if budget_total else "") + "; Ctrl-C stops"
        )
        try:
            while True:
                if max_cycles is not None and len(outcomes) >= max_cycles:
                    self.console.status(f"stopping: {max_cycles} cycles done")
                    break
                if budget_total is not None and spent >= budget_total:
                    self.console.status(f"stopping: total budget ${budget_total:.2f} reached (${spent:.2f} spent)")
                    break
                try:
                    result = self._run_pipeline()
                    cycle = self.store.state.cycle + 1
                    perception = self._perceive(result, cycle, None)
                except Exception as e:  # pipeline problems must not kill the loop
                    failures += 1
                    log.exception("pipeline/perception failed: %s", e)
                    self._sleep(self._backoff(interval, failures))
                    continue
                if not perception.diff["has_new_data"]:
                    if idle >= self.config.max_idle_cycles:
                        self.console.status(f"no new data and {idle} idle cycles already spent; waiting {interval:.0f}s for data")
                        self._sleep(interval)
                        continue
                    idle += 1
                else:
                    idle = 0
                cycle_budget = self.config.budget_usd_per_cycle
                if budget_total is not None:
                    cycle_budget = max(0.0, min(cycle_budget, budget_total - spent))
                try:
                    if self.llm is None:
                        outcome = self._offline_cycle(perception, cycle, None)
                    else:
                        outcome = self._think_cycle(perception, cycle, None, cycle_budget)
                except Exception as e:  # defensive: a bug in one cycle is logged, not fatal
                    failures += 1
                    log.exception("cycle %d crashed: %s", cycle, e)
                    self._sleep(self._backoff(interval, failures))
                    continue
                outcomes.append(outcome)
                self.outcomes.append(outcome)
                spent += outcome.cost_usd
                if outcome.status in ("error", "refused"):
                    failures += 1
                    delay = self._backoff(interval, failures)
                    self.console.status(f"cycle {cycle} ended with {outcome.status}; backing off {delay:.0f}s")
                    self._sleep(delay)
                    continue
                failures = 0
                if max_cycles is not None and len(outcomes) >= max_cycles:
                    self.console.status(f"stopping: {max_cycles} cycles done")
                    break
                self._sleep(interval)
        except KeyboardInterrupt:
            self.console.status("stopped by Ctrl-C")
        return outcomes

    # ------------------------------------------------------------- internals
    @staticmethod
    def _backoff(interval: float, failures: int) -> float:
        base = max(interval, 30.0)
        return float(min(base * (2 ** max(failures - 1, 0)), MAX_BACKOFF_SECONDS))

    def _run_pipeline(self) -> PipelineResult:
        runner = self._pipeline_runner
        if runner is None:
            try:
                from InSITE_Engine.pipeline import run_pipeline as runner  # noqa: N813
            except ImportError as e:
                raise RuntimeError("the pipeline is not available (InSITE_Engine.pipeline.run_pipeline could not be "
                                   f"imported: {e}); pass pipeline_runner= or install the pipeline package") from e
        result = runner(self.cfg)
        if not hasattr(result, "datasets") or not hasattr(result, "events") or not hasattr(result, "states"):
            raise TypeError("the pipeline runner must return a PipelineResult")
        return result

    def _perceive(self, result: PipelineResult, cycle: int, goal: str | None) -> Perception:
        state = self.store.state
        perception = perceive(result, state.fingerprint, state.seen_event_ids, self.store.digest(), cycle, goal)
        d = perception.diff
        self.console.status(
            f"cycle {cycle}: {len(result.datasets)} datasets, {perception.summary['event_count']} events "
            f"({len(d['new_events'])} new), {len(d['state_transitions'])} state changes"
            + ("; first cycle" if d["first_cycle"] else "" if d["has_new_data"] else "; no new data")
        )
        return perception

    def _think_cycle(self, perception: Perception, cycle: int, goal: str | None, budget_usd: float) -> CycleOutcome:
        started = _now()
        state = self.store.state
        staged = StagedChanges(state, cycle)
        ledger = EvidenceLedger()
        registry = ToolRegistry(perception.result, staged, self.store, ledger,
                                new_event_ids=perception.new_event_ids, on_send_to_user=self.console.user_message)
        agent = InvestigationAgent(self.llm, self.config, registry, self.console, self.system_prompt, self.tools,
                                   budget_usd=budget_usd)
        self.console.status(f"cycle {cycle}: investigating (effort {self.config.effort}, budget ${budget_usd:.2f}, "
                            f"max {self.config.max_steps} steps)")
        ar = agent.run(perception.brief)

        critic = CriticResult(status="skipped", model=self.config.critic_model)
        commit: CommitResult | None = None
        if ar.status in THINKING_STATUSES:
            critic = run_critic(self.llm, self.config, staged, ledger, self.console, goal)
            excerpts = {}
            for claim in staged.claims:
                info = staged.claim_text(claim["claim_id"])
                if info:
                    excerpts[claim["claim_id"]] = ledger.excerpts_for(info["evidence_ids"], chars=200)
            commit = self.store.commit(
                staged, critic.verdicts, fingerprint=perception.fingerprint,
                event_ids=perception.fingerprint["event_ids"], status=ar.status,
                cost_usd=ar.cost_usd + critic.cost_usd, critic_follow_ups=critic.follow_ups,
                evidence_excerpts=excerpts,
            )
        else:
            # refused / error: nothing staged is committed and the fingerprint stays, so the
            # next cycle sees the same new data again.
            self.store.record_cycle_without_thinking(cycle, ar.status, cost_usd=ar.cost_usd)

        total_cost = ar.cost_usd + critic.cost_usd
        usage = ar.usage.as_dict()
        if critic.usage:
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage[key] += critic.usage.get(key, 0)
            usage["calls"] += 1

        insights = (commit.insights_accepted + commit.insights_tentative) if commit else []
        rejected = commit.insights_rejected if commit else []
        board = board_rows(self.store)
        questions = open_question_rows(self.store)
        ctx = {
            "cycle": cycle, "status": ar.status, "generated_at": _now(), "goal": goal, "summary": ar.summary,
            "insights": commit.insights_accepted if commit else [], "tentative": commit.insights_tentative if commit else [],
            "rejected": rejected, "hypotheses": board, "hypothesis_changes": commit.hypothesis_changes if commit else [],
            "questions": questions, "questions_resolved": commit.questions_resolved if commit else [],
            "lessons": self.store.list_lessons(), "diff": perception.diff, "significant_events": perception.significant,
            "evidence": ledger.as_dict(), "critic": critic.as_dict(), "usage": usage, "cost_usd": total_cost,
            "total_cost_usd": self.store.state.total_cost_usd, "model": self.config.model, "steps": ar.steps,
            "served_by_fallback": ar.served_by_fallback, "next_focus": commit.agenda if commit else ar.next_focus,
            "sent_messages": registry.sent_messages, "stop_note": ar.stop_note if not ar.concluded else "",
            "pipeline_generated_at": perception.summary.get("generated_at"),
        }
        report_text = render_report(ctx)
        report_path, _ = write_report(self.store, cycle, report_text)
        journal = {
            "cycle": cycle, "status": ar.status, "goal": goal, "started_at": started, "finished_at": _now(),
            "model": self.config.model, "effort": self.config.effort, "critic_model": self.config.critic_model,
            "perception": {"summary": perception.summary, "fingerprint": perception.fingerprint, "diff": perception.diff,
                           "significant_events": perception.significant},
            "brief": perception.brief, "system_prompt": self.system_prompt, "tools": tool_names(),
            "transcript": messages_to_dicts(ar.messages), "turns": [t.as_dict() for t in ar.turns],
            "evidence": ledger.as_dict(), "send_to_user": registry.sent_messages, "staged": staged.summary_counts(),
            "critic": critic.as_dict(), "commit": asdict(commit) if commit else None,
            "refusal": ar.refusal, "error": ar.error, "stop_note": ar.stop_note, "served_by_fallback": ar.served_by_fallback,
            "usage": usage, "cost_usd": total_cost, "summary": ar.summary, "report_path": report_path,
        }
        journal_path = write_journal(self.store, cycle, journal)
        outcome = CycleOutcome(
            cycle=cycle, status=ar.status, summary=ar.summary, report_path=report_path, journal_path=journal_path,
            cost_usd=total_cost, usage=usage, insights=insights, rejected_insights=rejected, hypothesis_board=board,
            open_questions=questions, next_focus=ctx["next_focus"], critic=critic.as_dict(),
            perception={"diff": perception.diff, "event_count": perception.summary["event_count"],
                        "datasets": len(perception.result.datasets)},
            sent_messages=list(registry.sent_messages), steps=ar.steps, error=ar.error,
            served_by_fallback=ar.served_by_fallback, goal=goal,
        )
        self.console.status(f"cycle {cycle} {ar.status}: {ar.steps} steps, ${total_cost:.2f}, "
                            f"{len(insights)} insights ({len(rejected)} rejected); report {report_path}")
        return outcome

    def _offline_cycle(self, perception: Perception, cycle: int, goal: str | None) -> CycleOutcome:
        started = _now()
        summary = self._perception_summary(perception)
        self.store.record_cycle_without_thinking(cycle, "offline")
        ctx = {
            "cycle": cycle, "status": "offline", "generated_at": _now(), "goal": goal, "summary": summary,
            "hypotheses": board_rows(self.store), "questions": open_question_rows(self.store),
            "lessons": self.store.list_lessons(), "diff": perception.diff, "significant_events": perception.significant,
            "usage": {}, "cost_usd": 0.0, "total_cost_usd": self.store.state.total_cost_usd, "model": "none (offline)",
            "steps": 0, "stop_note": "perception-only cycle: no model was called, memory and the data fingerprint "
                                     "are unchanged, so the next thinking cycle will still see this data as new",
            "pipeline_generated_at": perception.summary.get("generated_at"),
        }
        report_path, _ = write_report(self.store, cycle, render_report(ctx))
        journal_path = write_journal(self.store, cycle, {
            "cycle": cycle, "status": "offline", "goal": goal, "started_at": started, "finished_at": _now(),
            "perception": {"summary": perception.summary, "fingerprint": perception.fingerprint, "diff": perception.diff,
                           "significant_events": perception.significant},
            "brief": perception.brief, "summary": summary, "report_path": report_path, "cost_usd": 0.0,
        })
        outcome = CycleOutcome(cycle=cycle, status="offline", summary=summary, report_path=report_path,
                               journal_path=journal_path, hypothesis_board=board_rows(self.store),
                               open_questions=open_question_rows(self.store),
                               perception={"diff": perception.diff, "event_count": perception.summary["event_count"],
                                           "datasets": len(perception.result.datasets)}, goal=goal)
        self.console.status(f"cycle {cycle} offline: perception-only report {report_path}")
        return outcome

    @staticmethod
    def _perception_summary(p: Perception) -> str:
        s, d = p.summary, p.diff
        parts = [f"Perception-only cycle (no model was called). The pipeline produced {len(s['datasets'])} datasets and "
                 f"{s['event_count']} events"]
        if s["events_by_severity"]:
            parts[-1] += " (" + ", ".join(f"{v} {k}" for k, v in s["events_by_severity"].items()) + ")"
        parts[-1] += "."
        abnormal = [k for k, v in s["current_states"].items() if v != "normal"]
        parts.append(f"{len(abnormal)} signal(s) are currently not normal: {', '.join(sorted(abnormal))}." if abnormal
                     else "All signals are currently in a normal state." if s["current_states"] else "No state information was produced.")
        if d["first_cycle"]:
            parts.append("This memory has not seen any data before, so everything is new.")
        else:
            parts.append(f"Since the Mind last thought: {len(d['new_events'])} new events, "
                         f"{len(d['updated_datasets'])} datasets with new rows, {len(d['state_transitions'])} state changes"
                         + (f", stale feeds: {', '.join(d['stale_datasets'])}" if d["stale_datasets"] else "") + ".")
        if p.significant:
            top = p.significant[:3]
            parts.append("Most significant events: " + "; ".join(
                f"{e['severity']} {e['kind']} on {e['dataset']}:{e['signal']} at {e['start']} ({e['detail']})" for e in top) + ".")
        parts.append("Run a thinking cycle to investigate causes and update the hypothesis board.")
        return " ".join(parts)
