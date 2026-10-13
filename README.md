# InSITE Engine

An autonomous engine that watches a site through its operational data exports and thinks about what it sees. It has two halves:

1. **The pipeline** (deterministic, no model): turns raw CSV exports from four systems (TRIO telemetry, TCL tanks and pumps, AX work orders, Meters) into clean time-series, rolling features, detected events (spikes, step changes, flatlines, gaps) and per-signal states (normal, anomalous, flatlined, offline).
2. **MŸND** (the Mind): a self-directed reasoning layer on Claude Fable 5.1. Each cycle it perceives what is new since it last thought, investigates with a tool set over the pipeline output, has every claim it makes graded by an independent critic, and only then commits what it learned to a persistent memory and writes a report for the operations team.

The point of the split: thresholds catch the obvious faults cheaply and reproducibly; MŸND is for what thresholds cannot do, which is joining evidence across systems (a pump trip that explains a pressure drop, a flow dip and a work order), finding slow problems (sensor drift, a rising minimum night flow that means a leak), and saying how sure it is and why.

## Quick start

```bash
pip install -r requirements.txt
python main.py demo            # write a synthetic site into InSITE_Engine/data/raw
python main.py run             # pipeline only: events.csv, states.csv, features/, summary.json
python main.py think --offline # perception-only MŸND cycle, no model calls
export ANTHROPIC_API_KEY=...   # or `ant auth login`
python main.py think           # one full cognitive cycle (costs money; see budgets below)
python main.py ask "why did outlet pressure drop on 3 October?"
python main.py live --max-cycles 4 --budget-usd 10
```

Real data goes under `input_path` (default `InSITE_Engine/data/raw`) in one folder per source: `TRIO/`, `TCL/`, `AX/`, `Meters/`, any number of CSV files each. Headers are normalised to snake_case, the time column is found and parsed to UTC (day-first formats are detected from the data), numeric text is coerced, and text-only datasets such as work orders pass through intact.

Outputs land under `output_path` (default `InSITE_Engine/output`): the pipeline files above, and MŸND's memory under `output/mind/` with `reports/latest.md`, one journal per cycle (full transcript, every tool call and its evidence, critic verdicts, cost) and `lessons/`.

## What a cycle does

1. **Perceive.** Run the pipeline, fingerprint the result and diff it against memory, so the model is briefed on what is new: new events, state transitions, stale feeds, new datasets. The brief also carries the memory digest: open hypotheses ranked by how much testing them would teach, open questions, lessons, and the agenda left by the previous cycle.
2. **Investigate.** An agent loop on Fable 5.1 with 20 tools. Data tools (`list_datasets`, `describe_signal`, `get_series`, `query_rows`, `get_events`, `get_states`, `correlate`, `profile`, `compare_windows`) are read-only over the pipeline output and every result is tagged with an evidence id. Memory tools (`propose_hypothesis`, `update_hypothesis`, `retire_hypothesis`, `add_question`, `resolve_question`, `record_insight`, `save_lesson`, `read_lesson`, `delete_lesson`) must cite evidence ids from the current cycle or they are rejected. `send_to_user` shows a message verbatim; `conclude_cycle` ends the cycle with the summary the team reads.
3. **Critique.** A fresh-context critic call (no tools, structured JSON) receives every claim with the evidence excerpts it cited and grades each one supported, weak or unsupported. The verdicts are applied deterministically: unsupported insights are rejected (and listed as such in the report), weak ones become tentative, unsupported hypothesis updates are reverted, weak ones count half, and the critic's follow-up questions join the agenda.
4. **Consolidate.** Commit the staged memory changes, store the fingerprint, write the journal and the report, set the next agenda.

Confidence is never asserted by the model. A hypothesis holds a probability; the model reports direction (supports or contradicts) and strength (weak, moderate, strong), and the harness applies a log-odds update with likelihood ratios 1.5, 3 and 8. Status follows from the probability (supported at 0.85 and above, refuted at 0.10 and below) and the full confidence history stays on the board across cycles.

## Cost and safety controls

All in `InSITE_Engine/config/config.yaml` under `mind:`.

- `budget_usd_per_cycle` (default $5): the model is told to wrap up at 80% and the loop stops at 100%. `budget_usd_total` and `--budget-usd` cap a whole `live` run. Spend is priced from the token usage the API reports.
- `max_steps`: model turns per cycle, with a wrap-up notice three steps before the limit.
- `effort`: `low` to `max` (the single biggest lever on cost and depth); `--effort` overrides it per run.
- `fallbacks: default`: a safety-classifier refusal on Fable is re-run server-side on Anthropic's recommended fallback model and the report says which model served the cycle. Set to `null` to disable; a refusal then ends the cycle with nothing committed.
- `live` mode idles when no new data has arrived (after `max_idle_cycles` cycles of thinking about its open agenda), backs off after an error or refusal, and stops on Ctrl-C.

Data content is treated as evidence, never as instructions: the system prompt says so, the critic is told so, and the tools never take file paths from the model. Nothing MŸND does writes back to the site systems; its only outputs are its memory and reports.

## The demo site

`python main.py demo` writes a week of 5-minute telemetry for a synthetic utility site with six embedded scenarios (the ground truth is written beside the data as `demo_ground_truth.json`):

| scenario | what the pipeline sees | what is left for MŸND |
|---|---|---|
| pump P2 trips on overload for 3 h on day 3; outlet pressure drops 10 min later, zone flow 15 min later; a work order is logged 2 h after | flatline and step changes on P2 current, step changes on pressure and flow | that these are one incident: `correlate` in the trip window gives r = 0.99 with P2 leading pressure by 10 minutes, and `query_rows` finds the work order |
| customer bulk meter frozen for 8 h on day 5 | high-severity flatline | that a frozen meter is not a real plateau |
| TRIO radio outage of 2 h on day 4 | dataset-level gap; every TRIO signal offline in states | nothing further |
| tank level B drifts up ~0.5 points/day | nothing (no threshold is crossed) | `describe_signal` reports the trend; comparing with tank A separates sensor drift from a real change |
| a leak from day 4: minimum night flow up ~25%; a low-pressure complaint on day 5 | nothing (the rise is gradual) | `profile` by day with `stat: min` over 01:00-04:00 shows the step from ~14.5 to ~17.6 l/s; the complaint is in AX |
| two single-sample glitch spikes in inlet pressure | spikes | that they are glitches, not process events |

On this data the pipeline reports 11 events, all of them real. The repository's tests include a scripted end-to-end MŸND cycle on this output, so the whole chain can be exercised without an API key.

## Layout

```
main.py                       command line: run | demo | think | ask | live
InSITE_Engine/
  config/config.yaml          paths, pipeline thresholds, MŸND settings and budgets
  contracts.py                Dataset / PipelineResult, event and state schemas (the seam between the halves)
  ingestion/ cleaning/ normalisation/ features/ events/ states/   pipeline stages
  pipeline.py                 run_pipeline(cfg) -> PipelineResult
  demo/                       synthetic site generator (python -m InSITE_Engine.demo)
  mind/                       MŸND: perception, agent loop, tools, critic, memory, reporting, console
tests/                        80 tests; `python -m pytest`
```

## Fable 5.1 notes

The model client streams every request (`client.beta.messages.stream`), keeps each cycle's conversation append-only so Fable's preserved thinking stays valid (frozen system prompt and tool list, assistant turns echoed exactly as received, harness notes appended after tool results and never removed), uses `output_config.effort` as the only depth control, leaves `tool_choice` on auto, validates every tool input against its schema (eager input streaming is on), and handles `refusal`, `max_tokens`, `pause_turn` and mid-output fallback blocks. The critic uses structured JSON output. Thinking is shown on the console as the summaries the API returns (`thinking_display: summarized`).
