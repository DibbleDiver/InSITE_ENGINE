"""Read-only data tools over a PipelineResult.

Every tool takes a validated input dict and returns compact text. They never
evaluate model-written expressions: filters are structured and applied with
pandas operations only.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ...contracts import ALL_SIGNALS, FEATURE_SEP, Dataset, PipelineResult

MAX_RESULT_CHARS = 4000
DEFAULT_POINTS = 200
MAX_POINTS = 500
NICE_MINUTES = [1, 2, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1440, 2880, 10080]


class ToolError(Exception):
    """A tool could not run with the given input; the message goes back to the model."""


# ----------------------------------------------------------------- formatting


def fmt_num(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return ""
    if f == 0:
        return "0"
    if abs(f) >= 1e6 or abs(f) < 1e-3:
        return f"{f:.3g}"
    if float(f).is_integer() and abs(f) < 1e9:
        return str(int(f))
    return f"{f:.4g}" if abs(f) < 1 else f"{f:.2f}".rstrip("0").rstrip(".")


def fmt_ts(ts: Any) -> str:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)):
        return ""
    try:
        t = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return str(ts)
    if pd.isna(t):
        return ""
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
        return t.strftime("%Y-%m-%dT%H:%M") + ("Z" if t.second == 0 else f":{t.second:02d}Z")
    return t.strftime("%Y-%m-%dT%H:%M")


def fmt_cell(value: Any, width: int = 160) -> str:
    if value is None:
        return ""
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return fmt_ts(value)
    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        return fmt_num(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).replace("\n", " ").replace("|", "/")
    return text if len(text) <= width else text[: width - 3] + "..."


def table(rows: list[list[str]], header: list[str]) -> str:
    lines = [" | ".join(header)]
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    note = (f"\n[output truncated: showing the first {limit} of {len(text)} characters; "
            f"narrow the query (window, filters, limit, resample) to see the rest]")
    return text[:limit] + note, True


def nice_freq(minutes: float) -> str:
    for m in NICE_MINUTES:
        if m >= minutes:
            break
    else:
        m = NICE_MINUTES[-1]
    if m % 1440 == 0:
        return f"{m // 1440}D"
    if m % 60 == 0:
        return f"{m // 60}h"
    return f"{m}min"


def freq_minutes(rule: str) -> float | None:
    try:
        offset = pd.tseries.frequencies.to_offset(rule)
        anchor = pd.Timestamp("2000-01-03T00:00:00Z")  # calendar offsets (e.g. '1D') are not Timedeltas in pandas 3
        minutes = (anchor + offset - anchor).total_seconds() / 60.0
    except (ValueError, TypeError, AttributeError):
        return None
    return minutes if minutes > 0 else None


# ----------------------------------------------------------------- access


class DataTools:
    def __init__(self, result: PipelineResult, new_event_ids: set[str] | None = None):
        self.result = result
        self.new_event_ids = set(new_event_ids or [])

    # lookups ---------------------------------------------------------------
    def dataset(self, ds_id: str) -> Dataset:
        ds = self.result.datasets.get(ds_id)
        if ds is None:
            # tolerate case / stray whitespace differences
            for key, candidate in self.result.datasets.items():
                if key.lower() == str(ds_id).strip().lower():
                    return candidate
            raise ToolError(f"unknown dataset {ds_id!r}; available: {', '.join(sorted(self.result.datasets)) or '(none)'}")
        return ds

    def numeric_column(self, ds: Dataset, name: str) -> str:
        if name in ds.frame.columns and pd.api.types.is_numeric_dtype(ds.frame[name]):
            return name
        if name in ds.frame.columns:
            raise ToolError(f"column {name!r} of {ds.id} is not numeric; numeric signals: {', '.join(ds.signals)}")
        raise ToolError(
            f"unknown signal {name!r} in {ds.id}; signals: {', '.join(ds.signals) or '(none)'}"
            + (f"; feature columns are available as <signal>{FEATURE_SEP}mean/std/z/roc" if ds.signals else "")
        )

    @staticmethod
    def parse_ts(value: Any, what: str) -> pd.Timestamp:
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError) as e:
            raise ToolError(f"{what} is not a valid timestamp: {value!r} ({e})") from e
        if pd.isna(ts):
            raise ToolError(f"{what} is not a valid timestamp: {value!r}")
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def window_bounds(self, args: dict, start_key: str = "start", end_key: str = "end") -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        start = self.parse_ts(args[start_key], start_key) if args.get(start_key) else None
        end = self.parse_ts(args[end_key], end_key) if args.get(end_key) else None
        if start is not None and end is not None and end < start:
            raise ToolError(f"{end_key} ({fmt_ts(end)}) is before {start_key} ({fmt_ts(start)})")
        return start, end

    def series(self, ds: Dataset, signal: str, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.Series:
        if not ds.is_timeseries:
            raise ToolError(f"dataset {ds.id} has no time axis; use query_rows for it")
        col = self.numeric_column(ds, signal)
        frame = ds.frame[[ds.time_column, col]].dropna(subset=[ds.time_column])
        ts = pd.to_datetime(frame[ds.time_column], utc=True)
        s = pd.Series(frame[col].astype(float).to_numpy(), index=pd.DatetimeIndex(ts), name=signal).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        return s

    @staticmethod
    def interval_minutes(index: pd.DatetimeIndex) -> float | None:
        if len(index) < 2:
            return None
        diffs = pd.Series(index).diff().dropna().dt.total_seconds().to_numpy() / 60.0
        diffs = diffs[diffs > 0]
        return float(np.median(diffs)) if len(diffs) else None

    def frame_window(self, ds: Dataset, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
        df = ds.frame
        if not ds.is_timeseries or (start is None and end is None):
            return df
        ts = pd.to_datetime(df[ds.time_column], utc=True)
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= ts >= start
        if end is not None:
            mask &= ts <= end
        return df[mask]

    def _events_for(self, ds_id: str | None = None, signal: str | None = None) -> pd.DataFrame:
        ev = self.result.events
        if ev is None or ev.empty:
            return pd.DataFrame(columns=list(ev.columns) if ev is not None else [])
        if ds_id is not None:
            ev = ev[ev["dataset"] == ds_id]
        if signal is not None:
            ev = ev[ev["signal"] == signal]
        return ev

    # tools -----------------------------------------------------------------
    def list_datasets(self, args: dict) -> str:
        lines = [f"{len(self.result.datasets)} datasets; pipeline output generated {fmt_ts(self.result.generated_at)}"]
        states = self.result.states
        events = self.result.events
        for ds_id, ds in sorted(self.result.datasets.items())[:40]:
            start, end = ds.time_range()
            interval = None
            if ds.is_timeseries and not ds.frame.empty:
                interval = self.interval_minutes(pd.DatetimeIndex(pd.to_datetime(ds.frame[ds.time_column], utc=True)).sort_values())
            n_events = int((events["dataset"] == ds_id).sum()) if events is not None and not events.empty else 0
            abnormal = []
            if states is not None and not states.empty:
                st = states[states["dataset"] == ds_id]
                if not st.empty:
                    latest = st.sort_values("end").groupby("signal").tail(1)
                    abnormal = [f"{r.signal}={r.state}" for r in latest.itertuples(index=False) if r.state != "normal"]
            text_cols = ds.text_columns()
            lines.append(
                f"- {ds_id} [{ds.source}] rows={len(ds.frame)}"
                + (f" time={fmt_ts(start)}..{fmt_ts(end)}" if start is not None else " (no time axis)")
                + (f" interval={fmt_num(interval)}min" if interval else "")
                + f" events={n_events}"
                + f"\n    signals: {', '.join(ds.signals) or '(none)'}"
                + (f"\n    text columns: {', '.join(text_cols[:12])}{' ...' if len(text_cols) > 12 else ''}" if text_cols else "")
                + (f"\n    current non-normal states: {', '.join(abnormal)}" if abnormal else "")
            )
        if len(self.result.datasets) > 40:
            lines.append(f"... {len(self.result.datasets) - 40} more datasets not shown")
        return "\n".join(lines)

    def describe_signal(self, args: dict) -> str:
        ds = self.dataset(args["dataset"])
        start, end = self.window_bounds(args)
        s = self.series(ds, args["signal"], start, end)
        total = len(s)
        valid = s.dropna()
        lines = [f"{ds.id}:{args['signal']} ({ds.source})"]
        if total == 0:
            lines.append("no samples in the requested window")
            return "\n".join(lines)
        lines.append(f"samples={total} missing={fmt_num(100.0 * (total - len(valid)) / total)}% "
                     f"range={fmt_ts(s.index[0])}..{fmt_ts(s.index[-1])}"
                     + (f" interval={fmt_num(self.interval_minutes(s.index))}min" if total > 1 else ""))
        if len(valid):
            q = valid.quantile([0.05, 0.5, 0.95])
            lines.append(f"min={fmt_num(valid.min())} p05={fmt_num(q.iloc[0])} median={fmt_num(q.iloc[1])} "
                         f"mean={fmt_num(valid.mean())} p95={fmt_num(q.iloc[2])} max={fmt_num(valid.max())} "
                         f"std={fmt_num(valid.std())}")
            lines.append(f"latest value={fmt_num(valid.iloc[-1])} at {fmt_ts(valid.index[-1])}")
            lines.append(self._trend_line(valid))
        state = self._latest_state(ds.id, args["signal"])
        lines.append(f"latest state: {state}")
        ev = self._events_for(ds.id, args["signal"])
        if ev.empty:
            lines.append("events on this signal: none")
        else:
            counts = ev["kind"].value_counts().to_dict()
            lines.append("events on this signal: " + ", ".join(f"{k}={int(v)}" for k, v in counts.items()))
            recent = ev.sort_values("start", ascending=False).head(3)
            for r in recent.itertuples(index=False):
                lines.append(f"  {r.event_id} {r.kind} {r.severity} {fmt_ts(r.start)}..{fmt_ts(r.end)} "
                             f"score={fmt_num(r.score)} {fmt_cell(r.detail, 100)}")
        return "\n".join(lines)

    def _trend_line(self, valid: pd.Series) -> str:
        if len(valid) < 3:
            return "trend: n/a (fewer than 3 samples)"
        days = ((valid.index - valid.index[0]).total_seconds() / 86400.0).to_numpy()
        if days[-1] <= 0:
            return "trend: n/a"
        slope, _ = np.polyfit(days, valid.to_numpy(), 1)
        mean = valid.mean()
        pct = (100.0 * slope / abs(mean)) if mean else float("nan")
        return (f"trend: slope={fmt_num(slope)} per day"
                + (f" ({fmt_num(pct)}% of mean per day)" if not math.isnan(pct) else "")
                + f" over {fmt_num(days[-1])} days")

    def _latest_state(self, ds_id: str, signal: str) -> str:
        st = self.result.states
        if st is None or st.empty:
            return "unknown (no states)"
        rows = st[(st["dataset"] == ds_id) & (st["signal"].isin([signal, ALL_SIGNALS]))]
        if rows.empty:
            return "unknown (no state rows for this signal)"
        last = rows.sort_values("end").iloc[-1]
        return f"{last['state']} since {fmt_ts(last['start'])} (until {fmt_ts(last['end'])}, {fmt_num(last['duration_minutes'])} min)"

    def get_series(self, args: dict) -> str:
        refs = args["signals"]
        if not refs:
            raise ToolError("signals must contain at least one {dataset, signal} pair")
        if len(refs) > 6:
            raise ToolError("at most 6 signals per call")
        start, end = self.window_bounds(args)
        max_points = int(args.get("max_points") or DEFAULT_POINTS)
        max_points = max(10, min(max_points, MAX_POINTS))
        agg = args.get("agg") or "mean"
        series = []
        datasets = set()
        for ref in refs:
            ds = self.dataset(ref["dataset"])
            datasets.add(ds.id)
            series.append((ds, ref["signal"], self.series(ds, ref["signal"], start, end)))
        raw_points = max(len(s) for _, _, s in series)
        if raw_points == 0:
            raise ToolError("no samples in the requested window")
        intervals = [self.interval_minutes(s.index) or 0 for _, _, s in series]
        lo = min(s.index.min() for _, _, s in series if len(s))
        hi = max(s.index.max() for _, _, s in series if len(s))
        span_min = max((hi - lo).total_seconds() / 60.0, 1.0)
        rule = args.get("resample")
        note = ""
        if rule:
            if freq_minutes(rule) is None:
                raise ToolError(f"resample {rule!r} is not a valid pandas offset alias (try '5min', '1h', '1D')")
            note = f"resampled to {rule} ({agg})"
        else:
            needed = span_min / max_points
            if raw_points > max_points or (len(datasets) > 1 and len(set(round(i) for i in intervals)) > 1):
                rule = nice_freq(max(needed, max(intervals)))
                note = f"downsampled to {rule} ({agg}) from {raw_points} raw points"
        if rule:
            frames = [self._resample(s, rule, agg) for _, _, s in series]
        else:
            frames = [s for _, _, s in series]
        single = len(datasets) == 1
        df = pd.concat(frames, axis=1, join="outer")
        df.columns = [sig if single else f"{ds.id}:{sig}" for ds, sig, _ in series]
        df = df.sort_index()
        if rule and len(df) > max_points:
            # the explicit grid still has too many rows: show the last max_points
            note += f"; showing the last {max_points} of {len(df)} rows"
            df = df.tail(max_points)
        header = f"{len(df)} rows from {fmt_ts(df.index.min())} to {fmt_ts(df.index.max())}"
        if note:
            header += f" ({note})"
        rows = [[fmt_ts(ts)] + [fmt_num(v) for v in vals] for ts, vals in zip(df.index, df.to_numpy())]
        return header + "\n" + table(rows, ["time"] + list(df.columns))

    @staticmethod
    def _resample(s: pd.Series, rule: str, agg: str) -> pd.Series:
        r = s.resample(rule)
        try:
            return getattr(r, agg)()
        except AttributeError as e:  # pragma: no cover - schema restricts agg
            raise ToolError(f"unsupported agg {agg!r}") from e

    def query_rows(self, args: dict) -> str:
        ds = self.dataset(args["dataset"])
        start, end = self.window_bounds(args)
        df = self.frame_window(ds, start, end)
        notes = []
        if (start is not None or end is not None) and not ds.is_timeseries:
            notes.append("dataset has no time axis: start/end ignored")
        total_before = len(df)
        for f in args.get("filters") or []:
            df = df[self._filter_mask(df, ds, f)]
        matched = len(df)
        sort_by = args.get("sort_by")
        if sort_by:
            if sort_by not in df.columns:
                raise ToolError(f"sort_by column {sort_by!r} not in {ds.id}; columns: {', '.join(map(str, df.columns))}")
            df = df.sort_values(sort_by, ascending=not args.get("descending", False), kind="stable")
        limit = max(1, min(int(args.get("limit") or 50), 200))
        shown = df.head(limit)
        columns = args.get("columns")
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ToolError(f"unknown columns {missing}; columns: {', '.join(map(str, df.columns))}")
        else:
            columns = [c for c in df.columns if FEATURE_SEP not in str(c)]
            if len(columns) > 12:
                notes.append(f"{len(columns) - 12} columns hidden (pass columns=[...] to pick): "
                             + ", ".join(map(str, columns[12:])))
                columns = columns[:12]
        header = f"{ds.id}: {matched} of {total_before} rows match; showing {len(shown)}"
        if notes:
            header += " (" + "; ".join(notes) + ")"
        rows = [[fmt_cell(v) for v in rec] for rec in shown[columns].itertuples(index=False, name=None)]
        return header + "\n" + table(rows, [str(c) for c in columns])

    def _filter_mask(self, df: pd.DataFrame, ds: Dataset, f: dict) -> pd.Series:
        col, op, value = f["column"], f["op"], f["value"]
        if col not in df.columns:
            raise ToolError(f"unknown column {col!r} in {ds.id}; columns: {', '.join(map(str, df.columns))}")
        s = df[col]
        if op == "contains":
            return s.astype(str).str.contains(str(value), case=False, regex=False, na=False)
        if op == "in":
            if not isinstance(value, list):
                raise ToolError("'in' needs a list value")
            if pd.api.types.is_numeric_dtype(s):
                nums = []
                for v in value:
                    try:
                        nums.append(float(v))
                    except (TypeError, ValueError):
                        raise ToolError(f"column {col!r} is numeric but {v!r} is not a number")
                return s.astype(float).isin(nums)
            lowered = {str(v).lower() for v in value}
            return s.astype(str).str.lower().isin(lowered)
        if pd.api.types.is_datetime64_any_dtype(s):
            target = self.parse_ts(value, f"filter value for {col}")
            left = pd.to_datetime(s, utc=True)
        elif pd.api.types.is_numeric_dtype(s):
            try:
                target = float(value)
            except (TypeError, ValueError):
                raise ToolError(f"column {col!r} is numeric but {value!r} is not a number")
            left = s.astype(float)
        else:
            if op in (">", ">=", "<", "<="):
                raise ToolError(f"column {col!r} is text; ordering comparisons need a numeric or timestamp column")
            left = s.astype(str).str.lower()
            target = str(value).lower()
        ops = {
            "==": left == target, "!=": left != target, ">": left > target, ">=": left >= target,
            "<": left < target, "<=": left <= target,
        }
        return ops[op].fillna(False) if hasattr(ops[op], "fillna") else ops[op]

    def get_events(self, args: dict) -> str:
        ev = self.result.events
        if ev is None or ev.empty:
            return "no events were detected by the pipeline"
        start, end = self.window_bounds(args)
        mask = pd.Series(True, index=ev.index)
        if args.get("dataset"):
            mask &= ev["dataset"] == self.dataset(args["dataset"]).id
        if args.get("signal"):
            mask &= ev["signal"] == args["signal"]
        if args.get("kind"):
            mask &= ev["kind"] == args["kind"]
        if args.get("severity"):
            mask &= ev["severity"] == args["severity"]
        if args.get("min_score") is not None:
            mask &= pd.to_numeric(ev["score"], errors="coerce") >= float(args["min_score"])
        starts = pd.to_datetime(ev["start"], utc=True)
        ends = pd.to_datetime(ev["end"], utc=True)
        if start is not None:
            mask &= ends >= start
        if end is not None:
            mask &= starts <= end
        sel = ev[mask]
        total = len(sel)
        if args.get("sort") == "score":
            sel = sel.assign(_score=pd.to_numeric(sel["score"], errors="coerce")).sort_values("_score", ascending=False)
        else:
            sel = sel.sort_values("start", ascending=False)
        limit = max(1, min(int(args.get("limit") or 30), 100))
        shown = sel.head(limit)
        new_count = int(shown["event_id"].isin(self.new_event_ids).sum()) if self.new_event_ids else 0
        header = f"{total} of {len(ev)} events match; showing {len(shown)}" + (f" ({new_count} marked NEW since the last cycle)" if new_count else "")
        rows = []
        for r in shown.itertuples(index=False):
            rows.append([
                str(r.event_id), "NEW" if r.event_id in self.new_event_ids else "", str(r.dataset), str(r.signal), str(r.kind),
                str(r.severity), fmt_ts(r.start), fmt_ts(r.end), fmt_num(r.peak_value), fmt_num(r.score), fmt_cell(r.detail, 120),
            ])
        return header + "\n" + table(rows, ["event_id", "new", "dataset", "signal", "kind", "severity", "start", "end", "peak", "score", "detail"])

    def get_states(self, args: dict) -> str:
        st = self.result.states
        if st is None or st.empty:
            return "no state intervals were produced by the pipeline"
        start, end = self.window_bounds(args)
        mask = pd.Series(True, index=st.index)
        if args.get("dataset"):
            mask &= st["dataset"] == self.dataset(args["dataset"]).id
        if args.get("signal"):
            mask &= st["signal"] == args["signal"]
        if args.get("state"):
            mask &= st["state"] == args["state"]
        starts = pd.to_datetime(st["start"], utc=True)
        ends = pd.to_datetime(st["end"], utc=True)
        if start is not None:
            mask &= ends >= start
        if end is not None:
            mask &= starts <= end
        sel = st[mask]
        if args.get("current_only"):
            sel = sel.sort_values("end").groupby(["dataset", "signal"]).tail(1)
        total = len(sel)
        sel = sel.sort_values("start", ascending=False)
        limit = max(1, min(int(args.get("limit") or 50), 200))
        shown = sel.head(limit)
        header = f"{total} state intervals match; showing {len(shown)}" + (" (current state of each signal)" if args.get("current_only") else "")
        rows = [[str(r.dataset), str(r.signal), str(r.state), fmt_ts(r.start), fmt_ts(r.end), fmt_num(r.duration_minutes)]
                for r in shown.itertuples(index=False)]
        return header + "\n" + table(rows, ["dataset", "signal", "state", "start", "end", "minutes"])

    def correlate(self, args: dict) -> str:
        start, end = self.window_bounds(args)
        ds_a = self.dataset(args["a"]["dataset"])
        ds_b = self.dataset(args["b"]["dataset"])
        a = self.series(ds_a, args["a"]["signal"], start, end)
        b = self.series(ds_b, args["b"]["signal"], start, end)
        if a.dropna().empty or b.dropna().empty:
            raise ToolError("one of the signals has no samples in the window")
        rule = args.get("resample")
        if rule:
            if freq_minutes(rule) is None:
                raise ToolError(f"resample {rule!r} is not a valid pandas offset alias")
        else:
            rule = nice_freq(max(self.interval_minutes(a.index) or 1, self.interval_minutes(b.index) or 1))
        step_min = freq_minutes(rule) or 0
        max_lag = max(0, min(int(args.get("max_lag") if args.get("max_lag") is not None else 12), 96))
        mode = args.get("mode") or "levels"
        ra = a.resample(rule).mean()
        rb = b.resample(rule).mean()
        df = pd.concat([ra.rename("a"), rb.rename("b")], axis=1, join="inner")
        if mode == "diff":
            df = df.diff()
        df = df.dropna()
        n = len(df)
        if n < 5:
            raise ToolError(f"only {n} aligned points on a {rule} grid; widen the window or choose a coarser resample")
        if df["a"].std() == 0 or df["b"].std() == 0:
            raise ToolError("one of the signals is constant on this grid (zero variance); correlation is undefined")
        label_a = f"{ds_a.id}:{args['a']['signal']}"
        label_b = f"{ds_b.id}:{args['b']['signal']}"
        results = []
        for k in range(-max_lag, max_lag + 1):
            shifted = df["b"].shift(-k)  # b at t+k against a at t
            pair = pd.concat([df["a"], shifted], axis=1).dropna()
            if len(pair) < 5:
                continue
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if r is None or math.isnan(r):
                continue
            results.append((k, float(r), len(pair)))
        if not results:
            raise ToolError("not enough overlapping points to correlate")
        r0 = next((r for k, r, _ in results if k == 0), float("nan"))
        best_k, best_r, best_n = max(results, key=lambda t: (abs(t[1]), -abs(t[0])))
        lines = [f"correlate {label_a} vs {label_b}: grid={rule}, mode={mode}, aligned points={n}, "
                 f"window={fmt_ts(df.index.min())}..{fmt_ts(df.index.max())}"]
        lines.append(f"r at lag 0 = {fmt_num(r0)}")
        if best_k == 0:
            lead = "best lag is 0: no measurable lead/lag at this grid"
        elif best_k > 0:
            lead = (f"best lag = +{best_k} steps ({fmt_num(best_k * step_min)} min): {label_a} LEADS {label_b}, "
                    f"i.e. a change in {label_a} is followed by {label_b} about {fmt_num(best_k * step_min)} min later")
        else:
            lead = (f"best lag = {best_k} steps ({fmt_num(-best_k * step_min)} min): {label_b} LEADS {label_a}, "
                    f"i.e. a change in {label_b} is followed by {label_a} about {fmt_num(-best_k * step_min)} min later")
        lines.append(f"{lead}; r at best lag = {fmt_num(best_r)} (n={best_n})")
        strength = "strong" if abs(best_r) >= 0.7 else "moderate" if abs(best_r) >= 0.4 else "weak" if abs(best_r) >= 0.2 else "negligible"
        lines.append(f"interpretation: {strength} {'positive' if best_r > 0 else 'negative'} linear association"
                     + ("; mode=levels can be inflated by shared trends or daily cycles, confirm with mode='diff'" if mode == "levels" else "")
                     + "; correlation alone does not establish cause")
        if n < 30:
            lines.append(f"caution: only {n} aligned points, estimates are noisy")
        if max_lag > 0:
            shown = results if len(results) <= 25 else [t for t in results if t[0] % max(1, round(len(results) / 25)) == 0 or t[0] in (0, best_k)]
            lines.append("r by lag (steps): " + ", ".join(f"{k:+d}:{fmt_num(r)}" for k, r, _ in shown))
        return "\n".join(lines)

    def profile(self, args: dict) -> str:
        ds = self.dataset(args["dataset"])
        start, end = self.window_bounds(args)
        s = self.series(ds, args["signal"], start, end).dropna()
        hours = args.get("hours")
        hours_note = ""
        if hours:
            h_from, h_to = int(hours["from"]), int(hours["to"])
            if not (0 <= h_from <= 23 and 0 <= h_to <= 23):
                raise ToolError("hours.from and hours.to must be between 0 and 23")
            hr = s.index.hour
            if h_from <= h_to:
                s = s[(hr >= h_from) & (hr <= h_to)]
            else:
                s = s[(hr >= h_from) | (hr <= h_to)]
            hours_note = f", hours {h_from:02d}-{h_to:02d} UTC inclusive"
        if s.empty:
            raise ToolError("no samples match the window/hours filter")
        group_by = args["group_by"]
        stat = args["stat"]
        if group_by == "hour":
            keys = s.index.hour
            key_label = "hour"
            key_fmt = lambda k: f"{int(k):02d}"  # noqa: E731
        elif group_by == "weekday":
            keys = s.index.dayofweek
            key_label = "weekday"
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            key_fmt = lambda k: names[int(k)]  # noqa: E731
        else:
            keys = s.index.floor("D")
            key_label = "day"
            key_fmt = lambda k: pd.Timestamp(k).strftime("%Y-%m-%d")  # noqa: E731
        grouped = s.groupby(keys)
        if stat == "p05":
            values = grouped.quantile(0.05)
        elif stat == "p95":
            values = grouped.quantile(0.95)
        else:
            values = getattr(grouped, stat)()
        counts = grouped.count()
        lines = [f"profile {ds.id}:{args['signal']} by {group_by}, stat={stat}{hours_note}; "
                 f"{len(s)} samples {fmt_ts(s.index.min())}..{fmt_ts(s.index.max())}, {len(values)} groups"]
        shown = values
        if group_by == "day" and len(values) > 60:
            lines.append(f"showing the last 60 of {len(values)} days")
            shown = values.tail(60)
        rows = [[key_fmt(k), fmt_num(v), fmt_num(counts.loc[k])] for k, v in shown.items()]
        lines.append(table(rows, [key_label, stat, "n"]))
        if group_by == "day" and len(values) >= 3:
            x = np.arange(len(values), dtype=float)
            y = values.to_numpy(dtype=float)
            ok = ~np.isnan(y)
            if ok.sum() >= 3:
                slope, _ = np.polyfit(x[ok], y[ok], 1)
                mean = float(np.nanmean(y))
                pct = 100.0 * slope / abs(mean) if mean else float("nan")
                lines.append(f"trend of the daily {stat}: {fmt_num(slope)} per day"
                             + (f" ({fmt_num(pct)}% of its mean per day)" if not math.isnan(pct) else "")
                             + f"; first={fmt_num(y[ok][0])} last={fmt_num(y[ok][-1])}")
        return "\n".join(lines)

    def compare_windows(self, args: dict) -> str:
        ds = self.dataset(args["dataset"])
        a0, a1 = self.window_bounds(args["window_a"])
        b0, b1 = self.window_bounds(args["window_b"])
        sa = self.series(ds, args["signal"], a0, a1).dropna()
        sb = self.series(ds, args["signal"], b0, b1).dropna()
        if len(sa) < 2 or len(sb) < 2:
            raise ToolError(f"need at least 2 samples in each window (got {len(sa)} and {len(sb)})")
        lines = [f"compare_windows {ds.id}:{args['signal']}"]
        for name, s in (("A", sa), ("B", sb)):
            lines.append(f"window {name} {fmt_ts(s.index.min())}..{fmt_ts(s.index.max())}: n={len(s)} mean={fmt_num(s.mean())} "
                         f"std={fmt_num(s.std())} min={fmt_num(s.min())} max={fmt_num(s.max())}")
        diff = float(sb.mean() - sa.mean())
        pct = 100.0 * diff / abs(sa.mean()) if sa.mean() else float("nan")
        var_a, var_b = float(sa.var()), float(sb.var())
        pooled = math.sqrt(((len(sa) - 1) * var_a + (len(sb) - 1) * var_b) / max(len(sa) + len(sb) - 2, 1))
        d = diff / pooled if pooled > 0 else float("inf") if diff else 0.0
        se = math.sqrt(var_a / len(sa) + var_b / len(sb))
        t = diff / se if se > 0 else float("inf") if diff else 0.0
        size = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        lines.append(f"change in mean B-A = {fmt_num(diff)}" + (f" ({fmt_num(pct)}%)" if not math.isnan(pct) else "")
                     + f"; Cohen's d = {fmt_num(d)} ({size} effect); Welch t = {fmt_num(t)}")
        lines.append("note: samples within a window are autocorrelated, so t overstates significance; weigh d and the raw change")
        return "\n".join(lines)
