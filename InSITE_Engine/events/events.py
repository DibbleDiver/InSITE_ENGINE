"""Event detection: turn featured telemetry into a small, precise event table.

Every row follows ``contracts.EVENT_COLUMNS``.  ``start``/``end`` are tz-aware
UTC sample timestamps and are inclusive: an event covers the samples from
``start`` to ``end``; point events have ``end == start``.

Kinds (``contracts.EVENT_KINDS``) and what ``score`` / ``peak_value`` mean:

* ``spike_high`` / ``spike_low``: samples whose robust z (``sig__z``) exceeds
  ``spike_z`` in magnitude.  Consecutive anomalous samples merge into one
  event; ``peak_value`` is the most extreme raw value, ``score`` the peak |z|.
* ``step_change``: a sustained level shift.  At every sample a straight line
  is fitted to the ``step_window`` samples before and to the ``step_window``
  samples from the sample on (real time as regressor); the shift is the jump
  between the two fitted levels at the boundary, measured in units of the
  larger of the two residual standard deviations ("pooled std", conservative,
  floored at the signal's global noise scale like the z-score scale).  Fitting lines rather than comparing plain
  means makes a demand ramp score ~0 while a genuine jump scores its full
  size.  The statistic is calibrated per signal against its own quiet
  boundaries (autocorrelated noise makes adjacent windows disagree more than
  white noise would; the test is never made more lenient than the white-noise
  ideal), boundaries whose windows straddle a gap are skipped, and consecutive
  candidate boundaries (plus same-direction candidates closer than
  ``step_window`` samples) merge into the single strongest one, so one physical
  step yields one point event.  Each candidate is then confirmed as a level
  jump rather than a change of slope (a tank starting to drain is a kink, not
  a step) by comparing a discontinuous two-line fit with the best continuous
  two-slope fit: the jump must beat the hinge by F >= step_z**2.  ``score`` is
  the calibrated shift in pooled-std units, ``peak_value`` the raw value at
  the boundary sample.
* ``flatline``: at least ``flatline_min_samples`` identical consecutive
  readings (and at least 4x the signal's median run of identical values, so
  coarsely quantised slow signals are not flagged).  Signals that are constant
  for their whole history, or whose dominant value covers > 95% of the samples,
  are constants/status values, not faults, and are skipped.  ``peak_value`` is
  the stuck value, ``score`` the run length in samples.
* ``gap``: with ``signal == ALL_SIGNALS``, a hole in the dataset's timestamps
  longer than ``gap_factor`` x the median interval and at least
  ``min_gap_minutes``; ``start``/``end`` are the first and last expected
  sample that is missing, ``score`` the number of missing samples.  With a
  signal name, a run of NaN readings of that signal lasting at least
  ``min_gap_minutes``; ``score`` is the number of missing readings.
  ``peak_value`` is NaN for gaps.

Severity (documented rule): the score is compared with its threshold.
spikes: |z| < 2x spike_z low, < 4x medium, else high, and a single-sample
spike is capped at medium (most often a telemetry glitch); steps: the same
2x / 4x rule on step_z; flatlines: run length < 2x flatline_min_samples low,
< 6x medium, else high; gaps: duration < 2x min_gap_minutes low, < 6x medium,
else high, and a dataset-level gap (the whole feed went silent) is one level
higher than a single signal dropping out.

Only telemetry is analysed: a dataset needs a ``timestamp`` axis, at least one
signal and regular sampling (at least half of its intervals within 25% of the
median).  Event logs such as work orders therefore produce no numeric events.

``event_id`` is the first 12 hex digits of sha1("dataset|signal|kind|start"),
so re-running on the same data yields identical ids.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from ..contracts import ALL_SIGNALS, EVENT_COLUMNS, FEATURE_SEP, SEVERITIES, Dataset, empty_events
from ..features.features import (
    compute_features,
    interval_minutes,
    minutes_since_start,
    noise_scale,
    sampling_regularity,
)
from ..normalisation.normalise import TIME_COLUMN, pipeline_config, select_signals

logger = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826
MIN_REGULAR_FRACTION = 0.5
STATIC_SHARE = 0.95            # dominant value share above which a signal is a status value
FLATLINE_RUN_MULTIPLIER = 4    # a flatline must also be this many median runs long
STEP_MAX_SPAN_FACTOR = 1.5     # a step window may not span more than 1.5x its nominal duration


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs (inclusive) of the True runs in a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not m.any():
        return []
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def fmt_duration(minutes: float) -> str:
    total = int(round(float(minutes)))
    days, rem = divmod(total, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h{mins:02d}m"
    if hours:
        return f"{hours}h{mins:02d}m"
    return f"{mins}m"


def fmt_value(value) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    text = f"{float(value):.6g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text


def fmt_time(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def event_id(dataset: str, signal: str, kind: str, start, salt: str = "") -> str:
    key = f"{dataset}|{signal}|{kind}|{pd.Timestamp(start).isoformat()}{salt}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _level(ratio: float, medium_at: float, high_at: float) -> int:
    if not np.isfinite(ratio) or ratio < medium_at:
        return 0
    return 1 if ratio < high_at else 2


def severity(kind: str, score: float, n_samples: int, duration_minutes: float, dataset_level: bool, pcfg: dict) -> str:
    if kind in ("spike_high", "spike_low"):
        level = _level(score / float(pcfg["spike_z"]), 2, 4)
        if n_samples == 1:
            level = min(level, 1)
    elif kind == "step_change":
        level = _level(score / float(pcfg["step_z"]), 2, 4)
    elif kind == "flatline":
        level = _level(n_samples / float(pcfg["flatline_min_samples"]), 2, 6)
    else:  # gap
        level = _level(duration_minutes / float(pcfg["min_gap_minutes"]), 2, 6)
        if dataset_level:
            level = min(level + 1, 2)
    return SEVERITIES[level]


def _record(dataset, source, signal, kind, start, end, peak_value, score, detail, n_samples, duration, pcfg):
    return {
        "event_id": "",
        "dataset": dataset,
        "source": source,
        "signal": signal,
        "kind": kind,
        "severity": severity(kind, score, n_samples, duration, signal == ALL_SIGNALS, pcfg),
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "peak_value": float(peak_value) if peak_value is not None and np.isfinite(peak_value) else np.nan,
        "score": float(score),
        "detail": detail,
    }


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #

def spike_events(x, z, stamps, sig, interval, pcfg, dataset, source) -> list[dict]:
    spike_z = float(pcfg["spike_z"])
    out = []
    with np.errstate(invalid="ignore"):
        masks = (("spike_high", z > spike_z), ("spike_low", z < -spike_z))
    for kind, mask in masks:
        for s, e in runs(mask):
            seg_x, seg_z = x[s:e + 1], z[s:e + 1]
            peak = np.nanmax(seg_x) if kind == "spike_high" else np.nanmin(seg_x)
            score = float(np.nanmax(np.abs(seg_z)))
            n = e - s + 1
            verb = "spiked to" if kind == "spike_high" else "dropped to"
            span = "1 sample" if n == 1 else f"{n} samples ({fmt_duration(n * interval)})"
            detail = f"{sig} {verb} {fmt_value(peak)} (z={score:.1f}) for {span} at {fmt_time(stamps[s])} UTC"
            out.append(_record(dataset, source, sig, kind, stamps[s], stamps[e], peak, score, detail, n, n * interval, pcfg))
    return out


def step_statistic(x: np.ndarray, minutes: np.ndarray, window: int, min_periods: int, floor: float,
                   interval: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample step statistic (see module docstring) plus the fitted levels before/after."""
    n = len(x)
    valid = np.isfinite(x)
    xf = np.where(valid, x, 0.0)
    tv = np.where(valid, minutes, 0.0)
    sums = pd.DataFrame({
        "n": valid.astype(float), "sx": xf, "st": tv, "stt": tv * tv, "stx": tv * xf, "sxx": xf * xf,
    })
    trailing = sums.rolling(window, min_periods=1).sum()
    before = trailing.shift(1)              # samples [i - window, i - 1]
    after = trailing.shift(-(window - 1))   # samples [i, i + window - 1]

    # The boundary sits half way between the last "before" sample and the first
    # "after" sample; both lines are evaluated there.
    t_boundary = np.concatenate(([np.nan], 0.5 * (minutes[1:] + minutes[:-1]))) if n > 1 else np.full(n, np.nan)

    def fit(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cnt, sx, st, stt, stx, sxx = (part[c].to_numpy(dtype=float) for c in ("n", "sx", "st", "stt", "stx", "sxx"))
        with np.errstate(divide="ignore", invalid="ignore"):
            den = cnt * stt - st * st
            slope = (cnt * stx - st * sx) / den
            intercept = (sx - slope * st) / cnt
            level = intercept + slope * t_boundary
            sse = sxx - 2 * intercept * sx - 2 * slope * stx + cnt * intercept ** 2 + 2 * intercept * slope * st + slope ** 2 * stt
            resid = np.sqrt(np.maximum(sse, 0.0) / np.maximum(cnt - 2, 1))
        ok = np.isfinite(den) & (den > 0) & (cnt >= min_periods)
        level[~ok] = np.nan
        resid[~ok] = np.nan
        return level, resid

    level_before, resid_before = fit(before)
    level_after, resid_after = fit(after)

    # Both windows and the boundary between them must be contiguous data: a line
    # extrapolated across a gap is not evidence of a step.
    tm = pd.Series(minutes)
    span = (tm.shift(-(window - 1)) - tm.shift(window)).to_numpy(dtype=float)
    max_span = STEP_MAX_SPAN_FACTOR * (2 * window - 1) * interval
    with np.errstate(invalid="ignore"):
        contiguous = span <= max_span
    pooled = np.fmax(resid_before, resid_after)
    if np.isfinite(floor):
        pooled = np.where(np.isfinite(pooled) & (pooled < floor), floor, pooled)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (level_after - level_before) / pooled
    d[~contiguous] = np.nan
    d[~np.isfinite(d)] = np.nan
    return d, level_before, level_after


def step_events(x, minutes, stamps, sig, interval, pcfg, dataset, source) -> list[dict]:
    window = max(3, int(pcfg["step_window"]))
    step_z = float(pcfg["step_z"])
    if len(x) < 2 * window:
        return []
    min_periods = max(3, int(np.ceil(0.75 * window)))
    floor = noise_scale(x, int(pcfg["robust_window"]))
    d, level_before, level_after = step_statistic(x, minutes, window, min_periods, floor, interval)
    d = d / null_inflation(d, window, step_z)
    with np.errstate(invalid="ignore"):
        candidate = np.abs(d) > step_z
    merged: list[tuple[int, int]] = []
    for s, e in runs(candidate):
        if merged:
            ps, pe = merged[-1]
            same_direction = np.sign(d[s:e + 1].sum()) == np.sign(d[ps:pe + 1].sum())
            if s - pe < window and same_direction:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    out = []
    for s, e in merged:
        seg = np.abs(d[s:e + 1])
        seg = np.where(np.isfinite(seg), seg, -np.inf)
        peak = s + int(np.argmax(seg))
        score = float(abs(d[peak]))
        f_stat = jump_versus_kink(x, minutes, peak, window, floor)
        if f_stat < step_z ** 2:
            logger.debug("%s: candidate step at %s rejected as a slope change (F=%.1f)", sig, stamps[peak], f_stat)
            continue
        before, after = float(level_before[peak]), float(level_after[peak])
        shift = after - before
        direction = "up" if shift > 0 else "down"
        detail = (f"{sig} stepped {direction} from {fmt_value(before)} to {fmt_value(after)} "
                  f"(shift {shift:+.4g}, {score:.1f} sd) at {fmt_time(stamps[peak])} UTC")
        out.append(_record(dataset, source, sig, "step_change", stamps[peak], stamps[peak], x[peak], score, detail, 1, interval, pcfg))
    return out


def null_inflation(d: np.ndarray, window: int, step_z: float, min_boundaries: int = 50) -> float:
    """How much noisier than white noise the step statistic is for this signal (>= 1).

    With white noise the difference of two fitted end-levels has a standard
    deviation of ``sqrt(2 * h_end)`` residual sds, where ``h_end`` is the
    leverage of a window's last sample.  Autocorrelated noise (and trend misfit)
    makes adjacent windows disagree more than that, so the robust spread of
    ``d`` over the quiet boundaries (|d| <= step_z) is compared with the
    white-noise value and the statistic is divided by the excess.  The factor
    is never below 1, so the test is never more lenient than the ideal.
    """
    finite = d[np.isfinite(d)]
    quiet = finite[np.abs(finite) <= step_z]
    if len(quiet) < min_boundaries:
        return 1.0
    null_sd = MAD_TO_SIGMA * float(np.median(np.abs(quiet - np.median(quiet))))
    h_end = 1.0 / window + 3.0 * (window - 1) / (window * (window + 1))
    white_sd = float(np.sqrt(2.0 * h_end))
    return max(1.0, null_sd / white_sd) if white_sd > 0 else 1.0


def jump_versus_kink(x: np.ndarray, minutes: np.ndarray, boundary: int, window: int, floor: float) -> float:
    """F statistic for "a level jump at ``boundary``" against "a continuous change of slope".

    Over the ``window`` samples either side of the boundary, a discontinuous
    two-line model (independent lines before and after) is compared with the
    best continuous hinge model (two slopes meeting at a knot searched over
    half a window either side).  A genuine step leaves the hinge model with a
    large residual; a kink (pump starts draining a tank, a demand ramp
    steepens) is explained equally well by both, so F stays small.  The
    residual variance is floored at ``floor**2`` so noise-free data cannot
    produce an infinite statistic.
    """
    n = len(x)
    lo, hi = max(0, boundary - window), min(n, boundary + window)
    xs, ts = x[lo:hi], minutes[lo:hi]
    valid = np.isfinite(xs)
    xs, ts = xs[valid], ts[valid]
    if len(xs) < 8 or boundary < 1:
        return float("inf")
    t_boundary = 0.5 * (minutes[boundary - 1] + minutes[boundary])
    tau = ts - t_boundary
    after = (tau > 0).astype(float)
    design = np.column_stack([np.ones_like(tau), tau, after, tau * after])
    rss_disc = _rss(design, xs)
    half = max(1, window // 2)
    knot_times = []
    for k in range(max(lo + 1, boundary - half), min(hi, boundary + half + 1)):
        knot_times.append(0.5 * (minutes[k - 1] + minutes[k]))  # a vertex between two samples
        knot_times.append(minutes[k])                            # a vertex exactly on a sample
    rss_cont = float("inf")
    for t_knot in knot_times:
        tk = ts - t_knot
        hinge = np.column_stack([np.ones_like(tk), tk, np.maximum(tk, 0.0)])
        rss_cont = min(rss_cont, _rss(hinge, xs))
    dof = max(1, len(xs) - 4)
    sigma2 = rss_disc / dof
    if np.isfinite(floor) and floor > 0:
        sigma2 = max(sigma2, floor * floor)
    if sigma2 <= 0:
        return float("inf") if rss_cont > rss_disc else 0.0
    return float((rss_cont - rss_disc) / sigma2)


def _rss(design: np.ndarray, y: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid)


def flatline_events(x, stamps, sig, interval, pcfg, dataset, source) -> list[dict]:
    min_samples = max(2, int(pcfg["flatline_min_samples"]))
    valid = np.isfinite(x)
    values = x[valid]
    if len(values) < 2 or np.all(values == values[0]):
        return []  # a constant, not a fault
    _, counts = np.unique(values, return_counts=True)
    if counts.max() / len(values) > STATIC_SHARE:
        return []  # a status/setpoint value that occasionally changes
    same = np.zeros(len(x), dtype=bool)
    same[1:] = valid[1:] & valid[:-1] & (x[1:] == x[:-1])
    starts = np.flatnonzero(~same)                      # every run (of valid or missing samples) starts here
    lengths = np.diff(np.append(starts, len(x)))
    run_is_valid = valid[starts]
    median_run = float(np.median(lengths[run_is_valid])) if run_is_valid.any() else 1.0
    min_run = max(min_samples, int(np.ceil(FLATLINE_RUN_MULTIPLIER * median_run)))
    out = []
    for s, length in zip(starts[run_is_valid].tolist(), lengths[run_is_valid].tolist()):
        if length < min_run:
            continue
        e = s + length - 1
        duration = length * interval
        value = x[s]
        detail = (f"{sig} stuck at {fmt_value(value)} for {fmt_duration(duration)} ({length} samples) "
                  f"from {fmt_time(stamps[s])} UTC")
        out.append(_record(dataset, source, sig, "flatline", stamps[s], stamps[e], value, float(length), detail, length, duration, pcfg))
    return out


def dataset_gap_events(stamps, interval, pcfg, dataset, source) -> list[dict]:
    gap_factor = float(pcfg["gap_factor"])
    min_gap = float(pcfg["min_gap_minutes"])
    if len(stamps) < 2:
        return []
    diffs = interval_minutes(pd.Series(stamps))
    hole_idx = np.flatnonzero((diffs > gap_factor * interval) & (diffs >= min_gap))
    out = []
    step = pd.Timedelta(minutes=interval)
    for i in hole_idx.tolist():
        last, nxt, hole = stamps[i], stamps[i + 1], float(diffs[i])
        missing = max(1, int(round(hole / interval)) - 1)
        start, end = last + step, nxt - step
        if end < start:
            end = start
        detail = (f"no data for {fmt_duration(hole)} between {fmt_time(last)} and {fmt_time(nxt)} UTC "
                  f"({missing} samples expected every {fmt_duration(interval)})")
        out.append(_record(dataset, source, ALL_SIGNALS, "gap", start, end, np.nan, float(missing), detail, missing, hole, pcfg))
    return out


def signal_gap_events(x, stamps, sig, interval, pcfg, dataset, source) -> list[dict]:
    min_gap = float(pcfg["min_gap_minutes"])
    out = []
    for s, e in runs(~np.isfinite(x)):
        duration = (stamps[e] - stamps[s]).total_seconds() / 60.0 + interval
        if duration < min_gap:
            continue
        n = e - s + 1
        detail = f"{sig} missing for {fmt_duration(duration)} ({n} samples) from {fmt_time(stamps[s])} UTC"
        out.append(_record(dataset, source, sig, "gap", stamps[s], stamps[e], np.nan, float(n), detail, n, duration, pcfg))
    return out


# --------------------------------------------------------------------------- #
# detect_events
# --------------------------------------------------------------------------- #

def _finish(records: list[dict]) -> pd.DataFrame:
    if not records:
        return empty_events()
    events = pd.DataFrame(records, columns=EVENT_COLUMNS)
    events = events.sort_values(["start", "signal", "kind"], kind="stable").reset_index(drop=True)
    ids = [event_id(r.dataset, r.signal, r.kind, r.start) for r in events.itertuples(index=False)]
    seen: dict[str, int] = {}
    for i, eid in enumerate(ids):  # identical keys are salted deterministically by order
        if eid in seen:
            seen[eid] += 1
            ids[i] = event_id(events.at[i, "dataset"], events.at[i, "signal"], events.at[i, "kind"], events.at[i, "start"], f"|{seen[eid]}")
        else:
            seen[eid] = 1
    events["event_id"] = ids
    events["start"] = pd.to_datetime(events["start"], utc=True).astype("datetime64[ns, UTC]")
    events["end"] = pd.to_datetime(events["end"], utc=True).astype("datetime64[ns, UTC]")
    events["peak_value"] = events["peak_value"].astype(float)
    events["score"] = events["score"].astype(float)
    return events


def detect_events(data, cfg: dict | None = None, *, dataset_id: str | None = None,
                  source: str | None = None, signals: list[str] | None = None) -> pd.DataFrame:
    """Events for one Dataset (or a featured frame). Returns ``empty_events()`` when there are none."""
    pcfg = pipeline_config(cfg)
    if isinstance(data, Dataset):
        df = data.frame
        dataset_id = dataset_id or data.id
        source = source or data.source
        if signals is None:
            signals = list(data.signals)
        time_column = data.time_column if data.is_timeseries else None
    else:
        df = data
        time_column = TIME_COLUMN if TIME_COLUMN in df.columns else None
        dataset_id = dataset_id or "dataset"
        source = source if source is not None else (dataset_id.split("/")[0] if "/" in dataset_id else "")
    if df is None or time_column is None or len(df) < 2:
        return empty_events()
    if signals is None:
        signals = select_signals(df, time_column)
    signals = [s for s in signals if s in df.columns]
    if not signals:
        return empty_events()

    ts = df[time_column]
    if ts.isna().any() or not ts.is_monotonic_increasing:
        df = df.dropna(subset=[time_column]).sort_values(time_column, kind="stable").reset_index(drop=True)
        ts = df[time_column]
    if len(df) < 2:
        return empty_events()
    interval, regular = sampling_regularity(ts)
    if not np.isfinite(interval) or interval <= 0 or regular < MIN_REGULAR_FRACTION:
        logger.info("%s: irregular sampling (%.0f%% regular); numeric event detection skipped", dataset_id, 100 * regular)
        return empty_events()
    if any(f"{sig}{FEATURE_SEP}z" not in df.columns for sig in signals):
        df = compute_features(df, signals, pcfg)

    stamps = pd.DatetimeIndex(ts)
    minutes = minutes_since_start(ts)
    records = dataset_gap_events(stamps, interval, pcfg, dataset_id, source)
    for sig in signals:
        x = df[sig].to_numpy(dtype=float)
        z = df[f"{sig}{FEATURE_SEP}z"].to_numpy(dtype=float)
        records += spike_events(x, z, stamps, sig, interval, pcfg, dataset_id, source)
        records += step_events(x, minutes, stamps, sig, interval, pcfg, dataset_id, source)
        records += flatline_events(x, stamps, sig, interval, pcfg, dataset_id, source)
        records += signal_gap_events(x, stamps, sig, interval, pcfg, dataset_id, source)
    events = _finish(records)
    logger.info("%s: %d event(s) across %d signal(s)", dataset_id, len(events), len(signals))
    return events
