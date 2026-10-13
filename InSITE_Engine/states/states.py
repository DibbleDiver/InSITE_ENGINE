"""States: what every signal was doing, as a timeline that tiles the dataset's span.

For each time-series dataset and each of its signals ``derive_states`` emits
contiguous, non-overlapping ``[start, end)`` segments (``STATE_COLUMNS``) that
together cover exactly the dataset's time span (first to last timestamp); the
``end`` of one segment is the ``start`` of the next and the last ``end`` is the
dataset's last timestamp.

Mapping and precedence: ``gap`` -> ``offline``, ``flatline`` -> ``flatlined``,
spikes and step changes -> ``anomalous``, everything else ``normal``.  Where
event windows overlap the most severe state wins: offline > flatlined >
anomalous > normal.  A dataset-level gap (``signal == ALL_SIGNALS``) makes
every signal of that dataset offline for its window.

Windows: an event covers samples ``start``..``end`` inclusive, and a sample
"lasts" until the next one, so its state window is
``[start, end + median_interval)`` (a point event such as a step change thus
occupies one sampling interval).  Windows are clipped to the dataset span.

``derive_states(events)`` with no datasets still works in a degraded form: the
span and the signals of each dataset are taken from its events and point
events contribute nothing (no interval is known).  ``main.py`` historically
called it that way; the pipeline always passes the datasets.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..contracts import ALL_SIGNALS, STATE_COLUMNS, Dataset, empty_states
from ..features.features import median_interval_minutes
from ..normalisation.normalise import TIME_COLUMN, pipeline_config, select_signals

logger = logging.getLogger(__name__)

PRECEDENCE = {"normal": 0, "anomalous": 1, "flatlined": 2, "offline": 3}
RANK_TO_STATE = {rank: state for state, rank in PRECEDENCE.items()}
KIND_TO_STATE = {
    "gap": "offline",
    "flatline": "flatlined",
    "spike_high": "anomalous",
    "spike_low": "anomalous",
    "step_change": "anomalous",
}
_NS_PER_MINUTE = 60_000_000_000


def _ns(ts) -> int:
    return int(pd.Timestamp(ts).value)


def _spec(dataset: str, source: str, signals: list[str], t_min, t_max, interval: float) -> dict:
    return {"dataset": dataset, "source": source, "signals": list(signals),
            "t_min": _ns(t_min), "t_max": _ns(t_max),
            "interval_ns": int(round(interval * _NS_PER_MINUTE)) if np.isfinite(interval) else 0}


def _spec_from_dataset(ds: Dataset) -> dict | None:
    if not ds.is_timeseries or not ds.signals:
        return None
    ts = ds.frame[ds.time_column].dropna()
    if len(ts) < 2:
        return None
    return _spec(ds.id, ds.source, ds.signals, ts.min(), ts.max(), median_interval_minutes(ts))


def _spec_from_frame(df: pd.DataFrame, events: pd.DataFrame) -> dict | None:
    if TIME_COLUMN not in df.columns:
        return None
    ts = df[TIME_COLUMN].dropna()
    signals = select_signals(df, TIME_COLUMN)
    if len(ts) < 2 or not signals:
        return None
    datasets = events["dataset"].unique().tolist() if not events.empty else []
    dataset = datasets[0] if len(datasets) == 1 else "dataset"
    sources = events["source"].unique().tolist() if not events.empty else []
    source = sources[0] if len(sources) == 1 else (dataset.split("/")[0] if "/" in dataset else "")
    return _spec(dataset, source, signals, ts.min(), ts.max(), median_interval_minutes(ts))


def _specs_from_events(events: pd.DataFrame) -> list[dict]:
    specs = []
    for dataset, group in events.groupby("dataset", sort=True):
        signals = sorted(s for s in group["signal"].unique().tolist() if s != ALL_SIGNALS)
        if not signals:
            continue
        source = group["source"].iloc[0]
        specs.append(_spec(dataset, source, signals, group["start"].min(), group["end"].max(), float("nan")))
    return specs


def _resolve_specs(events: pd.DataFrame, datasets_or_frame) -> list[dict]:
    if datasets_or_frame is None:
        return _specs_from_events(events)
    if isinstance(datasets_or_frame, Dataset):
        spec = _spec_from_dataset(datasets_or_frame)
        return [spec] if spec else []
    if isinstance(datasets_or_frame, pd.DataFrame):
        spec = _spec_from_frame(datasets_or_frame, events)
        return [spec] if spec else []
    items = datasets_or_frame.values() if isinstance(datasets_or_frame, dict) else datasets_or_frame
    specs = []
    for item in items:
        spec = _spec_from_dataset(item) if isinstance(item, Dataset) else None
        if spec:
            specs.append(spec)
    return specs


def _signal_segments(spec: dict, ev: pd.DataFrame, sig: str) -> list[dict]:
    t_min, t_max, interval_ns = spec["t_min"], spec["t_max"], spec["interval_ns"]
    if t_max <= t_min:
        return []
    rows = ev[(ev["signal"] == sig) | (ev["signal"] == ALL_SIGNALS)] if not ev.empty else ev
    windows: list[tuple[int, int, int]] = []
    for row in rows.itertuples(index=False):
        state = KIND_TO_STATE.get(row.kind)
        if state is None or pd.isna(row.start) or pd.isna(row.end):
            continue
        s = max(_ns(row.start), t_min)
        e = min(_ns(row.end) + interval_ns, t_max)
        if e > s:
            windows.append((s, e, PRECEDENCE[state]))
    if not windows:
        return [{"state": "normal", "start": t_min, "end": t_max}]
    points = np.unique(np.array([t_min, t_max] + [w[0] for w in windows] + [w[1] for w in windows], dtype=np.int64))
    rank = np.zeros(len(points) - 1, dtype=np.int8)
    for s, e, r in windows:
        i0 = int(np.searchsorted(points, s, side="left"))
        i1 = int(np.searchsorted(points, e, side="left"))
        rank[i0:i1] = np.maximum(rank[i0:i1], r)
    segments = []
    for s, e in _rank_runs(rank):
        segments.append({"state": RANK_TO_STATE[int(rank[s])], "start": int(points[s]), "end": int(points[e + 1])})
    return segments


def _rank_runs(rank: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of each run of equal ranks."""
    if len(rank) == 0:
        return []
    change = np.flatnonzero(np.diff(rank) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(rank) - 1]))
    return list(zip(starts.tolist(), ends.tolist()))


def derive_states(events: pd.DataFrame, datasets_or_frame=None, cfg: dict | None = None) -> pd.DataFrame:
    """Per-signal state timeline (see module docstring). Returns ``empty_states()`` if nothing applies."""
    pipeline_config(cfg)  # accepted for symmetry with the other stages
    if events is None or len(getattr(events, "columns", [])) == 0:
        events = pd.DataFrame(columns=["dataset", "source", "signal", "kind", "start", "end"])
    specs = _resolve_specs(events, datasets_or_frame)
    if not specs:
        return empty_states()
    records = []
    for spec in specs:
        ev = events[events["dataset"] == spec["dataset"]] if not events.empty else events
        for sig in spec["signals"]:
            for seg in _signal_segments(spec, ev, sig):
                records.append({
                    "dataset": spec["dataset"],
                    "source": spec["source"],
                    "signal": sig,
                    "state": seg["state"],
                    "start": pd.Timestamp(seg["start"], unit="ns", tz="UTC"),
                    "end": pd.Timestamp(seg["end"], unit="ns", tz="UTC"),
                    "duration_minutes": (seg["end"] - seg["start"]) / _NS_PER_MINUTE,
                })
    if not records:
        return empty_states()
    states = pd.DataFrame(records, columns=STATE_COLUMNS)
    states["start"] = pd.to_datetime(states["start"], utc=True).astype("datetime64[ns, UTC]")
    states["end"] = pd.to_datetime(states["end"], utc=True).astype("datetime64[ns, UTC]")
    states["duration_minutes"] = states["duration_minutes"].astype(float)
    states = states.sort_values(["dataset", "signal", "start"], kind="stable").reset_index(drop=True)
    logger.info("derived %d state segment(s) for %d dataset(s)", len(states), len(specs))
    return states
