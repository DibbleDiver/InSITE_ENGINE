"""Shared data contracts between the deterministic pipeline and the Mind.

The pipeline (ingest -> clean -> normalise -> features -> events -> states)
produces a PipelineResult. The Mind only ever reads a PipelineResult, so this
module is the single interface between the two halves of the engine.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

SOURCES = ["TRIO", "TCL", "AX", "Meters"]

# Feature columns are derived per signal as f"{signal}{FEATURE_SEP}{suffix}".
FEATURE_SEP = "__"
FEATURE_SUFFIXES = ["mean", "std", "z", "roc"]

# Signal name used for dataset-level events (e.g. the whole feed went silent).
ALL_SIGNALS = "__all__"

EVENT_KINDS = ["spike_high", "spike_low", "step_change", "flatline", "gap"]
SEVERITIES = ["low", "medium", "high"]
EVENT_COLUMNS = [
    "event_id",     # deterministic hash: the same data always yields the same id
    "dataset",      # Dataset.id
    "source",       # Dataset.source
    "signal",       # signal column, or ALL_SIGNALS
    "kind",         # one of EVENT_KINDS
    "severity",     # one of SEVERITIES
    "start",        # tz-aware UTC timestamp
    "end",          # tz-aware UTC timestamp (== start for point events)
    "peak_value",   # most extreme raw value inside the event (NaN for gaps)
    "score",        # kind-specific magnitude, e.g. peak |robust z|
    "detail",       # short human-readable description
]

STATES = ["normal", "anomalous", "flatlined", "offline"]
STATE_COLUMNS = [
    "dataset", "source", "signal", "state", "start", "end", "duration_minutes",
]


@dataclass
class Dataset:
    id: str                    # "<SOURCE>/<file_stem>", e.g. "Meters/site_flow"
    source: str                # folder name, one of SOURCES
    name: str                  # file stem
    path: str                  # file the data was read from
    frame: pd.DataFrame        # cleaned, normalised, with feature columns added
    signals: list[str] = field(default_factory=list)  # numeric signal columns (no feature columns)
    time_column: str | None = None  # "timestamp" when a time axis was found, else None

    @property
    def is_timeseries(self) -> bool:
        return self.time_column is not None and self.time_column in self.frame.columns

    def time_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if not self.is_timeseries or self.frame.empty:
            return None, None
        ts = self.frame[self.time_column]
        return ts.min(), ts.max()

    def text_columns(self) -> list[str]:
        feature_like = {c for c in self.frame.columns if FEATURE_SEP in c}
        return [
            c for c in self.frame.columns
            if c not in self.signals and c != self.time_column and c not in feature_like
        ]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def empty_states() -> pd.DataFrame:
    return pd.DataFrame(columns=STATE_COLUMNS)


def _iso(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


@dataclass
class PipelineResult:
    datasets: dict[str, Dataset]
    events: pd.DataFrame
    states: pd.DataFrame
    generated_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))

    def summary(self) -> dict:
        """JSON-serialisable overview of everything the pipeline produced."""
        datasets = {}
        for ds_id, ds in sorted(self.datasets.items()):
            start, end = ds.time_range()
            datasets[ds_id] = {
                "source": ds.source,
                "rows": int(len(ds.frame)),
                "signals": list(ds.signals),
                "text_columns": ds.text_columns(),
                "start": _iso(start),
                "end": _iso(end),
            }
        events = self.events
        by_kind = events["kind"].value_counts().to_dict() if not events.empty else {}
        by_severity = events["severity"].value_counts().to_dict() if not events.empty else {}
        current = {}
        if not self.states.empty:
            latest = self.states.sort_values("end").groupby(["dataset", "signal"]).tail(1)
            for row in latest.itertuples(index=False):
                current[f"{row.dataset}:{row.signal}"] = row.state
        return {
            "generated_at": _iso(self.generated_at),
            "datasets": datasets,
            "event_count": int(len(events)),
            "events_by_kind": {k: int(v) for k, v in by_kind.items()},
            "events_by_severity": {k: int(v) for k, v in by_severity.items()},
            "current_states": current,
        }

    def save(self, output_path: str) -> list[str]:
        """Write events, states, featured datasets and the summary. Returns written paths."""
        os.makedirs(output_path, exist_ok=True)
        written = []
        events_path = os.path.join(output_path, "events.csv")
        self.events.to_csv(events_path, index=False)
        written.append(events_path)
        states_path = os.path.join(output_path, "states.csv")
        self.states.to_csv(states_path, index=False)
        written.append(states_path)
        features_dir = os.path.join(output_path, "features")
        os.makedirs(features_dir, exist_ok=True)
        for ds in self.datasets.values():
            path = os.path.join(features_dir, f"{ds.source}__{ds.name}.csv")
            ds.frame.to_csv(path, index=False)
            written.append(path)
        summary_path = os.path.join(output_path, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        written.append(summary_path)
        return written
