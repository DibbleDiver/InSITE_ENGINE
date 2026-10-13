"""The deterministic half of the engine: raw CSV exports -> PipelineResult.

``run_pipeline(cfg)`` runs ingest -> clean -> normalise -> features -> events
-> states and returns the ``contracts.PipelineResult`` the Mind reads.  ``cfg``
is the loaded ``config.yaml`` (``input_path`` plus the ``pipeline`` section;
missing pipeline keys fall back to ``PIPELINE_DEFAULTS``).
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from .cleaning.clean import basic_clean
from .contracts import Dataset, PipelineResult, empty_events
from .events.events import detect_events
from .features.features import compute_features
from .ingestion.ingest import run_ingestion
from .normalisation.normalise import TIME_COLUMN, normalise, pipeline_config, select_signals
from .states.states import derive_states

logger = logging.getLogger(__name__)


def process_dataset(ds: Dataset, pcfg: dict) -> pd.DataFrame:
    """Clean, normalise and feature one raw Dataset in place; return its events."""
    frame = basic_clean(ds.frame, pcfg)
    frame = normalise(frame, pcfg)
    ds.time_column = TIME_COLUMN if TIME_COLUMN in frame.columns else None
    ds.signals = select_signals(frame, ds.time_column)
    if ds.time_column is not None and ds.signals:
        frame = compute_features(frame, ds.signals, pcfg)
    ds.frame = frame
    logger.info(
        "%s: %d rows, time axis=%s, signals=%s, text columns=%s",
        ds.id, len(frame), ds.time_column is not None, ds.signals, ds.text_columns(),
    )
    return detect_events(ds, pcfg)


def run_pipeline(cfg: dict) -> PipelineResult:
    """Run the whole deterministic pipeline over ``cfg["input_path"]``."""
    started = time.perf_counter()
    pcfg = pipeline_config(cfg)
    datasets = run_ingestion(cfg)
    per_dataset = []
    for ds in datasets.values():
        try:
            per_dataset.append(process_dataset(ds, pcfg))
        except Exception:  # one broken file must not take the whole run down
            logger.exception("%s: failed while processing; dataset kept raw, no events", ds.id)
    non_empty = [ev for ev in per_dataset if ev is not None and not ev.empty]
    if non_empty:
        events = pd.concat(non_empty, ignore_index=True)
        events = events.sort_values(["start", "dataset", "signal", "kind"], kind="stable").reset_index(drop=True)
    else:
        events = empty_events()
    states = derive_states(events, datasets, pcfg)
    result = PipelineResult(datasets=datasets, events=events, states=states)
    logger.info(
        "pipeline finished in %.2fs: %d dataset(s), %d event(s), %d state segment(s)",
        time.perf_counter() - started, len(datasets), len(events), len(states),
    )
    return result
