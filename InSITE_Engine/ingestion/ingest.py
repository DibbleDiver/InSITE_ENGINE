"""Ingestion: every ``*.csv`` under ``<input_path>/<SOURCE>/`` becomes a Dataset.

Dataset ids are ``"<SOURCE>/<file_stem>"``.  A missing folder or an empty
input path yields no datasets; a file that cannot be parsed is logged and
skipped.  Frames are returned raw (cleaning and normalisation happen later in
the pipeline), with the file read as leniently as possible: UTF-8 with BOM,
latin-1 fallback, and a separator sniff when the header is not comma-separated.
"""
from __future__ import annotations

import glob
import logging
import os

import pandas as pd

from ..contracts import SOURCES, Dataset

logger = logging.getLogger(__name__)

_ALT_SEPARATORS = (";", "\t", "|")


def read_csv_robust(path: str) -> pd.DataFrame | None:
    """Read a CSV leniently; return None (after logging) when it cannot be parsed."""
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        if df.shape[1] == 1 and any(sep in str(df.columns[0]) for sep in _ALT_SEPARATORS):
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        return df
    except pd.errors.EmptyDataError:
        logger.warning("%s: empty file, skipped", path)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        logger.error("%s: could not be parsed (%s: %s), skipped", path, type(exc).__name__, exc)
    return None


def load_folder(path: str, source: str | None = None) -> list[Dataset]:
    """Load every CSV in ``path`` as a raw Dataset. Missing folder -> []."""
    if not os.path.isdir(path):
        return []
    source = source or os.path.basename(os.path.normpath(path))
    datasets: list[Dataset] = []
    for file in sorted(glob.glob(os.path.join(path, "*.csv"))):
        df = read_csv_robust(file)
        if df is None:
            continue
        stem = os.path.splitext(os.path.basename(file))[0]
        datasets.append(Dataset(id=f"{source}/{stem}", source=source, name=stem, path=file, frame=df))
        logger.info("loaded %s/%s: %d rows x %d columns", source, stem, len(df), df.shape[1])
    return datasets


def run_ingestion(cfg: dict) -> dict[str, Dataset]:
    """Datasets keyed by id for every SOURCE folder under ``cfg["input_path"]``."""
    base = cfg.get("input_path") or ""
    datasets: dict[str, Dataset] = {}
    if not base or not os.path.isdir(base):
        logger.warning("input_path %r does not exist; no datasets ingested", base)
        return datasets
    for source in SOURCES:
        for ds in load_folder(os.path.join(base, source), source):
            datasets[ds.id] = ds
    if not datasets:
        logger.warning("no CSV files found under %s/{%s}", base, ",".join(SOURCES))
    return datasets
