"""Cleaning: make a raw CSV export tidy without inventing data.

* headers are normalised to snake_case ("Tank Level A (%)" -> "tank_level_a_pct",
  "P2 Current (A)" -> "p2_current_a"), duplicates get a numeric suffix;
* unnamed, entirely empty columns (trailing commas in the export) are dropped;
* text cells are stripped and blank cells become NaN;
* text columns that are numeric-looking in >= 90% of their non-blank cells
  ("12.5", " 7 ", "1,250.0", "12.5%") are coerced to numbers, the remaining
  non-numeric markers ("ERR", "n/a") become NaN;
* exact duplicate rows are dropped.

Missing values stay missing: nothing here fills NaN with zeros, because a
fabricated zero looks exactly like a pump that stopped or a sensor that spiked.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

logger = logging.getLogger(__name__)

NUMERIC_TEXT_MIN_FRACTION = 0.9
_THOUSANDS_RE = r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?"


def normalise_header(name, position: int = 0) -> str:
    """snake_case a column header; symbols become words where they carry meaning."""
    text = str(name).strip()
    if text.lower().startswith("unnamed:"):
        text = f"unnamed_{position}"
    text = (
        text.replace("%", " pct ")
        .replace("°", " deg ")
        .replace("&", " and ")
        .replace("#", " num ")
    )
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or f"col_{position}"


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalise_header to every column, de-duplicating with _2, _3, ..."""
    seen: dict[str, int] = {}
    new_names = []
    for position, name in enumerate(df.columns):
        base = normalise_header(name, position)
        if base in seen:
            seen[base] += 1
            new = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
            new = base
        new_names.append(new)
    out = df.copy()
    out.columns = new_names
    return out


def _is_text(series: pd.Series) -> bool:
    return is_string_dtype(series) or is_object_dtype(series)


def coerce_numeric_text(df: pd.DataFrame, min_fraction: float = NUMERIC_TEXT_MIN_FRACTION) -> pd.DataFrame:
    """Convert text columns that are overwhelmingly numeric-looking to numbers."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if not _is_text(series):
            continue
        nonnull = series.dropna()
        if len(nonnull) == 0:
            continue
        text = nonnull.astype("str").str.strip()
        thousands = text.str.fullmatch(_THOUSANDS_RE)
        text = text.where(~thousands.fillna(False), text.str.replace(",", "", regex=False))
        percent = text.str.endswith("%")
        text = text.where(~percent.fillna(False), text.str.rstrip("%"))
        numeric = pd.to_numeric(text, errors="coerce")
        fraction = float(numeric.notna().mean())
        if fraction < min_fraction:
            continue
        converted = pd.Series(np.nan, index=out.index, dtype="float64")
        converted.loc[numeric.index] = numeric.to_numpy(dtype=float)
        n_bad = int(numeric.isna().sum())
        if n_bad:
            bad_values = sorted(set(text[numeric.isna()].tolist()))[:5]
            logger.info("%s: coerced to numeric; %d non-numeric cell(s) %s set to NaN", col, n_bad, bad_values)
        if np.all(np.isfinite(converted.dropna())) and (converted.dropna() == converted.dropna().round()).all() \
                and not converted.isna().any():
            converted = converted.astype("int64")
        out[col] = converted
    return out


def strip_text(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace in text cells and turn blank cells into NaN."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if not _is_text(series):
            continue
        stripped = series.astype("str").str.strip()
        out[col] = stripped.mask(stripped == "")
    return out


def drop_empty_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    junk = [c for c in df.columns if str(c).startswith("unnamed_") and df[c].isna().all()]
    if junk:
        logger.debug("dropping %d empty unnamed column(s): %s", len(junk), junk)
        return df.drop(columns=junk)
    return df


def basic_clean(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Tidy a raw export (see module docstring). ``cfg`` is accepted for symmetry."""
    out = normalise_headers(df)
    out = drop_empty_unnamed_columns(out)
    out = strip_text(out)
    out = coerce_numeric_text(out)
    n_before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) < n_before:
        logger.info("dropped %d exact duplicate row(s)", n_before - len(out))
    return out
