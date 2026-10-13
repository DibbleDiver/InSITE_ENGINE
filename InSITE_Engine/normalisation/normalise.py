"""Normalisation: find and parse the time axis, order the rows, pick the signals.

This is the lowest layer of the pipeline, so it is also home to the pipeline
configuration defaults (``PIPELINE_DEFAULTS`` / ``pipeline_config``) that every
later stage (features, events, states) reads.

Conventions established here and relied on downstream:

* The time column is always renamed to ``"timestamp"`` (``TIME_COLUMN``) and is
  tz-aware UTC (``datetime64[ns, UTC]``).  Naive timestamps are interpreted in
  ``assume_timezone``; timestamps that carry an offset are converted.
* Day-first vs month-first is decided from the data (a first field > 12 means
  day-first, a second field > 12 means month-first).  When every row is
  ambiguous both readings are tried and the one with fewer parse failures,
  then the shorter overall span, then the more monotonic order wins; a complete
  tie falls back to ``dayfirst_default``.  The decision is logged so a parse is
  never silently wrong.
* ``select_signals`` returns the numeric columns that are real signals: not the
  timestamp, not feature columns, not all-NaN columns and not id-like integer
  columns (work-order numbers, row counters).
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from ..contracts import FEATURE_SEP

logger = logging.getLogger(__name__)

TIME_COLUMN = "timestamp"

PIPELINE_DEFAULTS: dict = {
    "assume_timezone": "UTC",   # timezone for timestamps that carry none
    "rolling_window": 12,       # samples in the rolling mean/std features
    "robust_window": 48,        # samples in the rolling median/MAD robust z-score
    "spike_z": 4.0,             # |robust z| above this is a spike
    "step_window": 24,          # samples either side when testing for a step change
    "step_z": 3.0,              # mean shift (in pooled std units) that counts as a step change
    "flatline_min_samples": 12, # identical consecutive readings that count as a flatline
    "gap_factor": 3.0,          # interval > gap_factor x median interval is a gap
    "min_gap_minutes": 30,      # gaps shorter than this are ignored
    "dayfirst_default": True,   # tie-break when day/month order cannot be decided from the data
}


def pipeline_config(cfg: dict | None = None) -> dict:
    """Return the pipeline parameters with defaults filled in.

    Accepts either the whole config (``cfg["pipeline"]`` is used) or the
    ``pipeline`` section itself, or ``None`` for pure defaults.
    """
    merged = dict(PIPELINE_DEFAULTS)
    if not cfg:
        return merged
    section = cfg.get("pipeline") if isinstance(cfg.get("pipeline"), dict) else cfg
    for key, value in section.items():
        if value is not None:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------- #
# Time column detection and parsing
# --------------------------------------------------------------------------- #

# Exact (normalised) header names that mean "time axis", in priority order.
_TIME_NAMES = [
    "timestamp", "time_stamp", "datetime", "date_time", "time", "date",
    "created_at", "ts", "recorded_at", "logged_at", "measured_at", "sampled_at",
    "sample_time", "reading_time", "event_time", "created", "updated_at",
]
_TIME_SUFFIX_RE = re.compile(r"(_at|_time|_timestamp|_datetime|_date|_ts)$")
_DMY_RE = r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})"
_ISO_RE = r"^\d{4}-\d{2}-\d{2}"
_TZ_SUFFIX_RE = r"(?:Z|[+-]\d{2}:?\d{2})$"
MIN_PARSED_FRACTION = 0.9


def _is_text(series: pd.Series) -> bool:
    return is_string_dtype(series) or is_object_dtype(series)


def _to_datetime(text: pd.Series, utc: bool, **kwargs) -> pd.Series:
    """pd.to_datetime with the mixed-offset failure turned into a utc=True retry."""
    try:
        return pd.to_datetime(text, errors="coerce", utc=utc, **kwargs)
    except (ValueError, TypeError):
        if utc:
            raise
        return pd.to_datetime(text, errors="coerce", utc=True, **kwargs)


def _decide_dayfirst(text: pd.Series, default: bool) -> tuple[bool, str]:
    """Decide day-first vs month-first for slash/dash/dot dates from the data."""
    fields = text.str.extract(_DMY_RE)
    first = pd.to_numeric(fields[0], errors="coerce")
    second = pd.to_numeric(fields[1], errors="coerce")
    if first.notna().sum() == 0:
        return bool(default), "no day/month fields (format is unambiguous)"
    n_first_big = int((first > 12).sum())
    n_second_big = int((second > 12).sum())
    if n_first_big and not n_second_big:
        return True, f"first field > 12 in {n_first_big} rows"
    if n_second_big and not n_first_big:
        return False, f"second field > 12 in {n_second_big} rows"
    if n_first_big and n_second_big:
        logger.warning(
            "inconsistent day/month order: first field > 12 in %d rows, second field > 12 in %d rows",
            n_first_big, n_second_big,
        )
        return n_first_big >= n_second_big, "inconsistent fields; majority wins"

    # Every row is ambiguous (both fields <= 12): try both readings.
    def _quality(dayfirst: bool) -> tuple:
        parsed = _to_datetime(text, utc=False, dayfirst=dayfirst)
        valid = parsed.dropna()
        n_nat = int(parsed.isna().sum()) - int(text.isna().sum())
        if len(valid) < 2:
            return (n_nat, 0.0, 0.0)
        span = float((valid.max() - valid.min()).total_seconds())
        diffs = valid.to_numpy().astype("datetime64[ns]").astype("int64")
        monotonic = float(np.mean(np.diff(diffs) >= 0))
        return (n_nat, span, -monotonic)

    q_day, q_month = _quality(True), _quality(False)
    if q_day < q_month:
        return True, "ambiguous fields; day-first gives the more plausible series"
    if q_month < q_day:
        return False, "ambiguous fields; month-first gives the more plausible series"
    return bool(default), "ambiguous fields; using dayfirst_default"


def parse_timestamps(raw: pd.Series, cfg: dict | None = None) -> tuple[pd.Series | None, dict]:
    """Parse a column to tz-aware UTC timestamps.

    Returns ``(parsed, info)`` where ``parsed`` is ``None`` when the column does
    not look like a time axis at all.  ``info`` holds ``method``, ``dayfirst``,
    ``reason`` and ``parsed_fraction`` (fraction of non-null values parsed).
    """
    pcfg = pipeline_config(cfg)
    tz = pcfg["assume_timezone"]
    n_nonnull = int(raw.notna().sum())
    info: dict = {"method": None, "dayfirst": None, "reason": "", "parsed_fraction": 0.0, "n": n_nonnull}
    if n_nonnull == 0:
        return None, info

    if is_datetime64_any_dtype(raw):
        parsed = raw
        info["method"] = "datetime"
    elif is_numeric_dtype(raw) and not is_bool_dtype(raw):
        median = float(raw.dropna().median())
        if 1e9 <= median < 1e10:
            unit = "s"
        elif 1e12 <= median < 1e13:
            unit = "ms"
        else:
            return None, info
        parsed = pd.to_datetime(raw, unit=unit, errors="coerce", utc=True)
        info["method"] = f"epoch_{unit}"
    else:
        text = raw.astype("str").str.strip()
        text = text.mask(text == "")
        nonnull = text.dropna()
        if len(nonnull) == 0:
            return None, info
        looks_iso = float(nonnull.str.match(_ISO_RE).mean()) >= MIN_PARSED_FRACTION
        has_tz = bool(nonnull.str.contains(_TZ_SUFFIX_RE, regex=True).any())
        dayfirst, reason = _decide_dayfirst(nonnull, pcfg["dayfirst_default"])
        info["dayfirst"], info["reason"] = dayfirst, reason
        attempts: list[tuple[str, dict]] = []
        if looks_iso:
            attempts.append(("iso8601", {"format": "ISO8601"}))
        attempts.append(("inferred", {"dayfirst": dayfirst}))
        attempts.append(("mixed", {"dayfirst": dayfirst, "format": "mixed"}))
        parsed, best = None, -1.0
        for name, kwargs in attempts:
            try:
                candidate = _to_datetime(text, utc=has_tz, **kwargs)
            except (ValueError, TypeError):
                continue
            fraction = float(candidate.notna().sum()) / n_nonnull
            if fraction > best:
                parsed, best, info["method"] = candidate, fraction, name
            if fraction >= 0.99:
                break
        if parsed is None:
            return None, info

    if parsed.dt.tz is None:
        try:
            localised = parsed.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:  # unsorted or no repeated times: treat ambiguous wall times as standard time
            localised = parsed.dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
        parsed = localised.dt.tz_convert("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    parsed = parsed.astype("datetime64[ns, UTC]")
    info["parsed_fraction"] = float(parsed.notna().sum()) / n_nonnull
    return parsed, info


def find_time_column(df: pd.DataFrame, cfg: dict | None = None) -> tuple[str | None, pd.Series | None, dict]:
    """Locate the time axis: by name first, then any text column that parses >= 90%."""
    pcfg = pipeline_config(cfg)
    columns = list(df.columns)
    candidates = [c for c in _TIME_NAMES if c in columns]
    candidates += [c for c in columns if c not in candidates and _TIME_SUFFIX_RE.search(str(c))]
    candidates += [c for c in columns if c not in candidates and (_is_text(df[c]) or is_datetime64_any_dtype(df[c]))]
    for col in candidates:
        series = df[col]
        if series.notna().sum() == 0:
            continue
        if _is_text(series):
            # Cheap screening on a sample before parsing a whole (possibly free-text) column.
            sample = series.dropna()
            if len(sample) > 200:
                sample = sample.iloc[:: max(1, len(sample) // 200)]
            screened, screen_info = parse_timestamps(sample, pcfg)
            if screened is None or screen_info["parsed_fraction"] < MIN_PARSED_FRACTION:
                continue
        parsed, info = parse_timestamps(series, pcfg)
        if parsed is not None and info["parsed_fraction"] >= MIN_PARSED_FRACTION:
            return col, parsed, info
    return None, None, {}


# --------------------------------------------------------------------------- #
# Signal selection
# --------------------------------------------------------------------------- #

_ID_NAME_RE = re.compile(
    r"(^|_)(id|ids|uid|uuid|guid|key|ref|reference|no|num|number|ticket|serial|index|idx|row|"
    r"work_order|wo|order|order_no|order_number)$"
)


def looks_like_id(name: str, series: pd.Series) -> bool:
    """True for integer columns that are identifiers rather than measurements."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0 or not np.all(np.isfinite(values)) or not np.all(values == np.round(values)):
        return False
    if _ID_NAME_RE.search(str(name)):
        return True
    n = len(values)
    if n >= 20:
        diffs = np.diff(values)
        monotonic = np.all(diffs > 0) or np.all(diffs < 0)
        compact = (values.max() - values.min() + 1) <= 1.5 * n
        if monotonic and compact:  # a row counter / sequence number
            return True
    return False


def select_signals(df: pd.DataFrame, time_column: str | None = TIME_COLUMN) -> list[str]:
    """Numeric columns that are real signals (see module docstring)."""
    signals: list[str] = []
    for col in df.columns:
        if col == time_column or FEATURE_SEP in str(col):
            continue
        series = df[col]
        if not is_numeric_dtype(series) or is_bool_dtype(series):
            continue
        if series.notna().sum() == 0:
            continue
        if looks_like_id(col, series):
            continue
        signals.append(col)
    return signals


# --------------------------------------------------------------------------- #
# normalise
# --------------------------------------------------------------------------- #

def normalise(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Give the frame a UTC ``timestamp`` axis and a clean row order.

    * finds the time column, parses it (see module docstring) and renames it
      ``timestamp``; frames without a time axis are returned unchanged;
    * drops rows without a timestamp;
    * sorts by time (stable, so file order is kept among equal timestamps);
    * for telemetry frames (at least one signal) collapses duplicate timestamps,
      keeping the last occurrence (a re-sent reading wins).  Event-log frames
      such as work orders keep every row.
    """
    pcfg = pipeline_config(cfg)
    if df.empty or len(df.columns) == 0:
        return df.reset_index(drop=True)
    col, parsed, info = find_time_column(df, pcfg)
    if col is None:
        logger.info("no time column found (%d columns); treating as a non-time-series table", len(df.columns))
        return df.reset_index(drop=True)
    logger.info(
        "time column %r -> %r via %s (dayfirst=%s: %s; %.1f%% of values parsed)",
        col, TIME_COLUMN, info.get("method"), info.get("dayfirst"), info.get("reason"),
        100.0 * info.get("parsed_fraction", 0.0),
    )
    out = df.drop(columns=[col])
    if TIME_COLUMN in out.columns:  # another column already had the reserved name
        out = out.rename(columns={TIME_COLUMN: f"{TIME_COLUMN}_raw"})
    out.insert(0, TIME_COLUMN, parsed)  # same index as df[col], so this aligns exactly

    n_before = len(out)
    out = out[out[TIME_COLUMN].notna()]
    if len(out) < n_before:
        logger.info("dropped %d rows without a timestamp", n_before - len(out))
    out = out.sort_values(TIME_COLUMN, kind="stable")

    if select_signals(out, TIME_COLUMN):
        duplicated = out[TIME_COLUMN].duplicated(keep="last")
        if duplicated.any():
            logger.info("collapsed %d duplicate timestamps (kept the last reading)", int(duplicated.sum()))
            out = out[~duplicated]
    return out.reset_index(drop=True)
