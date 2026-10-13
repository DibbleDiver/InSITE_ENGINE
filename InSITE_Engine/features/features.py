"""Features: per-signal rolling statistics the event detectors and the Mind read.

For every signal ``sig`` the frame gains (``FEATURE_SEP`` is ``"__"``):

* ``sig__mean`` / ``sig__std``: rolling mean / sample std over ``rolling_window``
  samples;
* ``sig__z``: robust z-score ``(x - local level) / (1.4826 * rolling MAD)``
  over ``robust_window`` samples.  The local level is the value, at the
  sample, of the least-squares line through the window (a detrended centre:
  on a demand ramp it sits where the plain rolling median would lag), and the
  MAD is the exact median absolute deviation of the window's residuals from
  that line (the line is refitted once without samples more than 3 MADs off
  it, so a spike inside the window cannot drag the line, and hence its
  neighbours' residuals, with it).  Detrending each window keeps the scale
  equal to the noise rather
  than noise plus trend, so a real dip during the overnight ramp still scores
  |z| >> spike_z, while a genuine step inside a window inflates the residual
  MAD and keeps its edge samples at |z| < 3 (plain median residuals would
  score hundreds there).
* ``sig__roc``: rate of change per minute, ``(x_t - x_prev) / minutes``, using
  the real time delta between the two most recent non-missing samples.

Window convention: every rolling feature is CENTRED on the sample (the window
extends ``window // 2`` samples back and the rest forward, truncated at the
edges).  Centred windows judge a sample against its neighbours on both sides,
so smooth trends such as daily demand ramps do not inflate ``|z|``; the trade
off is that the last ``window // 2`` samples of a feed use a partial window.

MAD = 0 guard and calibration: the scale of every window is floored at the
signal's global noise scale (1.4826 * median |residual| over the whole
signal, which is correct for autocorrelated noise too).  A window's own MAD can
therefore only raise the scale (around steps, in noisy periods), never drop
it below the typical noise, so with Gaussian noise |z| > 4 is a genuine
6e-5-per-sample tail event.  If the global noise scale is zero (quantised or
mostly repeated values) the global spread is used instead, and a truly
constant signal gets z = 0 everywhere.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contracts import FEATURE_SEP, FEATURE_SUFFIXES
from ..normalisation.normalise import TIME_COLUMN, pipeline_config, select_signals

logger = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826
_CHUNK_ROWS = 20_000  # rows per (rows, window) working block in local_trend_stats
_CLIP_MADS = 3.0      # residuals beyond this many (MAD-based) sigmas are left out of the refit


# --------------------------------------------------------------------------- #
# Sampling helpers shared with events and states
# --------------------------------------------------------------------------- #

def interval_minutes(ts: pd.Series) -> np.ndarray:
    """Minutes between consecutive timestamps (length n - 1)."""
    values = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    return np.diff(values) / 60e9


def median_interval_minutes(ts: pd.Series) -> float:
    """Median positive spacing between samples, NaN when undefined."""
    if len(ts) < 2:
        return float("nan")
    diffs = interval_minutes(ts)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else float("nan")


def sampling_regularity(ts: pd.Series, tolerance: float = 0.25) -> tuple[float, float]:
    """(median interval in minutes, fraction of intervals within +-tolerance of it)."""
    median = median_interval_minutes(ts)
    if not np.isfinite(median):
        return median, 0.0
    diffs = interval_minutes(ts)
    regular = float(np.mean(np.abs(diffs - median) <= tolerance * median)) if len(diffs) else 0.0
    return median, regular


def minutes_since_start(ts: pd.Series) -> np.ndarray:
    values = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    return (values - values[0]) / 60e9 if len(values) else np.array([], dtype=float)


# --------------------------------------------------------------------------- #
# Rolling statistics
# --------------------------------------------------------------------------- #

def local_trend_stats(values: np.ndarray, window: int, min_periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per sample: level of the centred window's least-squares line at the sample,
    exact MAD of the window's residuals from that line, and the valid count.

    Level and MAD are NaN where the window holds fewer than ``min_periods``
    valid samples (or no time spread).  Rows are processed in chunks so the
    (n, window) working matrices stay small.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    window = max(3, int(window))
    half = window // 2
    padded = np.concatenate([np.full(half, np.nan), x, np.full(window - 1 - half, np.nan)])
    windows = sliding_window_view(padded, window)  # (n, window), the sample sits at column `half`
    pos = (np.arange(window) - half).astype(float)
    level = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    def fit(rows: np.ndarray, use: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cnt = use.sum(axis=1)
        xf = np.where(use, rows, 0.0)
        p = np.where(use, pos[None, :], 0.0)
        sx, st, stt, stx = xf.sum(axis=1), p.sum(axis=1), (p * p).sum(axis=1), (p * xf).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = cnt * stt - st * st
            slope = (cnt * stx - st * sx) / den
            intercept = (sx - slope * st) / cnt  # the line's value at pos 0, i.e. at the sample
        ok = (cnt >= max(3, min_periods)) & np.isfinite(den) & (den > 0)
        intercept[~ok] = np.nan
        slope[~ok] = np.nan
        return intercept, slope, cnt

    for start in range(0, n, _CHUNK_ROWS):
        rows = windows[start:start + _CHUNK_ROWS]
        valid = ~np.isnan(rows)
        intercept, slope, cnt = fit(rows, valid)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            resid = np.where(valid, rows - (intercept[:, None] + slope[:, None] * pos[None, :]), np.nan)
            mad_chunk = np.nanmedian(np.abs(resid), axis=1)
            # One clipped refit: a spike inside the window must not drag the line
            # (and so the residuals of its neighbours) with it.  Residuals from a
            # step straddled by the window stay within the clip, so edges are unaffected.
            clip = _CLIP_MADS * MAD_TO_SIGMA * mad_chunk
            inlier = valid & (np.abs(np.nan_to_num(resid, nan=0.0)) <= clip[:, None]) & np.isfinite(clip)[:, None]
            refit_rows = (inlier.sum(axis=1) < valid.sum(axis=1)) & (inlier.sum(axis=1) >= max(3, min_periods))
            if refit_rows.any():
                intercept2, slope2, _ = fit(rows[refit_rows], inlier[refit_rows])
                keep = np.isfinite(intercept2)
                idx = np.flatnonzero(refit_rows)[keep]
                intercept[idx], slope[idx] = intercept2[keep], slope2[keep]
                resid[idx] = np.where(valid[idx], rows[idx] - (intercept[idx, None] + slope[idx, None] * pos[None, :]), np.nan)
                mad_chunk[idx] = np.nanmedian(np.abs(resid[idx]), axis=1)
        stop = start + len(rows)
        level[start:stop], mad[start:stop], counts[start:stop] = intercept, mad_chunk, cnt
    return level, mad, counts


def _global_spread(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    if len(finite) < 2:
        return float("nan")
    spread = MAD_TO_SIGMA * float(np.median(np.abs(finite - np.median(finite))))
    return spread if spread > 0 else float("nan")


def robust_z_and_scale(values: np.ndarray, window: int, min_periods: int | None = None) -> tuple[np.ndarray, float]:
    """(robust z, global noise scale) as described in the module docstring."""
    x = np.asarray(values, dtype=float)
    window = max(3, int(window))
    if min_periods is None:
        min_periods = max(3, window // 4)
    level, mad, _ = local_trend_stats(x, window, min_periods)
    resid = x - level
    scale = MAD_TO_SIGMA * mad
    finite_resid = resid[np.isfinite(resid)]
    noise = MAD_TO_SIGMA * float(np.median(np.abs(finite_resid))) if len(finite_resid) else float("nan")
    if not (noise > 0):
        noise = _global_spread(x)
    if np.isfinite(noise):
        scale = np.where(np.isfinite(scale) & (scale < noise), noise, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = resid / scale
    z[np.isfinite(scale) & (scale == 0)] = 0.0  # constant window of a constant signal: nothing to flag
    z[~np.isfinite(z)] = np.nan
    return z, float(noise)


def robust_z(values: np.ndarray, window: int, min_periods: int | None = None) -> np.ndarray:
    """Robust z-score (see module docstring); NaN only where the window is too empty."""
    return robust_z_and_scale(values, window, min_periods)[0]


def noise_scale(values: np.ndarray, window: int, min_periods: int | None = None) -> float:
    """The signal's global noise scale used to floor the z-score scale (NaN if constant)."""
    return robust_z_and_scale(values, window, min_periods)[1]


def rate_of_change(values: np.ndarray, minutes: np.ndarray | None) -> np.ndarray:
    """Per-minute change between consecutive non-missing samples (NaN without a time axis)."""
    x = np.asarray(values, dtype=float)
    roc = np.full(len(x), np.nan)
    if minutes is None or len(x) < 2:
        return roc
    idx = np.flatnonzero(np.isfinite(x))
    if len(idx) < 2:
        return roc
    dx = np.diff(x[idx])
    dt = np.diff(np.asarray(minutes, dtype=float)[idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(dt > 0, dx / dt, np.nan)
    roc[idx[1:]] = rate
    return roc


# --------------------------------------------------------------------------- #
# compute_features
# --------------------------------------------------------------------------- #

def feature_columns(signals: list[str]) -> list[str]:
    return [f"{sig}{FEATURE_SEP}{suffix}" for sig in signals for suffix in FEATURE_SUFFIXES]


def compute_features(df: pd.DataFrame, signals: list[str] | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Return ``df`` with the feature columns described in the module docstring appended."""
    pcfg = pipeline_config(cfg)
    time_column = TIME_COLUMN if TIME_COLUMN in df.columns else None
    if signals is None:
        signals = select_signals(df, time_column)
    # Re-running on an already featured frame must not duplicate columns.
    stale = [c for c in df.columns if FEATURE_SEP in str(c) and str(c).rsplit(FEATURE_SEP, 1)[-1] in FEATURE_SUFFIXES]
    base = df.drop(columns=stale) if stale else df
    if not signals or base.empty:
        return base.copy()

    rolling_window = max(2, int(pcfg["rolling_window"]))
    robust_window = max(3, int(pcfg["robust_window"]))
    rolling_min = max(2, rolling_window // 2)
    robust_min = max(3, robust_window // 4)
    minutes = minutes_since_start(base[time_column]) if time_column else None

    features: dict[str, np.ndarray] = {}
    for sig in signals:
        x = base[sig].to_numpy(dtype=float)
        series = pd.Series(x)
        roll = series.rolling(rolling_window, center=True, min_periods=rolling_min)
        features[f"{sig}{FEATURE_SEP}mean"] = roll.mean().to_numpy()
        features[f"{sig}{FEATURE_SEP}std"] = roll.std().to_numpy()
        features[f"{sig}{FEATURE_SEP}z"], _ = robust_z_and_scale(x, robust_window, robust_min)
        features[f"{sig}{FEATURE_SEP}roc"] = rate_of_change(x, minutes)
    featured = pd.DataFrame(features, index=base.index)
    return pd.concat([base, featured], axis=1)
