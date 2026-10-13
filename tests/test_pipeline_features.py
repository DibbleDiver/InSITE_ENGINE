"""Features: column naming, rate of change on real time deltas, robust z behaviour."""
import numpy as np
import pandas as pd

from InSITE_Engine.contracts import FEATURE_SEP, FEATURE_SUFFIXES
from InSITE_Engine.features.features import compute_features, robust_z, robust_z_and_scale


def test_feature_columns_and_roc_use_real_time_delta():
    t0 = pd.Timestamp("2026-10-01 00:00", tz="UTC")
    df = pd.DataFrame({
        "timestamp": [t0, t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(minutes=15), t0 + pd.Timedelta(minutes=20)],
        "x": [0.0, 1.0, 4.0, np.nan],
        "y": [10.0, 10.0, 10.0, 10.0],
    })
    out = compute_features(df, ["x", "y"], {"pipeline": {"rolling_window": 2, "robust_window": 3}})
    for sig in ("x", "y"):
        for suffix in FEATURE_SUFFIXES:
            assert f"{sig}{FEATURE_SEP}{suffix}" in out.columns
    roc = out["x__roc"].tolist()
    assert np.isnan(roc[0])
    assert roc[1] == 0.2          # +1 over 5 minutes
    assert roc[2] == 0.3          # +3 over the real 10-minute delta
    assert np.isnan(roc[3])       # missing sample, no rate
    assert out["x__mean"].iloc[1] == 0.5  # centred 2-sample window on [0, 1]
    again = compute_features(out, ["x", "y"], {"pipeline": {"rolling_window": 2, "robust_window": 3}})
    assert list(again.columns) == list(out.columns)  # re-running does not duplicate feature columns


def test_robust_z_ignores_ramps_flags_spikes_and_guards_constant_signals():
    rng = np.random.default_rng(0)
    n = 600
    noise = rng.normal(0.0, 0.3, n)
    ramp = np.linspace(0.0, 60.0, n) + noise
    z_ramp, z_flat = robust_z(ramp, 48), robust_z(noise, 48)
    assert np.allclose(z_ramp, z_flat, atol=1e-6, equal_nan=True), "a linear ramp must leave the z-score untouched"
    assert np.nanmax(np.abs(z_flat)) < 4.5  # the Gaussian tail of 600 samples

    spiked = ramp.copy()
    spiked[300] += 6.0
    z = robust_z(spiked, 48)
    assert z[300] > 8.0
    assert np.nanmax(np.abs(z[276:300])) < 4.0 and np.nanmax(np.abs(z[301:325])) < 4.0  # no collateral spikes

    z, noise = robust_z_and_scale(np.full(n, 3.5), 48)
    assert np.isnan(noise)
    assert np.all(z[np.isfinite(z)] == 0.0)
    assert not np.isinf(z).any()


def test_robust_z_edges_of_a_step_are_not_spikes():
    rng = np.random.default_rng(1)
    n = 500
    x = 7.0 + rng.normal(0.0, 0.07, n)
    x[200:260] = 0.0  # a pump trip: a step, a flat stretch, a step back
    z = robust_z(x, 48)
    assert np.nanmax(np.abs(z)) < 4.0
