"""Statistical helpers: HAC standard error, bootstrap CI, BH FDR, stability."""

import math

import numpy as np

from predicatelab.core.stats import (
    benjamini_hochberg,
    block_bootstrap_ci,
    mean_test,
    newey_west_se,
    paired_effect,
    pool_fold_series,
    stability,
)


def test_newey_west_se_is_finite_and_positive():
    rng = np.random.default_rng(0)
    x = rng.normal(0.2, 1.0, 200)
    se = newey_west_se(x, lag=5)
    assert math.isfinite(se) and se > 0
    # HAC SE should be in the same order of magnitude as the iid SE
    assert 0.2 * x.std(ddof=1) / math.sqrt(len(x)) < se < 5 * x.std(ddof=1) / math.sqrt(len(x))


def test_paired_effect_matches_mean_difference():
    cand = {f"d{i}": float(v) for i, v in enumerate(np.linspace(0.1, 0.3, 50))}
    base = {f"d{i}": float(v) for i, v in enumerate(np.linspace(0.0, 0.1, 50))}
    out = paired_effect(cand, base, nw_lag=2, n_boot=200, block=1, seed=7)
    assert out["n_dates"] == 50
    assert abs(out["effect"] - (np.mean(list(cand.values())) - np.mean(list(base.values())))) < 1e-12
    assert out["ci_low"] <= out["effect"] <= out["ci_high"]
    assert 0.0 <= out["p_value"] <= 1.0


def test_bootstrap_is_deterministic():
    x = list(np.linspace(-1, 1, 40))
    assert block_bootstrap_ci(x, n_boot=100, block=4, seed=3) == block_bootstrap_ci(
        x, n_boot=100, block=4, seed=3
    )


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert q == [0.04, 0.04, 0.04, 0.04]
    # NaN is preserved, and the valid values are corrected over their own count
    q2 = benjamini_hochberg([0.01, float("nan"), 0.5])
    assert math.isnan(q2[1])
    assert q2[0] <= q2[2] <= 1.0


def test_stability_counts_expected_sign():
    assert stability({"a": 0.1, "b": -0.2, "c": 0.3}, "higher") == 2 / 3
    assert stability({"a": -0.1, "b": -0.2}, "lower") == 1.0


def test_pool_fold_series_rejects_duplicate_dates():
    try:
        pool_fold_series({"f1": {"2020-01-01": 0.1}, "f2": {"2020-01-01": 0.2}})
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate dates must be rejected")
