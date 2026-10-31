"""Effect sizes, uncertainty and multiple-testing correction.

Daily cross-sectional IC values are serially correlated, so the standard error
of their mean uses a Newey-West (Bartlett) estimator.  Confidence intervals use
a moving-block bootstrap, which preserves short-range dependence better than an
iid bootstrap.  Within a round the p-values are corrected with
Benjamini-Hochberg.

All functions are deterministic given a seed.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def _clean(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


def newey_west_se(x: Iterable[float], lag: int = 5) -> float:
    """Standard error of the mean with Bartlett-kernel HAC."""
    arr = _clean(x)
    n = arr.size
    if n < 2:
        return float("nan")
    lag = max(0, min(int(lag), n - 1))
    e = arr - arr.mean()
    gamma0 = float(e @ e) / n
    var = gamma0
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1.0)
        gamma_l = float(e[l:] @ e[:-l]) / n
        var += 2.0 * weight * gamma_l
    if not math.isfinite(var) or var <= 0:
        var = gamma0
    return math.sqrt(var / n) if var > 0 else float("nan")


def normal_two_sided_p(t: float) -> float:
    if not math.isfinite(t):
        return float("nan")
    # erfc-based normal tail, no scipy dependency
    return math.erfc(abs(t) / math.sqrt(2.0))


def mean_test(x: Iterable[float], lag: int = 5) -> dict[str, float]:
    arr = _clean(x)
    n = arr.size
    if n < 2:
        return {"mean": float(arr.mean()) if n else float("nan"), "se": float("nan"),
                "t": float("nan"), "p": float("nan"), "n": float(n),
                "std": float("nan"), "cohen_dz": float("nan")}
    mean = float(arr.mean())
    se = newey_west_se(arr, lag)
    t = mean / se if se and math.isfinite(se) and se > 0 else float("nan")
    return {
        "mean": mean,
        "se": se,
        "t": t,
        "p": normal_two_sided_p(t),
        "n": float(n),
        "std": float(arr.std(ddof=1)),
        "cohen_dz": mean / float(arr.std(ddof=1)) if arr.std(ddof=1) > 0 else float("nan"),
    }


def block_bootstrap_ci(
    x: Iterable[float], n_boot: int = 1000, block: int = 5, seed: int = 2025, alpha: float = 0.05
) -> tuple[float, float]:
    arr = _clean(x)
    n = arr.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = arr[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_effect(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    nw_lag: int = 5,
    n_boot: int = 1000,
    block: int = 5,
    seed: int = 2025,
) -> dict[str, float]:
    """Distribution of (candidate - baseline) over dates present in both series."""
    common = sorted(set(candidate) & set(baseline), key=lambda d: str(d))
    diffs = [float(candidate[d]) - float(baseline[d]) for d in common]
    test = mean_test(diffs, lag=nw_lag)
    lo, hi = block_bootstrap_ci(diffs, n_boot=n_boot, block=block, seed=seed)
    return {
        "effect": test["mean"],
        "se": test["se"],
        "t": test["t"],
        "p_value": test["p"],
        "ci_low": lo,
        "ci_high": hi,
        "cohen_dz": test["cohen_dz"],
        "n_dates": float(len(common)),
    }


def benjamini_hochberg(pvalues: Iterable[float]) -> list[float]:
    """Return q-values in the input order. NaNs are preserved."""
    p = np.asarray(list(pvalues), dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q.tolist()
    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0.0, 1.0)
    out = np.empty(m)
    out[order] = qv
    q[valid] = out
    return q.tolist()


def pool_fold_series(fold_series: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Pool per-fold date→metric series into one series.

    Validation periods are disjoint by protocol, so duplicate dates indicate a
    protocol bug and raise rather than silently overwrite.
    """
    pooled: dict[str, float] = {}
    for fold_id, series in fold_series.items():
        for d, v in series.items():
            key = str(d)
            if key in pooled:
                raise ValueError(f"duplicate validation date {key} across folds (fold {fold_id})")
            pooled[key] = float(v)
    return dict(sorted(pooled.items()))


def stability(fold_effects: Mapping[str, float], direction: str = "higher") -> float:
    """Share of folds whose effect has the expected sign."""
    if not fold_effects:
        return float("nan")
    vals = [v for v in fold_effects.values() if math.isfinite(v)]
    if not vals:
        return float("nan")
    if direction == "higher":
        return float(np.mean([v > 0 for v in vals]))
    return float(np.mean([v < 0 for v in vals]))
