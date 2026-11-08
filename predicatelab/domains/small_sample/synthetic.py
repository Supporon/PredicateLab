"""Synthetic small-sample data with an explicit ground truth.

The small-sample scenario is a *controlled* application of the platform: the
labels are generated from a known latent score, so every candidate that the
pipeline calls positive can later be labelled true or false without a real
study.  Four regimes are supported:

``null_noise``
    ``y`` is independent of every feature (pure label noise).
``null_linear``
    ``y`` depends linearly on raw features.  A linear baseline already captures
    the signal, so no predicate should add anything -- a "baseline-sufficient"
    global null.
``signal_ratio_threshold``
    ``y`` depends on a thresholded ratio ``x_a / (|x_b| + 0.1)``, a nonlinear
    relation that a linear baseline cannot capture.  This is the positive
    control; the true descriptor family is ``ratio_threshold(a, b, q)``.
``signal_sign_interaction``
    ``y`` depends on ``sign(z_a * z_b)``, a pure interaction.  The true family
    is ``interaction(a, b)``.

The ratio offset uses the *population* standard deviation (1.0 for standard
normal features), matching the fold-local ``0.1 * sd_b`` convention in the
adapter's compiler.  The population threshold is computed from a fixed
reference draw and is independent of the replicate sample, so no label
information leaks into the candidate generation.

Everything is a pure function of ``seed`` and the scenario parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

MODES = (
    "null_noise",
    "null_linear",
    "signal_ratio_threshold",
    "signal_sign_interaction",
)
SIGNAL_MODES = ("signal_ratio_threshold", "signal_sign_interaction")

# A fixed reference draw makes the population threshold independent of every
# replicate sample (and therefore of its labels).
_REFERENCE_SEED = 9001
_REFERENCE_DRAWS = 100_000
_RATIO_OFFSET = 0.1


@dataclass(frozen=True)
class SyntheticData:
    X: np.ndarray
    y: np.ndarray
    truth: dict
    truth_indicator: np.ndarray | None


@lru_cache(maxsize=None)
def population_ratio_threshold(quantile: float) -> float:
    """Population quantile of ``x0 / (|x2| + 0.1)`` under standard normals."""
    rng = np.random.default_rng(_REFERENCE_SEED)
    z = rng.standard_normal((_REFERENCE_DRAWS, 2))
    ratio = z[:, 0] / (np.abs(z[:, 1]) + _RATIO_OFFSET)
    return float(np.quantile(ratio, float(quantile)))


def _standardise(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)


def _latent_score(mode: str, X: np.ndarray, strength: float,
                  threshold_quantile: float) -> tuple[np.ndarray, np.ndarray | None, dict | None]:
    """Return (score, true indicator or None, true descriptor or None)."""
    n, p = X.shape
    if mode == "null_noise":
        return np.zeros(n, dtype=float), None, None
    if mode == "null_linear":
        z = _standardise(X)
        score = 0.9 * z[:, 0] + 0.6 * z[:, 1]
        return score, None, None
    if mode == "signal_ratio_threshold":
        a, b = 0, 2
        ratio = X[:, a] / (np.abs(X[:, b]) + _RATIO_OFFSET)
        threshold = population_ratio_threshold(threshold_quantile)
        indicator = (ratio >= threshold).astype(float)
        centred = indicator - float(indicator.mean())
        nuisance = 0.35 * _standardise(X)[:, 3]
        descriptor = {"op": "ratio_threshold", "a": a, "b": b,
                      "quantile": float(threshold_quantile)}
        # ``strength * centred`` keeps the label balanced; the exposed truth
        # indicator stays the raw predicate output so post-hoc recovery scores
        # (AUC/correlation) compare against the predicate, not its centring.
        return float(strength) * centred + nuisance, indicator, descriptor
    if mode == "signal_sign_interaction":
        a, b = 1, 3
        # Population standardisation is the identity here (standard normals),
        # so the signal does not depend on the replicate's own sample moments.
        # Fold-local standardisation is the model's job.
        indicator = np.sign(X[:, a] * X[:, b])
        nuisance = 0.30 * X[:, 5] if p > 5 else np.zeros(n)
        descriptor = {"op": "interaction", "a": a, "b": b}
        return float(strength) * 0.5 * indicator + nuisance, indicator, descriptor
    raise ValueError(f"unknown small-sample mode {mode!r}; expected one of {MODES}")


def _calibrate_intercept(score: np.ndarray, target: float) -> float:
    """Solve for the intercept that gives the requested marginal positive rate."""
    lo, hi = -25.0, 25.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = float(np.mean(1.0 / (1.0 + np.exp(-(score + mid)))))
        if p < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate(*, seed: int, n_samples: int, n_features: int, mode: str,
             signal_strength: float = 0.0, threshold_quantile: float = 0.5,
             class_balance: float = 0.5) -> SyntheticData:
    """Generate one replicate of the small-sample scenario."""
    if mode not in MODES:
        raise ValueError(f"unknown small-sample mode {mode!r}; expected one of {MODES}")
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if not (0.0 < float(class_balance) < 1.0):
        raise ValueError("class_balance must be in (0,1)")
    if mode == "signal_ratio_threshold" and n_features < 4:
        raise ValueError("signal_ratio_threshold needs n_features >= 4")
    if mode == "signal_sign_interaction" and n_features < 4:
        raise ValueError("signal_sign_interaction needs n_features >= 4")

    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal((int(n_samples), int(n_features)))
    score, indicator, descriptor = _latent_score(
        mode, X, float(signal_strength), float(threshold_quantile)
    )
    intercept = _calibrate_intercept(score, float(class_balance))
    prob = 1.0 / (1.0 + np.exp(-(score + intercept)))
    y = (rng.random(int(n_samples)) < prob).astype(int)

    truth = {
        "mode": mode,
        "is_signal": mode in SIGNAL_MODES,
        "signal_strength": float(signal_strength) if mode in SIGNAL_MODES else 0.0,
        "threshold_quantile": float(threshold_quantile),
        "true_descriptor": descriptor,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "class_balance_target": float(class_balance),
        "realised_positive_rate": float(y.mean()),
        "seed": int(seed),
        "ratio_offset": _RATIO_OFFSET,
    }
    return SyntheticData(X=np.asarray(X, dtype=float), y=np.asarray(y, dtype=int),
                         truth=truth,
                         truth_indicator=None if indicator is None else np.asarray(indicator, dtype=float))


def descriptor_family(descriptor: dict) -> str:
    """Stable, human-readable family key used to score recovery."""
    op = str(descriptor.get("op", "?"))
    if op in {"ratio", "difference", "product", "ratio_threshold"}:
        return f"{op}({descriptor.get('a')},{descriptor.get('b')})"
    if op == "interaction":
        return f"interaction({descriptor.get('a')},{descriptor.get('b')})"
    return f"{op}({descriptor.get('a')})"


def truth_family(truth: dict) -> str | None:
    descriptor = truth.get("true_descriptor")
    return descriptor_family(descriptor) if descriptor else None


def matches_true_family(descriptor: dict, truth: dict) -> bool:
    expected = truth.get("true_descriptor")
    if not expected:
        return False
    return descriptor_family(descriptor) == descriptor_family(expected)
