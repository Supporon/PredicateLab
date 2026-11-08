"""Trusted compiler for TSIL predicate descriptors.

**Security / reproducibility model**

A candidate is JSON data.  It is *never* evaluated with ``eval``/``exec``.
This module maps a closed set of ``family`` names and validated integer/float
parameters to pure PyTorch functions that live in this file.  The only external
call is to the upstream TSIL predicate implementations, addressed by a
whitelisted key (``family="native"``).

**Mathematical compatibility**

The output of every predicate is ``(I, P)`` with

    phi_hat = phi / ||phi||,   P = phi_hat @ phi_hat.T,   I = eye(B)

so the injected function plugs into ``WeightedMSELoss`` exactly like the
upstream predicates and preserves TSIL's loss ``tau_hat * I + tau * P``.
Native predicates are delegated verbatim and are not re-normalised.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

# The 23 predicates shipped with TSIL (src/custom_phi_constructors.py) plus the
# timestamp predicate from src/temporal_predicates.py.
NATIVE_PREDICATES: dict[str, dict[str, Any]] = {
    "phi_timestamp": {"params": {}, "requires_timestamp": True},
    "phi_cycle": {"params": {}},
    "phi_phase": {"params": {}},
    "phi_momentum": {"params": {"lookback_windows": "int_list"}},
    "phi_volatility": {"params": {"window_sizes": "int_list"}},
    "phi_return_dist": {"params": {}},
    "phi_trend": {"params": {}},
    "phi_multiscale": {"params": {"scales": "int_list"}},
    "phi_temporal": {"params": {}},
    "phi_feature_corr": {"params": {}},
    "phi_price_volume": {"params": {}},
    "phi_factor": {"params": {}},
    "phi_technical": {"params": {"short_window": "int", "long_window": "int"}},
    "phi_vol_return": {"params": {}},
    "phi_return_sim": {"params": {}},
    "phi_market_beta": {"params": {}},
    "phi_cross_corr": {"params": {}},
    "phi_lead_lag": {"params": {}},
    "phi_common_factor": {"params": {}},
    "phi_vol_sync": {"params": {}},
    "phi_tail_risk": {"params": {}},
    "phi_market_regime": {"params": {}},
    "phi_sector_cluster": {"params": {}},
}

# families implemented in this file (trusted torch code)
OP_PARAMS = {
    "momentum": {"window": "int"},
    "volatility": {"window": "int"},
    "dispersion": {"window": "int"},
    "autocorr": {"window": "int"},
    "drawdown": {"window": "int"},
    "tail": {"window": "int"},
    "trend": {"window": "int"},
}

ALLOWED_KEYS = {"family", "params", "feature_slice", "predicate", "parts", "weights", "combine", "note"}


class DescriptorError(ValueError):
    pass


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _require_int(value: Any, name: str, lo: int, hi: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DescriptorError(f"{name} must be an integer, got {value!r}")
    if not (lo <= value <= hi):
        raise DescriptorError(f"{name}={value} outside [{lo}, {hi}]")


def validate_descriptor(
    descriptor: dict[str, Any],
    *,
    seq_len: int = 20,
    feature_dim: int | None = None,
    max_depth: int = 2,
    _depth: int = 0,
) -> None:
    """Strict static validation; raises DescriptorError on anything unusual."""
    if not isinstance(descriptor, dict):
        raise DescriptorError("descriptor must be a JSON object")
    unknown = set(descriptor) - ALLOWED_KEYS
    if unknown:
        raise DescriptorError(f"unknown descriptor keys: {sorted(unknown)}")
    family = descriptor.get("family")
    if not isinstance(family, str):
        raise DescriptorError("descriptor.family is required")
    params = descriptor.get("params", {})
    if not isinstance(params, dict):
        raise DescriptorError("params must be an object")

    fs = descriptor.get("feature_slice")
    if fs is not None:
        if (not isinstance(fs, (list, tuple)) or len(fs) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in fs)):
            raise DescriptorError("feature_slice must be [start, end] integers")
        if fs[0] < 0 or fs[1] <= fs[0]:
            raise DescriptorError(f"invalid feature_slice {fs}")
        if feature_dim is not None and fs[1] > feature_dim:
            raise DescriptorError(f"feature_slice end {fs[1]} exceeds feature_dim {feature_dim}")

    if family == "native":
        pred = descriptor.get("predicate")
        if pred not in NATIVE_PREDICATES:
            raise DescriptorError(f"unknown native predicate {pred!r}")
        schema = NATIVE_PREDICATES[pred]["params"]
        unknown_p = set(params) - set(schema)
        if unknown_p:
            raise DescriptorError(f"unknown params for {pred}: {sorted(unknown_p)}")
        for name, kind in schema.items():
            if name not in params:
                continue
            value = params[name]
            if kind == "int":
                _require_int(value, name, 2, max(2, seq_len))
            elif kind == "int_list":
                if not isinstance(value, list) or not value:
                    raise DescriptorError(f"{name} must be a non-empty list of integers")
                for v in value:
                    _require_int(v, name, 2, 250)
        if pred == "phi_technical" and "short_window" in params and "long_window" in params:
            if params["short_window"] >= params["long_window"]:
                raise DescriptorError("phi_technical requires short_window < long_window")
        return

    if family == "composite":
        if _depth >= max_depth:
            raise DescriptorError("composite nesting too deep")
        parts = descriptor.get("parts")
        if not isinstance(parts, list) or not (1 <= len(parts) <= 4):
            raise DescriptorError("composite.parts must contain 1..4 descriptors")
        for part in parts:
            validate_descriptor(part, seq_len=seq_len, feature_dim=feature_dim,
                                max_depth=max_depth, _depth=_depth + 1)
        weights = descriptor.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(parts):
                raise DescriptorError("composite.weights must match parts length")
            if not all(isinstance(w, (int, float)) and math.isfinite(w) and 0 < w <= 10 for w in weights):
                raise DescriptorError("composite weights must be finite and in (0, 10]")
        combine = descriptor.get("combine", "concat")
        if combine not in {"concat", "mix"}:
            raise DescriptorError("combine must be 'concat' or 'mix'")
        if combine == "concat" and any(p.get("family") == "native" for p in parts):
            raise DescriptorError("concat composites support DSL families only; use combine='mix' for native parts")
        return

    if family not in OP_PARAMS:
        raise DescriptorError(f"unknown family {family!r}")
    schema = OP_PARAMS[family]
    unknown_p = set(params) - set(schema)
    if unknown_p:
        raise DescriptorError(f"unknown params for {family}: {sorted(unknown_p)}")
    if "window" not in params:
        raise DescriptorError(f"{family} requires params.window")
    _require_int(params["window"], "window", 3, max(3, seq_len))


# ---------------------------------------------------------------------------
# numeric operators (pure torch)
# ---------------------------------------------------------------------------
def _sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _slice_features(x: torch.Tensor, descriptor: dict[str, Any]) -> torch.Tensor:
    fs = descriptor.get("feature_slice")
    if fs:
        x = x[:, :, int(fs[0]): int(fs[1])]
    if x.shape[-1] < 2:
        raise DescriptorError("predicate needs at least 2 features")
    return x


def _window(descriptor: dict[str, Any]) -> int:
    return int(descriptor.get("params", {}).get("window", 10))


def _last_window(x: torch.Tensor, k: int) -> torch.Tensor:
    return x[:, -min(k, x.shape[1]):, :]


def _slope(series: torch.Tensor) -> torch.Tensor:
    """Least-squares slope of each row over time, shape [B]."""
    b, t = series.shape
    tt = torch.linspace(0.0, 1.0, t, device=series.device, dtype=series.dtype)
    tt = tt - tt.mean()
    denom = (tt * tt).sum().clamp_min(1e-8)
    return (series * tt).sum(dim=1) / denom


def _standardise(phi: torch.Tensor) -> torch.Tensor:
    mean = phi.mean(dim=0, keepdim=True)
    std = phi.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (phi - mean) / std


def _normalise_phi(phi: torch.Tensor) -> torch.Tensor:
    phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    phi = _standardise(phi)
    norm = torch.linalg.norm(phi)
    if not torch.isfinite(norm) or norm < 1e-8:
        # Degenerate descriptor: return zeros (P = 0), never a constant vector.
        # A constant fallback is exactly the Phi_ones control and silently gave
        # a broken descriptor a non-trivial P; zero is deterministic and the
        # probe rejects it as degenerate before any training run.
        return torch.zeros_like(phi)
    return phi / norm


def raw_phi(descriptor: dict[str, Any], batch_x: torch.Tensor) -> torch.Tensor:
    """Un-normalised per-sample descriptor for DSL families (shape [B, d])."""
    family = descriptor["family"]
    x = _sanitize(batch_x)
    x = _slice_features(x, descriptor)
    k = _window(descriptor)
    xw = _last_window(x, k)
    m = xw.mean(dim=2)  # cross-feature mean series [B, k]
    b = x.shape[0]

    if family == "momentum":
        recent = xw.mean(dim=1)  # [B, F]
        if x.shape[1] > k:
            prev = x[:, :-k, :].mean(dim=1)
        else:
            prev = torch.zeros_like(recent)
        delta = recent - prev
        return torch.stack([delta.mean(dim=1), delta.std(dim=1), _slope(m)], dim=1)

    if family == "volatility":
        vol_per_feature = xw.std(dim=1)  # [B, F]
        return torch.stack(
            [vol_per_feature.mean(dim=1), vol_per_feature.amax(dim=1), vol_per_feature.std(dim=1)],
            dim=1,
        )

    if family == "dispersion":
        disp = xw.std(dim=2)  # [B, k]
        return torch.stack([disp.mean(dim=1), disp.std(dim=1), disp[:, -1]], dim=1)

    if family == "autocorr":
        if m.shape[1] < 3:
            return torch.zeros((b, 2), device=x.device)
        def ac(lag: int) -> torch.Tensor:
            a, c = m[:, lag:], m[:, :-lag]
            a = a - a.mean(dim=1, keepdim=True)
            c = c - c.mean(dim=1, keepdim=True)
            denom = (a.norm(dim=1) * c.norm(dim=1)).clamp_min(1e-8)
            return (a * c).sum(dim=1) / denom
        return torch.stack([ac(1), ac(2) if m.shape[1] > 2 else ac(1)], dim=1)

    if family == "drawdown":
        cum = torch.cumsum(m, dim=1)
        running_max = torch.cummax(cum, dim=1).values
        dd = running_max - cum
        scale = m.std(dim=1, keepdim=True).clamp_min(1e-6)
        dd = dd / scale
        return torch.stack([dd.amax(dim=1), dd[:, -1]], dim=1)

    if family == "tail":
        q = torch.quantile(m, torch.tensor([0.1, 0.9], device=m.device), dim=1)
        q10, q90 = q[0], q[1]
        ratio = q90 / q10.abs().clamp_min(1e-6)
        mean = m.mean(dim=1, keepdim=True)
        std = m.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = (m - mean) / std
        skew = (z ** 3).mean(dim=1)
        kurt = (z ** 4).mean(dim=1) - 3.0
        return torch.stack([ratio, skew, kurt], dim=1)

    if family == "trend":
        slope = _slope(m)
        tt = torch.linspace(0.0, 1.0, m.shape[1], device=m.device)
        fitted = m.mean(dim=1, keepdim=True) + slope[:, None] * (tt - tt.mean())[None, :]
        ss_res = ((m - fitted) ** 2).sum(dim=1)
        ss_tot = ((m - m.mean(dim=1, keepdim=True)) ** 2).sum(dim=1).clamp_min(1e-8)
        return torch.stack([slope, 1.0 - ss_res / ss_tot], dim=1)

    raise DescriptorError(f"unknown family {family!r}")


def _dsl_I_P(descriptor: dict[str, Any], batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    phi = _normalise_phi(raw_phi(descriptor, batch_x))
    p = phi @ phi.T
    i = torch.eye(phi.shape[0], device=phi.device, dtype=phi.dtype)
    return i, p


# ---------------------------------------------------------------------------
# native delegation
# ---------------------------------------------------------------------------
def _native_I_P(
    descriptor: dict[str, Any],
    batch_x: torch.Tensor,
    timestamp_x: torch.Tensor | None,
    tsil_root: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import sys
    from pathlib import Path

    if tsil_root and str(tsil_root) not in sys.path:
        sys.path.insert(0, str(Path(tsil_root)))
    pred = descriptor["predicate"]
    params = dict(descriptor.get("params", {}))
    device = batch_x.device
    if pred.startswith("phi_timestamp"):
        from src.temporal_predicates import get_temporal_I_P

        if timestamp_x is None:
            raise DescriptorError("phi_timestamp requires calendar features from the dataset")
        return get_temporal_I_P(pred, batch_x, timestamp_x=timestamp_x, device=device)
    if pred in {"phi_cycle", "phi_phase", "phi_momentum", "phi_volatility", "phi_return_dist"} and not params:
        # canonical paper implementation, no parameters
        from src.temporal_predicates import get_temporal_I_P

        return get_temporal_I_P(pred, batch_x, timestamp_x=timestamp_x, device=device)
    from src.custom_phi_constructors import get_custom_I_P

    # The parameterised variants (momentum look-back windows, volatility
    # windows, technical windows) exist in custom_phi_constructors.
    return get_custom_I_P(pred, batch_x, device=device, **params)


def _composite_I_P(
    descriptor: dict[str, Any],
    batch_x: torch.Tensor,
    timestamp_x: torch.Tensor | None,
    tsil_root: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts = descriptor["parts"]
    weights = descriptor.get("weights") or [1.0] * len(parts)
    combine = descriptor.get("combine", "concat")
    if combine == "concat":
        phis = [
            _standardise(raw_phi(p, batch_x))
            for p in parts
        ]
        phi = torch.cat([w * p for w, p in zip(weights, phis)], dim=1)
        phi = _normalise_phi(phi)
        return torch.eye(phi.shape[0], device=phi.device), phi @ phi.T
    i = torch.eye(batch_x.shape[0], device=batch_x.device, dtype=batch_x.dtype)
    p_mix = torch.zeros_like(i)
    for part, w in zip(parts, weights):
        _, p = _dispatch_I_P(part, batch_x, timestamp_x, tsil_root)
        p_mix = p_mix + float(w) * p
    # Trace-normalize for the same P-scale contract as native/DSL predicates
    # (trace(P)=1, lambda_max(P)<=1).  Frobenius normalization made the P-term
    # effect incomparable across predicates in one BH family.
    p_mix = p_mix / (torch.diagonal(p_mix).sum() + 1e-8)
    return i, p_mix


def _dispatch_I_P(
    descriptor: dict[str, Any],
    batch_x: torch.Tensor,
    timestamp_x: torch.Tensor | None,
    tsil_root: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    family = descriptor["family"]
    if family == "native":
        return _native_I_P(descriptor, batch_x, timestamp_x, tsil_root)
    if family == "composite":
        return _composite_I_P(descriptor, batch_x, timestamp_x, tsil_root)
    return _dsl_I_P(descriptor, batch_x)


def compile_descriptor(
    descriptor: dict[str, Any],
    *,
    tsil_root: str | None = None,
    seq_len: int = 20,
    feature_dim: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]:
    """Validate and return ``phi_fn(batch_x, timestamp_x) -> (I, P)``."""
    validate_descriptor(descriptor, seq_len=seq_len, feature_dim=feature_dim)

    def phi_fn(batch_x: torch.Tensor, timestamp_x: torch.Tensor | None = None):
        if batch_x.dim() != 3:
            raise DescriptorError(f"batch_x must be [B, T, F], got {tuple(batch_x.shape)}")
        with torch.no_grad():
            i, p = _dispatch_I_P(descriptor, batch_x, timestamp_x, tsil_root)
        if p.shape[0] != batch_x.shape[0] or p.shape[1] != batch_x.shape[0]:
            raise DescriptorError("predicate returned a P matrix with the wrong shape")
        return i, p

    return phi_fn


def probe_signature(p_matrix: torch.Tensor) -> list[float]:
    """Flatten the strict upper triangle of P for behavioural dedup."""
    p = p_matrix.detach().float().cpu().numpy()
    n = p.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    return [float(v) for v in p[idx[0].numpy(), idx[1].numpy()]]


def needs_timestamp(descriptor: dict[str, Any]) -> bool:
    family = descriptor.get("family")
    if family == "native":
        return bool(NATIVE_PREDICATES.get(descriptor.get("predicate"), {}).get("requires_timestamp"))
    if family == "composite":
        return any(needs_timestamp(p) for p in descriptor.get("parts", []))
    return False


CROSS_SAMPLE_PREDICATES = {
    "phi_cycle",
    "phi_return_sim",
    "phi_market_beta",
    "phi_cross_corr",
    "phi_lead_lag",
    "phi_common_factor",
    "phi_vol_sync",
    "phi_market_regime",
    "phi_sector_cluster",
}


def uses_cross_sample_coupling(descriptor: dict[str, Any]) -> bool:
    """True when the descriptor aggregates across batch rows to form a pseudo-market reference."""
    family = descriptor.get("family")
    if family == "native":
        return descriptor.get("predicate") in CROSS_SAMPLE_PREDICATES
    if family == "composite":
        return any(uses_cross_sample_coupling(p) for p in descriptor.get("parts", []))
    return False

