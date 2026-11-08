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
    "semivar_ratio": {"window": "int"},
    "vol_price_corr": {"window": "int"},
    "path_efficiency": {"window": "int"},
    "hurst_vr": {"window": "int"},
    "lead_lag_beta": {"window": "int"},
    "regime_transition": {"window": "int"},
    "tail_asymmetry": {"window": "int"},
}

SYMBOLIC_TIME_OPS = {
    "mean",
    "std",
    "slope",
    "accel",
    "ema_diff",
    "max_drawdown",
    "semivar_asymmetry",
    "skew",
    "kurt",
    "q_spread",
    "autocorr_1",
    "hurst_proxy",
    "path_efficiency",
}

SYMBOLIC_FEATURE_OPS = {
    "mean",
    "std",
    "max",
    "min",
    "iqr",
    "l2_norm",
    "top_bottom_spread",
    "corr_half",
}

SYMBOLIC_INTERACTIONS = {
    "none",
    "diff",
    "ratio",
    "product",
    "corr",
    "beta",
    "lead_lag_corr",
}

SYMBOLIC_TRANSFORMS = {
    "identity",
    "abs",
    "tanh",
    "sign_sqrt",
}

SYMBOLIC_CHANNEL_KEYS = {
    "time_op",
    "feature_op",
    "window",
    "feature_slice",
    "interaction",
    "rhs_slice",
    "rhs_time_op",
    "rhs_feature_op",
    "rhs_window",
    "transform",
    "weight",
    "note",
}

ALLOWED_KEYS = {
    "family",
    "params",
    "feature_slice",
    "predicate",
    "parts",
    "weights",
    "combine",
    "channels",
    "formula",
    "note",
}


class DescriptorError(ValueError):
    pass


def canonicalize_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a TSIL descriptor so semantically equivalent forms share one hash.

    - Normalizes composite ``parts`` ordering by canonical JSON representation.
    - Normalizes composite ``weights`` to sum to 1.0 (eliminating global scale factors
      so ``[1.0, 1.0]`` and ``[2.0, 2.0]`` map to ``[0.5, 0.5]``) and merges identical
      parts by summing their weights.
    - Normalizes ``weighted_p`` / ``concat`` aliases into canonical composite form.
    """
    from predicatelab.core.hashing import canonical_json

    if not isinstance(descriptor, dict):
        return descriptor
    out: dict[str, Any] = {}
    for k in sorted(descriptor.keys()):
        v = descriptor[k]
        if k == "params" and isinstance(v, dict):
            clean_params: dict[str, Any] = {}
            for pk in sorted(v.keys()):
                pv = v[pk]
                if isinstance(pv, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in pv):
                    clean_params[pk] = sorted(pv)
                else:
                    clean_params[pk] = pv
            out["params"] = clean_params
        elif k == "feature_slice" and v is not None and isinstance(v, (list, tuple)) and len(v) == 2:
            out["feature_slice"] = [int(v[0]), int(v[1])]
        else:
            out[k] = v

    family = out.get("family")
    if family in {"weighted_p", "concat"} and "parts" in out:
        out["combine"] = "mix" if family == "weighted_p" else "concat"
        out["family"] = "composite"
        family = "composite"

    if family == "composite" and isinstance(out.get("parts"), list):
        raw_parts = [canonicalize_descriptor(p) for p in out["parts"]]
        combine = out.get("combine", "concat")
        if combine == "weighted_p":
            combine = "mix"
        out["combine"] = combine
        raw_weights = out.get("weights")
        if (
            isinstance(raw_weights, list)
            and len(raw_weights) == len(raw_parts)
            and all(isinstance(w, (int, float)) and not isinstance(w, bool) and math.isfinite(float(w)) and float(w) > 0 for w in raw_weights)
        ):
            weights = [float(w) for w in raw_weights]
        else:
            weights = [1.0] * len(raw_parts)

        # Associative flattening for nested composites with the same combine mode
        flat_parts: list[dict[str, Any]] = []
        flat_weights: list[float] = []
        for part, w in zip(raw_parts, weights):
            if (
                isinstance(part, dict)
                and part.get("family") == "composite"
                and part.get("combine", "concat") == combine
                and isinstance(part.get("parts"), list)
            ):
                sub_parts = part["parts"]
                sub_weights = part.get("weights") or [1.0 / max(1, len(sub_parts))] * len(sub_parts)
                for sp, sw in zip(sub_parts, sub_weights):
                    flat_parts.append(sp)
                    flat_weights.append(float(w) * float(sw))
            else:
                flat_parts.append(part)
                flat_weights.append(float(w))

        # Sort parts deterministically by canonical JSON and merge identical parts
        paired = sorted(
            zip(flat_parts, flat_weights),
            key=lambda item: canonical_json(item[0]),
        )
        merged_parts: list[dict[str, Any]] = []
        merged_weights: list[float] = []
        for part, w in paired:
            if merged_parts and canonical_json(merged_parts[-1]) == canonical_json(part):
                merged_weights[-1] += w
            else:
                merged_parts.append(part)
                merged_weights.append(w)
        if len(merged_parts) == 1:
            return merged_parts[0]
        total_w = sum(merged_weights)
        if total_w > 0:
            norm_weights = [round(w / total_w, 8) for w in merged_weights]
        else:
            norm_weights = [round(1.0 / max(1, len(merged_parts)), 8)] * len(merged_parts)
        out["parts"] = merged_parts
        out["weights"] = norm_weights
    elif family == "symbolic" and isinstance(out.get("channels"), list):
        norm_channels: list[dict[str, Any]] = []
        for ch in out["channels"]:
            if not isinstance(ch, dict):
                continue
            c_clean: dict[str, Any] = {}
            for ck in sorted(ch.keys()):
                if ck == "note":
                    continue
                cv = ch[ck]
                if ck in {"feature_slice", "rhs_slice"} and cv is not None and isinstance(cv, (list, tuple)) and len(cv) == 2:
                    c_clean[ck] = [int(cv[0]), int(cv[1])]
                elif ck in {"window", "rhs_window"} and isinstance(cv, (int, float)) and not isinstance(cv, bool):
                    c_clean[ck] = int(cv)
                elif ck == "weight" and isinstance(cv, (int, float)) and not isinstance(cv, bool):
                    c_clean[ck] = round(float(cv), 6)
                else:
                    c_clean[ck] = cv
            c_clean.setdefault("time_op", "mean")
            c_clean.setdefault("feature_op", "mean")
            c_clean.setdefault("interaction", "none")
            c_clean.setdefault("transform", "identity")
            norm_channels.append(c_clean)
        # Deduplicate and sort channels deterministically by canonical JSON
        dedup_map: dict[str, dict[str, Any]] = {}
        for c_clean in norm_channels:
            key = canonical_json(c_clean)
            if key not in dedup_map:
                dedup_map[key] = c_clean
        out["channels"] = [dedup_map[k] for k in sorted(dedup_map.keys())]
    return out


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
        if combine not in {"concat", "mix", "weighted_p"}:
            raise DescriptorError("combine must be 'concat', 'mix', or 'weighted_p'")
        if combine == "concat" and any(p.get("family") == "native" for p in parts):
            raise DescriptorError("concat composites support DSL families only; use combine='mix' for native parts")
        return

    if family == "symbolic":
        channels = descriptor.get("channels")
        if not isinstance(channels, list) or not (2 <= len(channels) <= 8):
            raise DescriptorError("symbolic descriptor requires 'channels' list of length 2..8")
        if "window" in params:
            _require_int(params["window"], "params.window", 3, max(3, seq_len))
        for idx, ch in enumerate(channels):
            if not isinstance(ch, dict):
                raise DescriptorError(f"symbolic channel[{idx}] must be an object")
            unknown_ck = set(ch) - SYMBOLIC_CHANNEL_KEYS
            if unknown_ck:
                raise DescriptorError(f"symbolic channel[{idx}] has unknown keys: {sorted(unknown_ck)}")
            t_op = ch.get("time_op", "mean")
            if t_op not in SYMBOLIC_TIME_OPS:
                raise DescriptorError(f"symbolic channel[{idx}] unknown time_op {t_op!r}")
            f_op = ch.get("feature_op", "mean")
            if f_op not in SYMBOLIC_FEATURE_OPS:
                raise DescriptorError(f"symbolic channel[{idx}] unknown feature_op {f_op!r}")
            inter = ch.get("interaction", "none")
            if inter not in SYMBOLIC_INTERACTIONS:
                raise DescriptorError(f"symbolic channel[{idx}] unknown interaction {inter!r}")
            trans = ch.get("transform", "identity")
            if trans not in SYMBOLIC_TRANSFORMS:
                raise DescriptorError(f"symbolic channel[{idx}] unknown transform {trans!r}")
            if "window" in ch and ch["window"] is not None:
                _require_int(ch["window"], f"channel[{idx}].window", 3, max(3, seq_len))
            if "rhs_window" in ch and ch["rhs_window"] is not None:
                _require_int(ch["rhs_window"], f"channel[{idx}].rhs_window", 3, max(3, seq_len))
            if "rhs_time_op" in ch and ch["rhs_time_op"] is not None:
                if ch["rhs_time_op"] not in SYMBOLIC_TIME_OPS:
                    raise DescriptorError(f"symbolic channel[{idx}] unknown rhs_time_op {ch['rhs_time_op']!r}")
            if "rhs_feature_op" in ch and ch["rhs_feature_op"] is not None:
                if ch["rhs_feature_op"] not in SYMBOLIC_FEATURE_OPS:
                    raise DescriptorError(f"symbolic channel[{idx}] unknown rhs_feature_op {ch['rhs_feature_op']!r}")
            for s_key in ("feature_slice", "rhs_slice"):
                cfs = ch.get(s_key)
                if cfs is not None:
                    if (
                        not isinstance(cfs, (list, tuple))
                        or len(cfs) != 2
                        or not all(isinstance(v, int) and not isinstance(v, bool) for v in cfs)
                    ):
                        raise DescriptorError(f"channel[{idx}].{s_key} must be [start, end] integers")
                    if cfs[0] < 0 or cfs[1] - cfs[0] < 2:
                        raise DescriptorError(f"channel[{idx}].{s_key} {cfs} must span at least 2 features")
                    if feature_dim is not None and cfs[1] > feature_dim:
                        raise DescriptorError(
                            f"channel[{idx}].{s_key} end {cfs[1]} exceeds feature_dim {feature_dim}"
                        )
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


def _slope_3d(xw: torch.Tensor) -> torch.Tensor:
    """Least-squares slope along time (dim=1) for 3D tensor [B, k, F] -> [B, F]."""
    _, t, _ = xw.shape
    tt = torch.linspace(0.0, 1.0, t, device=xw.device, dtype=xw.dtype)
    tt = tt - tt.mean()
    denom = (tt * tt).sum().clamp_min(1e-8)
    return (xw * tt[None, :, None]).sum(dim=1) / denom


def _symbolic_time_reduce(xw: torch.Tensor, time_op: str) -> torch.Tensor:
    """Reduce [B, k, F] along time (dim=1) to [B, F]."""
    k = int(xw.shape[1])
    if time_op == "mean":
        return xw.mean(dim=1)
    if time_op == "std":
        return xw.std(dim=1)
    if time_op == "slope":
        return _slope_3d(xw)
    if time_op == "accel":
        if k < 4:
            return _slope_3d(xw)
        mid = max(2, k // 2)
        return _slope_3d(xw[:, mid:, :]) - _slope_3d(xw[:, :mid, :])
    if time_op == "ema_diff":
        w = torch.exp(torch.linspace(-1.5, 0.0, k, device=xw.device, dtype=xw.dtype))
        w = w / w.sum().clamp_min(1e-8)
        ema = (xw * w[None, :, None]).sum(dim=1)
        return ema - xw.mean(dim=1)
    if time_op == "max_drawdown":
        cum = torch.cumsum(xw, dim=1)
        rmax = torch.cummax(cum, dim=1).values
        dd = (rmax - cum) / xw.std(dim=1, keepdim=True).clamp_min(1e-6)
        return dd.amax(dim=1)
    if time_op == "semivar_asymmetry":
        xc = xw - xw.mean(dim=1, keepdim=True)
        up = torch.relu(xc).pow(2).mean(dim=1)
        dn = torch.relu(-xc).pow(2).mean(dim=1)
        return (up - dn) / (up + dn + 1e-8)
    if time_op == "skew":
        mean = xw.mean(dim=1, keepdim=True)
        std = xw.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = (xw - mean) / std
        return (z ** 3).mean(dim=1)
    if time_op == "kurt":
        mean = xw.mean(dim=1, keepdim=True)
        std = xw.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = (xw - mean) / std
        return (z ** 4).mean(dim=1) - 3.0
    if time_op == "q_spread":
        q = torch.quantile(xw, torch.tensor([0.15, 0.85], device=xw.device, dtype=xw.dtype), dim=1)
        std = xw.std(dim=1).clamp_min(1e-6)
        return (q[1] - q[0]) / std
    if time_op == "autocorr_1":
        if k < 3:
            return xw.mean(dim=1)
        a = xw[:, 1:, :] - xw[:, 1:, :].mean(dim=1, keepdim=True)
        c = xw[:, :-1, :] - xw[:, :-1, :].mean(dim=1, keepdim=True)
        denom = (a.norm(dim=1) * c.norm(dim=1)).clamp_min(1e-8)
        return (a * c).sum(dim=1) / denom
    if time_op == "hurst_proxy":
        if k < 3:
            return xw.std(dim=1)
        dx1 = xw[:, 1:, :] - xw[:, :-1, :]
        dx2 = xw[:, 2:, :] - xw[:, :-2, :]
        v1 = dx1.var(dim=1, unbiased=False)
        v2 = dx2.var(dim=1, unbiased=False)
        return v2 / (2.0 * v1 + 1e-8)
    if time_op == "path_efficiency":
        if k < 2:
            return xw.mean(dim=1).abs()
        net = (xw[:, -1, :] - xw[:, 0, :]).abs()
        tot = (xw[:, 1:, :] - xw[:, :-1, :]).abs().sum(dim=1)
        return net / (tot + 1e-8)
    raise DescriptorError(f"unknown symbolic time_op {time_op!r}")


def _symbolic_feature_reduce(feat_mat: torch.Tensor, feature_op: str) -> torch.Tensor:
    """Reduce [B, F] across features (dim=1) to [B]."""
    f_dim = int(feat_mat.shape[1])
    if feature_op == "mean":
        return feat_mat.mean(dim=1)
    if feature_op == "std":
        return feat_mat.std(dim=1)
    if feature_op == "max":
        return feat_mat.amax(dim=1)
    if feature_op == "min":
        return feat_mat.amin(dim=1)
    if feature_op == "iqr":
        q = torch.quantile(feat_mat, torch.tensor([0.25, 0.75], device=feat_mat.device, dtype=feat_mat.dtype), dim=1)
        return q[1] - q[0]
    if feature_op == "l2_norm":
        return torch.sqrt((feat_mat ** 2).mean(dim=1) + 1e-8)
    if feature_op == "top_bottom_spread":
        k_q = max(1, f_dim // 4)
        sorted_f, _ = torch.sort(feat_mat, dim=1)
        return sorted_f[:, -k_q:].mean(dim=1) - sorted_f[:, :k_q].mean(dim=1)
    if feature_op == "corr_half":
        mid = max(1, f_dim // 2)
        a = feat_mat[:, :mid]
        c = feat_mat[:, -mid:]
        ac = a - a.mean(dim=1, keepdim=True)
        cc = c - c.mean(dim=1, keepdim=True)
        denom = (ac.norm(dim=1) * cc.norm(dim=1)).clamp_min(1e-8)
        return (ac * cc).sum(dim=1) / denom
    raise DescriptorError(f"unknown symbolic feature_op {feature_op!r}")


def _eval_symbolic_channel(x_full: torch.Tensor, ch: dict[str, Any], default_window: int) -> torch.Tensor:
    """Evaluate one symbolic channel specification on x_full [B, T, F] -> [B]."""
    fs = ch.get("feature_slice")
    x_lhs = x_full[:, :, int(fs[0]): int(fs[1])] if fs else x_full
    w_lhs = int(ch.get("window") or default_window)
    xw_lhs = _last_window(x_lhs, w_lhs)
    t_op = ch.get("time_op", "mean")
    f_op = ch.get("feature_op", "mean")
    inter = ch.get("interaction", "none")

    lhs_feat = _symbolic_time_reduce(xw_lhs, t_op)
    lhs_val = _symbolic_feature_reduce(lhs_feat, f_op)

    if inter != "none":
        rhs_fs = ch.get("rhs_slice")
        if rhs_fs:
            x_rhs = x_full[:, :, int(rhs_fs[0]): int(rhs_fs[1])]
        else:
            # Default RHS is complementary half of features
            mid_f = max(2, int(x_full.shape[2]) // 2)
            x_rhs = x_full[:, :, mid_f:] if not fs or int(fs[0]) < mid_f else x_full[:, :, :mid_f]
        w_rhs = int(ch.get("rhs_window") or w_lhs)
        xw_rhs = _last_window(x_rhs, w_rhs)
        rhs_t_op = ch.get("rhs_time_op") or t_op
        rhs_f_op = ch.get("rhs_feature_op") or f_op

        if inter in {"diff", "ratio", "product"}:
            rhs_feat = _symbolic_time_reduce(xw_rhs, rhs_t_op)
            rhs_val = _symbolic_feature_reduce(rhs_feat, rhs_f_op)
            if inter == "diff":
                val = lhs_val - rhs_val
            elif inter == "ratio":
                val = (lhs_val - rhs_val) / (lhs_val.abs() + rhs_val.abs() + 1e-6)
            else:
                val = lhs_val * rhs_val
        else:
            k_common = min(int(xw_lhs.shape[1]), int(xw_rhs.shape[1]))
            s_lhs = xw_lhs[:, -k_common:, :].mean(dim=2)
            s_rhs = xw_rhs[:, -k_common:, :].mean(dim=2)
            if inter == "lead_lag_corr" and k_common >= 3:
                a = s_lhs[:, 1:] - s_lhs[:, 1:].mean(dim=1, keepdim=True)
                c = s_rhs[:, :-1] - s_rhs[:, :-1].mean(dim=1, keepdim=True)
                val = (a * c).sum(dim=1) / (a.norm(dim=1) * c.norm(dim=1) + 1e-8)
            elif inter == "beta":
                a = s_lhs - s_lhs.mean(dim=1, keepdim=True)
                c = s_rhs - s_rhs.mean(dim=1, keepdim=True)
                val = (a * c).sum(dim=1) / ((c * c).sum(dim=1) + 1e-6)
            else:  # corr
                a = s_lhs - s_lhs.mean(dim=1, keepdim=True)
                c = s_rhs - s_rhs.mean(dim=1, keepdim=True)
                val = (a * c).sum(dim=1) / (a.norm(dim=1) * c.norm(dim=1) + 1e-8)
    else:
        val = lhs_val

    trans = ch.get("transform", "identity")
    if trans == "abs":
        val = val.abs()
    elif trans == "tanh":
        val = torch.tanh(val)
    elif trans == "sign_sqrt":
        val = torch.sign(val) * torch.sqrt(val.abs() + 1e-8)
    weight = float(ch.get("weight", 1.0))
    return val * weight


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

    if family == "semivar_ratio":
        # Upside vs downside semivariance asymmetry across features and time
        xw_c = xw - xw.mean(dim=1, keepdim=True)
        up_sq = torch.relu(xw_c).pow(2).mean(dim=1)  # [B, F]
        dn_sq = torch.relu(-xw_c).pow(2).mean(dim=1)  # [B, F]
        dn_share = dn_sq / (up_sq + dn_sq + 1e-8)  # [B, F]
        f1 = dn_share.mean(dim=1)
        f2 = dn_share.std(dim=1)
        mc = m - m.mean(dim=1, keepdim=True)
        up_m = torch.relu(mc).pow(2).mean(dim=1)
        dn_m = torch.relu(-mc).pow(2).mean(dim=1)
        f3 = torch.log((up_m + 1e-6) / (dn_m + 1e-6))
        if xw.shape[1] >= 2:
            dx = xw[:, 1:, :] - xw[:, :-1, :]
            dx_up = torch.relu(dx).pow(2).mean(dim=(1, 2))
            dx_dn = torch.relu(-dx).pow(2).mean(dim=(1, 2))
            f4 = (dx_up - dx_dn) / (dx_up + dx_dn + 1e-8)
        else:
            f4 = f1
        f5 = (dn_sq.amax(dim=1) - up_sq.amax(dim=1)) / (dn_sq.amax(dim=1) + up_sq.amax(dim=1) + 1e-8)
        return torch.stack([f1, f2, f3, f4, f5], dim=1)

    if family == "vol_price_corr":
        # Multi-scale temporal synergy between price-change channels and volume/volatility channels
        n_feat = int(xw.shape[2])
        mid_f = max(1, n_feat // 2)
        p_feat = xw[:, :, :mid_f]
        v_feat = xw[:, :, mid_f:] if mid_f < n_feat else xw[:, :, :mid_f]
        p_ser = p_feat.mean(dim=2)  # [B, k]
        v_ser = v_feat.mean(dim=2)  # [B, k]

        def _row_corr(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            ac = a - a.mean(dim=1, keepdim=True)
            cc = c - c.mean(dim=1, keepdim=True)
            denom = (ac.norm(dim=1) * cc.norm(dim=1)).clamp_min(1e-8)
            return (ac * cc).sum(dim=1) / denom

        f1 = _row_corr(p_ser, v_ser)
        k_half = max(2, int(xw.shape[1]) // 2)
        f2 = _row_corr(p_ser[:, -k_half:], v_ser[:, -k_half:])
        if xw.shape[1] >= 3:
            f3 = _row_corr(p_ser[:, 1:], v_ser[:, :-1])
        else:
            f3 = f1
        f4 = _slope(p_ser) - _slope(v_ser)
        m_ch = min(int(p_feat.shape[2]), int(v_feat.shape[2]))
        if m_ch >= 2:
            pc_ch = p_feat[:, :, :m_ch] - p_feat[:, :, :m_ch].mean(dim=1, keepdim=True)
            vc_ch = v_feat[:, :, :m_ch] - v_feat[:, :, :m_ch].mean(dim=1, keepdim=True)
            ch_corr = (pc_ch * vc_ch).sum(dim=1) / (pc_ch.norm(dim=1) * vc_ch.norm(dim=1) + 1e-8)
            f5 = ch_corr.std(dim=1)
        else:
            f5 = _slope(p_ser) * _slope(v_ser)
        return torch.stack([f1, f2, f3, f4, f5], dim=1)

    if family == "path_efficiency":
        # Kaufman Efficiency Ratio, directional sign persistence, and path curvature
        if xw.shape[1] >= 2:
            dx = xw[:, 1:, :] - xw[:, :-1, :]
            net = (xw[:, -1, :] - xw[:, 0, :]).abs()
            tot = dx.abs().sum(dim=1).clamp_min(1e-8)
            er = net / tot  # [B, F]
            sign_net = torch.sign(xw[:, -1, :] - xw[:, 0, :])
            dir_persist = (torch.sign(dx) == sign_net[:, None, :]).float().mean(dim=1)
        else:
            er = xw.mean(dim=1).abs()
            dir_persist = torch.ones_like(er)
        if xw.shape[1] >= 3:
            ddx = xw[:, 2:, :] - 2.0 * xw[:, 1:-1, :] + xw[:, :-2, :]
            curv = ddx.std(dim=1) / xw.std(dim=1).clamp_min(1e-6)
        else:
            curv = er
        m_er = (m[:, -1] - m[:, 0]) / (m[:, 1:] - m[:, :-1]).abs().sum(dim=1).clamp_min(1e-8) if m.shape[1] >= 2 else m.mean(dim=1)
        return torch.stack(
            [er.mean(dim=1), er.std(dim=1), dir_persist.mean(dim=1), curv.mean(dim=1), m_er],
            dim=1,
        )

    if family == "hurst_vr":
        # Multi-horizon Variance Ratio and scaling exponent profile
        if xw.shape[1] >= 4:
            dx1 = xw[:, 1:, :] - xw[:, :-1, :]
            dx2 = xw[:, 2:, :] - xw[:, :-2, :]
            dx3 = xw[:, 3:, :] - xw[:, :-3, :]
            v1 = dx1.var(dim=1, unbiased=False).clamp_min(1e-8)
            vr2 = dx2.var(dim=1, unbiased=False) / (2.0 * v1)
            vr3 = dx3.var(dim=1, unbiased=False) / (3.0 * v1)
            dm1 = (m[:, 1:] - m[:, :-1]).var(dim=1, unbiased=False).clamp_min(1e-8)
            m_vr2 = (m[:, 2:] - m[:, :-2]).var(dim=1, unbiased=False) / (2.0 * dm1)
        else:
            vr2 = xw.std(dim=1)
            vr3 = vr2
            m_vr2 = m.std(dim=1)
        return torch.stack(
            [vr2.mean(dim=1), vr2.std(dim=1), vr3.mean(dim=1), (vr3 - vr2).mean(dim=1), m_vr2],
            dim=1,
        )

    if family == "lead_lag_beta":
        # Cross-channel lead-lag predictive beta and lag-1 correlation asymmetry
        n_feat = int(xw.shape[2])
        mid_f = max(1, n_feat // 2)
        p_ser = xw[:, :, :mid_f].mean(dim=2)
        v_ser = xw[:, :, mid_f:].mean(dim=2) if mid_f < n_feat else p_ser
        if xw.shape[1] >= 3:
            p_curr = p_ser[:, 1:] - p_ser[:, 1:].mean(dim=1, keepdim=True)
            v_prev = v_ser[:, :-1] - v_ser[:, :-1].mean(dim=1, keepdim=True)
            v_curr = v_ser[:, 1:] - v_ser[:, 1:].mean(dim=1, keepdim=True)
            p_prev = p_ser[:, :-1] - p_ser[:, :-1].mean(dim=1, keepdim=True)
            corr_v_leads_p = (p_curr * v_prev).sum(dim=1) / (p_curr.norm(dim=1) * v_prev.norm(dim=1) + 1e-8)
            corr_p_leads_v = (v_curr * p_prev).sum(dim=1) / (v_curr.norm(dim=1) * p_prev.norm(dim=1) + 1e-8)
            beta_vp = (p_curr * v_prev).sum(dim=1) / ((v_prev * v_prev).sum(dim=1) + 1e-6)
        else:
            corr_v_leads_p = _slope(p_ser)
            corr_p_leads_v = _slope(v_ser)
            beta_vp = corr_v_leads_p - corr_p_leads_v
        lead_asym = corr_v_leads_p - corr_p_leads_v
        ch_disp = (xw[:, :, :mid_f].std(dim=1).mean(dim=1) - xw[:, :, mid_f:].std(dim=1).mean(dim=1)) if mid_f < n_feat else lead_asym
        return torch.stack([corr_v_leads_p, corr_p_leads_v, lead_asym, beta_vp, ch_disp], dim=1)

    if family == "regime_transition":
        # Short-vs-prior window regime shift in volatility, slope, and cross-feature dispersion
        k_len = int(xw.shape[1])
        mid_t = max(1, k_len // 2)
        prior = xw[:, :mid_t, :]
        recent = xw[:, mid_t:, :] if mid_t < k_len else prior
        vol_prior = prior.std(dim=1).clamp_min(1e-6)
        vol_recent = recent.std(dim=1).clamp_min(1e-6)
        log_vol_ratio = torch.log(vol_recent / vol_prior)
        mean_shift = (recent.mean(dim=1) - prior.mean(dim=1)) / vol_prior
        disp_shift = recent.std(dim=2).mean(dim=1) - prior.std(dim=2).mean(dim=1)
        slope_shift = (_slope(m[:, mid_t:]) - _slope(m[:, :mid_t])) if mid_t >= 2 and (k_len - mid_t) >= 2 else _slope(m)
        return torch.stack(
            [log_vol_ratio.mean(dim=1), log_vol_ratio.std(dim=1), mean_shift.mean(dim=1), disp_shift, slope_shift],
            dim=1,
        )

    if family == "tail_asymmetry":
        # Upper vs lower tail expected shortfall and extreme excursion asymmetry
        xc = xw - xw.mean(dim=1, keepdim=True)
        std = xw.std(dim=1, keepdim=True).clamp_min(1e-6)
        z = xc / std  # [B, k, F]
        up_tail = torch.relu(z - 0.8).mean(dim=1)  # [B, F]
        dn_tail = torch.relu(-z - 0.8).mean(dim=1)  # [B, F]
        es_asym = (up_tail - dn_tail) / (up_tail + dn_tail + 1e-6)
        max_gain = z.amax(dim=1)
        max_loss = (-z).amax(dim=1)
        excursion_asym = (max_gain - max_loss) / (max_gain + max_loss + 1e-6)
        m_z = (m - m.mean(dim=1, keepdim=True)) / m.std(dim=1, keepdim=True).clamp_min(1e-6)
        m_tail = (torch.relu(m_z).pow(3).mean(dim=1) - torch.relu(-m_z).pow(3).mean(dim=1))
        return torch.stack(
            [es_asym.mean(dim=1), es_asym.std(dim=1), excursion_asym.mean(dim=1), excursion_asym.std(dim=1), m_tail],
            dim=1,
        )

    if family == "symbolic":
        channels = descriptor.get("channels") or []
        default_w = _window(descriptor)
        cols = [_eval_symbolic_channel(x, ch, default_w) for ch in channels]
        return torch.stack(cols, dim=1)

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

