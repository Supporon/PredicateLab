"""Subprocess worker for TSIL trials.

Invoked as ``python -m predicatelab.domains.tsil.worker job.json``.

The worker is the only process that imports qlib and the TSIL backbone.  It
receives a JSON job (data, fold, descriptor, training budget), builds the
dataset, injects the compiled predicate through ``QniverseModel(phi_fn=...)``,
trains, computes per-date cross-sectional metrics, and writes ``result.json``.

The job file is data.  The descriptor is compiled by the trusted DSL module.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def _write(path: str, payload: dict) -> None:
    """Atomic, strict JSON write: no NaN/Inf and no partial file on a crash."""
    from predicatelab.core.identity import atomic_write_text, sanitise_json, strict_json_dumps

    clean, null_paths = sanitise_json(payload)
    if null_paths:
        # Callers must not report a success when a statistic is not a number.
        if clean.get("status") == "success":
            clean["status"] = "failed"
            clean["error"] = (
                "non-finite result fields rejected: " + ", ".join(null_paths[:20])
            )
        clean["null_paths"] = null_paths
    atomic_write_text(path, strict_json_dumps(clean))


def _load_calendar(job: dict) -> list[str]:
    calendar = job.get("calendar")
    if calendar:
        return list(calendar)
    data = job["data"]
    cal_path = job.get("calendar_path") or data.get("calendar_path") \
        or (Path(job["provider_uri"]) / "calendars" / "day.txt")
    return [ln.strip() for ln in Path(cal_path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _split_segments(job: dict) -> tuple[dict, dict]:
    """Return disjoint train / early-stop / reported-test segments.

    For a fold with an explicit ``test`` window (only the holdout), the three
    segments are used as declared.

    For search and confirmation folds V1.1 no longer aliases ``test`` to the
    protocol validation window: that made early stopping and the reported
    metric use exactly the same dates.  Instead the tail of the training window
    is carved off as an early-stop segment, with the fold's purge gap between
    train and early-stop.  The protocol validation window remains the *only*
    reported segment, so model selection never sees the reported dates.
    """
    fold = job["fold"]
    purge = int(fold.get("purge_days", 2))
    if fold.get("test_start"):
        segments = {
            "train": [fold["train_start"], fold["train_end"]],
            "valid": [fold["valid_start"], fold["valid_end"]],
            "test": [fold["test_start"], fold["test_end"]],
        }
        return segments, {
            "mode": "declared_test",
            "early_stop": list(segments["valid"]),
            "reported": list(segments["test"]),
            "purge_days": purge,
        }

    calendar = _load_calendar(job)
    train_days = [d for d in calendar if fold["train_start"] <= d <= fold["train_end"]]
    frac = float(job.get("training", {}).get("early_stop_frac", 0.1))
    if len(train_days) < purge + 8:
        raise ValueError(
            f"fold {fold['fold_id']}: training period has only {len(train_days)} "
            f"trading days; cannot carve an early-stop tail with purge={purge}"
        )
    n_tail = int(round(len(train_days) * frac))
    n_tail = max(1, min(n_tail, len(train_days) - purge - 2))
    es_start_idx = len(train_days) - n_tail
    train_end_idx = es_start_idx - purge - 1
    if train_end_idx < 1:
        raise ValueError(f"fold {fold['fold_id']}: early-stop carve leaves no training data")
    segments = {
        "train": [train_days[0], train_days[train_end_idx]],
        "valid": [train_days[es_start_idx], fold["train_end"]],
        "test": [fold["valid_start"], fold["valid_end"]],
    }
    return segments, {
        "mode": "carved_early_stop_from_train",
        "early_stop": list(segments["valid"]),
        "reported": list(segments["test"]),
        "purge_days": purge,
        "early_stop_frac": frac,
        "n_train_days": len(train_days),
        "n_early_stop_days": n_tail,
    }


def _make_handler_config(job: dict, segments: dict) -> dict:
    data = job["data"]
    calendar = _load_calendar(job)
    seq_len = int(data["seq_len"])
    # warm-up history for the first training sample
    start = segments["train"][0]
    prior = [d for d in calendar if d < start][-max(seq_len * 2, 40):]
    start_time = prior[0] if prior else start
    end_time = max(segments["train"][1], segments["valid"][1], segments["test"][1])
    return {
        "start_time": start_time,
        "end_time": end_time,
        # Fit preprocessing on the carved training segment only.
        "fit_start_time": segments["train"][0],
        "fit_end_time": segments["train"][1],
        "instruments": data["market"],
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": [data["label"]],
    }


def _dataset_phi_type(job: dict) -> str:
    desc = job.get("descriptor") or {}
    from predicatelab.domains.tsil.dsl import needs_timestamp

    if job["variant"] == "baseline":
        return "mse"
    if job["variant"] == "control":
        ck = str(job.get("control_kind") or "ones").strip()
        if ck in {"", "ones", "control:ones"}:
            return "ones"
        return "external"
    if job["variant"] == "probe":
        return "phi_timestamp"
    return "phi_timestamp" if needs_timestamp(desc) else "external"


SUPPORTED_CONTROL_KINDS = {
    "",
    "ones",
    "control:ones",
    "perm",
    "control:perm",
    "random",
    "control:random",
    "block_diag",
    "control:block_diag",
}


def _make_control_phi_fn(
    control_kind: str,
    *,
    seed: int = 2025,
    descriptor: dict | None = None,
    tsil_root: str | None = None,
    seq_len: int = 20,
    feature_dim: int = 158,
):
    """Build the ``phi_fn(batch_x, timestamp_x) -> (I, P)`` for a control variant.

    Supported ``control_kind`` values:
    - ``ones`` / ``control:ones``: returns ``None`` (uses upstream ``get_ones_I_P``).
    - ``perm`` / ``control:perm``: computes a non-degenerate multi-feature kernel ``P``
      (from ``descriptor`` if provided, else the canonical momentum operator) and applies
      a deterministic row/column permutation ``P_pi = P[pi][:, pi]`` seeded by ``(seed, step)``.
      This preserves the exact eigenvalue spectrum ``{lambda_i}``, ``tr(P)=1``, and ``||P||_F``
      while breaking all semantic cross-sample pairings.
    - ``random`` / ``control:random``: random standardized Gaussian projection kernel ``P``.
    - ``block_diag`` / ``control:block_diag``: trace-normalized block-diagonal constant kernel.
    """
    import torch
    from predicatelab.domains.tsil.dsl import compile_descriptor

    ck = str(control_kind or "ones").strip()
    if ck not in SUPPORTED_CONTROL_KINDS:
        raise ValueError(f"unsupported control_kind: {control_kind!r}")
    if ck in {"", "ones", "control:ones"}:
        return None

    step_counter = [0]

    if ck in {"perm", "control:perm"}:
        base_desc = descriptor if (descriptor and isinstance(descriptor, dict) and descriptor.get("family")) else {
            "family": "momentum",
            "params": {"window": min(10, max(3, seq_len // 2))},
        }
        base_fn = compile_descriptor(
            base_desc,
            tsil_root=tsil_root,
            seq_len=seq_len,
            feature_dim=feature_dim,
        )

        def _perm_phi_fn(batch_x: torch.Tensor, timestamp_x: torch.Tensor | None = None):
            step = step_counter[0]
            step_counter[0] = step + 1
            i_mat, p_mat = base_fn(batch_x, timestamp_x)
            b = int(p_mat.shape[0])
            if b < 2:
                return i_mat, p_mat
            gen = torch.Generator(device="cpu")
            gen.manual_seed((int(seed) * 1000003 + step * 9176 + b) & 0x7FFFFFFF)
            perm = torch.randperm(b, generator=gen)
            if bool((perm == torch.arange(b)).all().item()):
                perm = torch.roll(perm, shifts=1)
            perm_dev = perm.to(device=p_mat.device)
            p_perm = p_mat.index_select(0, perm_dev).index_select(1, perm_dev)
            return i_mat, p_perm

        return _perm_phi_fn

    if ck in {"random", "control:random"}:
        def _random_phi_fn(batch_x: torch.Tensor, timestamp_x: torch.Tensor | None = None):
            step = step_counter[0]
            step_counter[0] = step + 1
            b = int(batch_x.shape[0])
            dev = batch_x.device
            dtype = batch_x.dtype
            i_mat = torch.eye(b, device=dev, dtype=dtype)
            if b < 2:
                return i_mat, i_mat
            gen = torch.Generator(device="cpu")
            gen.manual_seed((int(seed) * 1000033 + step * 7919 + b) & 0x7FFFFFFF)
            z = torch.randn((b, 4), generator=gen, dtype=torch.float32).to(device=dev, dtype=dtype)
            z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True).clamp_min(1e-6)
            norm = torch.linalg.norm(z).clamp_min(1e-8)
            z = z / norm
            return i_mat, z @ z.T

        return _random_phi_fn

    # block_diag / control:block_diag
    def _block_diag_phi_fn(batch_x: torch.Tensor, timestamp_x: torch.Tensor | None = None):
        b = int(batch_x.shape[0])
        dev = batch_x.device
        dtype = batch_x.dtype
        i_mat = torch.eye(b, device=dev, dtype=dtype)
        if b < 2:
            return i_mat, i_mat
        n_blocks = min(4, b)
        p_mat = torch.zeros((b, b), device=dev, dtype=dtype)
        bounds = torch.linspace(0, b, steps=n_blocks + 1, dtype=torch.long)
        for idx in range(n_blocks):
            s, e = int(bounds[idx].item()), int(bounds[idx + 1].item())
            if e > s:
                p_mat[s:e, s:e] = 1.0
        trace = torch.diagonal(p_mat).sum().clamp_min(1e-8)
        return i_mat, p_mat / trace

    return _block_diag_phi_fn


def _build_dataset(job: dict, batch_size: int, shuffle: bool):
    from src.dataset_ori_onehot import MTSDatasetH

    segments, split_meta = _split_segments(job)
    # MTSDatasetH (upstream) expects a handler *config dict* because it reads
    # ``handler['kwargs']`` before delegating construction to qlib's DatasetH.
    handler = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": _make_handler_config(job, segments),
    }
    dataset = MTSDatasetH(
        handler=handler,
        segments=segments,
        seq_len=int(job["data"]["seq_len"]),
        horizon=int(job["data"].get("horizon", 1)),
        batch_size=batch_size,
        shuffle=shuffle,
        phi_type=_dataset_phi_type(job),
    )
    return dataset, split_meta


def _date_level(index) -> int | None:
    import pandas as pd

    for level in range(index.nlevels):
        if pd.api.types.is_datetime64_any_dtype(index.get_level_values(level)):
            return level
    names = getattr(index, "names", None) or []
    for level, name in enumerate(names):
        if str(name or "").lower() in {"datetime", "date", "dt"}:
            return level
    return None


def _prediction_columns(preds):
    """Map a prediction frame to (instrument, date) regardless of level order.

    ``QniverseModel.test_epoch`` returns an index in ``(datetime, instrument)``
    order.  Writing level 0 as the instrument silently produced
    ``pd.Timestamp("000001.SZ")`` (a DateParseError) and an instrument column
    full of dates.  Detect the datetime level instead of assuming a position.
    """
    import numpy as np
    import pandas as pd

    index = preds.index
    date_level = _date_level(index)
    if date_level is None:
        if index.nlevels != 1:
            raise ValueError(
                "prediction index has no datetime level: "
                + repr([str(index.get_level_values(i).dtype) for i in range(index.nlevels)])
            )
        dates = pd.to_datetime(index.get_level_values(0), errors="raise")
        instruments = np.arange(len(index)).astype(str)
    else:
        other = [lvl for lvl in range(index.nlevels) if lvl != date_level]
        dates = pd.to_datetime(index.get_level_values(date_level), errors="raise")
        instruments = (index.get_level_values(other[0]).astype(str).values
                       if other else np.arange(len(index)).astype(str))
    date_values = [str(pd.Timestamp(d).date()) for d in dates]
    return np.asarray(instruments, dtype=object), np.asarray(date_values, dtype=object)


def _prediction_metrics(preds) -> tuple[dict, dict]:
    import numpy as np
    import pandas as pd

    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    idx = preds.index
    date_level = _date_level(idx)
    if date_level is None:
        # single index of timestamps
        dates = pd.to_datetime(idx.get_level_values(0))
        if "instrument" in preds.columns:
            instruments = np.asarray(preds["instrument"], dtype=str)
        else:
            instruments = np.arange(len(idx)).astype(str)
    else:
        dates = pd.to_datetime(idx.get_level_values(date_level))
        other = [lvl for lvl in range(idx.nlevels) if lvl != date_level]
        if other:
            instruments = idx.get_level_values(other[0]).astype(str).values
        elif "instrument" in preds.columns:
            instruments = np.asarray(preds["instrument"], dtype=str)
        else:
            instruments = np.arange(len(idx)).astype(str)
    df = pd.DataFrame({
        "score": np.asarray(preds["score"], dtype=float),
        "label": np.asarray(preds["label"], dtype=float),
        "dt": dates,
        "inst": instruments,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    ic, ric, mse, mae = {}, {}, {}, {}
    by_date_scores: dict[str, pd.Series] = {}
    for dt, g in df.groupby("dt"):
        if len(g) < 3:
            continue
        if g["score"].nunique() < 2 or g["label"].nunique() < 2:
            continue
        key = pd.Timestamp(dt).strftime("%Y-%m-%d")
        a = float(g["score"].corr(g["label"]))
        b = float(g["score"].corr(g["label"], method="spearman"))
        if np.isfinite(a):
            ic[key] = a
        if np.isfinite(b):
            ric[key] = b
        mse[key] = float(((g["score"] - g["label"]) ** 2).mean())
        mae[key] = float((g["score"] - g["label"]).abs().mean())
        by_date_scores[key] = g.groupby("inst")["score"].mean()

    def stats(series: dict) -> tuple[float, float, float]:
        if not series:
            return float("nan"), float("nan"), float("nan")
        arr = np.asarray(list(series.values()), dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        icir = float(mean / std * np.sqrt(252)) if std and np.isfinite(std) and std > 0 else float("nan")
        return mean, std, icir

    ic_mean, ic_std, icir = stats(ic)
    ric_mean, ric_std, ricir = stats(ric)

    sorted_dates = sorted(ic.keys())
    if len(sorted_dates) >= 2:
        mid = len(sorted_dates) // 2
        ic_first_half = float(np.mean([ic[d] for d in sorted_dates[:mid]]))
        ic_second_half = float(np.mean([ic[d] for d in sorted_dates[mid:]]))
        subperiod_sign_consistent = bool(
            (ic_first_half > 0.0 and ic_second_half > 0.0)
            or (ic_first_half < 0.0 and ic_second_half < 0.0)
        )
    elif len(sorted_dates) == 1:
        ic_first_half = float(ic[sorted_dates[0]])
        ic_second_half = float(ic[sorted_dates[0]])
        subperiod_sign_consistent = bool(ic_first_half != 0.0)
    else:
        ic_first_half = float("nan")
        ic_second_half = float("nan")
        subperiod_sign_consistent = False

    autocorr_list: list[float] = []
    score_dates = sorted(by_date_scores.keys())
    for i in range(1, len(score_dates)):
        s_prev = by_date_scores[score_dates[i - 1]]
        s_curr = by_date_scores[score_dates[i]]
        common = s_prev.index.intersection(s_curr.index)
        if len(common) < 3:
            continue
        p0 = s_prev.loc[common]
        p1 = s_curr.loc[common]
        if p0.nunique() < 2 or p1.nunique() < 2:
            continue
        ac = float(p0.corr(p1, method="spearman"))
        if np.isfinite(ac):
            autocorr_list.append(ac)
    if autocorr_list:
        pred_rank_autocorr = float(np.mean(autocorr_list))
    elif ic:
        pred_rank_autocorr = 0.0
    else:
        pred_rank_autocorr = float("nan")

    metrics = {
        "ic": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "rank_ic": ric_mean,
        "rank_ic_std": ric_std,
        "rank_icir": ricir,
        "mse": float(np.mean(list(mse.values()))) if mse else float("nan"),
        "mae": float(np.mean(list(mae.values()))) if mae else float("nan"),
        "n_dates": len(ic),
        "ic_first_half": ic_first_half,
        "ic_second_half": ic_second_half,
        "subperiod_sign_consistent": subperiod_sign_consistent,
        "pred_rank_autocorr": pred_rank_autocorr,
        "rank_autocorr": pred_rank_autocorr,
    }
    return metrics, {"ic": ic, "rank_ic": ric, "mse": mse, "mae": mae}


_evaluate_frame = _prediction_metrics


def run_trial(job: dict) -> dict:
    t0 = time.time()
    import numpy as np

    from predicatelab.domains.tsil.dsl import compile_descriptor

    descriptor = job.get("descriptor") or {}
    fold = job["fold"]
    seed = int(job.get("seed", 2025))
    training = job.get("training", {})
    model_cfg = dict(job["model_config"])
    model_cfg["phi_type"] = _dataset_phi_type(job)
    model_cfg["seq_len"] = int(job["data"]["seq_len"])

    phi_fn = None
    if job["variant"] == "candidate":
        phi_fn = compile_descriptor(
            descriptor,
            tsil_root=job["tsil_root"],
            seq_len=int(job["data"]["seq_len"]),
            feature_dim=int(job["data"].get("feature_dim", 158)),
        )
    elif job["variant"] == "control":
        phi_fn = _make_control_phi_fn(
            str(job.get("control_kind") or "ones"),
            seed=seed,
            descriptor=descriptor,
            tsil_root=job.get("tsil_root"),
            seq_len=int(job["data"]["seq_len"]),
            feature_dim=int(job["data"].get("feature_dim", 158)),
        )

    batch_size = int(training.get("batch_size", 800))
    dataset, split_meta = _build_dataset(job, batch_size, bool(training.get("shuffle", True)))

    from src.model_backbone_phitp import QniverseModel

    model = QniverseModel(
        model_config=model_cfg,
        model_type=training.get("model_type", "LSTM"),
        lr=float(training.get("lr", 1e-3)),
        n_epochs=int(training.get("n_epochs", 8)),
        early_stop=int(training.get("early_stop", 3)),
        smooth_steps=int(training.get("smooth_steps", 5)),
        max_steps_per_epoch=training.get("max_steps_per_epoch"),
        seed=seed,
        logdir=None,
        eval_train=False,
        eval_test=False,
        plot_training=False,
        plot_show=False,
        phi_fn=phi_fn,
        save_artifacts=False,
    )
    preds, fit_metrics = model.fit(dataset)
    metrics, series = _prediction_metrics(preds)
    primary = "ic"
    if not np.isfinite(metrics.get(primary, np.nan)):
        return {
            "status": "failed",
            "error": f"primary metric {primary!r} is not finite ({metrics.get(primary)!r}); "
                     "no valid independent samples",
            "metrics": metrics, "metric_series": series, "fitted_params": {},
            "predictions_path": "", "duration_s": time.time() - t0,
            "worker": {"variant": job["variant"], "fold": fold["fold_id"]},
        }
    if not series.get(primary):
        return {
            "status": "failed",
            "error": f"empty {primary!r} series; a success cannot be claimed",
            "metrics": metrics, "metric_series": series, "fitted_params": {},
            "predictions_path": "", "duration_s": time.time() - t0,
            "worker": {"variant": job["variant"], "fold": fold["fold_id"]},
        }

    pred_path = ""
    pred_index: dict = {}
    if job.get("save_predictions"):
        import numpy as np
        import pandas as pd

        pred_path = str(Path(job["output_dir"]) / "predictions.npz")
        inst, dates = _prediction_columns(preds)
        np.savez_compressed(
            pred_path,
            instrument=inst,
            date=dates,
            score=np.asarray(preds["score"], dtype=float),
            label=np.asarray(preds["label"], dtype=float),
        )
        pred_index = {
            "n_rows": int(len(inst)),
            "date_min": min(dates) if len(dates) else None,
            "date_max": max(dates) if len(dates) else None,
            "has_label": bool("label" in preds),
        }

    # Trained weights live in their own ``models/`` directory so a snapshot can
    # include or exclude them without touching predictions or reports.  The
    # engine only asks for this for full/final trials by default.
    model_path = ""
    model_sha = ""
    if job.get("save_model"):
        import hashlib

        import torch

        model_dir = Path(job["output_dir"]) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / "model.pt")
        torch.save({
            "model": model.model.state_dict(),
            "model_type": training.get("model_type", "LSTM"),
            "model_config": model_cfg,
            "seed": seed,
            "fold": fold["fold_id"],
            "descriptor": descriptor,
        }, model_path)
        digest = hashlib.sha256()
        with open(model_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        model_sha = digest.hexdigest()

    identity = job.get("identity") or {}
    return {
        "status": "success",
        "metrics": metrics,
        "fit_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (fit_metrics or {}).items()},
        "metric_series": series,
        "fitted_params": {},
        "predictions_path": pred_path,
        "model_path": model_path,
        "model_sha256": model_sha,
        "prediction_index": pred_index,
        "duration_s": time.time() - t0,
        "error": "",
        "worker": {
            "variant": job["variant"],
            "control_kind": job.get("control_kind", ""),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "fold": fold["fold_id"],
            "seed": seed,
            "n_epochs": training.get("n_epochs"),
            "max_steps_per_epoch": training.get("max_steps_per_epoch"),
            "train": [fold["train_start"], fold["train_end"]],
            "valid": [fold["valid_start"], fold["valid_end"]],
            "test": [fold.get("test_start"), fold.get("test_end")],
            "split": split_meta,
            "tau_mode": model_cfg.get("tau_mode", "learned"),
            "fixed_tau": model_cfg.get("fixed_tau"),
            "identity": identity,
        },
    }


def make_probe_batch(job: dict) -> dict:
    import numpy as np

    fold = job["fold"]
    dataset, _split = _build_dataset(job, batch_size=64, shuffle=False)
    train_ds = dataset.prepare("train")
    x = ts = None
    for batch in train_ds:
        data = batch["data"].cpu().numpy() if hasattr(batch["data"], "cpu") else np.asarray(batch["data"])
        x = data[:, :, :-1].astype("float32")
        t = batch.get("timestamp_feature")
        if t is not None:
            ts = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
            ts = ts.astype("float32")
        break
    if x is None:
        raise RuntimeError("no training batch produced for probe")
    out = {"x": x}
    if ts is not None:
        out["timestamp"] = ts
    path = job["output_path"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return {"status": "success", "path": path, "shape": list(x.shape),
            "has_timestamp": ts is not None}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m predicatelab.domains.tsil.worker job.json", file=sys.stderr)
        return 2
    job_path = argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    result_path = job["result_path"]
    # Remove any stale result before doing anything that could fail.
    if Path(result_path).exists():
        Path(result_path).unlink()
    # CUDA_VISIBLE_DEVICES must be set before torch is imported
    if job.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    tsil_root = job["tsil_root"]
    if tsil_root not in sys.path:
        sys.path.insert(0, tsil_root)

    try:
        mode = job.get("mode", "trial")
        import qlib
        try:
            from joblib import Parallel
            from joblib._parallel_backends import MultiprocessingBackend
            from qlib.utils import paral
            if not getattr(paral.ParallelExt, "_joblib13_patched", False):
                def _compat_init(self, *args, **kwargs):
                    maxtasksperchild = kwargs.pop("maxtasksperchild", None)
                    Parallel.__init__(self, *args, **kwargs)
                    if isinstance(getattr(self, "_backend", None), MultiprocessingBackend):
                        bag = getattr(self, "_backend_args", None)
                        if bag is None:
                            bag = getattr(self, "_backend_kwargs", None)
                        if isinstance(bag, dict):
                            bag["maxtasksperchild"] = maxtasksperchild

                paral.ParallelExt.__init__ = _compat_init
                paral.ParallelExt._joblib13_patched = True
        except Exception:
            pass
        qlib.init(provider_uri=job["provider_uri"], region="cn")
        if mode == "trial":
            result = run_trial(job)
        elif mode == "probe_batch":
            result = make_probe_batch(job)
        else:
            raise ValueError(f"unknown worker mode {mode!r}")
    except Exception as exc:
        result = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "metrics": {},
            "metric_series": {},
            "duration_s": 0.0,
        }
    _write(result_path, result)
    print(f"[worker] {result.get('status')} -> {result_path}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
