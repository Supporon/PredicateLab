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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _make_handler_config(job: dict) -> dict:
    data = job["data"]
    fold = job["fold"]
    calendar = job.get("calendar")
    if not calendar:
        cal_path = data.get("calendar_path") or (Path(job["provider_uri"]) / "calendars" / "day.txt")
        calendar = [ln.strip() for ln in Path(cal_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    seq_len = int(data["seq_len"])
    # warm-up history for the first training sample
    start = fold["train_start"]
    prior = [d for d in calendar if d < start][-max(seq_len * 2, 40):]
    start_time = prior[0] if prior else start
    end_time = fold.get("test_end") or fold["valid_end"]
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fold["train_start"],
        "fit_end_time": fold["train_end"],
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
        return "ones"
    if job["variant"] == "probe":
        return "phi_timestamp"
    return "phi_timestamp" if needs_timestamp(desc) else "external"


def _build_dataset(job: dict, batch_size: int, shuffle: bool):
    from src.dataset_ori_onehot import MTSDatasetH

    # MTSDatasetH (upstream) expects a handler *config dict* because it reads
    # ``handler['kwargs']`` before delegating construction to qlib's DatasetH.
    handler = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": _make_handler_config(job),
    }
    fold = job["fold"]
    segments = {
        "train": [fold["train_start"], fold["train_end"]],
        "valid": [fold["valid_start"], fold["valid_end"]],
        # During search the holdout is absent from the protocol; aliasing `test`
        # to the validation fold keeps QniverseModel.fit's 3-segment API intact
        # without ever touching held-out data.
        "test": [fold["test_start"], fold["test_end"]] if fold.get("test_start")
        else [fold["valid_start"], fold["valid_end"]],
    }
    return MTSDatasetH(
        handler=handler,
        segments=segments,
        seq_len=int(job["data"]["seq_len"]),
        horizon=int(job["data"].get("horizon", 1)),
        batch_size=batch_size,
        shuffle=shuffle,
        phi_type=_dataset_phi_type(job),
    )


def _prediction_metrics(preds) -> tuple[dict, dict]:
    import numpy as np
    import pandas as pd

    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    idx = preds.index
    date_level = None
    for lvl in range(idx.nlevels):
        vals = idx.get_level_values(lvl)
        if pd.api.types.is_datetime64_any_dtype(vals):
            date_level = lvl
            break
    if date_level is None:
        # single index of timestamps
        dates = pd.to_datetime(idx.get_level_values(0))
    else:
        dates = pd.to_datetime(idx.get_level_values(date_level))
    df = pd.DataFrame({
        "score": np.asarray(preds["score"], dtype=float),
        "label": np.asarray(preds["label"], dtype=float),
        "dt": dates,
    }).replace([np.inf, -np.inf], np.nan).dropna()
    ic, ric, mse, mae = {}, {}, {}, {}
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
    }
    return metrics, {"ic": ic, "rank_ic": ric, "mse": mse, "mae": mae}


def run_trial(job: dict) -> dict:
    t0 = time.time()
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

    batch_size = int(training.get("batch_size", 800))
    dataset = _build_dataset(job, batch_size, bool(training.get("shuffle", True)))

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

    pred_path = ""
    if job.get("save_predictions"):
        import numpy as np
        import pandas as pd

        pred_path = str(Path(job["output_dir"]) / "predictions.npz")
        index = preds.index
        inst = index.get_level_values(0).astype(str).values if index.nlevels > 1 else np.arange(len(index)).astype(str)
        dates = [str(pd.Timestamp(d).date()) for d in
                 (index.get_level_values(1) if index.nlevels > 1 else index)]
        np.savez_compressed(
            pred_path,
            instrument=inst,
            date=np.asarray(dates, dtype=object),
            score=np.asarray(preds["score"], dtype=float),
            label=np.asarray(preds["label"], dtype=float),
        )
    return {
        "status": "success",
        "metrics": metrics,
        "fit_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in (fit_metrics or {}).items()},
        "metric_series": series,
        "fitted_params": {},
        "predictions_path": pred_path,
        "duration_s": time.time() - t0,
        "error": "",
        "worker": {
            "variant": job["variant"],
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "fold": fold["fold_id"],
            "n_epochs": training.get("n_epochs"),
            "max_steps_per_epoch": training.get("max_steps_per_epoch"),
        },
    }


def make_probe_batch(job: dict) -> dict:
    import numpy as np

    fold = job["fold"]
    dataset = _build_dataset(job, batch_size=64, shuffle=False)
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
    # CUDA_VISIBLE_DEVICES must be set before torch is imported
    if job.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    tsil_root = job["tsil_root"]
    if tsil_root not in sys.path:
        sys.path.insert(0, tsil_root)

    try:
        mode = job.get("mode", "trial")
        import qlib
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
