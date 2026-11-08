#!/usr/bin/env python3
"""Zero-dependency external independent evidence verifier for PredicateLab.

IMPORTANT: This script intentionally imports ZERO modules from ``predicatelab``.
It relies strictly on Python's standard library (``argparse``, ``hashlib``,
``json``, ``math``, ``pathlib``, ``sqlite3``, ``sys``) and ``numpy`` so that
the referee shares no code path with the platform under audit.

It performs 5 independent checks on an exported bundle directory or an
artifacts directory + SQLite database:
  1. SHA-256 hash chain verification (predictions.npz, trial result_hash,
     frozen_family.json, frozen_manifest.json, bundle checksums).
  2. Raw prediction tensor re-evaluation from disk (predictions.npz -> daily
     Pearson IC & Spearman Rank IC for TSIL, or MSE/R2/MAE/LogLoss/Accuracy/AUC
     for Tabular).
  3. Independent statistical inference re-computation (paired delta series,
     Newey-West HAC SE, Moving-Block Bootstrap 95% CI, normal two-sided p-value,
     and Benjamini-Hochberg q-values) asserting < 1e-6 discrepancy against
     stored evidence rows.
  4. Ledger & freeze chronology verification across rounds, frozen_families,
     confirmation_access, holdout_access, and .global_holdout_ledger.sqlite.
  5. Scientific state legitimacy verification (independently_confirmed requires
     time_forward confirmation, directional 95% CI lower bound > 0, and
     beat_control == True when controls exist).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Hard assertion that no predicatelab module is loaded
assert not any(m.startswith("predicatelab") for m in sys.modules), (
    "verify_evidence_independent.py must never import predicatelab"
)


# ============================================================================
# 1. Pure-stdlib canonical JSON & SHA-256 helpers
# ============================================================================
def _canonicalise(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonicalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    if isinstance(obj, float):
        return 0.0 if obj == 0.0 else obj
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _canonicalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def recompute_trial_result_digest(trial: dict[str, Any]) -> str:
    keys = (
        "status",
        "metrics",
        "metric_series",
        "fitted_params",
        "predictions_path",
        "predictions_sha256",
        "null_paths",
    )
    payload = {k: trial.get(k) for k in keys}
    if payload["null_paths"] is None:
        payload["null_paths"] = []
    if payload["predictions_path"] is None:
        payload["predictions_path"] = ""
    if payload["predictions_sha256"] is None:
        payload["predictions_sha256"] = ""
    return sha256_obj(payload)


# ============================================================================
# 2. Pure-numpy raw prediction metrics (TSIL & Tabular)
# ============================================================================
def _rankdata_average(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    sorted_a = arr[order]
    diffs = np.concatenate(([True], sorted_a[1:] != sorted_a[:-1], [True]))
    idx = np.flatnonzero(diffs)
    for start, end in zip(idx[:-1], idx[1:]):
        if end - start > 1:
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not math.isfinite(c) else c


def recompute_tsil_npz_metrics(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        dates = data["datetime"]
        scores = np.asarray(data["score"], dtype=float)
        labels = np.asarray(data["label"], dtype=float)

    mask = np.isfinite(scores) & np.isfinite(labels)
    dates = dates[mask]
    scores = scores[mask]
    labels = labels[mask]

    # Group by date in sorted order
    uniq_dates = sorted({str(d) for d in dates.tolist()})
    ic_by_day: dict[str, float] = {}
    ric_by_day: dict[str, float] = {}
    for d in uniq_dates:
        idx = np.flatnonzero(dates == d)
        if len(idx) < 3:
            continue
        s = scores[idx]
        l = labels[idx]
        if float(np.std(l)) < 1e-12:
            continue
        ic_by_day[d] = _pearson(s, l)
        ric_by_day[d] = _pearson(_rankdata_average(s), _rankdata_average(l))

    ic_vals = np.array(list(ic_by_day.values()), dtype=float)
    ric_vals = np.array(list(ric_by_day.values()), dtype=float)
    mean_ic = float(np.mean(ic_vals)) if len(ic_vals) else float("nan")
    mean_ric = float(np.mean(ric_vals)) if len(ric_vals) else float("nan")
    return {
        "ic": mean_ic,
        "rank_ic": mean_ric,
        "ic_series": ic_by_day,
        "rank_ic_series": ric_by_day,
    }


def _roc_auc_pure_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata_average(y_score)
    sum_pos_ranks = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def recompute_tabular_npz_metrics(npz_path: Path, trial_metrics: dict[str, Any]) -> dict[str, float]:
    with np.load(npz_path, allow_pickle=False) as data:
        scores = np.asarray(data["score"], dtype=float)
        labels = np.asarray(data["label"], dtype=float)

    if "mse" in trial_metrics or "r2" in trial_metrics:
        mse = float(np.mean((labels - scores) ** 2))
        mae = float(np.mean(np.abs(labels - scores)))
        ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
        ss_res = float(np.sum((labels - scores) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        return {"mse": mse, "mae": mae, "r2": r2}
    else:
        y_int = labels.astype(int)
        p_clip = np.clip(scores, 1e-15, 1.0 - 1e-15)
        log_loss_val = float(-np.mean(y_int * np.log(p_clip) + (1 - y_int) * np.log(1.0 - p_clip)))
        acc_val = float(np.mean((scores >= 0.5).astype(int) == y_int))
        auc_val = _roc_auc_pure_numpy(y_int, scores)
        return {"auc": auc_val, "accuracy": acc_val, "log_loss": log_loss_val}


# ============================================================================
# 3. Pure-numpy independent statistical inference
# ============================================================================
def indep_newey_west_se(x: np.ndarray, max_lag: int | None = None) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return float("nan")
    if max_lag is None:
        max_lag = max(1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    max_lag = min(max_lag, n - 1)
    u = arr - arr.mean()
    gamma0 = float(np.dot(u, u) / n)
    var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(u[lag:], u[:-lag]) / n)
        var += 2.0 * w * gamma
    var = max(var, 1e-18)
    return math.sqrt(var / n)


def indep_block_bootstrap_ci(
    x: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 1000,
    block: int = 5,
    seed: int = 2025,
) -> tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or float(np.std(arr)) == 0.0:
        v = float(arr[0])
        return v, v
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(n_boot, -1)[:, :n]
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def indep_normal_two_sided_p(z: float) -> float:
    if not math.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def indep_paired_effect(
    candidate_series: dict[str, float],
    baseline_series: dict[str, float],
    *,
    nw_lag: int = 5,
    n_boot: int = 1000,
    block: int = 5,
    seed: int = 2025,
) -> dict[str, Any]:
    common = sorted(set(candidate_series) & set(baseline_series))
    if not common:
        return {
            "effect": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_dates": 0,
        }
    c = np.array([candidate_series[k] for k in common], dtype=float)
    b = np.array([baseline_series[k] for k in common], dtype=float)
    mask = np.isfinite(c) & np.isfinite(b)
    c = c[mask]
    b = b[mask]
    diff = c - b
    n = len(diff)
    if n == 0:
        return {
            "effect": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_dates": 0,
        }
    mean = float(diff.mean())
    se = indep_newey_west_se(diff, max_lag=nw_lag)
    ci_low, ci_high = indep_block_bootstrap_ci(diff, n_boot=n_boot, block=block, seed=seed)
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n >= 2 and sd == 0.0:
        p = 1.0 if mean == 0.0 else 0.0
    elif math.isfinite(se) and se > 1e-15:
        p = indep_normal_two_sided_p(mean / se)
    else:
        p = float("nan")
    return {
        "effect": mean,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p,
        "n_dates": int(n),
    }


def indep_benjamini_hochberg(p_values: list[float | None]) -> list[float]:
    n = len(p_values)
    out = [float("nan")] * n
    indexed = [
        (i, float(p))
        for i, p in enumerate(p_values)
        if p is not None and math.isfinite(float(p))
    ]
    m = len(indexed)
    if m == 0:
        return out
    indexed.sort(key=lambda t: t[1])
    raw = [min(1.0, p * m / (rank + 1)) for rank, (_, p) in enumerate(indexed)]
    for i in range(m - 2, -1, -1):
        raw[i] = min(raw[i], raw[i + 1])
    for (orig_idx, _), q in zip(indexed, raw):
        out[orig_idx] = float(q)
    return out


# ============================================================================
# 4. Data loading from Bundle or Artifacts + SQLite
# ============================================================================
def _load_sqlite_data(db_path: Path, run_id: str | None) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not run_id:
            row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
            if not row:
                raise RuntimeError(f"No runs found in {db_path}")
            run_id = str(row["run_id"])

        run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        run_dict = dict(run_row) if run_row else {"run_id": run_id}
        if isinstance(run_dict.get("config_json"), str) and run_dict["config_json"]:
            run_dict["config"] = json.loads(run_dict["config_json"])

        rounds = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM rounds WHERE run_id = ? ORDER BY round_index", (run_id,)
            ).fetchall()
        ]
        for r in rounds:
            raw_sum = r.get("summary") or r.get("summary_json")
            if isinstance(raw_sum, str) and raw_sum:
                r["summary"] = json.loads(raw_sum)

        trials = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM trials WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        ]
        for t in trials:
            for col in ("metrics", "metric_series", "fitted_params", "null_paths", "worker"):
                raw_val = t.get(col) if isinstance(t.get(col), str) else t.get(f"{col}_json")
                if isinstance(raw_val, str) and raw_val:
                    t[col] = json.loads(raw_val)
                elif t.get(col) is None:
                    t[col] = [] if col == "null_paths" else {}

        evidence = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM evidence WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall()
        ]
        for e in evidence:
            raw_m = e.get("metrics") if isinstance(e.get("metrics"), str) else e.get("metrics_json")
            e["metrics"] = json.loads(raw_m) if (isinstance(raw_m, str) and raw_m) else (e.get("metrics") or {})

        ff_row = conn.execute(
            "SELECT * FROM frozen_families WHERE run_id = ?", (run_id,)
        ).fetchone()
        frozen_family = None
        if ff_row:
            ff_dict = dict(ff_row)
            raw_ff = ff_dict.get("payload") or ff_dict.get("manifest_json")
            if isinstance(raw_ff, str) and raw_ff:
                frozen_family = json.loads(raw_ff)

        final_evals = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM final_evals WHERE run_id = ?", (run_id,)
            ).fetchall()
        ]
        for f in final_evals:
            raw_fm = f.get("metrics") if isinstance(f.get("metrics"), str) else f.get("metrics_json")
            f["metrics"] = json.loads(raw_fm) if (isinstance(raw_fm, str) and raw_fm) else (f.get("metrics") or {})

        holdout_access = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM holdout_access WHERE run_id = ?", (run_id,)
            ).fetchall()
        ]
        confirmation_access: list[dict[str, Any]] = []
        tbl_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='confirmation_access'"
        ).fetchone()
        if tbl_check:
            confirmation_access = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM confirmation_access WHERE run_id = ?", (run_id,)
                ).fetchall()
            ]
        return {
            "run_id": run_id,
            "run": run_dict,
            "rounds": rounds,
            "trials": trials,
            "evidence": evidence,
            "frozen_family": frozen_family,
            "final_evals": final_evals,
            "holdout_access": holdout_access,
            "confirmation_access": confirmation_access,
        }
    finally:
        conn.close()


# ============================================================================
# 5. The 5 Independent Verification Checks
# ============================================================================
def verify_all(
    *,
    bundle_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    db_path: Path | None = None,
    run_id: str | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    root_dir = bundle_dir or artifacts_dir
    if root_dir is None:
        raise ValueError("Provide --bundle or --artifacts")
    root_dir = root_dir.resolve()

    # Locate SQLite DB if available
    if db_path is None:
        for cand_db in (
            root_dir / "predicatelab.sqlite",
            root_dir.parent / "predicatelab.sqlite",
        ):
            if cand_db.is_file():
                db_path = cand_db
                break

    db_data = _load_sqlite_data(db_path, run_id) if (db_path and db_path.is_file()) else {}
    context_path = root_dir / "context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
    report_path = root_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}

    trials: list[dict[str, Any]] = db_data.get("trials") or (
        json.loads((root_dir / "trials.json").read_text(encoding="utf-8"))
        if (root_dir / "trials.json").is_file()
        else []
    )
    evidence: list[dict[str, Any]] = db_data.get("evidence") or (
        json.loads((root_dir / "evidence.json").read_text(encoding="utf-8"))
        if (root_dir / "evidence.json").is_file()
        else report.get("evidence", [])
    )
    rounds: list[dict[str, Any]] = db_data.get("rounds") or report.get("rounds", [])
    frozen_family: dict[str, Any] | None = db_data.get("frozen_family") or (
        json.loads((root_dir / "frozen_family.json").read_text(encoding="utf-8"))
        if (root_dir / "frozen_family.json").is_file()
        else report.get("frozen_family")
    )
    final_eval_file = root_dir / "final_evaluation.json"
    final_eval_bundle = (
        json.loads(final_eval_file.read_text(encoding="utf-8"))
        if final_eval_file.is_file()
        else None
    )
    confirmation_file = root_dir / "confirmation.json"
    confirmation_bundle = (
        json.loads(confirmation_file.read_text(encoding="utf-8"))
        if confirmation_file.is_file()
        else None
    )
    holdout_access = db_data.get("holdout_access") or (
        json.loads((root_dir / "holdout_access.json").read_text(encoding="utf-8"))
        if (root_dir / "holdout_access.json").is_file()
        else []
    )
    confirmation_access = db_data.get("confirmation_access") or []

    protocol = context.get("protocol") or {}
    policy = context.get("policy") or {}
    primary_metric = str(protocol.get("primary_metric") or "ic")
    direction_str = str(protocol.get("direction") or "higher")
    direction = 1 if direction_str == "higher" else -1
    confirmation_kind = str(protocol.get("confirmation_kind") or "time_forward")
    nw_lag = int(policy.get("nw_lag", 5))
    n_boot = int(policy.get("bootstrap_samples", 1000))
    block = int(policy.get("bootstrap_block", 5))
    baseline_variant = str(protocol.get("baseline_variant") or "baseline:mse")

    # ------------------------------------------------------------------
    # Check 1: SHA-256 Hash Chain Verification
    # ------------------------------------------------------------------
    hash_errors: list[str] = []
    n_trial_hashes_checked = 0
    n_pred_hashes_checked = 0

    bundle_manifest_path = root_dir / "bundle_manifest.json"
    checksums_path = root_dir / "checksums.sha256"
    if bundle_manifest_path.is_file() and checksums_path.is_file():
        bman = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        actual_cs_sha = sha256_file(checksums_path)
        if bman.get("checksums_sha256") != actual_cs_sha:
            hash_errors.append(
                f"checksums.sha256 mismatch: expected {bman.get('checksums_sha256')}, got {actual_cs_sha}"
            )
        bman_copy = {k: v for k, v in bman.items() if k != "bundle_hash"}
        expected_bundle_hash = sha256_obj(bman_copy)
        if bman.get("bundle_hash") != expected_bundle_hash:
            hash_errors.append(
                f"bundle_hash mismatch: expected {bman.get('bundle_hash')}, got {expected_bundle_hash}"
            )
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            expected_sha, rel_path = line.split("  ", 1)
            f_path = root_dir / rel_path
            if not f_path.is_file():
                hash_errors.append(f"missing file in checksums.sha256: {rel_path}")
            elif sha256_file(f_path) != expected_sha:
                hash_errors.append(f"file sha256 mismatch for {rel_path}")

    if frozen_family:
        ff_copy = {k: v for k, v in frozen_family.items() if k != "manifest_hash"}
        ff_expected = sha256_obj(ff_copy)
        if frozen_family.get("manifest_hash") and frozen_family["manifest_hash"] != ff_expected:
            hash_errors.append(
                f"frozen_family manifest_hash mismatch: {frozen_family['manifest_hash']} != {ff_expected}"
            )

    frozen_manifest_path = root_dir / "frozen_manifest.json"
    if frozen_manifest_path.is_file():
        fman = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
        fman_copy = {k: v for k, v in fman.items() if k not in {"manifest_hash", "frozen_at"}}
        fman_expected = sha256_obj(fman_copy)
        if fman.get("manifest_hash") and fman["manifest_hash"] != fman_expected:
            hash_errors.append(
                f"frozen_manifest manifest_hash mismatch: {fman['manifest_hash']} != {fman_expected}"
            )

    def _resolve_pred_file(trial: dict[str, Any]) -> Path | None:
        raw_p = str(trial.get("predictions_path") or "").strip()
        if raw_p:
            p = Path(raw_p)
            if p.is_file():
                return p
            if not p.is_absolute():
                for base in (root_dir, root_dir.parent):
                    cand_p = base / p
                    if cand_p.is_file():
                        return cand_p
        tid = trial.get("trial_id", "")
        bundled = root_dir / "predictions" / f"{tid}_predictions.npz"
        if bundled.is_file():
            return bundled
        return None

    for t in trials:
        expected_rhash = recompute_trial_result_digest(t)
        stored_rhash = str(t.get("result_hash") or "")
        if stored_rhash:
            n_trial_hashes_checked += 1
            if stored_rhash != expected_rhash:
                hash_errors.append(
                    f"trial {t.get('trial_id')} result_hash mismatch: stored={stored_rhash}, recomputed={expected_rhash}"
                )
        p_file = _resolve_pred_file(t)
        stored_psha = str(t.get("predictions_sha256") or "")
        if p_file is not None and stored_psha:
            n_pred_hashes_checked += 1
            actual_psha = sha256_file(p_file)
            if actual_psha != stored_psha:
                hash_errors.append(
                    f"trial {t.get('trial_id')} predictions_sha256 mismatch: stored={stored_psha}, actual={actual_psha}"
                )
        elif stored_psha and p_file is None:
            hash_errors.append(
                f"trial {t.get('trial_id')} records predictions_sha256={stored_psha[:16]} but predictions.npz is missing"
            )

    check_hash_chain = {
        "ok": len(hash_errors) == 0,
        "n_trial_hashes_checked": n_trial_hashes_checked,
        "n_prediction_files_checked": n_pred_hashes_checked,
        "errors": hash_errors,
    }

    # ------------------------------------------------------------------
    # Check 2: Raw Prediction Tensor Re-evaluation
    # ------------------------------------------------------------------
    pred_errors: list[str] = []
    n_pred_tensors_recomputed = 0
    max_pred_metric_err = 0.0

    for t in trials:
        if t.get("status") != "success":
            continue
        p_file = _resolve_pred_file(t)
        if p_file is None:
            continue
        n_pred_tensors_recomputed += 1
        with np.load(p_file, allow_pickle=False) as npz_peek:
            keys = set(npz_peek.files)
        t_metrics = t.get("metrics") or {}
        t_series = t.get("metric_series") or {}
        if "datetime" in keys:
            recomp = recompute_tsil_npz_metrics(p_file)
            for m_key in ("ic", "rank_ic"):
                if m_key in t_metrics and math.isfinite(float(t_metrics[m_key])):
                    err = abs(float(t_metrics[m_key]) - float(recomp[m_key]))
                    max_pred_metric_err = max(max_pred_metric_err, err)
                    if err > tol:
                        pred_errors.append(
                            f"trial {t.get('trial_id')} {m_key} mismatch: stored={t_metrics[m_key]}, recomputed={recomp[m_key]} (err={err:.3e})"
                        )
            stored_ic_ser = t_series.get("ic") or {}
            for d_key, d_val in stored_ic_ser.items():
                if d_key not in recomp["ic_series"]:
                    pred_errors.append(f"trial {t.get('trial_id')} date {d_key} missing in recomputed ic_series")
                else:
                    err = abs(float(d_val) - float(recomp["ic_series"][d_key]))
                    max_pred_metric_err = max(max_pred_metric_err, err)
                    if err > tol:
                        pred_errors.append(
                            f"trial {t.get('trial_id')} date {d_key} ic mismatch: err={err:.3e}"
                        )
        elif "sample" in keys:
            recomp_tab = recompute_tabular_npz_metrics(p_file, t_metrics)
            for m_key, m_val in recomp_tab.items():
                if m_key in t_metrics and math.isfinite(float(t_metrics[m_key])):
                    err = abs(float(t_metrics[m_key]) - float(m_val))
                    max_pred_metric_err = max(max_pred_metric_err, err)
                    if err > tol:
                        pred_errors.append(
                            f"trial {t.get('trial_id')} {m_key} mismatch: stored={t_metrics[m_key]}, recomputed={m_val} (err={err:.3e})"
                        )

    check_raw_predictions = {
        "ok": len(pred_errors) == 0,
        "n_prediction_tensors_recomputed": n_pred_tensors_recomputed,
        "max_abs_error": max_pred_metric_err,
        "errors": pred_errors,
    }

    # ------------------------------------------------------------------
    # Check 3: Independent Statistical Inference Re-computation
    # ------------------------------------------------------------------
    stat_errors: list[str] = []
    n_evidence_checked = 0
    max_stat_err = 0.0

    # Index trials by (candidate_id, fold_id, seed)
    trial_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for t in trials:
        if t.get("status") == "success":
            cid = str(t.get("candidate_id") or "")
            fid = str(t.get("fold_id") or "")
            sd = int(t.get("seed") if t.get("seed") is not None else 2025)
            trial_index[(cid, fid, sd)] = t

    # Group evidence by (stage, round_id) for the per-row statistic checks.
    ev_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in evidence:
        stg = str(e.get("stage") or "discovery")
        rid = str(e.get("round_id") or "")
        ev_groups.setdefault((stg, rid), []).append(e)

    # Benjamini-Hochberg family: the engine deliberately pools every fully
    # evaluated candidate of a stage across *all* discovery rounds, so the
    # family size never resets between rounds.  Mirror that here instead of
    # recomputing q-values per round, otherwise the reconciliation fails by
    # construction once a run has more than one round.
    bh_rows: list[dict[str, Any]] = []

    for (stg, rid), ev_list in ev_groups.items():
        # Filter to rows that underwent full statistical evaluation (have p_value or folds in metrics)
        evaluated_rows: list[dict[str, Any]] = []
        for e in ev_list:
            m = e.get("metrics") or {}
            folds_dict = m.get("folds") or {}
            per_seed_dict = m.get("per_seed") or {}
            if not folds_dict or not per_seed_dict:
                continue
            evaluated_rows.append(e)
            bh_rows.append(e)
            n_evidence_checked += 1
            cid = str(e.get("candidate_id") or "")
            eval_fold_ids = sorted(folds_dict.keys())
            seeds_used = sorted(int(s) for s in per_seed_dict.keys())

            recomp_seed_effects: list[float] = []
            recomp_seed_pvals: list[float] = []
            recomp_seed_lo: list[float] = []
            recomp_seed_hi: list[float] = []

            for sd in seeds_used:
                pooled_cand: dict[str, float] = {}
                pooled_base: dict[str, float] = {}
                for fid in eval_fold_ids:
                    ct = trial_index.get((cid, fid, sd))
                    bt = trial_index.get((baseline_variant, fid, sd))
                    if ct is None or bt is None:
                        continue
                    c_ser = (ct.get("metric_series") or {}).get(primary_metric) or {}
                    b_ser = (bt.get("metric_series") or {}).get(primary_metric) or {}
                    for k, v in c_ser.items():
                        pooled_cand[str(k)] = float(v)
                    for k, v in b_ser.items():
                        pooled_base[str(k)] = float(v)

                if not pooled_cand or not pooled_base:
                    continue
                res = indep_paired_effect(
                    pooled_cand,
                    pooled_base,
                    nw_lag=nw_lag,
                    n_boot=n_boot,
                    block=block,
                    seed=sd,
                )
                stored_ps = per_seed_dict[str(sd)]
                for k_field in ("effect", "se", "ci_low", "ci_high", "p_value"):
                    sv = stored_ps.get(k_field)
                    rv = res.get(k_field)
                    if sv is not None and rv is not None and math.isfinite(float(sv)) and math.isfinite(float(rv)):
                        err = abs(float(sv) - float(rv))
                        max_stat_err = max(max_stat_err, err)
                        if err > tol:
                            stat_errors.append(
                                f"evidence {e.get('evidence_id')} seed={sd} {k_field} mismatch: stored={sv}, recomputed={rv} (err={err:.3e})"
                            )
                if math.isfinite(res["effect"]):
                    recomp_seed_effects.append(res["effect"])
                if res["n_dates"] >= 2 and math.isfinite(res["p_value"]):
                    recomp_seed_pvals.append(res["p_value"])
                if res["n_dates"] >= 2 and math.isfinite(res["ci_low"]):
                    recomp_seed_lo.append(res["ci_low"])
                if res["n_dates"] >= 2 and math.isfinite(res["ci_high"]):
                    recomp_seed_hi.append(res["ci_high"])

            if recomp_seed_effects:
                agg_eff = float(np.mean(recomp_seed_effects))
                if e.get("effect") is not None:
                    err = abs(float(e["effect"]) - agg_eff)
                    max_stat_err = max(max_stat_err, err)
                    if err > tol:
                        stat_errors.append(
                            f"evidence {e.get('evidence_id')} aggregate effect mismatch: stored={e['effect']}, recomputed={agg_eff} (err={err:.3e})"
                        )
            if recomp_seed_pvals and e.get("p_value") is not None:
                agg_p = max(recomp_seed_pvals)
                err = abs(float(e["p_value"]) - agg_p)
                max_stat_err = max(max_stat_err, err)
                if err > tol:
                    stat_errors.append(
                        f"evidence {e.get('evidence_id')} aggregate p_value mismatch: stored={e['p_value']}, recomputed={agg_p} (err={err:.3e})"
                    )
            if recomp_seed_lo and e.get("ci_low") is not None:
                agg_lo = min(recomp_seed_lo)
                err = abs(float(e["ci_low"]) - agg_lo)
                max_stat_err = max(max_stat_err, err)
                if err > tol:
                    stat_errors.append(
                        f"evidence {e.get('evidence_id')} aggregate ci_low mismatch: stored={e['ci_low']}, recomputed={agg_lo} (err={err:.3e})"
                    )
            if recomp_seed_hi and e.get("ci_high") is not None:
                agg_hi = max(recomp_seed_hi)
                err = abs(float(e["ci_high"]) - agg_hi)
                max_stat_err = max(max_stat_err, err)
                if err > tol:
                    stat_errors.append(
                        f"evidence {e.get('evidence_id')} aggregate ci_high mismatch: stored={e['ci_high']}, recomputed={agg_hi} (err={err:.3e})"
                    )

    # Verify BH q-values across the whole run-level family (all rounds pooled).
    if bh_rows:
        p_list = [r.get("p_value") for r in bh_rows]
        recomp_q = indep_benjamini_hochberg(p_list)
        for r, rq in zip(bh_rows, recomp_q):
            sq = r.get("q_value")
            if sq is not None and math.isfinite(float(sq)) and math.isfinite(rq):
                err = abs(float(sq) - rq)
                max_stat_err = max(max_stat_err, err)
                if err > tol:
                    stat_errors.append(
                        f"evidence {r.get('evidence_id')} BH q_value mismatch: stored={sq}, recomputed={rq} (err={err:.3e})"
                    )

    check_independent_stats = {
        "ok": len(stat_errors) == 0,
        "n_evidence_checked": n_evidence_checked,
        "max_abs_error": max_stat_err,
        "errors": stat_errors,
    }

    # ------------------------------------------------------------------
    # Check 4: Ledger & Freeze Chronology Verification
    # ------------------------------------------------------------------
    chron_errors: list[str] = []
    round_end_times = [str(r.get("ended_at") or "") for r in rounds if r.get("ended_at")]
    round_start_times = [str(r.get("started_at") or "") for r in rounds if r.get("started_at")]
    frozen_at = str((frozen_family or {}).get("frozen_at") or "")
    conf_claimed_times = [str(c.get("claimed_at") or "") for c in confirmation_access if c.get("claimed_at")]
    holdout_claimed_times = [str(h.get("claimed_at") or "") for h in holdout_access if h.get("claimed_at")]

    if frozen_at:
        for r_end in round_end_times:
            if r_end > frozen_at:
                chron_errors.append(
                    f"discovery round ended_at ({r_end}) is later than frozen_family.frozen_at ({frozen_at})"
                )
        for r_start in round_start_times:
            if r_start > frozen_at:
                chron_errors.append(
                    f"discovery round started_at ({r_start}) is later than frozen_family.frozen_at ({frozen_at})"
                )
        for c_time in conf_claimed_times:
            if c_time < frozen_at:
                chron_errors.append(
                    f"confirmation claimed_at ({c_time}) precedes frozen_family.frozen_at ({frozen_at})"
                )
        for h_time in holdout_claimed_times:
            if h_time < frozen_at:
                chron_errors.append(
                    f"holdout claimed_at ({h_time}) precedes frozen_family.frozen_at ({frozen_at})"
                )

    if conf_claimed_times and holdout_claimed_times:
        if min(holdout_claimed_times) < min(conf_claimed_times):
            chron_errors.append(
                f"holdout claimed_at ({min(holdout_claimed_times)}) precedes confirmation claimed_at ({min(conf_claimed_times)})"
            )

    if (confirmation_access or confirmation_bundle) and not frozen_family:
        chron_errors.append("confirmation executed without a frozen_family manifest")
    if (holdout_access or final_eval_bundle) and not frozen_family:
        chron_errors.append("holdout finalization executed without a frozen_family manifest")

    check_ledger_chronology = {
        "ok": len(chron_errors) == 0,
        "max_round_ended_at": max(round_end_times) if round_end_times else None,
        "frozen_at": frozen_at or None,
        "min_confirmation_claimed_at": min(conf_claimed_times) if conf_claimed_times else None,
        "min_holdout_claimed_at": min(holdout_claimed_times) if holdout_claimed_times else None,
        "errors": chron_errors,
    }

    # ------------------------------------------------------------------
    # Check 5: Scientific State Legitimacy Verification
    # ------------------------------------------------------------------
    state_errors: list[str] = []
    n_indep_confirmed = 0

    for e in evidence:
        st = str(e.get("state") or "")
        stg = str(e.get("stage") or "")
        if stg == "discovery" and st == "independently_confirmed":
            state_errors.append(
                f"evidence {e.get('evidence_id')} in discovery stage illegally claims independently_confirmed"
            )
        if st == "independently_confirmed":
            n_indep_confirmed += 1
            if confirmation_kind != "time_forward":
                state_errors.append(
                    f"evidence {e.get('evidence_id')} claims independently_confirmed under confirmation_kind={confirmation_kind!r}"
                )
            ci_lo = e.get("ci_low")
            ci_hi = e.get("ci_high")
            if ci_lo is None or ci_hi is None:
                state_errors.append(
                    f"evidence {e.get('evidence_id')} claims independently_confirmed without finite 95% CI"
                )
            else:
                dir_ci_low = float(ci_lo) if direction == 1 else -float(ci_hi)
                if dir_ci_low <= 0.0:
                    state_errors.append(
                        f"evidence {e.get('evidence_id')} claims independently_confirmed with directional ci_low={dir_ci_low:.6f} <= 0"
                    )
            m = e.get("metrics") or {}
            if m.get("control_variant") or m.get("controls"):
                if m.get("beat_control") is False:
                    state_errors.append(
                        f"evidence {e.get('evidence_id')} claims independently_confirmed despite beat_control=False"
                    )

    if final_eval_bundle and isinstance(final_eval_bundle.get("result"), dict):
        f_res = final_eval_bundle["result"]
        if f_res.get("state") == "independently_confirmed":
            n_indep_confirmed += 1
            if final_eval_bundle.get("exploratory"):
                state_errors.append("final_evaluation is exploratory=True but claims independently_confirmed")
            if confirmation_kind != "time_forward":
                state_errors.append(
                    f"final_evaluation claims independently_confirmed under confirmation_kind={confirmation_kind!r}"
                )
            ci_lo = f_res.get("ci_low")
            ci_hi = f_res.get("ci_high")
            if ci_lo is None or ci_hi is None:
                state_errors.append("final_evaluation claims independently_confirmed without finite 95% CI")
            else:
                dir_ci_low = float(ci_lo) if direction == 1 else -float(ci_hi)
                if dir_ci_low <= 0.0:
                    state_errors.append(
                        f"final_evaluation claims independently_confirmed with directional ci_low={dir_ci_low:.6f} <= 0"
                    )

    check_state_legitimacy = {
        "ok": len(state_errors) == 0,
        "n_independently_confirmed_checked": n_indep_confirmed,
        "confirmation_kind": confirmation_kind,
        "errors": state_errors,
    }

    overall_ok = (
        check_hash_chain["ok"]
        and check_raw_predictions["ok"]
        and check_independent_stats["ok"]
        and check_ledger_chronology["ok"]
        and check_state_legitimacy["ok"]
    )

    return {
        "ok": overall_ok,
        "zero_predicatelab_imports": not any(m.startswith("predicatelab") for m in sys.modules),
        "tolerance": tol,
        "checks": {
            "hash_chain": check_hash_chain,
            "raw_predictions": check_raw_predictions,
            "independent_stats": check_independent_stats,
            "ledger_chronology": check_ledger_chronology,
            "state_legitimacy": check_state_legitimacy,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-dependency independent evidence verifier for PredicateLab"
    )
    parser.add_argument("--bundle", type=Path, default=None, help="Path to exported bundle directory")
    parser.add_argument("--artifacts", type=Path, default=None, help="Path to artifacts directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite database")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id")
    parser.add_argument("--tol", type=float, default=1e-6, help="Numerical tolerance (default 1e-6)")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args(argv)

    if args.bundle is None and args.artifacts is None:
        parser.error("Provide --bundle <bundle_dir> or --artifacts <artifacts_dir>")

    report = verify_all(
        bundle_dir=args.bundle,
        artifacts_dir=args.artifacts,
        db_path=args.db,
        run_id=args.run_id,
        tol=args.tol,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
