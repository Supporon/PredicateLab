"""TSIL domain adapter.

Wraps the upstream TSIL research code without changing its mathematics:

* data: qlib Alpha158 + ``MTSDatasetH`` from ``TSIL-main/src``;
* model: ``QniverseModel`` (LSTM by default) with the upstream
  ``WeightedMSELoss``;
* predicate injection: ``QniverseModel(phi_fn=compiled_descriptor)``, the hook
  added to the upstream loss;
* folds: expanding-window walk-forward with purge and embargo;
* holdout: 2024-01-01 .. 2025-08-31, reachable only through ``finalize``.

Each trial runs in a fresh subprocess (``worker.py``) so CUDA state, qlib and
crashes cannot contaminate the scheduler.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from predicatelab.core.adapter import CandidateGenerator, DomainAdapter, TrialRequest
from predicatelab.core.data import assess_common
from predicatelab.core.hashing import sha256_file, sha256_obj
from predicatelab.core.protocol import effective_time_fold
from predicatelab.core.schema import (
    DataContract,
    DataSnapshot,
    EvaluationSpec,
    ExperimentProtocol,
    FoldSpec,
)
from predicatelab.domains.tsil import dsl
from predicatelab.domains.tsil.generator import make_generator as _make_generator

REPO_ROOT = Path(__file__).resolve().parents[3]


class TSILAdapter(DomainAdapter):
    name = "tsil"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.provider_uri = Path(self.data_cfg["provider_uri"])
        self.tsil_root = Path(self.data_cfg["tsil_root"])
        self.seq_len = int(self.data_cfg.get("seq_len", 20))
        self.feature_dim = int(self.data_cfg.get("feature_dim", 158))
        self.market = self.data_cfg.get("market", "csi300")
        self.label = self.data_cfg.get("label", "Ref($close, -2) / Ref($close, -1) - 1")
        self._calendar = self._read_calendar()
        self._evaluation = EvaluationSpec.model_validate(
            config.get("evaluation", {
                "name": "tsil_rolling_ic",
                "primary_metric": "ic",
                "direction": "higher",
                "secondary_metrics": ["rank_ic", "icir", "mse"],
                "paired": True,
                "multiple_testing": "benjamini-hochberg",
                "notes": "daily cross-sectional IC on rolling validation folds",
            })
        )

    # ------------------------------------------------------------------ data
    def _read_calendar(self) -> list[str]:
        path = self.provider_uri / "calendars" / "day.txt"
        if not path.exists():
            raise FileNotFoundError(f"calendar not found: {path}")
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def data_contract(self, config: dict[str, Any]) -> DataContract:
        return DataContract(
            dataset_id=f"qlib-cn-{self.market}",
            domain="tsil",
            description="China A-share daily Alpha158 panel from the shared qlib store.",
            entity_key="instrument",
            time_key="datetime",
            label_definition=self.label,
            label_window={"horizon": int(self.data_cfg.get("horizon", 1)), "unit": "trading_day",
                          "uses_future_days": 2,
                          "note": "Ref($close,-2)/Ref($close,-1)-1 needs t+1 and t+2 prices"},
            feature_semantics={"alpha158": "158 qlib Alpha158 price-volume features"},
            universe=self.market,
            frequency="daily",
            time_range=(self._calendar[0], self._calendar[-1]),
            caveats=[
                "Survivorship/instrument membership is whatever the supplied qlib store contains.",
                "RobustZScoreNorm is fitted per fold on that fold's training period only.",
                "Adjusted prices are taken as-is from the provided qlib data.",
            ],
        )

    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        cal = self.provider_uri / "calendars" / "day.txt"
        inst = self.provider_uri / "instruments" / f"{self.market}.txt"
        feature_files = sorted((self.provider_uri / "features").glob("*/*.bin"))
        total_size = 0
        max_mtime = 0
        for f in feature_files:
            st = f.stat()
            total_size += st.st_size
            max_mtime = max(max_mtime, st.st_mtime_ns)
        sample = feature_files[:4] + feature_files[-4:] if len(feature_files) > 8 else feature_files
        sample_hashes = {str(f.relative_to(self.provider_uri)): sha256_file(f) for f in sample}
        details = {
            "calendar_sha256": sha256_file(cal),
            "instruments_sha256": sha256_file(inst),
            "n_feature_files": len(feature_files),
            "feature_total_size": total_size,
            "feature_max_mtime_ns": max_mtime,
            "sampled_feature_sha256": sample_hashes,
            "semantics": {
                "market": self.market,
                "label": self.label,
                "seq_len": self.seq_len,
                "feature_dim": self.feature_dim,
                "horizon": int(self.data_cfg.get("horizon", 1)),
            },
        }
        manifest_hash = sha256_obj(details)
        return DataSnapshot(
            dataset_id=f"qlib-cn-{self.market}",
            domain="tsil",
            manifest_method="calendar+instruments sha256 + feature-store metadata + 8 sampled file hashes",
            manifest_hash=manifest_hash,
            path=str(self.provider_uri),
            details=details,
        )

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        cal_start, cal_end = self._calendar[0], self._calendar[-1]
        protocol = self.build_protocol(config)
        folds = list(protocol.folds) + ([protocol.holdout_fold] if protocol.holdout_fold else [])
        coverage = {
            "ok": all(f.train_start >= cal_start and (f.test_end or f.valid_end) <= cal_end for f in folds),
            "severity": "block",
            "detail": f"calendar {cal_start}..{cal_end}",
        }
        labels = {"ok": bool(self.label), "severity": "block", "detail": self.label or "missing label"}
        inst_path = self.provider_uri / "instruments" / f"{self.market}.txt"
        n_inst = sum(1 for _ in inst_path.open()) if inst_path.exists() else 0
        entities = {"ok": n_inst > 0, "severity": "block", "detail": f"{n_inst} instrument rows"}
        purge_ok = all(f.purge_days >= 2 for f in protocol.folds)
        purge = {"ok": purge_ok, "severity": "warn",
                 "detail": "purge_days must be ≥ label future days (2)"}
        features = (self.provider_uri / "features").exists()
        feat = {"ok": features, "severity": "block", "detail": str(self.provider_uri / "features")}
        horizon = {"ok": int(self.data_cfg.get("horizon", 1)) == 1, "severity": "warn",
                   "detail": "MTSDatasetH masks the last `horizon` label columns"}
        return assess_common({
            "calendar_coverage": coverage,
            "label_definition": labels,
            "entities": entities,
            "purge_vs_label_window": purge,
            "feature_store": feat,
            "horizon": horizon,
        })

    # -------------------------------------------------------------- protocol
    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        p = config.get("protocol", {})
        purge = int(p.get("purge_days", 2))
        embargo = int(p.get("embargo_days", 3))
        folds = []
        for row in p.get("folds", []):
            f = FoldSpec(
                fold_id=row["id"],
                train_start=row["train"][0], train_end=row["train"][1],
                valid_start=row["valid"][0], valid_end=row["valid"][1],
                purge_days=int(row.get("purge_days", purge)),
                embargo_days=int(row.get("embargo_days", embargo)),
            )
            folds.append(effective_time_fold(f, self._calendar))
        holdout = None
        if p.get("holdout"):
            row = p["holdout"]
            h = FoldSpec(
                fold_id=row["id"],
                train_start=row["train"][0], train_end=row["train"][1],
                valid_start=row["valid"][0], valid_end=row["valid"][1],
                test_start=row["test"][0], test_end=row["test"][1],
                role="holdout",
                purge_days=int(row.get("purge_days", purge)),
                embargo_days=int(row.get("embargo_days", embargo)),
            )
            holdout = effective_time_fold(h, self._calendar)
            holdout = holdout.model_copy(update={"test_start": h.test_start, "test_end": h.test_end})
        return ExperimentProtocol(
            task_id=f"task-tsil-{self.market}",
            domain="tsil",
            primary_metric=self._evaluation.primary_metric,
            direction=self._evaluation.direction,
            folds=folds,
            holdout_fold=holdout,
            baseline_variant="baseline:mse",
            control_variant="control:ones",
            seeds=[int(x) for x in config.get("seeds", [config.get("training", {}).get("seed", 2025)])],
            model_spec={"model_type": config.get("training", {}).get("model_type", "LSTM"),
                          "model_config": config.get("model", {})},
            data_config={"provider_uri": str(self.provider_uri), "market": self.market,
                         "seq_len": self.seq_len, "horizon": self.data_cfg.get("horizon", 1)},
            evaluation=self._evaluation,
        )

    @property
    def evaluation(self) -> EvaluationSpec:
        return self._evaluation

    # ------------------------------------------------------------ candidates
    def validate_descriptor(self, kind: str, descriptor: dict[str, Any]) -> None:
        dsl.validate_descriptor(descriptor, seq_len=self.seq_len, feature_dim=self.feature_dim)
        family = descriptor.get("family")
        if family == "native" and kind != "native":
            raise dsl.DescriptorError("native descriptors must have kind='native'")
        if family == "composite" and kind != "composite":
            raise dsl.DescriptorError("composite descriptors must have kind='composite'")
        if family in dsl.OP_PARAMS and kind not in {"soft", "statistical"}:
            raise dsl.DescriptorError(f"DSL family {family} must be kind 'soft' or 'statistical'")

    def make_generator(self, config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
        return _make_generator(config, policy)

    # ----------------------------------------------------------------- probe
    def _probe_batch_path(self, snapshot: DataSnapshot, fold: FoldSpec) -> Path:
        root = Path(self.config.get("run", {}).get("artifacts_dir", "artifacts")) / "_probe_batches"
        return root / snapshot.snapshot_id / f"{fold.fold_id}.npz"

    def _ensure_probe_batch(self, snapshot: DataSnapshot, fold: FoldSpec) -> Path:
        path = self._probe_batch_path(snapshot, fold)
        if path.exists():
            return path
        job = {
            "mode": "probe_batch",
            "provider_uri": str(self.provider_uri),
            "tsil_root": str(self.tsil_root),
            "data": self._data_payload(),
            "fold": fold.model_dump(mode="json"),
            "variant": "probe",
            "descriptor": {},
            "seed": 2025,
            "gpu": None,
            "calendar_path": str(self.provider_uri / "calendars" / "day.txt"),
            "output_path": str(path),
            "result_path": str(path.with_suffix(".result.json")),
        }
        self._run_worker(job, workdir=path.parent, timeout=int(self.config.get("execution", {}).get("probe_timeout_s", 900)))
        if not path.exists():
            raise RuntimeError(f"probe batch was not produced: {path}")
        return path

    def probe(self, descriptor: dict[str, Any], fold: FoldSpec, snapshot: DataSnapshot,
              config: dict[str, Any], workdir: str) -> dict[str, Any]:
        import torch

        path = self._ensure_probe_batch(snapshot, fold)
        payload = np.load(path, allow_pickle=False)
        x = torch.from_numpy(payload["x"]).float()
        ts = torch.from_numpy(payload["timestamp"]).float() if "timestamp" in payload else None
        # CPU probe: deterministic and does not consume the shared GPUs
        if x.shape[0] < 4:
            return {"ok": False, "reason": "probe batch too small", "severity": "invalid"}
        if dsl.needs_timestamp(descriptor) and ts is None:
            return {"ok": False, "reason": "descriptor requires timestamp features but probe batch has none",
                    "severity": "invalid"}
        try:
            fn = dsl.compile_descriptor(descriptor, tsil_root=str(self.tsil_root),
                                        seq_len=self.seq_len, feature_dim=self.feature_dim)
            with torch.no_grad():
                i_mat, p_mat = fn(x, ts)
        except Exception as exc:
            return {"ok": False, "reason": f"compile/run failed: {type(exc).__name__}: {exc}",
                    "severity": "invalid"}
        checks: dict[str, Any] = {}
        finite = bool(torch.isfinite(p_mat).all().item() and torch.isfinite(i_mat).all().item())
        checks["finite"] = finite
        shape_ok = tuple(p_mat.shape) == (x.shape[0], x.shape[0])
        checks["shape"] = shape_ok
        sym = bool(torch.allclose(p_mat, p_mat.T, atol=1e-4))
        checks["symmetric"] = sym
        off = p_mat - torch.diag(torch.diag(p_mat))
        off_std = float(off.std().item()) if off.numel() > 1 else 0.0
        checks["offdiagonal_std"] = off_std
        degenerate = off_std < 1e-9
        checks["non_degenerate"] = not degenerate
        if not finite:
            return {"ok": False, "reason": "predicate produced non-finite values", "severity": "invalid", "checks": checks}
        if not shape_ok or not sym:
            return {"ok": False, "reason": "predicate produced a malformed P matrix", "severity": "invalid", "checks": checks}
        if degenerate:
            return {"ok": False, "reason": "degenerate predicate: all samples equally similar", "severity": "invalid", "checks": checks}
        # behavioural signature for dedup
        sig = dsl.probe_signature(p_mat)
        return {"ok": True, "checks": checks, "signature": sig,
                "reason": "ok" if not torch.allclose(p_mat, torch.eye(p_mat.shape[0]), atol=1e-3)
                else "ok (P close to identity)"}

    # ---------------------------------------------------------------- trials
    def _data_payload(self) -> dict[str, Any]:
        return {
            "provider_uri": str(self.provider_uri),
            "tsil_root": str(self.tsil_root),
            "market": self.market,
            "label": self.label,
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "horizon": int(self.data_cfg.get("horizon", 1)),
            "calendar_path": str(self.provider_uri / "calendars" / "day.txt"),
        }

    def _run_worker(self, job: dict[str, Any], workdir: Path, timeout: int) -> dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)
        job_path = workdir / "job.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log_path = workdir / "worker.log"
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    [sys.executable, "-m", "predicatelab.domains.tsil.worker", str(job_path)],
                    cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            return {"status": "failed", "error": f"worker timed out after {timeout}s", "metrics": {},
                    "metric_series": {}, "duration_s": float(timeout)}
        result_path = Path(job["result_path"])
        if not result_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:] if log_path.exists() else ""
            return {"status": "failed", "error": f"worker produced no result (rc={proc.returncode})\n{tail}",
                    "metrics": {}, "metric_series": {}, "duration_s": 0.0}
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.setdefault("worker", {})
        result["worker"].update({"returncode": proc.returncode, "job": str(job_path), "log": str(log_path)})
        return result

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        variant = {"baseline": "baseline", "control": "control"}.get(request.variant, "candidate")
        job = {
            "mode": "trial",
            "provider_uri": str(self.provider_uri),
            "tsil_root": str(self.tsil_root),
            "data": self._data_payload(),
            "fold": request.fold.model_dump(mode="json"),
            "variant": variant,
            "descriptor": request.descriptor if variant == "candidate" else {},
            "seed": request.seed,
            "training": request.training_config,
            "model_config": dict(self.config.get("model", {})),
            "save_predictions": bool(request.save_predictions),
            "output_dir": request.workdir,
            "gpu": self.config.get("execution", {}).get("gpu", "0"),
            "calendar_path": str(self.provider_uri / "calendars" / "day.txt"),
            "result_path": str(Path(request.workdir) / "result.json"),
        }
        timeout = int(self.config.get("execution", {}).get("trial_timeout_s", 3600))
        return self._run_worker(job, Path(request.workdir), timeout=timeout)
