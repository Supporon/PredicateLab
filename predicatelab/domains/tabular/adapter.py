"""Generic tabular adapter — the architecture-validation scenario.

This adapter is deliberately independent from TSIL: scikit-learn data, a
logistic-regression pipeline and stratified cross-validation.  It exists to
prove that the core engine (candidate identity, probing, screening, paired
statistics, evidence, feedback, holdout freeze) is not TSIL-specific.

All fitted quantities (thresholds, standardisation references) are computed on
the training part of the fold where the candidate is evaluated.  The 20%
holdout split is stored once and is never used by the search path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from predicatelab.core.adapter import CandidateGenerator, DomainAdapter, TrialRequest
from predicatelab.core.data import assess_common
from predicatelab.core.hashing import sha256_obj
from predicatelab.core.schema import (
    DataContract,
    DataSnapshot,
    EvaluationSpec,
    ExperimentProtocol,
    FoldSpec,
)
from predicatelab.domains.tabular.generator import TabularRuleGenerator

OPS = {"ratio", "product", "difference", "sigmoid", "threshold", "interaction"}
ALLOWED_KEYS = {"op", "a", "b", "quantile", "scale", "note"}


class DescriptorError(ValueError):
    pass


def validate_tabular_descriptor(desc: dict[str, Any], n_features: int) -> None:
    if not isinstance(desc, dict):
        raise DescriptorError("descriptor must be an object")
    unknown = set(desc) - ALLOWED_KEYS
    if unknown:
        raise DescriptorError(f"unknown keys: {sorted(unknown)}")
    op = desc.get("op")
    if op not in OPS:
        raise DescriptorError(f"unknown op {op!r}")
    a = desc.get("a")
    if not isinstance(a, int) or isinstance(a, bool) or not (0 <= a < n_features):
        raise DescriptorError(f"feature index a={a!r} outside [0,{n_features})")
    if op in {"ratio", "product", "difference", "interaction"}:
        b = desc.get("b")
        if not isinstance(b, int) or isinstance(b, bool) or not (0 <= b < n_features) or b == a:
            raise DescriptorError(f"feature index b={b!r} invalid")
    if op == "threshold":
        q = desc.get("quantile")
        if not isinstance(q, (int, float)) or not (0.0 < float(q) < 1.0):
            raise DescriptorError("quantile must be in (0,1)")
    if op == "sigmoid":
        s = desc.get("scale", 1.0)
        if not isinstance(s, (int, float)) or not (0.05 <= float(s) <= 20.0):
            raise DescriptorError("scale must be in [0.05,20]")


def fit_transform(desc: dict[str, Any], X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit predicate parameters on ``X_train`` only and transform both sets."""
    op = desc["op"]
    a = int(desc["a"])
    params: dict[str, Any] = {"op": op, "a": a}
    if op == "ratio":
        b = int(desc["b"])
        params["b"] = b
        scale = np.std(X_train[:, b]) + 1e-8
        train = X_train[:, a] / (np.abs(X_train[:, b]) + 0.1 * scale)
        other = X_other[:, a] / (np.abs(X_other[:, b]) + 0.1 * scale)
        params["denominator_scale"] = float(scale)
    elif op == "product":
        b = int(desc["b"])
        params["b"] = b
        train = X_train[:, a] * X_train[:, b]
        other = X_other[:, a] * X_other[:, b]
    elif op == "difference":
        b = int(desc["b"])
        params["b"] = b
        train = X_train[:, a] - X_train[:, b]
        other = X_other[:, a] - X_other[:, b]
    elif op == "sigmoid":
        median = float(np.median(X_train[:, a]))
        scale = float(np.quantile(np.abs(X_train[:, a] - median), 0.75) + 1e-6)
        k = float(desc.get("scale", 1.0))
        params.update({"median": median, "iqr": scale, "scale": k})
        train = 1.0 / (1.0 + np.exp(-k * (X_train[:, a] - median) / scale))
        other = 1.0 / (1.0 + np.exp(-k * (X_other[:, a] - median) / scale))
    elif op == "threshold":
        q = float(desc["quantile"])
        thr = float(np.quantile(X_train[:, a], q))
        params.update({"quantile": q, "threshold": thr})
        train = (X_train[:, a] >= thr).astype(float)
        other = (X_other[:, a] >= thr).astype(float)
    elif op == "interaction":
        b = int(desc["b"])
        params["b"] = b
        ma, mb = X_train[:, a].mean(), X_train[:, b].mean()
        sa, sb = X_train[:, a].std() + 1e-8, X_train[:, b].std() + 1e-8
        params.update({"mean_a": float(ma), "mean_b": float(mb), "std_a": float(sa), "std_b": float(sb)})
        za = (X_train[:, a] - ma) / sa
        zb = (X_train[:, b] - mb) / sb
        oa = (X_other[:, a] - ma) / sa
        ob = (X_other[:, b] - mb) / sb
        train = np.sign(za * zb)
        other = np.sign(oa * ob)
    else:  # pragma: no cover - validated earlier
        raise DescriptorError(op)
    return train.reshape(-1, 1), other.reshape(-1, 1), params


class TabularAdapter(DomainAdapter):
    name = "tabular"

    def __init__(self, config: dict[str, Any]):
        from sklearn.datasets import load_breast_cancer

        self.config = config
        self.data_cfg = config.get("data", {})
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.feature_names = list(load_breast_cancer().feature_names)
        self.n_features = int(self.X.shape[1])
        self.seed = int(config.get("training", {}).get("seed", 2025))
        self._evaluation = EvaluationSpec.model_validate(
            config.get("evaluation", {
                "name": "tabular_cv_auc",
                "primary_metric": "auc",
                "direction": "higher",
                "secondary_metrics": ["accuracy", "log_loss"],
                "paired": True,
                "multiple_testing": "benjamini-hochberg",
                "notes": "repeated stratified 5-fold cross-validation on the 80% search split",
            })
        )
        self._folds, self._holdout = self._build_folds()

    # ------------------------------------------------------------------ data
    def _split(self) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(self.y))
        pool, hold = train_test_split(
            idx, test_size=0.2, random_state=self.seed, stratify=self.y
        )
        return np.sort(pool), np.sort(hold)

    def _build_folds(self) -> tuple[list[FoldSpec], FoldSpec]:
        from sklearn.model_selection import StratifiedKFold

        pool, hold = self._split()
        folds: list[FoldSpec] = []
        n_splits = int(self.config.get("protocol", {}).get("cv_splits", 5))
        repeats = int(self.config.get("protocol", {}).get("cv_repeats", 2))
        for r in range(repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed + r)
            for k, (tr, va) in enumerate(skf.split(pool, self.y[pool])):
                train_idx = pool[tr].tolist()
                valid_idx = pool[va].tolist()
                folds.append(FoldSpec(
                    fold_id=f"cv_r{r}_f{k}", kind="cv", role="search",
                    metadata={"train_idx": train_idx, "valid_idx": valid_idx,
                              "repeat": r, "fold": k,
                              "cv_key": f"r{r}_f{k}"},
                    notes="stratified CV fold on the 80% search split",
                ))
        holdout = FoldSpec(
            fold_id="holdout_20pct", kind="cv", role="holdout",
            metadata={"train_idx": pool.tolist(), "valid_idx": hold.tolist(),
                      "test_idx": hold.tolist(), "cv_key": "holdout"},
            notes="fixed stratified 20% holdout, frozen before finalize",
        )
        return folds, holdout

    def data_contract(self, config: dict[str, Any]) -> DataContract:
        return DataContract(
            dataset_id="sklearn-breast-cancer",
            domain="tabular",
            description="Breast Cancer Wisconsin (diagnostic) tabular classification dataset.",
            entity_key="sample",
            time_key="",
            label_definition="malignant (1) vs benign (0)",
            label_window={"type": "static", "leakage_risk": "train/test split by sample"},
            feature_semantics={name: "measurement" for name in self.feature_names},
            universe="569 samples × 30 features",
            frequency="static",
            time_range=("", ""),
            caveats=[
                "Static dataset: no temporal ordering; repeated stratified CV is the validation protocol.",
                "Fixed 20% holdout is separated before any candidate search.",
            ],
        )

    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        details = {
            "n_samples": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "class_counts": {str(k): int(v) for k, v in zip(*np.unique(self.y, return_counts=True))},
            "X_sha256": sha256_obj(np.round(self.X, 10).tolist()),
            "y_sha256": sha256_obj(self.y.tolist()),
            "feature_names": self.feature_names,
            "holdout_idx": self._holdout.metadata["valid_idx"],
        }
        return DataSnapshot(
            dataset_id="sklearn-breast-cancer",
            domain="tabular",
            manifest_method="sha256 of rounded feature matrix and labels",
            manifest_hash=sha256_obj(details),
            path="sklearn.datasets.load_breast_cancer",
            details=details,
        )

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        counts = np.bincount(self.y.astype(int))
        return assess_common({
            "sample_count": {"ok": len(self.y) >= 200, "severity": "warn",
                             "detail": f"{len(self.y)} samples"},
            "class_balance": {"ok": counts.min() >= 30, "severity": "warn",
                              "detail": f"counts={counts.tolist()}"},
            "missing_values": {"ok": bool(np.isfinite(self.X).all()), "severity": "block",
                               "detail": "non-finite features"},
            "feature_count": {"ok": self.n_features >= 5, "severity": "warn",
                              "detail": f"{self.n_features} features"},
        })

    # -------------------------------------------------------------- protocol
    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        return ExperimentProtocol(
            task_id="task-tabular-breast-cancer",
            domain="tabular",
            primary_metric=self._evaluation.primary_metric,
            direction=self._evaluation.direction,
            folds=self._folds,
            holdout_fold=self._holdout,
            baseline_variant="baseline:logreg",
            control_variant=None,
            seeds=[self.seed],
            model_spec={"model": "LogisticRegression", "C": 1.0},
            data_config={"dataset": "sklearn-breast-cancer", "cv_splits": 5, "cv_repeats": 2},
            evaluation=self._evaluation,
        )

    @property
    def evaluation(self) -> EvaluationSpec:
        return self._evaluation

    # ------------------------------------------------------------ candidates
    def validate_descriptor(self, kind: str, descriptor: dict[str, Any]) -> None:
        validate_tabular_descriptor(descriptor, self.n_features)
        op = descriptor["op"]
        expected = {
            "ratio": {"statistical"},
            "product": {"statistical"},
            "difference": {"statistical"},
            "sigmoid": {"soft"},
            "threshold": {"conditional"},
            "interaction": {"relational"},
        }[op]
        if kind not in expected:
            raise DescriptorError(f"op {op} must have kind in {sorted(expected)}")

    def make_generator(self, config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
        return TabularRuleGenerator(config, self.feature_names, self.n_features)

    # ----------------------------------------------------------------- probe
    def probe(self, descriptor: dict[str, Any], fold: FoldSpec, snapshot: DataSnapshot,
              config: dict[str, Any], workdir: str) -> dict[str, Any]:
        try:
            tr = np.asarray(fold.metadata["train_idx"])
            Xtr = self.X[tr]
            col_tr, _, params = fit_transform(descriptor, Xtr, Xtr[:1])
            v = col_tr[:, 0]
            if not np.isfinite(v).all():
                return {"ok": False, "reason": "non-finite predicate values", "severity": "invalid"}
            if float(np.std(v)) < 1e-12:
                return {"ok": False, "reason": "constant predicate on training fold", "severity": "invalid"}
            corr = [float(np.corrcoef(v, Xtr[:, j])[0, 1]) if np.std(Xtr[:, j]) > 0 else 0.0
                    for j in range(min(6, self.n_features))]
            sig = [float(np.mean(v)), float(np.std(v)),
                   float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-12)),
                   *[0.0 if not math.isfinite(c) else c for c in corr]]
            return {"ok": True, "signature": sig, "checks": {"std": float(np.std(v)),
                    "fitted": params}, "reason": "ok"}
        except Exception as exc:
            return {"ok": False, "reason": f"probe failed: {type(exc).__name__}: {exc}", "severity": "invalid"}

    # ---------------------------------------------------------------- trials
    def _predict(self, desc: dict[str, Any] | None, train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        Xtr, ytr = self.X[train_idx], self.y[train_idx]
        Xva, yva = self.X[valid_idx], self.y[valid_idx]
        fitted: dict[str, Any] = {}
        if desc is not None:
            col_tr, col_va, fitted = fit_transform(desc, Xtr, Xva)
            Xtr = np.hstack([Xtr, col_tr])
            Xva = np.hstack([Xva, col_va])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, random_state=seed, solver="lbfgs"),
        )
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xva)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = {
            "auc": float(roc_auc_score(yva, prob)),
            "accuracy": float(accuracy_score(yva, pred)),
            "log_loss": float(log_loss(yva, prob)),
        }
        return metrics, fitted

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        import time

        t0 = time.time()
        meta = request.fold.metadata
        train_idx = np.asarray(meta["train_idx"], dtype=int)
        valid_idx = np.asarray(meta["valid_idx"], dtype=int)
        if request.variant == "candidate":
            metrics, fitted = self._predict(request.descriptor, train_idx, valid_idx, request.seed)
        else:
            metrics, fitted = self._predict(None, train_idx, valid_idx, request.seed)
        key = str(meta.get("cv_key", request.fold.fold_id))
        return {
            "status": "success",
            "metrics": metrics,
            "metric_series": {"auc": {key: metrics["auc"]},
                              "log_loss": {key: metrics["log_loss"]},
                              "accuracy": {key: metrics["accuracy"]}},
            "fitted_params": fitted,
            "fit_data_hash": sha256_obj({"train_idx": train_idx.tolist(),
                                         "dataset": "sklearn-breast-cancer"}),
            "predictions_path": "",
            "duration_s": time.time() - t0,
            "error": "",
            "worker": {"variant": request.variant, "fold": request.fold.fold_id, "in_process": True},
        }
