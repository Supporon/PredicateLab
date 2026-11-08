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
        self.config = config
        self.data_cfg = config.get("data", {})
        self.seed = int(config.get("training", {}).get("seed", 2025))
        self.external_path: Path | None = (
            Path(self.data_cfg["path"]) if self.data_cfg.get("path") else None
        )
        self._is_external = self.external_path is not None
        self.time_col: str | None = self.data_cfg.get("time_col")
        self.group_col: str | None = self.data_cfg.get("group_col")
        self.target_col: str | None = self.data_cfg.get("target_col")
        self.time_values: np.ndarray | None = None
        self.group_values: np.ndarray | None = None
        self.file_sha256: str = ""

        if not self._is_external:
            from sklearn.datasets import load_breast_cancer

            self.dataset_id = "sklearn-breast-cancer"
            self.task_id = "task-tabular-breast-cancer"
            self.X, self.y = load_breast_cancer(return_X_y=True)
            self.feature_names = list(load_breast_cancer().feature_names)
            self.n_features = int(self.X.shape[1])
            self.task_type = "classification"
            default_eval = {
                "name": "tabular_cv_logloss",
                "primary_metric": "auc",
                "direction": "higher",
                "secondary_metrics": ["accuracy", "log_loss"],
                "paired": True,
                "multiple_testing": "benjamini-hochberg",
                "notes": "repeated stratified 5-fold cross-validation on the 80% search split",
            }
        else:
            import pandas as pd
            from predicatelab.core.hashing import sha256_file

            assert self.external_path is not None
            if not self.external_path.exists():
                raise FileNotFoundError(f"tabular data path not found: {self.external_path}")
            if not self.target_col:
                raise ValueError("external tabular dataset requires data.target_col")
            self.file_sha256 = sha256_file(self.external_path)
            suffix = self.external_path.suffix.lower()
            if suffix in {".parquet", ".pq"}:
                df = pd.read_parquet(self.external_path)
            else:
                df = pd.read_csv(self.external_path)
            if self.target_col not in df.columns:
                raise ValueError(f"target_col {self.target_col!r} not in columns {list(df.columns)}")
            order_col = self.time_col or self.group_col
            if order_col:
                if order_col not in df.columns:
                    raise ValueError(f"order column {order_col!r} not in columns {list(df.columns)}")
                df = df.sort_values(order_col, kind="mergesort").reset_index(drop=True)
            exclude = {self.target_col, self.time_col, self.group_col} - {None}
            feat_cols = self.data_cfg.get("feature_cols") or [
                c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(feat_cols) < 2:
                raise ValueError("external tabular dataset needs at least 2 numeric feature columns")
            self.feature_names = [str(c) for c in feat_cols]
            self.X = np.asarray(df[self.feature_names].to_numpy(dtype=float), dtype=float)
            self.y = np.asarray(df[self.target_col].to_numpy(dtype=float), dtype=float)
            self.n_features = int(self.X.shape[1])
            if self.time_col:
                self.time_values = np.asarray([str(v) for v in df[self.time_col].tolist()], dtype=object)
            if self.group_col:
                self.group_values = np.asarray([str(v) for v in df[self.group_col].tolist()], dtype=object)
            ds_name = str(self.data_cfg.get("dataset") or self.external_path.stem)
            self.dataset_id = f"tabular-{ds_name}"
            self.task_id = f"task-tabular-{ds_name}"
            eval_cfg = config.get("evaluation") or {}
            pm = eval_cfg.get("primary_metric", "")
            uniq_y = np.unique(self.y[np.isfinite(self.y)])
            if self.data_cfg.get("task_type") in {"regression", "classification"}:
                self.task_type = str(self.data_cfg["task_type"])
            elif pm in {"r2", "mse", "mae"} or len(uniq_y) > 2:
                self.task_type = "regression"
            else:
                self.task_type = "classification"
            default_eval = (
                {
                    "name": "tabular_regression",
                    "primary_metric": "mse",
                    "direction": "lower",
                    "secondary_metrics": ["r2", "mae"],
                    "paired": True,
                    "multiple_testing": "benjamini-hochberg",
                    "notes": "time/group-ordered out-of-sample tabular evaluation",
                }
                if self.task_type == "regression"
                else {
                    "name": "tabular_cv_logloss",
                    "primary_metric": "auc",
                    "direction": "higher",
                    "secondary_metrics": ["accuracy", "log_loss"],
                    "paired": True,
                    "multiple_testing": "benjamini-hochberg",
                    "notes": "time/group-ordered out-of-sample tabular classification evaluation",
                }
            )

        self._evaluation = EvaluationSpec.model_validate(
            config.get("evaluation", default_eval)
        )
        self._folds, self._confirmation, self._holdout = self._build_folds()

    # ------------------------------------------------------------------ data
    def _split(self) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(self.y))
        pool, hold = train_test_split(
            idx, test_size=0.2, random_state=self.seed, stratify=self.y
        )
        return np.sort(pool), np.sort(hold)

    def _build_external_ordered_folds(self) -> tuple[list[FoldSpec], list[FoldSpec], FoldSpec]:
        n = len(self.y)
        order_vals = self.time_values if self.time_values is not None else self.group_values
        if order_vals is not None:
            # Partition by unique ordered time/group buckets (60% discovery, 20% confirmation, 20% holdout)
            unique_buckets: list[str] = []
            seen_b: set[str] = set()
            for v in order_vals.tolist():
                s = str(v)
                if s not in seen_b:
                    seen_b.add(s)
                    unique_buckets.append(s)
            n_b = len(unique_buckets)
            if n_b >= 10:
                b_disc_end = max(4, int(round(0.60 * n_b)))
                b_conf_end = max(b_disc_end + 1, int(round(0.80 * n_b)))
                disc_set = set(unique_buckets[:b_disc_end])
                conf_set = set(unique_buckets[b_disc_end:b_conf_end])
                hold_set = set(unique_buckets[b_conf_end:])
                disc_idx = np.array([i for i, v in enumerate(order_vals) if str(v) in disc_set], dtype=int)
                conf_idx = np.array([i for i, v in enumerate(order_vals) if str(v) in conf_set], dtype=int)
                hold_idx = np.array([i for i, v in enumerate(order_vals) if str(v) in hold_set], dtype=int)
            else:
                n_disc = int(round(0.60 * n))
                n_conf = int(round(0.80 * n))
                disc_idx = np.arange(0, n_disc, dtype=int)
                conf_idx = np.arange(n_disc, n_conf, dtype=int)
                hold_idx = np.arange(n_conf, n, dtype=int)
        else:
            n_disc = int(round(0.60 * n))
            n_conf = int(round(0.80 * n))
            disc_idx = np.arange(0, n_disc, dtype=int)
            conf_idx = np.arange(n_disc, n_conf, dtype=int)
            hold_idx = np.arange(n_conf, n, dtype=int)

        def _range_labels(indices: np.ndarray, fallback_prefix: str) -> tuple[str, str]:
            if order_vals is not None and len(indices) > 0:
                return str(order_vals[int(indices[0])]), str(order_vals[int(indices[-1])])
            if len(indices) > 0:
                return f"{fallback_prefix}_{int(indices[0]):06d}", f"{fallback_prefix}_{int(indices[-1]):06d}"
            return f"{fallback_prefix}_start", f"{fallback_prefix}_end"

        n_splits = int(self.config.get("protocol", {}).get("cv_splits", 3))
        n_splits = max(2, n_splits)
        folds: list[FoldSpec] = []
        # Expanding window over discovery pool: first chunk is base training, remaining chunks are validation folds
        chunks = np.array_split(disc_idx, n_splits + 1)
        for k in range(n_splits):
            tr_idx = np.concatenate(chunks[: k + 1])
            va_idx = chunks[k + 1]
            tr_s, tr_e = _range_labels(tr_idx, "idx")
            va_s, va_e = _range_labels(va_idx, "idx")
            folds.append(
                FoldSpec(
                    fold_id=f"disc_f{k}",
                    kind="cv",
                    role="search",
                    train_start=tr_s,
                    train_end=tr_e,
                    valid_start=va_s,
                    valid_end=va_e,
                    purge_days=0,
                    embargo_days=0,
                    metadata={
                        "train_idx": tr_idx.tolist(),
                        "valid_idx": va_idx.tolist(),
                        "cv_key": f"disc_f{k}",
                    },
                    notes="expanding-window discovery fold on first 60% ordered data",
                )
            )

        n_conf_splits = int(self.config.get("protocol", {}).get("confirmation_splits", 2))
        n_conf_splits = max(1, n_conf_splits)
        conf_chunks = np.array_split(conf_idx, n_conf_splits)
        confirmation: list[FoldSpec] = []
        for k, va_idx in enumerate(conf_chunks):
            if len(va_idx) == 0:
                continue
            tr_idx = disc_idx if k == 0 else np.concatenate([disc_idx, *conf_chunks[:k]])
            tr_s, tr_e = _range_labels(tr_idx, "idx")
            va_s, va_e = _range_labels(va_idx, "idx")
            confirmation.append(
                FoldSpec(
                    fold_id=f"conf_f{k}",
                    kind="cv",
                    role="confirmation",
                    train_start=tr_s,
                    train_end=tr_e,
                    valid_start=va_s,
                    valid_end=va_e,
                    purge_days=0,
                    embargo_days=0,
                    metadata={
                        "train_idx": tr_idx.tolist(),
                        "valid_idx": va_idx.tolist(),
                        "cv_key": f"conf_f{k}",
                    },
                    notes="time/group-forward confirmation fold on middle 20% data",
                )
            )

        pre_hold_idx = np.concatenate([disc_idx, conf_idx])
        tr_s, tr_e = _range_labels(pre_hold_idx, "idx")
        ho_s, ho_e = _range_labels(hold_idx, "idx")
        holdout = FoldSpec(
            fold_id="holdout_20pct",
            kind="cv",
            role="holdout",
            train_start=tr_s,
            train_end=tr_e,
            valid_start=ho_s,
            valid_end=ho_e,
            test_start=ho_s,
            test_end=ho_e,
            purge_days=0,
            embargo_days=0,
            metadata={
                "train_idx": pre_hold_idx.tolist(),
                "valid_idx": hold_idx.tolist(),
                "test_idx": hold_idx.tolist(),
                "cv_key": "holdout",
            },
            notes="time/group-forward 20% holdout split (last 20%)",
        )
        return folds, confirmation, holdout

    def _build_folds(self) -> tuple[list[FoldSpec], list[FoldSpec], FoldSpec]:
        if self._is_external and (
            self.time_col is not None
            or self.group_col is not None
            or self.config.get("protocol", {}).get("confirmation_kind") == "time_forward"
        ):
            return self._build_external_ordered_folds()

        from sklearn.model_selection import StratifiedKFold, KFold

        if self._is_external and self.task_type == "regression":
            idx = np.arange(len(self.y))
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(idx)
            n_hold = max(1, int(round(0.2 * len(idx))))
            hold = np.sort(perm[:n_hold])
            pool = np.sort(perm[n_hold:])
        else:
            pool, hold = self._split()
        folds: list[FoldSpec] = []
        confirmation: list[FoldSpec] = []
        n_splits = int(self.config.get("protocol", {}).get("cv_splits", 5))
        repeats = int(self.config.get("protocol", {}).get("cv_repeats", 2))
        discovery_repeats = int(self.config.get("protocol", {}).get(
            "discovery_repeats", max(1, repeats - 1)
        ))
        discovery_repeats = max(1, min(discovery_repeats, repeats))
        for r in range(repeats):
            if self.task_type == "classification":
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed + r)
                split_iter = splitter.split(pool, self.y[pool])
            else:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed + r)
                split_iter = splitter.split(pool)
            role = "search" if r < discovery_repeats else "confirmation"
            bucket = folds if role == "search" else confirmation
            for k, (tr, va) in enumerate(split_iter):
                train_idx = pool[tr].tolist()
                valid_idx = pool[va].tolist()
                bucket.append(FoldSpec(
                    fold_id=f"cv_r{r}_f{k}", kind="cv", role=role,
                    metadata={"train_idx": train_idx, "valid_idx": valid_idx,
                              "repeat": r, "fold": k,
                              "cv_key": f"r{r}_f{k}"},
                    notes=("stratified CV fold on the 80% search split"
                           if role == "search" else
                           "disjoint resampling confirmation repeat; same pool, not new data"),
                ))
        holdout = FoldSpec(
            fold_id="holdout_20pct", kind="cv", role="holdout",
            metadata={"train_idx": pool.tolist(), "valid_idx": hold.tolist(),
                      "test_idx": hold.tolist(), "cv_key": "holdout"},
            notes="fixed stratified 20% holdout, frozen before finalize",
        )
        return folds, confirmation, holdout

    def data_contract(self, config: dict[str, Any]) -> DataContract:
        if not self._is_external:
            return DataContract(
                dataset_id="sklearn-breast-cancer",
                domain="tabular",
                description="Breast Cancer Wisconsin (diagnostic) tabular classification dataset.",
                entity_key="sample",
                time_key="",
                label_definition="malignant (1) vs benign (0)",
                label_window={"type": "static", "uses_future_days": 0,
                              "leakage_risk": "train/test split by sample"},
                label_dependency={"future_offset_trading_days": 0,
                                  "same_sample_label": True,
                                  "note": "diagnosis label is observed with the sample; no time axis"},
                feature_semantics={name: "measurement" for name in self.feature_names},
                universe="569 samples × 30 features",
                frequency="static",
                preprocessing_version="tabular-standard-scaler-v1",
                time_range=("", ""),
                caveats=[
                    "Static dataset: no temporal ordering; repeated stratified CV is the validation protocol.",
                    "Fixed 20% holdout is separated before any candidate search.",
                ],
            )
        order_vals = self.time_values if self.time_values is not None else self.group_values
        t_range = (str(order_vals[0]), str(order_vals[-1])) if order_vals is not None and len(order_vals) > 0 else ("", "")
        return DataContract(
            dataset_id=self.dataset_id,
            domain="tabular",
            description=f"External tabular dataset ({self.task_type}) from {self.external_path.name if self.external_path else 'external'}.",
            entity_key=self.group_col or "sample",
            time_key=self.time_col or self.group_col or "",
            label_definition=f"{self.target_col} ({self.task_type})",
            label_window={"type": "ordered" if order_vals is not None else "static", "uses_future_days": 0,
                          "leakage_risk": "chronological/group split (60% discovery / 20% confirmation / 20% holdout)"},
            label_dependency={"future_offset_trading_days": 0,
                              "same_sample_label": True,
                              "note": "ordered split prevents future-to-past leakage"},
            feature_semantics={name: "measurement" for name in self.feature_names},
            universe=f"{len(self.y)} samples × {self.n_features} features",
            frequency="ordered" if order_vals is not None else "static",
            preprocessing_version="tabular-standard-scaler-v1",
            time_range=t_range,
            caveats=[
                "External tabular dataset with 60% discovery, 20% confirmation, and 20% holdout split.",
            ],
        )

    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        if not self._is_external:
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
        details = {
            "file_sha256": self.file_sha256,
            "n_samples": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "task_type": self.task_type,
            "X_sha256": sha256_obj(np.round(self.X, 10).tolist()),
            "y_sha256": sha256_obj(np.round(self.y, 10).tolist()),
            "feature_names": self.feature_names,
            "holdout_idx": self._holdout.metadata["valid_idx"],
        }
        return DataSnapshot(
            dataset_id=self.dataset_id,
            domain="tabular",
            manifest_method="sha256 of external tabular file and rounded feature/target arrays",
            manifest_hash=sha256_obj(details),
            path=str(self.external_path),
            details=details,
        )

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        if self.task_type == "classification":
            counts = np.bincount(self.y.astype(int))
            balance_check = {"ok": counts.min() >= 10, "severity": "warn",
                             "detail": f"counts={counts.tolist()}"}
        else:
            y_std = float(np.std(self.y))
            balance_check = {"ok": y_std > 1e-8, "severity": "block",
                             "detail": f"y_std={y_std:.6g}"}
        return assess_common({
            "sample_count": {"ok": len(self.y) >= 100, "severity": "warn",
                             "detail": f"{len(self.y)} samples"},
            "class_balance": balance_check,
            "missing_values": {"ok": bool(np.isfinite(self.X).all() and np.isfinite(self.y).all()),
                               "severity": "block",
                               "detail": "non-finite features or targets"},
            "feature_count": {"ok": self.n_features >= 2, "severity": "warn",
                              "detail": f"{self.n_features} features"},
        })

    # -------------------------------------------------------------- protocol
    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        if not self._is_external:
            return ExperimentProtocol(
                task_id="task-tabular-breast-cancer",
                domain="tabular",
                primary_metric=self._evaluation.primary_metric,
                direction=self._evaluation.direction,
                folds=self._folds,
                confirmation_folds=self._confirmation,
                confirmation_kind="resampling",
                holdout_fold=self._holdout,
                baseline_variant="baseline:logreg",
                control_variant=None,
                seeds=[int(x) for x in self.config.get("seeds", [self.seed])],
                model_spec={"model": "LogisticRegression", "C": 1.0},
                data_config={"dataset": "sklearn-breast-cancer", "cv_splits": 5,
                             "cv_repeats": int(self.config.get("protocol", {}).get("cv_repeats", 2))},
                evaluation=self._evaluation,
            )
        is_ordered = (
            self.time_col is not None
            or self.group_col is not None
            or self.config.get("protocol", {}).get("confirmation_kind") == "time_forward"
        )
        conf_kind = str(
            self.config.get("protocol", {}).get(
                "confirmation_kind", "time_forward" if is_ordered else "resampling"
            )
        )
        baseline_var = "baseline:ridge" if self.task_type == "regression" else "baseline:logreg"
        model_spec = (
            {"model": "Ridge", "alpha": float(self.config.get("training", {}).get("alpha", 1.0))}
            if self.task_type == "regression"
            else {"model": "LogisticRegression", "C": float(self.config.get("training", {}).get("C", 1.0))}
        )
        return ExperimentProtocol(
            task_id=self.task_id,
            domain="tabular",
            primary_metric=self._evaluation.primary_metric,
            direction=self._evaluation.direction,
            folds=self._folds,
            confirmation_folds=self._confirmation,
            confirmation_kind=conf_kind,
            holdout_fold=self._holdout,
            baseline_variant=baseline_var,
            control_variant=None,
            seeds=[int(x) for x in self.config.get("seeds", [self.seed])],
            model_spec=model_spec,
            data_config={
                "dataset": self.dataset_id,
                "file_sha256": self.file_sha256,
                "target_col": self.target_col,
                "time_col": self.time_col,
                "group_col": self.group_col,
                "task_type": self.task_type,
            },
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
        gen_name = str(policy.get("generator", "rule")).strip().lower()
        if gen_name in {"llm", "deepseek", "llm_hybrid"}:
            from predicatelab.domains.tabular.generator import TabularLLMGenerator

            return TabularLLMGenerator(
                config,
                self.feature_names,
                self.n_features,
                direction=self._evaluation.direction,
                gen_cfg=dict(policy.get("generator_config", {}) or {}),
            )
        return TabularRuleGenerator(config, self.feature_names, self.n_features,
                                    direction=self._evaluation.direction)

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
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        Xtr, ytr = self.X[train_idx], self.y[train_idx]
        Xva, yva = self.X[valid_idx], self.y[valid_idx]
        fitted: dict[str, Any] = {}
        if desc is not None:
            col_tr, col_va, fitted = fit_transform(desc, Xtr, Xva)
            Xtr = np.hstack([Xtr, col_tr])
            Xva = np.hstack([Xva, col_va])

        if self.task_type == "regression":
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            alpha = float(self.config.get("training", {}).get("alpha", 1.0))
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=seed))
            model.fit(Xtr, ytr)
            pred = np.asarray(model.predict(Xva), dtype=float)
            metrics = {
                "r2": float(r2_score(yva, pred)) if len(yva) >= 2 and np.std(yva) > 1e-12 else 0.0,
                "mse": float(mean_squared_error(yva, pred)),
                "mae": float(mean_absolute_error(yva, pred)),
            }
            predictions = {
                "sample": np.asarray(valid_idx, dtype=np.int64),
                "score": pred,
                "label": np.asarray(yva, dtype=float),
            }
            return metrics, fitted, predictions

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

        c_val = float(self.config.get("training", {}).get("C", 1.0))
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=c_val, random_state=seed, solver="lbfgs"),
        )
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xva)[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc_val = float(roc_auc_score(yva, prob)) if len(np.unique(yva)) >= 2 else 0.5
        metrics = {
            "auc": auc_val,
            "accuracy": float(accuracy_score(yva, pred)),
            "log_loss": float(log_loss(yva, np.clip(prob, 1e-7, 1.0 - 1e-7), labels=[0, 1])),
        }
        predictions = {
            "sample": np.asarray(valid_idx, dtype=np.int64),
            "score": np.asarray(prob, dtype=float),
            "label": np.asarray(yva, dtype=np.int64),
        }
        return metrics, fitted, predictions

    def _build_external_metric_series(
        self, key: str, predictions: dict[str, np.ndarray], metrics: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        scores = np.asarray(predictions["score"], dtype=float)
        labels = np.asarray(predictions["label"], dtype=float)
        n = len(labels)
        n_blocks = min(10, max(2, n // 8))
        if n < 4:
            return {m_name: {key: float(m_val)} for m_name, m_val in metrics.items()}
        idx_chunks = np.array_split(np.arange(n), n_blocks)
        if self.task_type == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            r2_ser: dict[str, float] = {}
            mse_ser: dict[str, float] = {}
            mae_ser: dict[str, float] = {}
            for b, ch in enumerate(idx_chunks):
                if len(ch) < 2:
                    continue
                b_key = f"{key}_b{b:02d}"
                y_b = labels[ch]
                p_b = scores[ch]
                r2_ser[b_key] = float(r2_score(y_b, p_b)) if np.std(y_b) > 1e-12 else 0.0
                mse_ser[b_key] = float(mean_squared_error(y_b, p_b))
                mae_ser[b_key] = float(mean_absolute_error(y_b, p_b))
            return {"r2": r2_ser, "mse": mse_ser, "mae": mae_ser}
        else:
            from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

            auc_ser: dict[str, float] = {}
            ll_ser: dict[str, float] = {}
            acc_ser: dict[str, float] = {}
            for b, ch in enumerate(idx_chunks):
                if len(ch) < 2:
                    continue
                b_key = f"{key}_b{b:02d}"
                y_b = labels[ch].astype(int)
                p_b = np.clip(scores[ch], 1e-7, 1.0 - 1e-7)
                auc_ser[b_key] = float(roc_auc_score(y_b, p_b)) if len(np.unique(y_b)) >= 2 else float(metrics["auc"])
                ll_ser[b_key] = float(log_loss(y_b, p_b, labels=[0, 1]))
                acc_ser[b_key] = float(accuracy_score(y_b, (p_b >= 0.5).astype(int)))
            return {"auc": auc_ser, "log_loss": ll_ser, "accuracy": acc_ser}

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        import time

        t0 = time.time()
        meta = request.fold.metadata
        train_idx = np.asarray(meta["train_idx"], dtype=int)
        valid_idx = np.asarray(meta["valid_idx"], dtype=int)
        if request.variant == "candidate":
            metrics, fitted, predictions = self._predict(
                request.descriptor, train_idx, valid_idx, request.seed)
        else:
            metrics, fitted, predictions = self._predict(
                None, train_idx, valid_idx, request.seed)
        pred_path = ""
        if request.save_predictions:
            out = Path(request.workdir) / "predictions.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, **predictions)
            tmp.replace(out)
            pred_path = str(out)
        key = str(meta.get("cv_key", request.fold.fold_id))
        if not self._is_external:
            metric_series = {
                "auc": {key: metrics["auc"]},
                "log_loss": {key: metrics["log_loss"]},
                "accuracy": {key: metrics["accuracy"]},
            }
        else:
            metric_series = self._build_external_metric_series(key, predictions, metrics)
        return {
            "status": "success",
            "metrics": metrics,
            "metric_series": metric_series,
            "fitted_params": fitted,
            "fit_data_hash": sha256_obj({"train_idx": train_idx.tolist(),
                                         "dataset": self.dataset_id}),
            "predictions_path": pred_path,
            "duration_s": time.time() - t0,
            "error": "",
            "worker": {"variant": request.variant, "fold": request.fold.fold_id, "in_process": True},
        }
