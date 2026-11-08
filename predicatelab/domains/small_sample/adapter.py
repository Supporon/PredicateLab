"""Small-sample application scenario.

This adapter turns a controlled synthetic dataset with a known latent signal
into a full PredicateLab domain.  It exists to answer a specific question that
neither TSIL (large daily cross-sections) nor the tabular demo (569 samples)
can answer: **what does the discovery -> confirmation -> holdout pipeline do
when every stage has very few samples?**

Design decisions that matter for small samples
----------------------------------------------
* The statistical unit is the *individual sample*, not a fold or a date.  Each
  validation fold contributes one ``neg_brier`` value per out-of-fold sample,
  so the paired test has ``n_effective`` units rather than 3-5 fold summaries.
  Samples are i.i.d. by construction, so ``nw_lag=0`` and ``bootstrap_block=1``
  are the pre-registered settings (HAC/block bootstrap default to serial
  dependence that does not exist here).
* The 25% holdout is fixed before any search and is the only truly independent
  evaluation.  Discovery and confirmation use disjoint parts of the remaining
  75% (``independent`` mode) or disjoint resamples of the same pool
  (``resampling`` mode).  The engine's ``confirmation_kind`` is therefore
  ``independent_split`` or ``resampling``; the former is conservatively
  reported by the core as ``resampling_confirmed`` because the core's state
  machine only distinguishes time-forward confirmation from everything else.
* Suitability fails closed: below 24 samples, or when a class cannot support
  the declared CV, the run is blocked before any trial executes.

The ground truth (mode, true descriptor, signal strength) is recorded in the
snapshot for post-hoc scoring.  It is never shown to the generator, the probe,
or the trial runner.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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
from predicatelab.domains.small_sample.generator import SmallSampleRuleGenerator
from predicatelab.domains.small_sample.synthetic import (
    MODES,
    SyntheticData,
    descriptor_family,
    generate,
    matches_true_family,
    truth_family,
)

OPS = {"ratio", "difference", "product", "ratio_threshold", "threshold",
       "sigmoid", "interaction"}
PAIRWISE = {"ratio", "difference", "product", "ratio_threshold", "interaction"}
ALLOWED_KEYS = {"op", "a", "b", "quantile", "scale", "note"}
MIN_SAMPLES = 24


class DescriptorError(ValueError):
    pass


def validate_small_sample_descriptor(desc: dict[str, Any], n_features: int) -> None:
    """Closed validator: reject anything the compiler cannot execute."""
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
    if op in PAIRWISE:
        b = desc.get("b")
        if not isinstance(b, int) or isinstance(b, bool) or not (0 <= b < n_features) or b == a:
            raise DescriptorError(f"feature index b={b!r} invalid")
    if op in {"threshold", "ratio_threshold"}:
        q = desc.get("quantile")
        if not isinstance(q, (int, float)) or isinstance(q, bool) or not (0.0 < float(q) < 1.0):
            raise DescriptorError("quantile must be in (0,1)")
    if op == "sigmoid":
        s = desc.get("scale", 1.0)
        if not isinstance(s, (int, float)) or isinstance(s, bool) or not (0.05 <= float(s) <= 20.0):
            raise DescriptorError("scale must be in [0.05,20]")


def _fit_transform(desc: dict[str, Any], X_train: np.ndarray,
                   X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit predicate parameters on ``X_train`` only and transform both sets."""
    op = desc["op"]
    a = int(desc["a"])
    params: dict[str, Any] = {"op": op, "a": a}
    if op in {"ratio", "ratio_threshold"}:
        b = int(desc["b"])
        sd_b = float(np.std(X_train[:, b])) + 1e-8
        offset = 0.1 * sd_b
        denom_train = np.abs(X_train[:, b]) + offset
        denom_other = np.abs(X_other[:, b]) + offset
        ratio_train = X_train[:, a] / denom_train
        ratio_other = X_other[:, a] / denom_other
        params.update({"b": b, "denominator_offset": float(offset)})
        if op == "ratio":
            train, other = ratio_train, ratio_other
        else:
            q = float(desc["quantile"])
            threshold = float(np.quantile(ratio_train, q))
            params.update({"quantile": q, "threshold": threshold})
            train = (ratio_train >= threshold).astype(float)
            other = (ratio_other >= threshold).astype(float)
    elif op == "difference":
        b = int(desc["b"])
        params["b"] = b
        train, other = X_train[:, a] - X_train[:, b], X_other[:, a] - X_other[:, b]
    elif op == "product":
        b = int(desc["b"])
        params["b"] = b
        train, other = X_train[:, a] * X_train[:, b], X_other[:, a] * X_other[:, b]
    elif op == "interaction":
        b = int(desc["b"])
        # A sign interaction is defined about the origin: sign(x_a * x_b).
        # Centring by the fold mean would move the boundary and produce a
        # predicate that no longer measures the same relation as the
        # population-level truth, so the raw signs are used.  Positive
        # rescaling is sign-preserving, and the downstream standardiser
        # handles scaling for the linear model.
        params.update({"b": b})
        train = np.sign(X_train[:, a]) * np.sign(X_train[:, b])
        other = np.sign(X_other[:, a]) * np.sign(X_other[:, b])
    elif op == "threshold":
        q = float(desc["quantile"])
        threshold = float(np.quantile(X_train[:, a], q))
        params.update({"quantile": q, "threshold": threshold})
        train = (X_train[:, a] >= threshold).astype(float)
        other = (X_other[:, a] >= threshold).astype(float)
    elif op == "sigmoid":
        median = float(np.median(X_train[:, a]))
        scale = float(np.quantile(np.abs(X_train[:, a] - median), 0.75) + 1e-6)
        k = float(desc.get("scale", 1.0))
        params.update({"median": median, "iqr": scale, "scale": k})
        train = 1.0 / (1.0 + np.exp(-k * (X_train[:, a] - median) / scale))
        other = 1.0 / (1.0 + np.exp(-k * (X_other[:, a] - median) / scale))
    else:  # pragma: no cover - validated earlier
        raise DescriptorError(op)
    return train.reshape(-1, 1), other.reshape(-1, 1), params


def _idx_label(indices: np.ndarray) -> str:
    if len(indices) == 0:
        return "idx:none"
    return f"idx:{int(indices[0])}-{int(indices[-1])}"


class SmallSampleAdapter(DomainAdapter):
    name = "small_sample"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_cfg = dict(config.get("data", {}))
        self.seed = int(config.get("training", {}).get("seed", 2025))
        self.scenario_seed = int(self.data_cfg.get("seed", self.seed))
        self.mode = str(self.data_cfg.get("generator", "null_noise"))
        if self.mode not in MODES:
            raise ValueError(f"data.generator must be one of {MODES}, got {self.mode!r}")
        self.n_samples = int(self.data_cfg.get("n_samples", 64))
        self.n_features = int(self.data_cfg.get("n_features", 8))
        self.signal_strength = float(self.data_cfg.get("signal_strength", 0.0))
        self.threshold_quantile = float(self.data_cfg.get("threshold_quantile", 0.5))
        self.class_balance = float(self.data_cfg.get("class_balance", 0.5))
        self.holdout_fraction = float(self.data_cfg.get("holdout_fraction", 0.25))
        self.discovery_fraction = float(self.data_cfg.get("discovery_fraction", 0.65))
        self.confirmation_mode = str(self.data_cfg.get("confirmation_mode", "independent"))
        if self.confirmation_mode not in {"independent", "resampling"}:
            raise ValueError("data.confirmation_mode must be 'independent' or 'resampling'")

        self.data: SyntheticData = generate(
            seed=self.scenario_seed, n_samples=self.n_samples,
            n_features=self.n_features, mode=self.mode,
            signal_strength=self.signal_strength,
            threshold_quantile=self.threshold_quantile,
            class_balance=self.class_balance,
        )
        self.X = self.data.X
        self.y = self.data.y
        self.truth = self.data.truth
        self.truth_indicator = self.data.truth_indicator
        self.feature_names = [f"x{i}" for i in range(self.n_features)]
        self.dataset_id = f"small-sample-{self.mode}-n{self.n_samples}-s{self.scenario_seed}"
        self.task_id = f"task-small-sample-{self.mode}"

        eval_cfg = config.get("evaluation") or {}
        default_eval = {
            "name": "small_sample_brier",
            "primary_metric": "neg_brier",
            "direction": "higher",
            "secondary_metrics": ["auc", "accuracy", "log_loss"],
            "paired": True,
            "multiple_testing": "benjamini-hochberg",
            "notes": ("per-sample paired Brier improvement on stratified CV folds; "
                      "i.i.d. sampling with nw_lag=0 and block bootstrap=1"),
        }
        self._evaluation = EvaluationSpec.model_validate(
            config.get("evaluation", default_eval)
        )
        self._folds, self._confirmation, self._holdout = self._build_folds()

    # ------------------------------------------------------------------ splits
    @staticmethod
    def _stratified_take(indices: np.ndarray, y: np.ndarray, frac: float,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Take a stratified fraction, keeping at least one sample per class."""
        selected: list[int] = []
        rest: list[int] = []
        for cls in np.unique(y[indices]):
            cls_idx = indices[y[indices] == cls]
            perm = rng.permutation(cls_idx)
            if len(perm) < 2:
                selected.extend(perm.tolist()[:0])
                rest.extend(perm.tolist())
                continue
            n_take = int(round(float(frac) * len(perm)))
            n_take = max(1, min(len(perm) - 1, n_take))
            selected.extend(perm[:n_take].tolist())
            rest.extend(perm[n_take:].tolist())
        return (np.sort(np.asarray(selected, dtype=int)),
                np.sort(np.asarray(rest, dtype=int)))

    def _build_folds(self) -> tuple[list[FoldSpec], list[FoldSpec], FoldSpec]:
        from sklearn.model_selection import StratifiedKFold

        protocol_cfg = self.config.get("protocol", {})
        cv_splits = max(2, int(protocol_cfg.get("cv_splits", 4)))
        cv_repeats = max(1, int(protocol_cfg.get("cv_repeats", 1)))
        discovery_repeats = max(1, int(protocol_cfg.get(
            "discovery_repeats", 1 if self.confirmation_mode == "independent" else 2
        )))
        discovery_repeats = min(discovery_repeats, cv_repeats)

        all_idx = np.arange(len(self.y), dtype=int)
        rng = np.random.default_rng(self.scenario_seed + 7001)
        holdout_idx, pool_idx = self._stratified_take(
            all_idx, self.y, self.holdout_fraction, rng
        )
        if self.confirmation_mode == "independent":
            conf_frac = max(0.2, min(0.6, 1.0 - self.discovery_fraction))
            rng_conf = np.random.default_rng(self.scenario_seed + 7002)
            confirmation_idx, discovery_idx = self._stratified_take(
                pool_idx, self.y, conf_frac, rng_conf
            )
        else:
            discovery_idx = pool_idx
            confirmation_idx = pool_idx

        folds: list[FoldSpec] = []
        confirmation: list[FoldSpec] = []
        if self.confirmation_mode == "independent":
            split_iter = StratifiedKFold(
                n_splits=cv_splits, shuffle=True, random_state=self.seed
            ).split(discovery_idx, self.y[discovery_idx])
            for k, (tr, va) in enumerate(split_iter):
                train_idx = np.sort(discovery_idx[tr])
                valid_idx = np.sort(discovery_idx[va])
                folds.append(FoldSpec(
                    fold_id=f"d0_f{k}", kind="cv", role="search",
                    train_start=_idx_label(train_idx), train_end=_idx_label(train_idx),
                    valid_start=_idx_label(valid_idx), valid_end=_idx_label(valid_idx),
                    purge_days=0, embargo_days=0,
                    metadata={"train_idx": train_idx.tolist(),
                              "valid_idx": valid_idx.tolist(),
                              "repeat": 0, "fold": k, "cv_key": f"d0_f{k}"},
                    notes="stratified discovery CV fold on the discovery pool",
                ))
            min_conf_class = int(np.bincount(self.y[confirmation_idx]).min()) \
                if len(confirmation_idx) else 0
            conf_splits = int(protocol_cfg.get("confirmation_splits",
                                               min(3, max(2, min_conf_class))))
            conf_splits = max(2, min(conf_splits, min_conf_class))
            conf_iter = StratifiedKFold(
                n_splits=conf_splits, shuffle=True, random_state=self.seed + 501
            ).split(confirmation_idx, self.y[confirmation_idx])
            for k, (tr, va) in enumerate(conf_iter):
                train_idx = np.sort(np.concatenate([discovery_idx, confirmation_idx[tr]]))
                valid_idx = np.sort(confirmation_idx[va])
                confirmation.append(FoldSpec(
                    fold_id=f"c{k}", kind="cv", role="confirmation",
                    train_start=_idx_label(train_idx), train_end=_idx_label(train_idx),
                    valid_start=_idx_label(valid_idx), valid_end=_idx_label(valid_idx),
                    purge_days=0, embargo_days=0,
                    metadata={"train_idx": train_idx.tolist(),
                              "valid_idx": valid_idx.tolist(),
                              "repeat": 100, "fold": k, "cv_key": f"c{k}"},
                    notes="confirmation fold on samples disjoint from discovery",
                ))
        else:
            for r in range(cv_repeats):
                role = "search" if r < discovery_repeats else "confirmation"
                split_iter = StratifiedKFold(
                    n_splits=cv_splits, shuffle=True, random_state=self.seed + r
                ).split(discovery_idx, self.y[discovery_idx])
                bucket = folds if role == "search" else confirmation
                for k, (tr, va) in enumerate(split_iter):
                    train_idx = np.sort(discovery_idx[tr])
                    valid_idx = np.sort(discovery_idx[va])
                    bucket.append(FoldSpec(
                        fold_id=f"r{r}_f{k}", kind="cv", role=role,
                        train_start=_idx_label(train_idx), train_end=_idx_label(train_idx),
                        valid_start=_idx_label(valid_idx), valid_end=_idx_label(valid_idx),
                        purge_days=0, embargo_days=0,
                        metadata={"train_idx": train_idx.tolist(),
                                  "valid_idx": valid_idx.tolist(),
                                  "repeat": r, "fold": k, "cv_key": f"r{r}_f{k}"},
                        notes=("stratified discovery CV fold"
                               if role == "search" else
                               "disjoint resampling confirmation; same pool, not new samples"),
                    ))

        pre_hold_idx = np.sort(np.concatenate([discovery_idx, confirmation_idx]))
        holdout = FoldSpec(
            fold_id="holdout", kind="cv", role="holdout",
            train_start=_idx_label(pre_hold_idx), train_end=_idx_label(pre_hold_idx),
            valid_start=_idx_label(holdout_idx), valid_end=_idx_label(holdout_idx),
            test_start=_idx_label(holdout_idx), test_end=_idx_label(holdout_idx),
            purge_days=0, embargo_days=0,
            metadata={"train_idx": pre_hold_idx.tolist(),
                      "valid_idx": holdout_idx.tolist(),
                      "test_idx": holdout_idx.tolist(), "cv_key": "holdout"},
            notes="fixed stratified holdout, separated before any search",
        )
        self._holdout_idx = holdout_idx
        self._discovery_idx = discovery_idx
        self._confirmation_idx = confirmation_idx
        return folds, confirmation, holdout

    # -------------------------------------------------------------------- data
    def data_contract(self, config: dict[str, Any]) -> DataContract:
        return DataContract(
            dataset_id=self.dataset_id,
            domain="small_sample",
            description=(f"Controlled synthetic small-sample dataset "
                         f"({self.mode}, n={self.n_samples}, p={self.n_features})."),
            entity_key="sample",
            time_key="",
            label_definition=(f"synthetic binary label from latent mode {self.mode!r}; "
                              "same-sample label"),
            label_window={"type": "static", "uses_future_days": 0,
                          "leakage_risk": "none: the label is generated per sample"},
            label_dependency={"future_offset_trading_days": 0,
                              "same_sample_label": True,
                              "note": "static i.i.d. samples; no time axis"},
            feature_semantics={name: "standard-normal synthetic measurement"
                               for name in self.feature_names},
            universe=f"{len(self.y)} samples x {self.n_features} features",
            frequency="static",
            preprocessing_version="small-sample-standardise-v1",
            time_range=("", ""),
            caveats=[
                "Synthetic data with a known ground truth; results validate the "
                "platform's small-sample behaviour, not a real scientific domain.",
                "The 25% holdout is fixed before search; discovery and confirmation "
                "use disjoint samples (independent mode) or disjoint resamples.",
            ],
        )

    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        counts = {str(int(k)): int(v) for k, v in
                  zip(*np.unique(self.y, return_counts=True))}
        details = {
            "generator": self.mode,
            "seed": self.scenario_seed,
            "n_samples": int(self.n_samples),
            "n_features": int(self.n_features),
            "class_counts": counts,
            "X_sha256": sha256_obj(np.round(self.X, 10).tolist()),
            "y_sha256": sha256_obj(self.y.tolist()),
            "holdout_idx": self._holdout_idx.tolist(),
            "discovery_size": int(len(self._discovery_idx)),
            "confirmation_size": int(len(self._confirmation_idx)),
            "holdout_size": int(len(self._holdout_idx)),
            "confirmation_mode": self.confirmation_mode,
            "truth": self.truth,
        }
        return DataSnapshot(
            dataset_id=self.dataset_id,
            domain="small_sample",
            manifest_method=("sha256 of synthetic feature matrix, labels, DGP "
                             "parameters and seed (deterministic)"),
            manifest_hash=sha256_obj(details),
            path="generated://small_sample",
            details=details,
        )

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        counts = np.bincount(self.y, minlength=2)
        disc_counts = (np.bincount(self.y[self._discovery_idx], minlength=2)
                       if len(self._discovery_idx) else np.zeros(2, dtype=int))
        conf_counts = (np.bincount(self.y[self._confirmation_idx], minlength=2)
                       if len(self._confirmation_idx) else np.zeros(2, dtype=int))
        hold_counts = np.bincount(self.y[self._holdout_idx], minlength=2)
        cv_splits = max(2, int(self.config.get("protocol", {}).get("cv_splits", 4)))
        conf_splits = max(2, int(self.config.get("protocol", {}).get("confirmation_splits",
                                                                   min(3, max(2, int(conf_counts.min()))))))
        feasibility = (
            int(disc_counts.min()) >= cv_splits
            and int(conf_counts.min()) >= min(2, conf_splits)
            and int(hold_counts.min()) >= 2
        )
        return assess_common({
            "sample_count": {
                "ok": len(self.y) >= MIN_SAMPLES, "severity": "block",
                "detail": f"{len(self.y)} samples (minimum {MIN_SAMPLES} for this protocol)",
            },
            "class_balance": {
                "ok": bool(int(counts.min()) >= 2), "severity": "block",
                "detail": f"class counts={counts.tolist()}",
            },
            "split_feasibility": {
                "ok": bool(feasibility), "severity": "block",
                "detail": (f"discovery={disc_counts.tolist()} (need >= {cv_splits}/class), "
                           f"confirmation={conf_counts.tolist()}, "
                           f"holdout={hold_counts.tolist()} (need >= 2/class)"),
            },
            "missing_values": {
                "ok": bool(np.isfinite(self.X).all()),
                "severity": "block", "detail": "non-finite features",
            },
            "small_sample_ratio": {
                "ok": len(self.y) >= 4 * self.n_features, "severity": "warn",
                "detail": f"n={len(self.y)} vs p={self.n_features}; n/p < 4",
            },
            "low_power": {
                "ok": len(self.y) >= 100, "severity": "warn",
                "detail": (f"n={len(self.y)}: inference is low-power; this is the "
                           "phenomenon under study, not a data defect"),
            },
        })

    # ---------------------------------------------------------------- protocol
    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        model_cfg = config.get("training", {})
        model_spec = {
            "model": "LogisticRegression",
            "C": float(model_cfg.get("C", 1.0)),
            "max_iter": int(model_cfg.get("max_iter", 5000)),
            "predicate_registry": "small_sample_v1",
        }
        data_config = {
            "generator": self.mode,
            "seed": self.scenario_seed,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "signal_strength": self.signal_strength,
            "threshold_quantile": self.threshold_quantile,
            "class_balance": self.class_balance,
            "holdout_fraction": self.holdout_fraction,
            "discovery_fraction": self.discovery_fraction,
            "confirmation_mode": self.confirmation_mode,
        }
        return ExperimentProtocol(
            task_id=self.task_id,
            domain="small_sample",
            primary_metric=self._evaluation.primary_metric,
            direction=self._evaluation.direction,
            folds=self._folds,
            confirmation_folds=self._confirmation,
            confirmation_kind=("independent_split"
                               if self.confirmation_mode == "independent" else "resampling"),
            holdout_fold=self._holdout,
            baseline_variant="baseline:logreg",
            control_variant=None,
            seeds=[int(x) for x in config.get("seeds", [self.seed])],
            model_spec=model_spec,
            data_config=data_config,
            evaluation=self._evaluation,
        )

    @property
    def evaluation(self) -> EvaluationSpec:
        return self._evaluation

    # -------------------------------------------------------------- candidates
    def validate_descriptor(self, kind: str, descriptor: dict[str, Any]) -> None:
        validate_small_sample_descriptor(descriptor, self.n_features)
        expected = {
            "ratio": {"statistical"},
            "difference": {"statistical"},
            "product": {"statistical"},
            "ratio_threshold": {"conditional"},
            "threshold": {"conditional"},
            "sigmoid": {"soft"},
            "interaction": {"relational"},
        }[descriptor["op"]]
        if kind not in expected:
            raise DescriptorError(f"op {descriptor['op']} must have kind in {sorted(expected)}")

    def make_generator(self, config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
        return SmallSampleRuleGenerator(
            config, self.feature_names, self.n_features,
            direction=self._evaluation.direction,
        )

    # ------------------------------------------------------------------- probe
    def probe(self, descriptor: dict[str, Any], fold: FoldSpec, snapshot: DataSnapshot,
              config: dict[str, Any], workdir: str) -> dict[str, Any]:
        try:
            train_idx = np.asarray(fold.metadata["train_idx"], dtype=int)
            Xtr = self.X[train_idx]
            col, _, params = _fit_transform(descriptor, Xtr, Xtr[:1])
            v = col[:, 0]
            if not np.isfinite(v).all():
                return {"ok": False, "reason": "non-finite predicate values",
                        "severity": "invalid"}
            if float(np.std(v)) < 1e-12:
                return {"ok": False, "reason": "constant predicate on training fold",
                        "severity": "invalid"}
            corr = []
            for j in range(min(6, self.n_features)):
                sd = float(np.std(Xtr[:, j]))
                corr.append(float(np.corrcoef(v, Xtr[:, j])[0, 1]) if sd > 0 else 0.0)
            sig = [float(np.mean(v)), float(np.std(v)),
                   float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-12)),
                   *[0.0 if not math.isfinite(c) else c for c in corr]]
            return {"ok": True, "signature": sig,
                    "checks": {"std": float(np.std(v)), "fitted": params},
                    "reason": "ok"}
        except Exception as exc:  # probe failures invalidate, never crash the round
            return {"ok": False, "reason": f"probe failed: {type(exc).__name__}: {exc}",
                    "severity": "invalid"}

    # ------------------------------------------------------------------ trials
    def _predict(self, desc: dict[str, Any] | None, train_idx: np.ndarray,
                 valid_idx: np.ndarray, seed: int,
                 training_config: dict[str, Any]) -> tuple[dict, dict, dict]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        Xtr, ytr = self.X[train_idx], self.y[train_idx]
        Xva, yva = self.X[valid_idx], self.y[valid_idx]
        fitted: dict[str, Any] = {"model": "LogisticRegression",
                                  "C": float(training_config.get("C", 1.0))}
        if desc is not None:
            col_tr, col_va, params = _fit_transform(desc, Xtr, Xva)
            Xtr = np.hstack([Xtr, col_tr])
            Xva = np.hstack([Xva, col_va])
            fitted.update(params)
        if len(np.unique(ytr)) < 2:
            raise ValueError("training fold contains a single class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(training_config.get("C", 1.0)),
                max_iter=int(training_config.get("max_iter", 5000)),
                solver="lbfgs",
                random_state=seed,
            ),
        )
        model.fit(Xtr, ytr)
        prob = np.asarray(model.predict_proba(Xva)[:, 1], dtype=float)
        pred = (prob >= 0.5).astype(int)
        auc = (float(roc_auc_score(yva, prob))
               if len(np.unique(yva)) >= 2 else 0.5)
        metrics = {
            "neg_brier": float(-np.mean((prob - yva) ** 2)),
            "auc": auc,
            "accuracy": float(accuracy_score(yva, pred)),
            "log_loss": float(log_loss(yva, np.clip(prob, 1e-7, 1.0 - 1e-7),
                                       labels=[0, 1])),
            "n_valid": int(len(yva)),
        }
        predictions = {
            "sample": np.asarray(valid_idx, dtype=np.int64),
            "score": prob,
            "label": np.asarray(yva, dtype=np.int64),
        }
        return metrics, fitted, predictions

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        import time

        t0 = time.time()
        meta = request.fold.metadata
        train_idx = np.asarray(meta["train_idx"], dtype=int)
        valid_idx = np.asarray(meta["valid_idx"], dtype=int)
        desc = request.descriptor if request.variant == "candidate" else None
        try:
            metrics, fitted, predictions = self._predict(
                desc, train_idx, valid_idx, request.seed, request.training_config
            )
        except Exception as exc:
            return {"status": "failed", "metrics": {}, "metric_series": {},
                    "fitted_params": {}, "predictions_path": "",
                    "duration_s": time.time() - t0,
                    "error": f"{type(exc).__name__}: {exc}", "worker": {}}

        pred_path = ""
        if request.save_predictions:
            out = Path(request.workdir) / "predictions.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            with tmp.open("wb") as fh:
                np.savez_compressed(fh, **predictions)
            tmp.replace(out)
            pred_path = str(out)

        # The statistical unit is the sample, not the fold: sample indices are
        # globally unique and each sample appears in exactly one fold per phase.
        series = {f"s{int(idx):06d}": float(-(float(p) - float(t)) ** 2)
                  for idx, p, t in zip(predictions["sample"],
                                       predictions["score"], predictions["label"])}
        return {
            "status": "success",
            "metrics": metrics,
            "metric_series": {"neg_brier": series},
            "fitted_params": fitted,
            "fit_data_hash": sha256_obj({
                "train_idx": train_idx.tolist(),
                "dataset": self.dataset_id,
                "descriptor": desc,
            }),
            "predictions_path": pred_path,
            "duration_s": time.time() - t0,
            "error": "",
            "worker": {"variant": request.variant, "fold": request.fold.fold_id,
                       "in_process": True},
        }

    # -------------------------------------------------------------- analysis
    def predicate_on_pool(self, descriptor: dict[str, Any]) -> np.ndarray:
        """Fit a descriptor on all non-holdout samples and return its column.

        Used only by the post-hoc analysis harness to score whether a validated
        candidate is behaviourally equivalent to the known latent signal.  It
        never runs during discovery, confirmation or finalization.
        """
        pool_idx = np.setdiff1d(np.arange(len(self.y), dtype=int), self._holdout_idx)
        col, _, _ = _fit_transform(descriptor, self.X[pool_idx], self.X[pool_idx])
        return col[:, 0]

    def recovery_correlation(self, descriptor: dict[str, Any]) -> float | None:
        if self.truth_indicator is None:
            return None
        pool_idx = np.setdiff1d(np.arange(len(self.y), dtype=int), self._holdout_idx)
        col = self.predicate_on_pool(descriptor)
        target = self.truth_indicator[pool_idx]
        if float(np.std(col)) < 1e-12 or float(np.std(target)) < 1e-12:
            return None
        return float(abs(np.corrcoef(col, target)[0, 1]))

    def recovery_auc(self, descriptor: dict[str, Any]) -> float | None:
        """AUC of the predicate column as a score for the true indicator.

        Unlike the family key, this credits a *monotone equivalent* predicate
        (e.g. ``ratio`` for a ``ratio_threshold`` truth) with full recovery.
        ``abs`` folds the direction so that a negated predicate still scores
        1.0 when it separates the truth perfectly.
        """
        if self.truth_indicator is None:
            return None
        from sklearn.metrics import roc_auc_score

        pool_idx = np.setdiff1d(np.arange(len(self.y), dtype=int), self._holdout_idx)
        col = self.predicate_on_pool(descriptor)
        target = self.truth_indicator[pool_idx]
        if len(np.unique(target)) < 2 or float(np.std(col)) < 1e-12:
            return None
        auc = float(roc_auc_score(target, col))
        return float(max(auc, 1.0 - auc))

    def family_of(self, descriptor: dict[str, Any]) -> str:
        return descriptor_family(descriptor)

    def true_family(self) -> str | None:
        return truth_family(self.truth)

    def matches_truth(self, descriptor: dict[str, Any]) -> bool:
        return matches_true_family(descriptor, self.truth)
