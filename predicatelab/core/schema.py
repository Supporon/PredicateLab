"""Pydantic schemas shared by every domain.

The central idea is a three-level candidate identity:

``PredicateTemplate``
    The *unbound* structured knowledge: what the predicate says, independent of
    any dataset.  Its identity is the content hash of ``(domain, kind,
    descriptor)``.

``BoundCandidate``
    A template bound to a specific dataset snapshot.  Thresholds and other
    data-dependent parameters must NOT be stored here, because binding says
    "this predicate applies to these data", not "this predicate was fitted to
    these data".

``FittedInstance``
    A candidate fitted inside one training fold (e.g. a quantile threshold).
    Fitted parameters live here with the fold id and the hash of the fitting
    data, so they can never be reused across folds or snapshots by accident.

Anything that cannot be expressed as a boolean predicate is represented by a
``CandidateKind`` plus a domain-owned descriptor dictionary; the core never
interprets the descriptor.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from predicatelab.core.hashing import sha256_obj, short_id


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CandidateKind(str, Enum):
    """Kinds of candidate knowledge supported by the core.

    V1.0 uses ``native``/``statistical``/``soft``/``composite`` in TSIL and
    ``statistical``/``soft``/``conditional``/``relational`` in tabular; the
    remaining kinds are first-class in the schema so new domains need no core
    change.
    """

    NATIVE = "native"  # a registered domain predicate implementation
    STATISTICAL = "statistical"  # a statistical descriptor (moments, dispersion ...)
    SOFT = "soft"  # a continuous similarity / graded predicate
    CONDITIONAL = "conditional"  # an if-condition predicate (threshold fitted in-fold)
    RELATIONAL = "relational"  # a relation between entities or features
    CONSTRAINT = "constraint"  # a monotonicity / admissibility constraint
    COMPOSITE = "composite"  # a composition of other candidates


class EvidenceStatus(str, Enum):
    """Mechanical outcome of one validation attempt.

    ``VALIDATED`` means "passed the pre-registered statistical bar at this
    stage"; whether that licenses a scientific claim is recorded separately in
    :class:`CandidateState` and ``EvidenceRecord.stage``.  Every generated
    candidate ends in exactly one of these outcomes; the platform deliberately
    persists failures and inconclusive results instead of keeping only winners.
    """

    VALIDATED = "validated"
    REJECTED = "rejected"  # significant negative effect, or chosen rejection rule
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SCREENED_OUT = "screened_out"  # passed structure/probe, not promoted
    DUPLICATE = "duplicate"  # semantic or behavioural duplicate
    INVALID_STRUCTURE = "invalid_structure"  # static validation failed
    EXECUTION_FAILED = "execution_failed"
    BLOCKED = "blocked"  # data/protocol blocker (e.g. holdout access attempt)


class CandidateState(str, Enum):
    """What a candidate is *allowed to claim*, as distinct from its test result.

    V1.0 collapsed "passed on search folds" into ``validated``, which readers
    could mistake for independent confirmation.  V1.1 separates the two:

    ``executable``              compiles/runs; no useful effect measured
    ``exploratory_positive``    positive on discovery data; cannot be reported
                                as independently confirmed
    ``resampling_confirmed``    confirmed on disjoint resamples of the *same*
                                data (e.g. extra CV repeats); not new data
    ``independently_confirmed`` confirmed on later, selection-free data
    ``insufficient``            not enough evidence / coverage
    ``rejected``                significant wrong-direction effect
    ``legacy_unknown``          historical row whose stage cannot be proven
    """

    EXECUTABLE = "executable"
    EXPLORATORY_POSITIVE = "exploratory_positive"
    RESAMPLING_CONFIRMED = "resampling_confirmed"
    INDEPENDENTLY_CONFIRMED = "independently_confirmed"
    INSUFFICIENT = "insufficient"
    REJECTED = "rejected"
    LEGACY_UNKNOWN = "legacy_unknown"


class Provenance(BaseModel):
    generator: str = "rule"
    generator_version: str = "1"
    round_id: str | None = None
    parents: list[str] = Field(default_factory=list)
    seed_family: str | None = None
    rationale: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class PredicateTemplate(BaseModel):
    template_id: str = ""
    domain: str
    kind: CandidateKind
    name: str
    descriptor: dict[str, Any]
    rationale: str = ""
    provenance: Provenance = Field(default_factory=Provenance)
    schema_version: str = "1"
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def content_hash(self) -> str:
        return sha256_obj(
            {
                "domain": self.domain,
                "kind": self.kind.value,
                "descriptor": self.descriptor,
                "schema_version": self.schema_version,
            }
        )

    def ensure_id(self) -> "PredicateTemplate":
        if not self.template_id:
            self.template_id = short_id("tpl", self.content_hash)
        return self


class BoundCandidate(BaseModel):
    candidate_id: str = ""
    template_id: str
    template_hash: str
    domain: str
    kind: CandidateKind
    name: str
    descriptor: dict[str, Any]
    dataset_id: str
    snapshot_hash: str
    bound_scope: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def binding_hash(self) -> str:
        return sha256_obj(
            {
                "template_hash": self.template_hash,
                "dataset_id": self.dataset_id,
                "snapshot_hash": self.snapshot_hash,
                "bound_scope": self.bound_scope,
            }
        )

    def ensure_id(self) -> "BoundCandidate":
        if not self.candidate_id:
            self.candidate_id = short_id("cand", self.binding_hash)
        return self


class FittedInstance(BaseModel):
    """Fold-local fitted state of a candidate (thresholds, quantiles, ...)."""

    instance_id: str = ""
    candidate_id: str
    fold_id: str
    fitted_params: dict[str, Any]
    fit_data_hash: str
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def content_hash(self) -> str:
        return sha256_obj(
            {
                "candidate_id": self.candidate_id,
                "fold_id": self.fold_id,
                "fitted_params": self.fitted_params,
                "fit_data_hash": self.fit_data_hash,
            }
        )

    def ensure_id(self) -> "FittedInstance":
        if not self.instance_id:
            self.instance_id = short_id("fit", self.content_hash)
        return self


class EvaluationSpec(BaseModel):
    """How a domain turns predictions into evidence.

    The core routes by ``name`` to the adapter evaluator; V1.0 registers
    ``tsil_rolling_ic`` and ``tabular_cv_auc``.
    """

    name: str
    primary_metric: str
    direction: Literal["higher", "lower"] = "higher"
    secondary_metrics: list[str] = Field(default_factory=list)
    paired: bool = True
    multiple_testing: str = "benjamini-hochberg"
    notes: str = ""


class FoldSpec(BaseModel):
    """One time-ordered train/validation split.

    ``valid_*`` is the label period of the validation segment and is the only
    segment whose metrics may drive screening and candidate selection.
    ``test_*`` is optional and reserved for the frozen final evaluation.

    V1.1 adds ``role='confirmation'``.  Confirmation folds are *not* search
    folds: they are temporally later than every discovery fold, are never seen
    by the generator or the screener, and are evaluated only after the
    candidate family is frozen.
    """

    fold_id: str
    kind: Literal["time", "cv"] = "time"
    role: Literal["search", "confirmation", "holdout"] = "search"
    train_start: str = ""
    train_end: str = ""
    valid_start: str = ""
    valid_end: str = ""
    test_start: str | None = None
    test_end: str | None = None
    purge_days: int = 2
    embargo_days: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @property
    def is_holdout(self) -> bool:
        return self.role == "holdout"


class ExperimentProtocol(BaseModel):
    protocol_id: str = ""
    task_id: str
    domain: str
    primary_metric: str
    direction: Literal["higher", "lower"] = "higher"
    folds: list[FoldSpec] = Field(default_factory=list)
    confirmation_folds: list[FoldSpec] = Field(default_factory=list)
    holdout_fold: FoldSpec | None = None
    baseline_variant: str = "baseline:mse"
    control_variant: str | None = "control:ones"
    control_variants: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [2025])
    model_spec: dict[str, Any] = Field(default_factory=dict)
    data_config: dict[str, Any] = Field(default_factory=dict)
    evaluation: EvaluationSpec | None = None
    # "time_forward" (later data) or "resampling" (same data, different splits)
    confirmation_kind: str = "time_forward"
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def is_temporal(self) -> bool:
        return any(f.kind == "time" for f in list(self.folds) + list(self.confirmation_folds))

    def content_hash(self, training_config: dict[str, Any] | None = None) -> str:
        return sha256_obj(
            {
                "domain": self.domain,
                "primary_metric": self.primary_metric,
                "direction": self.direction,
                "folds": [f.model_dump(mode="json") for f in self.folds],
                "confirmation_folds": [f.model_dump(mode="json") for f in self.confirmation_folds],
                "confirmation_kind": self.confirmation_kind,
                "holdout": self.holdout_fold.model_dump(mode="json") if self.holdout_fold else None,
                "seeds": self.seeds,
                "model_spec": self.model_spec,
                "data_config": self.data_config,
                "training_config": training_config or {},
            }
        )

    def ensure_id(self) -> "ExperimentProtocol":
        if not self.protocol_id:
            self.protocol_id = short_id("prot", self.content_hash())
        return self


class SearchPolicy(BaseModel):
    generator: str = "rule"
    generator_config: dict[str, Any] = Field(default_factory=dict)
    max_candidates_per_round: int = 24
    screen_folds: list[str] = Field(default_factory=list)
    validation_folds: list[str] = Field(default_factory=list)
    full_validation_top_k: int = 4
    min_screen_effect: float = 0.0
    fdr_alpha: float = 0.10
    min_effect: float = 0.005
    min_stability: float = 0.60
    # V1.1: partial execution must not become a validated conclusion.
    min_fold_coverage: float = 0.75
    min_seed_agreement: float = 0.5
    bootstrap_samples: int = 1000
    bootstrap_block: int = 5
    nw_lag: int = 5
    seed: int = 2025
    save_predictions: bool = False

    @field_validator("fdr_alpha", "min_stability", "min_fold_coverage", "min_seed_agreement")
    @classmethod
    def _in_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0) or (isinstance(v, float) and math.isnan(v)):
            raise ValueError("must be in [0, 1]")
        return v


class DataContract(BaseModel):
    dataset_id: str
    domain: str
    description: str
    entity_key: str = "instrument"
    time_key: str = "datetime"
    label_definition: str = ""
    label_window: dict[str, Any] = Field(default_factory=dict)
    feature_semantics: dict[str, str] = Field(default_factory=dict)
    universe: str = ""
    frequency: str = "daily"
    time_range: tuple[str, str] = ("", "")
    preprocessing_version: str = "v1"
    label_dependency: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)


class DataSnapshot(BaseModel):
    snapshot_id: str = ""
    dataset_id: str
    domain: str
    manifest_method: str
    manifest_hash: str
    path: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def snapshot_hash(self) -> str:
        """Alias used wherever a stable data identity is needed."""
        return self.manifest_hash

    def ensure_id(self) -> "DataSnapshot":
        if not self.snapshot_id:
            self.snapshot_id = short_id("snap", self.manifest_hash)
        return self


class ResearchTask(BaseModel):
    task_id: str = ""
    title: str
    question: str
    domain: str
    hypothesis: str = ""
    evaluation: EvaluationSpec
    data_contract: DataContract
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def ensure_id(self) -> "ResearchTask":
        if not self.task_id:
            # Stable semantic identity: timestamps, run ids and ad-hoc metadata
            # must not create a new task identity on every invocation.
            semantic = {
                "domain": self.domain,
                "title": self.title,
                "question": self.question,
                "hypothesis": self.hypothesis,
                "evaluation": self.evaluation.model_dump(mode="json"),
                "data_contract": self.data_contract.model_dump(mode="json"),
            }
            self.task_id = short_id("task", sha256_obj(semantic))
        return self


class TrialRecord(BaseModel):
    trial_id: str = ""
    candidate_id: str
    fold_id: str
    protocol_hash: str
    config_hash: str
    status: Literal["running", "success", "failed", "interrupted", "corrupt", "skipped"] = "success"
    phase: str = "unknown"
    seed: int | None = None
    snapshot_hash: str = ""
    identity_hash: str = ""
    attempt: int = 1
    result_hash: str = ""
    predictions_sha256: str = ""
    null_paths: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metric_series: dict[str, Any] = Field(default_factory=dict)
    fitted_params: dict[str, Any] = Field(default_factory=dict)
    predictions_path: str = ""
    duration_s: float = 0.0
    error: str = ""
    worker: dict[str, Any] = Field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def trial_key(self) -> str:
        return sha256_obj(
            {
                "candidate_id": self.candidate_id,
                "fold_id": self.fold_id,
                "protocol_hash": self.protocol_hash,
                "config_hash": self.config_hash,
            }
        )

    def ensure_id(self) -> "TrialRecord":
        if not self.trial_id:
            self.trial_id = short_id("trial", self.trial_key)
        return self


class EvidenceRecord(BaseModel):
    evidence_id: str = ""
    candidate_id: str
    round_id: str
    protocol_hash: str
    status: EvidenceStatus
    stage: Literal["discovery", "confirmation", "final", "legacy"] = "discovery"
    state: CandidateState = CandidateState.LEGACY_UNKNOWN
    seed: int | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    n_folds: int = 0
    stability: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    def ensure_id(self) -> "EvidenceRecord":
        if not self.evidence_id:
            # Stable per (run-round, candidate, stage): a retry updates the same
            # row instead of appending duplicate evidence.
            self.evidence_id = short_id(
                "ev",
                sha256_obj(
                    {
                        "candidate_id": self.candidate_id,
                        "round_id": self.round_id,
                        "protocol_hash": self.protocol_hash,
                        "stage": self.stage,
                    }
                ),
            )
        return self


class RoundRecord(BaseModel):
    round_id: str = ""
    task_id: str
    run_id: str
    round_index: int
    generator: str
    parent_round_id: str | None = None
    status: Literal["pending", "running", "completed", "failed", "interrupted", "legacy_unknown"] = "pending"
    status_origin: str = "v1.1"
    n_proposed: int = 0
    n_bound: int = 0
    n_validated: int = 0
    n_rejected: int = 0
    n_insufficient: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    started_at: str = ""
    ended_at: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    def ensure_id(self) -> "RoundRecord":
        if not self.round_id:
            self.round_id = short_id(
                "round",
                sha256_obj(
                    {
                        "run_id": self.run_id,
                        "round_index": self.round_index,
                        "created_at": self.created_at.isoformat(),
                    }
                ),
            )
        return self


class FinalEvaluation(BaseModel):
    final_id: str = ""
    task_id: str
    run_id: str = ""
    candidate_id: str
    protocol_hash: str
    config_hash: str
    manifest_hash: str
    holdout_key: str = ""
    access_id: str = ""
    status: Literal["frozen", "success", "failed", "interrupted", "legacy_unknown"] = "frozen"
    stage: Literal["final", "legacy"] = "final"
    state: CandidateState = CandidateState.LEGACY_UNKNOWN
    exploratory: bool = False
    result_hash: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    def ensure_id(self) -> "FinalEvaluation":
        if not self.final_id:
            self.final_id = short_id(
                "final",
                sha256_obj(
                    {
                        "candidate_id": self.candidate_id,
                        "manifest_hash": self.manifest_hash,
                        "protocol_hash": self.protocol_hash,
                    }
                ),
            )
        return self
