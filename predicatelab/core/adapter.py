"""Domain adapter contract.

The core knows *nothing* about tensors, qlib, sklearn or geology.  It only
knows how to schedule candidate generation, structural validation, probing,
screening, validation, statistics and evidence.  A domain adapter supplies:

* a data contract and an immutable snapshot identity;
* a structured descriptor validator (static validation);
* a trusted compiler / executor for descriptors;
* a fold protocol and trial runner;
* a candidate generator that may consume previous-round evidence.

Candidate descriptors are data (JSON), never code.  The adapter is the only
place allowed to turn a descriptor into an executable predicate, and it does so
through a fixed operator registry.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from predicatelab.core.hashing import sha256_obj
from predicatelab.core.schema import (
    DataContract,
    DataSnapshot,
    EvaluationSpec,
    ExperimentProtocol,
    EvidenceRecord,
    FoldSpec,
    PredicateTemplate,
    ResearchTask,
)


class DomainNotImplemented(NotImplementedError):
    """Raised by reserved adapters (mineral, image) in V1.0."""


@dataclass
class TrialRequest:
    candidate_id: str
    variant: str  # "candidate" | "baseline" | "control" | "control:<name>"
    descriptor: dict[str, Any]
    fold: FoldSpec
    seed: int
    training_config: dict[str, Any]
    workdir: str
    protocol_hash: str
    config_hash: str
    save_predictions: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    # ---- V1.1 identity: every field below changes cached numbers -----------
    phase: str = "discovery"  # discovery | confirmation | final
    snapshot_hash: str = ""
    data_manifest_hash: str = ""
    label_spec_hash: str = ""
    preprocessing_version: str = "v1"
    model_spec: dict[str, Any] = field(default_factory=dict)
    code_identity: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    attempt: int = 1
    label_expression: str = ""
    label_offset: int = 0
    control_kind: str = ""


class CandidateGenerator(ABC):
    """Turns research context into unbound, structured candidate templates."""

    @abstractmethod
    def propose(
        self,
        *,
        round_index: int,
        n: int,
        feedback: dict[str, Any] | None,
        known_template_hashes: set[str],
    ) -> list[PredicateTemplate]:
        raise NotImplementedError


class DomainAdapter(ABC):
    name: str = "abstract"

    # ------------------------------------------------------------------ data
    @abstractmethod
    def data_contract(self, config: dict[str, Any]) -> DataContract:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        """Build (or reuse) an immutable manifest of the data actually used."""
        raise NotImplementedError

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "checks": {}, "warnings": []}

    def trading_calendar(self, config: dict[str, Any]) -> list[str] | None:
        """Actual trading dates of the data; used for label-window isolation.

        Returning ``None`` is only acceptable when the label has no forward
        dependency.  The engine fails closed otherwise.
        """
        return None

    # -------------------------------------------------------------- protocol
    @abstractmethod
    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        raise NotImplementedError

    @property
    @abstractmethod
    def evaluation(self) -> EvaluationSpec:
        raise NotImplementedError

    # ------------------------------------------------------------- candidates
    @abstractmethod
    def validate_descriptor(self, kind: str, descriptor: dict[str, Any]) -> None:
        """Raise ValueError for anything the compiler will not accept."""
        raise NotImplementedError

    @abstractmethod
    def probe(
        self,
        descriptor: dict[str, Any],
        fold: FoldSpec,
        snapshot: DataSnapshot,
        config: dict[str, Any],
        workdir: str,
    ) -> dict[str, Any]:
        """Numeric smoke test on training-fold data.

        Returns a dict with at least ``ok`` (bool), ``reason`` (str) and, when
        ok, ``signature`` (list[float]) used for behavioural deduplication.
        The probe must only read training data.
        """
        raise NotImplementedError

    @abstractmethod
    def make_generator(
        self, config: dict[str, Any], policy: dict[str, Any]
    ) -> CandidateGenerator:
        raise NotImplementedError

    # ---------------------------------------------------------------- trials
    @abstractmethod
    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        """Execute one (candidate/baseline/control, fold) trial.

        Returns a dict with keys ``status``, ``metrics``, ``metric_series``,
        ``fitted_params``, ``predictions_path``, ``duration_s``, ``error``,
        ``worker``.  The engine converts it into a TrialRecord.
        """
        raise NotImplementedError

    def trial_config_hash(self, request: TrialRequest) -> str:
        """Full identity of one executable trial.

        V1.0 hashed only descriptor/fold/seed/training config, so a changed data
        snapshot, label expression, model budget, code revision or phase could
        reuse a stale cache.  V1.1 hashes all of them; every stage uses its own
        workdir keyed by this hash as well.
        """
        return sha256_obj(
            {
                "domain": self.name,
                "variant": request.variant,
                "control_kind": request.control_kind,
                "descriptor": request.descriptor,
                "fold": request.fold.model_dump(mode="json"),
                "phase": request.phase,
                "seed": request.seed,
                # ``attempt`` is deliberately excluded: a retry of the same
                # scientific configuration must find the same cache key.
                "training_config": request.training_config,
                "model_spec": request.model_spec,
                "snapshot_hash": request.snapshot_hash,
                "data_manifest_hash": request.data_manifest_hash,
                "label_spec_hash": request.label_spec_hash,
                "label_expression": request.label_expression,
                "label_offset": request.label_offset,
                "preprocessing_version": request.preprocessing_version,
                "code_fingerprint": (request.code_identity or {}).get("dirty_fingerprint"),
                "code_commit": (request.code_identity or {}).get("commit"),
                "environment_fingerprint": sha256_obj(request.environment or {}),
            }
        )

    def protocol_hash(self, protocol: ExperimentProtocol, config: dict[str, Any]) -> str:
        return protocol.content_hash(training_config=config.get("training", {}))


class DataProvider(ABC):
    """A data source plus its semantic mapping and suitability assessment.

    V1.0 adapters implement this internally; the interface exists so automatic
    data discovery can be added without changing the core.
    """

    @abstractmethod
    def describe(self) -> DataContract:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> DataSnapshot:
        raise NotImplementedError

    @abstractmethod
    def assess(self) -> dict[str, Any]:
        """Return {status, checks, warnings, blockers}."""
        raise NotImplementedError


class DomainRegistry:
    _factories: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, factory) -> None:
        cls._factories[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> DomainAdapter:
        if name not in cls._factories:
            raise KeyError(f"unknown domain {name!r}; registered: {sorted(cls._factories)}")
        return cls._factories[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._factories)
