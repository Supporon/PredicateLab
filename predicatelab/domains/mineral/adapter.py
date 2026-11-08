"""Mineral-prospectivity domain — **reserved interface only** (V1.0).

No geological, geochemical, geophysical or spatial algorithm is implemented.
What *is* implemented:

* descriptor contracts for the five candidate kinds;
* static validation of those descriptors;
* example templates showing how a future adapter would plug into the core;
* an explicit ``DomainNotImplemented`` on every execution path.

This keeps the extension surface honest and testable without fabricating
mineral-prospectivity results.
"""

from __future__ import annotations

from typing import Any

from predicatelab.core.adapter import (
    CandidateGenerator,
    DomainAdapter,
    DomainNotImplemented,
    TrialRequest,
)
from predicatelab.core.schema import (
    CandidateKind,
    DataContract,
    DataSnapshot,
    EvaluationSpec,
    ExperimentProtocol,
    FoldSpec,
    PredicateTemplate,
    Provenance,
)

FAMILIES = {
    "geochemical_anomaly": {
        "kind": {CandidateKind.CONDITIONAL, CandidateKind.STATISTICAL},
        "required": {"variable", "transform", "threshold_quantile"},
    },
    "geophysical_response": {
        "kind": {CandidateKind.STATISTICAL},
        "required": {"variable", "window_m", "statistic"},
    },
    "spatial_proximity": {
        "kind": {CandidateKind.SOFT, CandidateKind.RELATIONAL},
        "required": {"target", "max_distance_km", "decay"},
    },
    "structural_constraint": {
        "kind": {CandidateKind.CONSTRAINT},
        "required": {"feature", "relation", "distance_m"},
    },
    "multi_source_relation": {
        "kind": {CandidateKind.RELATIONAL},
        "required": {"sources", "relation"},
    },
}

TRANSFORMS = {"raw", "log", "zscore", "rank"}
DECAYS = {"linear", "exp"}
RELATIONS = {"within", "buffer", "intersects", "coincident", "lead_lag"}


class DescriptorError(ValueError):
    pass


def validate_mineral_descriptor(kind: str, descriptor: dict[str, Any]) -> None:
    if not isinstance(descriptor, dict):
        raise DescriptorError("descriptor must be an object")
    family = descriptor.get("family")
    if family not in FAMILIES:
        raise DescriptorError(f"unknown mineral family {family!r}")
    spec = FAMILIES[family]
    kinds = {k.value for k in spec["kind"]}
    if kind not in kinds:
        raise DescriptorError(f"family {family} requires kind in {sorted(kinds)}")
    missing = spec["required"] - set(descriptor)
    if missing:
        raise DescriptorError(f"missing fields: {sorted(missing)}")
    if family == "geochemical_anomaly":
        if descriptor["transform"] not in TRANSFORMS:
            raise DescriptorError("transform must be one of raw/log/zscore/rank")
        q = descriptor["threshold_quantile"]
        if not isinstance(q, (int, float)) or not (0.5 <= float(q) < 1.0):
            raise DescriptorError("threshold_quantile must be in [0.5,1)")
    if family == "geophysical_response":
        if not isinstance(descriptor["window_m"], int) or descriptor["window_m"] <= 0:
            raise DescriptorError("window_m must be a positive integer")
        if descriptor["statistic"] not in {"mean", "std", "gradient", "max"}:
            raise DescriptorError("unknown statistic")
    if family == "spatial_proximity":
        if not (0 < float(descriptor["max_distance_km"]) <= 1000):
            raise DescriptorError("max_distance_km must be in (0,1000]")
        if descriptor["decay"] not in DECAYS:
            raise DescriptorError("decay must be linear or exp")
    if family == "structural_constraint":
        if descriptor["relation"] not in RELATIONS:
            raise DescriptorError("unknown structural relation")
        if not isinstance(descriptor["distance_m"], int) or descriptor["distance_m"] < 0:
            raise DescriptorError("distance_m must be a non-negative integer")
    if family == "multi_source_relation":
        src = descriptor["sources"]
        if not isinstance(src, list) or len(src) != 2 or not all(isinstance(s, str) for s in src):
            raise DescriptorError("sources must be a list of two layer names")
        if descriptor["relation"] not in RELATIONS:
            raise DescriptorError("unknown multi-source relation")


class MineralAdapter(DomainAdapter):
    name = "mineral"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def data_contract(self, config: dict[str, Any]) -> DataContract:
        return DataContract(
            dataset_id="mineral-reserved",
            domain="mineral",
            description="Reserved mineral-prospectivity data contract (not implemented in V1.0).",
            entity_key="spatial_unit",
            time_key="",
            label_definition="presence / grade of mineralisation (to be supplied by the domain)",
            feature_semantics={
                "geochemistry": "multi-element assay layers",
                "geophysics": "gravity/magnetic/electrical survey layers",
                "remote_sensing": "alteration and structural lineament layers",
                "spatial": "distance to faults, intrusions, known deposits",
            },
            universe="spatial grid or geological units",
            frequency="spatial",
            time_range=("", ""),
            caveats=[
                "No data, model or algorithm is implemented in V1.0.",
                "All execution methods raise DomainNotImplemented.",
                "Deposit labels must be split spatially (blocked CV) to avoid leakage.",
            ],
        )

    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        raise DomainNotImplemented("mineral data snapshots are reserved for a future release")

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        return {"status": "block", "checks": {},
                "warnings": [], "blockers": ["mineral adapter is an interface stub in V1.0"]}

    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        raise DomainNotImplemented("mineral protocol/execution is reserved for a future release")

    @property
    def evaluation(self) -> EvaluationSpec:
        raise DomainNotImplemented("mineral evaluation is reserved for a future release")

    def validate_descriptor(self, kind: str, descriptor: dict[str, Any]) -> None:
        validate_mineral_descriptor(kind, descriptor)

    def probe(self, descriptor, fold: FoldSpec, snapshot, config, workdir) -> dict[str, Any]:
        raise DomainNotImplemented("mineral probe is reserved for a future release")

    def make_generator(self, config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
        raise DomainNotImplemented("mineral candidate generation is reserved for a future release")

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        raise DomainNotImplemented("mineral trial execution is reserved for a future release")

    # ------------------------------------------------------------ examples
    @staticmethod
    def example_templates() -> list[PredicateTemplate]:
        rows = [
            (CandidateKind.CONDITIONAL, "Geochemical Cu anomaly above regional q90",
             {"family": "geochemical_anomaly", "variable": "Cu_ppm", "transform": "log",
              "threshold_quantile": 0.9}),
            (CandidateKind.STATISTICAL, "Magnetic gradient within 500 m",
             {"family": "geophysical_response", "variable": "magnetics", "window_m": 500,
              "statistic": "gradient"}),
            (CandidateKind.SOFT, "Exponential proximity to known Cu deposits",
             {"family": "spatial_proximity", "target": "known_Cu_deposit",
              "max_distance_km": 25, "decay": "exp"}),
            (CandidateKind.CONSTRAINT, "Within 2 km of a major fault",
             {"family": "structural_constraint", "feature": "major_fault",
              "relation": "buffer", "distance_m": 2000}),
            (CandidateKind.RELATIONAL, "Coincident geochem and magnetic anomalies",
             {"family": "multi_source_relation", "sources": ["geochem_Cu", "magnetics"],
              "relation": "coincident"}),
        ]
        out = []
        for kind, name, desc in rows:
            validate_mineral_descriptor(kind.value, desc)
            out.append(PredicateTemplate(
                domain="mineral", kind=kind, name=name, descriptor=desc,
                rationale="example only; not executable in V1.0",
                provenance=Provenance(generator="example", rationale="extension contract example"),
            ))
        return out
