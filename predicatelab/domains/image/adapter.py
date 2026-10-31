"""Image-classification domain — **reserved interface only** (V1.0).

Descriptor contracts and examples are provided; no vision model, feature
extractor or training loop is implemented.  Every execution method raises
``DomainNotImplemented`` so no image result can be fabricated.
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
    "color_statistic": {
        "kind": {CandidateKind.STATISTICAL},
        "required": {"channel", "statistic"},
    },
    "texture_descriptor": {
        "kind": {CandidateKind.STATISTICAL},
        "required": {"method", "radius", "bins"},
    },
    "soft_visual_predicate": {
        "kind": {CandidateKind.SOFT},
        "required": {"concept", "threshold", "margin"},
    },
    "spatial_relation": {
        "kind": {CandidateKind.RELATIONAL},
        "required": {"subject", "object", "relation"},
    },
    "visual_constraint": {
        "kind": {CandidateKind.CONSTRAINT},
        "required": {"rule", "tolerance"},
    },
}

CHANNELS = {"r", "g", "b", "gray"}
STATISTICS = {"mean", "std", "entropy", "skew", "histogram"}
TEXTURE_METHODS = {"lbp", "glcm", "gabor", "hog"}
RELATIONS = {"above", "below", "near", "inside", "left_of", "right_of"}


class DescriptorError(ValueError):
    pass


def validate_image_descriptor(kind: str, descriptor: dict[str, Any]) -> None:
    if not isinstance(descriptor, dict):
        raise DescriptorError("descriptor must be an object")
    family = descriptor.get("family")
    if family not in FAMILIES:
        raise DescriptorError(f"unknown image family {family!r}")
    spec = FAMILIES[family]
    if kind not in {k.value for k in spec["kind"]}:
        raise DescriptorError(f"family {family} requires kind in {[k.value for k in spec['kind']]}")
    missing = spec["required"] - set(descriptor)
    if missing:
        raise DescriptorError(f"missing fields: {sorted(missing)}")
    if family == "color_statistic":
        if descriptor["channel"] not in CHANNELS:
            raise DescriptorError("unknown channel")
        if descriptor["statistic"] not in STATISTICS:
            raise DescriptorError("unknown statistic")
    if family == "texture_descriptor":
        if descriptor["method"] not in TEXTURE_METHODS:
            raise DescriptorError("unknown texture method")
        if not isinstance(descriptor["radius"], int) or descriptor["radius"] <= 0:
            raise DescriptorError("radius must be a positive integer")
        if not isinstance(descriptor["bins"], int) or not (4 <= descriptor["bins"] <= 512):
            raise DescriptorError("bins must be in [4,512]")
    if family == "soft_visual_predicate":
        t, m = descriptor["threshold"], descriptor["margin"]
        if not (0.0 <= float(t) <= 1.0 and 0.0 < float(m) <= 0.5):
            raise DescriptorError("threshold in [0,1], margin in (0,0.5]")
    if family == "spatial_relation":
        if descriptor["relation"] not in RELATIONS:
            raise DescriptorError("unknown spatial relation")
    if family == "visual_constraint":
        if not isinstance(descriptor["rule"], str) or not descriptor["rule"]:
            raise DescriptorError("rule must be a non-empty string")
        if not (0.0 <= float(descriptor["tolerance"]) <= 1.0):
            raise DescriptorError("tolerance must be in [0,1]")


class ImageAdapter(DomainAdapter):
    name = "image"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def data_contract(self, config: dict[str, Any]) -> DataContract:
        return DataContract(
            dataset_id="image-reserved",
            domain="image",
            description="Reserved image-classification data contract (not implemented in V1.0).",
            entity_key="image",
            time_key="",
            label_definition="class label (to be supplied by the domain)",
            feature_semantics={
                "pixels": "RGB or grayscale image",
                "visual_features": "learned embeddings / handcrafted descriptors",
            },
            universe="image dataset",
            frequency="static",
            time_range=("", ""),
            caveats=[
                "No vision model, feature extractor or training loop is implemented in V1.0.",
                "All execution methods raise DomainNotImplemented.",
                "Candidate splits must be grouped by scene/patient to avoid near-duplicate leakage.",
            ],
        )

    def snapshot(self, config: dict[str, Any]) -> DataSnapshot:
        raise DomainNotImplemented("image data snapshots are reserved for a future release")

    def suitability(self, snapshot: DataSnapshot, config: dict[str, Any]) -> dict[str, Any]:
        return {"status": "block", "checks": {}, "warnings": [],
                "blockers": ["image adapter is an interface stub in V1.0"]}

    def build_protocol(self, config: dict[str, Any]) -> ExperimentProtocol:
        raise DomainNotImplemented("image protocol/execution is reserved for a future release")

    @property
    def evaluation(self) -> EvaluationSpec:
        raise DomainNotImplemented("image evaluation is reserved for a future release")

    def validate_descriptor(self, kind: str, descriptor: dict[str, Any]) -> None:
        validate_image_descriptor(kind, descriptor)

    def probe(self, descriptor, fold: FoldSpec, snapshot, config, workdir) -> dict[str, Any]:
        raise DomainNotImplemented("image probe is reserved for a future release")

    def make_generator(self, config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
        raise DomainNotImplemented("image candidate generation is reserved for a future release")

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        raise DomainNotImplemented("image trial execution is reserved for a future release")

    @staticmethod
    def example_templates() -> list[PredicateTemplate]:
        rows = [
            (CandidateKind.STATISTICAL, "Mean red-channel intensity",
             {"family": "color_statistic", "channel": "r", "statistic": "mean"}),
            (CandidateKind.STATISTICAL, "LBP texture radius 2",
             {"family": "texture_descriptor", "method": "lbp", "radius": 2, "bins": 32}),
            (CandidateKind.SOFT, "Soft 'smooth surface' predicate",
             {"family": "soft_visual_predicate", "concept": "smooth_surface",
              "threshold": 0.6, "margin": 0.1}),
            (CandidateKind.RELATIONAL, "Object above horizon",
             {"family": "spatial_relation", "subject": "object", "object": "horizon",
              "relation": "above"}),
            (CandidateKind.CONSTRAINT, "Brightness admissibility",
             {"family": "visual_constraint", "rule": "mean_brightness in [0.1,0.9]",
              "tolerance": 0.05}),
        ]
        out = []
        for kind, name, desc in rows:
            validate_image_descriptor(kind.value, desc)
            out.append(PredicateTemplate(
                domain="image", kind=kind, name=name, descriptor=desc,
                rationale="example only; not executable in V1.0",
                provenance=Provenance(generator="example", rationale="extension contract example"),
            ))
        return out
