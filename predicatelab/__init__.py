"""PredicateLab: a generic predicate discovery and evidence-validation platform.

The package is split into

* :mod:`predicatelab.core`  -- domain-agnostic research engine (schema, state,
  protocol, statistics, evidence, scheduling);
* :mod:`predicatelab.domains` -- domain adapters (TSIL, tabular, mineral, image).

V1.0 implements the TSIL adapter end to end and a generic tabular adapter that
validates the core independently.  Mineral prospectivity and image
classification only provide extension contracts.
"""

__version__ = "1.0.0"

from predicatelab.core.schema import (  # noqa: F401
    BoundCandidate,
    CandidateKind,
    DataContract,
    DataSnapshot,
    EvaluationSpec,
    EvidenceRecord,
    EvidenceStatus,
    ExperimentProtocol,
    FittedInstance,
    FoldSpec,
    PredicateTemplate,
    ResearchTask,
    SearchPolicy,
)
