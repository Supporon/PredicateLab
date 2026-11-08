# Extending PredicateLab to a new domain

A domain adapter is the only place where domain knowledge enters the system.
It must supply a data contract, an immutable snapshot, a fold protocol, a
closed descriptor validator/compiler, a candidate generator, a probe and a
trial runner.  The core owns scheduling, statistics and evidence.

## 1. Contract

```python
from predicatelab.core.adapter import CandidateGenerator, DomainAdapter, TrialRequest
from predicatelab.core.schema import (
    DataContract, DataSnapshot, EvaluationSpec, ExperimentProtocol,
)

class MyAdapter(DomainAdapter):
    name = "my_domain"

    def data_contract(self, config) -> DataContract: ...
    def snapshot(self, config) -> DataSnapshot: ...
    def suitability(self, snapshot, config) -> dict: ...
    def build_protocol(self, config) -> ExperimentProtocol: ...
    @property
    def evaluation(self) -> EvaluationSpec: ...
    def validate_descriptor(self, kind: str, descriptor: dict) -> None: ...
    def probe(self, descriptor, fold, snapshot, config, workdir) -> dict: ...
    def make_generator(self, config, policy) -> CandidateGenerator: ...
    def run_trial(self, request: TrialRequest) -> dict: ...
```

`run_trial` returns:

```python
{
  "status": "success" | "failed",
  "metrics": {"<primary_metric>": 0.12, ...},
  "metric_series": {"<primary_metric>": {"<unit>": value, ...}},
  "fitted_params": {...},          # fold-local fitted state, optional
  "fit_data_hash": "...",          # hash of rows used for fitting
  "predictions_path": "...",
  "duration_s": 12.3,
  "error": "",
  "worker": {...},
}
```

The engine pairs `metric_series` between candidate and baseline by unit key
(trading date for TSIL, CV fold key for tabular).  Duplicate unit keys across
folds are rejected rather than silently overwritten.

## 2. Descriptor discipline

Descriptors are JSON.  A validator must reject anything the compiler cannot
execute, and the compiler must be a closed registry of pure functions:

* no `eval`/`exec` of descriptors or generated strings;
* no imports selected by the descriptor;
* bounded loops and integer ranges;
* unknown keys rejected, not ignored.

This is what allows an LLM (or a rule generator) to propose candidates while
the deterministic program remains the only execution path.  See
`predicatelab/domains/tsil/dsl.py` for a complete example: native predicates
are delegated by a whitelisted name, DSL families are implemented locally, and
composites are a validated tree of those parts.

## 3. Fitted state and folds

Anything data-dependent (a quantile threshold, a standardisation reference, a
cluster centre) must be fitted inside `run_trial` on that fold's training rows
and returned in `fitted_params` together with `fit_data_hash`.  It must not be
stored in the template or bound candidate, and it must not be reused across
folds.  The tabular adapter's `fit_transform` is a compact example.

For time-series, use `predicatelab.core.protocol.effective_time_fold` to apply
purge and embargo on the real calendar, and put the holdout in
`ExperimentProtocol.holdout_fold`, never in `folds`.  For grouped data (images
by scene, spatial data by block), build the folds so that no group appears in
both training and validation.

## 4. Data provider and suitability

`DataProvider` (`describe`, `snapshot`, `assess`) is the reserved interface for
automatic data discovery.  In V1.0 adapters implement the same three operations
directly.  `core.data.assess_common` aggregates checks into
`{status: ok|warn|block, checks, warnings, blockers}`; a `block` should stop
execution, a `warn` is recorded in `context.json` and the report.

A snapshot must state its method honestly.  Hashing a 20 GB binary store on
every run is usually not appropriate; `metadata-manifest(size+mtime_ns)` plus
sampled content hashes is an explicit, auditable compromise.  If a domain needs
byte-level identity, hash the selected files.

## 5. Mineral prospectivity (reserved)

`predicatelab/domains/mineral/adapter.py` defines five families:

| family | kind | required fields |
|---|---|---|
| `geochemical_anomaly` | conditional / statistical | `variable`, `transform`, `threshold_quantile` |
| `geophysical_response` | statistical | `variable`, `window_m`, `statistic` |
| `spatial_proximity` | soft / relational | `target`, `max_distance_km`, `decay` |
| `structural_constraint` | constraint | `feature`, `relation`, `distance_m` |
| `multi_source_relation` | relational | `sources`, `relation` |

`validate_mineral_descriptor` checks types and ranges; `example_templates()`
returns one valid example per family (`python -m predicatelab.cli examples
--domain mineral`).  Every execution method raises `DomainNotImplemented`.
To implement the domain later:

1. replace `snapshot`/`suitability` with a real spatial data manifest and
   checks (CRS, nodata, coverage, label definition, spatial blocking);
2. define a block/spatial CV protocol (never random split of neighbouring
   cells);
3. implement a closed descriptor compiler and a trial runner (e.g. gradient
   boosting or a geological model);
4. define an evaluation spec (AUC/precision at a prospectivity threshold,
   lift over a random baseline) and register it in the adapter.

No mineral result in this repository is real; the adapter is a contract.

## 6. Image classification (reserved)

`predicatelab/domains/image/adapter.py` defines `color_statistic`,
`texture_descriptor`, `soft_visual_predicate`, `spatial_relation` and
`visual_constraint`, with validators and examples.  Execution is reserved.
When implementing, split by scene/patient/group, not by image, to avoid
near-duplicate leakage.  A future evaluation would use accuracy/AUC with a
grouped CV protocol, and soft predicates would produce continuous evidence
rather than a boolean mask.

## 7. Registering a domain

Add a branch to `predicatelab.domains.create_adapter`, or use
`DomainRegistry.register(name, factory)`.  List executable domains in
`executable_domains()` only when they can actually run; reserved domains stay
in `reserved_domains()` so the CLI can advertise them without implying results.

## 8. Checklist for a new executable adapter

- [ ] data contract documents label window and leakage risks;
- [ ] snapshot method is explicit and stable;
- [ ] suitability returns a `block` when the data cannot support the protocol;
- [ ] validation folds and holdout are disjoint (grouped or time-ordered);
- [ ] descriptor validator and compiler are closed and side-effect free;
- [ ] probe reads training data only and returns a behavioural signature;
- [ ] every fitted quantity is fold-local and recorded with `fit_data_hash`;
- [ ] candidate and baseline use identical data, budget and seed;
- [ ] primary metric and direction are pre-registered;
- [ ] tests cover descriptor rejection, fold hygiene and one end-to-end round.
