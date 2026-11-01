# PredicateLab architecture (V1.1)

> **V1.1 addendum.**  Discovery, confirmation and final holdout are now three
> separate stages (`freeze-family` → `confirm` → `finalize`); holdout access is
> governed by a global, data-keyed ledger (`core/holdout.py`); label windows
> are resolved against the real trading calendar (`core/label_window.py`); and
> complete experiment identity, content manifests and atomic writes live in
> `core/identity.py`.  The normative rules are in
> [the V1.1 evaluation protocol](v1.1_evaluation_protocol.md); the V1.0 text
> below is retained for architecture orientation where it does not contradict
> that protocol.

PredicateLab is a modular monolith: one Python package, one SQLite state store,
and a small process boundary only where a domain needs it (TSIL training runs
in a subprocess).  The core is domain-agnostic; every domain-specific decision
lives in an adapter.

```
                    ┌──────────────────────────────────────────┐
                    │              core (domain-neutral)       │
  config.yaml ────► │ engine ─ protocol ─ stats ─ store        │
                    │   │        │          │        │         │
                    │   │        │          │        └─ SQLite │
                    │   │        │          └─ evidence/stats  │
                    │   └─ schema (template / bound / fitted)  │
                    └───────────────┬──────────────────────────┘
                                    │ DomainAdapter
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
        TSIL adapter          tabular adapter       mineral / image
     (qlib + LSTM + LUSI)   (sklearn logistic)     (contracts only)
              │
     subprocess worker
```

## 1. Core modules

| module | responsibility |
|---|---|
| `core/schema.py` | Pydantic models: `ResearchTask`, `DataContract`, `DataSnapshot`, `PredicateTemplate`, `BoundCandidate`, `FittedInstance`, `FoldSpec`, `ExperimentProtocol`, `SearchPolicy`, `TrialRecord`, `EvidenceRecord`, `RoundRecord`, `FinalEvaluation`. |
| `core/store.py` | SQLite (WAL) state: tasks, snapshots, templates, candidates, fitted instances, runs, rounds, trials, attempts, evidence, feedback, final evaluations, frozen families and the holdout ledger.  Trial lookup is `(candidate, fold, protocol_hash, config_hash)`, where `config_hash` covers the complete experiment identity. |
| `core/protocol.py` | Trading-day aware purge/embargo, fold-overlap checks and `HoldoutGuard`. |
| `core/holdout.py` | Global data-keyed holdout ledger: atomic claim, consumed/crash semantics, exact retry, exploratory reuse. |
| `core/identity.py` | Full content manifests and Merkle roots, code/environment fingerprints, strict JSON, atomic writes. |
| `core/label_window.py` | Label-expression offset parsing, real-calendar tail trimming, boundary leak counting. |
| `core/stats.py` | Newey-West HAC standard errors, moving-block bootstrap CI, paired effects, Benjamini-Hochberg q-values, stability. |
| `core/adapter.py` | `DomainAdapter`, `CandidateGenerator`, `DataProvider`, `DomainRegistry`, `TrialRequest`. |
| `core/data.py` | Snapshot manifests and shared data-suitability checks. |
| `core/engine.py` | The discovery/confirmation/final lifecycle, state machines, evidence classification and reports. |

The core never imports qlib, torch, sklearn or any domain code.  Adapters are
created by `predicatelab.domains.create_adapter(config)`.

## 2. Three-level candidate identity

```
PredicateTemplate   domain + kind + descriptor + schema_version   (content hash)
       │ bind to a dataset snapshot
BoundCandidate      template_hash + dataset_id + snapshot_hash      (binding hash)
       │ fit inside a training fold
FittedInstance      candidate_id + fold_id + fitted_params + fit_data_hash
```

* **Template** — "what the predicate says", independent of data.  Duplicate
  templates are collapsed by content hash regardless of display name.
* **Bound candidate** — "this predicate applies to this immutable snapshot".
  Thresholds and quantiles must not appear here.
* **Fitted instance** — "this candidate was fitted on this fold's training
  data".  The fit-data hash makes cross-fold reuse visible and auditable.

This is why a `threshold(mean texture, q=0.5)` candidate is defined once, but
each CV fold stores its own fitted threshold with the hash of the training rows
used to compute it.  A future adapter that needs richer fitted state can store
it in `fitted_params` without changing the core.

## 3. Candidate kinds

`CandidateKind` covers native predicates, statistical descriptors, soft/graded
predicates, conditional predicates, relational rules, constraints and
composites.  The core does not interpret the descriptor; it only stores it,
validates it through the adapter, and passes it to the adapter's trusted
compiler.  Continuous and soft predicates are therefore first-class, not
forced into a boolean `Predicate` abstraction.

## 4. Round lifecycle

A round is `propose → validate structure → bind → probe → screen → validate →
evidence → feedback`:

1. **Propose.**  The adapter's generator emits templates.  The rule generator
   is deterministic; `llm_json` reads LLM-written JSON.  Both emit *data*, never
   code.
2. **Static validation.**  The adapter's descriptor validator rejects unknown
   keys, out-of-range parameters, type errors and impossible compositions.
   Rejections are persisted as `invalid_structure` evidence.
3. **Bind and deduplicate.**  Content hash dedup runs first; after probing, a
   behavioural signature (strict upper triangle of the probe P matrix) removes
   candidates that are numerically almost identical to an existing candidate
   (`duplicate` evidence).
4. **Probe.**  A cheap training-fold execution checks shape, finiteness,
   symmetry and non-degeneracy.  It never sees validation or holdout labels.
5. **Screen.**  One cheap validation fold, paired against the baseline under
   the same budget.  Only the top-k with the expected sign are promoted; the
   rest are stored as `screened_out`.
6. **Full validation.**  All configured validation folds; every candidate and
   baseline is trained with the same profile and seed.
7. **Evidence.**  Pooled paired effect, bootstrap CI, HAC p-value, BH q-value,
   fold-level stability and the fold table are persisted.  Validated
   candidates that fail BH are downgraded; significant negative effects become
   `rejected`.
8. **Feedback.**  Validated, rejected, insufficient and screened-out rows feed
   the next round's generator.  Every candidate leaves a record.

Controls (`Phi_ones` in TSIL) are evaluated under the same conditions and
stored in the round summary, so the "any similarity matrix helps" explanation
can be checked.

## 5. Leakage controls

* The holdout fold is **not in `protocol.folds`**; it lives in
  `protocol.holdout_fold`.  Generators and the engine only receive search folds.
* `HoldoutGuard.assert_fold_accessible(fold_id, purpose)` raises
  `ProtocolViolation` if search code names the holdout.
* V1.1 separates the stages.  `freeze-family` records the exact discovery
  family; `confirm` evaluates only that set on confirmation folds (never fed
  back to the generator); `finalize` requires a `search_complete` run, a frozen
  family and `independently_confirmed`/`resampling_confirmed` evidence.  It
  writes `frozen_manifest.json`, claims the global holdout ledger atomically,
  and records a `FinalEvaluation` carrying `stage` and `state`.
* The holdout ledger is keyed by data identity (dataset, snapshot, full content
  manifest, label spec, test window), not run/candidate metadata.  Crashes
  count as consumption; only an exact retry returns the saved result.
* Fold construction applies purge and embargo.  TSIL uses expanding-window
  walk-forward validation on trading days; the label window
  (`Ref($close,-2)/Ref($close,-1)-1`) is two days, so the default purge is 2
  trading days.
* Fitted transformations are fold-local by construction; the TSIL worker fits
  `RobustZScoreNorm` on the training segment only, and the tabular adapter fits
  all thresholds on the fold's training rows.
* Trial results are cached by protocol and config hash, so a changed budget or
  fold cannot silently reuse an old metric.

Known limitation: "purge/embargo" are applied to the *segments*, but the
upstream TSIL `MTSDatasetH` builds fixed-length sequences and the handler
preprocessors are fit on the training segment.  The validation segment is never
used for fitting.  The holdout's `test` segment is only used by `finalize`.

## 6. Evaluation routing

`EvaluationSpec` records metric name, direction, paired design and the
multiple-testing method.  V1.0 implements:

* `tsil_rolling_ic` — daily cross-sectional IC on each validation fold, paired
  against the MSE baseline;
* `tabular_cv_logloss` — repeated stratified CV log-loss (AUC/accuracy as
  secondary), paired against a no-predicate logistic baseline.

The engine routes by the adapter's `evaluation` property and by the metric key
emitted in `metric_series`.  Other evaluation names can be added without
touching the core.

## 7. Reproducibility and state

* Every run writes `context.json` (task, protocol, snapshot, policy, config),
  `report.json`, `report.md`, `evidence.jsonl` and per-trial `job.json` /
  `worker.log` / `result.json`.
* `StateStore` is WAL-mode SQLite.  A failed trial can be retried; a successful
  one is never re-executed for the same identity.
* The data snapshot is a manifest, not a copy: calendar and instrument files
  are SHA-256 hashed, the feature store is represented by size/mtime plus eight
  sampled file hashes.  `manifest_method` says explicitly that this is a
  metadata-level manifest for large binary files.

## 8. Not in V1.0

* Automatic data discovery (interfaces `DataProvider`, data contract,
  suitability are present).
* Distributed scheduling, multi-agent orchestration, external stores
  (MLflow/DVC/Prefect/Ray) — deliberately deferred.
* Mineral-prospectivity and image-classification algorithms (contracts and
  examples only; all execution raises `DomainNotImplemented`).
* A general expression parser for arbitrary Boolean logic; the DSL is a closed
  operator registry.
