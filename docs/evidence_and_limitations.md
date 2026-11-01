# Evidence model, reproducibility and honest limitations

> **V1.1 note.**  `validated` is a mechanical stage-local status, not a
> scientific claim.  Candidate `state` (`exploratory_positive`,
> `resampling_confirmed`, `independently_confirmed`, `insufficient`,
> `rejected`, `legacy_unknown`) and evidence `stage` carry that meaning.  The
> normative rules, including the global holdout ledger and label isolation, are
> in [the V1.1 evaluation protocol](v1.1_evaluation_protocol.md).

## 1. Every candidate leaves evidence

`EvidenceStatus` is exhaustive:

| status | meaning |
|---|---|
| `validated` | expected-sign effect ≥ `min_effect`, stability ≥ `min_stability`, raw p ≤ `fdr_alpha`, survives BH |
| `rejected` | significant effect in the wrong direction |
| `insufficient_evidence` | executed, but effect/stability/significance did not meet the pre-registered bar (or BH downgraded it) |
| `screened_out` | passed structure/probe, not promoted from the cheap screen |
| `duplicate` | same content hash or behavioural duplicate (probe signature correlation ≥ 0.995) |
| `invalid_structure` | static descriptor validation failed |
| `execution_failed` | the trial failed; the error is stored |
| `blocked` | protocol/data blocker |

The store keeps all of them.  Reports show counts and the full `evidence.jsonl`
is exported for every run.  The next round's generator reads validated,
insufficient and screened-out rows; it never reads the holdout.

## 2. Statistics

For each promoted candidate the engine pools the per-fold series of the primary
metric and computes the paired difference against the baseline:

* **effect** = mean paired difference (candidate − baseline) on common units
  (trading dates for TSIL, CV folds for tabular);
* **uncertainty** = moving-block bootstrap CI (preserves short-range
  dependence) and Newey-West HAC standard error;
* **p-value** = two-sided normal approximation of `effect / HAC_SE`;
* **q-value** = Benjamini-Hochberg across the promoted candidates in the round;
* **stability** = fraction of folds with the pre-registered sign.

A raw p-value alone never validates a candidate.  `min_effect`,
`min_stability`, `fdr_alpha` and the metric direction are read from the config
and recorded in `context.json` / the protocol hash inputs.

Known limitations of the statistics:

* the HAC p-value uses a normal approximation, not a small-sample t
  distribution;
* for tabular repeated CV the "units" are folds, so the bootstrap treats folds
  as exchangeable; this is appropriate for the architecture check but is not a
  substitute for a hierarchical model;
* the tabular holdout is one split, so `finalize` reports a point estimate
  only; the TSIL holdout has 403 daily units, so a paired daily CI is
  computed, but a single holdout period is still one regime and cannot
  establish generalisation.  The TSIL holdout is evaluated once, for one
  frozen candidate, and is not reused to test other candidates.

## 3. Reproducibility

* Protocol/config identity is hashed; a trial is cached by
  `(candidate, fold, protocol_hash, config_hash)`.  Changing the budget, seed
  or fold produces a different key.
* Descriptor identity is content-addressed.  Fitted thresholds are stored per
  fold with `fit_data_hash`.
* The TSIL worker writes `job.json`, `worker.log` and `result.json` for every
  trial; the seed, GPU id, epochs and step cap are in the worker metadata.
* The snapshot manifest is metadata-level for the large qlib feature store
  (size/mtime + eight sampled SHA-256 hashes) and content-level for calendar
  and instrument files.  This is stated in `manifest_method`.

## 4. What is intentionally not implemented

* **Automatic data discovery.**  `DataProvider`, `DataContract`, snapshot and
  suitability interfaces exist; no crawler or schema-matching system does.
* **Mineral / image algorithms.**  Only contracts, validators and examples.
  All execution raises `DomainNotImplemented`.
* **LLM execution.**  The `llm_json` generator accepts structured proposals,
  but there is no LLM API in this environment and no LLM result is claimed.
  The executed runs use the rule generator.
* **Distributed scheduling / external experiment stores.**  Not needed for
  V1.0; the SQLite store and one subprocess per TSIL trial are sufficient.
* **Portfolio backtests / transaction costs.**  TSIL metrics are daily
  cross-sectional IC/RankIC/MSE/MAE; no backtest was run.

## 5. Environment-specific caveats

* **No `research_stack` directory exists** in the supplied repository.  Only
  `TSIL-main/` was available.  The platform therefore reuses TSIL directly and
  builds the core/state/evidence layer fresh; there was no pre-existing
  executor or candidate structure to adopt.
* **Shared GPU.**  The 8×A100 node was at 100% utilisation by other jobs.
  Reduced-profile TSIL trials took 55–110 s each.  Throughput, not semantics,
  was affected.
* **Reduced TSIL budget.**  `tsil_lstm_quick.yaml` uses 3/8 epochs and
  `d_model=32` for screening/full validation.  Results are conditional on that
  budget and are not a reproduction of the paper.  The full-budget config is
  provided for a dedicated allocation.
* **Tabular is a fixture.**  Its effect threshold (0.0005 log-loss) was chosen
  on the search-split scale to exercise validation/finalize, not as a
  scientific effect size.  Its holdout result is a single split and is reported
  as a point estimate only.
* **Universe.**  CSI300 membership comes from the supplied qlib store; no
  point-in-time index reconstruction or survivorship correction was performed.

## 6. How to read a report

1. Check `context.json`: domain, snapshot hash, protocol hash, suitability,
   policy thresholds.
2. Check `report.json` / `evidence.jsonl`: which candidates were validated,
   rejected, inconclusive, screened out, duplicate or invalid, and why.
3. Check the screen table: ties or saturation (e.g. AUC = 1.0) indicate the
   screen metric may be too coarse for the task.
4. Check `final_evaluation.json`: it must exist only after a frozen manifest,
   and it must name the frozen candidate/protocol/config hashes.
5. Treat a small effect with a wide CI as inconclusive, regardless of its rank.
