# Generic tabular scenario — architecture validation result

> **V1.1 status: historical / legacy, not V1.1 evidence.**  The run below
> predates the stage separation and the global holdout ledger.  Its holdout is
> recorded as consumed with an unknown window; its stored single-point CI/p
> values were NaN and are not an inferential result.  A new confirmatory use of
> this dataset requires a fresh database or an explicit exploratory-reuse
> record.  See [the V1.1 remediation report](v1.1_remediation_report.md).

This scenario exists to prove that the core engine works without TSIL.  It is
**not** a scientific claim about breast-cancer classification.

## Setup

* data: `sklearn.datasets.load_breast_cancer` (569 samples, 30 features), real
  data bundled with scikit-learn;
* split: fixed stratified 20% holdout (seed 2025), 80% search pool;
* protocol: repeated stratified 5-fold CV, 5 repeats = 25 validation folds
  (each fold's scaler and predicate thresholds are fitted on its training
  rows only);
* baseline: logistic regression on the 30 raw features;
* candidate: the same pipeline with one folded-in predicate feature;
* primary metric: log-loss (lower is better).  AUC is saturated on this easy
  dataset (many folds have baseline AUC ≈ 0.999), so log-loss was registered as
  primary; AUC and accuracy are secondary.

Run: `run_20261031_153310`, protocol hash `445b170f1adaceb7`.

## Search result (search split only, holdout untouched)

| candidate | descriptor | effect (log-loss) | 95% CI | p | q (BH) | stability |
|---|---|---|---|---|---|---|
| `cand_d84149af1369a0bd` | `difference(mean radius, area error)` | −0.00098 | [−0.00152, −0.00050] | 0.00021 | 0.00050 | 0.88 |
| `cand_9136a73816fdea65` | `ratio(mean radius, mean texture)` | −0.00066 | [−0.00104, −0.00030] | 0.00025 | 0.00050 | 0.88 |
| `cand_3e345814dd3f2bc2` | `ratio(feature 5, feature 7)` | −0.00068 | [−0.00108, −0.00025] | 0.00195 | 0.00779 | 0.80 |

Negative effect = lower log-loss than the baseline.  All three survive
Benjamini-Hochberg at α = 0.10 and have the expected sign in at least 80% of
folds.

The round also persisted 18 `screened_out`, 9 `duplicate`, 5
`insufficient_evidence` and 2 `invalid_structure` records.  The invalid cases
were a mutated `difference(a, b)` descriptor with `a == b`; static validation
rejected them before execution, which is the intended generator/validator
interaction.

**Threshold caveat (important).**  The minimum meaningful effect was set to
0.0005 log-loss for this architecture-validation fixture after inspecting the
search-split effect scale.  This is not a pre-registered scientific effect
size and says nothing about practical significance.  It was fixed before the
holdout was unlocked; the holdout was not used to choose candidates or
thresholds.  The search-split effect is small (~0.001 nats) because the task is
nearly saturated.

## Frozen final evaluation

`cand_d84149af1369a0bd` was frozen (`frozen_manifest.json`,
manifest hash `85165a50…97bca9ea`) and then evaluated once on the 20% holdout:

| model | accuracy | AUC | log-loss |
|---|---|---|---|
| baseline | 0.9474 | 0.98446 | 0.17361 |
| candidate | 0.9561 | 0.98479 | 0.17202 |

Holdout effect: **−0.00159 log-loss** (candidate better), consistent in sign
with the search folds.  The holdout is a single evaluation unit, so no
standard error, p-value or CI is reported for it; the point estimate must not
be over-interpreted.

## What this validates

* the engine can manage a non-TSIL task, candidates, trials, evidence and
  feedback;
* fold-local fitting and three-level identity work for a second domain;
* the holdout guard blocks search and the frozen finalize path runs;
* failures (duplicates, invalid structures, screened-out and inconclusive
  candidates) are retained, not discarded.
