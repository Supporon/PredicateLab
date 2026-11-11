# Small-sample scenario: validation against a known ground truth

> **Status.**  Executable domain introduced 2026-11-08.  Results below are from
> synthetic data with an explicit latent truth; they validate the *platform's
> behaviour* under small n, not any real-world claim.  Read
> [evidence and limitations](evidence_and_limitations.md) first.
>
> The pilot reported in §6–§7 is superseded by a **formal pre-registered
> study** (1,200 replicates, effect-size grid, two engine seeds):
> [results](results_small_sample_20261108.md) ·
> [pre-registration](preregistration_small_sample_20261108.md).

## 1. Why this scenario exists

PredicateLab's gates (paired effect, bootstrap CI, BH/ Romano-Wolf
multiplicity, stability, seed agreement) were designed for regimes with
hundreds or thousands of observations.  The question this scenario answers is
narrow and falsifiable:

> At n between 24 and 256, does the platform *avoid* confirming predicates when
> there is no signal, and does its power to recover and confirm a planted
> nonlinear signal degrade smoothly and predictably with n?

Because the labels are generated from a known latent score, every conclusion
can be scored against truth.  This turns "the pipeline feels conservative" into
measured type-I error and power.

## 2. Ground truth (data-generating process)

Eight independent standard-normal features; a binary label sampled from a
logistic link with a calibrated intercept so the marginal positive rate is
0.5.  Four modes (`predicatelab/domains/small_sample/synthetic.py`):

| mode | latent score | truth family | role |
|---|---|---|---|
| `null_noise` | 0 | none | pure label noise; every validation is false |
| `null_linear` | `0.9 z0 + 0.6 z1` | none | linear signal the baseline already captures; global null for *added* predicates |
| `signal_ratio_threshold` | `s · [1(x0/(\|x2\|+0.1) ≥ τ_q) − mean] + 0.35 z3` | `ratio_threshold(0,2,q)` | hard positive: real but modest nonlinear effect |
| `signal_sign_interaction` | `s · 0.5 · sign(x1·x3) + 0.30 x5` | `interaction(1,3)` | positive control with a large effect |

Details that matter for validity:

* The ratio threshold `τ_q` is a **population** quantile from a fixed reference
  draw (`_REFERENCE_SEED=9001`, 100k draws), never the replicate's own sample,
  so labels do not leak into the candidate threshold.
* The sign-interaction truth uses raw `sign(x1·x3)`.  The compiler's
  `interaction` operator is defined the same way, about the origin.  An earlier
  version standardised inside the DGP and centred inside the compiler; the two
  boundaries differed by the training-fold mean and confirmation power
  collapsed (discovery effect +0.16, confirmation +0.008).  Do not reintroduce
  fold-mean centring into a sign predicate.
* The DGP `strength` for the sign interaction is applied as `0.5·s` because
  `sign` already has unit variance; measured Brier gains on 2,000 fresh
  samples are ≈0.049/0.093/0.135 at `s=2/3/4`.

## 3. Small-sample design decisions

| decision | value | reason |
|---|---|---|
| statistical unit | individual sample | the task is i.i.d. classification; folds are only a device for out-of-fold prediction |
| metric | `neg_brier = −(p−y)²`, higher better | proper score, paired per sample |
| dependence | `nw_lag=0`, `bootstrap_block=1` | samples are i.i.d.; Newey-West/block structure would be wrong |
| discovery / confirmation | disjoint stratified samples (`independent`, 65/35 of the non-holdout pool) in `independent` mode | confirmation must see samples the generator never saw |
| confirmation folds | 3-fold stratified CV over the confirmation pool; train = discovery ∪ two folds | every confirmation prediction is out-of-fold and disjoint from discovery |
| holdout | fixed 25% stratified, split before any search | one-time final verdict, ledger-guarded |
| minimum n | 24 (`suitability` block) | below this the 4-fold discovery split cannot keep ≥4 samples/class |
| warnings | `n < 4p` (ratio) and `n < 100` (power) | recorded and acknowledged in every config via `run.warning_acknowledgements` |
| baseline | `LogisticRegression(C=1.0)` on standardised raw features | the predicate must add out-of-sample value over a linear model |
| search policy | BH `α=0.10`, `min_effect=0.0025`, `min_stability=0.60`, fold coverage 0.75, seed agreement 0.50, positive CI | pre-registered in `configs/small_sample_*.yaml`, identical across cells |

Note that the engine names an `independent_split` confirmation
`resampling_confirmed` (the state vocabulary predates this domain).  It is
**more** conservative than the name suggests: the confirmation samples are
disjoint from discovery, not resampled.

## 4. Files

| path | purpose |
|---|---|
| `predicatelab/domains/small_sample/synthetic.py` | DGP, truth descriptors, family keys |
| `predicatelab/domains/small_sample/generator.py` | closed hypothesis space (ratio/difference/product/threshold/sigmoid/interaction) and feedback mutations |
| `predicatelab/domains/small_sample/adapter.py` | folds, suitability, fold-local compilation, trials, post-hoc recovery scoring |
| `configs/small_sample_{smoke,signal,null,ratio}.yaml` | four pre-registered runs |
| `scripts/validate_small_sample.py` | Monte Carlo harness over the real lifecycle |
| `tests/test_small_sample_adapter.py` | 21 tests: descriptors, fold isolation, determinism, null/signal sanity, lifecycle |
| `artifacts_small_sample/sweep/summary.{md,json}` + `records.jsonl` | sweep outputs (regenerate with the command below) |

## 5. Running it

```bash
export PYTHONPATH=.
PY=/home/zhengzengrong/miniconda3/envs/predicatelab/bin/python

# 1. fast end-to-end smoke (n=96) -- full lifecycle
rm -rf artifacts_small_sample
$PY -m predicatelab.cli discover      --config configs/small_sample_smoke.yaml --rounds 1
RID=$(ls -t artifacts_small_sample | grep '^run_' | head -1)
$PY -m predicatelab.cli freeze-family --config configs/small_sample_smoke.yaml --run-id "$RID"
$PY -m predicatelab.cli confirm       --config configs/small_sample_smoke.yaml --run-id "$RID"
$PY -m predicatelab.cli finalize      --config configs/small_sample_smoke.yaml \
    --run-id "$RID" --candidate-id cand_...

# 2. Monte Carlo validation (n x mode grid; ~45 s at 8 workers)
$PY scripts/validate_small_sample.py --n-values 16,24,32,64,128,256 \
    --reps 12 --rounds 2 --workers 8 --output artifacts_small_sample/sweep

# 3. full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q
```

Each `small_sample_*.yaml` keeps its confirmation ledger inside
`artifacts_small_sample/`, so `rm -rf artifacts_small_sample` is a complete
reset.  The sweep goes further and gives every replicate its own database and
ledger (`<output>/runs/<cell>/`): a Monte Carlo re-run is not a second look at
the same confirmation fold, and reusing one ledger would silently downgrade
every re-run to `exploratory_positive`.

## 6. Smoke run (n=96, `signal_sign_interaction`, s=4)

| stage | sample | effect | 95% CI | p | state |
|---|---|---|---|---|---|
| discovery (4-fold CV on 46 discovery samples) | 46 | +0.1593 | — | 0.00034 | `exploratory_positive` |
| confirmation (26 disjoint samples, 3 folds of 9/9/8) | 26 | +0.1427 | (+0.0222, +0.2497) | 0.0110 | `resampling_confirmed` |
| final holdout (24 samples, one-time) | 24 | +0.1040 | (−0.0215, +0.1961) | 0.0639 | `insufficient` |

The true descriptor `interaction(1,3)` was the only frozen member, and the
fold-compiled column is identical to the latent indicator on the pool.  The
holdout's `insufficient` verdict is the intended behaviour: 24 samples cannot
exclude zero even when the point estimate is close to the population gain.
This is the single most important small-sample lesson in the scenario — a
confirmation is not a licence to claim the effect on a smaller final set.

## 7. Monte Carlo results

Grid: 4 modes × n ∈ {16, 24, 32, 64, 128, 256} × 12 replicates, 2 discovery
rounds, engine seed 2025, 8 workers.  `n=16` is refused by suitability in every
cell (shown as `-`).  Rates are over replicates that ran; Wilson 95% CI in
parentheses.  `recovered` credits the exact truth family or a monotone
equivalent with AUC ≥ 0.90 against the latent indicator; `spurious` is a
validated predicate that is neither.

| mode | n | discovery any | recovered | spurious | confirmation any | confirmed recovered | mean AUC | final states |
|---|---|---|---|---|---|---|---|---|
| null_linear | 24 | 0.17 | 0.00 | 0.17 | 0.00 | 0.00 | – | – |
| null_linear | 32 | 0.08 | 0.00 | 0.08 | 0.00 | 0.00 | – | – |
| null_linear | 64 | 0.08 | 0.00 | 0.08 | 0.00 | 0.00 | – | – |
| null_linear | 128 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | – | – |
| null_linear | 256 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | – | – |
| null_noise | 24 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | – | – |
| null_noise | 32 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | – | – |
| null_noise | 64 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | – | – |
| null_noise | 128 | 0.08 | 0.00 | 0.08 | 0.00 | 0.00 | – | – |
| null_noise | 256 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | – | – |
| signal_ratio_threshold | 24 | 0.17 | 0.08 | 0.08 | 0.00 | 0.00 | 0.877 | – |
| signal_ratio_threshold | 32 | 0.08 | 0.00 | 0.08 | 0.00 | 0.00 | 0.579 | – |
| signal_ratio_threshold | 64 | 0.17 | 0.17 | 0.00 | 0.00 | 0.00 | 0.984 | – |
| signal_ratio_threshold | 128 | 0.25 | 0.17 | 0.08 | 0.08 | 0.08 | 0.908 | confirmed:1 |
| signal_ratio_threshold | 256 | 0.33 | 0.33 | 0.08 | 0.00 | 0.00 | 0.976 | – |
| signal_sign_interaction | 24 | 0.45 | 0.18 | 0.36 | 0.09 | 0.09 | 0.791 | insufficient:1 |
| signal_sign_interaction | 32 | 0.58 | 0.25 | 0.42 | 0.00 | 0.00 | 0.751 | – |
| signal_sign_interaction | 64 | 0.33 | 0.33 | 0.08 | 0.17 | 0.17 | 1.000 | confirmed:1, insufficient:1 |
| signal_sign_interaction | 128 | 0.58 | 0.58 | 0.17 | 0.50 | 0.50 | 1.000 | confirmed:4, insufficient:2 |
| signal_sign_interaction | 256 | 0.83 | 0.83 | 0.17 | 0.67 | 0.67 | 1.000 | confirmed:8 |

Worst-case Wilson upper bounds in the null cells are 0.45 (n=24, 12 replicates);
the important number is **0 confirmations in 120 null replicates** across both
null modes.  The discovery gate is deliberately permissive (it is a search
stage); the confirmation gate is what carries the type-I guarantee, and it held
with zero false confirmations.

Other observations:

* **Power rises with n as expected** for the strong interaction: recovered
  18% → 25% → 33% → 58% → 83% from n=24 to n=256.  Confirmation lags recovery
  because the confirmation sample is only ~26% of n.
* **The ratio truth is hard**, as designed: maximum achievable Brier gain is
  ~0.044, so only 33% recovery and one confirmation at n=256.  The continuous
  `ratio(0,2)` separates the latent indicator perfectly (AUC 1.00, monotone
  equivalent) but rarely passes the effect/CI gates; the fitted
  `ratio_threshold` scores AUC 0.93–1.00 because its pool quantile differs
  slightly from the population threshold used to generate the labels.
* **Spurious discovery is high at very small n** (36–42% for the interaction at
  n=24–32) and falls to 8–17% by n≥64.  This is the small-sample optimism that
  the confirmation stage exists to absorb, and it did.
* **The final holdout is stricter than confirmation**: at n=64 and n=128 some
  confirmed candidates still ended `insufficient` on the 25% holdout, while at
  n=256 all 8 confirmations also passed the holdout.  n=96 (smoke) is exactly
  in the zone where confirmation passes and the holdout does not.

## 8. Limitations

1. **Synthetic by construction.**  The DGP is i.i.d. Gaussian; it says nothing
   about serial dependence, drift, or real measurement error.  The scenario is a
   calibration check of the *procedure*, not evidence about a domain.
2. **One engine seed per sweep replicate.**  Seed-agreement therefore cannot
   fail in the sweep; the four shipped configs use `seeds: [2025, 2026, 2027]`
   when multi-seed stability matters.
3. **Closed hypothesis space.**  The truth is always inside the generator's
   operator set.  Real discovery can miss the right operator entirely; this
   scenario measures power conditional on the truth being expressible.
4. **Monotone equivalence is a post-hoc definition.**  The AUC ≥ 0.90 label is
   computed after the run, on non-holdout samples, and never feeds discovery,
   confirmation or finalization.
5. **Confirmation state naming.**  `independent_split` is stored as
   `resampling_confirmed`; the folds are genuinely disjoint, but the state name
   understates the design.
6. **Replicate independence and the ledger.**  Each sweep replicate uses an
   isolated database and ledger.  This is appropriate for a simulation; it must
   not be copied into a real study, where the global ledger is the protection
   against re-using a confirmation fold.
