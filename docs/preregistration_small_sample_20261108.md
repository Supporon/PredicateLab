# Pre-registration — formal small-sample validation study

- **Date registered:** 2026-11-08, before the formal sweep was run.
- **Domain:** `small_sample` (synthetic, explicit latent truth), introduced the
  same day; see [the scenario description](small_sample_scenario.md).
- **Pilot disclosure:** a 12-replicate pilot (`artifacts_small_sample/sweep`,
  288 replicates, one engine seed) was run while building and debugging the
  harness.  Its results are in §7 of the scenario document.  This
  pre-registration scales that pilot (more replicates, an effect-size grid, two
  engine seeds) and fixes every decision threshold **before** the formal run.
  The pilot is disclosed because it informed the grid (e.g. which strengths are
  informative); no threshold below is chosen to fit a pilot cell.

## 1. Question

For a controlled synthetic classification task, what are the operating
characteristics of the PredicateLab gates at small n (24–256)?

Two estimands:

1. **Type-I behaviour.** Under a global null (no predicate can add value over
   the linear baseline), how often does the *confirmation* stage declare a
   candidate confirmed?
2. **Power.** Under a planted nonlinear signal, how often does the pipeline
   (a) validate a behaviourally equivalent predicate in discovery and
   (b) confirm it on disjoint samples, as a function of n and effect size?

## 2. Design grid (fixed now)

| factor | values |
|---|---|
| null modes | `null_noise`, `null_linear` |
| signal modes | `signal_sign_interaction` and `signal_ratio_threshold`, both over strengths 2.0, 3.0, 4.0 |
| n | 24, 32, 64, 128, 256 |
| replicates per cell | null: 40; signal: 30 |
| engine seeds per replicate | `[2025, 2026]` |
| discovery rounds | 2 |
| workers | 16 |

Total: 400 null + 450 interaction + 450 ratio = **1,300 replicates**
(2,600 engine-seed evaluations).

Held fixed byte-identical across all cells: the four base configs' policy
(`fdr_alpha=0.10`, `min_effect=0.0025`, `min_stability=0.60`, fold coverage
0.75, seed agreement 0.50, positive CI required), 8 features, 4 CV splits,
3 confirmation folds, 25% holdout, 65/35 discovery/confirmation split,
`bootstrap_samples=500`, `nw_lag=0`, `bootstrap_block=1`, 24 candidates/round.
Every replicate uses its own sqlite database and confirmation ledger under
`<output>/runs/`.

`n = 16` is **not** in the grid: `suitability` blocks it by design, and the
pilot confirmed 48/48 blocked.  It is reported separately as a boundary check.

## 3. Outcomes (definitions fixed now)

Per replicate, recorded by `scripts/validate_small_sample.py`:

- `discovery_any` — ≥1 candidate validated in discovery.
- `discovery_recovered` — ≥1 validated candidate is the true family **or** a
  monotone equivalent: AUC ≥ 0.90 of the compiled predicate column against the
  latent indicator, computed on non-holdout samples only, after the run.
- `discovery_spurious` — ≥1 validated candidate is neither.
- `confirmation_any` — ≥1 frozen member reaches `resampling_confirmed` or
  `independently_confirmed`.
- `confirmation_recovered` — as above **and** the confirmed member is a true or
  monotone-equivalent recovery.
- `final_state` — one-time holdout verdict for the confirmed member, when one
  exists (`independently_confirmed` / `insufficient` / `rejected` /
  `exploratory_positive`).

A **confirmed false positive** is a null replicate with `confirmation_any =
True`.  No other null outcome counts as an error.

## 4. Hypotheses and decision rules

All rates are reported over replicates with `status = ok` (a replicate blocked
by suitability is reported and excluded; blocking is itself an endpoint).
Wilson 95% intervals are reported but decisions use the pre-specified
thresholds, not interval overlap.

**H1 — calibration (primary).** Under the two null modes, the pooled
confirmation false-positive rate over 400 replicates is **≤ 0.02** (at most 8),
and no individual null cell exceeds **0.075** (at most 3/40).
*H1 is rejected if either bound is exceeded.*

**H2 — power (primary).** For `signal_sign_interaction`, s=3.0:
`confirmation_recovered` at n=256 is **≥ 0.40**, and at n=24 is **≤ 0.20**.
*H2 is supported only if both hold.*

**H3 — effect size (primary).** At n=128, the difference
`confirmation_recovered(s=4.0) − confirmation_recovered(s=2.0)` for
`signal_sign_interaction` is **≥ 0.15**.
*H3 is supported if the difference meets the threshold; otherwise reported as
not supported, whatever the sign.*

**H4 — holdout strictness (primary).** Among interaction replicates with a
confirmation at s=3.0, the holdout-validated fraction at n ∈ {24,32,64} is
**lower** than at n ∈ {128,256} (strict inequality; no fixed gap).
*H4 is supported if the pooled small-n fraction is lower.*

**H5 — discovery calibration (secondary).** The pooled `discovery_spurious`
rate under the null modes at n ∈ {24,32,64} is **≥ 0.10 higher** than at
n ∈ {128,256}, i.e. small-sample discovery is measurably optimistic and the
confirmation gate absorbs it.
*Reported as a descriptive expectation, not used to reject the platform.*

**H6 — ratio hard positive (secondary).** `signal_ratio_threshold` at s=4.0
has `confirmation_recovered` **≤ 0.20 in every n ≤ 128** and **≤ 0.50 at
n=256**.  The ratio effect's maximum achievable Brier gain is ≈0.044, so this
is a pre-specified expectation of low power, not a failure criterion.

**Boundary check.** All `n=16` replicates are blocked by suitability with
`sample_count` in the blockers; 0 execution failures occur anywhere in the
grid.

## 5. Analysis plan

1. Run the grid exactly as specified in §2, once.  No interim analysis, no
   early stopping, no re-running of individual cells.
2. Produce `artifacts_small_sample/formal_20261108/summary.{json,md}` and keep
   `records.jsonl` (every replicate, with its seed and status).
3. Evaluate H1–H6 in the order above and record met / not met with the observed
   point estimate and Wilson interval.
4. Report **all** cells, including ones that contradict an expectation.
5. The formal report (`docs/results_small_sample_20261108.md`) must state the
   run timestamp, the exact command, worker count, and the git/code fingerprint
   recorded in each replicate's context.

## 6. Deviations policy

- A harness bug found during the formal run is a **deviation**: fix it, record
  a dated note in the results document, and re-run the **entire** grid.  No
  partial re-runs and no post-hoc cell additions.
- If a cell's replicates are all blocked or all error, report the cell with
  `rate = null`; do not substitute a different n.
- The post-hoc equivalence rule (AUC ≥ 0.90) is fixed here.  It never feeds
  discovery, confirmation or finalization; changing it after seeing results
  would be a deviation.

## 7. Command (frozen)

```bash
export PYTHONPATH=.
PY=/home/zhengzengrong/miniconda3/envs/predicatelab/bin/python

$PY scripts/validate_small_sample.py \
    --n-values 24,32,64,128,256 \
    --modes null_noise,null_linear,signal_sign_interaction,signal_ratio_threshold \
    --strengths 2,3,4 \
    --reps 30 --rounds 2 --seeds 2025,2026 --workers 16 \
    --output artifacts_small_sample/formal_20261108
```

(With `--strengths`, both signal modes expand over 2/3/4 while null modes stay
at strength 0.  The ratio mode's strength parameter shifts the latent score,
not the population threshold; all its cells are reported in full, but H6 refers
to s=4.0 only.  This is stated now, before the run, so no cell is dropped
afterwards.)

## 8. Limitations stated up front

1. Synthetic i.i.d. Gaussian data; conclusions are about the procedure, not a
   real domain.
2. The truth is always inside the generator's operator space; power is
   conditional on expressibility.
3. Two engine seeds per replicate test the seed-agreement gate but do not
   exhaust modelling variability.
4. `n=24` cells are at the edge of the protocol: the discovery pool is ~11
   samples and split feasibility can block a replicate; the blocked fraction is
   reported rather than corrected.
5. The engine stores an `independent_split` confirmation as
   `resampling_confirmed`; the folds are disjoint, so this is conservative.
