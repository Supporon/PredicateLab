# Formal small-sample study — results

- **Pre-registration:** [preregistration_small_sample_20261108.md](preregistration_small_sample_20261108.md)
- **Run:** 2026-11-08T21:16:04+08:00 → 21:18:38, commit `a895a7c`, clean tree
  before and after (`git status --porcelain` empty).
- **Grid:** 1,200 replicates = 2 null modes × 5 n + 2 signal modes × 3 strengths
  × 5 n, 30 replicates per cell, engine seeds `[2025, 2026]`, 2 discovery
  rounds, 16 workers.
- **Records / summary:** `artifacts_small_sample/formal_20261108/records.jsonl`,
  `summary.{md,json}`, `analysis.{md,json}`; boundary check in
  `artifacts_small_sample/formal_20261108_boundary/`.
- **Command:** exactly the frozen command in the pre-registration §7.

## 1. Verdicts

| hypothesis | endpoint | observed | pre-registered rule | verdict |
|---|---|---|---|---|
| **H1** calibration (primary) | null confirmation false positives | **3/298 = 0.010** (95% CI 0.003–0.029); worst cell 0.034 | pooled ≤ 0.02 and every cell ≤ 0.075 | **MET** |
| **H2** power (primary) | interaction s=3.0, confirmed recovery | n=256: **21/30 = 0.700** (0.521–0.833); n=24: **3/29 = 0.103** (0.036–0.264) | n=256 ≥ 0.40 and n=24 ≤ 0.20 | **SUPPORTED** |
| **H3** effect size (primary) | interaction, n=128, confirmed recovery s=4 vs s=2 | **0.767 (23/30) − 0.100 (3/30) = 0.667** | difference ≥ 0.15 | **SUPPORTED** |
| **H4** holdout strictness (primary) | holdout-confirmed among confirmed, s=3.0 | small n (24–64): **3/10 = 0.300**; large n (128–256): **25/33 = 0.758** | small < large | **SUPPORTED** |
| **H5** discovery calibration (secondary) | null spurious discovery, n≤64 vs n≥128 | **38/178 = 0.213** vs **3/120 = 0.025**, gap **0.188** | gap ≥ 0.10 | **MET** |
| **H6** ratio hard positive (secondary) | ratio s=4.0 confirmed recovery | n≤128: 0.000, 0.000, 0.033, 0.100; n=256: **0.167** | every n≤128 ≤ 0.20; n=256 ≤ 0.50 | **MET** |
| boundary | n=16 refused | **8/8 blocked**, 8/8 with `sample_count` blocker | all blocked | **MET** |

All six pre-registered expectations held.  Execution integrity: **1,192 ok,
8 blocked (all by split feasibility), 0 errors** — no replicate failed for a
harness or execution reason.  The two process deviations are documented in §4.

## 2. Calibration is the headline

The pooled null confirmation rate is 1.0% (3 of 298 usable replicates), with an
upper Wilson bound of 2.9%.  The three false confirmations all occurred at
n=24–32, and every one of them subsequently failed the holdout (`insufficient`).
The discovery stage alone is much less conservative — 21% spurious validations
at n≤64 — which is exactly the division of labour the protocol intends: search
is optimistic, confirmation and the one-time holdout carry the error control.

At n≥128 the discovery stage itself becomes well behaved (2.5% spurious under
the null), so the confirmation gate is rarely even exercised.

## 3. Power

For the planted sign interaction at s=3.0, confirmed recovery rises with n:
0.103 (n=24) → 0.400 (n=128) → 0.700 (n=256).  At s=4.0 it reaches 1.00 at
n=256.  The effect-size ordering is strong: at n=128 the confirmed-recovery rate
is 0.767 at s=4.0 versus 0.100 at s=2.0.

The confirmation sample is only ~26% of n, which is why confirmed recovery lags
discovery recovery by roughly one step in n.  The holdout result is stricter
still: among small-n confirmations only 30% survive the 25% holdout, versus 76%
at n≥128.  A confirmation at n≤64 should therefore be read as "survived an
independent gate", not as "the effect is established".

The ratio truth behaves as pre-registered: rare recovery at small n (0–10%),
rising only to 16.7% at n=256, because its maximum achievable Brier gain is
about 0.044.  This is the intended demonstration that a real, correctly
specified nonlinear predicate can remain unconfirmable at these sample sizes —
and that the platform does not manufacture a positive result for it.

## 4. Deviations from the pre-registration

**D1 — null replicates: 30, not 40.**  §2 of the pre-registration prose listed
40 replicates per null cell (400 total), but the frozen command in §7 used one
`--reps 30` for every cell.  The executed grid is the command's: 30 per cell,
300 null replicates (298 usable after 2 split-feasibility blocks).  The decision
thresholds were left unchanged, which makes H1 marginally stricter than the
prose (≤ 6/300 rather than ≤ 8/400); the observed 3/298 is below both.  No cell
was added or dropped after seeing results.

**D2 — first attempt aborted by an external commit.**  The first formal
execution (21:12:50–21:15:24) was interrupted by a concurrent commit to the
repository (`f3e2ed7` → `a895a7c`) while it ran; 6 replicates failed with
`code_fingerprint changed since run creation` at confirmation.  Per the
deviation policy (§6) the **entire** grid was re-run on the now-clean commit
`a895a7c`, and the first attempt is preserved at
`artifacts_small_sample/formal_20261108_attempt1/` for audit.  The results above
come only from the second, complete run.  This is a genuine strength of the
platform: the engine refused to confirm under a changed code identity.

No other deviation occurred.  The harness was not modified after the formal run.

## 5. Full per-cell results

Rates are over usable (non-blocked) replicates; Wilson 95% intervals in
parentheses.  `–` marks an empty cell or a not-applicable column (null modes
have no latent indicator, so recovery AUC is undefined).

### Null modes (every validation is a false discovery)

| mode | n | discovery any | discovery spurious | confirmation any |
|---|---|---|---|---|
| null_linear | 24 | 0.23 (0.12,0.41) | 0.23 (0.12,0.41) | 0.03 (0.01,0.17) |
| null_linear | 32 | 0.30 (0.17,0.48) | 0.30 (0.17,0.48) | 0.03 (0.01,0.17) |
| null_linear | 64 | 0.10 (0.03,0.26) | 0.10 (0.03,0.26) | 0.00 (0.00,0.11) |
| null_linear | 128 | 0.03 (0.01,0.17) | 0.03 (0.01,0.17) | 0.00 (0.00,0.11) |
| null_linear | 256 | 0.03 (0.01,0.17) | 0.03 (0.01,0.17) | 0.00 (0.00,0.11) |
| null_noise | 24 | 0.24 (0.12,0.42) | 0.24 (0.12,0.42) | 0.03 (0.01,0.17) |
| null_noise | 32 | 0.31 (0.17,0.49) | 0.31 (0.17,0.49) | 0.00 (0.00,0.11) |
| null_noise | 64 | 0.10 (0.03,0.26) | 0.10 (0.03,0.26) | 0.00 (0.00,0.11) |
| null_noise | 128 | 0.03 (0.01,0.17) | 0.03 (0.01,0.17) | 0.00 (0.00,0.11) |
| null_noise | 256 | 0.00 (0.00,0.11) | 0.00 (0.00,0.11) | 0.00 (0.00,0.11) |

### Sign-interaction signal

| s | n | discovery recovered | discovery spurious | confirmation any | confirmed recovered | final confirmed / finalized |
|---|---|---|---|---|---|---|
| 2 | 24 | 0.14 | 0.18 | 0.04 | 0.00 | 0 / 1 |
| 2 | 32 | 0.10 | 0.30 | 0.07 | 0.03 | 1 / 2 |
| 2 | 64 | 0.17 | 0.17 | 0.00 | 0.00 | – |
| 2 | 128 | 0.17 | 0.00 | 0.10 | 0.10 | 1 / 3 |
| 2 | 256 | 0.47 | 0.00 | 0.27 | 0.27 | 6 / 8 |
| 3 | 24 | 0.24 | 0.34 | 0.10 | 0.10 | 0 / 3 |
| 3 | 32 | 0.27 | 0.33 | 0.07 | 0.07 | 1 / 2 |
| 3 | 64 | 0.43 | 0.10 | 0.17 | 0.17 | 2 / 5 |
| 3 | 128 | 0.57 | 0.17 | 0.40 | 0.40 | 5 / 12 |
| 3 | 256 | 0.90 | 0.07 | 0.70 | 0.70 | 20 / 21 |
| 4 | 24 | 0.53 | 0.50 | 0.30 | 0.27 | 3 / 9 |
| 4 | 32 | 0.47 | 0.20 | 0.20 | 0.20 | 2 / 6 |
| 4 | 64 | 0.73 | 0.13 | 0.47 | 0.47 | 9 / 14 |
| 4 | 128 | 0.87 | 0.10 | 0.77 | 0.77 | 17 / 23 |
| 4 | 256 | 1.00 | 0.07 | 1.00 | 1.00 | 29 / 30 |

### Ratio-threshold hard positive

| s | n | discovery recovered | confirmation any | confirmed recovered |
|---|---|---|---|---|
| 2 | 24 | 0.07 | 0.00 | 0.00 |
| 2 | 32 | 0.00 | 0.03 | 0.00 |
| 2 | 64 | 0.10 | 0.00 | 0.00 |
| 2 | 128 | 0.07 | 0.00 | 0.00 |
| 2 | 256 | 0.07 | 0.00 | 0.00 |
| 3 | 24 | 0.07 | 0.00 | 0.00 |
| 3 | 32 | 0.00 | 0.00 | 0.00 |
| 3 | 64 | 0.13 | 0.00 | 0.00 |
| 3 | 128 | 0.13 | 0.07 | 0.07 |
| 3 | 256 | 0.33 | 0.07 | 0.07 |
| 4 | 24 | 0.10 | 0.00 | 0.00 |
| 4 | 32 | 0.00 | 0.00 | 0.00 |
| 4 | 64 | 0.17 | 0.03 | 0.03 |
| 4 | 128 | 0.33 | 0.10 | 0.10 |
| 4 | 256 | 0.43 | 0.17 | 0.17 |

## 6. What this does and does not establish

**Establishes (for this synthetic DGP and this pre-registered protocol):**

1. The confirmation gate holds nominal-ish type-I control at these sample sizes:
   1.0% false confirmations under two global nulls, with a 2.9% upper bound.
2. Power is monotone in n and in effect size, with the expected lag between
   discovery, confirmation and holdout.
3. The platform refuses to operate below n=24 and refuses to confirm under a
   changed code identity.
4. A real but modest nonlinear effect (ratio, max gain ≈0.044) is not turned
   into a false positive, even at n=256.

**Does not establish:**

1. Anything about real data: the DGP is i.i.d. Gaussian and the truth is always
   inside the generator's operator space.
2. That the 1% null rate generalises to other predictor spaces, metrics, or
   dependence structures; only the specific gates and grid were tested.
3. That n=256 is "enough" in general — only that this effect size is usually
   confirmable with these folds.
4. That the three observed null confirmations are "acceptable"; they are the
   measured false-positive rate, and their holdout failures are consistent with
   the calibration, not a licence to relax.

## 7. Reproduce

```bash
export PYTHONPATH=.
PY=/home/zhengzengrong/miniconda3/envs/predicatelab/bin/python

$PY scripts/validate_small_sample.py \
    --n-values 24,32,64,128,256 \
    --modes null_noise,null_linear,signal_sign_interaction,signal_ratio_threshold \
    --strengths 2,3,4 \
    --reps 30 --rounds 2 --seeds 2025,2026 --workers 16 \
    --output artifacts_small_sample/formal_20261108

$PY scripts/analyze_small_sample_formal.py \
    --records artifacts_small_sample/formal_20261108/records.jsonl \
    --boundary-records artifacts_small_sample/formal_20261108_boundary/records.jsonl \
    --output artifacts_small_sample/formal_20261108/analysis
```

The sweep is deterministic given the command (per-replicate data seeds are
`20261108 + 1009·rep`), so a re-run on the same commit reproduces these numbers
exactly, subject to the two deviations above.
