# Results — LUSI kernel-weight (tau) sweep

- **Pre-registration:** `docs/preregistration_tau_sweep_20261108.md` (written before any
  sweep trial was run).
- **Runs:** all five arms `run_20261108_110259`, one round each, completed
  `search_complete` on 2026-11-08 (~19:53 local).
- **Provenance:** data snapshot `96cc5fb17be56224`, protocol `73277d2b7fe9d2a0`,
  code fingerprint `0ad2f23e528ed57b` (unchanged from run creation — verified against
  `runs.code_fingerprint`).
- **Artifacts:** `artifacts/tsil_tau_{k005,k010,k020,k050,learned}/`,
  analysis `artifacts/exp_logs/tau_analysis_20261108.txt`,
  verifier output `artifacts/exp_logs/verify_tau_*.json`.

## 0. Integrity checks

| check | result |
|---|---|
| candidate trials actually ran at the intended tau | k005 `fixed_tau=0.95` ×36, k010 `0.90` ×36, k020 `0.80` ×36, k050 `0.50` ×36, learned `tau_mode=learned` via `sigmoid(alpha)` ×36 |
| `baseline:mse` identical across arms | yes — screen seed 2025 IC `0.047835`, MSE `0.99656` in all five arms, bit-identical |
| holdout untouched | yes — `holdout-audit` shows only the pre-seeded `builtin_historic` row |
| independent verifier (`scripts/verify_evidence_independent.py`) | `ok=true`, 0 errors, `zero_predicatelab_imports=true` on all five arms |
| confirmation / holdout folds | `f2024` and `2025-09-01…2026-03-31` still unused |

Note on `phi_type`: `job.json` stores the pre-override `model_config.phi_type="mse"`.
`worker.py:465` overwrites it per variant before constructing the model
(`baseline→mse`, `control:ones→ones`, `candidate→external|phi_timestamp`), so
`use_weighted=true` for candidates and the LUSI loss is active. The presence of
`fixed_tau`/`tau_history` in a candidate's `fit_metrics` (absent for baseline) is the
runtime evidence.

## 1. Out-of-sample result (full validation f2020+f2021 × seeds 2025,2026)

| arm | kernel weight | promoted | validated | arm-mean paired effect | best | mean stability | min q |
|---|---|---|---|---|---|---|---|
| k005 | 0.05 | 4 | **0** | −0.0024 | −0.0014 | 0.31 | 0.496 |
| k010 | 0.10 | 4 | **0** | −0.0014 | −0.0007 | 0.56 | 0.532 |
| k020 | 0.20 | 4 | **0** | −0.0020 | −0.0001 | 0.44 | 0.471 |
| k050 | 0.50 | 4 | **0** | −0.0025 | −0.0017 | 0.31 | 0.442 |
| learned | learnable | 4 | **0** | −0.0019 | −0.0014 | 0.38 | 0.537 |

Effect = candidate IC − `baseline:mse` IC, paired on fold × seed.
**20/20 promoted candidates across the five arms have a negative effect.** No arm's
smallest p comes near the FDR threshold (min q ≈ 0.44).

Per-arm t-test of the 4 promoted effects against zero: t = −2.8 … −7.0. The sign is
reliable; what is not reliable is any *ordering* between arms.

## 2. In-sample (screen f2022, 5 epochs)

| arm | mean screen effect | min … max | n>0 | control:ones f2022 IC | baseline MSE IC |
|---|---|---|---|---|---|
| k005 | +0.0025 | +0.0006 … +0.0043 | 20/20 | 0.0501 | 0.0478 |
| k010 | +0.0019 | −0.0045 … +0.0038 | 19/20 | 0.0502 | 0.0478 |
| k020 | +0.0018 | −0.0007 … +0.0040 | 17/20 | 0.0507 | 0.0478 |
| k050 | +0.0029 | +0.0005 … +0.0052 | 20/20 | 0.0511 | 0.0478 |
| learned | +0.0015 | −0.0014 … +0.0050 | 19/20 | 0.0508 | 0.0478 |

Every kernel, at every weight including 0.05, buys roughly +0.002 in-sample over MSE —
and `control:ones` (no predicate content) buys the same amount. Out of sample the sign
reverses for all of them. This reproduces the four-arm diagnostic: the screen gain
belongs to the presence of the kernel term, not to predicate content or to its weight.

## 3. Pre-registered decision

Decision rule (fixed in the pre-registration):

- `positive` = ≥1 promoted candidate reaches `state=validated`.
- H1 supported iff a validated candidate appears at kernel ≤0.10 **and** the arm-mean
  effect curve increases monotonically as kernel weight decreases.
- H1 rejected iff no validated candidate **and** arm-mean effects are flat in tau
  (spread < 0.001).
- Anything else is reported as-is, no re-gating.

Observed:

- validated at kernel ≤0.10: **0**; validated anywhere: **0**
- arm-mean effect curve (kernel 0.05 / 0.10 / 0.20 / 0.50):
  −0.0024 / −0.0014 / −0.0020 / −0.0025 → **not monotone**
- spread across kernel weights: **0.0012**

**VERDICT: H1 is not supported.** No tau setting rescues a candidate, and the ordering
does not follow the predicted dose–response. By the letter the outcome is
*intermediate* rather than a clean H0 rejection, because the spread (0.0012) exceeds the
pre-registered flatness bound (0.001) by 0.0002 — that margin is far smaller than the
sampling error of a 4-candidate arm mean (SE 0.0003–0.0006), so it must not be read as
evidence of a tau effect.

Supporting statistics:

| test | result |
|---|---|
| full effect ~ kernel weight (n=16 promoted) | slope −0.00127 ± 0.00142, p=0.39 |
| screen effect ~ kernel weight (n=80 candidates) | slope +0.00155 ± 0.00090, p=0.088 |
| pairwise arm differences (Welch) | all p ≥ 0.107, none significant |

The screen-fold slope is the only dose–response that appears at all: more kernel weight
gives more in-sample gain. It does not survive out of sample.

## 4. Learned arm — the predicted collapse is real

`sigmoid(alpha)` at each epoch, mean over the candidate trials of each phase:

| phase | trials | epochs | τ̂ first eval | τ̂ last eval | monotone decreasing |
|---|---|---|---|---|---|
| screen f2022 | 20 | 5 | 0.4206 | 0.2347 | 20/20 |
| full f2020 | 8 | 18 (early stop) | 0.4606 | 0.1263 | 8/8 |
| full f2021 | 8 | 7 (early stop) | 0.4404 | 0.2030 | 8/8 |

This confirms the upstream code comment: with `tau_mode='learned'` the objective drifts
toward the rank-1 P term and τ̂ is driven to zero. The *secondary* prediction that this
arm would be the worst of the five is **not** confirmed — it is −0.0019, better than
k005 and k050.

## 5. Mechanistic diagnostic — why the sweep is null

Same candidate, same fold/seed, different tau ⇒ materially different models:

| candidate | fold/seed | rank corr vs baseline (k005 / k010 / k050) | cross-arm rank corr |
|---|---|---|---|
| a1924c4c | f2020 s2025 | 0.852 / 0.849 / 0.830 | 0.893 … 0.936 |
| a1924c4c | f2020 s2026 | 0.819 / 0.820 / 0.812 | 0.947 … 0.977 |
| 71b00c29 | f2020 s2026 | — / 0.994 / 0.738 | 0.742 … 0.885 |
| 7839cc5f | f2020 s2026 | — / 0.992 / 0.808 (k020) | 0.814 |

Two facts together:

1. tau **does** change the trained model (predictions correlate only 0.74–0.99 across
   arms), so the sweep is not a no-op.
2. But the out-of-sample outcome is the same negative effect at kernel weight 0.05 as at
   0.50, and at 0.05 the model is still far from the MSE baseline (rank corr 0.85 on
   f2020).

That is the Adam scale-invariance limitation written into the pre-registration, now
measured: the optimizer normalises per-parameter gradient magnitude, so shrinking the
kernel's *scale* does not shrink its *direction* share. The blend weight is therefore
not a usable knob — the P term keeps steering the solution even at 5 % weight, and the
steering is what costs out-of-sample IC. The candidate-vs-MSE gap is a property of
injecting P, not of how much of it is injected.

## 6. Limitations of this sweep

1. n = 4 promoted candidates per arm; between-arm differences at this effect size are
   not resolvable. The arm-mean test is an ordering check, not a powered contrast.
2. The promoted set differs across arms (only 14 unique candidates over 20 slots), so
   candidate-level cross-arm pairing is only available for 4 candidates (analysed in §5).
3. The bank is 20 fixed LLM proposals; a null here says nothing about the LUSI term for
   predicates outside this bank.
4. `f2024` and the holdout remain unused, by design.

## 7. Consequence

No arm produced a validated candidate, so no `freeze-family → confirm → finalize` chain
was opened; the confirmation fold and the holdout stay fresh. The next discriminating
experiment is not another tau value — it is a test of the P-term mechanism itself
(e.g. holding the kernel gradient's *direction* fixed while varying its projection, or
disabling Adam's per-parameter normalisation for the loss blend), because the existing
evidence says the weight is the wrong axis to vary.
