# Pre-registration — LUSI kernel-weight (tau) sweep

- **Date registered:** 2026-11-08, before any sweep trial was run.
- **Motivating result:** the four-arm LLM-vs-DSL discovery run of 2026-11-08
  (`artifacts/tsil_llm_*`) produced **0 validated candidates out of 32 promoted**.
  Every promoted candidate had a *negative* paired effect against the plain MSE
  baseline (range −0.0006 … −0.0041), including the deterministic rule-search arm.
- **Diagnostic that motivates this sweep:** across 154 distinct kernels from four
  arms the screen-fold (f2022) IC had mean +0.0505 and sd 0.0015, while the
  degenerate `control:ones` kernel scored +0.0511 and the MSE baseline +0.0478.
  The screen-fold "improvement" therefore tracks the presence of the LUSI kernel
  term, not the predicate's content. Out of sample the ordering reverses
  (f2020: MSE 0.0405 > candidates ≈0.0355 ≈ ones 0.0343).

## Parameterisation (important)

TSIL's `fixed_tau` **is tau_hat**, the coefficient on the identity (MSE) term:

```
loss = tau_hat * eᵀ I e + (1 - tau_hat) * eᵀ P e      (e = y - y_hat)
```

so the *kernel weight* is `1 - fixed_tau`. This pre-registration names arms by the
kernel weight, not by the config field.

## Arms (5), all else held byte-identical

| arm | config | `tau_mode` | `fixed_tau` | kernel weight |
|---|---|---|---|---|
| k005 | `configs/tsil_tau_k005.yaml` | fixed | 0.95 | 0.05 |
| k010 | `configs/tsil_tau_k010.yaml` | fixed | 0.90 | 0.10 |
| k020 | `configs/tsil_tau_k020.yaml` | fixed | 0.80 | 0.20 |
| k050 | `configs/tsil_tau_k050.yaml` | fixed | 0.50 | 0.50 (anchor = the failed run) |
| learned | `configs/tsil_tau_learned.yaml` | learned | — | learnable, `tau_hat_init = 0.0` |

Held fixed across all five arms:

- **Predicate bank**: `artifacts/llm_proposal_bank/tsil_hybrid_bank.json`, the 20
  DeepSeek-designed proposals (8 DSL composites + 12 symbolic/atomic) already
  probe-verified on the real CSI300 f2022 batch. The generator's live LLM call is
  disabled (`generator_config.enabled: false`), so the bank is identical in every arm.
- `rounds: 1` — every arm screens the *same* 20 candidates; no round-1 evolution can
  diverge the candidate sets.
- Protocol: screen `f2022`; validation `f2020`, `f2021`; seeds `[2025, 2026]`;
  LSTM, `lr=1e-3`, screen 5 epochs / 400 steps, full 20 epochs / 800 steps.
- Gates: `min_effect=0.002`, `min_stability=0.5`, `require_positive_ci=true`,
  `require_beat_control=true` (vs `control:ones`), FDR 0.10, `full_validation_top_k=4`.
- Confirmation fold `f2024` and holdout `2025-09-01…2026-03-31` are **not touched**
  by this sweep; they remain unused.

## Hypotheses

- **H1 (primary):** with the predicate bank fixed, the paired effect
  (candidate − `baseline:mse`, f2020/f2021 × 2 seeds) is monotonically decreasing in
  the kernel weight, and is positive, if at all, only at kernel weight ≤ 0.10.
  Equivalently: the previous negative result is a *tau artefact*, not a property of
  the predicates.
- **H0 (falsifier of H1):** the effect is flat in tau within ±0.001 across
  kernel weights 0.05…0.50; then the negative result is a property of the predicate
  family and the LUSI term, not of the blend weight.
- **Secondary prediction:** the learnable-`tau_hat` arm collapses onto the rank-1 P
  term (as documented in `TSIL-main/src/model_backbone_phitp.py`), i.e. tau_hat → 0
  and its effect is the worst of the five arms.

## Decision rule (fixed now, not after seeing results)

- A tau setting is declared **positive** if at least one promoted candidate reaches
  `state = validated` under the unchanged gates above.
- H1 is **supported** iff a validated candidate appears at kernel weight ≤ 0.10
  and the arm-mean effect curve is increasing as the kernel weight decreases.
- H1 is **rejected** iff no arm produces a validated candidate and the arm-mean
  effects are flat in tau (spread < 0.001).
- Any intermediate outcome (e.g. validated only at 0.20) is reported as-is; no arm
  is re-run or re-gated after the fact.

## Known limitations, stated up front

1. Adam is approximately invariant to a constant rescaling of the loss, so the
   changed total scale (`tau_hat` vs `1 − tau_hat`) is not expected to matter much;
   what changes is the kernel:MSE ratio. This is an assumption, not a verified fact.
2. The `learned` arm's tau_hat trajectory is logged in the worker metrics; a
   collapse to 0 is a prediction being tested, not a failure mode to be fixed.
3. n = 20 candidates × 4 evaluated hypotheses per arm; a null result here cannot
   prove the LUSI term is useless for *all* predicates, only for this bank.
