# TSIL experiment: predicates under the real LUSI loss

This is the primary scientific scenario of V1.0.  It uses the upstream TSIL
code in `TSIL-main/` unchanged (except for an optional `phi_fn` injection hook
that already existed in the loss) and the qlib CSI300 Alpha158 panel at
`/home/dmz-ai/zhengzengrong/FinTSB-main/qlib_data/cn_data`.

## 1. What is reused and what is not changed

| upstream piece | use |
|---|---|
| `src/model_backbone_phitp.py` | `QniverseModel`, LSTM backbone, `WeightedMSELoss` |
| LUSI loss | `M = τ̂ I + τ P`, `L = mean(eᵀ M e)`, `e = outputs - targets` (raw residual), `τ̂ = sigmoid(alpha)`, `τ = 1 - τ̂` |
| `src/temporal_predicates.py` | six paper predicates (`phi_timestamp`, `phi_cycle`, `phi_phase`, `phi_momentum`, `phi_volatility`, `phi_return_dist`) |
| `src/custom_phi_constructors.py` | 17 exploratory predicates and the parameterised variants |
| `src/dataset_ori_onehot.py` | `MTSDatasetH`, sequence construction, timestamp features |
| qlib `Alpha158` | 158 features, label `Ref($close,-2)/Ref($close,-1)-1` |

The mathematical semantics are untouched:

* candidates are injected through `QniverseModel(phi_fn=compiled)` → the loss
  calls `proxy(batch_x, timestamp_x) -> (I, P)`;
* native predicates are delegated verbatim to the upstream functions; the
  platform does not re-normalise or re-weight them;
* DSL descriptors implement the same contract: a per-sample descriptor φ,
  global L2 normalization, `P = φ̂ φ̂ᵀ`, `I = eye(B)`;
* baseline `phi_type='mse'` is the upstream plain-MSE path (`use_weighted =
  False`); control `phi_type='ones'` is the upstream Φ_ones path; candidates
  use `phi_type='external'` with the compiled `phi_fn`.

`test_eval` / `fit` semantics were checked: `QniverseModel.fit` always calls
`dataset.prepare(["train","valid","test"])` and reports metrics on the `test`
segment.  During search the worker aliases the `test` segment to the validation
fold, so the upstream API stays intact and the holdout is never loaded.  The
`finalize` entry builds the dataset with the real `test` dates.

## 2. Candidate knowledge

Two generator families are available (`search.generator`):

* `rule` — deterministic seeds plus feedback-driven mutations and exploration.
  Seeds are the six paper predicates, the 17 exploratory predicates and
  structured DSL descriptors (momentum, volatility, dispersion, autocorrelation,
  drawdown, tail, trend).
* `llm_json` — reads a JSON list of proposals (`generator_config.proposals_path`).
  The file is data only; every descriptor is statically validated, probed and
  compiled by the trusted DSL.  The LLM never receives an execution path.

Mutations include window shifts, feature-slice changes, parameterised native
variants and `combine='mix'` composites that blend two P matrices.  Composites
of DSL families may also use `combine='concat'`.

## 3. Protocol

Expanding-window walk-forward with a one-year validation block:

| fold | train | validation |
|---|---|---|
| f2020 | 2018-01-01 .. 2019-12-31 | 2020-01-01 .. 2020-12-31 |
| f2021 | 2018-01-01 .. 2020-12-31 | 2021-01-01 .. 2021-12-31 |
| f2022 | 2018-01-01 .. 2021-12-31 | 2022-01-01 .. 2022-12-31 |
| f2023 | 2018-01-01 .. 2022-12-31 | 2023-01-01 .. 2023-12-31 |

* purge: 2 trading days from the end of each training block (label window is
  two days);
* embargo: 3 trading days from the start of each validation block;
* V1.1 discovery fold policy: screening `f2022`; full validation `f2020`–`f2022`;
* V1.1 confirmation fold: `f2023` (`confirmation_kind: time_forward`); it is
  not visible to the generator or the search policy;
* holdout: 2024-01-01 .. 2025-08-31, stored as `holdout_fold` and reachable
  only through the one-time ledger-guarded `finalize`;
* 2023-12-28/29 label tails are trimmed by the real trading calendar (600
  leaked sample-dates at 300 instruments/date in the unchecked V1.0 layout).

The worker's handler fits `RobustZScoreNorm` and `Fillna` on the training
segment (`fit_start_time`/`fit_end_time`), and uses `DropnaLabel` +
`CSRankNorm` for the label.  Sequence length is 20 trading days.

## 4. Commands

```bash
export PYTHONPATH=.
PY=/home/dmz-ai/miniconda3/envs/zzr_py312/bin/python

$PY -m predicatelab.cli validate-config --config configs/tsil_lstm_quick.yaml
$PY -m predicatelab.cli discover --config configs/tsil_lstm_quick.yaml --rounds 2
$PY -m predicatelab.cli evidence --config configs/tsil_lstm_quick.yaml --run-id run_...
$PY -m predicatelab.cli report   --config configs/tsil_lstm_quick.yaml --run-id run_...
$PY -m predicatelab.cli freeze-family --config configs/tsil_lstm_quick.yaml --run-id run_...
$PY -m predicatelab.cli confirm  --config configs/tsil_lstm_quick.yaml --run-id run_...
$PY -m predicatelab.cli finalize --config configs/tsil_lstm_quick.yaml \
    --run-id run_... --candidate-id cand_...
```

Do not run the finalize step for 2024–2025 in this phase: that holdout is
already consumed and the ledger will refuse it.  The V1.1 TSIL discovery and
confirmation path is validated by config checks and unit tests but has not yet
been executed on real data (`CODE_ONLY`).

`configs/tsil_lstm_quick.yaml` is a compute-limited profile: 3 epochs /
150 steps for screening, 8 epochs / 300 steps for full validation, LSTM
`d_model=32`.  The data, model family, loss, folds and metrics are real; only
the training budget is reduced.  `configs/tsil_lstm_full.yaml` is the
paper-scale budget (50 epochs, `d_model=64`) and is intended for a dedicated
GPU allocation.

## 5. Results

See `docs/results_tsil.md` for the run actually executed in this environment
(`run_20261031_153702`).  Outcome in one line: four candidates (one
`volatility` family plus a borderline `drawdown` mutation) validated on the
rolling search folds, but the strongest candidate **failed to confirm** on the
frozen 2024–2025 holdout (paired IC −0.00103, CI [−0.00328, +0.00110],
p = 0.386).  The document also records the shared-GPU contention, reduced
budget and rule-based generator caveats.

## 6. Limitations and blockers

* **No `research_stack` exists in this repository.**  Only `TSIL-main/` is
  present.  The platform was built around the TSIL code and a fresh core; there
  was no existing executor/state manager to reuse beyond TSIL itself.
* **No LLM API is configured.**  The `llm_json` generator interface is
  implemented and tested, but the executed discovery runs use the deterministic
  rule generator.  No LLM-proposed result is claimed.
* **GPU contention.**  The shared 8×A100 node was at 100% utilisation by other
  jobs during the run; per-trial time was 55–110 s for the reduced profile.
  This is a throughput limitation, not a change to the experiment.
* **Reduced training budget.**  Conclusions are conditional on the reduced
  budget; they should not be read as reproducing the TSIL paper's numbers.
* **Survivorship / universe.**  The instrument membership is whatever the
  supplied qlib store contains; no point-in-time index reconstruction was done.
