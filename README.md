# PredicateLab V1.1

A generic predicate discovery and evidence-validation platform built around a
domain-neutral core and domain adapters.

* **TSIL** (time-series): real qlib CSI300 data, the upstream TSIL LSTM/LUSI
  loss and 23 upstream predicates plus six temporal predicates.
* **Tabular**: an independent scikit-learn adapter that validates the core
  engine without TSIL.
* **Small sample**: a controlled synthetic domain (n = 24–256) with an explicit
  latent truth, used to measure type-I error and power of the
  discovery→confirmation→holdout gates; see
  [the small-sample scenario](docs/small_sample_scenario.md).
* **Mineral prospectivity / image classification**: descriptor and adapter
  contracts only; every execution path raises `DomainNotImplemented`.

V1.1 is a scientific-trustworthiness release.  It makes the holdout a globally
consumed resource, isolates labels by the real trading calendar, keys every
cached trial by the complete experiment identity, adds run/round state
machines with resume and atomic writes, and separates discovery, confirmation
and one-time final evaluation.  It does **not** add new real TSIL results.

**Historical result, not V1.1 evidence.**  The V1.0 TSIL loop completed two
real rounds (`run_20261031_153702`); its strongest search-fold candidate was
evaluated once on the 2024–2025 holdout and did not confirm (paired IC
−0.00103, p = 0.386).  Under V1.1 that holdout is marked consumed and cannot be
read again.  Details: [TSIL results](docs/results_tsil.md),
[tabular results](docs/results_tabular.md).

## Quick start

```bash
export PYTHONPATH=.
PY=/home/dmz-ai/miniconda3/envs/zzr_py312/bin/python

# list domains
$PY -m predicatelab.cli domains

# validate data, protocol, label isolation and effective folds
$PY -m predicatelab.cli validate-config --config configs/tsil_lstm_smoke.yaml

# lifecycle: discovery -> frozen family -> confirmation -> one-time holdout
$PY -m predicatelab.cli discover --config configs/tsil_lstm_smoke.yaml --rounds 1
$PY -m predicatelab.cli freeze-family --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli confirm --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli finalize --config configs/tsil_lstm_smoke.yaml \
    --run-id run_... --candidate-id cand_...

# report, evidence and immutable bundle
$PY -m predicatelab.cli report   --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli evidence --config configs/tsil_lstm_smoke.yaml --run-id run_...
$PY -m predicatelab.cli export   --config configs/tsil_lstm_smoke.yaml \
    --run-id run_... --out artifacts/run_.../bundle

# database migration and global holdout audit
$PY -m predicatelab.cli migrate --db artifacts/predicatelab.sqlite \
    --backfill-artifacts artifacts
$PY -m predicatelab.cli holdout-audit --db artifacts/predicatelab.sqlite

# reserved-domain contract examples
$PY -m predicatelab.cli examples --domain mineral
$PY -m predicatelab.cli examples --domain image
```

Development-data lifecycle smoke (separate database, relaxed gates, not a
scientific result):

```bash
$PY -m predicatelab.cli discover --config configs/tabular_dev_v11.yaml --rounds 1
$PY -m predicatelab.cli freeze-family --config configs/tabular_dev_v11.yaml --run-id run_...
$PY -m predicatelab.cli confirm --config configs/tabular_dev_v11.yaml --run-id run_...
$PY -m predicatelab.cli finalize --config configs/tabular_dev_v11.yaml \
    --run-id run_... --candidate-id cand_...
$PY -m predicatelab.cli export --config configs/tabular_dev_v11.yaml \
    --run-id run_... --out artifacts_v11_dev_e2e/run_.../bundle
```

`discover` writes `artifacts/run_<timestamp>/` with `context.json`,
`report.json`, `report.md`, `evidence.jsonl` and per-trial logs.
`freeze-family` adds `frozen_family.json`; `confirm` adds
`confirmation.json`; `finalize` adds `frozen_manifest.json` and
`final_evaluation.json`.  `export` adds a bundle with predictions (when the
adapter emits them), `checksums.sha256` and `bundle_manifest.json`.

**Read `result.state`, not only the exit code.**  `finalize` exits 0 when
execution completed; the scientific state (`independently_confirmed`,
`resampling_confirmed`, `insufficient`, `rejected`) is in the payload, and
undefined statistics are `null` plus `null_paths`, never NaN.

## Artifacts and lightweight snapshots

Raw runs mix small reports with heavy arrays, databases and optional weights.
To migrate or commit a run, build a categorised snapshot instead of copying
`artifacts/`:

```bash
# key results + logs + figures + small analysis code, with a change record
python scripts/package_snapshot.py --artifacts-root artifacts --out published

# trained weights and prediction arrays too (local archive only)
python scripts/package_snapshot.py --include-models --include-data \
    --out snapshots/full --archive snapshots/full.tar.gz
```

The snapshot separates `results/`, `logs/`, `figures/`, `code/`, `models/`
and `data/`, and writes `MANIFEST.json`, `checksums.sha256`, a README per
experiment and `changes/` (git status/diff/log, redacted config, environment).
`models/` and `data/` are excluded unless explicitly requested, which is what
keeps the committed tree small.  Per-trial `result.json`/`job.json` are
compacted into `results/trials_summary.csv`; pass `--include-trial-details`
to copy them individually.  Result figures are produced by
`scripts/plot_run_results.py --run-dir <run>` into the run's `figures/`.
Full details: [docs/artifact_layout.md](docs/artifact_layout.md).

## Documentation

- [Code guide (中文)](docs/code_guide.md) — end-to-end walkthrough of the
  codebase: identity, lifecycle, config, artifacts, debugging, and recipes for
  extending predicates, statistics, domains and the database schema.
- [V1.1 remediation report](docs/v1.1_remediation_report.md) — what changed,
  migration record, test evidence, dev end-to-end run, remaining risks and
  `VERIFIED` / `CODE_ONLY` / `NOT_VERIFIED` / `FAILED` status.
- [V1.1 evaluation protocol](docs/v1.1_evaluation_protocol.md) — normative
  holdout, label-isolation, identity, state-machine and statistics rules.
- [V1.1 compatibility report](docs/v1.1_compatibility_report.md) — CLI, config,
  database and artifact compatibility, migration and rollback.
- [Architecture](docs/architecture.md) — core modules, identity, round
  lifecycle, leakage controls.
- [TSIL experiment](docs/tsil_experiment.md) — upstream reuse, LUSI math,
  folds, commands, limitations.
- [TSIL results](docs/results_tsil.md) — historical run.
- [Generic tabular results](docs/results_tabular.md).
- [Small-sample scenario](docs/small_sample_scenario.md) and its
  [formal pre-registered study](docs/results_small_sample_20261108.md) —
  measured type-I error and power of the discovery→confirmation→holdout gates
  at n = 24–256.
- [Extending domains](docs/extending_domains.md) — adapter contract,
  mineral/image reservations, checklist.
- [Evidence and limitations](docs/evidence_and_limitations.md).
- [交付说明 / handover guide (中文)](docs/HANDOVER_zh.md) — what the package
  contains, how to run it, how to read results, and which claims are not yet
  supported.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. $PY -m pytest -q
# 190 passed
```

(`pytest.ini` disables qlib's optional `fugue_test` plugin, which imports
`fsspec` at configure time and is unrelated to this project.)

## Environment

Developed with Python 3.12, pydantic 2.11, numpy/pandas, scikit-learn 1.5,
torch (CUDA), qlib 0.9.6 and PyYAML 6.  TSIL data:
`/home/dmz-ai/zhengzengrong/FinTSB-main/qlib_data/cn_data` (override in the
config).  A cached TSIL `validate-config` takes about 28 s and ~590 MB because
the full data store is hashed; the manifest cache is self-validating.  See the
limitations documents for what was not run.
