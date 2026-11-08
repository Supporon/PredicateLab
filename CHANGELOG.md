# Changelog

All notable changes to the repository. Dates are ISO-8601 (UTC+8 local time).

## 2026-11-08 — Artifact taxonomy and lightweight snapshots

**Goal.** Keep raw experiment output local while making migration snapshots and
GitHub commits small, without losing training logs, key results or the record
of what changed.

### Added

* `predicatelab/core/artifacts.py` — one classification point for every
  artifact: `results`, `logs`, `figures`, `models`, `data` or `code`.
  Unrecognised small files default to `results`; models/data are the explicit
  heavy categories.
* `scripts/package_snapshot.py` — builds a categorised `published/` tree from
  `artifacts/`, with `INDEX.md`, `MANIFEST.json`, `checksums.sha256` and a
  `changes/` record (git status/diff/log, redacted config snapshot, package
  versions). Supports `--include-models`, `--include-data`, `--archive`.
* `scripts/plot_run_results.py` — deterministic result figures written to
  `<run>/figures/`: `effects_forest.{png,svg}` (paired effect + 95% CI,
  status by colour **and** marker shape) and `ic_series.{png,svg}` (IC vs MSE
  baseline and control kernel, one small-multiple panel per fold, raw hairline
  plus 20-day rolling mean).
* CLI `publish` and `publish-all` subcommands, plus
  `PredicateLabEngine.publish_snapshot(...)`.
* Per-trial `result.json`/`job.json` are compacted into
  `results/trials_summary.csv` by default (columns: run, phase, fold,
  candidate, variant, seed, status, ic, rank_ic, icir, mse, duration, model
  path, error). The omitted file count and bytes are recorded in
  `MANIFEST.json` under `excluded_trial_details`; `--include-trial-details`
  copies them individually. This is what took the snapshot from 44 MB to
  18 MB.
* `docs/artifact_layout.md`, this changelog, and a README section describing
  the raw-vs-published split.

### Changed

* `execution.save_models` / `execution.save_final_models` control whether the
  TSIL worker writes `models/model.pt` next to a trial. Screen/discovery trials
  stay weightless by default; final/holdout trials keep their model by default.
  The model path and SHA-256 are recorded in the trial's `worker` metadata, so
  the trial schema and result hashes are unchanged.
* `.gitignore` re-includes logs and CSV **inside** `published/` and
  `snapshots/`, while keeping `models/`, `data/`, databases and archives out
  of Git.

### Not changed

* Raw `artifacts/` layout, report/evidence formats, database schema and all
  scientific gates. Existing runs can be packaged as-is.

### Verification

* `CUDA_VISIBLE_DEVICES="" python -m pytest tests -q` — 182 passed before the
  change, 190 after, including 8 new artifact/snapshot tests.
* Rebuilding `published/` from the current runs republishes 13 experiments at
  18 MB (results 5.3 MB, logs 4.6 MB, figures 5.9 MB, code 22 KB) while
  excluding 478 MB of data and 24.7 MB of per-trial detail; `checksums.sha256`
  verifies cleanly.
