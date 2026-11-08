# Artifact layout and snapshots

A run writes a lot of files, and they do not have the same lifetime:

* `report.md`, `final_evaluation.json` and `worker.log` explain what happened
  and should survive a migration or a GitHub commit;
* `predictions.npz`, probe batches, the 13 MB data manifest and the SQLite
  ledger are bulk data — useful locally, expensive in a repository;
* trained weights are the bulkiest output of all and are usually rebuilt from
  the config, not archived with the source.

PredicateLab keeps the raw run as the source of truth and builds a separate,
categorised snapshot from it. Nothing below ever deletes or rewrites a raw
artifact.

## 1. Categories

`predicatelab/core/artifacts.py` classifies every file by directory name and
suffix.

| kind | contents | default snapshot |
|---|---|---|
| `results` | `report.json` / `report.md`, `context.json`, `evidence.jsonl`, `frozen_family.json`, `confirmation.json`, `frozen_manifest.json`, `final_evaluation.json`; per-trial `result.json` is folded into `results/trials_summary.csv` | yes |
| `logs` | runner, worker and training logs (`*.log`, `*.out`, `*.err`) | yes |
| `figures` | plots (`*.png`, `*.svg`, `*.pdf`, `*.jpg`) | yes |
| `code` | small analysis / reproduction scripts (`*.py`, `*.sh`) | yes |
| `models` | trained weights (`*.pt`, `*.pth`, `*.bin`, `*.ckpt`, `*.safetensors`) | **no** |
| `data` | `*.npz`/`*.npy`/`*.pkl`, `_manifests/`, `_probe_batches/`, `predictions/`, SQLite databases | **no** |

Explicit directories win over suffixes: a file in `models/` is always a
model, a file in `logs/` is always a log. Unrecognised small files fall back
to `results` — publishing a forgotten report is safer than publishing a
forgotten weight file.

## 2. Raw versus published

```
artifacts/                       # raw, git-ignored, local only
  tsil_tau_k005/
    predicatelab.sqlite          # data
    _manifests/…json             # data (13 MB)
    _probe_batches/…             # data
    llm_proposals/…json          # results (small provenance)
    run_20261108_110259/
      context.json               # results
      report.json / report.md    # results
      evidence.jsonl             # results
      trials/…/result.json       # results
      trials/…/predictions.npz   # data
      trials/…/models/model.pt   # models (opt-in)
      trials/…/worker.log        # logs
      figures/…                  # figures (generated)

published/                       # lightweight, safe to commit
  INDEX.md
  MANIFEST.json
  checksums.sha256
  changes/
    git_status.txt
    git_diff.patch
    git_diff_stat.txt
    git_log.txt
    environment.json
    config_snapshot.yaml
    configs/*.yaml
  tsil_tau_k005/
    README.md
    snapshot_summary.json
    results/…
    logs/…
    figures/…
    code/…
```

`published/` is produced by the packager, never edited by hand.

## 3. Commands

One run:

```bash
python -m predicatelab.cli publish \
    --config configs/tsil_tau_k005.yaml \
    --run-id run_20261108_110259 \
    --out published/tsil_tau_k005/run_20261108_110259
```

Every experiment under an artifacts root:

```bash
python scripts/package_snapshot.py \
    --artifacts-root artifacts --out published --configs-dir configs
```

Heavy variants (local archive, not for GitHub):

```bash
# include trained weights and prediction arrays
python scripts/package_snapshot.py --include-models --include-data \
    --out snapshots/20261108_full --archive snapshots/20261108_full.tar.gz
```

`publish`/`publish-all` in the CLI are equivalent to the script for the
common cases. The script adds `--clean` and `--archive`.

### Per-trial detail

Hundreds of trial directories each hold a `result.json` (and sometimes a
`job.json`). Copying them all costs tens of megabytes and says little that a
table cannot. The default snapshot therefore keeps run-level results and writes
one compact table per experiment:

```
published/<experiment>/results/trials_summary.csv
```

with the columns `run_id, phase, fold, candidate, variant, seed, status, ic,
rank_ic, icir, mse, duration_s, model_path, error`. The count and size of the
omitted files are recorded in `MANIFEST.json` under
`excluded_trial_details`, and the per-experiment README states the same.

Pass `--include-trial-details` when you need every raw `result.json`/`job.json`
(for example to audit one failing fold). The raw runs are never modified
either way.

## 4. Trained models

The TSIL worker does not save weights on every screen trial by default.
`execution` in the config controls it:

```yaml
execution:
  save_models: false        # every trial — heavy
  save_final_models: true   # final/holdout trials — keeps an auditable model
```

When enabled, the worker writes `models/model.pt` inside the trial directory
with the state dict plus the model type, model config, seed, fold and
descriptor. The path and SHA-256 are recorded in the trial's `worker`
metadata, so a model can be matched to its result without a schema change.

`models/` is excluded from snapshots unless `--include-models` is passed, and
`.gitignore` keeps it out of commits even then.

## 5. Figures

`scripts/plot_run_results.py` renders two static figures into
`<run>/figures/`:

* `effects_forest.{png,svg}` — candidate paired effect with 95% CI (if the
  evidence record has one), status encoded by colour **and** marker shape;
* `ic_series.{png,svg}` — daily cross-sectional IC for the largest-|effect|
  candidate vs the MSE baseline and the control kernel, one small-multiple
  panel per fold, raw hairline plus a 20-day rolling mean.

```bash
python scripts/plot_run_results.py \
    --run-dir artifacts/tsil_tau_k005/run_20261108_110259
```

Run this before packaging and the figures land in the snapshot automatically.
The plots are deterministic; there is no interactive layer in a static PNG.

## 6. What a snapshot records

`MANIFEST.json` is the machine-readable index:

* `include` / `excluded_kinds` — what was and was not copied, with file
  counts and byte totals for every category;
* `files[]` — destination path, kind, source path, byte size and SHA-256;
* `run_facts` (single-run) or `experiments[]` (workspace) — run status,
  evidence counts, rounds and final-evaluation state;
* `tool` — Python/platform and package versions;
* `change_info` — pointers into `changes/`.

`checksums.sha256` is `sha256  relative/path` and covers every published file.
`README.md`/`INDEX.md` are the human-readable equivalent, including the
credibility note: discovery-positive is `exploratory_positive`, not
independent confirmation, and `docs/audit_20261107/REPORT.md` must be read
before citing TSIL/LUSI results.

## 7. Git policy

The repository `.gitignore` keeps `artifacts/`, `models`, `data` arrays and
databases out of Git, and re-includes logs/CSV **inside** `published/` and
`snapshots/`. The intent:

| tree | committed | why |
|---|---|---|
| `artifacts/` | no | raw, reproduced from config + data |
| `published/<experiment>/` | yes | results, logs, figures, change record |
| `published/**/models/` | no | heavy, opt-in archive only |
| `published/**/data/` | no | heavy, opt-in archive only |
| `snapshots/*.tar.gz` | no | local archive |

## 8. Migrating an old run

Nothing special is required: the packager reads the existing layout. Run

```bash
python scripts/package_snapshot.py --artifacts-root artifacts --out published
```

and the old `run_*` directories are categorised by the same rules. To verify
what will be included before writing anything:

```bash
python -c "from predicatelab.core.artifacts import collect_artifacts, summarize_entries; \
import json; print(json.dumps(summarize_entries(collect_artifacts('artifacts')), indent=2))"
```
