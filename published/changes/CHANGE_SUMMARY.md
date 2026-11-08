# Change record

This directory describes the code and configuration state used to build
the snapshot. The raw run directories are the source of truth; this is
the lightweight, portable record.

- published kinds: `results, logs, figures, code`
- git commit: `f3e2ed7f389f` on `master`
- working tree diff: `changes/git_diff.patch`

## Experiments in this snapshot

- `exp_logs`: 20 files, 40.4 KiB
- `llm_proposal_bank`: 3 files, 55.5 KiB
- `_root`: 1 files, 22.4 KiB
- `run_s1`: 1 files, 190.7 KiB
- `tsil_llm_dsl_composite`: 95 files, 2.1 MiB
- `tsil_llm_hybrid`: 97 files, 2.1 MiB
- `tsil_llm_novel_symbolic`: 97 files, 2.1 MiB
- `tsil_llm_rule_baseline`: 89 files, 1.8 MiB
- `tsil_tau_k005`: 59 files, 1.3 MiB
- `tsil_tau_k010`: 59 files, 1.3 MiB
- `tsil_tau_k020`: 59 files, 1.3 MiB
- `tsil_tau_k050`: 59 files, 1.3 MiB
- `tsil_tau_learned`: 59 files, 1.3 MiB

## Reproduce

```bash
python -m predicatelab.cli discover --config <config> --rounds <n>
python scripts/package_snapshot.py --artifacts-root artifacts --out published
```
