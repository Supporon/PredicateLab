#!/usr/bin/env python
"""Package experiment artifacts into a lightweight, portable snapshot.

The raw ``artifacts/`` tree is the source of truth and stays untouched.  This
script writes a separate ``published/`` tree with only the categories you ask
for:

    results   report.json/.md, context, evidence, frozen family/confirmation,
              final evaluation, per-trial result.json
    logs      runner, worker and training logs
    figures   generated plots (see ``scripts/plot_run_results.py``)
    code      small analysis/reproduction scripts
    models    trained weights            (opt-in, ``--include-models``)
    data      predictions, probes, DBs   (opt-in, ``--include-data``)

Every snapshot also carries ``changes/`` (git status/diff/log, a redacted
config snapshot, package versions), ``MANIFEST.json`` and
``checksums.sha256``.  Models and data are excluded by default so the tree is
safe to commit; use ``--archive`` for a full local tarball instead.

Examples
--------
    # lightweight snapshot of every experiment (what goes to GitHub)
    python scripts/package_snapshot.py

    # one experiment only, with figures already generated
    python scripts/package_snapshot.py --experiment tsil_tau_k005

    # full archival tarball, weights and prediction arrays included
    python scripts/package_snapshot.py --include-models --include-data \
        --out snapshots/20261108_full --archive snapshots/20261108_full.tar.gz
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predicatelab.core.artifacts import (  # noqa: E402
    DEFAULT_INCLUDE,
    build_workspace_snapshot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts-root", default=str(REPO_ROOT / "artifacts"),
                        help="raw experiment output (read-only)")
    parser.add_argument("--out", default=str(REPO_ROOT / "published"),
                        help="snapshot destination (default: published/)")
    parser.add_argument("--experiment", action="append", default=None,
                        help="limit to one experiment directory (repeatable)")
    parser.add_argument("--include-models", action="store_true",
                        help="copy trained weights (heavy)")
    parser.add_argument("--include-data", action="store_true",
                        help="copy predictions/probes/databases (heavy)")
    parser.add_argument("--include-trial-details", action="store_true",
                        help="copy per-trial result.json/job.json (bulk; default: "
                             "only results/trials_summary.csv)")
    parser.add_argument("--configs-dir", default=str(REPO_ROOT / "configs"),
                        help="copy *.yaml configs into changes/configs")
    parser.add_argument("--clean", action="store_true",
                        help="remove the destination first (never touches artifacts/)")
    parser.add_argument("--archive", default=None,
                        help="write a .tar.gz of the finished snapshot")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_root = Path(args.artifacts_root).resolve()
    out = Path(args.out).resolve()
    if not artifacts_root.is_dir():
        raise SystemExit(f"artifacts root not found: {artifacts_root}")
    if out == artifacts_root or artifacts_root.is_relative_to(out):
        raise SystemExit("refusing to write the snapshot inside the artifacts root")
    if args.clean and out.exists():
        shutil.rmtree(out)

    include = list(DEFAULT_INCLUDE)
    if args.include_models:
        include.append("models")
    if args.include_data:
        include.append("data")

    configs_dir = Path(args.configs_dir)
    snapshot = build_workspace_snapshot(
        artifacts_root,
        out,
        include=include,
        include_models=args.include_models,
        include_data=args.include_data,
        include_trial_details=args.include_trial_details,
        configs_dir=configs_dir if configs_dir.is_dir() else None,
        repo_root=REPO_ROOT,
        experiments=args.experiment or None,
    )
    print(f"snapshot={snapshot}")
    print(f"index={snapshot / 'INDEX.md'}")
    print(f"manifest={snapshot / 'MANIFEST.json'}")
    print(f"kinds={','.join(include)}")

    if args.archive:
        archive = Path(args.archive).resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(snapshot, arcname=snapshot.name)
        print(f"archive={archive} ({archive.stat().st_size / 1048576:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
