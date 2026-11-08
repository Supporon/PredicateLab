"""PredicateLab command-line interface.

Examples
--------
    python -m predicatelab.cli domains
    python -m predicatelab.cli validate-config --config configs/tabular_quick.yaml
    python -m predicatelab.cli discover --config configs/tabular_quick.yaml --rounds 2
    python -m predicatelab.cli report --config configs/tabular_quick.yaml --run-id run_...
    python -m predicatelab.cli finalize --config configs/tsil_lstm_quick.yaml \
        --run-id run_... --candidate-id cand_...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root for the snapshot's ``changes/`` record.

    Prefer the git work tree containing the package; fall back to the package
    parent so a source checkout still works without git.
    """
    candidate = Path(__file__).resolve().parents[1]
    try:
        import subprocess

        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(candidate), capture_output=True, text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip())
    except OSError:
        pass
    return candidate


def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {p}")
    with p.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit("config must be a mapping")
    return cfg


def cmd_domains(_args) -> int:
    from predicatelab.domains import executable_domains, reserved_domains

    print(json.dumps({"executable": executable_domains(), "reserved": reserved_domains()}, indent=2))
    return 0


def cmd_validate_config(args) -> int:
    import tempfile

    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    tmp = Path(tempfile.mkdtemp(prefix="predicatelab_validate_"))
    cfg = json.loads(json.dumps(cfg, default=str))
    cfg.setdefault("run", {})
    cfg["run"]["artifacts_dir"] = str(tmp / "artifacts")
    cfg["run"]["db"] = str(tmp / "validate.sqlite")
    engine = PredicateLabEngine(cfg, db_path=cfg["run"]["db"], run_id="run_validate")
    try:
        ctx = engine._build_context()
        out = {
            "domain": ctx.adapter.name,
            "protocol_hash": ctx.protocol_hash,
            "identity_hash": ctx.context_hash,
            "snapshot": ctx.snapshot.snapshot_id,
            "snapshot_method": ctx.snapshot.manifest_method,
            "data_manifest_hash": ctx.data_manifest_hash,
            "label_expression": ctx.label_expression,
            "label_offset_trading_days": ctx.label_offset,
            "label_isolation": ctx.label_report,
            "folds": [f.fold_id for f in ctx.protocol.folds],
            "confirmation_folds": [f.fold_id for f in ctx.protocol.confirmation_folds],
            "holdout": ctx.protocol.holdout_fold.fold_id if ctx.protocol.holdout_fold else None,
            "effective_folds": [f.model_dump(mode="json") for f in ctx.effective_folds],
            "effective_confirmation": [f.model_dump(mode="json") for f in ctx.effective_confirmation],
            "effective_holdout": (ctx.effective_holdout.model_dump(mode="json")
                                  if ctx.effective_holdout else None),
            "seeds": ctx.protocol.seeds,
            "suitability": ctx.task.metadata.get("suitability"),
        }
        print(json.dumps(out, indent=2, default=str))
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error": f"{type(exc).__name__}: {exc}"},
                         indent=2, default=str), file=sys.stderr)
        return 2
    finally:
        engine.store.close()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def cmd_discover(args) -> int:
    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    engine = PredicateLabEngine(cfg, db_path=args.db, run_id=args.run_id)
    try:
        run_id = engine.run(rounds=args.rounds)
        print(f"run_id={run_id}")
        print(f"artifacts={engine.artifacts}")
    finally:
        engine.store.close()
    return 0


def cmd_report(args) -> int:
    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    if not args.run_id:
        raise SystemExit("--run-id is required")
    engine = PredicateLabEngine(cfg, db_path=args.db, run_id=args.run_id)
    try:
        engine._build_context()
        engine._write_report()
        report = json.loads((engine.artifacts / "report.json").read_text(encoding="utf-8"))
        print(json.dumps(report["evidence_counts"], indent=2))
        print(f"report={engine.artifacts / 'report.md'}")
    finally:
        engine.store.close()
    return 0


def cmd_freeze(args) -> int:
    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    if not args.run_id:
        raise SystemExit("--run-id is required")
    engine = PredicateLabEngine(cfg, db_path=args.db, run_id=args.run_id)
    try:
        manifest = engine.freeze_family(
            candidate_ids=args.candidate_id or None,
            top_k=args.top_k,
        )
        print(json.dumps({k: manifest[k] for k in
                          ("family_id", "manifest_hash", "run_id", "members")},
                         indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"freeze blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.store.close()


def cmd_confirm(args) -> int:
    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    if not args.run_id:
        raise SystemExit("--run-id is required")
    engine = PredicateLabEngine(cfg, db_path=args.db, run_id=args.run_id)
    try:
        summary = engine.confirm(candidate_ids=args.candidate_id or None, top_k=args.top_k)
        print(json.dumps(summary, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"confirmation blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.store.close()


def cmd_finalize(args) -> int:
    from predicatelab.core.engine import finalize_run

    cfg = load_config(args.config)
    if not args.run_id or not args.candidate_id:
        raise SystemExit("--run-id and --candidate-id are required")
    try:
        result = finalize_run(cfg, args.run_id, args.candidate_id, db_path=args.db)
    except Exception as exc:
        print(f"finalize blocked: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_migrate(args) -> int:
    from predicatelab.core.store import SCHEMA_VERSION, StateStore

    db = args.db or "artifacts/predicatelab.sqlite"
    store = StateStore(db)
    try:
        out: dict[str, Any] = {
            "db": db,
            "schema_version": store.schema_version,
            "target_version": SCHEMA_VERSION,
            "migrations": [dict(r) for r in store._conn.execute(
                "SELECT * FROM schema_migrations ORDER BY version")],
            "counts": {},
        }
        for table in ("tasks", "snapshots", "templates", "candidates", "runs", "rounds",
                      "trials", "trial_attempts", "evidence", "feedback", "final_evals",
                      "holdout_access", "holdout_access_events", "frozen_families"):
            out["counts"][table] = store._conn.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if args.backfill_artifacts:
            out["holdout_backfill"] = store.backfill_holdout_ledger(args.backfill_artifacts)
        print(json.dumps(out, indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_holdout_audit(args) -> int:
    from predicatelab.core.store import StateStore

    db = args.db or "artifacts/predicatelab.sqlite"
    store = StateStore(db)
    try:
        rows = store.list_holdout_access()
        print(json.dumps({
            "n_rows": len(rows),
            "rows": [{
                "access_id": r["access_id"], "holdout_key": r["holdout_key"][:16],
                "run_id": r["run_id"], "candidate_id": r["candidate_id"],
                "status": r["status"], "purpose": r["purpose"], "origin": r["origin"],
                "test_start": r["test_start"], "test_end": r["test_end"],
            } for r in rows],
        }, indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_evidence(args) -> int:
    from predicatelab.core.store import StateStore

    cfg = load_config(args.config) if args.config else {}
    db = args.db or cfg.get("run", {}).get("db", "artifacts/predicatelab.sqlite")
    store = StateStore(db)
    try:
        rows = store.list_evidence(run_id=args.run_id)
        print(f"{'candidate':22s} {'status':22s} {'effect':>9s} {'q':>7s} {'stab':>5s}  name")
        for r in rows:
            print(f"{r['candidate_id'][:22]:22s} {r['status']:22s} "
                  f"{(r['effect'] if r['effect'] is not None else float('nan')):+9.4f} "
                  f"{(r['q_value'] if r['q_value'] is not None else float('nan')):7.3g} "
                  f"{(r['stability'] if r['stability'] is not None else float('nan')):5.2f}  "
                  f"{(r.get('reason') or '')[:70]}")
    finally:
        store.close()
    return 0


def cmd_examples(args) -> int:
    from predicatelab.domains.image.adapter import ImageAdapter
    from predicatelab.domains.mineral.adapter import MineralAdapter

    if args.domain == "mineral":
        rows = MineralAdapter.example_templates()
    elif args.domain == "image":
        rows = ImageAdapter.example_templates()
    else:
        raise SystemExit("examples are available for mineral and image")
    print(json.dumps([t.model_dump(mode="json") for t in rows], indent=2))
    return 0


def cmd_export(args) -> int:
    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    engine = PredicateLabEngine(cfg, db_path=args.db, run_id=args.run_id)
    try:
        out = engine.export(args.out)
        print(f"exported to {out}")
    finally:
        engine.store.close()
    return 0


def cmd_publish(args) -> int:
    from predicatelab.core.artifacts import build_run_snapshot

    cfg = load_config(args.config)
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_id:
        run_dir = Path(run_cfg.get("artifacts_dir", "artifacts")) / args.run_id
    else:
        raise SystemExit("publish requires --run-id or --run-dir")
    out = build_run_snapshot(
        run_dir, Path(args.out),
        include_models=args.include_models,
        include_data=args.include_data,
        include_trial_details=args.include_trial_details,
        config=cfg,
        config_path=args.config,
        repo_root=repo_root(),
    )
    print(f"snapshot={out}")
    print(f"manifest={out / 'MANIFEST.json'}")
    return 0


def cmd_publish_all(args) -> int:
    from predicatelab.core.artifacts import build_workspace_snapshot

    cfg = load_config(args.config) if args.config else {}
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    root = Path(args.artifacts_dir or run_cfg.get("artifacts_dir", "artifacts"))
    configs_dir = Path(args.configs_dir) if args.configs_dir else None
    out = build_workspace_snapshot(
        root, Path(args.out),
        include_models=args.include_models,
        include_data=args.include_data,
        include_trial_details=args.include_trial_details,
        configs_dir=configs_dir,
        repo_root=repo_root(),
        experiments=args.experiment or None,
    )
    print(f"snapshot={out}")
    print(f"index={out / 'INDEX.md'}")
    print(f"manifest={out / 'MANIFEST.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="predicatelab", description="Predicate discovery & evidence-validation platform")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("domains", help="list registered domains")
    d.set_defaults(func=cmd_domains)

    v = sub.add_parser("validate-config", help="validate config, protocol and data suitability")
    v.add_argument("--config", required=True)
    v.set_defaults(func=cmd_validate_config)

    s = sub.add_parser("discover", help="run candidate discovery rounds")
    s.add_argument("--config", required=True)
    s.add_argument("--rounds", type=int, default=None)
    s.add_argument("--run-id", default=None)
    s.add_argument("--db", default=None)
    s.set_defaults(func=cmd_discover)

    r = sub.add_parser("report", help="rebuild report for a run")
    r.add_argument("--config", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--db", default=None)
    r.set_defaults(func=cmd_report)

    fz = sub.add_parser("freeze-family", help="freeze the discovery family before confirmation")
    fz.add_argument("--config", required=True)
    fz.add_argument("--run-id", required=True)
    fz.add_argument("--candidate-id", action="append", default=None)
    fz.add_argument("--top-k", type=int, default=None)
    fz.add_argument("--db", default=None)
    fz.set_defaults(func=cmd_freeze)

    cf = sub.add_parser("confirm", help="independent confirmation on frozen family")
    cf.add_argument("--config", required=True)
    cf.add_argument("--run-id", required=True)
    cf.add_argument("--candidate-id", action="append", default=None)
    cf.add_argument("--top-k", type=int, default=None)
    cf.add_argument("--db", default=None)
    cf.set_defaults(func=cmd_confirm)

    f = sub.add_parser("finalize", help="freeze a validated candidate and evaluate on the holdout")
    f.add_argument("--config", required=True)
    f.add_argument("--run-id", required=True)
    f.add_argument("--candidate-id", required=True)
    f.add_argument("--db", default=None)
    f.set_defaults(func=cmd_finalize)

    mg = sub.add_parser("migrate", help="upgrade SQLite schema in place and verify preservation")
    mg.add_argument("--db", default=None)
    mg.add_argument("--backfill-artifacts", default=None,
                    help="artifacts root used to backfill historical holdout consumption")
    mg.set_defaults(func=cmd_migrate)

    ha = sub.add_parser("holdout-audit", help="list the global holdout access ledger")
    ha.add_argument("--db", default=None)
    ha.set_defaults(func=cmd_holdout_audit)

    e = sub.add_parser("evidence", help="list evidence for a run")
    e.add_argument("--config", default=None)
    e.add_argument("--run-id", default=None)
    e.add_argument("--db", default=None)
    e.set_defaults(func=cmd_evidence)

    x = sub.add_parser("examples", help="show reserved-domain descriptor examples")
    x.add_argument("--domain", required=True, choices=["mineral", "image"])
    x.set_defaults(func=cmd_examples)

    ex = sub.add_parser("export", help="copy the evidence bundle")
    ex.add_argument("--config", required=True)
    ex.add_argument("--run-id", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--db", default=None)
    ex.set_defaults(func=cmd_export)

    pub = sub.add_parser("publish", help="build a lightweight snapshot of one run")
    pub.add_argument("--config", required=True)
    pub.add_argument("--run-id", default=None)
    pub.add_argument("--run-dir", default=None)
    pub.add_argument("--out", required=True)
    pub.add_argument("--include-models", action="store_true",
                     help="also copy trained weights (models/)")
    pub.add_argument("--include-data", action="store_true",
                     help="also copy predictions, probes and databases (data/)")
    pub.add_argument("--include-trial-details", action="store_true",
                     help="also copy per-trial result.json/job.json (bulk)")
    pub.set_defaults(func=cmd_publish)

    pa = sub.add_parser("publish-all",
                        help="snapshot every experiment under an artifacts root")
    pa.add_argument("--config", default=None)
    pa.add_argument("--artifacts-dir", default=None)
    pa.add_argument("--out", default="published")
    pa.add_argument("--configs-dir", default=None,
                    help="copy *.yaml configs into changes/configs")
    pa.add_argument("--experiment", action="append", default=None,
                    help="limit to one experiment directory (repeatable)")
    pa.add_argument("--include-models", action="store_true")
    pa.add_argument("--include-data", action="store_true")
    pa.add_argument("--include-trial-details", action="store_true",
                    help="also copy per-trial result.json/job.json (bulk)")
    pa.set_defaults(func=cmd_publish_all)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
