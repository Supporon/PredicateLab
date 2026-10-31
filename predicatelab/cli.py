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
    from predicatelab.core.engine import PredicateLabEngine

    cfg = load_config(args.config)
    engine = PredicateLabEngine(cfg, run_id="run_validate")
    try:
        ctx = engine._build_context()
        print(json.dumps({
            "domain": ctx.adapter.name,
            "protocol_hash": ctx.protocol_hash,
            "snapshot": ctx.snapshot.snapshot_id,
            "snapshot_method": ctx.snapshot.manifest_method,
            "folds": [f.fold_id for f in ctx.protocol.folds],
            "holdout": ctx.protocol.holdout_fold.fold_id if ctx.protocol.holdout_fold else None,
            "suitability": ctx.task.metadata.get("suitability"),
        }, indent=2, default=str))
    finally:
        engine.store.close()
        # validation should not leave a fake run behind
        for p in (engine.artifacts / "context.json",):
            if p.exists():
                p.unlink()
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


def cmd_finalize(args) -> int:
    from predicatelab.core.engine import finalize_run

    cfg = load_config(args.config)
    if not args.run_id or not args.candidate_id:
        raise SystemExit("--run-id and --candidate-id are required")
    result = finalize_run(cfg, args.run_id, args.candidate_id, db_path=args.db)
    print(json.dumps(result, indent=2, default=str))
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

    f = sub.add_parser("finalize", help="freeze a validated candidate and evaluate on the holdout")
    f.add_argument("--config", required=True)
    f.add_argument("--run-id", required=True)
    f.add_argument("--candidate-id", required=True)
    f.add_argument("--db", default=None)
    f.set_defaults(func=cmd_finalize)

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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
