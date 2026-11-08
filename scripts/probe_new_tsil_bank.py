"""Real-data CSI300 probe verification for the new TSIL symbolic / operator / composite bank.

Builds the discovery probe batch from the real qlib CSI300 Alpha158 store, then
runs the production ``TSILAdapter.probe`` gate on every candidate in the curated
LLM bank plus all new atomic operator families across several windows.

Usage::

    python scripts/probe_new_tsil_bank.py --config configs/tsil_llm_experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from predicatelab.core.engine import PredicateLabEngine  # noqa: E402
from predicatelab.domains.tsil import dsl  # noqa: E402
from predicatelab.domains.tsil.generator import build_symbolic_and_composite_bank  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="artifacts/probe_bank_report.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_id = "run_probe_bank_check"
    eng = PredicateLabEngine(cfg, db_path=":memory:", run_id=run_id)
    rows: list[dict] = []
    try:
        ctx = eng._build_context()
        adapter = ctx.adapter
        fold = eng._fold(cfg["search"]["screen_folds"][0], "search")
        t0 = time.time()
        batch_path = adapter._ensure_probe_batch(ctx.snapshot, fold)
        print(f"[probe-batch] {batch_path} (built in {time.time() - t0:.1f}s)")

        cases: list[tuple[str, str, dict]] = []
        banks = build_symbolic_and_composite_bank(
            seq_len=adapter.seq_len, feature_dim=adapter.feature_dim
        )
        for group, bank in banks.items():
            for row in bank:
                cases.append((group, row["name"], row["descriptor"]))
        for fam in dsl.OP_PARAMS:
            for w in (5, 10, 15):
                cases.append(("atomic", f"{fam} window={w}", {"family": fam, "params": {"window": w}}))

        for group, name, desc in cases:
            try:
                kind = (
                    "composite" if desc.get("family") == "composite"
                    else "native" if desc.get("family") == "native"
                    else "statistical"
                )
                adapter.validate_descriptor(kind, desc)
            except Exception as exc:
                rows.append({"group": group, "name": name, "ok": False,
                             "error": f"static validation: {type(exc).__name__}: {exc}"})
                print(f"  FAIL(static) {name[:64]:64s} {exc}")
                continue
            workdir = str(Path(cfg["run"].get("artifacts_dir", "artifacts")) / "probe_check" / f"c{len(rows):03d}")
            res = adapter.probe(desc, fold, ctx.snapshot, cfg, workdir)
            checks = res.get("checks", {}) or {}
            rec = {
                "group": group, "name": name, "ok": bool(res.get("ok")),
                "effective_rank": checks.get("effective_rank"),
                "ones_kernel_alignment": checks.get("ones_kernel_alignment"),
                "offdiagonal_std": checks.get("offdiagonal_std"),
                "trace": checks.get("trace"),
                "lambda_min": checks.get("lambda_min"),
                "permutation_equivariant": checks.get("permutation_equivariant"),
                "reason": res.get("reason", "")[:300],
            }
            rows.append(rec)
            flag = "OK  " if rec["ok"] else "FAIL"
            print(f"  {flag} {group:14s} {name[:60]:60s} "
                  f"r_eff={rec['effective_rank']} ones={rec['ones_kernel_alignment']} "
                  f"{(rec['reason'] or '')[:70]}")
    finally:
        eng.store.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n[summary] {n_ok}/{len(rows)} descriptors passed the real CSI300 probe gate -> {out}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
