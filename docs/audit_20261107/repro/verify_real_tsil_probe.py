"""Verify all 23 upstream predicates, 6 DSL families, and prediction files on real CSI300 data."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/zhengzengrong/PredicateLab/Predicate_Lab_v1.1")

import numpy as np
import yaml

from predicatelab.core.engine import PredicateLabEngine
from predicatelab.domains.tsil import dsl


def main() -> None:
    cfg_path = "/home/zhengzengrong/PredicateLab/audit_20261107_credibility/tmpdata/tsil_smoke_verify/tsil_smoke_verify.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    eng = PredicateLabEngine(cfg, db_path=cfg["run"]["db"], run_id="run_tsil_gpu_verify")
    ctx = eng._build_context()
    adapter = ctx.adapter
    snap = ctx.snapshot
    f2022 = [f for f in ctx.effective_folds if f.fold_id == "f2022"][0]
    probe_path = adapter._probe_batch_path(snap, f2022)
    pb = np.load(probe_path)
    pdir = str(Path(cfg["run"]["artifacts_dir"]) / "probe_verify")

    native_results = {}
    for fn_name in dsl.NATIVE_PREDICATES:
        desc = {"family": "native", "predicate": fn_name, "params": {}}
        res = adapter.probe(desc, f2022, snap, cfg, pdir)
        native_results[fn_name] = {
            "ok": res["ok"],
            "reason": res.get("reason"),
            "perm_equiv": res.get("checks", {}).get("permutation_equivariant"),
            "perm_diff": res.get("checks", {}).get("permutation_max_abs_diff"),
            "batch_coupling": res.get("checks", {}).get("batch_coupling"),
            "trace": res.get("checks", {}).get("trace"),
        }

    dsl_results = {}
    for fam in dsl.OP_PARAMS:
        desc = {"family": fam, "params": {"window": 10}, "feature_slice": [0, 32]}
        res = adapter.probe(desc, f2022, snap, cfg, pdir)
        dsl_results[fam] = {
            "ok": res["ok"],
            "reason": res.get("reason"),
            "perm_equiv": res.get("checks", {}).get("permutation_equivariant"),
            "perm_diff": res.get("checks", {}).get("permutation_max_abs_diff"),
            "batch_coupling": res.get("checks", {}).get("batch_coupling"),
        }

    art_root = Path(cfg["run"]["artifacts_dir"])
    pred_files = sorted(glob.glob(str(art_root / "**" / "predictions.npz"), recursive=True))
    pred_summary = []
    for pf in pred_files:
        arr = np.load(pf, allow_pickle=True)
        n_rows = int(len(arr["score"]))
        pred_summary.append({
            "rel": str(Path(pf).relative_to(art_root)),
            "n_rows": n_rows,
            "mod_800": n_rows % 800,
            "n_unique_dates": int(len(set(arr["date"]))),
            "n_unique_instruments": int(len(set(arr["instrument"]))),
            "date_min": str(min(arr["date"])),
            "date_max": str(max(arr["date"])),
        })

    eng.store.close()
    print(json.dumps({
        "probe_batch_path": str(probe_path),
        "probe_batch_x_shape": list(pb["x"].shape),
        "probe_batch_ts_shape": list(pb["timestamp"].shape),
        "n_native_total": len(native_results),
        "n_native_ok": sum(v["ok"] for v in native_results.values()),
        "n_native_perm_equiv": sum(bool(v["perm_equiv"]) for v in native_results.values()),
        "native_non_ok": {k: v for k, v in native_results.items() if not v["ok"]},
        "cycle_check": native_results["phi_cycle"],
        "n_dsl_total": len(dsl_results),
        "n_dsl_ok": sum(v["ok"] for v in dsl_results.values()),
        "dsl_results": dsl_results,
        "n_prediction_files": len(pred_summary),
        "sample_predictions": pred_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
