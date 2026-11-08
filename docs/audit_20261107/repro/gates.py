"""Gate / ledger / identity reproductions against the real engine.

R03  holdout ledger is per-SQLite-file: a new --db allows a second
     "confirmatory" final on the same data period.
R04  confirmation folds have no exposure ledger: the same confirmation period
     can be used by any number of runs, each able to produce
     `independently_confirmed`; nothing records prior exposure.
R08  code identity without git is a constant: editing source does not change
     the trial cache key; TSIL code at data.tsil_root is never fingerprinted.
R09  n_dates in aggregated evidence is summed over seeds.

Usage: python -I gates.py <src_copy> <calendar_path> <workdir>
"""
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
cal = sys.argv[2]
work = Path(sys.argv[3]).resolve()
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(src))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth_adapter as sa  # noqa: E402
sa.install()
from predicatelab.core.engine import PredicateLabEngine  # noqa: E402
from predicatelab.core.holdout import HoldoutConsumed  # noqa: E402
from predicatelab.core.identity import code_version  # noqa: E402

# a scenario with a real effect so candidates reach confirmation/final
SYN = dict(replicate_seed=7, sd_daily=0.03, sd_family_year=0.0, sd_seed_fold=0.0,
           true_family_effects={"volatility": 0.01, "native:phi_volatility": 0.01})
out = {}


def lifecycle(tag, db, run_id=None, do_final=True):
    cfg = sa.base_config(work / tag, cal, **SYN)
    cfg["run"]["db"] = str(db)
    cfg["run"]["rounds"] = 1
    eng = PredicateLabEngine(cfg, db_path=str(db), run_id=run_id)
    try:
        rid = eng.run(rounds=1)
        fam = eng.freeze_family(top_k=1)
        conf = eng.confirm()
        res = {"run_id": rid, "frozen": [m["candidate_id"] for m in fam["members"]],
               "confirmation_states": [r["state"] for r in conf["results"]]}
        if do_final:
            cid = fam["members"][0]["candidate_id"]
            try:
                fin = eng.finalize(cid)
                res["final"] = {"status": fin.get("status"), "exploratory": fin.get("exploratory"),
                                "state": (fin.get("result") or {}).get("state"),
                                "purpose": fin["manifest"]["purpose"]}
            except HoldoutConsumed as exc:
                res["final"] = {"blocked": f"HoldoutConsumed: {str(exc)[:160]}"}
            except Exception as exc:
                res["final"] = {"blocked": f"{type(exc).__name__}: {str(exc)[:160]}"}
        res["holdout_ledger_rows"] = [
            {k: r[k] for k in ("run_id", "status", "purpose", "test_start", "test_end")}
            for r in eng.store.list_holdout_access()]
        return res
    finally:
        eng.store.close()


import time  # noqa: E402

# ---- R03 / R04 -----------------------------------------------------------
db1 = work / "db_A.sqlite"
out["R03_R04_run1_dbA"] = lifecycle("A1", db1)
time.sleep(1.1)  # run ids are second-resolution timestamps
out["R04_run2_same_dbA"] = lifecycle("A2", db1)
time.sleep(1.1)
db2 = work / "db_B.sqlite"
out["R03_run3_fresh_dbB"] = lifecycle("B1", db2)
out["R03_R04_expected"] = {
    "run2_same_db_final": "blocked or exploratory (holdout consumed) -- engine does this",
    "run2_confirmation": "should be refused or downgraded: f2023 already used by run1 (no ledger)",
    "run3_fresh_db_final": "should be refused: same dataset/period already consumed in db_A",
}

# ---- R08 code identity without git -----------------------------------------
tree = work / "code_tree"
if not tree.exists():
    shutil.copytree(src / "predicatelab", tree / "predicatelab")
before = code_version(tree)
f = tree / "predicatelab" / "core" / "stats.py"
f.write_text(f.read_text() + "\n# audit edit: changes numerics in a real fix\n")
after = code_version(tree)
out["R08"] = {"git_available_in_tree": (tree / ".git").exists(),
              "fingerprint_before_edit": before["dirty_fingerprint"],
              "fingerprint_after_edit": after["dirty_fingerprint"],
              "unchanged_after_source_edit": before["dirty_fingerprint"] == after["dirty_fingerprint"],
              "commit": before["commit"]}

# ---- R09 n_dates summed over seeds ----------------------------------------
import sqlite3  # noqa: E402
con = sqlite3.connect(f"file:{db1}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute("SELECT metrics FROM evidence WHERE stage='discovery' AND status IN "
                   "('validated','insufficient_evidence','rejected') LIMIT 1").fetchall()
if rows:
    m = json.loads(rows[0]["metrics"])
    per_seed = {k: v.get("n_dates") for k, v in (m.get("per_seed") or {}).items()}
    out["R09"] = {"evidence_n_dates": m.get("n_dates"), "per_seed_n_dates": per_seed,
                  "n_validation_dates_field": m.get("n_validation_dates")}
con.close()
print(json.dumps(out, indent=2, default=str))
