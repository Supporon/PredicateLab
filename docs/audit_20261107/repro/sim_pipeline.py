"""Whole-pipeline calibration of the real PredicateLab engine with ground truth.

Pre-registered targets (fixed before running; see report section 5):
  * statistical unit for the generalisation claim: (fold/year) x (training seed);
  * target error rate: P(a globally-null run yields >=1 `independently_confirmed`
    at the confirmation stage) <= 0.10 (= configured fdr_alpha; under the global
    null FDR = FWER);  same target for the final stage;
  * minimum meaningful effect = config min_effect = 0.002 IC; positive control at
    0.004 IC (2x MME) for the `volatility` DSL family;
  * verdict per scenario: PASS if the Wilson 95% upper bound <= 0.10, FAIL if
    the Wilson lower bound > 0.10, otherwise INCONCLUSIVE.

Each replicate is one complete run with its own SQLite file:
  2 discovery rounds (real RuleGenerator + feedback mutations, real screen /
  BH / classify) -> freeze all validated -> confirm on f2023 -> finalize the
  best confirmed candidate on the synthetic holdout.

Usage: python -I sim_pipeline.py <src_copy> <calendar> <workdir> <out.jsonl> <scenario> <n_reps> <n_procs>
"""
import json
import multiprocessing as mp
import sys
import tempfile
import traceback
from pathlib import Path

SRC, CAL, WORK, OUT, SCEN, NREP, NPROC = sys.argv[1:8]
NREP, NPROC = int(NREP), int(NPROC)
START_REP = int(sys.argv[8]) if len(sys.argv) > 8 else 0
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(SRC).resolve()))
sys.path.insert(0, str(HERE))

SCENARIOS = {
    # name: (sd_seed_fold, sd_family_year, true effects)
    "N0_iid": (0.0, 0.0, {}),
    "N1_seed1.5e-3": (0.0015, 0.0, {}),
    "N2_seed3e-3": (0.003, 0.0, {}),
    "N3_seed1.5e-3_famyr2e-3": (0.0015, 0.002, {}),
    "P1_vol+4e-3_seed1.5e-3_famyr2e-3": (0.0015, 0.002, {"volatility": 0.004}),
    "P2_vol+1e-2_seed1.5e-3_famyr2e-3": (0.0015, 0.002, {"volatility": 0.01}),
}


def one(rep: int) -> dict:
    sys.path.insert(0, str(Path(SRC).resolve()))
    sys.path.insert(0, str(HERE))
    import contextlib
    import io

    import synth_adapter as sa
    sa.install()
    from predicatelab.core.engine import PredicateLabEngine
    sd_e, sd_u, eff = SCENARIOS[SCEN]
    tmp = Path(tempfile.mkdtemp(prefix=f"{SCEN}_{rep}_", dir=WORK))
    cfg = sa.base_config(tmp, CAL, replicate_seed=rep, sd_daily=0.04,
                         sd_seed_fold=sd_e, sd_family_year=sd_u, true_family_effects=eff)
    row = {"scenario": SCEN, "rep": rep}
    eng = PredicateLabEngine(cfg, db_path=cfg["run"]["db"], run_id=f"run_sim_{rep}")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            eng.run(rounds=2)
            disc = eng.store.list_evidence(run_id=eng.run_id, stage="discovery")
            rounds = {r["round_id"]: r["round_index"] for r in eng.store.list_rounds(eng.run_id)}
            val = [e for e in disc if e["status"] == "validated"]
            fam_of = {}
            for e in val:
                c = eng.store.get_candidate(e["candidate_id"])
                fam_of[e["candidate_id"]] = sa.family_of(c["descriptor"])
            row["n_validated"] = len(val)
            row["n_validated_by_round"] = {str(i): sum(1 for e in val if rounds.get(e["round_id"]) == i)
                                           for i in (0, 1)}
            row["validated_true_family"] = sum(1 for c, f in fam_of.items() if f in eff)
            row["validated_families"] = sorted(set(fam_of.values()))
            if val:
                eng.freeze_family()
                conf = eng.confirm()
                states = [r["state"] for r in conf["results"]]
                row["n_frozen"] = len(states)
                row["n_conf_confirmed"] = states.count("independently_confirmed")
                confirmed = [r for r in conf["results"] if r["state"] == "independently_confirmed"]
                row["conf_confirmed_true_family"] = sum(1 for r in confirmed if fam_of.get(r["candidate_id"]) in eff)
                if confirmed:
                    best = max(confirmed, key=lambda r: r["effect"])
                    fin = eng.finalize(best["candidate_id"])
                    row["final_state"] = fin["result"]["state"]
                    row["final_true_family"] = fam_of.get(best["candidate_id"]) in eff
                    row["final_effect"] = fin["result"]["effect"]
            row.setdefault("n_conf_confirmed", 0)
            row.setdefault("final_state", None)
    except Exception:
        row["error"] = traceback.format_exc()[-1500:]
    finally:
        eng.store.close()
    return row


if __name__ == "__main__":
    Path(WORK).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(NPROC) as pool, open(OUT, "a") as fh:
        for r in pool.imap_unordered(one, range(START_REP, START_REP + NREP)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
