#!/usr/bin/env python
"""Validate the small-sample scenario against a known ground truth.

The harness runs the *real* lifecycle -- ``discover -> freeze-family ->
confirm -> finalize`` -- once per replicate, over a grid of sample sizes and
latent regimes, and scores what the platform concluded against the ground
truth that generated the data.

What the sweep measures
-----------------------
For every ``(mode, n, replicate)`` cell it records:

* ``discovery_any``      -- did discovery validate at least one candidate?
* ``discovery_true``     -- did discovery validate the *true* family?
* ``discovery_false``    -- validated candidates that are not the true family
                           (for the two null modes every validation is false)
* ``confirmation_any``   -- did any frozen member confirm?
* ``confirmation_true``  -- did a candidate of the true family confirm?
* ``final_state``        -- the one-time holdout verdict for the confirmed
                           candidate (``insufficient`` is expected at small n)
* ``recovery_auc``       -- AUC of the best validated predicate column for the
                           latent truth (monotone-equivalent predicates count)

The summary reports per-cell rates with Wilson 95% intervals, so the null
cells ("should stay quiet") and the signal cells ("should be recoverable with
limited power") can be read side by side.

Isolation
---------
Every replicate is a fresh ``sqlite`` database *and* a fresh confirmation
ledger under ``<output>/runs/<cell>/.``  A sweep is a Monte Carlo experiment,
so re-running the same grid must not be blocked by the ledger that protects a
real study's one-time confirmation fold; each synthetic dataset is also
identified by its replicate seed.

Examples
--------
    python scripts/validate_small_sample.py --quick
    python scripts/validate_small_sample.py --reps 10 --n-values 32,64,128,256 \
        --modes null_linear,signal_sign_interaction --output artifacts_small_sample/sweep
    python scripts/validate_small_sample.py --workers 4 --reps 20
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from predicatelab.core.engine import PredicateLabEngine  # noqa: E402
from predicatelab.domains.small_sample.adapter import SmallSampleAdapter  # noqa: E402

BASE_CONFIG = REPO_ROOT / "configs" / "small_sample_signal.yaml"
MODES = ("null_noise", "null_linear", "signal_ratio_threshold",
         "signal_sign_interaction")
DEFAULT_STRENGTH = {
    "null_noise": 0.0,
    "null_linear": 0.0,
    "signal_ratio_threshold": 4.0,
    "signal_sign_interaction": 3.0,
}
CONFIRMED_STATES = {
    "resampling_confirmed", "independently_confirmed", "time_forward_confirmed",
}
# A validated predicate with this AUC against the latent truth is treated as a
# behavioural recovery even if its operator key differs (e.g. the continuous
# product x1*x3 for a sign(x1*x3) truth).
EQUIVALENCE_AUC = 0.90


# --------------------------------------------------------------------------- #
# config plumbing
# --------------------------------------------------------------------------- #
def load_base_config() -> dict[str, Any]:
    with BASE_CONFIG.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit(f"base config is not a mapping: {BASE_CONFIG}")
    return cfg


def build_config(base: dict[str, Any], *, mode: str, n_samples: int, rep: int,
                 workdir: Path, rounds: int, strength: float | None,
                 seeds: list[int], cv_splits: int, save_predictions: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    scenario_seed = int(base.get("data", {}).get("seed", 20261108)) + 1009 * rep
    cfg.setdefault("run", {})
    cfg["run"].update({
        "name": f"sweep_{mode}_n{n_samples}_r{rep}",
        "artifacts_dir": str(workdir / "artifacts"),
        "db": str(workdir / "sweep.sqlite"),
        # Each replicate is an independent synthetic study.  The repository
        # ledger exists to stop a *real* confirmation fold being consumed
        # twice; a Monte Carlo sweep deliberately re-runs the same grid, so it
        # gets an isolated ledger per replicate (also removes lock contention).
        "global_ledger_db": str(workdir / "ledger.sqlite"),
        "rounds": rounds,
        "require_confirmation": True,
        "warning_acknowledgements": "all",
        "warning_ack_reason": ("Pre-registered synthetic small-sample sweep; low power "
                               "is the phenomenon under study, not a data defect."),
    })
    cfg["seeds"] = list(seeds)
    cfg.setdefault("data", {})
    cfg["data"].update({
        "generator": mode,
        "seed": scenario_seed,
        "n_samples": int(n_samples),
        "n_features": 8,
        "signal_strength": float(DEFAULT_STRENGTH[mode] if strength is None else strength),
        "threshold_quantile": 0.75,
        "class_balance": 0.5,
        "holdout_fraction": 0.25,
        "discovery_fraction": 0.65,
        "confirmation_mode": "independent",
    })
    cfg.setdefault("training", {})
    cfg["training"].update({"seed": int(seeds[0]), "C": 1.0, "max_iter": 5000})
    cfg.setdefault("protocol", {})
    cfg["protocol"].update({"cv_splits": int(cv_splits), "cv_repeats": 1,
                            "discovery_repeats": 1, "confirmation_splits": 3})
    cfg.setdefault("search", {})
    cfg["search"].update({"save_predictions": bool(save_predictions),
                          "max_candidates_per_round": 24})
    cfg.setdefault("execution", {})
    cfg["execution"]["workers"] = 1
    return cfg


# --------------------------------------------------------------------------- #
# one replicate
# --------------------------------------------------------------------------- #
def _latest_evidence(store, run_id: str, stage: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for ev in store.list_evidence(run_id=run_id, stage=stage):
        latest[ev["candidate_id"]] = ev
    return latest


def _describe_candidate(store, candidate_id: str) -> dict[str, Any]:
    cand = store.get_candidate(candidate_id) or {}
    return {"candidate_id": candidate_id,
            "name": cand.get("name"),
            "descriptor": cand.get("descriptor")}


def run_replicate(job: dict[str, Any]) -> dict[str, Any]:
    """Run one full lifecycle and return a flat, JSON-safe record."""
    t0 = time.time()
    mode = job["mode"]
    n_samples = int(job["n_samples"])
    rep = int(job["rep"])
    tmp_root = Path(job["workdir"])
    strength = job["config"]["data"].get("signal_strength")
    record: dict[str, Any] = {
        "mode": mode, "n_samples": n_samples, "rep": rep,
        "strength": float(strength) if mode.startswith("signal_") else None,
        "scenario_seed": job["config"]["data"]["seed"],
        "is_signal": mode.startswith("signal_"),
        "status": "error", "error": "",
        "suitability_status": None,
        "n_discovery": None, "n_confirmation": None, "n_holdout": None,
        "discovery_any": False, "discovery_true": False, "discovery_false": False,
        "discovery_recovered": False, "discovery_spurious": False,
        "n_validated": 0,
        "confirmation_any": False, "confirmation_true": False,
        "confirmation_recovered": False,
        "confirmation_states": {},
        "final_state": None, "final_effect": None,
        "final_ci_low": None, "final_ci_high": None,
        "recovery_auc": None, "recovery_family": None,
        "true_family": None, "validated": [],
        "duration_s": None,
    }
    engine = None
    try:
        cfg = job["config"]
        # Every replicate starts from a clean state.  Re-using a sqlite file
        # would let the confirmation ledger remember an earlier sweep and
        # silently downgrade this replicate to exploratory_positive.
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        engine = PredicateLabEngine(cfg, db_path=cfg["run"]["db"], run_id=None)
        adapter = SmallSampleAdapter(cfg)
        record["true_family"] = adapter.true_family()
        record["n_discovery"] = int(len(adapter._discovery_idx))
        record["n_confirmation"] = int(len(adapter._confirmation_idx))
        record["n_holdout"] = int(len(adapter._holdout_idx))
        record["suitability_status"] = None

        run_id = engine.run(rounds=int(job["rounds"]))
        try:
            ctx = engine._build_context()
            suitability = (ctx.task.metadata or {}).get("suitability") or {}
            record["suitability_status"] = suitability.get("status")
        except Exception:
            record["suitability_status"] = None
        latest = _latest_evidence(engine.store, run_id, "discovery")
        validated = [cid for cid, ev in latest.items()
                     if ev.get("status") == "validated"]
        record["n_validated"] = len(validated)
        record["discovery_any"] = bool(validated)
        members = []
        best_auc = None
        best_family = None
        for cid in validated:
            info = _describe_candidate(engine.store, cid)
            desc = info["descriptor"] or {}
            family = adapter.family_of(desc)
            is_true = adapter.matches_truth(desc)
            auc = adapter.recovery_auc(desc)
            equivalent = bool(is_true) or (auc is not None and auc >= EQUIVALENCE_AUC)
            info.update({"family": family, "is_true": bool(is_true),
                         "equivalent": equivalent, "recovery_auc": auc,
                         "effect": latest[cid].get("effect"),
                         "state": latest[cid].get("state")})
            members.append(info)
            if auc is not None and (best_auc is None or auc > best_auc):
                best_auc = float(auc)
                best_family = family
        record["validated"] = members
        record["discovery_true"] = any(m["is_true"] for m in members)
        record["discovery_false"] = any(not m["is_true"] for m in members)
        record["discovery_recovered"] = any(m["equivalent"] for m in members)
        record["discovery_spurious"] = any(not m["equivalent"] for m in members)
        record["recovery_auc"] = best_auc
        record["recovery_family"] = best_family

        # --- freeze + confirm (only meaningful if something validated)
        if validated:
            engine.freeze_family()
            engine.confirm()
            conf_latest = _latest_evidence(engine.store, run_id, "confirmation")
            states = {}
            for cid, ev in conf_latest.items():
                states[cid] = ev.get("state")
            record["confirmation_states"] = states
            confirmed = [cid for cid, st in states.items() if st in CONFIRMED_STATES]
            record["confirmation_any"] = bool(confirmed)
            record["confirmation_true"] = any(
                m["is_true"] and m["candidate_id"] in confirmed for m in members
            )
            record["confirmation_recovered"] = any(
                m["equivalent"] and m["candidate_id"] in confirmed for m in members
            )

            # --- finalize exactly the candidate that confirmed (if any)
            if confirmed:
                by_id = {m["candidate_id"]: m for m in members}
                confirmed.sort(key=lambda cid: (
                    not by_id[cid]["equivalent"],
                    -(by_id[cid]["effect"] or 0.0),
                ))
                final_cid = confirmed[0]
                res = engine.finalize(final_cid)
                payload = res.get("result", res)
                state = res.get("state") or payload.get("state")
                cand_block = payload.get("candidate", {}) if isinstance(payload, dict) else {}
                metrics = cand_block.get("metrics", {}) if isinstance(cand_block, dict) else {}
                first = next(iter(metrics.values()), {}) if isinstance(metrics, dict) else {}
                record["final_state"] = state
                record["final_effect"] = first.get("effect")
                record["final_ci_low"] = first.get("ci_low")
                record["final_ci_high"] = first.get("ci_high")
                record["final_candidate"] = final_cid
        record["status"] = "ok"
    except Exception as exc:  # blocked / protocol violation / execution failure
        kind = type(exc).__name__
        suitability_block = kind == "RuntimeError" and "suitability" in str(exc)
        record["status"] = "blocked" if kind in {
            "SuitabilityError", "ProtocolViolation", "DescriptorError"
        } or suitability_block else "error"
        record["error"] = f"{kind}: {exc}"[:800]
        # Record *why* a cell was refused, even when the engine short-circuited.
        try:
            adapter = SmallSampleAdapter(job["config"])
            report = adapter.suitability(adapter.snapshot(job["config"]), job["config"])
            record["suitability_status"] = report.get("status")
            record["blockers"] = list(report.get("blockers") or [])
        except Exception:
            pass
        if job.get("debug"):
            record["traceback"] = traceback.format_exc()[-4000:]
    finally:
        if engine is not None:
            try:
                engine.store.close()
            except Exception:
                pass
        if not job.get("keep_artifacts"):
            shutil.rmtree(tmp_root, ignore_errors=True)
        record["duration_s"] = round(time.time() - t0, 3)
    return record


# --------------------------------------------------------------------------- #
# summary statistics
# --------------------------------------------------------------------------- #
def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _rate(records: list[dict], key: str, ok_status: str = "ok") -> dict[str, Any]:
    usable = [r for r in records if r["status"] == ok_status]
    k = sum(1 for r in usable if r.get(key))
    lo, hi = wilson(k, len(usable))
    return {"k": k, "n": len(usable), "rate": (k / len(usable)) if usable else None,
            "ci_low": lo, "ci_high": hi}


def summarise(records: list[dict]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for r in records:
        strength = r.get("strength")
        key = (f"{r['mode']}|s={strength}|n={r['n_samples']}"
               if r["mode"].startswith("signal_") else f"{r['mode']}|n={r['n_samples']}")
        cells.setdefault(key, {"mode": r["mode"], "n_samples": r["n_samples"],
                               "strength": strength, "records": []})["records"].append(r)

    out: dict[str, Any] = {"cells": {}, "totals": {
        "replicates": len(records),
        "status": {},
    }}
    def _sort_key(kv):
        cell = kv[1]
        strength = cell["strength"] if cell["strength"] is not None else 0.0
        return (cell["mode"], float(strength), cell["n_samples"])

    for key, cell in sorted(cells.items(), key=_sort_key):
        recs = cell["records"]
        status_counts: dict[str, int] = {}
        for r in recs:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        aucs = [r["recovery_auc"] for r in recs if r["recovery_auc"] is not None]
        finals = [r["final_state"] for r in recs if r["final_state"]]
        out["cells"][key] = {
            "mode": cell["mode"],
            "n_samples": cell["n_samples"],
            "strength": cell["strength"],
            "n_replicates": len(recs),
            "status_counts": status_counts,
            "discovery_any": _rate(recs, "discovery_any"),
            "discovery_true": _rate(recs, "discovery_true"),
            "discovery_recovered": _rate(recs, "discovery_recovered"),
            "discovery_spurious": _rate(recs, "discovery_spurious"),
            "discovery_false": _rate(recs, "discovery_false"),
            "confirmation_any": _rate(recs, "confirmation_any"),
            "confirmation_true": _rate(recs, "confirmation_true"),
            "confirmation_recovered": _rate(recs, "confirmation_recovered"),
            "mean_validated": (
                sum(r["n_validated"] for r in recs if r["status"] == "ok")
                / max(1, sum(1 for r in recs if r["status"] == "ok"))
            ),
            "mean_recovery_auc": (sum(aucs) / len(aucs)) if aucs else None,
            "final_states": {s: finals.count(s) for s in sorted(set(finals))},
        }
        for r in recs:
            out["totals"]["status"][r["status"]] = \
                out["totals"]["status"].get(r["status"], 0) + 1
    return out


def render_markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = ["# Small-sample validation sweep", ""]
    lines.append(f"- base config: `{meta['base_config']}`")
    lines.append(f"- rounds: {meta['rounds']}, seeds: {meta['seeds']}, "
                 f"replicates per cell: {meta['reps']}, workers: {meta['workers']}")
    lines.append(f"- started: {meta['started']}, finished: {meta['finished']}")
    lines.append("")
    lines += ["## Per-cell results", "",
              "Rates are over replicates that completed without a blocking error; "
              "Wilson 95% intervals in parentheses.", "",
              "| mode | strength | n | reps | discovery any | discovered true | "
              "recovered (true or AUC≥0.90) | spurious | confirmation any | "
              "confirmed true | confirmed recovered | mean AUC | final states |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in summary["cells"].values():
        def fmt(rate):
            if rate["rate"] is None:
                return "-"
            return f"{rate['rate']:.2f} ({rate['ci_low']:.2f},{rate['ci_high']:.2f})"
        auc = cell["mean_recovery_auc"]
        finals = ", ".join(f"{k}:{v}" for k, v in
                           sorted(cell["final_states"].items())) or "-"
        strength = cell.get("strength")
        lines.append(
            f"| {cell['mode']} | {'-' if strength is None else f'{float(strength):g}'} | "
            f"{cell['n_samples']} | {cell['n_replicates']} | "
            f"{fmt(cell['discovery_any'])} | {fmt(cell['discovery_true'])} | "
            f"{fmt(cell['discovery_recovered'])} | {fmt(cell['discovery_spurious'])} | "
            f"{fmt(cell['confirmation_any'])} | {fmt(cell['confirmation_true'])} | "
            f"{fmt(cell['confirmation_recovered'])} | "
            f"{'-' if auc is None else f'{auc:.3f}'} | {finals} |"
        )
    lines += ["", "## Reading this table", "",
              "- `null_noise` / `null_linear` are global nulls: **every** discovery "
              "is a false discovery, and the whole column should be near zero.",
              "- `discovered true` requires the exact operator family of the planted "
              "truth; `recovered` also credits a monotone-equivalent predicate "
              f"(AUC ≥ {EQUIVALENCE_AUC:.2f} against the latent indicator, e.g. the "
              "continuous product for a sign-interaction truth).",
              "- `spurious` counts validated predicates that are neither: pure "
              "small-sample false discoveries at the discovery gate.",
              "- `confirmed *` is the pre-registration gate; `final states` shows the "
              "one-time holdout verdict and is expected to degrade faster than "
              "confirmation as n shrinks.",
              ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-values", default="32,64,128,256",
                   help="comma-separated sample sizes")
    p.add_argument("--modes", default=",".join(MODES),
                   help="comma-separated modes (default: all four)")
    p.add_argument("--reps", type=int, default=6,
                   help="replicates per (mode, n) cell (default: 6)")
    p.add_argument("--rounds", type=int, default=2,
                   help="discovery rounds per replicate (default: 2)")
    p.add_argument("--strength", type=float, default=None,
                   help="override signal strength for all signal modes")
    p.add_argument("--strengths", default="",
                   help="comma-separated signal strengths; expands signal-mode "
                        "cells (null modes are unaffected)")
    p.add_argument("--seeds", default="2025",
                   help="comma-separated engine seeds (default: 2025)")
    p.add_argument("--cv-splits", type=int, default=4)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", default="artifacts_small_sample/sweep",
                   help="output directory for JSONL + summaries")
    p.add_argument("--save-predictions", action="store_true",
                   help="keep per-trial prediction files (slow, disk heavy)")
    p.add_argument("--keep-artifacts", action="store_true",
                   help="keep each replicate's sqlite + artifacts under output/runs")
    p.add_argument("--debug", action="store_true",
                   help="record tracebacks for failed replicates")
    p.add_argument("--quick", action="store_true",
                   help="preset: n=32,64,128, all modes, 2 reps, 1 round")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.quick:
        args.n_values = "32,64,128"
        args.reps = 2
        args.rounds = 1
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown mode(s): {unknown}; expected {MODES}")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    strength_grid = [float(x) for x in args.strengths.split(",") if x.strip()]
    if args.strength is not None and strength_grid:
        raise SystemExit("use either --strength or --strengths, not both")

    base = load_base_config()
    out_dir = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for mode in modes:
        if mode.startswith("signal_"):
            mode_strengths = (strength_grid if strength_grid
                              else [args.strength if args.strength is not None
                                    else DEFAULT_STRENGTH[mode]])
        else:
            mode_strengths = [0.0]
        for strength in mode_strengths:
            tag = f"_s{float(strength):g}" if mode.startswith("signal_") else ""
            for n_samples in n_values:
                for rep in range(args.reps):
                    workdir = runs_dir / f"{mode}{tag}_n{n_samples}_r{rep}"
                    workdir.mkdir(parents=True, exist_ok=True)
                    cfg = build_config(
                        base, mode=mode, n_samples=n_samples, rep=rep,
                        workdir=workdir, rounds=args.rounds, strength=strength,
                        seeds=seeds, cv_splits=args.cv_splits,
                        save_predictions=args.save_predictions,
                    )
                    jobs.append({
                        "mode": mode, "n_samples": n_samples, "rep": rep,
                        "strength": float(strength) if mode.startswith("signal_") else None,
                        "config": cfg, "rounds": args.rounds,
                        "workdir": str(workdir),
                        "keep_artifacts": args.keep_artifacts,
                        "debug": args.debug,
                    })

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[sweep] {len(jobs)} replicates over {len(modes)} modes x "
          f"{len(n_values)} sample sizes", flush=True)
    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        for k, job in enumerate(jobs, 1):
            rec = run_replicate(job)
            records.append(rec)
            print(f"[{k}/{len(jobs)}] {rec['mode']} n={rec['n_samples']} r={rec['rep']}: "
                  f"{rec['status']} validated={rec['n_validated']} "
                  f"conf_true={rec['confirmation_true']} final={rec['final_state']} "
                  f"({rec['duration_s']}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_replicate, job): job for job in jobs}
            for k, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                records.append(rec)
                print(f"[{k}/{len(jobs)}] {rec['mode']} n={rec['n_samples']} r={rec['rep']}: "
                      f"{rec['status']} validated={rec['n_validated']} "
                      f"conf_true={rec['confirmation_true']} final={rec['final_state']} "
                      f"({rec['duration_s']}s)", flush=True)
    records.sort(key=lambda r: (r["mode"], r["n_samples"], r["rep"]))

    jsonl = out_dir / "records.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    summary = summarise(records)
    finished = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {"base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
            "rounds": args.rounds, "seeds": seeds, "reps": args.reps,
            "workers": args.workers, "started": started, "finished": finished,
            "n_values": n_values, "modes": modes,
            "strength_override": args.strength,
            "strengths": strength_grid or None}
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2, default=str),
        encoding="utf-8")
    (out_dir / "summary.md").write_text(
        render_markdown(summary, meta), encoding="utf-8")
    print(f"[sweep] wrote {jsonl}, {out_dir / 'summary.json'}, "
          f"{out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
