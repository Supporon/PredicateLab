#!/usr/bin/env python
"""Live progress table for the TSIL LLM-vs-DSL four-arm experiment.

Usage (from the repo root):
    python artifacts/exp_logs/progress.py          # one snapshot
    watch -n 60 python artifacts/exp_logs/progress.py

Read-only: opens every arm database with ``mode=ro`` so it never blocks the
training processes.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import pathlib
import sqlite3
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ARMS = [
    ("A novel_symbolic", "tsil_llm_novel_symbolic", "LLM: novel symbolic + new operators"),
    ("B dsl_composite", "tsil_llm_dsl_composite", "LLM: DSL composition only"),
    ("C hybrid", "tsil_llm_hybrid", "LLM: balanced novel+composite hybrid"),
    ("D rule_baseline", "tsil_llm_rule_baseline", "deterministic rule search (no LLM)"),
]


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:+.4f}"


def arm_snapshot(arm: str) -> dict:
    db = REPO / "artifacts" / arm / "predicatelab.sqlite"
    out: dict = {"arm": arm, "ok": False}
    if not db.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
    except Exception as exc:  # pragma: no cover - transient lock
        out["error"] = str(exc)
        return out
    try:
        row = con.execute(
            "select run_id, status, started_at from runs order by started_at desc limit 1"
        ).fetchone()
        if not row:
            return out
        run_id, run_status, started = row
        out.update(run_id=run_id, run_status=run_status, started=started, ok=True)

        rounds = con.execute(
            "select round_index, status, n_proposed from rounds where run_id=?"
            " order by round_index",
            (run_id,),
        ).fetchall()
        out["rounds"] = rounds
        by_phase: dict[str, dict[str, int]] = {}
        for phase, status, n in con.execute(
            "select phase, status, count(*) from trials where run_id=? group by phase, status",
            (run_id,),
        ):
            by_phase.setdefault(str(phase), {})[str(status)] = int(n)
        out["trials"] = by_phase

        best_ic = best_ric = None
        for (metrics,) in con.execute(
            "select metrics from trials where run_id=? and status='success'", (run_id,)
        ):
            try:
                m = json.loads(metrics) or {}
            except Exception:
                continue
            for key, bucket in (("ic", "ic"), ("rank_ic", "rank_ic")):
                val = m.get(key)
                if isinstance(val, (int, float)):
                    if bucket == "ic":
                        best_ic = val if best_ic is None else max(best_ic, val)
                    else:
                        best_ric = val if best_ric is None else max(best_ric, val)
        out["best_ic"], out["best_rank_ic"] = best_ic, best_ric

        root = REPO / "artifacts" / arm / run_id / "trials"
        on_disk = 0
        if root.exists():
            for f in root.rglob("result.json"):
                s = str(f)
                if "/baseline" not in s and "/control" not in s:
                    on_disk += 1
        out["results_on_disk"] = on_disk
    except Exception as exc:  # pragma: no cover - transient lock
        out["error"] = str(exc)
    finally:
        con.close()
    return out


def main() -> int:
    print(f"PredicateLab TSIL experiment snapshot  {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    header = (f"{'arm':<20}{'run':<24}{'round':<11}{'screen ok/fail/disk':<22}"
              f"{'full ok/fail':<15}{'best IC':<10}{'best RIC':<10}")
    print(header)
    print("-" * len(header))
    for label, arm, note in ARMS:
        s = arm_snapshot(arm)
        if not s.get("ok"):
            print(f"{label:<20}{'not started / db busy':<24}")
            continue
        rnd = s["rounds"][-1] if s["rounds"] else None
        rnd_s = f"{rnd[0]}:{rnd[1]}" if rnd else "-"
        scr = s["trials"].get("screen", {})
        full = s["trials"].get("full", {})
        scr_s = f"{scr.get('success', 0)}/{scr.get('failed', 0)}/{s['results_on_disk']}"
        full_s = f"{full.get('success', 0)}/{full.get('failed', 0)}"
        print(
            f"{label:<20}{s['run_id']:<24}{rnd_s:<11}{scr_s:<22}{full_s:<15}"
            f"{_fmt(s['best_ic']):<10}{_fmt(s['best_rank_ic']):<10}"
        )
    print()
    print("legends: round=r<index>:<status>   screen = success/failed trials recorded in the DB / candidate")
    print("         trials already finished on disk (the running batch records DB rows only when it ends)")
    print("         arms: " + " | ".join(f"{label}={arm}" for label, arm, _ in ARMS))
    logs = sorted(glob.glob(str(REPO / "artifacts" / "exp_logs" / "arm_*.log")))
    print("         logs: " + "  ".join(pathlib.Path(p).name for p in logs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
