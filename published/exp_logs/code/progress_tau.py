#!/usr/bin/env python
"""Progress table for the LUSI kernel-weight (tau) sweep.

Usage (from the repo root):
    python artifacts/exp_logs/progress_tau.py
    watch -n 60 python artifacts/exp_logs/progress_tau.py
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
    ("k005", 0.95, "0.05"),
    ("k010", 0.90, "0.10"),
    ("k020", 0.80, "0.20"),
    ("k050", 0.50, "0.50 (anchor)"),
    ("learned", None, "learnable"),
]


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:+.{nd}f}"


def snapshot(slug: str) -> dict:
    db = REPO / "artifacts" / f"tsil_tau_{slug}" / "predicatelab.sqlite"
    out: dict = {"ok": False}
    if not db.exists():
        return out
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
    except Exception:
        return out
    try:
        row = con.execute(
            "select run_id, status from runs order by started_at desc limit 1"
        ).fetchone()
        if not row:
            return out
        rid, status = row
        out.update(ok=True, run_id=rid, run_status=status)
        out["rounds"] = con.execute(
            "select round_index, status, n_validated from rounds where run_id=?"
            " order by round_index", (rid,),
        ).fetchall()
        out["trials"] = dict(
            con.execute(
                "select phase || ':' || status, count(*) from trials where run_id=?"
                " group by phase, status", (rid,),
            ).fetchall()
        )
        best = None
        for (m,) in con.execute(
            "select metrics from trials where run_id=? and phase='screen' and status='success'", (rid,)
        ):
            try:
                ic = (json.loads(m) or {}).get("ic")
            except Exception:
                continue
            if isinstance(ic, (int, float)):
                best = ic if best is None else max(best, ic)
        out["best_screen_ic"] = best
        effects = []
        for (summary,) in con.execute(
            "select summary from rounds where run_id=?", (rid,)
        ):
            try:
                d = json.loads(summary or "{}")
            except Exception:
                continue
            for v in d.get("validation", []) or []:
                if isinstance(v.get("effect"), (int, float)):
                    effects.append((v["effect"], v.get("state"), v.get("ci")))
        out["effects"] = effects
    except Exception:
        pass
    finally:
        con.close()
    return out


def main() -> int:
    print(f"TSIL tau sweep — {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    hdr = (f"{'arm':<9}{'tau_hat':>8}{'kernel':>18}   {'run':<22}{'round':<10}"
           f"{'screen':<12}{'full':<10}{'bestIC':>9}{'mean_effect':>13}{'validated':>11}")
    print(hdr)
    print("-" * len(hdr))
    for slug, tau_hat, kw in ARMS:
        s = snapshot(slug)
        th = "-" if tau_hat is None else f"{tau_hat:.2f}"
        if not s.get("ok"):
            print(f"{slug:<9}{th:>8}{kw:>18}   not started")
            continue
        rnd = s["rounds"][-1] if s["rounds"] else None
        rnd_s = f"{rnd[0]}:{rnd[1]}" if rnd else "-"
        scr = f"{s['trials'].get('screen:success', 0)}/{s['trials'].get('screen:failed', 0)}"
        full = f"{s['trials'].get('full:success', 0)}/{s['trials'].get('full:failed', 0)}"
        effs = s["effects"]
        mean_eff = sum(e for e, _, _ in effs) / len(effs) if effs else None
        nval = sum(1 for e in effs if e[1] == "validated")
        print(f"{slug:<9}{th:>8}{kw:>18}   {s['run_id']:<22}{rnd_s:<10}"
              f"{scr:<12}{full:<10}{_fmt(s['best_screen_ic']):>9}{_fmt(mean_eff):>13}{nval:>11}")
    print()
    print("screen/full = success/failed trials in the DB; mean_effect = mean paired effect of the")
    print("promoted candidates (candidate - baseline:mse); validated = candidates passing every gate.")
    logs = sorted(glob.glob(str(REPO / "artifacts" / "exp_logs" / "tau_*.log")))
    print("logs: " + "  ".join(pathlib.Path(p).name for p in logs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
