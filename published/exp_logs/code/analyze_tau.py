#!/usr/bin/env python
"""H1/H0 analysis for the pre-registered LUSI kernel-weight (tau) sweep.

Reads the five arm databases, verifies at runtime which tau each arm actually
used, and prints the evidence for/against H1 as fixed in
docs/preregistration_tau_sweep_20261108.md.

Usage (repo root):  python artifacts/exp_logs/analyze_tau.py
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ARMS = [
    ("k005", 0.95, 0.05),
    ("k010", 0.90, 0.10),
    ("k020", 0.80, 0.20),
    ("k050", 0.50, 0.50),
    ("learned", None, None),
]
TOL_FLAT = 0.001  # pre-registered flatness threshold


def _f(v, nd=4, sign=True):
    if v is None:
        return "-"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def load(slug: str) -> dict:
    db = REPO / "artifacts" / f"tsil_tau_{slug}" / "predicatelab.sqlite"
    out = {"ok": False}
    if not db.exists():
        return out
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    try:
        row = con.execute(
            "select run_id, status from runs order by started_at desc limit 1"
        ).fetchone()
        if not row:
            return out
        rid, rstatus = row
        out.update(ok=True, run_id=rid, run_status=rstatus)
        # use the last round that has a summary (rounds=1 in this sweep)
        srow = con.execute(
            "select round_index, status, summary from rounds where run_id=?"
            " order by round_index desc", (rid,),
        ).fetchall()
        summary = {}
        for idx, rstat, summ in srow:
            if summ:
                summary = json.loads(summ)
                out["round_index"] = idx
                out["round_status"] = rstat
                break
        out["summary"] = summary

        # runtime tau verification from trial results
        tau_used: dict[tuple, int] = {}
        learned_traj: list[list[float]] = []
        for p in (REPO / "artifacts" / f"tsil_tau_{slug}").glob(
            "run_*/trials/*/**/result.json"
        ):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            w = d.get("worker") or {}
            if w.get("variant") != "candidate":
                continue
            fm = d.get("fit_metrics") or {}
            key = (w.get("tau_mode"), w.get("fixed_tau"))
            tau_used[key] = tau_used.get(key, 0) + 1
            th = fm.get("tau_history")
            if isinstance(th, list) and th:
                learned_traj.append([float(x) for x in th])
        out["tau_used"] = tau_used
        out["learned_traj"] = learned_traj
    finally:
        con.close()
    return out


def arm_validation(summary: dict) -> dict[str, dict]:
    rows = {}
    for v in summary.get("validation") or []:
        if v.get("candidate_id"):
            rows[v["candidate_id"]] = v
    return rows


def control_ic(summary: dict) -> dict[str, dict]:
    """control:ones full-validation IC per fold."""
    c = (summary.get("control") or {}).get("control:ones") or {}
    full = c.get("full") or {}
    return {
        fold: (v.get("metrics") or {}).get("ic")
        for fold, v in full.items()
        if isinstance(v, dict) and v.get("status") == "success"
    }


def main() -> int:
    print("=" * 100)
    print("LUSI kernel-weight (tau) sweep — pre-registered analysis")
    print("=" * 100)

    data = {slug: load(slug) for slug, _, _ in ARMS}

    # ---- 0. data availability -------------------------------------------------
    missing = [s for s, d in data.items() if not d.get("ok")]
    if missing:
        print(f"\nINCOMPLETE: no database for {missing}. Re-run when the sweep finishes.")
        return 1

    print("\n[0] runtime tau actually used by the candidate trials")
    for slug, tau_hat, kw in ARMS:
        d = data[slug]
        used = d.get("tau_used") or {}
        traj = d.get("learned_traj") or []
        desc = ", ".join(f"{m}/fixed={ft}×{n}" for (m, ft), n in used.items()) or "none"
        extra = ""
        if traj:
            mins = [x[0] for x in traj]
            maxs = [x[-1] for x in traj]
            extra = (f"  | tau_hat first-eval {st.mean(maxs):.4f} -> last-eval "
                     f"{st.mean(mins):.4f} (per-candidate means)")
        print(f"    {slug:<8} {desc}{extra}")

    # ---- 1. per-arm table -----------------------------------------------------
    print("\n[1] per-arm result (promoted candidates only)")
    hdr = (f"{'arm':<8}{'kernel':>8}{'n':>4}{'valid':>7}{'mean_eff':>10}"
           f"{'best_eff':>10}{'mean_stab':>11}{'min_q':>8}{'controlIC_f2020':>17}{'f2021':>9}")
    print(hdr)
    print("-" * len(hdr))
    per_arm = {}
    for slug, tau_hat, kw in ARMS:
        d = data[slug]
        rows = list(arm_validation(d["summary"]).values())
        effects = [r.get("effect") for r in rows if isinstance(r.get("effect"), (int, float))]
        stabs = [r.get("stability") for r in rows if isinstance(r.get("stability"), (int, float))]
        qs = [r.get("q") for r in rows if isinstance(r.get("q"), (int, float))]
        nval = sum(1 for r in rows if r.get("state") == "validated")
        per_arm[slug] = {
            "kw": kw, "n": len(rows), "nval": nval,
            "mean_eff": st.mean(effects) if effects else None,
            "best_eff": max(effects) if effects else None,
            "mean_stab": st.mean(stabs) if stabs else None,
            "min_q": min(qs) if qs else None,
            "effects": effects,
        }
        c_ic = control_ic(d["summary"])
        kw_s = "learn" if kw is None else f"{kw:.2f}"
        print(f"{slug:<8}{kw_s:>8}{len(rows):>4}{nval:>7}"
              f"{_f(per_arm[slug]['mean_eff']):>10}{_f(per_arm[slug]['best_eff']):>10}"
              f"{_f(per_arm[slug]['mean_stab'],2):>11}{_f(per_arm[slug]['min_q'],3):>8}"
              f"{_f(c_ic.get('f2020')):>17}{_f(c_ic.get('f2021')):>9}")

    # ---- 2. per-candidate effect matrix --------------------------------------
    print("\n[2] paired effect (candidate - baseline:mse) per candidate")
    all_ids = sorted({cid for d in data.values() for cid in arm_validation(d["summary"])})
    print(f"    candidates evaluated: {len(all_ids)}   (identical bank in every arm)")
    print(f"    {'candidate':<22}" + "".join(f"{s:>10}" for s, _, _ in ARMS))
    common = []
    for cid in all_ids:
        vals = []
        ok = True
        for slug, _, _ in ARMS:
            v = arm_validation(data[slug]["summary"]).get(cid)
            if not v or not isinstance(v.get("effect"), (int, float)):
                ok = False
                vals.append(None)
            else:
                vals.append(v["effect"])
        if ok:
            common.append((cid, vals))
        print(f"    {cid:<22}" + "".join(f"{_f(v):>10}" for v in vals))
    print(f"    {'ARM MEAN':<22}" + "".join(
        f"{_f(per_arm[s]['mean_eff']):>10}" for s, _, _ in ARMS))

    # ---- 3. paired cross-arm contrasts --------------------------------------
    if common:
        print("\n[3] paired contrast across arms (same candidate, same protocol)")
        print(f"    {'contrast':<26}{'mean':>10}{'median':>10}{'n_pos':>7}{'n':>4}")
        for a, b in [("k005", "k050"), ("k010", "k050"), ("k020", "k050"),
                     ("k005", "k020"), ("k005", "k010")]:
            diffs = []
            for cid, vals in common:
                diffs.append(vals[[s for s, _, _ in ARMS].index(a)] -
                             vals[[s for s, _, _ in ARMS].index(b)])
            npos = sum(1 for x in diffs if x > 0)
            print(f"    {a+' - '+b:<26}{_f(st.mean(diffs)):>10}"
                  f"{_f(st.median(diffs)):>10}{npos:>7}{len(diffs):>4}")

    # ---- 4. H1 / H0 decision ------------------------------------------------
    print("\n[4] pre-registered decision")
    fixed_order = [("k005", 0.05), ("k010", 0.10), ("k020", 0.20), ("k050", 0.50)]
    means = [per_arm[s]["mean_eff"] for s, _ in fixed_order]
    spread = (max(means) - min(means)) if all(m is not None for m in means) else None
    monotone = all(
        means[i] is not None and means[i + 1] is not None and means[i] >= means[i + 1] - 1e-9
        for i in range(len(means) - 1)
    )
    val_low = per_arm["k005"]["nval"] + per_arm["k010"]["nval"]
    val_any = sum(per_arm[s]["nval"] for s in per_arm)
    print(f"    kernel 0.05 / 0.10 / 0.20 / 0.50 arm-mean effect: "
          + " / ".join(_f(m) for m in means))
    print(f"    spread across kernel weights (0.05..0.50): {_f(spread) if spread is not None else '-'}")
    print(f"    monotone increasing as kernel weight decreases: {monotone}")
    print(f"    validated candidates at kernel <= 0.10: {val_low}; at any kernel: {val_any}")

    if val_low > 0 and monotone:
        verdict = "H1 SUPPORTED — a validated candidate exists at kernel <= 0.10 and the effect curve is monotone."
    elif val_any == 0 and spread is not None and spread < TOL_FLAT:
        verdict = "H0 / H1 REJECTED — no validated candidate and effects are flat in tau (spread < 0.001)."
    elif val_any == 0:
        verdict = ("INTERMEDIATE — no validated candidate in any arm, but the effect spread "
                   f"({_f(spread)}) exceeds the pre-registered flatness bound 0.001: report as-is, do not re-gate.")
    else:
        verdict = ("INTERMEDIATE — validated candidate(s) exist, but not in the pattern H1 predicted "
                   "(kernel<=0.10 + monotone). Report as-is, no arm is re-run or re-gated.")
    print(f"    VERDICT: {verdict}")

    # ---- 5. learned arm trajectory ------------------------------------------
    traj = data["learned"].get("learned_traj") or []
    if traj:
        print("\n[5] learned arm tau_hat trajectory (secondary prediction: collapse to 0)")
        first = st.mean(x[0] for x in traj)
        last = st.mean(x[-1] for x in traj)
        mins = min(min(x) for x in traj)
        print(f"    candidates with a trajectory: {len(traj)}; epochs logged: {len(traj[0])}")
        print(f"    mean tau_hat at first eval = {first:.4f}, at last eval = {last:.4f}, min over all = {mins:.4f}")
        print("    direction: " + ("decreasing (supports collapse)" if last < first else "NOT decreasing (prediction fails)"))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
