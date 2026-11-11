#!/usr/bin/env python
"""Evaluate the pre-registered small-sample hypotheses on a sweep's records.

Reads ``records.jsonl`` produced by ``scripts/validate_small_sample.py`` and
applies the decision rules fixed in
``docs/preregistration_small_sample_20261108.md`` §4.  Nothing here re-runs the
platform or changes any rate; it is a deterministic reduction of the records.

Usage
-----
    python scripts/analyze_small_sample_formal.py \
        --records artifacts_small_sample/formal_20261108/records.jsonl \
        --boundary-records artifacts_small_sample/formal_20261108_boundary/records.jsonl \
        --output artifacts_small_sample/formal_20261108/analysis
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def wilson(k: int, n: int, z: float = 1.96) -> dict[str, Any]:
    if n <= 0:
        return {"k": k, "n": 0, "rate": None, "ci_low": None, "ci_high": None}
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"k": k, "n": n, "rate": p,
            "ci_low": max(0.0, centre - margin),
            "ci_high": min(1.0, centre + margin)}


def load(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def usable(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("status") == "ok"]


def cell(records: Iterable[dict[str, Any]], mode: str, n: int,
         strength: float | None = None) -> list[dict[str, Any]]:
    out = []
    for r in usable(records):
        if r["mode"] != mode or int(r["n_samples"]) != int(n):
            continue
        if strength is not None and abs(float(r.get("strength") or 0.0) - float(strength)) > 1e-9:
            continue
        out.append(r)
    return out


def rate(records: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    recs = list(records)
    return wilson(sum(1 for r in recs if r.get(key)), len(recs))


def evaluate(records: list[dict[str, Any]],
             boundary: list[dict[str, Any]] | None) -> dict[str, Any]:
    null_modes = ("null_noise", "null_linear")
    null_recs = [r for r in usable(records) if r["mode"] in null_modes]
    null_cells: dict[str, dict[str, Any]] = {}
    for r in null_recs:
        key = f"{r['mode']}|n={r['n_samples']}"
        null_cells.setdefault(key, [])
        null_cells[key].append(r)
    cell_rates = {k: rate(v, "confirmation_any") for k, v in null_cells.items()}
    pooled_h1 = rate(null_recs, "confirmation_any")
    max_cell = max((v["rate"] or 0.0) for v in cell_rates.values()) if cell_rates else 0.0

    h1 = {
        "pooled": pooled_h1,
        "cells": {k: v for k, v in sorted(cell_rates.items())},
        "max_cell_rate": max_cell,
        "threshold_pooled": 0.02,
        "threshold_cell": 0.075,
        "met": bool(pooled_h1["rate"] is not None
                    and pooled_h1["rate"] <= 0.02
                    and max_cell <= 0.075),
    }

    h2_n256 = rate(cell(records, "signal_sign_interaction", 256, 3.0),
                   "confirmation_recovered")
    h2_n24 = rate(cell(records, "signal_sign_interaction", 24, 3.0),
                  "confirmation_recovered")
    h2 = {
        "n256": h2_n256, "n24": h2_n24,
        "threshold_n256_min": 0.40, "threshold_n24_max": 0.20,
        "supported": bool(h2_n256["rate"] is not None and h2_n256["rate"] >= 0.40
                          and h2_n24["rate"] is not None and h2_n24["rate"] <= 0.20),
    }

    h3_s4 = rate(cell(records, "signal_sign_interaction", 128, 4.0),
                 "confirmation_recovered")
    h3_s2 = rate(cell(records, "signal_sign_interaction", 128, 2.0),
                 "confirmation_recovered")
    diff = ((h3_s4["rate"] or 0.0) - (h3_s2["rate"] or 0.0)
            if h3_s4["rate"] is not None and h3_s2["rate"] is not None else None)
    h3 = {"s4": h3_s4, "s2": h3_s2, "difference": diff,
          "threshold_difference": 0.15,
          "supported": bool(diff is not None and diff >= 0.15)}

    small = [r for n in (24, 32, 64)
             for r in cell(records, "signal_sign_interaction", n, 3.0)]
    large = [r for n in (128, 256)
             for r in cell(records, "signal_sign_interaction", n, 3.0)]

    def holdout_ratio(recs):
        conf = [r for r in recs if r.get("confirmation_any")]
        ok = [r for r in conf if r.get("final_state") == "independently_confirmed"]
        return wilson(len(ok), len(conf))

    h4 = {"small_n": holdout_ratio(small), "large_n": holdout_ratio(large),
          "supported": bool(
              holdout_ratio(small)["rate"] is not None
              and holdout_ratio(large)["rate"] is not None
              and holdout_ratio(small)["rate"] < holdout_ratio(large)["rate"])}

    null_small = [r for r in null_recs if int(r["n_samples"]) in (24, 32, 64)]
    null_large = [r for r in null_recs if int(r["n_samples"]) in (128, 256)]
    sp_small = rate(null_small, "discovery_spurious")
    sp_large = rate(null_large, "discovery_spurious")
    gap = ((sp_small["rate"] or 0.0) - (sp_large["rate"] or 0.0)
           if sp_small["rate"] is not None and sp_large["rate"] is not None else None)
    h5 = {"small_n": sp_small, "large_n": sp_large, "difference": gap,
          "threshold_difference": 0.10,
          "met": bool(gap is not None and gap >= 0.10)}

    ratio_cells = {n: rate(cell(records, "signal_ratio_threshold", n, 4.0),
                           "confirmation_recovered")
                   for n in (24, 32, 64, 128, 256)}
    h6 = {
        "cells": {str(k): v for k, v in ratio_cells.items()},
        "max_rate_n_le_128": max((v["rate"] or 0.0) for n, v in ratio_cells.items()
                                 if n <= 128),
        "n256_rate": ratio_cells[256]["rate"],
        "threshold_n_le_128_max": 0.20, "threshold_n256_max": 0.50,
        "met": bool(all((ratio_cells[n]["rate"] is not None
                         and ratio_cells[n]["rate"] <= 0.20)
                        for n in (24, 32, 64, 128))
                    and ratio_cells[256]["rate"] is not None
                    and ratio_cells[256]["rate"] <= 0.50),
    }

    boundary_out = None
    if boundary is not None:
        blocked = [r for r in boundary if r.get("status") == "blocked"]
        with_sample_count = [
            r for r in blocked
            if any("sample_count" in b for b in (r.get("blockers") or []))
        ]
        boundary_out = {
            "records": len(boundary),
            "blocked": len(blocked),
            "blocked_with_sample_count": len(with_sample_count),
            "met": bool(boundary and len(blocked) == len(boundary)
                        and len(with_sample_count) == len(boundary)),
        }

    executions = {
        "total": len(records),
        "ok": sum(1 for r in records if r["status"] == "ok"),
        "blocked": sum(1 for r in records if r["status"] == "blocked"),
        "error": sum(1 for r in records if r["status"] == "error"),
        "blocked_cells": {
            f"{r['mode']}|s={r.get('strength')}|n={r['n_samples']}": True
            for r in records if r["status"] == "blocked"
        },
        "zero_execution_failures": not any(r["status"] == "error" for r in records),
    }

    return {"H1_calibration": h1, "H2_power": h2, "H3_effect_size": h3,
            "H4_holdout_strictness": h4, "H5_discovery_calibration": h5,
            "H6_ratio_hard_positive": h6,
            "boundary_n16": boundary_out, "executions": executions}


def fmt(rate_dict: dict[str, Any]) -> str:
    if rate_dict["rate"] is None:
        return "n/a"
    return (f"{rate_dict['k']}/{rate_dict['n']} = {rate_dict['rate']:.3f} "
            f"(95% CI {rate_dict['ci_low']:.3f}–{rate_dict['ci_high']:.3f})")


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Pre-registered hypothesis evaluation", ""]
    h1 = result["H1_calibration"]
    lines += ["## H1 — null calibration (primary)", "",
              f"- pooled confirmation false positives: {fmt(h1['pooled'])}",
              f"- worst cell: {max(v['rate'] or 0 for v in h1['cells'].values()):.3f} "
              f"(threshold 0.075)",
              f"- **{'MET' if h1['met'] else 'NOT MET'}**", ""]
    h2 = result["H2_power"]
    lines += ["## H2 — power, interaction s=3.0 (primary)", "",
              f"- n=256 confirmed recovery: {fmt(h2['n256'])} (need ≥ 0.40)",
              f"- n=24 confirmed recovery: {fmt(h2['n24'])} (need ≤ 0.20)",
              f"- **{'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'}**", ""]
    h3 = result["H3_effect_size"]
    lines += ["## H3 — effect-size ordering at n=128 (primary)", "",
              f"- s=4.0: {fmt(h3['s4'])}",
              f"- s=2.0: {fmt(h3['s2'])}",
              f"- difference: {h3['difference']:.3f} (need ≥ 0.15)",
              f"- **{'SUPPORTED' if h3['supported'] else 'NOT SUPPORTED'}**", ""]
    h4 = result["H4_holdout_strictness"]
    lines += ["## H4 — holdout strictness, interaction s=3.0 (primary)", "",
              f"- small n (24–64), holdout-confirmed among confirmed: {fmt(h4['small_n'])}",
              f"- large n (128–256): {fmt(h4['large_n'])}",
              f"- **{'SUPPORTED' if h4['supported'] else 'NOT SUPPORTED'}**", ""]
    h5 = result["H5_discovery_calibration"]
    lines += ["## H5 — discovery calibration (secondary)", "",
              f"- null spurious discovery, n≤64: {fmt(h5['small_n'])}",
              f"- null spurious discovery, n≥128: {fmt(h5['large_n'])}",
              f"- difference: {h5['difference']:.3f} (expected ≥ 0.10)",
              f"- **{'MET' if h5['met'] else 'NOT MET'}**", ""]
    h6 = result["H6_ratio_hard_positive"]
    lines += ["## H6 — ratio hard positive, s=4.0 (secondary)", ""]
    for n, v in h6["cells"].items():
        lines.append(f"- n={n}: {fmt(v)}")
    lines += [f"- **{'MET' if h6['met'] else 'NOT MET'}**", ""]
    b = result.get("boundary_n16")
    if b:
        lines += ["## Boundary check — n=16", "",
                  f"- records: {b['records']}, blocked: {b['blocked']}, "
                  f"with `sample_count` blocker: {b['blocked_with_sample_count']}",
                  f"- **{'MET' if b['met'] else 'NOT MET'}**", ""]
    e = result["executions"]
    lines += ["## Execution integrity", "",
              f"- records: {e['total']} (ok {e['ok']}, blocked {e['blocked']}, "
              f"error {e['error']})",
              f"- zero execution failures: {e['zero_execution_failures']}", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--boundary-records", default=None)
    p.add_argument("--output", required=True, help="output prefix (no extension)")
    args = p.parse_args()
    result = evaluate(load(args.records),
                      load(args.boundary_records) if args.boundary_records else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
