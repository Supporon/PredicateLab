#!/usr/bin/env python
"""Render the key result figures for one run into ``<run>/figures/``.

Two static figures, chosen for the two jobs a report reader has:

``effects_forest.png``
    Per-candidate paired effect against zero, with the 95% CI where the
    evidence record has one.  Status is encoded by both colour and marker
    shape, so the figure survives colour-vision deficiency and greyscale.

``ic_series.png``
    Daily cross-sectional IC for the strongest candidate vs the MSE baseline
    and the ``control:*`` kernel, one small-multiple panel per validation
    fold.  Raw daily values are a faint hairline; the 20-day rolling mean is
    the readable line.

The figures are written next to the run's other artifacts, in the category
directory ``figures/``, so ``scripts/package_snapshot.py`` picks them up with
no special case.  The raw run data is never modified.

Examples
--------
    python scripts/plot_run_results.py --run-dir artifacts/tsil_tau_k005/run_20261108_110259
    python scripts/plot_run_results.py --run-dir ... --candidate cand_344d5916077d28b5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reference palette (dataviz skill).  Categorical slots are used in fixed
# order; status hues are reserved for candidate state and always paired with
# a marker shape.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
MUTED = "#8a8a85"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREEN = "#008300"
RED = "#e34948"

STATUS_STYLE = {
    "validated": (GREEN, "o", "validated"),
    "exploratory_positive": (GREEN, "o", "validated"),
    "rejected": (RED, "X", "rejected"),
}
FALLBACK_STATUS_STYLE = (MUTED, "s", "insufficient / screened")


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": TEXT,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": False,
        "legend.frameon": False,
    })


def _load(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _status_style(status: str) -> tuple[str, str, str]:
    for key, style in STATUS_STYLE.items():
        if key in str(status):
            return style
    return FALLBACK_STATUS_STYLE


def plot_effects(run_dir: Path, figures_dir: Path, *, max_rows: int = 24) -> Path | None:
    """Forest plot of paired effects, sorted by effect size."""
    import matplotlib.pyplot as plt

    report = _load(run_dir / "report.json")
    if not report:
        return None
    rows = []
    for item in report.get("evidence") or []:
        if not isinstance(item, dict) or not _finite(item.get("effect")):
            continue
        rows.append(item)
    if not rows:
        return None
    rows.sort(key=lambda r: float(r["effect"]))
    if len(rows) > max_rows:
        # Keep both tails: the extremes are the story, the middle is noise.
        head = rows[: max_rows // 2]
        tail = rows[-(max_rows - len(head)):]
        rows = head + tail

    fig_height = max(3.0, 1.3 + 0.26 * len(rows))
    fig, ax = plt.subplots(figsize=(7.2, fig_height), constrained_layout=True)
    ax.axvline(0.0, color=MUTED, linewidth=0.9, zorder=1)

    used: dict[str, Any] = {}
    for y, item in enumerate(rows):
        effect = float(item["effect"])
        lo, hi = item.get("ci_low"), item.get("ci_high")
        color, marker, label = _status_style(str(item.get("status", "")))
        if _finite(lo) and _finite(hi):
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.4, alpha=0.75,
                    solid_capstyle="round", zorder=2)
        handle = ax.scatter([effect], [y], s=40, color=color, marker=marker,
                            edgecolors=SURFACE, linewidths=1.2, zorder=3)
        used.setdefault(label, (handle, color, marker))

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{str(r.get('candidate_id', '?'))[:12]}  {r.get('status', '')}"
                        for r in rows])
    ax.set_xlabel("paired effect vs baseline (primary metric; 0 = no difference)")
    ax.set_title("Candidate effects with 95% CI", loc="left", color=TEXT, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.7, solid_capstyle="butt")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if used:
        handles = [entry[0] for entry in used.values()]
        labels = list(used)
        ax.legend(handles, labels, loc="lower right", fontsize=8,
                  markerscale=1.0, handletextpad=0.4)
    path = figures_dir / "effects_forest.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def _trial_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Parse ``trials/<phase>/<fold>/<candidate>/<variant>/seed_<n>/<hash>/``."""
    root = run_dir / "trials"
    if not root.is_dir():
        return []
    rows = []
    for result_path in root.rglob("result.json"):
        rel = result_path.relative_to(root).parts
        if len(rel) < 6:
            continue
        phase, fold, candidate_dir, variant, seed_part = rel[:5]
        data = _load(result_path)
        if not data or data.get("status") != "success":
            continue
        series = (data.get("metric_series") or {}).get("ic")
        if not isinstance(series, dict) or len(series) < 2:
            continue
        rows.append({
            "phase": phase,
            "fold": fold,
            "candidate_id": candidate_dir,
            "variant": variant,
            "seed": int(seed_part.replace("seed_", "")) if seed_part.startswith("seed_") else 0,
            "ic": series,
        })
    return rows


def _pick_candidate(report: dict[str, Any], trial_rows: list[dict[str, Any]]) -> str | None:
    present = {row["candidate_id"] for row in trial_rows if row["variant"] == "candidate"}
    candidates = [
        item for item in (report.get("evidence") or [])
        if isinstance(item, dict) and item.get("candidate_id") in present
        and _finite(item.get("effect"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: abs(float(item["effect"])), reverse=True)
    return str(candidates[0]["candidate_id"])


def _rolling(dates: list[str], values: list[float], window: int = 20):
    import pandas as pd

    series = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    return series.index, series, series.rolling(window, min_periods=max(2, window // 2)).mean()


def plot_ic_series(run_dir: Path, figures_dir: Path,
                   candidate: str | None = None) -> Path | None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    report = _load(run_dir / "report.json") or {}
    rows = _trial_rows(run_dir)
    if not rows:
        return None

    # Prefer full-validation trials; fall back to screen.  If a candidate was
    # requested, use whichever phase actually ran it.  One seed keeps the panel
    # readable, and the seed is stated in the subtitle.
    phase_rows = []
    for phase in ("full", "screen"):
        candidates = [row for row in rows
                      if row["phase"] == phase and row["variant"] == "candidate"]
        if candidate is None:
            if candidates:
                phase_rows = [row for row in rows if row["phase"] == phase]
                candidate = _pick_candidate(report, phase_rows)
                break
        elif any(row["candidate_id"] == candidate for row in candidates):
            phase_rows = [row for row in rows if row["phase"] == phase]
            break
    if not phase_rows or not candidate:
        return None
    seeds = sorted({row["seed"] for row in phase_rows if row["candidate_id"] == candidate})
    if not seeds:
        return None
    seed = seeds[0]
    folds = sorted({row["fold"] for row in phase_rows})
    folds = folds[:3]
    if not folds:
        return None

    def series_for(fold: str, variant_kind: str) -> dict[str, float] | None:
        for row in phase_rows:
            if row["fold"] != fold or row["seed"] != seed:
                continue
            if variant_kind == "candidate" and row["candidate_id"] == candidate \
                    and row["variant"] == "candidate":
                return row["ic"]
            if variant_kind == "baseline" and row["variant"] == "baseline":
                return row["ic"]
            if variant_kind == "control" and row["variant"] == "control":
                return row["ic"]
        return None

    panels = [(fold, series_for(fold, "candidate"), series_for(fold, "baseline"),
               series_for(fold, "control")) for fold in folds]
    panels = [panel for panel in panels if panel[1]]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4),
                             sharey=True, constrained_layout=True, squeeze=False)
    axes = axes[0]
    entries = [
        ("candidate", "candidate", BLUE),
        ("baseline (MSE)", "baseline", ORANGE),
        ("control kernel", "control", AQUA),
    ]
    handles = {}
    for ax, (fold, cand_ic, base_ic, ctrl_ic) in zip(axes, panels):
        ax.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
        for label, key, color in entries:
            series = {"candidate": cand_ic, "baseline": base_ic, "control": ctrl_ic}[key]
            if not series:
                continue
            dates = sorted(series)
            values = [float(series[d]) for d in dates]
            xs, raw, smooth = _rolling(dates, values)
            ax.plot(xs, raw.values, color=color, linewidth=0.7, alpha=0.22, zorder=2)
            handle, = ax.plot(xs, smooth.values, color=color, linewidth=2.0,
                              label=label, zorder=3)
            handles.setdefault(label, handle)
        ax.set_title(f"fold {fold}", loc="left", color=TEXT, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("daily cross-sectional IC")
    fig.suptitle(f"IC series — {candidate[:12]} (seed {seed}, {phase})",
                 x=0.01, ha="left", color=TEXT, fontsize=11)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="lower center",
                   ncol=len(handles), bbox_to_anchor=(0.5, -0.04), fontsize=8)
    path = figures_dir / "ic_series.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--candidate", default=None,
                        help="candidate id for the IC series (default: largest |effect|)")
    parser.add_argument("--max-effects", type=int, default=24)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "report.json").exists():
        raise SystemExit(f"report.json not found under {run_dir}")
    figures_dir = Path(args.figures_dir) if args.figures_dir else run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _style()
    produced = []
    forest = plot_effects(run_dir, figures_dir, max_rows=args.max_effects)
    if forest:
        produced.append(forest)
    series = plot_ic_series(run_dir, figures_dir, candidate=args.candidate)
    if series:
        produced.append(series)
    if not produced:
        print("no figures produced (no effect/IC data found)", file=sys.stderr)
        return 1
    for path in produced:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
