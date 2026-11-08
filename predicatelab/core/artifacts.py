"""Artifact taxonomy and portable snapshot packaging.

A local run under ``artifacts/`` deliberately mixes many kinds of output: the
SQLite ledger, 13 MB data manifests, per-trial prediction arrays and optional
model weights sit next to the handful of small files that actually describe
what happened.  That is convenient while running and wrong for a snapshot you
want to migrate or commit to GitHub.

This module draws the boundary once, by content type:

=========== =============================================================
kind        contents
=========== =============================================================
results     ``report.json`` / ``report.md``, ``context.json``,
            ``evidence.jsonl``, ``frozen_*.json``, ``confirmation.json``,
            ``final_evaluation.json`` and per-trial ``result.json``
logs        runner / worker / training logs (``*.log``, ``*.out``, ``*.err``)
figures     plots (``*.png``, ``*.svg``, ``*.pdf``, ``*.jpg``)
models      trained weights (``*.pt``, ``*.pth``, ``*.bin``, ``*.ckpt``)
data        prediction arrays, probe batches, manifests, SQLite databases
code        small analysis / reproduction scripts (``*.py``, ``*.sh``)
=========== =============================================================

``build_run_snapshot`` and ``build_workspace_snapshot`` copy only the kinds
requested and write ``MANIFEST.json``, ``checksums.sha256``, a human-readable
``README.md`` and a ``changes/`` directory with git state, a config snapshot
and environment versions.  Models and data are excluded by default: that is
what keeps a committed snapshot small.  Pass ``include_models`` /
``include_data`` when building a full archival tarball instead.

The functions never delete or rewrite a raw run: they only read from
``artifacts/`` and write to the snapshot destination.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as _metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from predicatelab.core.identity import atomic_write_text, strict_json_dumps

SNAPSHOT_SCHEMA = "predicatelab.snapshot/1"

KINDS = ("results", "logs", "figures", "models", "data", "code")
DEFAULT_INCLUDE = ("results", "logs", "figures", "code")
HEAVY_INCLUDE = ("models", "data")

MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".webp"}
LOG_SUFFIXES = {".log", ".out", ".err"}
CODE_SUFFIXES = {".py", ".sh", ".ipynb"}
DATA_SUFFIXES = {
    ".npz", ".npy", ".pkl", ".pickle", ".h5", ".hdf5", ".parquet",
    ".feather", ".arrow", ".sqlite", ".sqlite3", ".db", ".gz", ".zip",
    ".tar", ".tgz", ".7z",
}
RESULT_SUFFIXES = {
    ".json", ".jsonl", ".ndjson", ".md", ".csv", ".tsv", ".txt",
    ".yaml", ".yml", ".html", ".sha256",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".lock", ".shm", ".wal", ".tmp", ".swp"}
SKIP_NAMES = {".ds_store", "thumbs.db"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}

MODEL_DIRS = {"models", "checkpoints"}
FIGURE_DIRS = {"figures", "plots"}
LOG_DIRS = {"logs"}
DATA_DIRS = {"_manifests", "predictions"}

DIFF_CAP_BYTES = 2 * 1024 * 1024
STATUS_CAP_BYTES = 256 * 1024
SECRET_KEYS = {
    "api_key", "apikey", "token", "access_token", "password", "passwd",
    "secret", "client_secret", "access_key", "secret_key", "deepseek_key",
}


# --------------------------------------------------------------------- taxonomy
def classify_artifact(path: str | Path) -> str | None:
    """Return the artifact kind for *path*, or ``None`` when it is transient.

    Classification is by directory name first (an explicit ``models/`` or
    ``logs/`` directory always wins), then by suffix.  Unrecognised small files
    default to ``results`` because that is the safe side of the boundary: a
    forgotten report is published, a forgotten weight file is not.
    """
    p = Path(path)
    name = p.name.lower()
    parts = {part.lower() for part in p.parts[:-1]}
    suffix = p.suffix.lower()

    if name in SKIP_NAMES or suffix in SKIP_SUFFIXES:
        return None
    if name.endswith("-shm") or name.endswith("-wal"):
        return None
    if parts & MODEL_DIRS:
        return "models"
    if suffix in MODEL_SUFFIXES:
        return "models"
    if parts & FIGURE_DIRS:
        return "figures"
    if suffix in FIGURE_SUFFIXES:
        return "figures"
    if parts & LOG_DIRS or name in {"worker.log", "run.log"} or suffix in LOG_SUFFIXES:
        return "logs"
    if parts & DATA_DIRS:
        return "data"
    if suffix in CODE_SUFFIXES:
        return "code"
    if suffix in DATA_SUFFIXES:
        return "data"
    return "results"


TRIAL_DETAIL_NAMES = {
    "result.json", "trial_result.json", "worker_result.json", "job.json",
}


def is_trial_detail(path: str | Path) -> bool:
    """Per-trial JSON is bulk detail, not a key run-level result.

    A full run can hold hundreds of these (each with a 200+ point metric
    series); copying all of them would dominate a lightweight snapshot.  They
    stay in the raw run and are summarised into ``results/trials_summary.csv``
    unless ``include_trial_details`` is requested.
    """
    p = Path(path)
    return p.name.lower() in TRIAL_DETAIL_NAMES and "trials" in {
        part.lower() for part in p.parts[:-1]
    }


@dataclass
class ArtifactEntry:
    """One source file and the category it belongs to."""

    kind: str
    source: Path
    relpath: str
    size: int
    sha256: str = ""
    bulk: bool = False

    def record(self, dest: str) -> dict[str, Any]:
        return {
            "path": dest,
            "kind": self.kind,
            "source": str(self.source),
            "bytes": self.size,
            "sha256": self.sha256,
        }


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def collect_artifacts(root: str | Path) -> list[ArtifactEntry]:
    """Classify every publishable file under *root* (no hashing yet)."""
    base = Path(root)
    entries: list[ArtifactEntry] = []
    for path in _iter_files(base):
        kind = classify_artifact(path)
        if kind is None:
            continue
        try:
            rel = path.relative_to(base).as_posix()
            size = path.stat().st_size
        except OSError:
            continue
        entries.append(ArtifactEntry(
            kind=kind, source=path, relpath=rel, size=size,
            bulk=is_trial_detail(path),
        ))
    entries.sort(key=lambda e: (e.kind, e.relpath))
    return entries


def summarize_entries(entries: Iterable[ArtifactEntry]) -> dict[str, dict[str, int]]:
    summary = {kind: {"files": 0, "bytes": 0} for kind in KINDS}
    for entry in entries:
        bucket = summary.setdefault(entry.kind, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += entry.size
    return summary


def summarize_bulk(entries: Iterable[ArtifactEntry]) -> dict[str, int]:
    """Count the per-trial detail files skipped by a compact snapshot."""
    files = 0
    size = 0
    for entry in entries:
        if entry.bulk:
            files += 1
            size += entry.size
    return {"files": files, "bytes": size}


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def _kind_table(
    summary: Mapping[str, Mapping[str, int]],
    include: Sequence[str],
    published: Iterable[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Source totals per kind next to how many files actually got copied.

    The distinction matters: a run can hold 99 small ``results`` files but the
    snapshot only copies the run-level ones, with per-trial detail folded into
    ``results/trials_summary.csv``.
    """
    published_counts: dict[str, int] = {}
    for rec in published or ():
        if rec.get("kind") in KINDS:
            published_counts[rec["kind"]] = published_counts.get(rec["kind"], 0) + 1
    lines = ["| kind | source files | source size | published files |",
             "|---|---:|---:|---:|"]
    for kind in KINDS:
        counts = summary.get(kind, {"files": 0, "bytes": 0})
        if not counts["files"]:
            continue
        n_published = published_counts.get(kind, 0) if kind in include else 0
        lines.append(
            f"| `{kind}` | {counts['files']} | {_human_bytes(counts['bytes'])} "
            f"| {n_published} |"
        )
    return lines


# ------------------------------------------------------------------- copying
def _copy_and_hash(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with src.open("rb") as fh_in, dst.open("wb") as fh_out:
        for chunk in iter(lambda: fh_in.read(1 << 20), b""):
            digest.update(chunk)
            fh_out.write(chunk)
    try:
        shutil.copystat(src, dst)
    except OSError:
        pass
    return digest.hexdigest()


def _dest_relpath(entry: ArtifactEntry) -> str:
    """Destination path inside the snapshot, without a duplicated kind folder.

    A source path already ending in its kind directory (``figures/plot.png``,
    ``trials/.../models/model.pt``) would otherwise produce
    ``figures/figures/plot.png``; the redundant component is dropped so the
    kind folder is always the first level.
    """
    parts = list(Path(entry.relpath).parts)
    if parts and parts[0].lower() == entry.kind:
        parts = parts[1:]
    elif len(parts) >= 2 and parts[-2].lower() == entry.kind:
        parts = parts[:-2] + [parts[-1]]
    return f"{entry.kind}/{'/'.join(parts)}"


def _publish_entries(
    entries: Sequence[ArtifactEntry],
    include: Sequence[str],
    out_dir: Path,
    *,
    include_trial_details: bool = False,
) -> tuple[list[dict[str, Any]], list[ArtifactEntry]]:
    """Copy included entries into ``out_dir/<kind>/<relpath>``.

    Per-trial detail JSON is skipped unless ``include_trial_details`` is set;
    the caller writes a compact CSV summary instead.  Returns
    ``(file_records, published_entries)``.  The entries are mutated in place so
    their ``sha256`` is available to the caller for manifests.
    """
    records: list[dict[str, Any]] = []
    published: list[ArtifactEntry] = []
    for entry in entries:
        if entry.kind not in include:
            continue
        if entry.bulk and not include_trial_details:
            continue
        dest_rel = _dest_relpath(entry)
        dest = out_dir / dest_rel
        entry.sha256 = _copy_and_hash(entry.source, dest)
        records.append(entry.record(dest_rel))
        published.append(entry)
    return records, published


def _write_checksums(out_dir: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    lines = [f"{rec['sha256']}  {rec['path']}" for rec in sorted(records, key=lambda r: r["path"])]
    path = out_dir / "checksums.sha256"
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return path


def _write_manifest(
    out_dir: Path,
    *,
    scope: str,
    source_roots: Sequence[str],
    include: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    excluded: Mapping[str, Mapping[str, int]],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    kinds = {kind: {"files": 0, "bytes": 0} for kind in KINDS}
    for rec in records:
        kinds[rec["kind"]]["files"] += 1
        kinds[rec["kind"]]["bytes"] += rec["bytes"]
    manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": list(source_roots),
        "include": list(include),
        "excluded_kinds": dict(excluded),
        "kinds": kinds,
        "files": sorted(records, key=lambda r: r["path"]),
        "tool": _tool_versions(),
    }
    if extra:
        manifest.update(extra)
    atomic_write_text(out_dir / "MANIFEST.json", strict_json_dumps(manifest))
    return out_dir / "MANIFEST.json"


# --------------------------------------------------------------- change info
def _cap(text: str, limit: int) -> tuple[str, bool]:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", errors="ignore") + "\n... [truncated]\n", True


def _git(repo: Path, *args: str) -> tuple[str, int]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return f"git unavailable: {type(exc).__name__}: {exc}\n", 127
    out = proc.stdout or ""
    if proc.returncode != 0 and proc.stderr:
        out += ("\n" + proc.stderr) if out else proc.stderr
    return out, proc.returncode


def _tool_versions() -> dict[str, Any]:
    packages = [
        "torch", "numpy", "pandas", "scipy", "scikit-learn", "pydantic",
        "qlib", "matplotlib", "PyYAML",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = _metadata.version(package)
        except Exception:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }


def _sanitise_config(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        clean: dict[str, Any] = {}
        for key, value in obj.items():
            if str(key).lower() in SECRET_KEYS:
                clean[str(key)] = "***redacted***"
            else:
                clean[str(key)] = _sanitise_config(value)
        return clean
    if isinstance(obj, (list, tuple)):
        return [_sanitise_config(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_config_snapshot(path: Path, config: Mapping[str, Any] | None) -> None:
    if config is None:
        return
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_sanitise_config(dict(config)), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def write_change_info(
    out_dir: Path,
    *,
    repo_root: str | Path | None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    include: Sequence[str] = DEFAULT_INCLUDE,
    summary_lines: Sequence[str] = (),
) -> dict[str, Any]:
    """Write the small ``changes/`` directory describing how this snapshot was made.

    It captures the working-tree diff (not the repository), the current commit
    and status, a redacted config snapshot and package versions.  Everything is
    capped so a large uncommitted diff cannot silently bloat the snapshot.
    """
    changes = out_dir / "changes"
    changes.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"dir": "changes"}

    if repo_root is not None:
        repo = Path(repo_root)
        commit, rc = _git(repo, "rev-parse", "HEAD")
        branch, _ = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        status, _ = _git(repo, "status", "--porcelain=v1")
        diff_stat, _ = _git(repo, "diff", "--stat", "--no-color")
        diff, _ = _git(repo, "diff", "--no-color")
        log, _ = _git(repo, "log", "-n", "20", "--oneline", "--decorate")
        status, status_cut = _cap(status, STATUS_CAP_BYTES)
        diff, diff_cut = _cap(diff, DIFF_CAP_BYTES)
        payloads = {
            "git_commit.txt": commit.strip() or "unknown",
            "git_branch.txt": branch.strip() or "unknown",
            "git_status.txt": status,
            "git_diff_stat.txt": diff_stat,
            "git_diff.patch": diff,
            "git_log.txt": log,
        }
        for name, text in payloads.items():
            atomic_write_text(changes / name, text if text.endswith("\n") or not text else text + "\n")
        info.update({
            "repo_root": str(repo),
            "commit": commit.strip(),
            "branch": branch.strip(),
            "diff_truncated": diff_cut,
            "status_truncated": status_cut,
            "git_status": "changes/git_status.txt",
            "git_diff": "changes/git_diff.patch",
        })

    atomic_write_text(changes / "environment.json", strict_json_dumps(_tool_versions()))
    info["environment"] = "changes/environment.json"

    if config is not None:
        _write_config_snapshot(changes / "config_snapshot.yaml", config)
        info["config_snapshot"] = "changes/config_snapshot.yaml"
    if config_path is not None:
        src = Path(config_path)
        if src.is_file():
            dst = changes / "config_source.yaml"
            shutil.copy2(src, dst)
            info["config_source"] = "changes/config_source.yaml"

    lines = [
        "# Change record",
        "",
        "This directory describes the code and configuration state used to build",
        "the snapshot. The raw run directories are the source of truth; this is",
        "the lightweight, portable record.",
        "",
        f"- published kinds: `{', '.join(include)}`",
        f"- git commit: `{(info.get('commit') or 'unknown')[:12]}`"
        f" on `{info.get('branch') or 'unknown'}`",
        f"- working tree diff: `changes/git_diff.patch`"
        + (" (truncated)" if info.get("diff_truncated") else ""),
        "",
    ]
    if summary_lines:
        lines.extend(["## Experiments in this snapshot", ""])
        lines.extend(summary_lines)
        lines.append("")
    lines.extend([
        "## Reproduce",
        "",
        "```bash",
        "python -m predicatelab.cli discover --config <config> --rounds <n>",
        "python scripts/package_snapshot.py --artifacts-root artifacts --out published",
        "```",
        "",
    ])
    atomic_write_text(changes / "CHANGE_SUMMARY.md", "\n".join(lines))
    info["summary"] = "changes/CHANGE_SUMMARY.md"
    return info


# ------------------------------------------------------------- run summaries
def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def run_facts(run_dir: str | Path) -> dict[str, Any]:
    """Extract a compact, human-readable verdict from a run directory."""
    root = Path(run_dir)
    facts: dict[str, Any] = {"run_id": root.name}
    report = _load_json(root / "report.json")
    if report:
        facts.update({
            "run_id": report.get("run_id", root.name),
            "domain": report.get("domain"),
            "run_status": report.get("run_status"),
            "protocol_hash": report.get("protocol_hash"),
            "identity_hash": report.get("identity_hash"),
            "evidence_counts": report.get("evidence_counts") or {},
            "state_counts": report.get("state_counts") or {},
            "rounds": [
                {
                    "index": row.get("round_index"),
                    "status": row.get("status"),
                    "proposed": row.get("n_proposed"),
                    "validated": row.get("n_validated"),
                    "insufficient": row.get("n_insufficient"),
                }
                for row in (report.get("rounds") or [])
                if isinstance(row, dict)
            ],
            "final_evals": [
                {
                    "candidate_id": row.get("candidate_id"),
                    "state": row.get("state"),
                    "effect": row.get("effect"),
                    "p_value": row.get("p_value"),
                }
                for row in (report.get("final_evals") or [])
                if isinstance(row, dict)
            ],
        })
    final = _load_json(root / "final_evaluation.json")
    if final:
        facts["final_evaluation"] = {
            key: final.get(key)
            for key in ("state", "effect", "ci_low", "ci_high", "p_value", "exploratory")
            if key in final
        }
    return facts


def write_trials_summary(run_dirs: Sequence[str | Path], dest: str | Path) -> Path | None:
    """Write one compact CSV row per per-trial ``result.json``.

    The full per-trial JSON stays in the raw run; this keeps the *numbers*
    (status, fold, seed, headline metrics) in the lightweight snapshot without
    copying hundreds of kilobyte-sized metric series.
    """
    import csv

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        trials_root = root / "trials"
        if not trials_root.is_dir():
            continue
        for path in sorted(trials_root.rglob("result.json")):
            data = _load_json(path)
            if not data:
                continue
            parts = path.relative_to(trials_root).parts
            worker = data.get("worker") if isinstance(data.get("worker"), dict) else {}
            metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
            error_lines = str(data.get("error") or "").splitlines()
            rows.append({
                "run_id": root.name,
                "phase": parts[0] if len(parts) > 0 else "",
                "fold": parts[1] if len(parts) > 1 else "",
                "candidate": parts[2] if len(parts) > 2 else "",
                "variant": parts[3] if len(parts) > 3 else "",
                "seed": worker.get("seed", ""),
                "status": data.get("status", ""),
                "ic": metrics.get("ic", ""),
                "rank_ic": metrics.get("rank_ic", ""),
                "icir": metrics.get("icir", ""),
                "mse": metrics.get("mse", ""),
                "duration_s": data.get("duration_s", ""),
                "model_path": worker.get("model_path", ""),
                "error": (error_lines[0] if error_lines else "")[:300],
            })
    if not rows:
        return None
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "phase", "fold", "candidate", "variant", "seed", "status",
        "ic", "rank_ic", "icir", "mse", "duration_s", "model_path", "error",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _generated_record(path: Path, out_dir: Path, *, kind: str = "results",
                      source: str = "generated") -> dict[str, Any]:
    return {
        "path": path.relative_to(out_dir).as_posix(),
        "kind": kind,
        "source": source,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _facts_table(facts: Mapping[str, Any]) -> list[str]:
    lines = [
        f"- run: `{facts.get('run_id')}`",
        f"- domain: `{facts.get('domain') or 'unknown'}`",
        f"- run status: `{facts.get('run_status') or 'unknown'}`",
    ]
    counts = facts.get("evidence_counts") or {}
    if counts:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- evidence: {rendered}")
    for row in facts.get("rounds") or []:
        lines.append(
            f"- round {row.get('index')}: {row.get('status')} "
            f"(proposed {row.get('proposed')}, validated {row.get('validated')}, "
            f"insufficient {row.get('insufficient')})"
        )
    for row in facts.get("final_evals") or []:
        lines.append(
            f"- final `{row.get('candidate_id')}`: {row.get('state')} "
            f"(effect {row.get('effect')}, p {row.get('p_value')})"
        )
    if facts.get("final_evaluation"):
        lines.append(f"- final evaluation: `{facts['final_evaluation']}`")
    return lines


def _caveat_lines(repo_root: Path | None) -> list[str]:
    lines = [
        "> Credibility note: before citing any TSIL/LUSI result from this repository,",
        "> read `docs/audit_20261107/REPORT.md` (or `audit_20261107_credibility/REPORT.md`).",
        "> Discovery-positive is `exploratory_positive`; it is not independent confirmation.",
    ]
    if repo_root is not None:
        audit = Path(repo_root) / "docs" / "audit_20261107" / "REPORT.md"
        if not audit.exists():
            audit = Path(repo_root).parent / "audit_20261107_credibility" / "REPORT.md"
        if audit.exists():
            lines.append(f"> Audit report: `{audit}`.")
    return lines


def _render_run_readme(
    out_dir: Path,
    *,
    run_dir: Path,
    include: Sequence[str],
    summary: Mapping[str, Mapping[str, int]],
    facts: Mapping[str, Any],
    config_path: str | Path | None,
    repo_root: Path | None,
    bulk: Mapping[str, int] | None = None,
    published: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    lines = [f"# Snapshot — {facts.get('run_id', run_dir.name)}", ""]
    lines.append(f"Source: `{run_dir}`")
    lines.append("")
    lines.extend(_kind_table(summary, include, published))
    lines.append("")
    if bulk and bulk.get("files"):
        lines.append(
            f"Per-trial `result.json`/`job.json` are not copied: {bulk['files']} files "
            f"({_human_bytes(bulk['bytes'])}) are summarised in "
            "`results/trials_summary.csv`. They remain in the raw run."
        )
        lines.append("")
    lines.extend(["## Outcome", ""])
    lines.extend(_facts_table(facts))
    lines.append("")
    key_results = [
        "results/report.md", "results/report.json", "results/evidence.jsonl",
        "results/final_evaluation.json", "results/context.json",
    ]
    present = [name for name in key_results if (out_dir / name).exists()]
    if present:
        lines.extend(["## Key results", ""])
        lines.extend(f"- `{name}`" for name in present)
        lines.append("")
    if config_path:
        lines.extend(["## Config", "", f"`{config_path}` (copied to `changes/`).", ""])
    lines.extend(["## Credibility", ""])
    lines.extend(_caveat_lines(repo_root))
    lines.append("")
    path = out_dir / "README.md"
    atomic_write_text(path, "\n".join(lines))
    return path


# ---------------------------------------------------------------- build APIs
def build_run_snapshot(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    include: Sequence[str] = DEFAULT_INCLUDE,
    include_models: bool = False,
    include_data: bool = False,
    include_trial_details: bool = False,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Build a lightweight (or full) snapshot of one run directory.

    Parameters mirror the CLI: ``include_models`` and ``include_data`` are the
    two opt-in heavy categories, and ``include_trial_details`` opts back into
    per-trial ``result.json``/``job.json`` (otherwise only
    ``results/trials_summary.csv`` is written).  The destination is
    created/updated; no existing file outside it is touched.
    """
    source = Path(run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"run directory not found: {source}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kinds = list(include)
    if include_models and "models" not in kinds:
        kinds.append("models")
    if include_data and "data" not in kinds:
        kinds.append("data")

    entries = collect_artifacts(source)
    summary = summarize_entries(entries)
    excluded = {
        kind: summary.get(kind, {"files": 0, "bytes": 0})
        for kind in KINDS if kind not in kinds
    }
    bulk = summarize_bulk(entries)
    records, published = _publish_entries(
        entries, kinds, out, include_trial_details=include_trial_details,
    )
    if not include_trial_details:
        summary_csv = write_trials_summary([source], out / "results" / "trials_summary.csv")
        if summary_csv is not None:
            records.append(_generated_record(summary_csv, out))

    repo_path = Path(repo_root) if repo_root is not None else None
    facts = run_facts(source)
    _render_run_readme(
        out,
        run_dir=source,
        include=kinds,
        summary=summary,
        facts=facts,
        config_path=config_path,
        repo_root=repo_path,
        bulk=bulk,
        published=records,
    )
    change_info = write_change_info(
        out,
        repo_root=repo_path,
        config=config,
        config_path=config_path,
        include=kinds,
        summary_lines=[f"- `{facts.get('run_id', source.name)}`: {facts.get('run_status', 'unknown')}"],
    )
    _write_manifest(
        out,
        scope="run",
        source_roots=[str(source)],
        include=kinds,
        records=records,
        excluded=excluded,
        extra={
            "run_facts": facts,
            "change_info": change_info,
            "excluded_trial_details": {} if include_trial_details else bulk,
        },
    )
    _write_checksums(out, records)
    return out


def discover_groups(artifacts_root: str | Path, include: Sequence[str]) -> dict[str, list[Path]]:
    """Group an artifacts root into experiment buckets.

    Immediate child directories are experiments.  A leading underscore marks
    scratch/data-only directories (``_probe_batches``); they are skipped unless
    the caller explicitly publishes ``data``.  Loose files belong to ``_root``.
    """
    root = Path(artifacts_root)
    groups: dict[str, list[Path]] = {}
    if not root.is_dir():
        return groups
    for child in sorted(root.iterdir()):
        if child.is_dir():
            if child.name.startswith("_") and "data" not in include:
                continue
            groups[child.name] = [child]
        elif child.is_file():
            groups.setdefault("_root", []).append(child)
    return groups


def build_workspace_snapshot(
    artifacts_root: str | Path,
    out_dir: str | Path,
    *,
    include: Sequence[str] = DEFAULT_INCLUDE,
    include_models: bool = False,
    include_data: bool = False,
    include_trial_details: bool = False,
    configs_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    experiments: Sequence[str] | None = None,
) -> Path:
    """Build one snapshot for every experiment under *artifacts_root*.

    Layout::

        published/
          INDEX.md
          MANIFEST.json
          checksums.sha256
          changes/...
          <experiment>/README.md
          <experiment>/<kind>/<path-inside-experiment>
    """
    root = Path(artifacts_root)
    if not root.is_dir():
        raise FileNotFoundError(f"artifacts root not found: {root}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    kinds = list(include)
    if include_models and "models" not in kinds:
        kinds.append("models")
    if include_data and "data" not in kinds:
        kinds.append("data")

    groups = discover_groups(root, kinds)
    if experiments is not None:
        wanted = set(experiments)
        groups = {name: paths for name, paths in groups.items() if name in wanted}
    if not groups:
        raise RuntimeError(f"no experiment directories found under {root}")

    repo_path = Path(repo_root) if repo_root is not None else None
    all_records: list[dict[str, Any]] = []
    excluded_total: dict[str, dict[str, int]] = {
        kind: {"files": 0, "bytes": 0} for kind in KINDS if kind not in kinds
    }
    bulk_total = {"files": 0, "bytes": 0}
    index_rows: list[dict[str, Any]] = []
    summary_lines: list[str] = []

    for name, sources in groups.items():
        group_out = out / name
        entries: list[ArtifactEntry] = []
        for source in sources:
            entries.extend(collect_artifacts(source))
        # Recompute relpaths relative to the experiment root for a stable layout.
        base = sources[0] if sources and sources[0].is_dir() else root
        for entry in entries:
            try:
                entry.relpath = entry.source.relative_to(base).as_posix()
            except ValueError:
                entry.relpath = entry.source.name
        entries.sort(key=lambda e: (e.kind, e.relpath))
        summary = summarize_entries(entries)
        for kind in excluded_total:
            excluded_total[kind]["files"] += summary.get(kind, {}).get("files", 0)
            excluded_total[kind]["bytes"] += summary.get(kind, {}).get("bytes", 0)

        records, _published = _publish_entries(
            entries, kinds, group_out, include_trial_details=include_trial_details,
        )
        run_dirs = sorted(
            [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")]
            if base.is_dir() else []
        )
        bulk = {"files": 0, "bytes": 0}
        if not include_trial_details:
            bulk = summarize_bulk(entries)
            bulk_total["files"] += bulk["files"]
            bulk_total["bytes"] += bulk["bytes"]
            summary_csv = write_trials_summary(
                run_dirs or [base], group_out / "results" / "trials_summary.csv",
            )
            if summary_csv is not None:
                records.append(_generated_record(summary_csv, group_out))
        # The global manifest/checksums are rooted at the snapshot destination,
        # while per-experiment records stay relative to their own folder.
        all_records.extend({**rec, "path": f"{name}/{rec['path']}"} for rec in records)

        # Per-experiment README with its runs and outcome summary.
        lines = [f"# Experiment snapshot — {name}", "", f"Source: `{base}`", ""]
        lines.extend(_kind_table(summary, kinds, records))
        lines.append("")
        if bulk.get("files"):
            lines.append(
                f"Per-trial `result.json`/`job.json` are not copied: {bulk['files']} files "
                f"({_human_bytes(bulk['bytes'])}) are summarised in "
                "`results/trials_summary.csv`. They remain in the raw runs."
            )
            lines.append("")
        if run_dirs:
            lines.extend(["## Runs", "", "| run | status | validated | insufficient |", "|---|---|---:|---:|"])
            for run_dir in run_dirs:
                facts = run_facts(run_dir)
                lines.append(
                    f"| `{facts.get('run_id')}` | {facts.get('run_status', 'unknown')} "
                    f"| {(facts.get('evidence_counts') or {}).get('validated', 0)} "
                    f"| {(facts.get('evidence_counts') or {}).get('insufficient_evidence', 0)} |"
                )
            lines.append("")
        else:
            facts = {"run_id": name, "run_status": "n/a (auxiliary output)"}
            lines.extend(_facts_table(facts))
            lines.append("")
        lines.extend(["## Credibility", ""])
        lines.extend(_caveat_lines(repo_path))
        lines.append("")
        atomic_write_text(group_out / "README.md", "\n".join(lines))

        kinds_present = {
            kind: {"files": summary.get(kind, {}).get("files", 0),
                   "bytes": summary.get(kind, {}).get("bytes", 0)}
            for kind in KINDS
        }
        published_counts: dict[str, int] = {}
        for rec in records:
            published_counts[rec["kind"]] = published_counts.get(rec["kind"], 0) + 1
        atomic_write_text(
            group_out / "snapshot_summary.json",
            strict_json_dumps({
                "schema": SNAPSHOT_SCHEMA,
                "experiment": name,
                "source": str(base),
                "include": kinds,
                "kinds": kinds_present,
                "published": published_counts,
                "files": sorted(rec["path"] for rec in records),
            }),
        )
        index_rows.append({
            "experiment": name,
            "files": len(records),
            "bytes": sum(rec["bytes"] for rec in records),
            "kinds": published_counts,
        })
        summary_lines.append(
            f"- `{name}`: {len(records)} files, {_human_bytes(sum(rec['bytes'] for rec in records))}"
        )

    # Global index and change record.
    lines = [
        "# PredicateLab lightweight snapshot",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Source: `{root}`",
        "",
        "Published kinds: " + ", ".join(f"`{kind}`" for kind in kinds),
        "",
        "| experiment | files | size | kinds |",
        "|---|---:|---:|---|",
    ]
    for row in index_rows:
        kinds_text = ", ".join(sorted(row["kinds"]))
        lines.append(
            f"| [`{row['experiment']}`]({row['experiment']}/README.md) "
            f"| {row['files']} | {_human_bytes(row['bytes'])} | {kinds_text} |"
        )
    lines.extend(["", "## Not published", ""])
    for kind in KINDS:
        counts = excluded_total.get(kind)
        if counts and counts["files"]:
            lines.append(f"- `{kind}`: {counts['files']} files, {_human_bytes(counts['bytes'])}")
    lines.extend(["", "## Credibility", ""])
    lines.extend(_caveat_lines(repo_path))
    lines.append("")
    atomic_write_text(out / "INDEX.md", "\n".join(lines))

    change_info = write_change_info(
        out,
        repo_root=repo_path,
        include=kinds,
        summary_lines=summary_lines,
    )
    if configs_dir is not None:
        src = Path(configs_dir)
        if src.is_dir():
            dst = out / "changes" / "configs"
            dst.mkdir(parents=True, exist_ok=True)
            for cfg in sorted(src.glob("*.yaml")):
                shutil.copy2(cfg, dst / cfg.name)
            for cfg in sorted(src.glob("*.yml")):
                shutil.copy2(cfg, dst / cfg.name)
            change_info["configs"] = "changes/configs"

    _write_manifest(
        out,
        scope="workspace",
        source_roots=[str(root)],
        include=kinds,
        records=all_records,
        excluded=excluded_total,
        extra={
            "experiments": index_rows,
            "change_info": change_info,
            "excluded_trial_details": {} if include_trial_details else bulk_total,
        },
    )
    _write_checksums(out, all_records)
    return out


def write_figures_index(figures_dir: str | Path, *, run_id: str = "") -> Path | None:
    """Write a tiny ``index.json`` for generated figures, if any exist."""
    root = Path(figures_dir)
    if not root.is_dir():
        return None
    files = sorted(
        p for p in root.iterdir() if p.is_file() and classify_artifact(p) == "figures"
    )
    if not files:
        return None
    payload = {
        "run_id": run_id,
        "figures": [
            {"name": p.name, "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files
        ],
    }
    path = root / "index.json"
    atomic_write_text(path, strict_json_dumps(payload))
    return path
