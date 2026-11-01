"""Stable experiment identity: code, environment, data content and strict JSON.

V1.1 makes every cached execution traceable to *everything* that can change a
number:

* the code revision (commit + a hash of the dirty diff),
* the runtime versions that affect numerics,
* the full content identity of the data (Merkle root over per-file SHA-256),
* the label expression / availability window and preprocessing version,
* the fold's actual dates and phase,
* the candidate/control/baseline identity, model and training budget, seed.

None of these may be replaced by a timestamp, a file count or a sample of
files.  The large qlib store is handled with a persisted, verifiable content
manifest: every file has a SHA-256; a cached manifest is only trusted after its
size/mtime entries still match, and any changed file is re-hashed.  A
``verify=True`` scan re-hashes everything.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from predicatelab.core.hashing import canonical_json, sha256_file, sha256_obj

CONTENT_MANIFEST_VERSION = "1"


# --------------------------------------------------------------------------- code
def _run_git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=30, check=False
        )
        if out.returncode != 0:
            return None
        return out.stdout
    except Exception:
        return None


def code_version(repo_root: str | Path) -> dict[str, Any]:
    """Commit + dirty state of the working tree.

    The dirty fingerprint hashes ``git diff HEAD`` *and* the porcelain status so
    that untracked files are visible in the identity too.  This is deliberately
    conservative: any local edit invalidates all cached executions unless the
    operator explicitly acknowledges drift.
    """
    repo = Path(repo_root)
    commit = (_run_git(repo, "rev-parse", "HEAD") or "").strip() or "unknown"
    status = _run_git(repo, "status", "--porcelain=v1") or ""
    diff = _run_git(repo, "diff", "HEAD", "--binary") or ""
    dirty = bool(status.strip())
    fingerprint = sha256_obj({"commit": commit, "status": status, "diff": diff})
    return {
        "commit": commit,
        "dirty": dirty,
        "dirty_fingerprint": fingerprint,
        "status_lines": len([ln for ln in status.splitlines() if ln.strip()]),
    }


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "pydantic", "sklearn", "torch", "qlib", "scipy"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def environment_fingerprint() -> str:
    return sha256_obj(environment_versions())


# ---------------------------------------------------------------------- strict json
def sanitise_json(value: Any) -> tuple[Any, list[str]]:
    """Recursively replace NaN/Inf with ``None`` and record where.

    Returns ``(clean_value, null_paths)``.  The engine stores ``null_paths`` next
    to the result so a missing statistic is never silently read as zero.
    """
    nulls: list[str] = []

    def walk(obj: Any, path: str) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                nulls.append(path or "value")
                return None
            return obj
        if isinstance(obj, dict):
            return {str(k): walk(v, f"{path}.{k}" if path else str(k)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        return obj

    return walk(value, ""), nulls


def strict_json_dumps(obj: Any) -> str:
    clean, _ = sanitise_json(obj)
    return json.dumps(clean, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via a same-directory temp file + fsync + os.replace."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


# ------------------------------------------------------------------ content identity
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def full_content_manifest(
    root: str | Path,
    files: Iterable[str | Path],
    *,
    cache_path: str | Path | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """Full SHA-256 content manifest with a Merkle root.

    ``cache_path`` makes repeated scans cheap: unchanged ``(size, mtime_ns)``
    entries reuse their recorded digest; changed/new files are re-hashed.  The
    cache itself is content-addressed and carries the Merkle root, so a
    corrupted cache is detected and rebuilt.  ``verify=True`` re-hashes every
    file and is the authoritative check.
    """
    root = Path(root)
    rel_files = sorted(str(Path(f)) for f in files)
    cached: dict[str, dict[str, Any]] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists() and not verify:
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            fields = ("version", "root", "n_files", "n_missing", "merkle_root", "files")
            expected = sha256_obj({k: payload.get(k) for k in fields})
            valid = (
                payload.get("version") == CONTENT_MANIFEST_VERSION
                and payload.get("root") == str(root)
                and isinstance(payload.get("files"), list)
                and payload.get("manifest_hash") == expected
            )
            if valid:
                cached = {e["path"]: e for e in payload["files"]}
        except Exception:
            cached = {}

    entries: list[dict[str, Any]] = []
    rehashed = 0
    for rel in rel_files:
        full = root / rel
        if not full.exists():
            entries.append({"path": rel, "exists": False})
            continue
        st = full.stat()
        prev = cached.get(rel)
        if (
            not verify
            and prev
            and prev.get("exists")
            and prev.get("size") == st.st_size
            and prev.get("mtime_ns") == st.st_mtime_ns
            and prev.get("sha256")
        ):
            digest = prev["sha256"]
        else:
            digest = sha256_file(full)
            rehashed += 1
        entries.append({
            "path": rel,
            "exists": True,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "sha256": digest,
        })

    leaf_hashes = [sha256_obj(e) for e in entries]
    merkle_root = _merkle_root(leaf_hashes)
    manifest = {
        "version": CONTENT_MANIFEST_VERSION,
        "root": str(root),
        "n_files": len(entries),
        "n_missing": sum(1 for e in entries if not e.get("exists")),
        "n_rehashed": rehashed,
        "generated_at": _now(),
        "merkle_root": merkle_root,
        "files": entries,
    }
    manifest["manifest_hash"] = sha256_obj({k: manifest[k] for k in
                                            ("version", "root", "n_files", "n_missing", "merkle_root", "files")})
    if cache_file:
        atomic_write_text(cache_file, strict_json_dumps(manifest))
    return manifest


def _merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(bytes.fromhex(h)).hexdigest() for h in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(bytes.fromhex(a + b)).hexdigest()
                 for a, b in zip(level[0::2], level[1::2])]
    return level[0]


def fold_content_hash(fold: Any) -> str:
    """Identity of a fold including actual dates, metadata and phase-relevant fields."""
    dump = fold.model_dump(mode="json") if hasattr(fold, "model_dump") else dict(fold)
    return sha256_obj(dump)


def label_spec_hash(contract: Any) -> str:
    payload = {
        "label_definition": getattr(contract, "label_definition", ""),
        "label_window": getattr(contract, "label_window", {}),
        "time_key": getattr(contract, "time_key", ""),
        "frequency": getattr(contract, "frequency", ""),
    }
    return sha256_obj(payload)


def experiment_identity(
    *,
    domain: str,
    snapshot_hash: str,
    data_manifest_hash: str,
    label_spec: str,
    protocol_hash: str,
    code: dict[str, Any],
    environment: dict[str, Any],
    model_spec: dict[str, Any],
    preprocessing_version: str = "v1",
) -> dict[str, Any]:
    return {
        "domain": domain,
        "snapshot_hash": snapshot_hash,
        "data_manifest_hash": data_manifest_hash,
        "label_spec_hash": label_spec,
        "protocol_hash": protocol_hash,
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "code_fingerprint": code.get("dirty_fingerprint"),
        "environment": environment,
        "model_spec": model_spec,
        "preprocessing_version": preprocessing_version,
    }


def identity_hash(identity: dict[str, Any]) -> str:
    return sha256_obj(identity)


def canonical_dumps(obj: Any) -> str:
    return canonical_json(obj)
