"""Data snapshot helpers.

A snapshot is *immutable identity*, not a copy.  V1.0 never mutates source data;
it records enough metadata to detect that the bytes underneath changed:

* full SHA-256 for small, semantically critical files (calendar, instrument
  lists, configuration);
* ``(relative_path, size, mtime_ns)`` for large binary stores, aggregated into
  a manifest hash.  This is explicitly a metadata-level manifest and is labelled
  as such in ``DataSnapshot.manifest_method``.

Domains that need stronger guarantees can hash full content by setting
``hash_content=True`` for selected files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from predicatelab.core.hashing import sha256_file, sha256_obj


def file_manifest(
    root: str | Path,
    paths: Iterable[str | Path],
    *,
    hash_content: bool = False,
) -> tuple[str, dict]:
    root = Path(root)
    entries = []
    for p in sorted(Path(x) for x in paths):
        full = p if p.is_absolute() else root / p
        if not full.exists():
            entries.append({"path": str(p), "exists": False})
            continue
        st = full.stat()
        entry = {
            "path": str(p),
            "exists": True,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
        if hash_content:
            entry["sha256"] = sha256_file(full)
        entries.append(entry)
    digest = sha256_obj(entries)
    method = "sha256-file-list" if hash_content else "metadata-manifest(size+mtime_ns)"
    return digest, {"method": method, "root": str(root), "n_files": len(entries), "files": entries}


def directory_manifest(
    root: str | Path, *, patterns: tuple[str, ...] = ("**/*",), limit: int | None = None
) -> tuple[str, dict]:
    root = Path(root)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    files = [f for f in files if f.is_file()]
    if limit is not None:
        files = files[:limit]
    return file_manifest(root, files)


def content_hash_of_files(paths: Iterable[str | Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(str(p).encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()


def assess_common(checks: dict[str, dict]) -> dict:
    """Aggregate check dicts into {status, checks, warnings, blockers}.

    Each check is ``{"ok": bool, "severity": "warn"|"block", "detail": str}``.
    """
    warnings, blockers = [], []
    clean: dict[str, dict] = {}
    for name, chk in checks.items():
        c = dict(chk)
        # numpy scalars are not JSON/Pydantic-serialisable
        if "ok" in c:
            c["ok"] = bool(c["ok"])
        clean[name] = c
        if c.get("ok", True):
            continue
        sev = c.get("severity", "warn")
        msg = f"{name}: {c.get('detail', '')}"
        (blockers if sev == "block" else warnings).append(msg)
    status = "block" if blockers else ("warn" if warnings else "ok")
    return {"status": status, "checks": clean, "warnings": warnings, "blockers": blockers}
