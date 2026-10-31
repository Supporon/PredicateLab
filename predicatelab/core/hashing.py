"""Content-addressed identities.

Every object that participates in deduplication (predicate templates, data
snapshots, protocol configurations) is identified by a SHA-256 digest of its
canonical JSON representation.  Canonicalisation is deliberately explicit so
that the same semantic object always hashes to the same value regardless of
dict ordering or float formatting.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def _canonicalise(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _canonicalise(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    if isinstance(obj, float):
        # -0.0 and 0.0 are the same semantic number
        return 0.0 if obj == 0.0 else obj
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _canonicalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def short_id(prefix: str, digest: str, length: int = 16) -> str:
    return f"{prefix}_{digest[:length]}"
