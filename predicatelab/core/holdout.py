"""One-time holdout access ledger.

A holdout is a *consumable scientific resource*, not a fold that can be read
again whenever a new ``run_id`` appears.  V1.1 therefore keeps a global ledger
whose identity is the data, not the study metadata:

``holdout_key = H(dataset_id, snapshot_hash, label_spec_hash, test_start, test_end)``

Changing ``run_id``, candidate, protocol, manifest timestamp or artifact path
does **not** change this key.  SQLite enforces a partial unique index over the
active statuses, and the claim is made inside ``BEGIN IMMEDIATE`` so two
concurrent finalizers cannot both win.

Status machine::

    intent -> claimed -> running -> completed
                              \\-> failed
                              \\-> interrupted   (process died; still consumed)

``failed``/``interrupted`` remain *consumed*: the data may already have been
read.  Only a retry of the exact same ``(run_id, candidate_id, manifest_hash)``
is allowed, and even then it reuses the same frozen manifest instead of
creating a new claim.

``exploratory_reuse`` rows are recorded for analyses that deliberately reuse
already-consumed data.  They can never be reported as fresh confirmatory
evidence.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from predicatelab.core.hashing import sha256_obj

ACTIVE_STATUSES = ("intent", "claimed", "running", "completed", "failed", "interrupted")
CONSUMED_STATUSES = ("claimed", "running", "completed", "failed", "interrupted")


class HoldoutConsumed(RuntimeError):
    """Raised when a holdout has already been consumed by another study."""


class HoldoutLedgerError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def holdout_key(
    *,
    dataset_id: str,
    snapshot_hash: str,
    label_spec_hash: str,
    test_start: str,
    test_end: str,
    data_manifest_hash: str | None = None,
) -> str:
    """Data-only identity.  Study/run/candidate/protocol are deliberately absent."""
    return sha256_obj({
        "dataset_id": dataset_id,
        "snapshot_hash": snapshot_hash,
        "data_manifest_hash": data_manifest_hash,
        "label_spec_hash": label_spec_hash,
        "test_start": test_start,
        "test_end": test_end,
    })


def claim_holdout(
    conn: sqlite3.Connection,
    *,
    key: str,
    study_id: str,
    run_id: str,
    candidate_id: str,
    protocol_hash: str,
    snapshot_hash: str,
    dataset_id: str,
    label_spec_hash: str,
    test_start: str,
    test_end: str,
    purpose: str = "confirmatory_final",
    manifest_hash: str | None = None,
    code_fingerprint: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically claim a holdout.

    Returns the ledger row.  Raises :class:`HoldoutConsumed` when another study
    or another candidate already owns the only confirmatory access.  A retry of
    the same ``(run_id, candidate_id, manifest_hash)`` returns the existing row
    with ``"retry": True`` and does not arm a second test.
    """
    import json

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM holdout_access WHERE holdout_key=? AND status IN (%s)"
            % ",".join("?" for _ in ACTIVE_STATUSES),
            (key, *ACTIVE_STATUSES),
        ).fetchone()

        if row is not None:
            same = (
                row["run_id"] == run_id
                and row["candidate_id"] == candidate_id
                and (manifest_hash is None or row["manifest_hash"] == manifest_hash)
            )
            if not same:
                raise HoldoutConsumed(
                    f"holdout key {key[:16]} already consumed by run={row['run_id']} "
                    f"candidate={row['candidate_id']} status={row['status']}; "
                    "a fresh confirmatory evaluation is not allowed"
                )
            if row["status"] == "completed":
                conn.commit()
                return {**dict(row), "retry": True, "already_completed": True}
            # Same study + candidate + manifest: retry the same test, never a new one.
            conn.execute(
                "UPDATE holdout_access SET status='claimed', updated_at=?, "
                "details=? WHERE access_id=?",
                (utcnow(), json.dumps(details or {}, sort_keys=True, default=str), row["access_id"]),
            )
            _event(conn, row["access_id"], "retry_claim", run_id, candidate_id, "same run/candidate retry")
            conn.commit()
            out = dict(conn.execute(
                "SELECT * FROM holdout_access WHERE access_id=?", (row["access_id"],)
            ).fetchone())
            out["retry"] = True
            return out

        # Period-level guard: the same dataset dates are consumed even when the
        # snapshot identity changed (e.g. V1.0 sampled manifests vs V1.1 full
        # content manifests, or a historical row whose identity was unprovable).
        # Only exact-key retries above are exempt.
        conflict = conn.execute(
            """SELECT * FROM holdout_access
               WHERE dataset_id=? AND status IN (%s)
                 AND (test_start='legacy_unknown' OR test_end='legacy_unknown'
                      OR (test_start <= ? AND test_end >= ?))
               LIMIT 1""" % ",".join("?" for _ in ACTIVE_STATUSES),
            (dataset_id, *ACTIVE_STATUSES, test_end, test_start),
        ).fetchone()
        if conflict is not None:
            raise HoldoutConsumed(
                f"dataset {dataset_id!r} window {test_start}..{test_end} overlaps an "
                f"already-consumed holdout ({conflict['test_start']}..{conflict['test_end']}, "
                f"origin={conflict['origin']}, status={conflict['status']}); "
                "a fresh confirmatory evaluation is not allowed -- an explicit "
                "exploratory reuse must be recorded instead"
            )

        access_id = "hoa_" + sha256_obj({"key": key, "run": run_id, "cand": candidate_id,
                                         "manifest": manifest_hash})[:20]
        now = utcnow()
        conn.execute(
            """INSERT INTO holdout_access(
                   access_id, holdout_key, study_id, run_id, candidate_id, protocol_hash,
                   snapshot_hash, dataset_id, label_spec_hash, test_start, test_end,
                   purpose, status, manifest_hash, code_fingerprint, details,
                   created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (access_id, key, study_id, run_id, candidate_id, protocol_hash, snapshot_hash,
             dataset_id, label_spec_hash, test_start, test_end, purpose, "claimed",
             manifest_hash, code_fingerprint,
             json.dumps(details or {}, sort_keys=True, default=str), now, now),
        )
        _event(conn, access_id, "claimed", run_id, candidate_id, "exclusive confirmatory claim")
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM holdout_access WHERE access_id=?", (access_id,)
        ).fetchone())
    except Exception:
        conn.rollback()
        raise


def mark_holdout(conn: sqlite3.Connection, access_id: str, status: str,
                 *, run_id: str = "", candidate_id: str = "", note: str = "") -> None:
    if status not in {"intent", "claimed", "running", "completed", "failed", "interrupted", "exploratory_reuse"}:
        raise HoldoutLedgerError(f"unknown holdout status {status!r}")
    with conn:
        conn.execute(
            "UPDATE holdout_access SET status=?, updated_at=? WHERE access_id=?",
            (status, utcnow(), access_id),
        )
        _event(conn, access_id, status, run_id, candidate_id, note)


def record_exploratory_reuse(conn: sqlite3.Connection, **kwargs: Any) -> str:
    """Record a deliberate reuse of consumed data; it is never confirmatory."""
    import json
    access_id = "hoa_expl_" + sha256_obj(kwargs)[:20]
    now = utcnow()
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO holdout_access(
                   access_id, holdout_key, study_id, run_id, candidate_id, protocol_hash,
                   snapshot_hash, dataset_id, label_spec_hash, test_start, test_end,
                   purpose, status, manifest_hash, code_fingerprint, details,
                   created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (access_id, kwargs["key"], kwargs["study_id"], kwargs["run_id"],
             kwargs["candidate_id"], kwargs["protocol_hash"], kwargs["snapshot_hash"],
             kwargs["dataset_id"], kwargs["label_spec_hash"], kwargs["test_start"],
             kwargs["test_end"], "exploratory_reuse", "exploratory_reuse",
             kwargs.get("manifest_hash"), kwargs.get("code_fingerprint"),
             json.dumps(kwargs.get("details") or {}, sort_keys=True, default=str), now, now),
        )
        _event(conn, access_id, "exploratory_reuse", kwargs["run_id"],
               kwargs["candidate_id"], "reuse of consumed holdout marked exploratory")
    return access_id


def _event(conn: sqlite3.Connection, access_id: str, event: str,
           run_id: str = "", candidate_id: str = "", note: str = "") -> None:
    conn.execute(
        """INSERT INTO holdout_access_events(access_id, event, run_id, candidate_id, note, created_at)
           VALUES (?,?,?,?,?,?)""",
        (access_id, event, run_id, candidate_id, note, utcnow()),
    )
