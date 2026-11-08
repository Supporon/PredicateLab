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

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from predicatelab.core.hashing import sha256_obj

ACTIVE_STATUSES = ("intent", "claimed", "running", "completed", "failed", "interrupted")
CONSUMED_STATUSES = ("claimed", "running", "completed", "failed", "interrupted")

# Immutable historical exposure seeds (CSI300 2024-01-01..2025-08-31 holdout
# and f2023 2023-01-01..2023-12-31 confirmation fold already exposed in V1.0/V1.1).
# Creating a fresh SQLite DB in a new subdirectory cannot bypass these records.
BUILTIN_CONSUMED_HOLDOUTS: tuple[dict[str, Any], ...] = (
    {
        "access_id": "hoa_builtin_csi300_2024_2025",
        "holdout_key": "builtin_historical_qlib_cn_csi300_20240101_20250831",
        "study_id": "task_tsil_csi300_historical_v10_v11",
        "run_id": "historical_v10_v11_exposed",
        "candidate_id": "cand_be80bc12ff574b3d",
        "protocol_hash": "historical_exposed",
        "snapshot_hash": "snap_14111cf557f10859",
        "dataset_id": "qlib-cn-csi300",
        "label_spec_hash": "historical_ref_close_m2_m1",
        "test_start": "2024-01-01",
        "test_end": "2025-08-31",
        "purpose": "confirmatory_final",
        "status": "completed",
        "origin": "builtin_historical_seed",
        "manifest_hash": "historical_exposed_manifest",
        "code_fingerprint": "historical_exposed",
        "details": '{"note": "CSI300 2024-01-01..2025-08-31 holdout consumed during V1.0/V1.1 development"}',
        "created_at": "2026-04-12T00:00:00+00:00",
        "updated_at": "2026-04-12T00:00:00+00:00",
    },
)

BUILTIN_CONSUMED_CONFIRMATIONS: tuple[dict[str, Any], ...] = (
    {
        "access_id": "cfa_builtin_csi300_f2023",
        "confirmation_key": "builtin_historical_qlib_cn_csi300_f2023_20230101_20231231",
        "study_id": "task_tsil_csi300_historical_v11",
        "run_id": "historical_v11_quick_exposed",
        "family_id": "historical_v11_quick_family",
        "manifest_hash": "historical_v11_quick_manifest",
        "protocol_hash": "historical_exposed",
        "snapshot_hash": "snap_14111cf557f10859",
        "dataset_id": "qlib-cn-csi300",
        "label_spec_hash": "historical_ref_close_m2_m1",
        "fold_id": "f2023",
        "valid_start": "2023-01-01",
        "valid_end": "2023-12-31",
        "purpose": "independent_confirmation",
        "status": "completed",
        "code_fingerprint": "historical_exposed",
        "details": '{"note": "CSI300 f2023 2023-01-01..2023-12-31 confirmation fold exposed during V1.1 smoke/quick runs"}',
        "created_at": "2026-04-12T00:00:00+00:00",
        "updated_at": "2026-04-12T00:00:00+00:00",
    },
)


def seed_builtin_exposures(conn: sqlite3.Connection) -> None:
    """Ensure immutable historical exposure rows exist in a ledger connection."""
    if os.environ.get("PREDICATELAB_IGNORE_BUILTIN_EXPOSURE") == "1":
        return
    with conn:
        for h in BUILTIN_CONSUMED_HOLDOUTS:
            conn.execute(
                """INSERT OR IGNORE INTO holdout_access(
                       access_id, holdout_key, study_id, run_id, candidate_id, protocol_hash,
                       snapshot_hash, dataset_id, label_spec_hash, test_start, test_end,
                       purpose, status, origin, manifest_hash, code_fingerprint, details,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    h["access_id"], h["holdout_key"], h["study_id"], h["run_id"],
                    h["candidate_id"], h["protocol_hash"], h["snapshot_hash"],
                    h["dataset_id"], h["label_spec_hash"], h["test_start"], h["test_end"],
                    h["purpose"], h["status"], h["origin"], h["manifest_hash"],
                    h["code_fingerprint"], h["details"], h["created_at"], h["updated_at"],
                ),
            )
        for c in BUILTIN_CONSUMED_CONFIRMATIONS:
            conn.execute(
                """INSERT OR IGNORE INTO confirmation_access(
                       access_id, confirmation_key, study_id, run_id, family_id, manifest_hash,
                       protocol_hash, snapshot_hash, dataset_id, label_spec_hash, fold_id,
                       valid_start, valid_end, purpose, status, code_fingerprint, details,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    c["access_id"], c["confirmation_key"], c["study_id"], c["run_id"],
                    c["family_id"], c["manifest_hash"], c["protocol_hash"],
                    c["snapshot_hash"], c["dataset_id"], c["label_spec_hash"],
                    c["fold_id"], c["valid_start"], c["valid_end"],
                    c["purpose"], c["status"], c["code_fingerprint"],
                    c["details"], c["created_at"], c["updated_at"],
                ),
            )


def _match_builtin_holdout(dataset_id: str, test_start: str, test_end: str) -> dict[str, Any] | None:
    if os.environ.get("PREDICATELAB_IGNORE_BUILTIN_EXPOSURE") == "1":
        return None
    if not test_start or not test_end:
        return None
    for h in BUILTIN_CONSUMED_HOLDOUTS:
        if h["dataset_id"] == dataset_id and h["test_start"] <= test_end and h["test_end"] >= test_start:
            return dict(h)
    return None


def _match_builtin_confirmation(dataset_id: str, valid_start: str, valid_end: str) -> dict[str, Any] | None:
    if os.environ.get("PREDICATELAB_IGNORE_BUILTIN_EXPOSURE") == "1":
        return None
    if not valid_start or not valid_end:
        return None
    for c in BUILTIN_CONSUMED_CONFIRMATIONS:
        if c["dataset_id"] == dataset_id and c["valid_start"] <= valid_end and c["valid_end"] >= valid_start:
            return dict(c)
    return None


class HoldoutConsumed(RuntimeError):
    """Raised when a holdout has already been consumed by another study."""


class ConfirmationConsumed(RuntimeError):
    """Raised when a confirmation fold has already been consumed by another study."""


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
        if conflict is None:
            conflict = _match_builtin_holdout(dataset_id, test_start, test_end)
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


def assert_holdout_available(
    conn: sqlite3.Connection,
    *,
    key: str,
    dataset_id: str,
    test_start: str,
    test_end: str,
    run_id: str,
    candidate_id: str,
    manifest_hash: str | None = None,
) -> None:
    """Read-only check raising :class:`HoldoutConsumed` if another study already owns the holdout."""
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
        return
    conflict = conn.execute(
        """SELECT * FROM holdout_access
           WHERE dataset_id=? AND status IN (%s)
             AND (test_start='legacy_unknown' OR test_end='legacy_unknown'
                  OR (test_start <= ? AND test_end >= ?))
           LIMIT 1""" % ",".join("?" for _ in ACTIVE_STATUSES),
        (dataset_id, *ACTIVE_STATUSES, test_end, test_start),
    ).fetchone()
    if conflict is None:
        conflict = _match_builtin_holdout(dataset_id, test_start, test_end)
    if conflict is not None and not (
        conflict["run_id"] == run_id and conflict["candidate_id"] == candidate_id
    ):
        raise HoldoutConsumed(
            f"dataset {dataset_id!r} window {test_start}..{test_end} overlaps an "
            f"already-consumed holdout ({conflict['test_start']}..{conflict['test_end']}, "
            f"origin={conflict['origin']}, status={conflict['status']}); "
            "a fresh confirmatory evaluation is not allowed -- an explicit "
            "exploratory reuse must be recorded instead"
        )


def confirmation_key(
    *,
    dataset_id: str,
    snapshot_hash: str,
    label_spec_hash: str,
    fold_id: str,
    valid_start: str,
    valid_end: str,
    data_manifest_hash: str | None = None,
) -> str:
    """Data-only identity of a confirmation fold window."""
    return sha256_obj({
        "dataset_id": dataset_id,
        "snapshot_hash": snapshot_hash,
        "data_manifest_hash": data_manifest_hash,
        "label_spec_hash": label_spec_hash,
        "fold_id": fold_id,
        "valid_start": valid_start,
        "valid_end": valid_end,
    })


def _confirmation_event(
    conn: sqlite3.Connection, access_id: str, event: str,
    run_id: str = "", family_id: str = "", note: str = "",
) -> None:
    conn.execute(
        """INSERT INTO confirmation_access_events(access_id, event, run_id, family_id, note, created_at)
           VALUES (?,?,?,?,?,?)""",
        (access_id, event, run_id, family_id, note, utcnow()),
    )


def claim_confirmation(
    conn: sqlite3.Connection,
    *,
    key: str,
    study_id: str,
    run_id: str,
    family_id: str,
    manifest_hash: str,
    protocol_hash: str,
    snapshot_hash: str,
    dataset_id: str,
    label_spec_hash: str,
    fold_id: str,
    valid_start: str,
    valid_end: str,
    code_fingerprint: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Claim a confirmation fold window or record an exploratory reuse if already consumed."""
    import json

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM confirmation_access WHERE confirmation_key=? AND status IN (%s)"
            % ",".join("?" for _ in ACTIVE_STATUSES),
            (key, *ACTIVE_STATUSES),
        ).fetchone()
        if row is None and valid_start and valid_end:
            row = conn.execute(
                """SELECT * FROM confirmation_access
                   WHERE dataset_id=? AND status IN (%s)
                     AND valid_start != '' AND valid_end != ''
                     AND valid_start <= ? AND valid_end >= ?
                   LIMIT 1""" % ",".join("?" for _ in ACTIVE_STATUSES),
                (dataset_id, *ACTIVE_STATUSES, valid_end, valid_start),
            ).fetchone()
        if row is None and valid_start and valid_end:
            row = _match_builtin_confirmation(dataset_id, valid_start, valid_end)

        if row is not None:
            same = (row["run_id"] == run_id and row["manifest_hash"] == manifest_hash)
            if same:
                conn.commit()
                return {**dict(row), "retry": True, "reused": False, "prior_exposure": None}
            reuse_id = "cfa_expl_" + sha256_obj({
                "key": key, "run": run_id, "family": family_id, "manifest": manifest_hash,
            })[:20]
            now = utcnow()
            reuse_details = {
                **(details or {}),
                "prior_run_id": row["run_id"],
                "prior_family_id": row["family_id"],
                "prior_manifest_hash": row["manifest_hash"],
                "prior_status": row["status"],
            }
            conn.execute(
                """INSERT OR IGNORE INTO confirmation_access(
                       access_id, confirmation_key, study_id, run_id, family_id, manifest_hash,
                       protocol_hash, snapshot_hash, dataset_id, label_spec_hash, fold_id,
                       valid_start, valid_end, purpose, status, code_fingerprint, details,
                       created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (reuse_id, key, study_id, run_id, family_id, manifest_hash, protocol_hash,
                 snapshot_hash, dataset_id, label_spec_hash, fold_id, valid_start, valid_end,
                 "exploratory_reuse", "exploratory_reuse", code_fingerprint,
                 json.dumps(reuse_details, sort_keys=True, default=str), now, now),
            )
            _confirmation_event(
                conn, reuse_id, "exploratory_reuse", run_id, family_id,
                f"reused confirmation fold already consumed by run={row['run_id']}",
            )
            conn.commit()
            reuse_row = conn.execute(
                "SELECT * FROM confirmation_access WHERE access_id=?", (reuse_id,)
            ).fetchone()
            return {
                **(dict(reuse_row) if reuse_row else {"access_id": reuse_id}),
                "retry": False,
                "reused": True,
                "prior_exposure": dict(row),
            }

        access_id = "cfa_" + sha256_obj({
            "key": key, "run": run_id, "family": family_id, "manifest": manifest_hash,
        })[:20]
        now = utcnow()
        conn.execute(
            """INSERT INTO confirmation_access(
                   access_id, confirmation_key, study_id, run_id, family_id, manifest_hash,
                   protocol_hash, snapshot_hash, dataset_id, label_spec_hash, fold_id,
                   valid_start, valid_end, purpose, status, code_fingerprint, details,
                   created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (access_id, key, study_id, run_id, family_id, manifest_hash, protocol_hash,
             snapshot_hash, dataset_id, label_spec_hash, fold_id, valid_start, valid_end,
             "independent_confirmation", "claimed", code_fingerprint,
             json.dumps(details or {}, sort_keys=True, default=str), now, now),
        )
        _confirmation_event(conn, access_id, "claimed", run_id, family_id, "exclusive confirmation claim")
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM confirmation_access WHERE access_id=?", (access_id,)
        ).fetchone()
        return {**dict(claimed), "retry": False, "reused": False, "prior_exposure": None}
    except Exception:
        conn.rollback()
        raise


def mark_confirmation(
    conn: sqlite3.Connection, access_id: str, status: str,
    *, run_id: str = "", family_id: str = "", note: str = "",
) -> None:
    if status not in {"intent", "claimed", "running", "completed", "failed", "interrupted", "exploratory_reuse"}:
        raise HoldoutLedgerError(f"unknown confirmation status {status!r}")
    with conn:
        conn.execute(
            "UPDATE confirmation_access SET status=?, updated_at=? WHERE access_id=?",
            (status, utcnow(), access_id),
        )
        _confirmation_event(conn, access_id, status, run_id, family_id, note)

