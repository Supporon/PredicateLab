"""SQLite state store with schema migration and experiment state machines.

V1.1 guarantees:

* schema changes are additive and versioned (``schema_migrations``); historical
  rows are preserved and missing fields are marked ``legacy_*`` rather than
  invented;
* every trial attempt is logged immutably (``trial_attempts``) and the canonical
  ``trials`` row is only trusted when its result hash verifies;
* a round row's existence does **not** mean completion -- only
  ``rounds.status='completed'`` does;
* the holdout ledger (``holdout_access``) has a partial unique index over active
  claims, so concurrent finalizers cannot both consume a holdout.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from predicatelab.core.hashing import sha256_obj
from predicatelab.core.holdout import (
    ACTIVE_STATUSES,
    assert_holdout_available as _assert_holdout_available,
    claim_confirmation as _claim_confirmation,
    claim_holdout as _claim_holdout,
    confirmation_key as _confirmation_key,
    holdout_key as _holdout_key,
    mark_confirmation as _mark_confirmation,
    mark_holdout as _mark_holdout,
)
from predicatelab.core.identity import sanitise_json, strict_json_dumps
from predicatelab.core.schema import (
    BoundCandidate,
    DataSnapshot,
    EvidenceRecord,
    FinalEvaluation,
    PredicateTemplate,
    ResearchTask,
    RoundRecord,
    TrialRecord,
)

SCHEMA_VERSION = 3
MIGRATION_HISTORY = {
    2: "v1.1 identity, state machines, holdout ledger",
    3: "v1.1.1 per-seed trial/attempt recording",
}


def _dump(obj: Any) -> str:
    clean, _ = sanitise_json(obj)
    return json.dumps(clean, ensure_ascii=True, sort_keys=True, allow_nan=False, default=str)


def _loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    manifest_method TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    path TEXT,
    details TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    template_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    domain TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    descriptor TEXT NOT NULL,
    rationale TEXT,
    provenance TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    template_hash TEXT NOT NULL,
    domain TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    descriptor TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    bound_scope TEXT,
    probe_signature TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(template_hash, dataset_id, snapshot_hash)
);
CREATE TABLE IF NOT EXISTS fitted_instances (
    instance_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    fitted_params TEXT NOT NULL,
    fit_data_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT DEFAULT 'discovery',
    config TEXT,
    artifacts TEXT,
    snapshot_hash TEXT,
    protocol_hash TEXT,
    code_fingerprint TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    generator TEXT NOT NULL,
    parent_round_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    status_origin TEXT NOT NULL DEFAULT 'v1.1',
    n_proposed INTEGER DEFAULT 0,
    n_bound INTEGER DEFAULT 0,
    n_validated INTEGER DEFAULT 0,
    n_rejected INTEGER DEFAULT 0,
    n_insufficient INTEGER DEFAULT 0,
    summary TEXT,
    error TEXT DEFAULT '',
    started_at TEXT DEFAULT '',
    ended_at TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(run_id, round_index)
);
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT PRIMARY KEY,
    run_id TEXT,
    candidate_id TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT DEFAULT 'unknown',
    seed INTEGER,
    snapshot_hash TEXT DEFAULT '',
    identity_hash TEXT DEFAULT '',
    attempt INTEGER DEFAULT 1,
    result_hash TEXT DEFAULT '',
    predictions_sha256 TEXT DEFAULT '',
    null_paths TEXT DEFAULT '[]',
    metrics TEXT,
    metric_series TEXT,
    fitted_params TEXT,
    predictions_path TEXT,
    duration_s REAL DEFAULT 0,
    error TEXT,
    worker TEXT,
    started_at TEXT DEFAULT '',
    finished_at TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, fold_id, protocol_hash, config_hash)
);
CREATE TABLE IF NOT EXISTS trial_attempts (
    attempt_id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    run_id TEXT,
    candidate_id TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    phase TEXT DEFAULT '',
    seed INTEGER,
    status TEXT NOT NULL,
    result_hash TEXT DEFAULT '',
    metrics TEXT,
    error TEXT,
    worker TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT,
    candidate_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'legacy',
    state TEXT NOT NULL DEFAULT 'legacy_unknown',
    seed INTEGER,
    status TEXT NOT NULL,
    effect REAL,
    ci_low REAL,
    ci_high REAL,
    p_value REAL,
    q_value REAL,
    n_folds INTEGER DEFAULT 0,
    stability REAL,
    metrics TEXT,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    round_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS final_evals (
    final_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    candidate_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    holdout_key TEXT DEFAULT '',
    access_id TEXT DEFAULT '',
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'final',
    state TEXT NOT NULL DEFAULT 'legacy_unknown',
    exploratory INTEGER DEFAULT 0,
    result_hash TEXT DEFAULT '',
    metrics TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, protocol_hash, manifest_hash)
);
CREATE TABLE IF NOT EXISTS frozen_families (
    run_id TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL,
    task_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    data_manifest_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdout_access (
    access_id TEXT PRIMARY KEY,
    holdout_key TEXT NOT NULL,
    study_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    label_spec_hash TEXT NOT NULL,
    test_start TEXT NOT NULL,
    test_end TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'v1.1',
    manifest_hash TEXT,
    code_fingerprint TEXT,
    details TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(access_id)
);
CREATE TABLE IF NOT EXISTS holdout_access_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    access_id TEXT NOT NULL,
    event TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    candidate_id TEXT DEFAULT '',
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS confirmation_access (
    access_id TEXT PRIMARY KEY,
    confirmation_key TEXT NOT NULL,
    study_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    label_spec_hash TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    valid_start TEXT NOT NULL,
    valid_end TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT NOT NULL,
    code_fingerprint TEXT,
    details TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(access_id)
);
CREATE TABLE IF NOT EXISTS confirmation_access_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    access_id TEXT NOT NULL,
    event TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    family_id TEXT DEFAULT '',
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trials_candidate ON trials(candidate_id, protocol_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_round ON evidence(round_id);
CREATE INDEX IF NOT EXISTS idx_evidence_candidate ON evidence(candidate_id);
CREATE INDEX IF NOT EXISTS idx_attempts_trial ON trial_attempts(trial_id);
CREATE INDEX IF NOT EXISTS idx_holdout_key ON holdout_access(holdout_key);
CREATE UNIQUE INDEX IF NOT EXISTS ux_holdout_active
    ON holdout_access(holdout_key)
    WHERE status IN ('intent','claimed','running','completed','failed','interrupted');
CREATE INDEX IF NOT EXISTS idx_confirmation_key ON confirmation_access(confirmation_key);
CREATE UNIQUE INDEX IF NOT EXISTS ux_confirmation_active
    ON confirmation_access(confirmation_key)
    WHERE status IN ('intent','claimed','running','completed','failed','interrupted');
"""

_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdout_access (
    access_id TEXT PRIMARY KEY,
    holdout_key TEXT NOT NULL,
    study_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    label_spec_hash TEXT NOT NULL,
    test_start TEXT NOT NULL,
    test_end TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'v1.1',
    manifest_hash TEXT,
    code_fingerprint TEXT,
    details TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(access_id)
);
CREATE TABLE IF NOT EXISTS holdout_access_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    access_id TEXT NOT NULL,
    event TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    candidate_id TEXT DEFAULT '',
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS confirmation_access (
    access_id TEXT PRIMARY KEY,
    confirmation_key TEXT NOT NULL,
    study_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    label_spec_hash TEXT NOT NULL,
    fold_id TEXT NOT NULL,
    valid_start TEXT NOT NULL,
    valid_end TEXT NOT NULL,
    purpose TEXT NOT NULL,
    status TEXT NOT NULL,
    code_fingerprint TEXT,
    details TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(access_id)
);
CREATE TABLE IF NOT EXISTS confirmation_access_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    access_id TEXT NOT NULL,
    event TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    family_id TEXT DEFAULT '',
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_holdout_key ON holdout_access(holdout_key);
CREATE UNIQUE INDEX IF NOT EXISTS ux_holdout_active
    ON holdout_access(holdout_key)
    WHERE status IN ('intent','claimed','running','completed','failed','interrupted');
CREATE INDEX IF NOT EXISTS idx_confirmation_key ON confirmation_access(confirmation_key);
CREATE UNIQUE INDEX IF NOT EXISTS ux_confirmation_active
    ON confirmation_access(confirmation_key)
    WHERE status IN ('intent','claimed','running','completed','failed','interrupted');
"""

# Columns added in V1.1 for databases created by V1.0.
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "runs": {
        "stage": "TEXT DEFAULT 'discovery'",
        "snapshot_hash": "TEXT",
        "protocol_hash": "TEXT",
        "code_fingerprint": "TEXT",
        "updated_at": "TEXT",
    },
    "rounds": {
        "status": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
        "status_origin": "TEXT NOT NULL DEFAULT 'legacy'",
        "error": "TEXT DEFAULT ''",
        "started_at": "TEXT DEFAULT ''",
        "ended_at": "TEXT DEFAULT ''",
    },
    "trials": {
        "phase": "TEXT DEFAULT 'legacy_unknown'",
        "seed": "INTEGER",
        "snapshot_hash": "TEXT DEFAULT ''",
        "identity_hash": "TEXT DEFAULT ''",
        "attempt": "INTEGER DEFAULT 1",
        "result_hash": "TEXT DEFAULT ''",
        "predictions_sha256": "TEXT DEFAULT ''",
        "null_paths": "TEXT DEFAULT '[]'",
        "started_at": "TEXT DEFAULT ''",
        "finished_at": "TEXT DEFAULT ''",
    },
    "trial_attempts": {
        "seed": "INTEGER",
    },
    "evidence": {
        "stage": "TEXT NOT NULL DEFAULT 'legacy'",
        "state": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
        "seed": "INTEGER",
    },
    "final_evals": {
        "run_id": "TEXT DEFAULT ''",
        "holdout_key": "TEXT DEFAULT ''",
        "access_id": "TEXT DEFAULT ''",
        "stage": "TEXT NOT NULL DEFAULT 'legacy'",
        "state": "TEXT NOT NULL DEFAULT 'legacy_unknown'",
        "exploratory": "INTEGER DEFAULT 0",
        "result_hash": "TEXT DEFAULT ''",
    },
}


def utcnow_value() -> str:
    from predicatelab.core.schema import utcnow

    return utcnow().isoformat()


def result_digest(payload: dict[str, Any]) -> str:
    keys = ("status", "metrics", "metric_series", "fitted_params", "predictions_path",
            "predictions_sha256", "null_paths")
    return sha256_obj({k: payload.get(k) for k in keys})


class StateStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        migrate: bool = True,
        global_ledger_path: str | Path | None = None,
    ):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=60.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        if migrate:
            self._migrate()
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

        self._ledger_conn: sqlite3.Connection | None = None
        ledger_env = os.environ.get("PREDICATELAB_GLOBAL_LEDGER")
        ledger_target: Path | None = None
        if global_ledger_path is not None:
            ledger_target = Path(global_ledger_path).resolve()
        elif ledger_env:
            ledger_target = Path(ledger_env).resolve()
        elif str(self.path) != ":memory:" and self.path.name != ".predicatelab_global_ledger.sqlite":
            ledger_target = self.path.resolve().parent / ".predicatelab_global_ledger.sqlite"
        if ledger_target is not None and ledger_target != self.path.resolve():
            ledger_target.parent.mkdir(parents=True, exist_ok=True)
            self._ledger_conn = sqlite3.connect(str(ledger_target), timeout=60.0, isolation_level=None)
            self._ledger_conn.row_factory = sqlite3.Row
            self._ledger_conn.execute("PRAGMA journal_mode=WAL")
            self._ledger_conn.executescript(_LEDGER_SCHEMA)

    # ------------------------------------------------------------------ basics
    def close(self) -> None:
        if self._ledger_conn is not None:
            try:
                self._ledger_conn.close()
            except Exception:
                pass
            self._ledger_conn = None
        self._conn.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    # --------------------------------------------------------------- migration
    def _columns(self, table: str) -> set[str]:
        return {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}

    def _migrate(self) -> None:
        from predicatelab.core.schema import utcnow

        changed = False
        for table, cols in _ADDED_COLUMNS.items():
            existing = self._columns(table)
            for name, decl in cols.items():
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    changed = True
        # Unique/partial indexes that may not exist on a V1.0 database.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_evidence_stage "
            "ON evidence(run_id, round_id, candidate_id, stage) WHERE run_id IS NOT NULL"
        )
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_holdout_active ON holdout_access(holdout_key) "
            "WHERE status IN ('intent','claimed','running','completed','failed','interrupted')"
        )
        for version, description in MIGRATION_HISTORY.items():
            row = self._conn.execute(
                "SELECT 1 FROM schema_migrations WHERE version=?", (version,)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations(version, description, applied_at) "
                    "VALUES (?,?,?)", (version, description, utcnow().isoformat()),
                )
                changed = True
        if changed:
            self._backfill_legacy_columns()

    def _backfill_legacy_columns(self) -> None:
        """Annotate historical rows; never invent confirmatory status."""
        from predicatelab.core.schema import utcnow

        now = utcnow().isoformat()
        # rounds: only rows still carrying the migrated legacy marker
        rounds = self._conn.execute(
            "SELECT r.round_id, r.run_id, r.status_origin, "
            "  EXISTS(SELECT 1 FROM feedback f WHERE f.round_id=r.round_id) AS has_feedback, "
            "  (SELECT status FROM runs u WHERE u.run_id=r.run_id) AS run_status "
            "FROM rounds r"
        ).fetchall()
        for r in rounds:
            if r["status_origin"] != "legacy":
                continue
            completed = bool(r["has_feedback"]) or r["run_status"] in ("search_complete",)
            status = "completed" if completed else "legacy_unknown"
            self._conn.execute(
                "UPDATE rounds SET status=?, status_origin=? WHERE round_id=?",
                (status, "backfill_feedback" if completed else "legacy", r["round_id"]),
            )
        # trials: legacy rows stay explicitly legacy.  No result hash is
        # invented for them: V1.0 never stored the fields the hash covers, so a
        # fabricated digest would either never verify or, worse, verify numbers
        # under an incomplete identity.
        self._conn.execute(
            "UPDATE trials SET phase='legacy_unknown' WHERE phase IS NULL"
        )
        # evidence: legacy rows are explicitly not confirmation
        ev = self._conn.execute(
            "SELECT evidence_id, status FROM evidence WHERE stage='legacy' AND state='legacy_unknown'"
        ).fetchall()
        for row in ev:
            state = {
                "validated": "exploratory_positive",
                "rejected": "rejected",
                "insufficient_evidence": "insufficient",
            }.get(row["status"], "executable")
            self._conn.execute("UPDATE evidence SET state=? WHERE evidence_id=?", (state, row["evidence_id"]))
        # final evals: legacy, never independently_confirmed
        self._conn.execute(
            "UPDATE final_evals SET stage='legacy', state='legacy_unknown' "
            "WHERE stage='legacy' AND state='legacy_unknown'"
        )
        for row in self._conn.execute("SELECT final_id, notes FROM final_evals WHERE stage='legacy'").fetchall():
            note = (row["notes"] or "")
            if "V1.1 legacy" not in note:
                self._conn.execute(
                    "UPDATE final_evals SET notes=? WHERE final_id=?",
                    ((note + "; V1.1 legacy: stage unprovable, not re-labelled confirmed").strip("; "),
                     row["final_id"]),
                )
        self._conn.execute("UPDATE runs SET updated_at=? WHERE updated_at IS NULL", (now,))

    # ------------------------------------------------------------------- tasks
    def save_task(self, task: ResearchTask) -> str:
        task.ensure_id()
        with self.transaction() as c:
            c.execute(
                "INSERT OR IGNORE INTO tasks(task_id, domain, payload, created_at) VALUES (?,?,?,?)",
                (task.task_id, task.domain, _dump(task.model_dump(mode="json")), task.created_at.isoformat()),
            )
        return task.task_id

    def get_task(self, task_id: str) -> ResearchTask | None:
        row = self._conn.execute("SELECT payload FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return ResearchTask.model_validate_json(row["payload"]) if row else None

    # --------------------------------------------------------------- snapshots
    def save_snapshot(self, snap: DataSnapshot) -> str:
        snap.ensure_id()
        with self.transaction() as c:
            c.execute(
                """INSERT OR REPLACE INTO snapshots(snapshot_id, dataset_id, domain, manifest_method,
                       manifest_hash, path, details, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (snap.snapshot_id, snap.dataset_id, snap.domain, snap.manifest_method,
                 snap.manifest_hash, snap.path, _dump(snap.details), snap.created_at.isoformat()),
            )
        return snap.snapshot_id

    def get_snapshot(self, snapshot_id: str) -> DataSnapshot | None:
        row = self._conn.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if not row:
            return None
        return DataSnapshot(
            snapshot_id=row["snapshot_id"], dataset_id=row["dataset_id"], domain=row["domain"],
            manifest_method=row["manifest_method"], manifest_hash=row["manifest_hash"],
            path=row["path"] or "", details=_loads(row["details"], {}), created_at=row["created_at"],
        )

    # --------------------------------------------------------------- templates
    def add_template(self, tpl: PredicateTemplate) -> tuple[str, bool]:
        tpl.ensure_id()
        existing = self._conn.execute(
            "SELECT template_id FROM templates WHERE content_hash=?", (tpl.content_hash,)
        ).fetchone()
        if existing:
            return existing["template_id"], False
        with self.transaction() as c:
            c.execute(
                """INSERT INTO templates(template_id, content_hash, domain, kind, name,
                       descriptor, rationale, provenance, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (tpl.template_id, tpl.content_hash, tpl.domain, tpl.kind.value, tpl.name,
                 _dump(tpl.descriptor), tpl.rationale,
                 _dump(tpl.provenance.model_dump(mode="json")), tpl.created_at.isoformat()),
            )
        return tpl.template_id, True

    def get_template(self, template_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM templates WHERE template_id=?", (template_id,)).fetchone()
        return dict(row) if row else None

    # -------------------------------------------------------------- candidates
    def add_candidate(self, cand: BoundCandidate) -> tuple[str, bool]:
        cand.ensure_id()
        existing = self._conn.execute(
            "SELECT candidate_id FROM candidates WHERE template_hash=? AND dataset_id=? AND snapshot_hash=?",
            (cand.template_hash, cand.dataset_id, cand.snapshot_hash),
        ).fetchone()
        if existing:
            return existing["candidate_id"], False
        with self.transaction() as c:
            c.execute(
                """INSERT INTO candidates(candidate_id, template_id, template_hash, domain, kind,
                       name, descriptor, dataset_id, snapshot_hash, bound_scope, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (cand.candidate_id, cand.template_id, cand.template_hash, cand.domain, cand.kind.value,
                 cand.name, _dump(cand.descriptor), cand.dataset_id, cand.snapshot_hash,
                 _dump(cand.bound_scope), cand.created_at.isoformat()),
            )
        return cand.candidate_id, True

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["descriptor"] = _loads(d["descriptor"], {})
        d["bound_scope"] = _loads(d["bound_scope"], {})
        return d

    def list_candidates(self, snapshot_hash: str | None = None) -> list[dict[str, Any]]:
        if snapshot_hash:
            rows = self._conn.execute(
                "SELECT * FROM candidates WHERE snapshot_hash=? ORDER BY created_at", (snapshot_hash,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM candidates ORDER BY created_at").fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["descriptor"] = _loads(d["descriptor"], {})
            d["bound_scope"] = _loads(d["bound_scope"], {})
            out.append(d)
        return out

    def set_probe_signature(self, candidate_id: str, signature: list[float]) -> None:
        with self.transaction() as c:
            c.execute("UPDATE candidates SET probe_signature=? WHERE candidate_id=?",
                      (_dump(signature), candidate_id))

    def list_probe_signatures(self, snapshot_hash: str) -> dict[str, list[float]]:
        rows = self._conn.execute(
            "SELECT candidate_id, probe_signature FROM candidates "
            "WHERE snapshot_hash=? AND probe_signature IS NOT NULL", (snapshot_hash,),
        ).fetchall()
        return {r["candidate_id"]: _loads(r["probe_signature"], []) for r in rows}

    # -------------------------------------------------------------------- runs
    def create_run(self, run_id: str, task_id: str, config: dict[str, Any], artifacts: str,
                   *, snapshot_hash: str = "", protocol_hash: str = "",
                   code_fingerprint: str = "") -> None:
        from predicatelab.core.schema import utcnow
        now = utcnow().isoformat()
        with self.transaction() as c:
            c.execute(
                """INSERT INTO runs(run_id, task_id, status, stage, config, artifacts, snapshot_hash,
                       protocol_hash, code_fingerprint, started_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, task_id, "running", "discovery", _dump(config), artifacts,
                 snapshot_hash, protocol_hash, code_fingerprint, now, now),
            )

    def finish_run(self, run_id: str, status: str, *, stage: str | None = None) -> None:
        from predicatelab.core.schema import utcnow
        now = utcnow().isoformat()
        with self.transaction() as c:
            if stage:
                c.execute("UPDATE runs SET status=?, stage=?, updated_at=?, ended_at=? WHERE run_id=?",
                          (status, stage, now, now, run_id))
            else:
                c.execute("UPDATE runs SET status=?, updated_at=?, ended_at=? WHERE run_id=?",
                          (status, now, now, run_id))

    def set_run_status(self, run_id: str, status: str) -> None:
        from predicatelab.core.schema import utcnow
        with self.transaction() as c:
            c.execute("UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                      (status, utcnow().isoformat(), run_id))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = _loads(d["config"], {})
        return d

    def list_runs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM runs ORDER BY started_at")]

    # ------------------------------------------------------------------ rounds
    def save_round(self, rnd: RoundRecord) -> str:
        rnd.ensure_id()
        with self.transaction() as c:
            c.execute(
                """INSERT INTO rounds(round_id, run_id, task_id, round_index, generator,
                       parent_round_id, status, status_origin, n_proposed, n_bound, n_validated,
                       n_rejected, n_insufficient, summary, error, started_at, ended_at, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rnd.round_id, rnd.run_id, rnd.task_id, rnd.round_index, rnd.generator,
                 rnd.parent_round_id, rnd.status, rnd.status_origin, rnd.n_proposed, rnd.n_bound,
                 rnd.n_validated, rnd.n_rejected, rnd.n_insufficient, _dump(rnd.summary),
                 rnd.error, rnd.started_at, rnd.ended_at, rnd.created_at.isoformat()),
            )
        return rnd.round_id

    def get_round(self, run_id: str, round_index: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM rounds WHERE run_id=? AND round_index=?", (run_id, round_index)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["summary"] = _loads(d["summary"], {})
        return d

    def update_round(self, round_id: str, **fields: Any) -> None:
        allowed = {"n_proposed", "n_bound", "n_validated", "n_rejected", "n_insufficient",
                   "summary", "status", "error", "started_at", "ended_at", "parent_round_id"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = [_dump(v) if k == "summary" else v for k, v in updates.items()]
        with self.transaction() as c:
            c.execute(f"UPDATE rounds SET {cols} WHERE round_id=?", (*vals, round_id))

    def start_round(self, round_id: str) -> None:
        from predicatelab.core.schema import utcnow
        with self.transaction() as c:
            c.execute(
                "UPDATE rounds SET status='running', status_origin='v1.1', started_at=?, error='' "
                "WHERE round_id=?", (utcnow().isoformat(), round_id),
            )

    def fail_round(self, round_id: str, status: str, error: str) -> None:
        from predicatelab.core.schema import utcnow
        if status not in {"failed", "interrupted"}:
            raise ValueError(status)
        with self.transaction() as c:
            c.execute(
                "UPDATE rounds SET status=?, error=?, ended_at=? WHERE round_id=?",
                (status, error[:4000], utcnow().isoformat(), round_id),
            )

    def list_rounds(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM rounds WHERE run_id=? ORDER BY round_index", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["summary"] = _loads(d["summary"], {})
            out.append(d)
        return out

    # ------------------------------------------------------------------ trials
    def record_attempt(self, trial: TrialRecord, run_id: str = "") -> str:
        from predicatelab.core.schema import utcnow
        attempt_id = "att_" + sha256_obj({
            "trial": trial.trial_key if hasattr(trial, "trial_key") else "",
            "candidate": trial.candidate_id, "fold": trial.fold_id,
            "protocol": trial.protocol_hash, "config": trial.config_hash,
            "attempt": trial.attempt, "started": trial.started_at,
        })[:20]
        with self.transaction() as c:
            c.execute(
                """INSERT OR REPLACE INTO trial_attempts(attempt_id, trial_id, run_id, candidate_id,
                       fold_id, protocol_hash, config_hash, phase, seed, status, result_hash,
                       metrics, error, worker, started_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (attempt_id, trial.trial_id, run_id, trial.candidate_id, trial.fold_id,
                 trial.protocol_hash, trial.config_hash, trial.phase, trial.seed, trial.status,
                 trial.result_hash, _dump(trial.metrics), trial.error[:4000],
                 _dump(trial.worker), trial.started_at or utcnow().isoformat(),
                 trial.finished_at or utcnow().isoformat()),
            )
        return attempt_id

    def record_trial(self, trial: TrialRecord, run_id: str = "") -> bool:
        """Record the canonical trial row; successful rows win over failures."""
        trial.ensure_id()
        existing = self.get_trial_by_key(
            trial.candidate_id, trial.fold_id, trial.protocol_hash, trial.config_hash
        )
        if existing and existing["status"] == "success" and trial.status != "success":
            return False
        with self.transaction() as c:
            c.execute(
                """INSERT INTO trials(trial_id, run_id, candidate_id, fold_id, protocol_hash,
                       config_hash, status, phase, seed, snapshot_hash, identity_hash, attempt,
                       result_hash, predictions_sha256, null_paths, metrics, metric_series,
                       fitted_params, predictions_path, duration_s, error, worker,
                       started_at, finished_at, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(candidate_id, fold_id, protocol_hash, config_hash) DO UPDATE SET
                       status=excluded.status, phase=excluded.phase, seed=excluded.seed,
                       snapshot_hash=excluded.snapshot_hash, identity_hash=excluded.identity_hash,
                       attempt=excluded.attempt, result_hash=excluded.result_hash,
                       predictions_sha256=excluded.predictions_sha256, null_paths=excluded.null_paths,
                       metrics=excluded.metrics, metric_series=excluded.metric_series,
                       fitted_params=excluded.fitted_params, predictions_path=excluded.predictions_path,
                       duration_s=excluded.duration_s, error=excluded.error, worker=excluded.worker,
                       started_at=excluded.started_at, finished_at=excluded.finished_at,
                       created_at=excluded.created_at""",
                (trial.trial_id, run_id, trial.candidate_id, trial.fold_id, trial.protocol_hash,
                 trial.config_hash, trial.status, trial.phase, trial.seed, trial.snapshot_hash,
                 trial.identity_hash, trial.attempt, trial.result_hash, trial.predictions_sha256,
                 _dump(trial.null_paths), _dump(trial.metrics), _dump(trial.metric_series),
                 _dump(trial.fitted_params), trial.predictions_path, trial.duration_s,
                 trial.error, _dump(trial.worker), trial.started_at, trial.finished_at,
                 trial.created_at.isoformat()),
            )
        return True

    def get_trial_by_key(self, candidate_id: str, fold_id: str, protocol_hash: str,
                         config_hash: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM trials WHERE candidate_id=? AND fold_id=? AND protocol_hash=? AND config_hash=?",
            (candidate_id, fold_id, protocol_hash, config_hash),
        ).fetchone()
        return self._trial_dict(row) if row else None

    def trial_result_valid(self, row: dict[str, Any]) -> tuple[bool, str]:
        """Reject a cache row that is incomplete, non-finite or hash-mismatched."""
        if row.get("status") != "success":
            return False, f"status={row.get('status')}"
        if not row.get("result_hash"):
            return False, "missing result hash (legacy or incomplete write)"
        digest = result_digest(row)
        if digest != row["result_hash"]:
            return False, "result hash mismatch (corrupt cache)"
        if not row.get("metrics"):
            return False, "empty metrics"
        if not row.get("metric_series"):
            return False, "empty metric series"
        path = row.get("predictions_path") or ""
        if path:
            from pathlib import Path
            p = Path(path)
            if not p.exists():
                return False, "predictions file missing"
            if row.get("predictions_sha256"):
                from predicatelab.core.hashing import sha256_file
                if sha256_file(p) != row["predictions_sha256"]:
                    return False, "predictions hash mismatch"
        return True, "ok"

    def get_cached_trial(self, candidate_id: str, fold_id: str, protocol_hash: str,
                         config_hash: str) -> dict[str, Any] | None:
        row = self.get_trial_by_key(candidate_id, fold_id, protocol_hash, config_hash)
        if not row:
            return None
        ok, reason = self.trial_result_valid(row)
        if not ok:
            # Legacy rows are never relabelled: they are simply untrusted and
            # re-executed.  A current row with a written-but-invalid result is
            # marked corrupt so the damage stays visible.
            if row.get("status") == "success" and row.get("phase") != "legacy_unknown":
                with self.transaction() as c:
                    c.execute("UPDATE trials SET status='corrupt', error=? WHERE trial_id=?",
                              (f"cache rejected: {reason}", row["trial_id"]))
            return None
        return row

    @staticmethod
    def _trial_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("metrics", "metric_series", "fitted_params", "worker"):
            d[key] = _loads(d.get(key), {})
        d["null_paths"] = _loads(d.get("null_paths"), [])
        return d

    def list_trials(self, candidate_id: str | None = None, protocol_hash: str | None = None,
                    run_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if candidate_id:
            clauses.append("candidate_id=?"); params.append(candidate_id)
        if protocol_hash:
            clauses.append("protocol_hash=?"); params.append(protocol_hash)
        if run_id:
            clauses.append("run_id=?"); params.append(run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM trials {where} ORDER BY created_at", params).fetchall()
        return [self._trial_dict(r) for r in rows]

    def list_attempts(self, trial_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM trial_attempts WHERE trial_id=? ORDER BY started_at", (trial_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["metrics"] = _loads(d["metrics"], {})
            d["worker"] = _loads(d["worker"], {})
            out.append(d)
        return out

    # -------------------------------------------------------- fitted instances
    def add_fitted_instance(self, inst) -> str:
        inst.ensure_id()
        with self.transaction() as c:
            c.execute(
                """INSERT OR REPLACE INTO fitted_instances(instance_id, candidate_id, fold_id,
                       fitted_params, fit_data_hash, created_at) VALUES (?,?,?,?,?,?)""",
                (inst.instance_id, inst.candidate_id, inst.fold_id,
                 _dump(inst.fitted_params), inst.fit_data_hash, inst.created_at.isoformat()),
            )
        return inst.instance_id

    def list_fitted_instances(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM fitted_instances WHERE candidate_id=? ORDER BY created_at", (candidate_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["fitted_params"] = _loads(d["fitted_params"], {})
            out.append(d)
        return out

    # ---------------------------------------------------------------- evidence
    def record_evidence(self, ev: EvidenceRecord, run_id: str = "") -> str:
        from predicatelab.core.protocol import ProtocolViolation

        ev.ensure_id()
        with self.transaction() as c:
            existing = c.execute(
                "SELECT run_id, protocol_hash, stage FROM evidence WHERE evidence_id=?",
                (ev.evidence_id,),
            ).fetchone()
            if existing is not None:
                # ``evidence_id`` is (candidate, round, protocol_hash, stage),
                # so a same-id conflict is a retry of the same scientific
                # configuration.  The state may advance (candidate ->
                # validated), but a changed protocol hash or run must never
                # overwrite the historical row: that is how a threshold change
                # used to rewrite the same evidence from insufficient to
                # validated.
                if existing["protocol_hash"] != ev.protocol_hash:
                    raise ProtocolViolation(
                        f"evidence {ev.evidence_id} already exists with protocol "
                        f"{existing['protocol_hash'][:16]}, refusing to overwrite with "
                        f"{ev.protocol_hash[:16]}"
                    )
                if existing["stage"] != ev.stage:
                    raise ProtocolViolation(
                        f"evidence {ev.evidence_id} stage changed "
                        f"{existing['stage']!r} -> {ev.stage!r}"
                    )
                if run_id and existing["run_id"] and existing["run_id"] != run_id:
                    raise ProtocolViolation(
                        f"evidence {ev.evidence_id} belongs to run {existing['run_id']}, "
                        f"not {run_id}"
                    )
            c.execute(
                """INSERT INTO evidence(evidence_id, run_id, candidate_id, round_id, protocol_hash,
                       stage, state, seed, status, effect, ci_low, ci_high, p_value, q_value,
                       n_folds, stability, metrics, reason, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(evidence_id) DO UPDATE SET
                       run_id=excluded.run_id, protocol_hash=excluded.protocol_hash,
                       state=excluded.state, seed=excluded.seed, status=excluded.status,
                       effect=excluded.effect, ci_low=excluded.ci_low, ci_high=excluded.ci_high,
                       p_value=excluded.p_value, q_value=excluded.q_value, n_folds=excluded.n_folds,
                       stability=excluded.stability, metrics=excluded.metrics,
                       reason=excluded.reason, created_at=excluded.created_at""",
                (ev.evidence_id, run_id or None, ev.candidate_id, ev.round_id, ev.protocol_hash,
                 ev.stage, ev.state.value, ev.seed, ev.status.value, ev.effect, ev.ci_low,
                 ev.ci_high, ev.p_value, ev.q_value, ev.n_folds, ev.stability, _dump(ev.metrics),
                 ev.reason, ev.created_at.isoformat()),
            )
        return ev.evidence_id

    def list_evidence(self, run_id: str | None = None, round_id: str | None = None,
                      stage: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if run_id:
            clauses.append("run_id=?"); params.append(run_id)
        if round_id:
            clauses.append("round_id=?"); params.append(round_id)
        if stage:
            clauses.append("stage=?"); params.append(stage)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM evidence {where} ORDER BY created_at", params).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["metrics"] = _loads(d["metrics"], {})
            out.append(d)
        return out

    def get_evidence(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE candidate_id=? ORDER BY created_at", (candidate_id,)
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["metrics"] = _loads(d["metrics"], {})
            out.append(d)
        return out

    def latest_evidence(self, run_id: str, candidate_id: str, stage: str | None = None) -> dict[str, Any] | None:
        rows = [e for e in self.list_evidence(run_id=run_id, stage=stage)
                if e["candidate_id"] == candidate_id]
        return rows[-1] if rows else None

    # ---------------------------------------------------------------- feedback
    def add_feedback(self, round_id: str, payload: dict[str, Any], run_id: str = "") -> None:
        from predicatelab.core.schema import utcnow
        with self.transaction() as c:
            c.execute(
                "INSERT INTO feedback(run_id, round_id, payload, created_at) VALUES (?,?,?,?)",
                (run_id, round_id, _dump(payload), utcnow().isoformat()),
            )

    def latest_feedback(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT payload FROM feedback WHERE run_id=? ORDER BY feedback_id DESC LIMIT 1", (run_id,)
        ).fetchone()
        return _loads(row["payload"], {}) if row else None

    # ------------------------------------------------------------ final evals
    def add_final_eval(self, final: FinalEvaluation) -> str:
        final.ensure_id()
        with self.transaction() as c:
            c.execute(
                """INSERT INTO final_evals(final_id, task_id, run_id, candidate_id, protocol_hash,
                       config_hash, manifest_hash, holdout_key, access_id, status, stage, state,
                       exploratory, result_hash, metrics, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT DO UPDATE SET
                       run_id=excluded.run_id, holdout_key=excluded.holdout_key,
                       access_id=excluded.access_id, status=excluded.status,
                       stage=excluded.stage, state=excluded.state, exploratory=excluded.exploratory,
                       result_hash=excluded.result_hash, metrics=excluded.metrics,
                       notes=excluded.notes""",
                (final.final_id, final.task_id, final.run_id, final.candidate_id,
                 final.protocol_hash, final.config_hash, final.manifest_hash, final.holdout_key,
                 final.access_id, final.status, final.stage, final.state.value,
                 1 if final.exploratory else 0, final.result_hash, _dump(final.metrics),
                 final.notes, final.created_at.isoformat()),
            )
        return final.final_id

    def get_final_eval(self, manifest_hash: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM final_evals WHERE manifest_hash=?", (manifest_hash,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = _loads(d["metrics"], {})
        return d

    def get_final_eval_for(self, run_id: str, candidate_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM final_evals WHERE run_id=? AND candidate_id=? "
            "ORDER BY created_at DESC LIMIT 1", (run_id, candidate_id),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = _loads(d["metrics"], {})
        return d

    def list_final_evals(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            rows = self._conn.execute(
                "SELECT * FROM final_evals WHERE run_id=? ORDER BY created_at", (run_id,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM final_evals ORDER BY created_at").fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["metrics"] = _loads(d["metrics"], {})
            out.append(d)
        return out

    def has_frozen(self, candidate_id: str, protocol_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM final_evals WHERE candidate_id=? AND protocol_hash=? "
            "AND status IN ('frozen','success','failed','interrupted')",
            (candidate_id, protocol_hash),
        ).fetchone()
        return row is not None

    # --------------------------------------------------------- frozen families
    def save_frozen_family(self, run_id: str, manifest: dict[str, Any]) -> None:
        from predicatelab.core.protocol import ProtocolViolation
        from predicatelab.core.schema import utcnow
        with self.transaction() as c:
            existing = c.execute(
                "SELECT manifest_hash FROM frozen_families WHERE run_id=?", (run_id,)
            ).fetchone()
            if existing is not None:
                # The family is frozen *before* confirmation.  Allowing an
                # upsert lets a later call after seeing confirmation results
                # replace the family, which is selection on the test data.
                raise ProtocolViolation(
                    f"run {run_id} already has frozen family "
                    f"{existing['manifest_hash'][:16]}; frozen families are immutable"
                )
            c.execute(
                """INSERT INTO frozen_families(run_id, manifest_hash, task_id, protocol_hash,
                       snapshot_hash, data_manifest_hash, payload, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (run_id, manifest["manifest_hash"], manifest["task_id"],
                 manifest["protocol_hash"], manifest["snapshot_hash"],
                 manifest.get("data_manifest_hash", ""), _dump(manifest),
                 utcnow().isoformat()),
            )

    def get_frozen_family(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT payload FROM frozen_families WHERE run_id=?", (run_id,)).fetchone()
        return _loads(row["payload"], {}) if row else None

    # ---------------------------------------------------------- holdout ledger
    @staticmethod
    def _sync_table(src: sqlite3.Connection, dst: sqlite3.Connection, table: str, key_col: str) -> None:
        cols = [r["name"] for r in src.execute(f"PRAGMA table_info({table})")]
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        rows = src.execute(f"SELECT {col_list} FROM {table}").fetchall()
        if not rows:
            return
        with dst:
            for r in rows:
                existing = dst.execute(
                    f"SELECT status, updated_at FROM {table} WHERE access_id=?",
                    (r["access_id"],),
                ).fetchone()
                if existing is not None:
                    if existing["status"] != r["status"] and str(r["updated_at"]) >= str(existing["updated_at"]):
                        dst.execute(
                            f"UPDATE {table} SET status=?, updated_at=?, details=? WHERE access_id=?",
                            (r["status"], r["updated_at"], r["details"], r["access_id"]),
                        )
                    continue
                if r["status"] in ACTIVE_STATUSES:
                    conflict = dst.execute(
                        f"SELECT 1 FROM {table} WHERE {key_col}=? AND status IN (%s)"
                        % ",".join("?" for _ in ACTIVE_STATUSES),
                        (r[key_col], *ACTIVE_STATUSES),
                    ).fetchone()
                    if conflict is not None:
                        continue
                dst.execute(
                    f"INSERT OR IGNORE INTO {table}({col_list}) VALUES ({placeholders})",
                    tuple(r[c] for c in cols),
                )

    def _sync_ledgers(self) -> None:
        if self._ledger_conn is None:
            return
        self._sync_table(self._conn, self._ledger_conn, "holdout_access", "holdout_key")
        self._sync_table(self._ledger_conn, self._conn, "holdout_access", "holdout_key")
        self._sync_table(self._conn, self._ledger_conn, "confirmation_access", "confirmation_key")
        self._sync_table(self._ledger_conn, self._conn, "confirmation_access", "confirmation_key")

    def _push_ledgers(self) -> None:
        if self._ledger_conn is None:
            return
        self._sync_table(self._conn, self._ledger_conn, "holdout_access", "holdout_key")
        self._sync_table(self._conn, self._ledger_conn, "confirmation_access", "confirmation_key")

    @staticmethod
    def holdout_key(**kwargs: Any) -> str:
        return _holdout_key(**kwargs)

    @staticmethod
    def confirmation_key(**kwargs: Any) -> str:
        return _confirmation_key(**kwargs)

    def assert_holdout_available(self, **kwargs: Any) -> None:
        self._sync_ledgers()
        _assert_holdout_available(self._conn, **kwargs)

    def claim_holdout(self, **kwargs: Any) -> dict[str, Any]:
        self._sync_ledgers()
        out = _claim_holdout(self._conn, **kwargs)
        self._push_ledgers()
        return out

    def mark_holdout(self, access_id: str, status: str, **kwargs: Any) -> None:
        self._sync_ledgers()
        _mark_holdout(self._conn, access_id, status, **kwargs)
        self._push_ledgers()

    def record_exploratory_reuse(self, **kwargs: Any) -> str:
        from predicatelab.core.holdout import record_exploratory_reuse

        self._sync_ledgers()
        out = record_exploratory_reuse(self._conn, **kwargs)
        self._push_ledgers()
        return out

    def get_holdout_access(self, access_id: str) -> dict[str, Any] | None:
        self._sync_ledgers()
        row = self._conn.execute("SELECT * FROM holdout_access WHERE access_id=?", (access_id,)).fetchone()
        return dict(row) if row else None

    def list_holdout_access(self, holdout_key: str | None = None) -> list[dict[str, Any]]:
        self._sync_ledgers()
        if holdout_key:
            rows = self._conn.execute(
                "SELECT * FROM holdout_access WHERE holdout_key=? ORDER BY created_at", (holdout_key,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM holdout_access ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def holdout_events(self, access_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM holdout_access_events WHERE access_id=? ORDER BY event_id", (access_id,)
        ).fetchall()]

    def consumed_holdout(self, holdout_key: str) -> list[dict[str, Any]]:
        return [r for r in self.list_holdout_access(holdout_key)
                if r["status"] in ACTIVE_STATUSES and r["status"] != "intent"]

    def claim_confirmation(self, **kwargs: Any) -> dict[str, Any]:
        self._sync_ledgers()
        out = _claim_confirmation(self._conn, **kwargs)
        self._push_ledgers()
        return out

    def mark_confirmation(self, access_id: str, status: str, **kwargs: Any) -> None:
        self._sync_ledgers()
        _mark_confirmation(self._conn, access_id, status, **kwargs)
        self._push_ledgers()

    def list_confirmation_access(self, confirmation_key: str | None = None) -> list[dict[str, Any]]:
        self._sync_ledgers()
        if confirmation_key:
            rows = self._conn.execute(
                "SELECT * FROM confirmation_access WHERE confirmation_key=? ORDER BY created_at",
                (confirmation_key,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM confirmation_access ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def backfill_holdout_ledger(self, artifacts_root: str | Path) -> dict[str, Any]:
        """Record already-consumed historical holdouts as permanently consumed.

        Missing fields are stored as ``legacy_unknown`` (never guessed).  The
        ledger key is then scoped by run/final id so it blocks re-use of that
        known study without pretending to know an exact data identity.
        """
        import hashlib

        root = Path(artifacts_root)
        finals = self.list_final_evals()
        inserted, skipped, unknown = 0, 0, 0
        details_rows = []
        for f in finals:
            if f.get("status") != "success":
                skipped += 1
                continue
            run_id = f.get("run_id") or ""
            context: dict[str, Any] = {}
            if run_id:
                ctx_path = root / run_id / "context.json"
                if ctx_path.exists():
                    context = _loads(ctx_path.read_text(encoding="utf-8"), {}) or {}
            if not context:
                # fall back to scanning artifact directories for this candidate
                for candidate in sorted(root.glob("*/final_evaluation.json")):
                    try:
                        payload = _loads(candidate.read_text(encoding="utf-8"), {}) or {}
                    except Exception:
                        continue
                    if (payload.get("manifest") or {}).get("candidate_id") == f["candidate_id"]:
                        run_id = candidate.parent.name
                        ctx_path = candidate.parent / "context.json"
                        if ctx_path.exists():
                            context = _loads(ctx_path.read_text(encoding="utf-8"), {}) or {}
                        break
            snapshot = (context.get("snapshot") or {})
            protocol = (context.get("effective_protocol") or context.get("protocol") or {})
            holdout = (protocol.get("holdout") or protocol.get("holdout_fold") or {})
            contract = (context.get("data_contract") or {})
            missing = []
            snapshot_hash = snapshot.get("manifest_hash") or "legacy_unknown"
            data_manifest_hash = context.get("data_manifest_hash") or snapshot_hash
            label_spec = context.get("label_spec_hash") or ""
            if not label_spec and contract:
                # The historical contract is recorded data; hashing it is not a
                # guess.  A missing contract stays legacy_unknown.
                try:
                    from predicatelab.core.identity import label_spec_hash
                    from predicatelab.core.schema import DataContract

                    label_spec = label_spec_hash(DataContract.model_validate(contract))
                except Exception:
                    label_spec = ""
            label_spec = label_spec or "legacy_unknown"
            test_start = holdout.get("test_start") or "legacy_unknown"
            test_end = holdout.get("test_end") or "legacy_unknown"
            dataset_id = snapshot.get("dataset_id") or contract.get("dataset_id") or "legacy_unknown"
            for name, value in (("snapshot_hash", snapshot_hash),
                                ("label_spec_hash", label_spec),
                                ("test_start", test_start),
                                ("test_end", test_end),
                                ("dataset_id", dataset_id)):
                if value == "legacy_unknown":
                    missing.append(name)
            unknown += 1 if missing else 0
            key = _holdout_key(
                dataset_id=dataset_id, snapshot_hash=snapshot_hash,
                data_manifest_hash=data_manifest_hash, label_spec_hash=label_spec,
                test_start=test_start, test_end=test_end,
            )
            if missing:
                key = sha256_obj({"legacy_key": key, "final_id": f["final_id"],
                                  "run_id": run_id, "candidate": f["candidate_id"]})
            access_id = "hoa_backfill_" + sha256_obj({
                "final": f["final_id"], "key": key})[:20]
            now = utcnow_value()
            exists = self._conn.execute(
                "SELECT 1 FROM holdout_access WHERE access_id=?", (access_id,)
            ).fetchone()
            if exists:
                skipped += 1
                continue
            with self.transaction() as c:
                c.execute(
                    """INSERT INTO holdout_access(
                           access_id, holdout_key, study_id, run_id, candidate_id,
                           protocol_hash, snapshot_hash, dataset_id, label_spec_hash,
                           test_start, test_end, purpose, status, origin, manifest_hash,
                           code_fingerprint, details, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (access_id, key, f["task_id"], run_id, f["candidate_id"],
                     f["protocol_hash"], snapshot_hash, dataset_id, label_spec,
                     test_start, test_end, "legacy_final", "completed", "backfill_legacy",
                     f["manifest_hash"], None,
                     _dump({"final_id": f["final_id"], "missing_fields": missing,
                            "source": "historical final_evaluation"}),
                     now, now),
                )
                c.execute(
                    """INSERT INTO holdout_access_events(access_id, event, run_id,
                           candidate_id, note, created_at) VALUES (?,?,?,?,?,?)""",
                    (access_id, "backfill_consumed", run_id, f["candidate_id"],
                     "historical holdout treated as already consumed", now),
                )
                c.execute("UPDATE final_evals SET run_id=? WHERE final_id=?",
                          (run_id or f.get("run_id") or "", f["final_id"]))
            inserted += 1
            details_rows.append({"final_id": f["final_id"], "access_id": access_id,
                                 "holdout_key": key, "missing_fields": missing})
        self._push_ledgers()
        return {"inserted": inserted, "skipped": skipped,
                "n_with_unknown_fields": unknown, "rows": details_rows}

    # ------------------------------------------------------------------ export
    def export_evidence_jsonl(self, path: str | Path, run_id: str | None = None) -> int:
        rows = self.list_evidence(run_id=run_id)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return len(rows)
