"""SQLite state store.

A single SQLite file is enough for V1.0: the engine is a modular monolith and
trials are executed one per subprocess, so there is one writer.  WAL mode is
enabled so a long-running report process can read while a search writes.

The store is the basis of *resume*: a trial is identified by
``(candidate, fold, protocol_hash, config_hash)`` and a successful trial is
never re-executed, while failed trials are kept (and can be retried explicitly).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

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


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=str)


def _loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)


_SCHEMA = """
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
    config TEXT,
    artifacts TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    round_index INTEGER NOT NULL,
    generator TEXT NOT NULL,
    parent_round_id TEXT,
    n_proposed INTEGER DEFAULT 0,
    n_bound INTEGER DEFAULT 0,
    n_validated INTEGER DEFAULT 0,
    n_rejected INTEGER DEFAULT 0,
    n_insufficient INTEGER DEFAULT 0,
    summary TEXT,
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
    metrics TEXT,
    metric_series TEXT,
    fitted_params TEXT,
    predictions_path TEXT,
    duration_s REAL DEFAULT 0,
    error TEXT,
    worker TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, fold_id, protocol_hash, config_hash)
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT,
    candidate_id TEXT NOT NULL,
    round_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
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
    candidate_id TEXT NOT NULL,
    protocol_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(candidate_id, protocol_hash, manifest_hash)
);
CREATE INDEX IF NOT EXISTS idx_trials_candidate ON trials(candidate_id, protocol_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_round ON evidence(round_id);
CREATE INDEX IF NOT EXISTS idx_evidence_candidate ON evidence(candidate_id);
"""


class StateStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ basics
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

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
                (
                    snap.snapshot_id,
                    snap.dataset_id,
                    snap.domain,
                    snap.manifest_method,
                    snap.manifest_hash,
                    snap.path,
                    _dump(snap.details),
                    snap.created_at.isoformat(),
                ),
            )
        return snap.snapshot_id

    def get_snapshot(self, snapshot_id: str) -> DataSnapshot | None:
        row = self._conn.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if not row:
            return None
        return DataSnapshot(
            snapshot_id=row["snapshot_id"],
            dataset_id=row["dataset_id"],
            domain=row["domain"],
            manifest_method=row["manifest_method"],
            manifest_hash=row["manifest_hash"],
            path=row["path"] or "",
            details=_loads(row["details"], {}),
            created_at=row["created_at"],
        )

    # --------------------------------------------------------------- templates
    def add_template(self, tpl: PredicateTemplate) -> tuple[str, bool]:
        """Insert if the semantic content is new.  Returns (template_id, created)."""
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
                (
                    tpl.template_id,
                    tpl.content_hash,
                    tpl.domain,
                    tpl.kind.value,
                    tpl.name,
                    _dump(tpl.descriptor),
                    tpl.rationale,
                    _dump(tpl.provenance.model_dump(mode="json")),
                    tpl.created_at.isoformat(),
                ),
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
                (
                    cand.candidate_id,
                    cand.template_id,
                    cand.template_hash,
                    cand.domain,
                    cand.kind.value,
                    cand.name,
                    _dump(cand.descriptor),
                    cand.dataset_id,
                    cand.snapshot_hash,
                    _dump(cand.bound_scope),
                    cand.created_at.isoformat(),
                ),
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
            c.execute(
                "UPDATE candidates SET probe_signature=? WHERE candidate_id=?",
                (_dump(signature), candidate_id),
            )

    def list_probe_signatures(self, snapshot_hash: str) -> dict[str, list[float]]:
        rows = self._conn.execute(
            "SELECT candidate_id, probe_signature FROM candidates WHERE snapshot_hash=? AND probe_signature IS NOT NULL",
            (snapshot_hash,),
        ).fetchall()
        return {r["candidate_id"]: _loads(r["probe_signature"], []) for r in rows}

    # -------------------------------------------------------------------- runs
    def create_run(self, run_id: str, task_id: str, config: dict[str, Any], artifacts: str) -> None:
        from predicatelab.core.schema import utcnow

        with self.transaction() as c:
            c.execute(
                "INSERT INTO runs(run_id, task_id, status, config, artifacts, started_at) VALUES (?,?,?,?,?,?)",
                (run_id, task_id, "running", _dump(config), artifacts, utcnow().isoformat()),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        from predicatelab.core.schema import utcnow

        with self.transaction() as c:
            c.execute(
                "UPDATE runs SET status=?, ended_at=? WHERE run_id=?",
                (status, utcnow().isoformat(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["config"] = _loads(d["config"], {})
        return d

    # ------------------------------------------------------------------ rounds
    def save_round(self, rnd: RoundRecord) -> str:
        rnd.ensure_id()
        with self.transaction() as c:
            c.execute(
                """INSERT INTO rounds(round_id, run_id, task_id, round_index, generator,
                       parent_round_id, n_proposed, n_bound, n_validated, n_rejected,
                       n_insufficient, summary, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rnd.round_id,
                    rnd.run_id,
                    rnd.task_id,
                    rnd.round_index,
                    rnd.generator,
                    rnd.parent_round_id,
                    rnd.n_proposed,
                    rnd.n_bound,
                    rnd.n_validated,
                    rnd.n_rejected,
                    rnd.n_insufficient,
                    _dump(rnd.summary),
                    rnd.created_at.isoformat(),
                ),
            )
        return rnd.round_id

    def update_round(self, round_id: str, **fields: Any) -> None:
        allowed = {
            "n_proposed",
            "n_bound",
            "n_validated",
            "n_rejected",
            "n_insufficient",
            "summary",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = [_dump(v) if k == "summary" else v for k, v in updates.items()]
        with self.transaction() as c:
            c.execute(f"UPDATE rounds SET {cols} WHERE round_id=?", (*vals, round_id))

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
    def record_trial(self, trial: TrialRecord, run_id: str = "") -> bool:
        """Record a trial.  Existing successful trials win; failed ones may be replaced."""
        trial.ensure_id()
        existing = self.get_trial_by_key(
            trial.candidate_id, trial.fold_id, trial.protocol_hash, trial.config_hash
        )
        if existing and existing["status"] == "success" and trial.status != "success":
            return False
        with self.transaction() as c:
            c.execute(
                """INSERT INTO trials(trial_id, run_id, candidate_id, fold_id, protocol_hash,
                       config_hash, status, metrics, metric_series, fitted_params,
                       predictions_path, duration_s, error, worker, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(candidate_id, fold_id, protocol_hash, config_hash) DO UPDATE SET
                       status=excluded.status, metrics=excluded.metrics,
                       metric_series=excluded.metric_series, fitted_params=excluded.fitted_params,
                       predictions_path=excluded.predictions_path, duration_s=excluded.duration_s,
                       error=excluded.error, worker=excluded.worker, created_at=excluded.created_at""",
                (
                    trial.trial_id,
                    run_id,
                    trial.candidate_id,
                    trial.fold_id,
                    trial.protocol_hash,
                    trial.config_hash,
                    trial.status,
                    _dump(trial.metrics),
                    _dump(trial.metric_series),
                    _dump(trial.fitted_params),
                    trial.predictions_path,
                    trial.duration_s,
                    trial.error,
                    _dump(trial.worker),
                    trial.created_at.isoformat(),
                ),
            )
        return True

    def get_trial_by_key(
        self, candidate_id: str, fold_id: str, protocol_hash: str, config_hash: str
    ) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT * FROM trials WHERE candidate_id=? AND fold_id=? AND protocol_hash=? AND config_hash=?""",
            (candidate_id, fold_id, protocol_hash, config_hash),
        ).fetchone()
        return self._trial_dict(row) if row else None

    def get_cached_trial(
        self, candidate_id: str, fold_id: str, protocol_hash: str, config_hash: str
    ) -> dict[str, Any] | None:
        row = self.get_trial_by_key(candidate_id, fold_id, protocol_hash, config_hash)
        return row if row and row["status"] == "success" else None

    @staticmethod
    def _trial_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["metrics"] = _loads(d["metrics"], {})
        d["metric_series"] = _loads(d["metric_series"], {})
        d["fitted_params"] = _loads(d["fitted_params"], {})
        d["worker"] = _loads(d["worker"], {})
        return d

    def list_trials(self, candidate_id: str | None = None, protocol_hash: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if candidate_id:
            clauses.append("candidate_id=?"); params.append(candidate_id)
        if protocol_hash:
            clauses.append("protocol_hash=?"); params.append(protocol_hash)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM trials {where} ORDER BY created_at", params).fetchall()
        return [self._trial_dict(r) for r in rows]

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
        ev.ensure_id()
        with self.transaction() as c:
            c.execute(
                """INSERT OR REPLACE INTO evidence(evidence_id, run_id, candidate_id, round_id,
                       protocol_hash, status, effect, ci_low, ci_high, p_value, q_value,
                       n_folds, stability, metrics, reason, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ev.evidence_id,
                    run_id,
                    ev.candidate_id,
                    ev.round_id,
                    ev.protocol_hash,
                    ev.status.value,
                    ev.effect,
                    ev.ci_low,
                    ev.ci_high,
                    ev.p_value,
                    ev.q_value,
                    ev.n_folds,
                    ev.stability,
                    _dump(ev.metrics),
                    ev.reason,
                    ev.created_at.isoformat(),
                ),
            )
        return ev.evidence_id

    def list_evidence(self, run_id: str | None = None, round_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if run_id:
            clauses.append("run_id=?"); params.append(run_id)
        if round_id:
            clauses.append("round_id=?"); params.append(round_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM evidence {where} ORDER BY created_at", params
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["metrics"] = _loads(d["metrics"], {})
            out.append(d)
        return out

    def get_evidence(self, candidate_id: str) -> list[dict[str, Any]]:
        return [e for e in self.list_evidence() if e["candidate_id"] == candidate_id]

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
                """INSERT OR REPLACE INTO final_evals(final_id, task_id, candidate_id, protocol_hash,
                       config_hash, manifest_hash, status, metrics, notes, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    final.final_id,
                    final.task_id,
                    final.candidate_id,
                    final.protocol_hash,
                    final.config_hash,
                    final.manifest_hash,
                    final.status,
                    _dump(final.metrics),
                    final.notes,
                    final.created_at.isoformat(),
                ),
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

    def has_frozen(self, candidate_id: str, protocol_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM final_evals WHERE candidate_id=? AND protocol_hash=? AND status IN ('frozen','success')",
            (candidate_id, protocol_hash),
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------ export
    def export_evidence_jsonl(self, path: str | Path, run_id: str | None = None) -> int:
        rows = self.list_evidence(run_id=run_id)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return len(rows)
