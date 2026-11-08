"""V1.0 -> V1.1 SQLite migration: additive, verifiable, history-preserving.

The fixture below is the exact V1.0 schema.  Migration must add the V1.1
identity/state columns, keep every historical row, and label unprovable fields
``legacy_unknown`` instead of inventing confirmation.
"""

import json
import sqlite3

from predicatelab.core.store import SCHEMA_VERSION, StateStore

V10_SCHEMA = """
CREATE TABLE tasks (task_id TEXT PRIMARY KEY, domain TEXT NOT NULL,
    payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE snapshots (snapshot_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
    domain TEXT NOT NULL, manifest_method TEXT NOT NULL, manifest_hash TEXT NOT NULL,
    path TEXT, details TEXT, created_at TEXT NOT NULL);
CREATE TABLE templates (template_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL UNIQUE,
    domain TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL, descriptor TEXT NOT NULL,
    rationale TEXT, provenance TEXT, created_at TEXT NOT NULL);
CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY, template_id TEXT NOT NULL,
    template_hash TEXT NOT NULL, domain TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
    descriptor TEXT NOT NULL, dataset_id TEXT NOT NULL, snapshot_hash TEXT NOT NULL,
    bound_scope TEXT, probe_signature TEXT, created_at TEXT NOT NULL,
    UNIQUE(template_hash, dataset_id, snapshot_hash));
CREATE TABLE fitted_instances (instance_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL,
    fold_id TEXT NOT NULL, fitted_params TEXT NOT NULL, fit_data_hash TEXT NOT NULL,
    created_at TEXT NOT NULL);
CREATE TABLE runs (run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, status TEXT NOT NULL,
    config TEXT, artifacts TEXT, started_at TEXT NOT NULL, ended_at TEXT);
CREATE TABLE rounds (round_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT NOT NULL,
    round_index INTEGER NOT NULL, generator TEXT NOT NULL, parent_round_id TEXT,
    n_proposed INTEGER DEFAULT 0, n_bound INTEGER DEFAULT 0, n_validated INTEGER DEFAULT 0,
    n_rejected INTEGER DEFAULT 0, n_insufficient INTEGER DEFAULT 0, summary TEXT,
    created_at TEXT NOT NULL, UNIQUE(run_id, round_index));
CREATE TABLE trials (trial_id TEXT PRIMARY KEY, run_id TEXT, candidate_id TEXT NOT NULL,
    fold_id TEXT NOT NULL, protocol_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
    status TEXT NOT NULL, metrics TEXT, metric_series TEXT, fitted_params TEXT,
    predictions_path TEXT, duration_s REAL DEFAULT 0, error TEXT, worker TEXT,
    created_at TEXT NOT NULL, UNIQUE(candidate_id, fold_id, protocol_hash, config_hash));
CREATE TABLE evidence (evidence_id TEXT PRIMARY KEY, run_id TEXT, candidate_id TEXT NOT NULL,
    round_id TEXT NOT NULL, protocol_hash TEXT NOT NULL, status TEXT NOT NULL,
    effect REAL, ci_low REAL, ci_high REAL, p_value REAL, q_value REAL,
    n_folds INTEGER DEFAULT 0, stability REAL, metrics TEXT, reason TEXT,
    created_at TEXT NOT NULL);
CREATE TABLE feedback (feedback_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,
    round_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE final_evals (final_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL, protocol_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
    manifest_hash TEXT NOT NULL, status TEXT NOT NULL, metrics TEXT, notes TEXT,
    created_at TEXT NOT NULL, UNIQUE(candidate_id, protocol_hash, manifest_hash));
CREATE INDEX idx_trials_candidate ON trials(candidate_id, protocol_hash);
CREATE INDEX idx_evidence_round ON evidence(round_id);
CREATE INDEX idx_evidence_candidate ON evidence(candidate_id);
"""


def _make_v10_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(V10_SCHEMA)
    conn.execute("INSERT INTO tasks VALUES ('t1','tsil','{}','2025-01-01')")
    conn.execute("INSERT INTO runs(run_id, task_id, status, config, artifacts, started_at) "
                 "VALUES ('run_old','t1','search_complete','{}','artifacts/run_old','2025-01-01')")
    conn.execute("INSERT INTO rounds(round_id, run_id, task_id, round_index, generator, "
                 "n_proposed, created_at) VALUES ('round_old','run_old','t1',0,'rule',3,'2025-01-01')")
    conn.execute("INSERT INTO feedback(run_id, round_id, payload, created_at) "
                 "VALUES ('run_old','round_old','{}','2025-01-01')")
    conn.execute("INSERT INTO trials(trial_id, run_id, candidate_id, fold_id, protocol_hash, "
                 "config_hash, status, metrics, metric_series, created_at) "
                 "VALUES ('trial_old','run_old','cand_old','f2020','ph_old','ch_old','success',"
                 "'{\"ic\":0.1}','{\"ic\":{\"2020-01-02\":0.1}}','2025-01-01')")
    conn.execute("INSERT INTO evidence(evidence_id, run_id, candidate_id, round_id, "
                 "protocol_hash, status, effect, p_value, created_at) "
                 "VALUES ('ev_old','run_old','cand_old','round_old','ph_old','validated',0.1,0.01,"
                 "'2025-01-01')")
    conn.execute("INSERT INTO final_evals(final_id, task_id, candidate_id, protocol_hash, "
                 "config_hash, manifest_hash, status, metrics, notes, created_at) "
                 "VALUES ('final_old','t1','cand_old','ph_old','ch_old','mh_old','success',"
                 "'{\"ic\":0.1}','old holdout','2025-01-01')")
    conn.commit()
    conn.close()


def test_v10_database_migrates_without_losing_rows(tmp_path):
    db = tmp_path / "old.sqlite"
    _make_v10_db(db)

    with StateStore(db) as store:
        assert store.schema_version == SCHEMA_VERSION
        migrations = [dict(r) for r in store._conn.execute(
            "SELECT * FROM schema_migrations ORDER BY version")]
        assert [m["version"] for m in migrations] == [2, SCHEMA_VERSION]
        assert SCHEMA_VERSION == 3

        round_row = dict(store._conn.execute(
            "SELECT * FROM rounds WHERE round_id='round_old'").fetchone())
        assert round_row["status"] == "completed"
        assert round_row["status_origin"] == "backfill_feedback"

        trial = store.get_trial_by_key("cand_old", "f2020", "ph_old", "ch_old")
        assert trial["status"] == "success"
        assert trial["phase"] == "legacy_unknown"
        assert trial["metrics"]["ic"] == 0.1

        ev = store.list_evidence(run_id="run_old")[0]
        assert ev["stage"] == "legacy"
        assert ev["state"] == "exploratory_positive"

        final = store.list_final_evals()[0]
        assert final["status"] == "success"
        assert final["stage"] == "legacy"
        assert final["state"] == "legacy_unknown"
        assert "V1.1 legacy" in final["notes"]

        for table, expected in {"tasks": 1, "runs": 1, "rounds": 1, "trials": 1,
                                "evidence": 1, "feedback": 1, "final_evals": 1}.items():
            assert store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "old.sqlite"
    _make_v10_db(db)
    StateStore(db).close()
    with StateStore(db) as store:
        assert store.schema_version == SCHEMA_VERSION
        assert store._conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 2
        assert store._conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 1
        assert len(store.list_holdout_access()) == 0


def test_historical_holdout_is_backfilled_as_consumed(tmp_path):
    db = tmp_path / "old.sqlite"
    _make_v10_db(db)
    artifacts = tmp_path / "artifacts" / "run_old"
    artifacts.mkdir(parents=True)
    (artifacts / "context.json").write_text(json.dumps({
        "snapshot": {"manifest_hash": "snap_old", "dataset_id": "qlib-cn-csi300"},
        "data_manifest_hash": "data_old",
        "label_spec_hash": "label_old",
        "effective_protocol": {"holdout": {"test_start": "2024-01-01",
                                           "test_end": "2025-08-31"}},
    }))
    (artifacts / "final_evaluation.json").write_text(json.dumps({
        "manifest": {"candidate_id": "cand_old"},
    }))

    with StateStore(db) as store:
        out = store.backfill_holdout_ledger(tmp_path / "artifacts")
        assert out["inserted"] == 1
        assert out["n_with_unknown_fields"] == 0
        rows = store.list_holdout_access()
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"
        assert rows[0]["origin"] == "backfill_legacy"
        assert rows[0]["snapshot_hash"] == "snap_old"
        assert rows[0]["test_start"] == "2024-01-01"
        assert store.list_final_evals()[0]["run_id"] == "run_old"
        # Re-running the backfill is idempotent.
        again = store.backfill_holdout_ledger(tmp_path / "artifacts")
        assert again["inserted"] == 0
        assert len(store.list_holdout_access()) == 1


def test_backfill_marks_missing_fields_legacy_unknown(tmp_path):
    db = tmp_path / "old.sqlite"
    _make_v10_db(db)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    with StateStore(db) as store:
        out = store.backfill_holdout_ledger(tmp_path / "artifacts")
        assert out["inserted"] == 1
        assert out["n_with_unknown_fields"] == 1
        row = store.list_holdout_access()[0]
        details = json.loads(row["details"])
        assert "snapshot_hash" in details["missing_fields"]
        assert row["snapshot_hash"] == "legacy_unknown"
        assert row["test_end"] == "legacy_unknown"
