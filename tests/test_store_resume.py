"""State store identity, caching and resume semantics."""

import json

from predicatelab.core.hashing import sha256_file
from predicatelab.core.schema import (
    EvidenceRecord,
    EvidenceStatus,
    RoundRecord,
    TrialRecord,
)
from predicatelab.core.store import StateStore, result_digest


def _trial(tmp_path, status="success", acc=0.8):
    pred = tmp_path / f"predictions_{status}_{acc}.json"
    pred.write_text(json.dumps({"index": ["2020-01-02"], "pred": [acc]}), encoding="utf-8")
    trial = TrialRecord(
        candidate_id="cand",
        fold_id="f2020",
        protocol_hash="ph",
        config_hash="ch",
        status=status,
        metrics={"acc": acc},
        metric_series={"acc": {"2020-01-02": acc}},
        predictions_path=str(pred),
        predictions_sha256=sha256_file(pred),
    )
    trial.result_hash = result_digest({
        "status": trial.status, "metrics": trial.metrics,
        "metric_series": trial.metric_series, "fitted_params": trial.fitted_params,
        "predictions_path": trial.predictions_path,
        "predictions_sha256": trial.predictions_sha256,
        "null_paths": trial.null_paths,
    })
    return trial


def test_trial_cache_returns_successful_trial(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        store.record_trial(_trial(tmp_path), run_id="r1")
        cached = store.get_cached_trial("cand", "f2020", "ph", "ch")
        assert cached is not None
        assert cached["metrics"]["acc"] == 0.8
        # different config hash must miss
        assert store.get_cached_trial("cand", "f2020", "ph", "other") is None


def test_failed_trial_is_replaced_by_success(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        store.record_trial(_trial(tmp_path, status="failed", acc=0.0), run_id="r1")
        assert store.get_trial_by_key("cand", "f2020", "ph", "ch")["status"] == "failed"
        store.record_trial(_trial(tmp_path, status="success", acc=0.9), run_id="r1")
        row = store.get_trial_by_key("cand", "f2020", "ph", "ch")
        assert row["status"] == "success"
        assert row["metrics"]["acc"] == 0.9


def test_evidence_and_feedback_roundtrip(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        rec = RoundRecord(task_id="t", run_id="r", round_index=0, generator="rule").ensure_id()
        store.save_round(rec)
        ev = EvidenceRecord(
            candidate_id="cand", round_id=rec.round_id, protocol_hash="ph",
            status=EvidenceStatus.INSUFFICIENT_EVIDENCE, effect=0.001, p_value=0.4,
            metrics={"note": "kept even though it did not validate"},
        ).ensure_id()
        store.record_evidence(ev, run_id="r")
        rows = store.list_evidence(run_id="r")
        assert len(rows) == 1
        assert rows[0]["status"] == "insufficient_evidence"
        assert rows[0]["metrics"]["note"].startswith("kept")

        store.add_feedback(rec.round_id, {"validated": [], "insufficient": [{"candidate_id": "cand"}]}, run_id="r")
        fb = store.latest_feedback("r")
        assert fb["insufficient"][0]["candidate_id"] == "cand"
