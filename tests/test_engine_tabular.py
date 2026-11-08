"""End-to-end engine test on the generic tabular adapter.

The statistical thresholds are deliberately relaxed: this test checks the
*mechanics* (proposal → probe → screen → validation → evidence → freeze →
holdout), not whether a predicate is scientifically useful.
"""

import json

import pytest

from predicatelab.core.engine import PredicateLabEngine
from predicatelab.core.protocol import ProtocolViolation


def _config(tmp_path):
    return {
        "run": {
            "name": "engine_test",
            "domain": "tabular",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "db": str(tmp_path / "artifacts" / "state.sqlite"),
            "rounds": 1,
        },
        "training": {"seed": 2025},
        # Discovery uses repeat 0; repeat 1 is a disjoint resampling
        # confirmation family.  Holdout access still requires freeze+confirm.
        "protocol": {"cv_splits": 5, "cv_repeats": 2, "discovery_repeats": 1,
                     "confirmation_kind": "resampling"},
        "evaluation": {
            "name": "tabular_cv_logloss",
            "primary_metric": "log_loss",
            "direction": "lower",
            "secondary_metrics": ["auc", "accuracy"],
            "paired": True,
            "multiple_testing": "benjamini-hochberg",
        },
        "search": {
            "generator": "rule",
            "max_candidates_per_round": 3,
            "screen_folds": ["cv_r0_f0"],
            # Screening and validation folds must be disjoint (D-10).
            "validation_folds": ["cv_r0_f1", "cv_r0_f2", "cv_r0_f3", "cv_r0_f4"],
            "full_validation_top_k": 1,
            # test-only relaxation: promote/validate any finite effect
            "min_screen_effect": -1e9,
            "min_effect": -1.0,
            "min_stability": 0.0,
            "fdr_alpha": 1.0,
            "bootstrap_samples": 200,
            "bootstrap_block": 1,
            "nw_lag": 0,
            "seed": 2025,
        },
    }


def test_round_produces_evidence_and_finalize_isolates_holdout(tmp_path):
    cfg = _config(tmp_path)
    engine = PredicateLabEngine(cfg)
    run_id = engine.run(rounds=1)

    rounds = engine.store.list_rounds(run_id)
    assert len(rounds) == 1
    assert rounds[0]["n_proposed"] >= 1

    evidence = engine.store.list_evidence(run_id=run_id)
    assert evidence, "every candidate must leave an evidence record"
    assert all(e["status"] in {
        "validated", "rejected", "insufficient_evidence", "screened_out",
        "duplicate", "invalid_structure", "execution_failed", "blocked",
    } for e in evidence)

    validated = [e for e in evidence if e["status"] == "validated"]
    assert validated, "relaxed test thresholds should validate the promoted candidate"
    candidate_id = validated[0]["candidate_id"]

    # search context cannot touch the holdout
    ctx = engine._build_context()
    with pytest.raises(ProtocolViolation, match="holdout"):
        ctx.guard.assert_fold_accessible(ctx.protocol.holdout_fold.fold_id, "search")

    # The holdout is only reachable after freeze -> confirmation.
    frozen = engine.freeze_family()
    assert candidate_id in {m["candidate_id"] for m in frozen["members"]}
    confirmation = engine.confirm()
    assert any(r["candidate_id"] == candidate_id and r["state"]
               in {"independently_confirmed", "resampling_confirmed"}
               for r in confirmation["results"]), confirmation

    summary = engine.finalize(candidate_id)
    assert summary["status"] == "success"
    assert summary["result"]["candidate"]["candidate_id"] == candidate_id
    assert summary["result"]["effect"] is not None
    assert engine.artifacts.joinpath("frozen_family.json").exists()
    assert engine.artifacts.joinpath("final_evaluation.json").exists()

    final = engine.store.get_final_eval(summary["manifest_hash"])
    assert final is not None and final["status"] == "success"
    engine.store.close()


def test_finalize_refuses_unvalidated_candidate(tmp_path):
    cfg = _config(tmp_path)
    engine = PredicateLabEngine(cfg)
    run_id = engine.run(rounds=1)
    evidence = engine.store.list_evidence(run_id=run_id)
    candidate_id = evidence[0]["candidate_id"]
    # force every evidence row for this candidate to a non-validated status
    with engine.store.transaction() as conn:
        conn.execute("UPDATE evidence SET status='insufficient_evidence' WHERE candidate_id=?", (candidate_id,))
    with pytest.raises(ProtocolViolation, match="not validated"):
        engine.finalize(candidate_id)
    engine.store.close()
