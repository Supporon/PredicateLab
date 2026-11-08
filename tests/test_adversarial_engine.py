"""Adversarial engine tests: unfinished runs, retries, resume, partial folds.

Every test here corresponds to a V1.0 failure that could produce a
"validated" result from incomplete or stale state.
"""

import pytest

from predicatelab.core.engine import PredicateLabEngine
from predicatelab.core.protocol import ProtocolViolation


def _config(tmp_path, *, seeds=None, max_candidates=2, require_confirmation=True):
    cfg = {
        "run": {
            "name": "adversarial",
            "domain": "tabular",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "db": str(tmp_path / "artifacts" / "state.sqlite"),
            "rounds": 1,
            "require_confirmation": require_confirmation,
        },
        "training": {"seed": 2025},
        "protocol": {"cv_splits": 5, "cv_repeats": 2, "discovery_repeats": 1,
                     "confirmation_kind": "resampling"},
        "evaluation": {
            "name": "tabular_cv_logloss", "primary_metric": "log_loss",
            "direction": "lower", "secondary_metrics": ["auc", "accuracy"],
            "paired": True, "multiple_testing": "benjamini-hochberg",
        },
        "search": {
            "generator": "rule",
            "max_candidates_per_round": max_candidates,
            "screen_folds": ["cv_r0_f0"],
            # Screening and validation folds must be disjoint (D-10).
            "validation_folds": ["cv_r0_f1", "cv_r0_f2", "cv_r0_f3", "cv_r0_f4"],
            "full_validation_top_k": 1,
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
    if seeds:
        cfg["seeds"] = seeds
    return cfg


def _validated_id(engine, run_id):
    rows = [e for e in engine.store.list_evidence(run_id=run_id, stage="discovery")
            if e["status"] == "validated"]
    assert rows, "relaxed test thresholds should validate a candidate"
    return rows[0]["candidate_id"]


def _run_to_confirmed(tmp_path, **cfg_kw):
    engine = PredicateLabEngine(_config(tmp_path, **cfg_kw))
    run_id = engine.run(rounds=1)
    candidate_id = _validated_id(engine, run_id)
    engine.freeze_family()
    engine.confirm()
    return engine, run_id, candidate_id


def test_finalize_refuses_a_run_that_is_not_search_complete(tmp_path):
    engine, run_id, candidate_id = _run_to_confirmed(tmp_path)
    engine.store.set_run_status(run_id, "running")
    with pytest.raises(ProtocolViolation, match="unfinished run"):
        engine.finalize(candidate_id)
    engine.store.close()


def test_finalize_refuses_a_validated_but_unfrozen_candidate(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    run_id = engine.run(rounds=1)
    candidate_id = _validated_id(engine, run_id)
    with pytest.raises(ProtocolViolation, match="not frozen"):
        engine.finalize(candidate_id)
    engine.store.close()


def test_finalize_requires_independent_confirmation(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    run_id = engine.run(rounds=1)
    candidate_id = _validated_id(engine, run_id)
    engine.freeze_family()
    with pytest.raises(ProtocolViolation, match="not independently confirmed"):
        engine.finalize(candidate_id)
    engine.store.close()


def test_finalize_detects_snapshot_drift(tmp_path):
    engine, _run_id, candidate_id = _run_to_confirmed(tmp_path)
    with engine.store.transaction() as c:
        c.execute("UPDATE candidates SET snapshot_hash='different' WHERE candidate_id=?",
                  (candidate_id,))
    with pytest.raises(ProtocolViolation, match="different snapshot"):
        engine.finalize(candidate_id)
    engine.store.close()


def test_repeated_finalize_returns_saved_result_without_consuming_again(tmp_path):
    engine, _run_id, candidate_id = _run_to_confirmed(tmp_path)
    first = engine.finalize(candidate_id)
    assert first["status"] == "success"
    second = engine.finalize(candidate_id)
    assert second["already_completed"] is True
    assert len(engine.store.list_final_evals()) == 1
    assert len(engine.store.list_holdout_access()) == 1
    engine.store.close()


def test_resume_does_not_duplicate_evidence_or_rounds(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    run_id = engine.run(rounds=1)
    before = engine.store.list_evidence(run_id=run_id)
    engine.run(rounds=1)
    after = engine.store.list_evidence(run_id=run_id)
    assert len(before) == len(after)
    assert len({e["evidence_id"] for e in after}) == len(after)
    assert len(engine.store.list_rounds(run_id)) == 1
    engine.store.close()


def test_interrupted_round_is_rerun_and_evidence_is_upserted(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    run_id = engine.run(rounds=1)
    before = engine.store.list_evidence(run_id=run_id)
    with engine.store.transaction() as c:
        c.execute("UPDATE rounds SET status='interrupted' WHERE run_id=?", (run_id,))
    engine.run(rounds=1)
    rounds = engine.store.list_rounds(run_id)
    assert rounds[0]["status"] == "completed"
    after = engine.store.list_evidence(run_id=run_id)
    assert len(after) == len(before)
    assert len({e["evidence_id"] for e in after}) == len(after)
    engine.store.close()


def test_multi_seed_actually_trains_each_seed(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path, seeds=[2025, 2026], max_candidates=1))
    run_id = engine.run(rounds=1)
    candidate_id = _validated_id(engine, run_id)
    rows = engine.store.list_trials(candidate_id=candidate_id, run_id=run_id)
    seeds = {r["seed"] for r in rows}
    assert rows, "multi-seed run must persist trial rows"
    assert {"2025", "2026"} <= {str(s) for s in seeds}
    # Each seed also has its own phase/fold/seed-isolated workdir.
    dirs = {p.name for p in (engine.artifacts / "trials").rglob("seed_*")}
    assert {"seed_2025", "seed_2026"} <= dirs
    engine.store.close()


def test_partial_fold_failure_below_coverage_is_insufficient(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    engine._build_context()
    per_seed = {
        2025: {"effect": 0.1, "p_value": 0.01, "ci_low": 0.05, "ci_high": 0.15,
               "stability": 1.0, "n_folds": 4, "fold_effects": {"a": 0.1}, "inferential": True},
        2026: {"effect": 0.1, "p_value": 0.01, "ci_low": 0.05, "ci_high": 0.15,
               "stability": 1.0, "n_folds": 1, "fold_effects": {"a": 0.1}, "inferential": True},
    }
    agg = engine._aggregate(per_seed, n_expected_folds=4,
                            failures=["fold b failed", "fold c failed", "fold d failed"])
    assert agg["fold_coverage"] == pytest.approx(0.25)
    assert engine._classify(agg, engine._ctx.policy).value == "insufficient_evidence"
    engine.store.close()


def test_non_finite_aggregate_cannot_be_classified(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    engine._build_context()
    per_seed = {2025: {"effect": float("nan"), "p_value": 0.01, "ci_low": 0.0,
                       "ci_high": 0.1, "stability": 1.0, "n_folds": 4,
                       "fold_effects": {"a": 0.1}, "inferential": True}}
    agg = engine._aggregate(per_seed, n_expected_folds=4, failures=["execution failed"])
    assert "error" in agg
    engine.store.close()
