"""Small-sample scenario: ground truth, fold hygiene and lifecycle behaviour."""

import copy

import numpy as np
import pytest

from predicatelab.core.adapter import TrialRequest
from predicatelab.domains.small_sample.adapter import (
    MIN_SAMPLES,
    DescriptorError,
    SmallSampleAdapter,
    _fit_transform,
    validate_small_sample_descriptor,
)
from predicatelab.domains.small_sample.synthetic import (
    MODES,
    descriptor_family,
    generate,
    matches_true_family,
    truth_family,
)


def _config(mode="signal_sign_interaction", n_samples=96, *, seed=20261108,
            rounds=1, cv_splits=4, strength=3.0, ack=True):
    cfg = {
        "run": {
            "name": "test_small_sample",
            "domain": "small_sample",
            "rounds": rounds,
            "require_confirmation": True,
        },
        "seeds": [2025],
        "data": {
            "generator": mode,
            "seed": seed,
            "n_samples": n_samples,
            "n_features": 8,
            "signal_strength": strength,
            "threshold_quantile": 0.75,
            "class_balance": 0.5,
            "holdout_fraction": 0.25,
            "discovery_fraction": 0.65,
            "confirmation_mode": "independent",
        },
        "training": {"seed": 2025, "C": 1.0, "max_iter": 5000},
        "protocol": {"cv_splits": cv_splits, "cv_repeats": 1,
                     "discovery_repeats": 1, "confirmation_splits": 3},
        "search": {"generator": "rule", "max_candidates_per_round": 24,
                   "save_predictions": False, "fdr_alpha": 0.10,
                   "min_effect": 0.0025, "min_stability": 0.60,
                   "bootstrap_samples": 200, "bootstrap_block": 1, "nw_lag": 0,
                   "seed": 2025},
        "execution": {"workers": 1},
    }
    if ack:
        cfg["run"]["warning_acknowledgements"] = "all"
        cfg["run"]["warning_ack_reason"] = "test: synthetic small-sample scenario"
    return cfg


# ------------------------------------------------------------------ descriptors
@pytest.mark.parametrize("desc", [
    {"op": "ratio", "a": 0, "b": 0},
    {"op": "interaction", "a": -1, "b": 2},
    {"op": "threshold", "a": 0, "quantile": 1.5},
    {"op": "ratio_threshold", "a": 0, "b": 8, "quantile": 0.5},
    {"op": "sigmoid", "a": 0, "scale": 100.0},
    {"op": "banana", "a": 0},
])
def test_descriptor_validation_rejects_bad_input(desc):
    with pytest.raises(DescriptorError):
        validate_small_sample_descriptor(desc, n_features=8)


def test_kind_must_match_operator():
    adapter = SmallSampleAdapter(_config())
    with pytest.raises(DescriptorError):
        adapter.validate_descriptor("conditional", {"op": "ratio", "a": 0, "b": 1})
    adapter.validate_descriptor("statistical", {"op": "ratio", "a": 0, "b": 1})
    adapter.validate_descriptor("relational", {"op": "interaction", "a": 1, "b": 3})


def test_ratio_threshold_parameter_is_fitted_on_training_only():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((40, 3))
    other = rng.standard_normal((30, 3)) * 50.0  # must not move the threshold
    desc = {"op": "ratio_threshold", "a": 0, "b": 2, "quantile": 0.7}
    tr, va, params = _fit_transform(desc, train, other)
    ratio = train[:, 0] / (np.abs(train[:, 2]) + 0.1 * np.std(train[:, 2]))
    assert params["threshold"] == pytest.approx(np.quantile(ratio, 0.7))
    assert tr.shape == (40, 1) and va.shape == (30, 1)


def test_interaction_predicate_is_origin_defined_and_fold_independent():
    rng = np.random.default_rng(1)
    train = rng.standard_normal((50, 4))
    other = rng.standard_normal((50, 4))
    desc = {"op": "interaction", "a": 1, "b": 3}
    tr, va, _ = _fit_transform(desc, train, other)
    assert np.array_equal(tr[:, 0], np.sign(train[:, 1]) * np.sign(train[:, 3]))
    assert np.array_equal(va[:, 0], np.sign(other[:, 1]) * np.sign(other[:, 3]))
    # changing the training sample must not change the prediction on `other`
    _, va2, _ = _fit_transform(desc, train[:20], other)
    assert np.array_equal(va, va2)


# ---------------------------------------------------------------------- splits
def test_discovery_confirmation_and_holdout_are_pairwise_disjoint():
    adapter = SmallSampleAdapter(_config())
    protocol = adapter.build_protocol({})
    disc = set(adapter._discovery_idx.tolist())
    conf = set(adapter._confirmation_idx.tolist())
    hold = set(adapter._holdout_idx.tolist())
    assert disc and conf and hold
    assert not (disc & conf)
    assert not (disc & hold)
    assert not (conf & hold)
    assert disc | conf | hold == set(range(len(adapter.y)))

    search_valid = {i for f in protocol.folds for i in f.metadata["valid_idx"]}
    conf_valid = {i for f in protocol.confirmation_folds for i in f.metadata["valid_idx"]}
    hold_valid = set(protocol.holdout_fold.metadata["valid_idx"])
    assert search_valid <= disc
    assert conf_valid <= conf
    assert hold_valid == hold
    assert not (search_valid & conf_valid)
    assert not (search_valid & hold_valid)


def test_confirmation_training_folds_include_discovery_but_never_their_own_valid():
    adapter = SmallSampleAdapter(_config())
    protocol = adapter.build_protocol({})
    disc = set(adapter._discovery_idx.tolist())
    for fold in protocol.confirmation_folds:
        train = set(fold.metadata["train_idx"])
        valid = set(fold.metadata["valid_idx"])
        assert disc <= train
        assert not (train & valid)


# ---------------------------------------------------------------- data/model
def test_generation_is_deterministic_and_seed_sensitive():
    a = generate(seed=7, n_samples=64, n_features=6, mode="signal_sign_interaction")
    b = generate(seed=7, n_samples=64, n_features=6, mode="signal_sign_interaction")
    c = generate(seed=8, n_samples=64, n_features=6, mode="signal_sign_interaction")
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.X, c.X)


def test_null_modes_have_no_truth_indicator_and_signal_modes_do():
    for mode in MODES:
        data = generate(seed=3, n_samples=64, n_features=6, mode=mode,
                        signal_strength=3.0, threshold_quantile=0.75)
        if mode.startswith("null"):
            assert data.truth_indicator is None
            assert truth_family(data.truth) is None
        else:
            assert data.truth_indicator is not None
            assert truth_family(data.truth) is not None


def test_true_family_is_recovered_exactly_for_the_interaction_truth():
    adapter = SmallSampleAdapter(_config(strength=4.0))
    true_desc = adapter.truth["true_descriptor"]
    assert adapter.matches_truth(true_desc)
    assert adapter.truth_indicator is not None
    # the compiler's interaction column *is* the latent indicator (raw signs)
    col = adapter.predicate_on_pool(true_desc)
    pool = np.setdiff1d(np.arange(len(adapter.y)), adapter._holdout_idx)
    assert np.array_equal(col, adapter.truth_indicator[pool])
    assert adapter.recovery_auc(true_desc) == pytest.approx(1.0)

    # a wrong pair is not credited
    assert not adapter.matches_truth({"op": "interaction", "a": 0, "b": 2})
    assert descriptor_family({"op": "interaction", "a": 1, "b": 3}) == "interaction(1,3)"


def test_recovery_auc_credits_monotone_equivalent_predicates():
    adapter = SmallSampleAdapter(_config(mode="signal_ratio_threshold", n_samples=256,
                                         strength=4.0))
    truth_family_key = adapter.true_family()
    assert truth_family_key == "ratio_threshold(0,2)"
    # the continuous ratio is monotone in the ratio threshold at the same
    # quantile, so it should separate the latent indicator almost perfectly
    auc = adapter.recovery_auc({"op": "ratio", "a": 0, "b": 2})
    assert auc is not None and auc > 0.95
    assert not adapter.matches_truth({"op": "ratio", "a": 0, "b": 2})


def test_generator_proposes_the_true_family():
    adapter = SmallSampleAdapter(_config(strength=4.0))
    gen = adapter.make_generator(adapter.config, {})
    proposals = gen.propose(round_index=0, n=24, feedback=None,
                            known_template_hashes=set())
    families = {descriptor_family(t.descriptor) for t in proposals}
    assert adapter.true_family() in families
    assert matches_true_family(adapter.truth["true_descriptor"], adapter.truth)


# ----------------------------------------------------------------- suitability
def test_suitability_blocks_below_minimum_samples():
    adapter = SmallSampleAdapter(_config(n_samples=MIN_SAMPLES - 8))
    snap = adapter.snapshot(adapter.config)
    report = adapter.suitability(snap, adapter.config)
    assert report["status"] == "block"
    assert any("sample_count" in b for b in report["blockers"])


def test_suitability_warns_at_small_n_but_does_not_block():
    adapter = SmallSampleAdapter(_config(n_samples=96))
    snap = adapter.snapshot(adapter.config)
    report = adapter.suitability(snap, adapter.config)
    assert report["status"] == "warn"
    assert report["warnings"]
    assert not report["blockers"]


# -------------------------------------------------------------------- trials
def test_trial_series_is_keyed_per_sample_and_unique_across_folds():
    adapter = SmallSampleAdapter(_config())
    protocol = adapter.build_protocol({})
    seen = set()
    for fold in protocol.folds + protocol.confirmation_folds:
        req = TrialRequest(
            candidate_id="cand", variant="candidate",
            descriptor={"op": "interaction", "a": 1, "b": 3},
            fold=fold, seed=2025, training_config={}, workdir="",
            protocol_hash="p", config_hash="c",
        )
        result = adapter.run_trial(req)
        assert result["status"] == "success", result.get("error")
        keys = set(result["metric_series"]["neg_brier"])
        assert keys and not (keys & seen)
        seen |= keys
        assert result["fitted_params"]["op"] == "interaction"


# ------------------------------------------------------------------ lifecycle
def test_engine_runs_one_discovery_round_and_recovers_the_true_family(tmp_path):
    from predicatelab.core.engine import PredicateLabEngine

    cfg = copy.deepcopy(_config(rounds=1, strength=4.0))
    cfg["run"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["run"]["db"] = str(tmp_path / "test.sqlite")
    engine = PredicateLabEngine(cfg, db_path=cfg["run"]["db"])
    try:
        run_id = engine.run(rounds=1)
        evidence = engine.store.list_evidence(run_id=run_id, stage="discovery")
        validated = [e for e in evidence if e["status"] == "validated"]
        assert validated, "expected the true interaction to validate at n=96, strength=4"
        adapter = SmallSampleAdapter(cfg)
        hits = []
        for ev in validated:
            cand = engine.store.get_candidate(ev["candidate_id"])
            hits.append(adapter.matches_truth(cand["descriptor"]))
        assert any(hits), [engine.store.get_candidate(e["candidate_id"])["descriptor"]
                           for e in validated]
    finally:
        engine.store.close()


def test_engine_full_lifecycle_at_very_small_n_terminates_in_a_defined_state(tmp_path):
    """Freeze/confirm/finalize must either conclude or refuse -- never crash."""
    from predicatelab.core.engine import PredicateLabEngine

    cfg = copy.deepcopy(_config(n_samples=64, rounds=1, strength=2.0))
    cfg["run"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["run"]["db"] = str(tmp_path / "test.sqlite")
    engine = PredicateLabEngine(cfg, db_path=cfg["run"]["db"])
    try:
        run_id = engine.run(rounds=1)
        engine._write_report()
        report_path = engine.artifacts / "report.json"
        assert report_path.exists()
        frozen = None
        try:
            frozen = engine.freeze_family()
        except RuntimeError:
            frozen = None
        if frozen is not None:
            summary = engine.confirm()
            assert isinstance(summary, dict)
        confirmed = [
            e for e in engine.store.list_evidence(run_id=run_id, stage="confirmation")
            if e["state"] in {"resampling_confirmed", "independently_confirmed"}
        ]
        if confirmed:
            result = engine.finalize(confirmed[0]["candidate_id"])
            assert result["status"] == "success"
            assert result["result"]["state"] in {
                "independently_confirmed", "exploratory_positive",
                "insufficient", "rejected",
            }
    finally:
        engine.store.close()
