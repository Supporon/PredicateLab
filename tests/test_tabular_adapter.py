"""Generic tabular adapter: fold-local fitting and real trial execution."""

import numpy as np
import pytest

from predicatelab.domains.tabular.adapter import (
    DescriptorError,
    TabularAdapter,
    fit_transform,
    validate_tabular_descriptor,
)
from predicatelab.core.adapter import TrialRequest


def _adapter():
    return TabularAdapter({"training": {"seed": 2025}, "protocol": {"cv_splits": 3, "cv_repeats": 1}})


def test_threshold_is_fitted_on_training_data_only():
    X_train = np.arange(100, dtype=float).reshape(-1, 1)
    X_valid = np.array([[1000.0], [-1000.0]])
    desc = {"op": "threshold", "a": 0, "quantile": 0.5}
    tr, va, params = fit_transform(desc, X_train, X_valid)
    assert params["threshold"] == np.quantile(X_train[:, 0], 0.5)
    # both extremes are above the training median; the valid data must not
    # change the threshold or its encoding
    assert va.tolist() == [[1.0], [0.0]]


def test_sigmoid_is_fitted_on_training_data_only():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    X_valid = np.array([[99.0]])
    tr, va, params = fit_transform({"op": "sigmoid", "a": 0, "scale": 2.0}, X_train, X_valid)
    assert params["median"] == np.median(X_train[:, 0])
    assert tr.shape == (4, 1) and va.shape == (1, 1)


@pytest.mark.parametrize("desc", [
    {"op": "ratio", "a": 0, "b": 0},
    {"op": "ratio", "a": -1, "b": 2},
    {"op": "threshold", "a": 0, "quantile": 1.5},
    {"op": "sigmoid", "a": 0, "scale": 100.0},
    {"op": "banana", "a": 0},
])
def test_descriptor_validation_rejects_bad_input(desc):
    with pytest.raises(DescriptorError):
        validate_tabular_descriptor(desc, n_features=30)


def test_kind_must_match_operator():
    adapter = _adapter()
    with pytest.raises(DescriptorError):
        adapter.validate_descriptor("conditional", {"op": "ratio", "a": 0, "b": 1})
    adapter.validate_descriptor("statistical", {"op": "ratio", "a": 0, "b": 1})


def test_trial_runs_and_uses_a_single_fold_at_a_time():
    adapter = _adapter()
    folds = adapter.build_protocol({}).folds
    fold = folds[0]
    req = TrialRequest(
        candidate_id="cand", variant="candidate",
        descriptor={"op": "ratio", "a": 0, "b": 1},
        fold=fold, seed=2025, training_config={}, workdir="",
        protocol_hash="p", config_hash="c",
    )
    result = adapter.run_trial(req)
    assert result["status"] == "success"
    assert np.isfinite(result["metrics"]["auc"])
    assert result["metric_series"]["log_loss"]
    assert result["fitted_params"]["op"] == "ratio"


def test_mutations_never_make_pairwise_operators_degenerate():
    from predicatelab.domains.tabular.generator import TabularRuleGenerator

    gen = TabularRuleGenerator({}, [f"f{i}" for i in range(30)], 30, direction="lower")
    feedback = {
        "validated": [{
            "candidate_id": "c1", "effect": -0.001,
            "descriptor": {"op": "difference", "a": 5, "b": 7},
        }],
        "insufficient": [], "screened_out": [],
    }
    proposals = gen.propose(round_index=1, n=10, feedback=feedback, known_template_hashes=set())
    assert proposals
    for t in proposals:
        d = t.descriptor
        if d["op"] in {"ratio", "product", "difference", "interaction"}:
            assert d["a"] != d["b"], d
        validate_tabular_descriptor(d, 30)


def test_holdout_is_separate_from_search_folds():
    adapter = _adapter()
    protocol = adapter.build_protocol({})
    search_ids = {f.fold_id for f in protocol.folds}
    assert protocol.holdout_fold is not None
    assert protocol.holdout_fold.fold_id not in search_ids
    assert protocol.holdout_fold.role == "holdout"
    search_valid = set()
    for f in protocol.folds:
        search_valid.update(f.metadata["valid_idx"])
    assert not (search_valid & set(protocol.holdout_fold.metadata["valid_idx"]))
