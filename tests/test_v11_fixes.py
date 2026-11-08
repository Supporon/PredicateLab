"""Regression tests for the V1.1 audit defect cards (D-01 .. D-16).

Each test names the defect it pins down.  The tests use synthetic data for the
TSIL pieces (no GPU, no qlib data, and never the consumed 2024-01..2025-08
holdout) and the packaged breast-cancer dataset for engine-level tests.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

# Never initialise CUDA in the test process: the shared GPU nodes may be busy.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from predicatelab.core.engine import PredicateLabEngine, _selection_score
from predicatelab.core.identity import full_content_manifest
from predicatelab.core.label_window import validate_and_isolate
from predicatelab.core.protocol import ProtocolViolation
from predicatelab.core.schema import (
    EvidenceRecord,
    EvidenceStatus,
    ExperimentProtocol,
    FoldSpec,
)
from predicatelab.core.store import StateStore
from predicatelab.domains.tsil.adapter import TSILAdapter, _offdiag_stats
from predicatelab.domains.tsil.dsl import _normalise_phi
from predicatelab.domains.tsil.worker import _prediction_columns, _split_segments

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _generated_calendar(start: str = "2018-01-01", end: str = "2025-12-31") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _fake_tsil_provider(tmp_path: Path, *, feature_bytes: bytes = b"AAAA") -> Path:
    root = tmp_path / "provider"
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features" / "s1").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text(
        "\n".join(_generated_calendar()) + "\n", encoding="utf-8"
    )
    (root / "instruments" / "csi300.txt").write_text(
        "000001.SZ\t2018-01-01\t2025-12-31\n600000.SH\t2018-01-01\t2025-12-31\n",
        encoding="utf-8",
    )
    (root / "features" / "s1" / "a.bin").write_bytes(feature_bytes)
    return root


def _tsil_adapter(tmp_path: Path, provider: Path) -> TSILAdapter:
    return TSILAdapter({
        "run": {"artifacts_dir": str(tmp_path / "art128")},
        "data": {
            "provider_uri": str(provider),
            "tsil_root": str(REPO / "TSIL-main"),
            "market": "csi300",
            "seq_len": 20,
            "feature_dim": 158,
            "label": "Ref($close, -2) / Ref($close, -1) - 1",
        },
    })


def _engine_config(tmp_path: Path, *, require_confirmation: bool = False,
                   screen=("cv_r0_f0",), validation=("cv_r0_f1", "cv_r0_f2", "cv_r0_f3", "cv_r0_f4"),
                   max_candidates: int = 2) -> dict:
    return {
        "run": {
            "name": "v11_fixes",
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
            "screen_folds": list(screen),
            "validation_folds": list(validation),
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


def _validated_candidate(engine: PredicateLabEngine) -> str:
    rows = [e for e in engine.store.list_evidence(run_id=engine.run_id, stage="discovery")
            if e["status"] == EvidenceStatus.VALIDATED.value]
    assert rows, "relaxed thresholds should validate at least one candidate"
    return rows[0]["candidate_id"]


# ---------------------------------------------------------------------------
# D-01 / D-02: TSIL snapshot identity and verified content manifests
# ---------------------------------------------------------------------------
def test_tsil_snapshot_identity_is_stable_across_calls(tmp_path):
    adapter = _tsil_adapter(tmp_path, _fake_tsil_provider(tmp_path))
    first = adapter.snapshot({})
    time.sleep(0.01)
    second = adapter.snapshot({})

    assert first.manifest_hash == second.manifest_hash
    # The volatile fields still exist as observability metadata, but they are
    # not part of the identity hash.
    assert first.details["data_manifest_generated_at"] != second.details["data_manifest_generated_at"]
    assert "data_manifest_cache" in first.details
    assert "data_manifest_hash" in first.details


def test_tsil_snapshot_identity_ignores_volatile_metadata(tmp_path):
    adapter = _tsil_adapter(tmp_path, _fake_tsil_provider(tmp_path))
    snap = adapter.snapshot({})
    # Recompute the hash over the identity subset only.
    from predicatelab.core.hashing import sha256_obj

    identity = {k: v for k, v in snap.details.items()
                if k not in {"data_manifest_cache", "data_manifest_generated_at"}}
    assert sha256_obj(identity) == snap.manifest_hash


def test_verified_manifest_rehashes_same_size_same_mtime_edit(tmp_path):
    provider = _fake_tsil_provider(tmp_path, feature_bytes=b"AAAA")
    adapter = _tsil_adapter(tmp_path, provider)
    before = adapter.snapshot({})

    target = provider / "features" / "s1" / "a.bin"
    st = target.stat()
    target.write_bytes(b"BBBB")  # same size, different content
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))  # restore mtime

    after = adapter.snapshot({})
    assert after.manifest_hash != before.manifest_hash

    # The underlying identity is content-only: a size/mtime cache entry must
    # not be accepted when verify=True.
    manifest = full_content_manifest(
        provider,
        ["calendars/day.txt", "instruments/csi300.txt", "features/s1/a.bin"],
        cache_path=tmp_path / "cache.json",
        verify=True,
    )
    assert manifest["n_rehashed"] == 3


# ---------------------------------------------------------------------------
# D-03: prediction NPZ index levels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order", [("datetime", "instrument"), ("instrument", "datetime")])
def test_prediction_columns_detect_datetime_level(order):
    keys = [(pd.Timestamp("2024-01-02"), "000001.SZ"),
            (pd.Timestamp("2024-01-02"), "600000.SH"),
            (pd.Timestamp("2024-01-03"), "000001.SZ")]
    if order == ("datetime", "instrument"):
        index = pd.MultiIndex.from_tuples(keys, names=list(order))
    else:
        index = pd.MultiIndex.from_tuples([(b, a) for a, b in keys], names=list(order))
    frame = pd.DataFrame({"score": [0.1, 0.2, 0.3], "label": [1.0, 2.0, 3.0]}, index=index)

    instruments, dates = _prediction_columns(frame)
    assert list(instruments) == ["000001.SZ", "600000.SH", "000001.SZ"]
    assert list(dates) == ["2024-01-02", "2024-01-02", "2024-01-03"]


def test_prediction_columns_reject_non_date_index():
    frame = pd.DataFrame(
        {"score": [0.1, 0.2], "label": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")]),
    )
    with pytest.raises(ValueError):
        _prediction_columns(frame)


# ---------------------------------------------------------------------------
# D-04: strict off-diagonal degeneracy and zero (not ones) fallback
# ---------------------------------------------------------------------------
def test_constant_p_is_degenerate_on_strict_offdiagonal():
    constant = torch.ones(6, 6) * 0.7
    std, abs_max, spread = _offdiag_stats(constant)
    assert std < 1e-6
    assert abs_max > 0
    assert spread < 1e-6  # every pair is equally similar
    # The old computation included the diagonal zeros and is not the criterion.
    old = (constant - torch.diag(torch.diag(constant))).std().item()
    assert old > 1e-3


def test_degenerate_dsl_descriptor_returns_zero_not_ones():
    phi = _normalise_phi(torch.ones(8, 3))
    assert torch.count_nonzero(phi).item() == 0
    # P = phi phi^T is zero, so the probe rejects it instead of turning the
    # descriptor into the Phi_ones control.
    p = phi @ phi.T
    assert torch.count_nonzero(p).item() == 0


# ---------------------------------------------------------------------------
# D-05: search policy is part of experiment identity; evidence is immutable
# ---------------------------------------------------------------------------
def _protocol() -> ExperimentProtocol:
    fold = FoldSpec(fold_id="f1", train_start="2020-01-01", train_end="2020-12-31",
                    valid_start="2021-01-01", valid_end="2021-12-31")
    return ExperimentProtocol(
        task_id="t", domain="tabular", primary_metric="log_loss", direction="lower",
        folds=[fold], seeds=[1],
    )


def test_search_policy_changes_protocol_hash():
    policy_a = {"min_effect": 0.001, "fdr_alpha": 0.1}
    policy_b = {"min_effect": 0.01, "fdr_alpha": 0.1}
    base = _protocol().content_hash(training_config={"seed": 1})
    assert base == _protocol().content_hash(training_config={"seed": 1})
    a = _protocol().content_hash(training_config={"seed": 1}, search_config=policy_a)
    b = _protocol().content_hash(training_config={"seed": 1}, search_config=policy_b)
    assert a != b
    assert a != base


def test_evidence_row_cannot_be_overwritten_under_another_protocol(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        ev1 = EvidenceRecord(
            candidate_id="c1", round_id="r1", protocol_hash="p1",
            status=EvidenceStatus.INSUFFICIENT_EVIDENCE, stage="discovery",
        ).ensure_id()
        store.record_evidence(ev1)
        # Simulate the old failure: same row id, different protocol identity.
        ev2 = EvidenceRecord(
            candidate_id="c1", round_id="r1", protocol_hash="p2",
            status=EvidenceStatus.VALIDATED, stage="discovery",
        ).ensure_id()
        ev2.evidence_id = ev1.evidence_id
        with pytest.raises(ProtocolViolation, match="refusing to overwrite"):
            store.record_evidence(ev2)
        assert store.list_evidence()[0]["status"] == EvidenceStatus.INSUFFICIENT_EVIDENCE.value


def test_changing_search_policy_cannot_resume_same_run(tmp_path):
    cfg = _engine_config(tmp_path)
    engine = PredicateLabEngine(cfg, run_id="run_policy_identity")
    engine.run(rounds=1)
    engine.store.close()

    mutated = _engine_config(tmp_path)
    mutated["search"]["min_effect"] = 0.5
    other = PredicateLabEngine(mutated, run_id="run_policy_identity")
    with pytest.raises(ProtocolViolation, match="protocol"):
        other._build_context()
    # A rejected identity check must not leave a usable cached context: a
    # caller that swallows the exception and retries run() must be rejected
    # again, not proceed under the mutated policy.
    assert other._ctx is None
    with pytest.raises(ProtocolViolation, match="protocol"):
        other.run(rounds=1)
    other.store.close()


# ---------------------------------------------------------------------------
# D-06 / D-15: frozen family is immutable and records the full policy
# ---------------------------------------------------------------------------
def test_frozen_family_is_immutable_and_records_policy(tmp_path):
    engine = PredicateLabEngine(_engine_config(tmp_path), run_id="run_freeze_immutable")
    engine.run(rounds=1)
    cid = _validated_candidate(engine)
    manifest = engine.freeze_family(candidate_ids=[cid])

    assert manifest["search_policy"]["min_effect"] == -1.0
    assert manifest["policy_hash"]
    assert manifest["selection_rule"]["mode"] == "explicit_list"
    assert manifest["selection_rule"]["candidate_ids"] == [cid]

    before = engine.store.get_frozen_family(engine.run_id)["manifest_hash"]
    with pytest.raises(ProtocolViolation, match="immutable"):
        engine.freeze_family(candidate_ids=[cid])
    assert engine.store.get_frozen_family(engine.run_id)["manifest_hash"] == before

    # The store itself refuses a second write for the same run.
    with pytest.raises(ProtocolViolation, match="immutable"):
        engine.store.save_frozen_family(engine.run_id, manifest)
    engine.store.close()


# ---------------------------------------------------------------------------
# D-07: a failed finalize can be retried with the same frozen manifest
# ---------------------------------------------------------------------------
def test_finalize_retry_after_worker_crash_reuses_manifest_and_claim(tmp_path, capsys):
    engine = PredicateLabEngine(
        _engine_config(tmp_path, require_confirmation=False), run_id="run_finalize_retry",
    )
    engine.run(rounds=1)
    cid = _validated_candidate(engine)
    engine.freeze_family(candidate_ids=[cid])

    original = engine._execute_final

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated worker crash after holdout claim")

    engine._execute_final = boom
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        engine.finalize(cid)
    assert engine.store.get_run(engine.run_id)["status"] == "final_failed"
    failed = engine.store.get_final_eval_for(engine.run_id, cid)
    assert failed["status"] == "failed"

    engine._execute_final = original
    bundle = engine.finalize(cid)
    assert bundle["status"] == "success"
    assert bundle["manifest_hash"] == failed["manifest_hash"]
    assert engine.store.get_run(engine.run_id)["status"] == "final_complete"
    engine.store.close()


# ---------------------------------------------------------------------------
# D-08: early stopping and the reported metric use disjoint dates
# ---------------------------------------------------------------------------
def test_search_fold_carves_disjoint_early_stop_segment():
    calendar = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2019-01-01", "2022-12-31")]
    fold = {
        "fold_id": "f2022", "train_start": "2020-01-01", "train_end": "2021-12-31",
        "valid_start": "2022-01-01", "valid_end": "2022-12-31", "purge_days": 2,
    }
    segments, meta = _split_segments({
        "fold": fold, "data": {"seq_len": 20}, "calendar": calendar, "training": {},
    })

    assert segments["test"] == ["2022-01-01", "2022-12-31"]
    assert meta["mode"] == "carved_early_stop_from_train"
    assert meta["early_stop"] == segments["valid"]
    assert meta["reported"] == segments["test"]
    assert segments["valid"] != segments["test"]
    assert segments["train"][1] < segments["valid"][0]

    train_idx = calendar.index(segments["train"][1])
    es_idx = calendar.index(segments["valid"][0])
    assert es_idx - train_idx == fold["purge_days"] + 1


def test_holdout_fold_keeps_declared_test_segment():
    fold = {
        "fold_id": "holdout", "train_start": "2018-01-01", "train_end": "2022-12-31",
        "valid_start": "2023-01-01", "valid_end": "2023-12-31",
        "test_start": "2024-01-01", "test_end": "2025-08-31", "purge_days": 2,
    }
    segments, meta = _split_segments({
        "fold": fold, "data": {"seq_len": 20},
        "calendar": _generated_calendar("2018-01-01", "2025-12-31"), "training": {},
    })
    assert meta["mode"] == "declared_test"
    assert segments["test"] == ["2024-01-01", "2025-08-31"]


# ---------------------------------------------------------------------------
# D-09: freeze selection respects direction
# ---------------------------------------------------------------------------
def test_freeze_selection_score_is_direction_aware():
    assert _selection_score("lower", -0.5) > _selection_score("lower", -0.1)
    assert _selection_score("higher", 0.5) > _selection_score("higher", 0.1)
    assert _selection_score("lower", -0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# D-10: screen and validation folds must be disjoint
# ---------------------------------------------------------------------------
def test_overlapping_screen_and_validation_folds_are_rejected(tmp_path):
    cfg = _engine_config(tmp_path, validation=("cv_r0_f0", "cv_r0_f1"))
    engine = PredicateLabEngine(cfg, run_id="run_overlap")
    with pytest.raises(ProtocolViolation, match="disjoint"):
        engine._build_context()
    engine.store.close()


# ---------------------------------------------------------------------------
# D-11: per-date IC, not per-batch (per-instrument) IC
# ---------------------------------------------------------------------------
def _import_backbone():
    if str(REPO / "TSIL-main") not in sys.path:
        sys.path.insert(0, str(REPO / "TSIL-main"))
    from src.model_backbone_phitp import _evaluate_by_date, evaluate
    return _evaluate_by_date, evaluate


def test_evaluate_by_date_differs_from_pooled_correlation():
    _evaluate_by_date, evaluate = _import_backbone()
    dates = [pd.Timestamp("2024-01-02")] * 3 + [pd.Timestamp("2024-01-03")] * 3
    instruments = ["a", "b", "c"] * 2
    frame = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "label": [100.0, 101.0, 102.0, 0.0, 1.0, 2.0],
        },
        index=pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"]),
    )
    daily = _evaluate_by_date(frame)
    assert daily["n_dates"] == 2
    assert daily["IC"] == pytest.approx(1.0)
    pooled = float(evaluate(frame)["IC"])
    assert abs(pooled - 1.0) > 0.1  # a single pooled correlation is not the metric


# ---------------------------------------------------------------------------
# D-12: fixed tau removes the one-way alpha ratchet
# ---------------------------------------------------------------------------
def _backbone_model(**overrides):
    _import_backbone()
    from src.model_backbone_phitp import QniverseModel

    model_config = {
        "seq_len": 20, "pred_len": 1, "e_layers": 1, "factor": 3,
        "enc_in": 158, "c_out": 1, "d_model": 8, "d_ff": 8, "n_heads": 2,
        "dropout": 0.0, "use_attn": False, "noise_level": 0.0,
        "tau_hat_init": 0.0, "plot_show": False, "phi_type": "external",
        **overrides,
    }
    return QniverseModel(
        model_config=model_config, model_type="LSTM", lr=1e-3, n_epochs=1,
        early_stop=1, smooth_steps=1, max_steps_per_epoch=1, seed=1,
        logdir=None, eval_train=False, eval_test=False, plot_training=False,
        plot_show=False, phi_fn=None, save_artifacts=False,
    )


def test_fixed_tau_has_no_alpha_gradient_path():
    model = _backbone_model(tau_mode="fixed", fixed_tau=0.3)
    tau_hat, tau = model._tau()
    assert float(tau_hat) == pytest.approx(0.3)
    assert float(tau) == pytest.approx(0.7)
    assert not tau_hat.requires_grad
    assert model.tau_mode == "fixed"

    learned = _backbone_model(tau_mode="learned")
    assert learned._tau()[0].requires_grad


def test_unknown_or_invalid_fixed_tau_is_rejected():
    with pytest.raises(ValueError):
        _backbone_model(tau_mode="fixed")
    with pytest.raises(ValueError):
        _backbone_model(tau_mode="fixed", fixed_tau=1.0)
    with pytest.raises(ValueError):
        _backbone_model(tau_mode="mystery")


def test_shipped_tsil_configs_use_fixed_tau():
    """D-12 must be active in the configs, not only available in code."""
    for name in ("tsil_lstm_smoke.yaml", "tsil_lstm_quick.yaml", "tsil_lstm_full.yaml"):
        cfg = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
        model = cfg["model"]
        assert model.get("tau_mode") == "fixed", name
        tau = float(model["fixed_tau"])
        assert 0.0 < tau < 1.0, name


# ---------------------------------------------------------------------------
# D-13: cross-correlation P is trace-normalized and bounded
# ---------------------------------------------------------------------------
def test_cross_corr_trace_and_lambda_max():
    if str(REPO / "TSIL-main") not in sys.path:
        sys.path.insert(0, str(REPO / "TSIL-main"))
    from src.custom_phi_constructors import get_cross_correlation_I_P

    torch.manual_seed(0)
    x = torch.randn(16, 20, 158)
    _, p = get_cross_correlation_I_P(x, device="cpu")
    trace = float(torch.diagonal(p).sum())
    eig = torch.linalg.eigvalsh((p + p.T) / 2)
    assert trace == pytest.approx(1.0, abs=1e-4)
    assert float(eig.min()) >= -1e-4
    assert float(eig.max()) <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# D-16: holdout test-tail trimming is explicit and preserves the nominal window
# ---------------------------------------------------------------------------
def test_holdout_test_tail_is_trimmed_at_calendar_end():
    calendar = _generated_calendar("2020-01-01", "2024-12-31")
    search = FoldSpec(
        fold_id="f1", train_start="2020-01-01", train_end="2021-12-31",
        valid_start="2022-01-01", valid_end="2022-12-15", purge_days=2, embargo_days=3,
    )
    holdout = FoldSpec(
        fold_id="holdout", role="holdout",
        train_start="2020-01-01", train_end="2022-12-31",
        valid_start="2023-01-01", valid_end="2023-12-15",
        test_start="2024-01-01", test_end="2024-12-31",
        purge_days=2, embargo_days=3,
    )
    adjusted, adjusted_holdout, report = validate_and_isolate(
        [search], holdout, calendar,
        "Ref($close, -2) / Ref($close, -1) - 1", {"uses_future_days": 2},
    )
    assert adjusted_holdout is not None
    assert adjusted_holdout.test_end < "2024-12-31"
    test_report = report["folds"]["holdout"]["test"]
    assert test_report["nominal_test_end"] == "2024-12-31"
    assert test_report["effective_test_end"] == adjusted_holdout.test_end
    assert test_report["n_tail_labels_unavailable_dates"] >= 1


def test_tsil_holdout_fold_preserves_nominal_window_as_metadata(tmp_path):
    adapter = _tsil_adapter(tmp_path, _fake_tsil_provider(tmp_path))
    protocol = adapter.build_protocol({
        "protocol": {
            "purge_days": 2, "embargo_days": 3,
            "folds": [{
                "id": "f2022", "train": ["2020-01-01", "2021-12-31"],
                "valid": ["2022-01-01", "2022-12-31"],
            }],
            "holdout": {
                "id": "holdout", "train": ["2020-01-01", "2022-12-31"],
                "valid": ["2023-01-01", "2023-12-15"],
                "test": ["2024-01-01", "2024-12-31"],
            },
        },
        "seeds": [1],
    })
    holdout = protocol.holdout_fold
    assert holdout is not None
    assert holdout.metadata["nominal_test_start"] == "2024-01-01"
    assert holdout.metadata["nominal_test_end"] == "2024-12-31"
    assert holdout.metadata["label_future_offset_trading_days"] == 2
