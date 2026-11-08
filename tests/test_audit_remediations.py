"""Regression tests for the 2026-11-07 credibility audit remediations (R01-R09)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from predicatelab.core.engine import PredicateLabEngine
from predicatelab.core.holdout import ConfirmationConsumed, HoldoutConsumed
from predicatelab.core.identity import code_version
from predicatelab.core.schema import (
    CandidateState,
    DataSnapshot,
    EvidenceRecord,
    EvidenceStatus,
    FoldSpec,
    RoundRecord,
)
from predicatelab.domains.tsil.adapter import TSILAdapter
from predicatelab.domains.tsil.dsl import (
    CROSS_SAMPLE_PREDICATES,
    NATIVE_PREDICATES,
    compile_descriptor,
    uses_cross_sample_coupling,
)
from predicatelab.domains.tsil.generator import RuleGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]
TSIL_ROOT = REPO_ROOT / "TSIL-main"
TSIL_SRC = TSIL_ROOT / "src"
for p in (str(TSIL_ROOT), str(TSIL_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def test_r01_r02_ts_slices_never_cross_instrument_boundary_and_eval_keeps_all_dates():
    from dataset_ori_onehot import _create_ts_slices

    # 3 instruments with 25 dates each (total 75 rows), seq_len=20.
    dates = pd.date_range("2022-01-03", periods=25, freq="B")
    insts = ["SH600000", "SH600004", "SH600006"]
    idx = pd.MultiIndex.from_product([insts, dates], names=["instrument", "datetime"])
    slices = _create_ts_slices(idx, seq_len=20)

    assert len(slices) == len(idx)
    for slc in slices:
        inst_in_window = idx.get_level_values("instrument")[slc]
        assert len(set(inst_in_window)) == 1, f"cross-instrument slice detected: {set(inst_in_window)}"


def test_r05_r06_all_23_native_predicates_are_batch_permutation_equivariant(tmp_path):
    torch.manual_seed(2025)
    B, T, D = 24, 20, 16
    x = torch.randn(B, T, D, dtype=torch.float32)
    ts = torch.randn(B, T, 5, dtype=torch.float32)
    perm = torch.randperm(B)
    x_perm = x[perm]
    ts_perm = ts[perm]

    for pred_name in sorted(NATIVE_PREDICATES):
        fn = compile_descriptor(
            {"family": "native", "predicate": pred_name, "params": {}},
            tsil_root=str(TSIL_ROOT),
        )
        I1, P1 = fn(x, ts)
        I2, P2 = fn(x_perm, ts_perm)
        P1_perm = P1.index_select(0, perm).index_select(1, perm)
        diff = float((P1_perm - P2).abs().max().item())
        assert diff < 1e-5, f"{pred_name} violated batch-row permutation equivariance: max_diff={diff}"

    (tmp_path / "calendars").mkdir(parents=True, exist_ok=True)
    (tmp_path / "calendars" / "day.txt").write_text("2022-01-04\n2022-01-05\n", encoding="utf-8")
    tsil_cfg = {
        "run": {"artifacts_dir": str(tmp_path / "art")},
        "data": {"provider_uri": str(tmp_path), "tsil_root": str(TSIL_ROOT), "seq_len": 20, "feature_dim": 16},
    }
    adapter = TSILAdapter(tsil_cfg)
    snap = DataSnapshot(
        domain="tsil",
        dataset_id="csi300",
        snapshot_id="s1",
        manifest_hash="m1",
        manifest_method="full_content",
        details={"data_manifest_hash": "dm1"},
    )
    fold = FoldSpec(fold_id="f2022", train_start="2020-01-01", train_end="2021-12-31",
                    valid_start="2022-01-01", valid_end="2022-12-31")
    batch_path = adapter._probe_batch_path(snap, fold)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(batch_path, x=x.numpy(), ts=ts.numpy())

    probe_cycle = adapter.probe(
        {"family": "native", "predicate": "phi_cycle", "params": {}},
        fold,
        snap,
        tsil_cfg,
        str(tmp_path / "probe_cycle"),
    )
    assert probe_cycle["ok"] is True
    assert probe_cycle["checks"]["permutation_equivariant"] is True
    assert probe_cycle["checks"]["batch_coupling"] == "pooled_minibatch_cross_sample"
    assert "phi_cycle" in CROSS_SAMPLE_PREDICATES
    assert uses_cross_sample_coupling({"family": "native", "predicate": "phi_momentum", "params": {}}) is False


def test_r03_r04_cross_db_holdout_and_confirmation_exposure_ledger(tmp_path):
    def _cfg(db_path: Path, strict_confirm: bool = False, rounds: int = 1) -> dict:
        return {
            "run": {
                "name": "ledger_test",
                "domain": "tabular",
                "artifacts_dir": str(tmp_path / f"art_{db_path.stem}"),
                "db": str(db_path),
                "rounds": rounds,
                "require_confirmation": True,
                "strict_confirmation_ledger": strict_confirm,
            },
            "seeds": [11, 13, 17],
            "data": {"dataset": "breast_cancer"},
            "protocol": {"cv_splits": 5, "cv_repeats": 2, "discovery_repeats": 1, "confirmation_kind": "resampling"},
            "evaluation": {"name": "tabular_cv_logloss", "primary_metric": "log_loss", "direction": "lower"},
            "search": {
                "generator": "rule",
                "max_candidates_per_round": 6,
                "screen_folds": ["cv_r0_f0"],
                "validation_folds": ["cv_r0_f1", "cv_r0_f2"],
                "full_validation_top_k": 2,
                "min_screen_effect": -0.05,
                "fdr_alpha": 0.99,
                "min_effect": -0.05,
                "min_stability": 0.0,
                "bootstrap_samples": 100,
                "bootstrap_block": 1,
                "nw_lag": 0,
                "seed": 11,
            },
        }

    db_a = tmp_path / "study_a.sqlite"
    db_b = tmp_path / "study_b.sqlite"

    with PredicateLabEngine(_cfg(db_a), run_id="run_a1") as eng_a1:
        eng_a1.run(rounds=1)
        ev_a1 = eng_a1.store.list_evidence(run_id="run_a1", stage="discovery")
        cands_a1 = [r["candidate_id"] for r in ev_a1 if r["n_folds"] > 0][:1]
        eng_a1.freeze_family(candidate_ids=cands_a1)
        conf_a1 = eng_a1.confirm()
        assert conf_a1["confirmation_reused"] is False
        fin_a1 = eng_a1.finalize(candidate_id=cands_a1[0])
        assert fin_a1["status"] == "success"
        assert fin_a1["exploratory"] is False

    # Same DB, new run: confirmation is downgraded to exploratory_positive and holdout is blocked.
    with PredicateLabEngine(_cfg(db_a), run_id="run_a2") as eng_a2:
        eng_a2.run(rounds=1)
        ev_a2 = eng_a2.store.list_evidence(run_id="run_a2", stage="discovery")
        cands_a2 = [r["candidate_id"] for r in ev_a2 if r["n_folds"] > 0][:1]
        eng_a2.freeze_family(candidate_ids=cands_a2)
        conf_a2 = eng_a2.confirm()
        assert conf_a2["confirmation_reused"] is True
        assert conf_a2["results"][0]["state"] in {
            CandidateState.EXPLORATORY_POSITIVE.value,
            CandidateState.INSUFFICIENT.value,
            CandidateState.REJECTED.value,
        }
        with pytest.raises(HoldoutConsumed):
            eng_a2.finalize(candidate_id=cands_a2[0])

    # Fresh DB in the same directory: global peer ledger still downgrades confirm() and blocks finalize().
    with PredicateLabEngine(_cfg(db_b), run_id="run_b1") as eng_b1:
        eng_b1.run(rounds=1)
        ev_b1 = eng_b1.store.list_evidence(run_id="run_b1", stage="discovery")
        cands_b1 = [r["candidate_id"] for r in ev_b1 if r["n_folds"] > 0][:1]
        eng_b1.freeze_family(candidate_ids=cands_b1)
        conf_b1 = eng_b1.confirm()
        assert conf_b1["confirmation_reused"] is True
        with pytest.raises(HoldoutConsumed):
            eng_b1.finalize(candidate_id=cands_b1[0])

    # Strict confirmation ledger mode raises ConfirmationConsumed on re-use.
    db_c = tmp_path / "study_c.sqlite"
    with PredicateLabEngine(_cfg(db_c, strict_confirm=True), run_id="run_c1") as eng_c1:
        eng_c1.run(rounds=1)
        ev_c1 = eng_c1.store.list_evidence(run_id="run_c1", stage="discovery")
        cands_c1 = [r["candidate_id"] for r in ev_c1 if r["n_folds"] > 0][:1]
        eng_c1.freeze_family(candidate_ids=cands_c1)
        with pytest.raises(ConfirmationConsumed):
            eng_c1.confirm()


def test_r07_generator_selects_parents_by_screen_effect_not_validation_effect():
    gen = RuleGenerator({"search": {"generator_config": {"mutation_top_k": 1}}})
    feedback = {
        "validated": [
            {
                "candidate_id": "cand_val_high_screen_low",
                "screen_effect": 0.001,
                "effect": 0.050,
                "descriptor": {"family": "volatility", "params": {"window": 10}},
            }
        ],
        "insufficient": [
            {
                "candidate_id": "cand_screen_high_val_low",
                "screen_effect": 0.020,
                "effect": 0.0005,
                "descriptor": {"family": "momentum", "params": {"window": 5}},
            }
        ],
        "screened_out": [],
    }
    parents = gen._parents(feedback)
    assert len(parents) == 1
    assert parents[0]["candidate_id"] == "cand_screen_high_val_low"
    assert parents[0]["family"] == "momentum"


def test_r07_run_wide_bh_fdr_across_multiple_rounds(tmp_path):
    from predicatelab.core.stats import benjamini_hochberg

    cfg = {
        "run": {
            "name": "r07_multi_round",
            "domain": "tabular",
            "artifacts_dir": str(tmp_path / "art"),
            "db": str(tmp_path / "r07.sqlite"),
            "rounds": 2,
        },
        "seeds": [11, 13, 17],
        "data": {"dataset": "breast_cancer"},
        "protocol": {"cv_splits": 5, "cv_repeats": 2, "discovery_repeats": 1, "confirmation_kind": "resampling"},
        "evaluation": {"name": "tabular_cv_logloss", "primary_metric": "log_loss", "direction": "lower"},
        "search": {
            "generator": "rule",
            "max_candidates_per_round": 4,
            "screen_folds": ["cv_r0_f0"],
            "validation_folds": ["cv_r0_f1", "cv_r0_f2"],
            "full_validation_top_k": 2,
            "min_screen_effect": -0.05,
            "fdr_alpha": 0.10,
            "min_effect": -0.05,
            "min_stability": 0.0,
            "bootstrap_samples": 100,
            "bootstrap_block": 1,
            "nw_lag": 0,
            "seed": 11,
        },
    }
    with PredicateLabEngine(cfg, run_id="r07_run") as eng:
        eng.run(rounds=2)
        ev_rows = [
            r for r in eng.store.list_evidence(run_id="r07_run", stage="discovery")
            if (r.get("metrics") or {}).get("full_validation")
        ]
        assert len(ev_rows) == 4
        expected_q = benjamini_hochberg([r["p_value"] for r in ev_rows])
        for row, q_exp in zip(ev_rows, expected_q):
            assert row["q_value"] == pytest.approx(q_exp, abs=1e-9)
        fb = eng.store.latest_feedback("r07_run")
        for bucket in ("validated", "insufficient", "rejected"):
            for item in fb.get(bucket, []):
                assert "screen_effect" in item and item["screen_effect"] is not None


def test_r08_code_version_detects_edits_without_git_and_hashes_external_roots(tmp_path):
    pkg = tmp_path / "repo" / "pkg"
    ext = tmp_path / "ext_tsil"
    pkg.mkdir(parents=True)
    ext.mkdir(parents=True)
    mod_file = pkg / "mod.py"
    ext_file = ext / "loss.py"
    mod_file.write_text("x = 1\n", encoding="utf-8")
    ext_file.write_text("tau = 0.5\n", encoding="utf-8")

    v1 = code_version(tmp_path / "repo", extra_roots=[ext])
    assert v1["commit"].startswith("tree:")
    mod_file.write_text("x = 2\n", encoding="utf-8")
    v2 = code_version(tmp_path / "repo", extra_roots=[ext])
    assert v2["dirty_fingerprint"] != v1["dirty_fingerprint"]
    assert v2["source_tree_hash"] != v1["source_tree_hash"]

    ext_file.write_text("tau = 0.25\n", encoding="utf-8")
    v3 = code_version(tmp_path / "repo", extra_roots=[ext])
    assert v3["dirty_fingerprint"] != v2["dirty_fingerprint"]


def test_r09_aggregate_reports_unique_n_dates_not_sum_across_seeds(tmp_path):
    cfg = {
        "run": {
            "name": "r09_test",
            "domain": "tabular",
            "artifacts_dir": str(tmp_path / "art"),
            "db": str(tmp_path / "r09.sqlite"),
            "rounds": 1,
        },
        "seeds": [2025, 2026, 2027],
        "data": {"dataset": "breast_cancer"},
        "protocol": {"cv_splits": 5, "cv_repeats": 2, "discovery_repeats": 1, "confirmation_kind": "resampling"},
        "evaluation": {"name": "tabular_cv_logloss", "primary_metric": "log_loss", "direction": "lower"},
        "search": {"screen_folds": ["cv_r0_f0"], "validation_folds": ["cv_r0_f1", "cv_r0_f2"]},
    }
    with PredicateLabEngine(cfg, run_id="r09_run") as eng:
        eng._build_context()
        per_seed = {
            2025: {
                "effect": -0.01, "ci_low": -0.02, "ci_high": -0.002, "p_value": 0.01,
                "stability": 1.0, "n_dates": 476, "n_folds": 2,
                "fold_effects": {"cv_r0_f1": -0.01, "cv_r0_f2": -0.01},
                "series": {f"d{i}": -0.01 for i in range(476)},
            },
            2026: {
                "effect": -0.011, "ci_low": -0.02, "ci_high": -0.002, "p_value": 0.01,
                "stability": 1.0, "n_dates": 476, "n_folds": 2,
                "fold_effects": {"cv_r0_f1": -0.011, "cv_r0_f2": -0.011},
                "series": {f"d{i}": -0.011 for i in range(476)},
            },
            2027: {
                "effect": -0.009, "ci_low": -0.02, "ci_high": -0.002, "p_value": 0.01,
                "stability": 1.0, "n_dates": 476, "n_folds": 2,
                "fold_effects": {"cv_r0_f1": -0.009, "cv_r0_f2": -0.009},
                "series": {f"d{i}": -0.009 for i in range(476)},
            },
        }
        agg = eng._aggregate(per_seed, n_expected_folds=2, failures=[])
        assert agg["n_dates"] == 476
        assert agg["n_seed_dates_total"] == 476 * 3


def test_real_path_parallel_ext_and_probe_batch_infra_error_and_final_state_str(tmp_path, monkeypatch):
    from dataset_ori_onehot import _patch_qlib_parallel_ext
    from qlib.utils import paral

    _patch_qlib_parallel_ext()
    p = paral.ParallelExt(n_jobs=2, backend="multiprocessing", maxtasksperchild=1)
    bag = getattr(p, "_backend_args", None) or getattr(p, "_backend_kwargs", None)
    assert isinstance(bag, dict) and bag.get("maxtasksperchild") == 1

    cfg = {
        "run": {
            "name": "probe_infra_test",
            "domain": "tabular",
            "artifacts_dir": str(tmp_path / "art"),
            "db": str(tmp_path / "probe_infra.sqlite"),
            "rounds": 1,
        },
        "seeds": [11],
        "data": {"dataset": "breast_cancer"},
        "protocol": {"cv_splits": 5, "cv_repeats": 2, "discovery_repeats": 1, "confirmation_kind": "resampling"},
        "evaluation": {"name": "tabular_cv_logloss", "primary_metric": "log_loss", "direction": "lower"},
        "search": {"screen_folds": ["cv_r0_f0"], "validation_folds": ["cv_r0_f1"]},
    }
    with PredicateLabEngine(cfg, run_id="probe_infra_run") as eng:
        ctx = eng._build_context()
        monkeypatch.setattr(
            ctx.adapter,
            "probe",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("probe batch failed (failed): worker crashed")),
        )
        with pytest.raises(RuntimeError, match="probe batch failed"):
            eng._probe({"family": "native"}, ctx.effective_folds[0], str(tmp_path / "pr"))

