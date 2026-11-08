"""Comprehensive regression tests for Stage 0-3 scientific protocol closure:
  - Stage 0: Git 1.8.3.1 compatible code_version()
  - Stage 1.1: SearchPolicy & _classify() require_positive_ci and require_beat_control
  - Stage 1.2: Cross-DB atomic global holdout/confirmation ledger + pre-seeded CSI300
  - Stage 1.3: Irreversible freeze_family() and code_fingerprint consistency enforcement
  - Stage 2.1: LUSI effective_rank & ones_kernel_alignment probe gate
  - Stage 2.2: Isospectral row-permuted control (control:perm) + subperiod IC + rank_autocorr
  - Stage 2.3: Cheverud-Nyholt effective_num_tests & Romano-Wolf step-down FWER p-values
  - Stage 2.4: Algebraic canonicalization of composite / DSL descriptors
  - Stage 3.1: Zero-dependency external verifier scripts/verify_evidence_independent.py
  - Stage 3.2: External time/group-ordered TabularAdapter -> independently_confirmed
  - Stage 3.3: New TSIL operators semivar_ratio and vol_price_corr
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predicatelab.core.engine import (
    HoldoutConsumed,
    PredicateLabEngine,
    ProtocolViolation,
    code_version,
    finalize_run,
)
from predicatelab.core.schema import CandidateKind, CandidateState, EvidenceStatus, PredicateTemplate, SearchPolicy
from predicatelab.core.stats import effective_num_tests, romano_wolf_stepdown
from predicatelab.domains.tsil.adapter import TSILAdapter, _kernel_rank_and_ones_alignment
from predicatelab.domains.tsil.dsl import canonicalize_descriptor, compile_descriptor
from predicatelab.domains.tsil.worker import _make_control_phi_fn, _prediction_metrics


def test_stage0_code_version_git_1_8_compatible() -> None:
    repo = Path(__file__).resolve().parents[1]
    cv = code_version(repo)
    assert len(cv["commit"]) == 40
    assert "unknown" not in cv["commit"]
    assert len(cv["source_tree_hash"]) == 64
    assert len(cv["dirty_fingerprint"]) == 64


def test_stage1_1_classify_require_positive_ci_and_beat_control() -> None:
    cfg = {
        "run": {"name": "s1_test", "domain": "tabular"},
        "data": {"dataset": "sklearn-breast-cancer"},
    }
    eng = PredicateLabEngine(cfg, db_path=":memory:", run_id="run_s1")
    try:
        eng._build_context()
        policy = SearchPolicy(
            min_effect=0.0005,
            min_stability=0.60,
            min_fold_coverage=0.75,
            min_seed_agreement=0.50,
            fdr_alpha=0.10,
            require_positive_ci=True,
            require_beat_control=True,
            min_control_delta=0.0,
        )
        # Case A: positive effect & p <= alpha, but ci_low <= 0 -> INSUFFICIENT_EVIDENCE
        agg_neg_ci = {
            "effect": 0.0009,
            "ci_low": -0.0020,
            "ci_high": 0.0038,
            "p_value": 0.01,
            "stability": 1.0,
            "fold_coverage": 1.0,
            "n_folds": 3,
            "seed_agreement": 1.0,
            "inferential": True,
        }
        assert eng._classify(agg_neg_ci, policy) == EvidenceStatus.INSUFFICIENT_EVIDENCE

        # Case B: positive CI, but fails to beat control:ones -> INSUFFICIENT_EVIDENCE
        agg_lose_ctrl = {
            "effect": 0.0030,
            "ci_low": 0.0010,
            "ci_high": 0.0050,
            "p_value": 0.005,
            "stability": 1.0,
            "fold_coverage": 1.0,
            "n_folds": 3,
            "seed_agreement": 1.0,
            "inferential": True,
            "control_variant": "control:ones",
            "control_effect": -0.0002,
            "beat_control": False,
        }
        assert eng._classify(agg_lose_ctrl, policy) == EvidenceStatus.INSUFFICIENT_EVIDENCE

        # Case C: both positive CI and beats control -> VALIDATED
        agg_good = {
            **agg_lose_ctrl,
            "control_effect": 0.0015,
            "beat_control": True,
            "min_control_delta": 0.0015,
        }
        assert eng._classify(agg_good, policy) == EvidenceStatus.VALIDATED
    finally:
        eng.store.close()


def test_stage2_1_and_3_3_kernel_rank_gate_and_new_tsil_operators(tmp_path: Path) -> None:
    import torch

    rng = np.random.default_rng(2025)
    # Construct a realistic (B=64, T=20, F=16) financial sequence batch
    X = rng.normal(0.0, 1.0, size=(64, 20, 16)).astype(np.float32)

    # 1. Near-constant descriptor -> rank-1 kernel collapse -> rejected
    p_collapsed = np.ones((64, 64), dtype=float) / 64.0
    r_eff_col, align_col = _kernel_rank_and_ones_alignment(p_collapsed)
    assert r_eff_col == pytest.approx(1.0, abs=1e-6)
    assert align_col == pytest.approx(1.0, abs=1e-6)

    # 2. Compile and evaluate new operators semivar_ratio & vol_price_corr
    for fam in ("semivar_ratio", "vol_price_corr"):
        desc = {"family": fam, "params": {"window": 10}}
        fn = compile_descriptor(desc, seq_len=20, feature_dim=16)
        i_mat, p_mat = fn(torch.from_numpy(X))
        assert p_mat.shape == (64, 64)
        assert torch.isfinite(p_mat).all()
        r_eff, align = _kernel_rank_and_ones_alignment(p_mat)
        assert r_eff >= 1.25, f"{fam} effective_rank={r_eff} < 1.25"
        assert align <= 0.98, f"{fam} ones_kernel_alignment={align} > 0.98"


def test_stage2_2_isospectral_control_perm_and_subperiod_metrics() -> None:
    import torch

    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.normal(size=(32, 15, 8)).astype(np.float32))
    desc = {"family": "semivar_ratio", "params": {"window": 10}}
    base_fn = compile_descriptor(desc, seq_len=15, feature_dim=8)
    perm_fn = _make_control_phi_fn(
        "control:perm", seed=2025, descriptor=desc, seq_len=15, feature_dim=8
    )

    _, p_orig = base_fn(X)
    _, p_perm = perm_fn(X)
    assert p_orig.shape == p_perm.shape == (32, 32)
    # Isospectral check: trace, Frobenius norm, and eigenvalues of P and P_perm are identical!
    eig_orig = np.linalg.eigvalsh(0.5 * (p_orig + p_orig.t()).numpy())
    eig_perm = np.linalg.eigvalsh(0.5 * (p_perm + p_perm.t()).numpy())
    np.testing.assert_allclose(eig_orig, eig_perm, atol=1e-5)
    # Sample alignment is broken
    assert not np.allclose(p_orig.numpy(), p_perm.numpy())

    # Sub-period IC & rank_autocorr check
    dates = []
    insts = []
    scores = []
    labels = []
    for d_idx in range(10):
        d_str = f"2024-01-{d_idx + 1:02d}"
        for i in range(20):
            dates.append(d_str)
            insts.append(f"SH6000{i:02d}")
            s = float(i) + 0.1 * d_idx
            scores.append(s)
            labels.append(s + rng.normal(0.0, 1.0))
    df = pd.DataFrame(
        {"score": scores, "label": labels},
        index=pd.MultiIndex.from_arrays([dates, insts], names=["datetime", "instrument"]),
    )
    metrics, series = _prediction_metrics(df)
    assert "ic_first_half" in metrics
    assert "ic_second_half" in metrics
    assert "subperiod_sign_consistent" in metrics
    assert "pred_rank_autocorr" in metrics
    assert "rank_autocorr" in metrics
    assert metrics["subperiod_sign_consistent"] == 1.0
    assert metrics["pred_rank_autocorr"] > 0.95


def test_stage2_3_effective_num_tests_and_romano_wolf() -> None:
    rng = np.random.default_rng(2025)
    base = rng.normal(0.01, 0.02, size=(100, 1))
    # 5 nearly identical candidates -> M_eff close to 1
    dup_mat = np.repeat(base, 5, axis=1) + rng.normal(0.0, 1e-6, size=(100, 5))
    m_eff_dup = effective_num_tests(dup_mat)
    assert 1.0 <= m_eff_dup < 1.05

    # 5 independent candidates -> M_eff close to 5
    ind_mat = rng.normal(0.0, 0.02, size=(200, 5))
    ind_mat[:, 0] += 0.015  # candidate 0 has strong positive effect
    m_eff_ind = effective_num_tests(ind_mat)
    assert m_eff_ind > 4.0

    rw_p = romano_wolf_stepdown(ind_mat, n_boot=400, block=5, seed=2025, nw_lag=3)
    assert len(rw_p) == 5
    assert all(0.0 <= p <= 1.0 for p in rw_p)
    assert rw_p[0] < 0.05


def test_stage2_4_algebraic_canonicalization_deduplication() -> None:
    a = {"family": "momentum", "params": {"window": 5}}
    b = {"family": "volatility", "params": {"window": 10}}

    t1 = PredicateTemplate(
        domain="tsil",
        kind=CandidateKind.COMPOSITE,
        name="mix_ab_scaled",
        descriptor={"family": "composite", "combine": "mix", "parts": [a, b], "weights": [2.0, 4.0]},
    ).ensure_id()
    t2 = PredicateTemplate(
        domain="tsil",
        kind=CandidateKind.COMPOSITE,
        name="mix_ba_unit",
        descriptor={"family": "composite", "combine": "mix", "parts": [b, a], "weights": [2.0, 1.0]},
    ).ensure_id()
    # Nested associative flattening: mix(mix(a, a), b) with weights -> combines identical 'a' parts
    t3 = PredicateTemplate(
        domain="tsil",
        kind=CandidateKind.COMPOSITE,
        name="nested_mix",
        descriptor={
            "family": "composite",
            "combine": "mix",
            "parts": [
                {"family": "composite", "combine": "mix", "parts": [a, a], "weights": [1.0, 1.0]},
                b,
            ],
            "weights": [1.0, 2.0],
        },
    ).ensure_id()

    assert t1.content_hash == t2.content_hash
    assert t1.content_hash == t3.content_hash
    assert t1.template_id == t2.template_id == t3.template_id


def test_stage3_1_and_3_2_external_tabular_e2e_and_zero_trust_verifier(tmp_path: Path) -> None:
    # Generate an external time-ordered panel dataset where a non-linear interaction
    # x0 * x1 strongly drives target y across all time periods (discovery, confirmation, holdout).
    rng = np.random.default_rng(2025)
    n_periods = 30
    samples_per_period = 20
    rows = []
    for t in range(n_periods):
        period_str = f"2024-{(t // 10) + 1:02d}-{(t % 10) + 1:02d}"
        for s in range(samples_per_period):
            x0 = float(rng.normal(0.0, 1.0))
            x1 = float(rng.normal(0.0, 1.0))
            x2 = float(rng.normal(0.0, 1.0))
            x3 = float(rng.normal(0.0, 1.0))
            # Strong quadratic/product signal x0 * x1 that linear Ridge without predicate cannot capture
            y = 0.2 * x0 - 0.1 * x1 + 2.5 * (x0 * x1) + float(rng.normal(0.0, 0.15))
            rows.append(
                {
                    "time_period": period_str,
                    "f0": x0,
                    "f1": x1,
                    "f2": x2,
                    "f3": x3,
                    "target": y,
                }
            )
    csv_path = tmp_path / "external_panel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "predicatelab.sqlite"
    bundle_dir = tmp_path / "bundle"

    cfg = {
        "run": {
            "name": "ext_tabular_e2e",
            "domain": "tabular",
            "artifacts_dir": str(artifacts_dir),
            "db": str(db_path),
            "rounds": 1,
            "require_confirmation": True,
        },
        "seeds": [2025, 2026],
        "task": {
            "title": "External time-forward tabular test",
            "question": "Does product(f0, f1) improve MSE on future periods?",
            "hypothesis": "x0 * x1 interaction generalizes across time_forward splits.",
        },
        "data": {
            "dataset": "synthetic_panel_2025",
            "path": str(csv_path),
            "target_col": "target",
            "time_col": "time_period",
            "task_type": "regression",
        },
        "training": {"seed": 2025, "alpha": 1.0},
        "protocol": {
            "cv_splits": 3,
            "confirmation_splits": 2,
            "confirmation_kind": "time_forward",
        },
        "evaluation": {
            "name": "tabular_regression",
            "primary_metric": "mse",
            "direction": "lower",
            "secondary_metrics": ["r2", "mae"],
            "paired": True,
            "multiple_testing": "benjamini-hochberg",
        },
        "search": {
            "generator": "rule",
            "save_predictions": True,
            "max_candidates_per_round": 8,
            "screen_folds": ["disc_f0"],
            "validation_folds": ["disc_f1", "disc_f2"],
            "full_validation_top_k": 4,
            "min_screen_effect": 0.0,
            "fdr_alpha": 0.10,
            "min_effect": 0.01,
            "min_stability": 0.60,
            "require_positive_ci": True,
            "require_beat_control": False,
            "bootstrap_samples": 400,
            "bootstrap_block": 2,
            "nw_lag": 2,
            "seed": 2025,
        },
    }

    eng = PredicateLabEngine(cfg, db_path=str(db_path), run_id="run_ext_tab")
    try:
        run_id = eng.run(rounds=1)
        assert run_id == "run_ext_tab"

        # Stage 1.3: freeze_family is irreversible and blocks subsequent discovery rounds
        frozen = eng.freeze_family(top_k=2)
        assert len(frozen["members"]) >= 1
        with pytest.raises(ProtocolViolation, match="already has frozen family"):
            eng.freeze_family(top_k=1)
        with pytest.raises(ProtocolViolation, match="frozen / post-discovery"):
            eng.run(rounds=2)

        # Confirmation on time_forward middle 20% window -> independently_confirmed
        conf_summary = eng.confirm()
        confirmed_cands = [
            r for r in conf_summary["results"] if r["state"] == CandidateState.INDEPENDENTLY_CONFIRMED.value
        ]
        assert len(confirmed_cands) >= 1
        best_cid = confirmed_cands[0]["candidate_id"]
        assert conf_summary["effective_num_tests"] >= 1.0

        # Finalize on unconsumed last 20% holdout window -> independently_confirmed
        final_bundle = eng.finalize(best_cid)
        assert final_bundle["status"] == "success"
        assert final_bundle["exploratory"] is False
        assert final_bundle["result"]["state"] == CandidateState.INDEPENDENTLY_CONFIRMED.value

        # Export immutable evidence bundle
        eng.export(bundle_dir)
    finally:
        eng.store.close()

    # Stage 3.1: Run zero-dependency external verifier script in a fresh subprocess
    verifier_script = Path(__file__).resolve().parents[1] / "scripts" / "verify_evidence_independent.py"
    proc = subprocess.run(
        [sys.executable, str(verifier_script), "--bundle", str(bundle_dir), "--db", str(db_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Verifier failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    ver_report = json.loads(proc.stdout)
    assert ver_report["ok"] is True
    assert ver_report["zero_predicatelab_imports"] is True
    for check_name, check_res in ver_report["checks"].items():
        assert check_res["ok"] is True, f"Check {check_name} failed: {check_res}"
    assert ver_report["checks"]["raw_predictions"]["max_abs_error"] < 1e-6
    assert ver_report["checks"]["independent_stats"]["max_abs_error"] < 1e-6
