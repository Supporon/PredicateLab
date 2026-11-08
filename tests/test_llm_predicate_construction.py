"""Tests for LLM-driven TSIL predicate construction (DeepSeek integration).

Covers:
  - JSON extraction / repair from imperfect LLM output
  - Credential resolution precedence and the explicit ``enabled: false`` kill-switch
  - Symbolic descriptor validation, compilation, canonicalisation and deduplication
  - New atomic operator families (path_efficiency, hurst_vr, lead_lag_beta,
    regime_transition, tail_asymmetry)
  - LLM generator strategy separation (novel symbolic vs DSL composite vs hybrid)
    with the API disabled, proving the curated bank + rule fallback are safe
  - Descriptor sanitisation clamping out-of-range windows/slices
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from predicatelab.core.llm import DeepSeekLLMClient, extract_json_object, resolve_llm_credentials
from predicatelab.core.schema import CandidateKind, PredicateTemplate
from predicatelab.domains.tsil import dsl
from predicatelab.domains.tsil.adapter import _kernel_rank_and_ones_alignment
from predicatelab.domains.tsil.generator import (
    LLMGenerator,
    build_symbolic_and_composite_bank,
    make_generator,
)

NEW_OPERATOR_FAMILIES = (
    "path_efficiency",
    "hurst_vr",
    "lead_lag_beta",
    "regime_transition",
    "tail_asymmetry",
)


def _batch(b: int = 40, t: int = 20, f: int = 158, seed: int = 11) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=(b, t, f)).astype(np.float32))


# ---------------------------------------------------------------- JSON parsing
def test_extract_json_object_handles_fenced_and_trailing_commas() -> None:
    fenced = '```json\n{"proposals": [{"kind": "soft"},],}\n```'
    assert extract_json_object(fenced) == {"proposals": [{"kind": "soft"}]}
    prose = 'Sure. Here is the result:\n{"ok": true, "n": 2}\nHope that helps.'
    assert extract_json_object(prose) == {"ok": True, "n": 2}
    truncated = '{"round_analysis": "x", "proposals": [{"a": 1}, {"b": 2}, {"c"'
    salvaged = extract_json_object(truncated)
    assert isinstance(salvaged, dict)
    assert len(salvaged["proposals"]) == 2
    with pytest.raises(ValueError):
        extract_json_object("no json here at all")


# ------------------------------------------------------------ credentials
def test_credentials_enabled_false_disables_key() -> None:
    creds = resolve_llm_credentials({"enabled": False})
    assert creds["api_key"] == ""
    assert DeepSeekLLMClient({"enabled": False}).has_api_key is False


def test_credentials_env_placeholder_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_LLM_KEY", "unit-test-key-value")
    creds = resolve_llm_credentials({"api_key": "${MY_TEST_LLM_KEY}", "model": "deepseek-chat"})
    assert creds["api_key"] == "unit-test-key-value"


# ------------------------------------------------------- symbolic descriptors
def test_symbolic_descriptor_compiles_and_is_high_rank() -> None:
    desc = {
        "family": "symbolic",
        "params": {"window": 10},
        "channels": [
            {"time_op": "path_efficiency", "feature_op": "mean", "window": 5, "feature_slice": [29, 95]},
            {"time_op": "semivar_asymmetry", "feature_op": "std", "window": 10, "feature_slice": [9, 29]},
            {
                "time_op": "mean",
                "feature_op": "mean",
                "window": 10,
                "feature_slice": [9, 29],
                "rhs_slice": [95, 158],
                "interaction": "lead_lag_corr",
            },
        ],
    }
    dsl.validate_descriptor(desc, seq_len=20, feature_dim=158)
    fn = dsl.compile_descriptor(desc, seq_len=20, feature_dim=158)
    i_mat, p_mat = fn(_batch())
    assert p_mat.shape == (40, 40)
    assert torch.isfinite(p_mat).all()
    r_eff, align = _kernel_rank_and_ones_alignment(p_mat)
    assert r_eff >= 1.25 and align <= 0.98
    assert torch.allclose(torch.diagonal(p_mat).sum(), torch.tensor(1.0), atol=1e-4)


def test_symbolic_descriptor_canonicalisation_dedups_channel_order() -> None:
    a = {"time_op": "mean", "feature_op": "mean", "window": 5, "feature_slice": [0, 9]}
    b = {"time_op": "std", "feature_op": "iqr", "window": 10, "feature_slice": [9, 29]}
    c = {"time_op": "slope", "feature_op": "max", "window": 15, "feature_slice": [29, 95]}
    d1 = {"family": "symbolic", "params": {"window": 10}, "channels": [a, b, c]}
    d2 = {"family": "symbolic", "params": {"window": 10}, "channels": [c, a, b]}
    t1 = PredicateTemplate(domain="tsil", kind=CandidateKind.STATISTICAL,
                           name="s1", descriptor=d1).ensure_id()
    t2 = PredicateTemplate(domain="tsil", kind=CandidateKind.STATISTICAL,
                           name="s2", descriptor=d2).ensure_id()
    assert t1.content_hash == t2.content_hash


def test_symbolic_validation_rejects_bad_channel_fields() -> None:
    bad = {
        "family": "symbolic",
        "params": {"window": 10},
        "channels": [
            {"time_op": "mean", "feature_op": "mean"},
            {"time_op": "nope", "feature_op": "mean"},
        ],
    }
    with pytest.raises(dsl.DescriptorError):
        dsl.validate_descriptor(bad, seq_len=20, feature_dim=158)
    too_few = {"family": "symbolic", "params": {"window": 10},
               "channels": [{"time_op": "mean", "feature_op": "mean"}]}
    with pytest.raises(dsl.DescriptorError):
        dsl.validate_descriptor(too_few, seq_len=20, feature_dim=158)


# --------------------------------------------------------- atomic operators
def test_new_operator_families_compile_and_are_non_degenerate() -> None:
    x = _batch()
    for fam in NEW_OPERATOR_FAMILIES:
        desc = {"family": fam, "params": {"window": 10}}
        dsl.validate_descriptor(desc, seq_len=20, feature_dim=158)
        fn = dsl.compile_descriptor(desc, seq_len=20, feature_dim=158)
        i_mat, p_mat = fn(x)
        assert torch.isfinite(p_mat).all(), fam
        assert p_mat.shape == (40, 40)
        r_eff, align = _kernel_rank_and_ones_alignment(p_mat)
        assert r_eff >= 1.25, f"{fam} effective_rank={r_eff}"
        assert align <= 0.98, f"{fam} ones_kernel_alignment={align}"


def test_operator_family_requires_valid_window() -> None:
    with pytest.raises(dsl.DescriptorError):
        dsl.validate_descriptor({"family": "path_efficiency", "params": {"window": 2}}, seq_len=20)
    with pytest.raises(dsl.DescriptorError):
        dsl.validate_descriptor({"family": "hurst_vr", "params": {"window": 21}}, seq_len=20)


# ------------------------------------------------------------- generator
def _cfg(mode: str, tmp_path, **extra) -> dict:
    return {
        "run": {"name": f"t_{mode}", "domain": "tsil", "artifacts_dir": str(tmp_path)},
        "data": {"seq_len": 20, "feature_dim": 158},
        "training": {},
        "generator_config": {
            "mode": mode,
            "enabled": False,  # never hit the network in unit tests
            "max_tokens": 128,
            "max_retries": 0,
            **extra,
        },
    }


@pytest.mark.parametrize(
    "mode,symbolic_floor",
    [("novel_symbolic", 0.5), ("dsl_composite", 0.0), ("hybrid", 0.25)],
)
def test_llm_generator_strategy_composition(mode: str, symbolic_floor: float, tmp_path) -> None:
    cfg = _cfg(mode, tmp_path)
    gen = make_generator(cfg, {"generator": "llm", "generator_config": cfg["generator_config"]})
    assert isinstance(gen, LLMGenerator)
    tpls = gen.propose(round_index=0, n=14, feedback=None, known_template_hashes=set())
    assert len(tpls) == 14
    families = [t.descriptor.get("family") for t in tpls]
    assert any(f == "symbolic" for f in families)
    assert any(f == "composite" for f in families)
    # "Novel symbolic" covers the multi-channel symbolic family AND the LLM-designed
    # operator primitives; "DSL composite" is the composite family.
    novel = sum(
        1 for t in tpls
        if t.descriptor.get("family") == "symbolic"
        or t.descriptor.get("family") in NEW_OPERATOR_FAMILIES
    )
    n_symbolic = novel
    assert n_symbolic / len(families) >= symbolic_floor
    # All proposals must be statically valid and hash-distinct
    hashes = set()
    for t in tpls:
        dsl.validate_descriptor(t.descriptor, seq_len=20, feature_dim=158)
        assert t.content_hash not in hashes
        hashes.add(t.content_hash)


def test_dsl_composite_mode_prefers_composites(tmp_path) -> None:
    cfg = _cfg("dsl_composite", tmp_path)
    gen = make_generator(cfg, {"generator": "llm_composite", "generator_config": cfg["generator_config"]})
    tpls = gen.propose(round_index=0, n=12, feedback=None, known_template_hashes=set())
    families = [t.descriptor.get("family") for t in tpls]
    assert families.count("composite") >= 6


def test_generator_never_exceeds_n_and_respects_known_hashes(tmp_path) -> None:
    cfg = _cfg("hybrid", tmp_path)
    gen = make_generator(cfg, {"generator": "llm", "generator_config": cfg["generator_config"]})
    first = gen.propose(round_index=0, n=10, feedback=None, known_template_hashes=set())
    known = {t.content_hash for t in first}
    second = gen.propose(round_index=0, n=10, feedback=None, known_template_hashes=known)
    assert len(second) <= 10
    assert not ({t.content_hash for t in second} & known)


def test_descriptor_sanitisation_clamps_windows_and_slices(tmp_path) -> None:
    cfg = _cfg("hybrid", tmp_path)
    gen = make_generator(cfg, {"generator": "llm", "generator_config": cfg["generator_config"]})
    raw = {
        "family": "symbolic",
        "params": {"window": 999},
        "channels": [
            {"time_op": "mean", "feature_op": "mean", "window": 1, "feature_slice": [150, 900]},
            {"time_op": "bogus", "feature_op": "bogus", "interaction": "bogus"},
            {"time_op": "slope", "feature_op": "std", "window": 12},
        ],
    }
    clean = gen._sanitize_descriptor(raw)
    assert clean["params"]["window"] == 20
    ch = clean["channels"]
    assert len(ch) == 3
    assert all(3 <= int(c.get("window", 10)) <= 20 for c in ch)
    assert all(0 <= c["feature_slice"][0] < c["feature_slice"][1] <= 158
               for c in ch if "feature_slice" in c)
    assert ch[0]["time_op"] == "mean"
    dsl.validate_descriptor(clean, seq_len=20, feature_dim=158)


def test_bank_is_valid_and_hash_distinct() -> None:
    banks = build_symbolic_and_composite_bank(seq_len=20, feature_dim=158)
    seen: set[str] = set()
    for group in ("novel_symbolic", "dsl_composite"):
        for row in banks[group]:
            dsl.validate_descriptor(row["descriptor"], seq_len=20, feature_dim=158)
            t = PredicateTemplate(
                domain="tsil",
                kind=CandidateKind.COMPOSITE if row["descriptor"]["family"] == "composite" else CandidateKind.STATISTICAL,
                name=row["name"],
                descriptor=row["descriptor"],
            ).ensure_id()
            assert t.content_hash not in seen, row["name"]
            seen.add(t.content_hash)
    assert len(banks["novel_symbolic"]) >= 8
    assert len(banks["dsl_composite"]) >= 6
