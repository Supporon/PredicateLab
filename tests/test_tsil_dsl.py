"""TSIL descriptor validation and the trusted predicate compiler.

No qlib data is required: the compiler and native predicates are exercised on
small random tensors, on CPU.
"""

from pathlib import Path

import pytest
import torch

from predicatelab.domains.tsil import dsl

TSIL_ROOT = str(Path(__file__).resolve().parents[1] / "TSIL-main")


def _batch(b=8, t=20, f=158, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, f, generator=g)


@pytest.mark.parametrize("bad", [
    {"family": "momentum", "params": {"window": 10}, "extra": 1},
    {"family": "nope", "params": {"window": 10}},
    {"family": "momentum", "params": {}},
    {"family": "momentum", "params": {"window": 2}},
    {"family": "momentum", "params": {"window": 10}, "feature_slice": [5, 5]},
    {"family": "native", "predicate": "phi_not_a_predicate", "params": {}},
    {"family": "native", "predicate": "phi_momentum", "params": {"bogus": [5]}},
    {"family": "native", "predicate": "phi_technical",
     "params": {"short_window": 20, "long_window": 10}},
    {"family": "composite", "combine": "concat", "parts": [
        {"family": "native", "predicate": "phi_momentum", "params": {}},
    ]},
])
def test_invalid_descriptors_are_rejected(bad):
    with pytest.raises(dsl.DescriptorError):
        dsl.validate_descriptor(bad, seq_len=20, feature_dim=158)


def test_momentum_compiles_to_a_valid_similarity_matrix():
    desc = {"family": "momentum", "params": {"window": 5}, "feature_slice": [0, 8]}
    fn = dsl.compile_descriptor(desc, seq_len=20, feature_dim=158)
    i, p = fn(_batch(), None)
    assert i.shape == (8, 8) and p.shape == (8, 8)
    assert torch.isfinite(p).all()
    assert torch.allclose(p, p.T, atol=1e-5)
    assert float((p - torch.diag(torch.diag(p))).std()) > 0


def test_native_predicate_is_delegated_verbatim():
    desc = {"family": "native", "predicate": "phi_volatility",
            "params": {"window_sizes": [5, 10]}}
    fn = dsl.compile_descriptor(desc, tsil_root=TSIL_ROOT, seq_len=20, feature_dim=158)
    i, p = fn(_batch(), None)
    assert p.shape == (8, 8) and torch.isfinite(p).all()
    assert torch.allclose(p, p.T, atol=1e-4)


def test_timestamp_predicate_requires_calendar_features():
    desc = {"family": "native", "predicate": "phi_timestamp", "params": {}}
    assert dsl.needs_timestamp(desc)
    fn = dsl.compile_descriptor(desc, tsil_root=TSIL_ROOT, seq_len=20, feature_dim=158)
    with pytest.raises(dsl.DescriptorError, match="timestamp"):
        fn(_batch(), None)
    ts = torch.rand(8, 20, 3)
    i, p = fn(_batch(), ts)
    assert p.shape == (8, 8) and torch.isfinite(p).all()


def test_concat_composite_combines_dsl_parts():
    desc = {"family": "composite", "combine": "concat", "parts": [
        {"family": "momentum", "params": {"window": 5}, "feature_slice": [0, 6]},
        {"family": "volatility", "params": {"window": 10}, "feature_slice": [20, 30]},
    ]}
    fn = dsl.compile_descriptor(desc, seq_len=20, feature_dim=158)
    i, p = fn(_batch(), None)
    assert p.shape == (8, 8)
    assert torch.isfinite(p).all()


def test_probe_signature_uses_strict_upper_triangle():
    p = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    sig = dsl.probe_signature(p)
    assert len(sig) == 6
    assert sig == [1.0, 2.0, 3.0, 6.0, 7.0, 11.0]
