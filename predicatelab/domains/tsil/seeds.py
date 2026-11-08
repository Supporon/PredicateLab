"""Round-1 seed predicates for the TSIL domain.

The 23 native predicates are exactly the ones shipped with ``TSIL-main``.  Six
of them are the paper's temporal predicates, the rest are exploratory
constructors from ``src/custom_phi_constructors.py``.  Native descriptors
delegate to the upstream implementation, so the mathematical mechanism is
unchanged.

The DSL seeds are new, parameterised statistical descriptors compiled by
``predicatelab.domains.tsil.dsl``.  They are deliberately modest in number:
the point of V1.0 is a trustworthy loop, not a benchmark sweep.
"""

from __future__ import annotations

from predicatelab.core.schema import CandidateKind, PredicateTemplate, Provenance

PAPER_SIX = [
    ("phi_timestamp", "Timestamp: calendar one-hot embedding", "paper: timestamp effect"),
    ("phi_cycle", "Market cycle: batch-level dominant period", "paper: market cycle"),
    ("phi_phase", "Asset phase: per-sample dominant period", "paper: asset phase"),
    ("phi_momentum", "Multi-horizon relative momentum", "paper: momentum"),
    ("phi_volatility", "Multi-window rolling volatility", "paper: volatility"),
    ("phi_return_dist", "Return distribution [mean, std, skew, kurt]", "paper: distribution shape"),
]

EXPLORATORY = [
    ("phi_trend", "Trend strength (ADX-style)", "exploratory: trend"),
    ("phi_multiscale", "Multi-scale temporal features", "exploratory: multi-scale"),
    ("phi_temporal", "Temporal pattern descriptor", "exploratory: temporal pattern"),
    ("phi_feature_corr", "Feature correlation structure", "exploratory: feature relatedness"),
    ("phi_price_volume", "Price-volume relationship", "exploratory: price-volume"),
    ("phi_factor", "Factor exposure", "exploratory: factor exposure"),
    ("phi_technical", "Composite technical indicators", "exploratory: technical"),
    ("phi_vol_return", "Volatility-return joint structure", "exploratory: vol-return"),
    ("phi_return_sim", "Return similarity clustering", "exploratory: cross-stock similarity"),
    ("phi_market_beta", "Market beta", "exploratory: market beta"),
    ("phi_cross_corr", "Cross-stock correlation", "exploratory: cross correlation"),
    ("phi_lead_lag", "Lead-lag relation", "exploratory: lead-lag"),
    ("phi_common_factor", "Common factor exposure", "exploratory: common factor"),
    ("phi_vol_sync", "Volatility synchronisation", "exploratory: vol sync"),
    ("phi_tail_risk", "Tail risk co-movement", "exploratory: tail risk"),
    ("phi_market_regime", "Market regime descriptor", "exploratory: regime"),
    ("phi_sector_cluster", "Sector clustering", "exploratory: sector cluster"),
]

DSL_SEEDS = [
    (CandidateKind.SOFT, "Momentum (window 5, cross-feature)", {"family": "momentum", "params": {"window": 5}}),
    (CandidateKind.SOFT, "Momentum (window 10, cross-feature)", {"family": "momentum", "params": {"window": 10}}),
    (CandidateKind.SOFT, "Momentum (window 20, cross-feature)", {"family": "momentum", "params": {"window": 20}}),
    (CandidateKind.STATISTICAL, "Volatility profile (window 10)", {"family": "volatility", "params": {"window": 10}}),
    (CandidateKind.STATISTICAL, "Volatility profile (window 20)", {"family": "volatility", "params": {"window": 20}}),
    (CandidateKind.STATISTICAL, "Cross-sectional dispersion (window 10)", {"family": "dispersion", "params": {"window": 10}}),
    (CandidateKind.STATISTICAL, "Return autocorrelation (window 20)", {"family": "autocorr", "params": {"window": 20}}),
    (CandidateKind.STATISTICAL, "Drawdown profile (window 20)", {"family": "drawdown", "params": {"window": 20}}),
    (CandidateKind.STATISTICAL, "Tail shape (window 20)", {"family": "tail", "params": {"window": 20}}),
    (CandidateKind.STATISTICAL, "Trend slope and fit (window 10)", {"family": "trend", "params": {"window": 10}}),
    (CandidateKind.COMPOSITE, "Momentum+Volatility+Trend composite",
     {"family": "composite", "combine": "concat", "parts": [
         {"family": "momentum", "params": {"window": 5}},
         {"family": "volatility", "params": {"window": 10}},
         {"family": "trend", "params": {"window": 10}},
     ], "weights": [1.0, 1.0, 1.0]}),
]


def native_template(predicate: str, name: str, rationale: str, params: dict | None = None) -> PredicateTemplate:
    return PredicateTemplate(
        domain="tsil",
        kind=CandidateKind.NATIVE,
        name=name,
        descriptor={"family": "native", "predicate": predicate, "params": params or {}},
        rationale=rationale,
        provenance=Provenance(generator="seed", seed_family=predicate, rationale=rationale),
    )


def seed_templates(pool: str = "paper6+dsl") -> list[PredicateTemplate]:
    """Return the ordered round-1 pool.

    ``paper6``      the six paper predicates
    ``paper6+dsl``  the six paper predicates followed by DSL descriptors
    ``all``         paper six + all exploratory + DSL
    ``dsl``         DSL descriptors only
    """
    templates: list[PredicateTemplate] = []
    if pool in {"paper6", "paper6+dsl", "all"}:
        for pred, name, why in PAPER_SIX:
            templates.append(native_template(pred, name, why))
    if pool == "all":
        for pred, name, why in EXPLORATORY:
            templates.append(native_template(pred, name, why))
    if pool in {"paper6+dsl", "dsl"}:
        for kind, name, desc in DSL_SEEDS:
            templates.append(PredicateTemplate(
                domain="tsil", kind=kind, name=name, descriptor=desc,
                rationale=f"DSL seed: {name}",
                provenance=Provenance(generator="seed", seed_family=desc["family"], rationale=name),
            ))
    return templates


def exploratory_templates() -> list[PredicateTemplate]:
    return [native_template(p, n, w) for p, n, w in EXPLORATORY]
