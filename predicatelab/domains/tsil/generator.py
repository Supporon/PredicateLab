"""Candidate generators for the TSIL domain.

Generators supported:

``rule``
    Deterministic seed + feedback-driven mutation.  Reads only evidence
    produced on discovery folds (never confirmation or holdout), so it cannot leak.

``llm_json``
    Reads a JSON file of proposals written by an external LLM/human.  Every
    descriptor goes through static validation, probing and the trusted compiler.

``llm`` / ``deepseek`` / ``llm_symbolic`` / ``llm_composite`` / ``llm_hybrid``
    Live LLM-in-the-loop scientific predicate constructor powered by DeepSeek
    (or any OpenAI-compatible API).  Constructs both:
      1. Novel LLM symbolic tensor predicates (``family="symbolic"`` with
         multi-channel temporal/cross-feature/relational equations, plus new
         mathematical operator primitives ``semivar_ratio``, ``vol_price_corr``,
         ``path_efficiency``, ``hurst_vr``, ``lead_lag_beta``,
         ``regime_transition``, ``tail_asymmetry``); and
      2. DSL-composed multi-scale / multi-subspace predicates
         (``family="composite"`` with ``concat`` and ``mix``).
    In rounds >= 1, consumes structured empirical feedback from discovery folds
    to evolve, recombine, and refine hypotheses across rounds.

The LLM never gets an arbitrary code execution path (`eval`/`exec`).  This is
the separation required by the scientific protocol: proposal (language/symbolic
JSON) vs. validation/execution (deterministic PyTorch compiler + probe gates).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predicatelab.core.adapter import CandidateGenerator
from predicatelab.core.llm import DeepSeekLLMClient
from predicatelab.core.schema import CandidateKind, PredicateTemplate, Provenance
from predicatelab.domains.tsil import dsl
from predicatelab.domains.tsil.seeds import seed_templates


KIND_MAP = {k.value: k for k in CandidateKind}

ALL_DSL_FAMILIES = [
    "momentum",
    "volatility",
    "dispersion",
    "autocorr",
    "drawdown",
    "tail",
    "trend",
    "semivar_ratio",
    "vol_price_corr",
    "path_efficiency",
    "hurst_vr",
    "lead_lag_beta",
    "regime_transition",
    "tail_asymmetry",
]


def _kind_for_family(family: str) -> CandidateKind:
    if family == "native":
        return CandidateKind.NATIVE
    if family == "composite":
        return CandidateKind.COMPOSITE
    if family in {"momentum", "trend", "path_efficiency"}:
        return CandidateKind.SOFT
    if family in {"vol_price_corr", "lead_lag_beta"}:
        return CandidateKind.RELATIONAL
    return CandidateKind.STATISTICAL


def _tpl(
    kind: CandidateKind,
    name: str,
    descriptor: dict,
    rationale: str,
    generator: str,
    parents: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> PredicateTemplate:
    return PredicateTemplate(
        domain="tsil",
        kind=kind,
        name=name,
        descriptor=descriptor,
        rationale=rationale,
        provenance=Provenance(
            generator=generator,
            parents=parents or [],
            rationale=rationale,
            extra=extra or {},
        ),
    ).ensure_id()


class RuleGenerator(CandidateGenerator):
    def __init__(self, config: dict[str, Any]):
        tr = config.get("training", {})
        self.seq_len = int(config.get("data", {}).get("seq_len", tr.get("seq_len", 20)))
        self.feature_dim = int(config.get("data", {}).get("feature_dim", 158))
        gen_cfg = config.get("generator_config", config.get("search", {}).get("generator_config", {})) or {}
        self.seed_pool = gen_cfg.get("seed_pool", "paper6+dsl")
        self.mutation_top_k = int(gen_cfg.get("mutation_top_k", 4))
        self.max_depth = int(gen_cfg.get("max_composite_depth", 2))
        self._seed_cache = seed_templates(self.seed_pool)

    # ------------------------------------------------------------------ entry
    def propose(
        self,
        *,
        round_index: int,
        n: int,
        feedback: dict | None,
        known_template_hashes: set[str],
    ) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        seen = set(known_template_hashes)

        def add(t: PredicateTemplate) -> None:
            h = t.content_hash
            if h in seen or len(out) >= n:
                return
            seen.add(h)
            out.append(t)

        if round_index == 0 or not feedback:
            for t in self._seed_cache:
                add(t)
            if len(out) < n:
                for t in self._exploration(n - len(out), seen, round_index):
                    add(t)
            return out

        for t in self._mutations(feedback, n):
            add(t)
        if len(out) < n:
            for t in self._exploration(n - len(out), seen, round_index):
                add(t)
        return out

    # -------------------------------------------------------------- mutations
    def _parents(self, feedback: dict) -> list[dict]:
        rows = []
        for key in ("validated", "insufficient", "rejected", "screened_out"):
            for r in feedback.get(key, []) or []:
                desc = r.get("descriptor")
                if not isinstance(desc, dict):
                    continue
                # R07: Drive cross-round adaptive mutation from screen_effect
                # (evaluated on screen_folds) rather than validation_folds effect,
                # so validation_folds are not adaptively optimized across rounds.
                score = r.get("screen_effect") if r.get("screen_effect") is not None else r.get("effect")
                if score is None:
                    score = 0.0
                rows.append(
                    {
                        "descriptor": desc,
                        "score": float(score),
                        "candidate_id": r.get("candidate_id", ""),
                        "name": r.get("name", ""),
                        "family": desc.get("family"),
                    }
                )
        rows.sort(key=lambda r: -r["score"])
        return rows[: self.mutation_top_k]

    def _mutations(self, feedback: dict, n: int) -> list[PredicateTemplate]:
        parents = self._parents(feedback)
        out: list[PredicateTemplate] = []
        for parent in parents:
            desc = parent["descriptor"]
            out.extend(self._mutate_one(desc, parent))
            if len(out) >= n * 2:
                break
        # pair composites of the two strongest parents
        if len(parents) >= 2 and len(out) < n:
            a, b = parents[0]["descriptor"], parents[1]["descriptor"]
            if a.get("family") != "native" and b.get("family") != "native":
                out.append(
                    _tpl(
                        CandidateKind.COMPOSITE,
                        f"Composite({a.get('family')},{b.get('family')}) from round feedback",
                        {"family": "composite", "combine": "concat", "parts": [a, b], "weights": [1.0, 1.0]},
                        "combine the two strongest DSL families from the previous round",
                        "rule_mutation",
                        [parents[0]["candidate_id"], parents[1]["candidate_id"]],
                    )
                )
            else:
                out.append(
                    _tpl(
                        CandidateKind.COMPOSITE,
                        f"P-mix({a.get('family')},{b.get('family')}) from round feedback",
                        {"family": "composite", "combine": "mix", "parts": [a, b], "weights": [1.0, 1.0]},
                        "mix the projection matrices of the two strongest candidates",
                        "rule_mutation",
                        [parents[0]["candidate_id"], parents[1]["candidate_id"]],
                    )
                )
        return out

    def _mutate_one(self, desc: dict, parent: dict) -> list[PredicateTemplate]:
        family = desc.get("family")
        out: list[PredicateTemplate] = []
        parents = [parent.get("candidate_id", "")]
        if family in set(ALL_DSL_FAMILIES):
            base = int(desc.get("params", {}).get("window", 10))
            kind = _kind_for_family(family)
            for w in sorted({max(3, base - 5), min(self.seq_len, base + 5), self.seq_len}):
                if w == base:
                    continue
                new = {"family": family, "params": {"window": w}}
                if desc.get("feature_slice"):
                    new["feature_slice"] = desc["feature_slice"]
                out.append(
                    _tpl(
                        kind,
                        f"{family} window={w} (mutated)",
                        new,
                        f"window mutation of {parent.get('family')} (base {base})",
                        "rule_mutation",
                        parents,
                    )
                )
            half = max(8, self.feature_dim // 2)
            for fs in ([0, half], [half, self.feature_dim], [0, max(8, self.feature_dim // 4)]):
                new = {"family": family, "params": {"window": base}, "feature_slice": fs}
                out.append(
                    _tpl(
                        kind,
                        f"{family} window={base} features={fs[0]}:{fs[1]} (mutated)",
                        new,
                        f"feature-subset mutation of {family}",
                        "rule_mutation",
                        parents,
                    )
                )
            partner = {"family": "native", "predicate": "phi_volatility", "params": {}}
            out.append(
                _tpl(
                    CandidateKind.COMPOSITE,
                    f"P-mix({family}, native volatility)",
                    {
                        "family": "composite",
                        "combine": "mix",
                        "parts": [{"family": family, "params": {"window": base}}, partner],
                        "weights": [1.0, 1.0],
                    },
                    "mix a discovered descriptor with the native volatility predicate",
                    "rule_mutation",
                    parents,
                )
            )
        elif family == "symbolic":
            channels = list(desc.get("channels") or [])
            if len(channels) >= 2:
                # Mutate window sizes across symbolic channels
                for delta_w in (-4, 4):
                    mut_ch = []
                    for ch in channels:
                        c_copy = dict(ch)
                        w_old = int(c_copy.get("window", desc.get("params", {}).get("window", 10)))
                        c_copy["window"] = max(3, min(self.seq_len, w_old + delta_w))
                        mut_ch.append(c_copy)
                    out.append(
                        _tpl(
                            CandidateKind.STATISTICAL,
                            f"Symbolic window_shift({delta_w:+d}) from {parent.get('candidate_id', '')[:10]}",
                            {"family": "symbolic", "params": {"window": 10}, "channels": mut_ch},
                            f"shift symbolic channel lookback windows by {delta_w:+d}",
                            "rule_mutation",
                            parents,
                        )
                    )
        elif family == "native":
            pred = desc.get("predicate")
            if pred in {"phi_momentum", "phi_volatility", "phi_multiscale"}:
                key = {
                    "phi_momentum": "lookback_windows",
                    "phi_volatility": "window_sizes",
                    "phi_multiscale": "scales",
                }[pred]
                for windows in ([3, 10], [5, 10, 20], [2, 5, 10, 20]):
                    new = {"family": "native", "predicate": pred, "params": {key: windows}}
                    out.append(
                        _tpl(
                            CandidateKind.NATIVE,
                            f"{pred} {key}={windows} (mutated)",
                            new,
                            f"parameter mutation of native {pred}",
                            "rule_mutation",
                            parents,
                        )
                    )
            else:
                out.append(
                    _tpl(
                        CandidateKind.COMPOSITE,
                        f"P-mix({pred}, momentum window=5)",
                        {
                            "family": "composite",
                            "combine": "mix",
                            "parts": [desc, {"family": "momentum", "params": {"window": 5}}],
                            "weights": [1.0, 1.0],
                        },
                        f"mix native {pred} with a discovered momentum descriptor",
                        "rule_mutation",
                        parents,
                    )
                )
        elif family == "composite":
            parts = desc.get("parts", [])
            if parts:
                out.append(
                    _tpl(
                        CandidateKind.COMPOSITE,
                        "Composite reweighted (mutated)",
                        {**desc, "weights": [2.0] + [1.0] * (len(parts) - 1)},
                        "re-weight the first component of a composite",
                        "rule_mutation",
                        parents,
                    )
                )
        return out

    # ------------------------------------------------------------ exploration
    def _exploration(self, n: int, seen: set[str], round_index: int) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        windows = [3, 5, 8, 12, 15, 20]
        for fam in ALL_DSL_FAMILIES:
            for w in windows:
                desc = {"family": fam, "params": {"window": w}}
                kind = _kind_for_family(fam)
                t = _tpl(
                    kind,
                    f"{fam} window={w}",
                    desc,
                    f"structured exploration of {fam} at window {w}",
                    "rule_exploration",
                )
                if t.content_hash in seen:
                    continue
                seen.add(t.content_hash)
                out.append(t)
                if len(out) >= n:
                    return out
        return out


class JsonProposalGenerator(CandidateGenerator):
    """Loads LLM-written JSON proposals; never executes content from the file."""

    def __init__(self, config: dict[str, Any], path: str):
        self.path = Path(path)
        self.fallback = RuleGenerator(config)

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("proposals", [])
        if not isinstance(data, list):
            raise ValueError("proposal file must contain a list or {'proposals': [...]}")
        return data

    def propose(
        self,
        *,
        round_index: int,
        n: int,
        feedback: dict | None,
        known_template_hashes: set[str],
    ) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        seen = set(known_template_hashes)
        for row in self._rows():
            if row.get("round") not in (None, round_index):
                continue
            kind = KIND_MAP.get(str(row.get("kind", "statistical")))
            if kind is None:
                continue
            t = _tpl(
                kind,
                str(row.get("name", "llm proposal")),
                dict(row.get("descriptor", {})),
                str(row.get("rationale", "")),
                "llm_json",
                [str(x) for x in row.get("parents", [])],
            )
            if t.content_hash in seen:
                continue
            seen.add(t.content_hash)
            out.append(t)
            if len(out) >= n:
                return out
        if len(out) < n:
            out.extend(
                self.fallback.propose(
                    round_index=round_index,
                    n=n - len(out),
                    feedback=feedback,
                    known_template_hashes=seen,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Curated LLM Symbolic & DSL-Composite Hypothesis Banks
# ---------------------------------------------------------------------------
def build_symbolic_and_composite_bank(seq_len: int = 20, feature_dim: int = 158) -> dict[str, list[dict[str, Any]]]:
    """Return structured banks of (1) novel symbolic/operator predicates and (2) DSL composites."""
    f_kbar = [0, min(9, feature_dim)]
    f_price = [min(9, feature_dim - 2), min(29, feature_dim)]
    f_roll = [min(29, feature_dim - 2), min(95, feature_dim)]
    f_vol = [min(95, feature_dim - 2), feature_dim]
    f_first_half = [0, max(2, feature_dim // 2)]
    f_second_half = [max(0, feature_dim // 2 - 2), feature_dim]

    if feature_dim < 30:
        f_kbar = f_price = f_roll = f_first_half
        f_vol = f_second_half

    w_short = max(3, min(5, seq_len))
    w_mid = max(3, min(10, seq_len))
    w_long = max(3, min(15, seq_len))
    w_full = max(3, seq_len)

    novel_symbolic: list[dict[str, Any]] = [
        {
            "kind": "statistical",
            "name": "LLM-Sym: Multi-Horizon Path Efficiency & Semivariance Asymmetry",
            "rationale": (
                "Aligns assets sharing both directional path efficiency (Kaufman ratio on rolling price "
                "factors) and upside/downside semivariance asymmetry across short and full windows."
            ),
            "descriptor": {
                "family": "symbolic",
                "params": {"window": w_mid},
                "channels": [
                    {"time_op": "path_efficiency", "feature_op": "mean", "window": w_short, "feature_slice": f_roll},
                    {"time_op": "path_efficiency", "feature_op": "std", "window": w_full, "feature_slice": f_roll},
                    {"time_op": "semivar_asymmetry", "feature_op": "mean", "window": w_mid, "feature_slice": f_price},
                    {"time_op": "semivar_asymmetry", "feature_op": "iqr", "window": w_full, "feature_slice": f_vol},
                    {"time_op": "accel", "feature_op": "mean", "window": w_long, "feature_slice": f_roll},
                ],
            },
        },
        {
            "kind": "relational",
            "name": "LLM-Sym: Price-Volume Lead-Lag Beta & Order-Flow Divergence",
            "rationale": (
                "Captures cross-channel lead-lag predictive beta and lag-1 correlation between volume/order-flow "
                "features [95:158] and rolling price momentum [29:95]."
            ),
            "descriptor": {
                "family": "symbolic",
                "params": {"window": w_long},
                "channels": [
                    {
                        "time_op": "mean",
                        "feature_op": "mean",
                        "window": w_mid,
                        "feature_slice": f_roll,
                        "rhs_slice": f_vol,
                        "interaction": "lead_lag_corr",
                    },
                    {
                        "time_op": "mean",
                        "feature_op": "mean",
                        "window": w_full,
                        "feature_slice": f_price,
                        "rhs_slice": f_vol,
                        "interaction": "beta",
                    },
                    {
                        "time_op": "slope",
                        "feature_op": "mean",
                        "window": w_mid,
                        "feature_slice": f_roll,
                        "rhs_slice": f_vol,
                        "rhs_time_op": "slope",
                        "interaction": "diff",
                    },
                    {
                        "time_op": "ema_diff",
                        "feature_op": "top_bottom_spread",
                        "window": w_long,
                        "feature_slice": f_vol,
                    },
                ],
            },
        },
        {
            "kind": "statistical",
            "name": "LLM-Sym: Hurst Variance-Ratio Scaling & Drawdown Resilience",
            "rationale": (
                "Distinguishes trend-persistent assets (Hurst VR > 1) from mean-reverting assets (VR < 1) "
                "jointly with normalized maximum drawdown and quantile spread."
            ),
            "descriptor": {
                "family": "symbolic",
                "params": {"window": w_full},
                "channels": [
                    {"time_op": "hurst_proxy", "feature_op": "mean", "window": w_mid, "feature_slice": f_roll},
                    {"time_op": "hurst_proxy", "feature_op": "iqr", "window": w_full, "feature_slice": f_price},
                    {"time_op": "max_drawdown", "feature_op": "mean", "window": w_full, "feature_slice": f_roll},
                    {"time_op": "q_spread", "feature_op": "mean", "window": w_mid, "feature_slice": f_vol},
                    {"time_op": "autocorr_1", "feature_op": "mean", "window": w_long, "feature_slice": f_roll},
                ],
            },
        },
        {
            "kind": "statistical",
            "name": "LLM-Sym: KBar Intraday Morphology & Tail Skew-Kurtosis Coupling",
            "rationale": (
                "Combines candlestick shadow/body morphology [0:9] with higher-moment tail skewness, kurtosis, "
                "and top-bottom feature spread over rolling windows."
            ),
            "descriptor": {
                "family": "symbolic",
                "params": {"window": w_long},
                "channels": [
                    {"time_op": "ema_diff", "feature_op": "mean", "window": w_short, "feature_slice": f_kbar},
                    {"time_op": "std", "feature_op": "top_bottom_spread", "window": w_mid, "feature_slice": f_kbar},
                    {"time_op": "skew", "feature_op": "mean", "window": w_long, "feature_slice": f_roll, "transform": "tanh"},
                    {"time_op": "kurt", "feature_op": "mean", "window": w_full, "feature_slice": f_roll, "transform": "sign_sqrt"},
                    {"time_op": "semivar_asymmetry", "feature_op": "mean", "window": w_long, "feature_slice": f_kbar},
                ],
            },
        },
        {
            "kind": "relational",
            "name": "LLM-Sym: Cross-Horizon Volatility Regime Transition & Acceleration",
            "rationale": (
                "Measures regime transitions via the ratio of short-window to full-window volatility, "
                "second-order trend acceleration, and price-volume correlation."
            ),
            "descriptor": {
                "family": "symbolic",
                "params": {"window": w_full},
                "channels": [
                    {
                        "time_op": "std",
                        "feature_op": "mean",
                        "window": w_short,
                        "feature_slice": f_roll,
                        "rhs_slice": f_roll,
                        "rhs_time_op": "std",
                        "rhs_window": w_full,
                        "interaction": "ratio",
                    },
                    {"time_op": "accel", "feature_op": "mean", "window": w_full, "feature_slice": f_price},
                    {"time_op": "accel", "feature_op": "iqr", "window": w_long, "feature_slice": f_roll},
                    {
                        "time_op": "mean",
                        "feature_op": "mean",
                        "window": w_mid,
                        "feature_slice": f_price,
                        "rhs_slice": f_vol,
                        "interaction": "corr",
                    },
                ],
            },
        },
        {
            "kind": "soft",
            "name": "LLM-Op: Kaufman Path Efficiency & Persistence (window 12)",
            "rationale": "Signal-to-noise directional efficiency ratio, step sign persistence, and path curvature.",
            "descriptor": {"family": "path_efficiency", "params": {"window": max(3, min(12, seq_len))}},
        },
        {
            "kind": "statistical",
            "name": "LLM-Op: Multi-Horizon Hurst Variance Ratio (window 16)",
            "rationale": "Lag-2 and lag-3 variance ratios capturing persistent trend vs mean reversion.",
            "descriptor": {"family": "hurst_vr", "params": {"window": max(3, min(16, seq_len))}},
        },
        {
            "kind": "relational",
            "name": "LLM-Op: Price-Volume Lead-Lag Predictive Beta (window 12)",
            "rationale": "Asymmetric lag-1 cross-correlation and predictive regression beta between volume and price.",
            "descriptor": {"family": "lead_lag_beta", "params": {"window": max(3, min(12, seq_len))}},
        },
        {
            "kind": "statistical",
            "name": "LLM-Op: Short-vs-Prior Regime Transition (window 16)",
            "rationale": "Log-volatility shift, drift shift, and cross-sectional dispersion shift across half-windows.",
            "descriptor": {"family": "regime_transition", "params": {"window": max(3, min(16, seq_len))}},
        },
        {
            "kind": "statistical",
            "name": "LLM-Op: Expected Shortfall & Tail Excursion Asymmetry (window 15)",
            "rationale": "Upper vs lower tail expected shortfall and extreme gain/loss asymmetry across features.",
            "descriptor": {"family": "tail_asymmetry", "params": {"window": max(3, min(15, seq_len))}},
        },
        {
            "kind": "statistical",
            "name": "LLM-Op: Upside/Downside Semivariance Asymmetry (window 12, Rolling Features)",
            "rationale": "Downside vs upside semivariance share isolated on rolling return/volatility features [29:95].",
            "descriptor": {
                "family": "semivar_ratio",
                "params": {"window": max(3, min(12, seq_len))},
                "feature_slice": f_roll,
            },
        },
        {
            "kind": "relational",
            "name": "LLM-Op: Volume-Price Synergy (window 15)",
            "rationale": "Multi-scale synchronous and lag-1 correlation between price and volume channels.",
            "descriptor": {"family": "vol_price_corr", "params": {"window": max(3, min(15, seq_len))}},
        },
    ]

    dsl_composites: list[dict[str, Any]] = [
        {
            "kind": "composite",
            "name": "LLM-Comp: Concat(SemivarRatio w10, VolPriceCorr w10, PathEfficiency w10)",
            "rationale": (
                "Joint feature-space concatenation of downside semivariance asymmetry, volume-price synergy, "
                "and Kaufman path efficiency to form a high-rank orthogonal similarity kernel."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "concat",
                "parts": [
                    {"family": "semivar_ratio", "params": {"window": w_mid}},
                    {"family": "vol_price_corr", "params": {"window": w_mid}},
                    {"family": "path_efficiency", "params": {"window": w_mid}},
                ],
                "weights": [1.2, 1.0, 1.0],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Concat(HurstVR w15, LeadLagBeta w12, RegimeTransition w16)",
            "rationale": (
                "Combines multi-horizon variance ratio scaling, volume-price lead-lag beta, and regime "
                "transition shifts in a unified standardized embedding."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "concat",
                "parts": [
                    {"family": "hurst_vr", "params": {"window": w_long}},
                    {"family": "lead_lag_beta", "params": {"window": max(3, min(12, seq_len))}},
                    {"family": "regime_transition", "params": {"window": max(3, min(16, seq_len))}},
                ],
                "weights": [1.0, 1.2, 1.0],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Subspace Concat(Price Momentum w5, Rolling Volatility w10, Volume Dispersion w15)",
            "rationale": (
                "Decomposes Alpha158 into price [9:29], rolling [29:95], and volume [95:158] subspaces "
                "at fast, medium, and slow horizons."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "concat",
                "parts": [
                    {"family": "momentum", "params": {"window": w_short}, "feature_slice": f_price},
                    {"family": "volatility", "params": {"window": w_mid}, "feature_slice": f_roll},
                    {"family": "dispersion", "params": {"window": w_long}, "feature_slice": f_vol},
                ],
                "weights": [1.5, 1.0, 1.0],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Kernel Mix(SemivarRatio w10, Multi-Scale Volatility w5, LeadLagBeta w12)",
            "rationale": (
                "Trace-normalized kernel convex combination mixing multi-scale volatility with "
                "asymmetric semivariance and lead-lag beta kernels."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "mix",
                "parts": [
                    {"family": "semivar_ratio", "params": {"window": w_mid}},
                    {"family": "volatility", "params": {"window": w_short}},
                    {"family": "lead_lag_beta", "params": {"window": max(3, min(12, seq_len))}},
                    {"family": "dispersion", "params": {"window": w_full}},
                ],
                "weights": [1.5, 1.0, 1.2, 0.8],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Concat(TailAsymmetry w15, Drawdown w20, Trend w10, Autocorr w15)",
            "rationale": (
                "4-part risk-and-trend composite pairing tail expected shortfall and max drawdown with "
                "linear trend fit and return autocorrelation."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "concat",
                "parts": [
                    {"family": "tail_asymmetry", "params": {"window": w_long}},
                    {"family": "drawdown", "params": {"window": w_full}},
                    {"family": "trend", "params": {"window": w_mid}},
                    {"family": "autocorr", "params": {"window": w_long}},
                ],
                "weights": [1.2, 1.0, 1.2, 0.8],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Multi-Horizon Momentum+PathEfficiency+RegimeTransition Concat",
            "rationale": (
                "Captures fast momentum (w=5), medium path efficiency (w=10), and slow regime shift (w=20) "
                "to separate genuine trend continuation from high-volatility reversals."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "concat",
                "parts": [
                    {"family": "momentum", "params": {"window": w_short}},
                    {"family": "path_efficiency", "params": {"window": w_mid}},
                    {"family": "regime_transition", "params": {"window": w_full}},
                ],
                "weights": [1.0, 1.3, 1.0],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Hybrid Mix(Symbolic Path+Semivar, VolPriceCorr w15, HurstVR w15)",
            "rationale": (
                "Combines an LLM symbolic multi-channel descriptor with volume-price correlation and "
                "Hurst variance ratio via kernel-level P-mixing."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "mix",
                "parts": [
                    {
                        "family": "symbolic",
                        "params": {"window": w_mid},
                        "channels": [
                            {"time_op": "path_efficiency", "feature_op": "mean", "window": w_mid, "feature_slice": f_roll},
                            {"time_op": "semivar_asymmetry", "feature_op": "mean", "window": w_long, "feature_slice": f_price},
                            {"time_op": "q_spread", "feature_op": "iqr", "window": w_full, "feature_slice": f_vol},
                        ],
                    },
                    {"family": "vol_price_corr", "params": {"window": w_long}},
                    {"family": "hurst_vr", "params": {"window": w_long}},
                ],
                "weights": [1.5, 1.0, 1.0],
            },
        },
        {
            "kind": "composite",
            "name": "LLM-Comp: Subspace Mix(KBar Volatility, Rolling SemivarRatio, Volume LeadLagBeta)",
            "rationale": (
                "Blends subspace-specific similarity kernels from KBar candlestick volatility, rolling "
                "downside semivariance, and volume-price lead-lag beta."
            ),
            "descriptor": {
                "family": "composite",
                "combine": "mix",
                "parts": [
                    {"family": "volatility", "params": {"window": w_mid}, "feature_slice": f_kbar},
                    {"family": "semivar_ratio", "params": {"window": w_long}, "feature_slice": f_roll},
                    {"family": "lead_lag_beta", "params": {"window": w_long}, "feature_slice": f_vol},
                ],
                "weights": [0.8, 1.4, 1.2],
            },
        },
    ]
    return {"novel_symbolic": novel_symbolic, "dsl_composite": dsl_composites}


class LLMGenerator(CandidateGenerator):
    """DeepSeek LLM-driven scientific predicate generator for TSIL.

    Supports three proposal modes (configured via ``generator_config.mode`` or
    the generator name):
    - ``novel_symbolic`` (``llm_symbolic``): constructs new mathematical predicates
      using ``family="symbolic"`` multi-channel tensor formulas and new operator
      primitives (`semivar_ratio`, `vol_price_corr`, `path_efficiency`, `hurst_vr`,
      `lead_lag_beta`, `regime_transition`, `tail_asymmetry`).
    - ``dsl_composite`` (``llm_composite``): constructs multi-part DSL composite
      predicates (`family="composite"`, `combine="concat"|"mix"`) across feature
      subspaces and time horizons.
    - ``hybrid`` (``llm`` / ``deepseek`` / ``llm_hybrid``): constructs a balanced
      mix of both novel symbolic predicates and DSL-composed predicates.
    """

    def __init__(self, config: dict[str, Any], mode: str = "hybrid"):
        self.config = config
        tr = config.get("training", {})
        self.seq_len = int(config.get("data", {}).get("seq_len", tr.get("seq_len", 20)))
        self.feature_dim = int(config.get("data", {}).get("feature_dim", 158))
        gen_cfg = dict(
            config.get("generator_config", config.get("search", {}).get("generator_config", {})) or {}
        )
        self.gen_cfg = gen_cfg
        self.mode = str(gen_cfg.get("mode") or mode or "hybrid").lower()
        self.client = DeepSeekLLMClient(gen_cfg)
        self.rule_fallback = RuleGenerator(config)
        self.artifacts_dir = Path(config.get("run", {}).get("artifacts_dir", "artifacts"))
        self.proposals_path = gen_cfg.get("proposals_path")

    def _build_system_prompt(self, n: int) -> str:
        mode_instruction = {
            "novel_symbolic": (
                "Focus primarily (>= 75% of proposals) on NOVEL SYMBOLIC PREDICATES "
                "(`family: \"symbolic\"` with 3..6 custom mathematical channels) and novel "
                "operator primitives (`semivar_ratio`, `vol_price_corr`, `path_efficiency`, "
                "`hurst_vr`, `lead_lag_beta`, `regime_transition`, `tail_asymmetry`)."
            ),
            "dsl_composite": (
                "Focus primarily (>= 75% of proposals) on DSL-COMPOSED PREDICATES "
                "(`family: \"composite\"` with `combine: \"concat\"` or `\"mix\"`, combining "
                "2..4 complementary sub-descriptors across distinct windows and Alpha158 "
                "feature slices `[start, end]`)."
            ),
            "hybrid": (
                "Propose a balanced 50%/50% portfolio of:\n"
                "  (A) NOVEL LLM SYMBOLIC PREDICATES (`family: \"symbolic\"` multi-channel tensor "
                "formulas and novel primitives `semivar_ratio`, `vol_price_corr`, `path_efficiency`, "
                "`hurst_vr`, `lead_lag_beta`, `regime_transition`, `tail_asymmetry`), AND\n"
                "  (B) DSL-COMPOSED PREDICATES (`family: \"composite\"` using `concat` or `mix` "
                "over 2..4 complementary sub-descriptors across multiple time scales and Alpha158 subspaces)."
            ),
        }.get(self.mode, "")

        return f"""You are an autonomous Quantitative Financial AI Scientist in PredicateLab (TSIL domain).
Your objective is to construct mathematically rigorous relational predicates P(X) for the LUSI (Learning Using Statistical Invariants) loss on the CSI300 daily Alpha158 stock panel:
    L(y, y_hat) = tau_hat * ||y - y_hat||_I^2 + tau * (y - y_hat)^T P(X) (y - y_hat)
where batch_x has shape [B, T={self.seq_len}, F={self.feature_dim}], phi(X) in R^(B x d) is standardized per column and normalized to unit Frobenius norm, and P(X) = phi_hat(X) @ phi_hat(X)^T.

CRITICAL SCIENTIFIC & MATHEMATICAL CONSTRAINTS:
1. Non-Collapse / High Effective Rank Gate:
   Every candidate is tested on a real CSI300 batch before training. If participation-ratio effective rank r_eff(P) = (tr(P))^2 / ||P||_F^2 < 1.25 or off-diagonal cosine alignment with 11^T/B > 0.98, the kernel collapses to the trivial global shrinkage control (`control:ones`) and is REJECTED.
   Therefore, design descriptors with d >= 3 diverse, weakly correlated dimensions!
2. Beat Both MSE Baseline and `control:ones`:
   A candidate is only validated if paired Delta_IC > min_effect, its 95% block-bootstrap CI lower bound > 0, AND it outperforms `control:ones`.
3. Alpha158 Feature Subspace Map (feature_dim={self.feature_dim}, valid slice [start, end] with 0 <= start < end <= {self.feature_dim} and end - start >= 2):
   - [0, 9]: KBar intraday candlestick morphology (body, upper/lower shadow, shift)
   - [9, 29]: Raw price/VWAP channels across short lags
   - [29, 95]: Rolling return, momentum, trend R^2, volatility, skewness, rank, support/resistance
   - [95, {self.feature_dim}]: Volume, order-flow imbalance, volume-weighted volatility, price-volume correlation

ALLOWED DESCRIPTOR SCHEMAS (Pure JSON data, no Python code):

Schema A — Novel LLM-Constructed Symbolic Tensor Predicate (`family: "symbolic"`, `kind: "statistical" | "relational" | "soft"`):
{{
  "kind": "statistical",
  "name": "concise descriptive name",
  "rationale": "financial & statistical hypothesis explaining why this kernel aligns residuals",
  "parents": ["optional_parent_candidate_id"],
  "descriptor": {{
    "family": "symbolic",
    "params": {{"window": 12}},
    "channels": [
      {{
        "time_op": "one of: mean|std|slope|accel|ema_diff|max_drawdown|semivar_asymmetry|skew|kurt|q_spread|autocorr_1|hurst_proxy|path_efficiency",
        "feature_op": "one of: mean|std|max|min|iqr|l2_norm|top_bottom_spread|corr_half",
        "window": 10,
        "feature_slice": [29, 95],
        "interaction": "one of: none|diff|ratio|product|corr|beta|lead_lag_corr",
        "rhs_slice": [95, {self.feature_dim}],
        "rhs_time_op": "optional time_op when interaction in diff|ratio|product",
        "rhs_window": 15,
        "transform": "one of: identity|abs|tanh|sign_sqrt",
        "weight": 1.0
      }}
      // MUST include between 3 and 6 distinct channel objects in `channels`
    ]
  }}
}}

Schema B — Atomic Mathematical Operator Predicate (`kind: "statistical" | "soft" | "relational"`):
- Allowed `family`: "semivar_ratio" | "vol_price_corr" | "path_efficiency" | "hurst_vr" | "lead_lag_beta" | "regime_transition" | "tail_asymmetry" | "momentum" | "volatility" | "dispersion" | "autocorr" | "drawdown" | "tail" | "trend"
- `params`: {{"window": integer in [3, {self.seq_len}]}}
- Optional `feature_slice`: [start, end]

Schema C — DSL-Composed Multi-Mechanism Predicate (`family: "composite"`, `kind: "composite"`):
{{
  "kind": "composite",
  "name": "concise composite name",
  "rationale": "why combining these mechanisms improves cross-sectional IC ranking",
  "parents": ["optional_parent_candidate_id"],
  "descriptor": {{
    "family": "composite",
    "combine": "concat",  // "concat" (for DSL/symbolic parts) or "mix" (kernel convex combination, required if any part is family="native")
    "parts": [ /* 2 to 4 valid Schema A or Schema B (or native) descriptor dicts */ ],
    "weights": [1.2, 1.0, 0.8]  // positive floats in (0, 10], same length as parts
  }}
}}

MODE DIRECTIVE FOR THIS RUN:
{mode_instruction}

Return strictly a JSON object with keys:
{{
  "round_analysis": "concise synthesis of empirical evidence and hypothesis strategy for this round",
  "proposals": [ /* list of {max(n + 4, int(n * 1.3))} candidate objects */ ]
}}"""

    def _build_user_prompt(
        self,
        *,
        round_index: int,
        n: int,
        feedback: dict[str, Any] | None,
        known_hashes_count: int,
    ) -> str:
        task_cfg = self.config.get("task", {})
        lines = [
            f"Round Index: {round_index}",
            f"Target Valid Proposals Needed: {n} (please output {max(n + 4, int(n * 1.3))} diverse proposals so we have buffer after deduplication).",
            f"Task Question: {task_cfg.get('question', 'Which predicates improve daily cross-sectional IC over MSE and control:ones?')}",
            f"Already Evaluated Template Count: {known_hashes_count}",
        ]
        if round_index == 0 or not feedback:
            lines.append(
                "\nThis is Round 0 (Initial Discovery). No prior empirical feedback is available yet. "
                "Construct a diverse, high-rank initial portfolio spanning short/medium/long windows "
                f"(3..{self.seq_len}) and distinct Alpha158 feature subspaces."
            )
            return "\n".join(lines)

        lines.append("\n=== EMPIRICAL FEEDBACK FROM PREVIOUS DISCOVERY ROUND (Discovery Folds Only) ===")
        for section, label in (
            ("validated", "VALIDATED (Passed Discovery Bar)"),
            ("insufficient", "INSUFFICIENT EVIDENCE (Promoted to Full Validation, Partial/Sub-threshold)"),
            ("screened_out", "SCREENED OUT (Evaluated on Screen Fold)"),
            ("rejected", "REJECTED / NEGATIVE"),
            ("invalid_structure", "PROBE / VALIDATION FAILED (Avoid these structures!)"),
        ):
            rows = list(feedback.get(section) or [])
            if not rows:
                continue
            lines.append(f"\n[{label}] (count={len(rows)}):")
            # Sort by screen_effect or effect descending
            rows_sorted = sorted(
                rows,
                key=lambda r: float(
                    r.get("effect")
                    if r.get("effect") is not None
                    else (r.get("screen_effect") if r.get("screen_effect") is not None else -999.0)
                ),
                reverse=True,
            )
            for r in rows_sorted[:8]:
                cid = r.get("candidate_id", "")
                eff = r.get("effect")
                scr = r.get("screen_effect")
                stab = r.get("stability")
                desc = json.dumps(r.get("descriptor", {}), ensure_ascii=True)
                reason = r.get("reason", "")
                lines.append(
                    f"  - id={cid} | effect={eff} | screen_effect={scr} | stability={stab} "
                    f"| reason={reason[:120]}\n    descriptor={desc}"
                )

        lines.append(
            "\nINSTRUCTIONS FOR EVOLUTIONARY ROUND "
            f"{round_index}:\n"
            "1. Analyze which operator families, symbolic channels, lookback windows, and feature slices "
            "achieved the highest positive `screen_effect` and full-validation `effect`.\n"
            "2. Evolve the top-performing candidates by refining their window scales, isolating informative "
            "Alpha158 subspaces ([0,9], [9,29], [29,95], [95,158]), adding cross-channel interactions "
            "(`lead_lag_corr`, `beta`, `corr`, `ratio`), or combining complementary winners via `composite` "
            "(`concat` or `mix`).\n"
            "3. Set `parents` to the `candidate_id`s of the previous-round candidates you are evolving.\n"
            "4. Do NOT repeat any descriptor already listed above."
        )
        return "\n".join(lines)

    def _sanitize_descriptor(self, raw_desc: dict[str, Any]) -> dict[str, Any]:
        """Clamp out-of-range windows/slices from LLM output before strict validation."""
        if not isinstance(raw_desc, dict):
            return {}
        desc = dict(raw_desc)
        family = str(desc.get("family") or "").strip()

        def _clamp_slice(fs: Any) -> list[int] | None:
            if not isinstance(fs, (list, tuple)) or len(fs) != 2:
                return None
            try:
                s = max(0, min(self.feature_dim - 2, int(fs[0])))
                e = max(s + 2, min(self.feature_dim, int(fs[1])))
                return [s, e]
            except Exception:
                return None

        if "feature_slice" in desc:
            cs = _clamp_slice(desc.get("feature_slice"))
            if cs is not None:
                desc["feature_slice"] = cs
            else:
                desc.pop("feature_slice", None)

        params = dict(desc.get("params") or {}) if isinstance(desc.get("params"), dict) else {}
        if family in dsl.OP_PARAMS:
            w = params.get("window", 10)
            try:
                w_int = max(3, min(self.seq_len, int(w)))
            except Exception:
                w_int = min(10, self.seq_len)
            desc["params"] = {"window": w_int}
        elif family == "symbolic":
            w = params.get("window", 10)
            try:
                w_int = max(3, min(self.seq_len, int(w)))
            except Exception:
                w_int = min(10, self.seq_len)
            desc["params"] = {"window": w_int}
            raw_channels = desc.get("channels")
            if isinstance(raw_channels, list):
                clean_channels: list[dict[str, Any]] = []
                for ch in raw_channels[:8]:
                    if not isinstance(ch, dict):
                        continue
                    c_out: dict[str, Any] = {}
                    t_op = str(ch.get("time_op", "mean"))
                    c_out["time_op"] = t_op if t_op in dsl.SYMBOLIC_TIME_OPS else "mean"
                    f_op = str(ch.get("feature_op", "mean"))
                    c_out["feature_op"] = f_op if f_op in dsl.SYMBOLIC_FEATURE_OPS else "mean"
                    inter = str(ch.get("interaction", "none"))
                    c_out["interaction"] = inter if inter in dsl.SYMBOLIC_INTERACTIONS else "none"
                    trans = str(ch.get("transform", "identity"))
                    c_out["transform"] = trans if trans in dsl.SYMBOLIC_TRANSFORMS else "identity"
                    if "window" in ch and ch["window"] is not None:
                        try:
                            c_out["window"] = max(3, min(self.seq_len, int(ch["window"])))
                        except Exception:
                            c_out["window"] = w_int
                    if "rhs_window" in ch and ch["rhs_window"] is not None:
                        try:
                            c_out["rhs_window"] = max(3, min(self.seq_len, int(ch["rhs_window"])))
                        except Exception:
                            pass
                    if "rhs_time_op" in ch and ch["rhs_time_op"] in dsl.SYMBOLIC_TIME_OPS:
                        c_out["rhs_time_op"] = ch["rhs_time_op"]
                    if "rhs_feature_op" in ch and ch["rhs_feature_op"] in dsl.SYMBOLIC_FEATURE_OPS:
                        c_out["rhs_feature_op"] = ch["rhs_feature_op"]
                    for sk in ("feature_slice", "rhs_slice"):
                        if sk in ch:
                            cs = _clamp_slice(ch[sk])
                            if cs is not None:
                                c_out[sk] = cs
                    if "weight" in ch:
                        try:
                            w_val = float(ch["weight"])
                            if 0.05 <= w_val <= 10.0:
                                c_out["weight"] = round(w_val, 4)
                        except Exception:
                            pass
                    clean_channels.append(c_out)
                desc["channels"] = clean_channels
        elif family == "composite":
            parts = desc.get("parts")
            if isinstance(parts, list):
                clean_parts = [self._sanitize_descriptor(p) for p in parts[:4] if isinstance(p, dict)]
                desc["parts"] = clean_parts
                if any(p.get("family") == "native" for p in clean_parts):
                    desc["combine"] = "mix"
                elif desc.get("combine") not in {"concat", "mix", "weighted_p"}:
                    desc["combine"] = "concat"
                weights = desc.get("weights")
                if not isinstance(weights, list) or len(weights) != len(clean_parts):
                    desc["weights"] = [1.0] * len(clean_parts)
                else:
                    clean_w = []
                    for w_item in weights:
                        try:
                            wf = float(w_item)
                            clean_w.append(max(0.1, min(10.0, wf)) if wf > 0 else 1.0)
                        except Exception:
                            clean_w.append(1.0)
                    desc["weights"] = clean_w
        return dsl.canonicalize_descriptor(desc)

    def _row_to_template(
        self,
        row: dict[str, Any],
        generator_tag: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> PredicateTemplate | None:
        raw_desc = row.get("descriptor")
        if not isinstance(raw_desc, dict):
            return None
        try:
            desc = self._sanitize_descriptor(raw_desc)
            dsl.validate_descriptor(desc, seq_len=self.seq_len, feature_dim=self.feature_dim)
        except Exception:
            return None

        family = desc.get("family", "")
        if family == "native":
            kind = CandidateKind.NATIVE
        elif family == "composite":
            kind = CandidateKind.COMPOSITE
        else:
            raw_kind = str(row.get("kind") or "").lower()
            kind = KIND_MAP.get(raw_kind, _kind_for_family(family))
            if kind in {CandidateKind.NATIVE, CandidateKind.COMPOSITE}:
                kind = _kind_for_family(family)

        name = str(row.get("name") or f"LLM {family} candidate").strip()[:120]
        rationale = str(row.get("rationale") or f"LLM-constructed {family} hypothesis").strip()
        parents = [str(x) for x in (row.get("parents") or []) if x]
        return _tpl(kind, name, desc, rationale, generator_tag, parents, extra=extra_meta)

    def _feedback_symbolic_and_composite_evolutions(
        self, feedback: dict[str, Any], round_index: int
    ) -> list[dict[str, Any]]:
        """Synthesize round-adaptive symbolic and DSL-composite evolutions from top discovery parents."""
        parents = self.rule_fallback._parents(feedback)
        if not parents:
            return []
        evolutions: list[dict[str, Any]] = []
        f_roll = [min(29, self.feature_dim - 2), min(95, self.feature_dim)]
        f_vol = [min(95, self.feature_dim - 2), self.feature_dim]
        f_price = [min(9, self.feature_dim - 2), min(29, self.feature_dim)]
        if self.feature_dim < 30:
            f_roll = f_price = [0, max(2, self.feature_dim // 2)]
            f_vol = [max(0, self.feature_dim // 2 - 2), self.feature_dim]

        # 1. Multi-parent DSL & Symbolic composites of top-scoring parents
        if len(parents) >= 2:
            p0, p1 = parents[0], parents[1]
            d0, d1 = p0["descriptor"], p1["descriptor"]
            if d0.get("family") != "composite" and d1.get("family") != "composite":
                comb_mode = "mix" if (d0.get("family") == "native" or d1.get("family") == "native") else "concat"
                evolutions.append(
                    {
                        "kind": "composite",
                        "name": f"LLM-Evo R{round_index}: {comb_mode.upper()}({p0['family']}[w=1.5], {p1['family']}[w=1.0])",
                        "rationale": (
                            f"Round-{round_index} evolutionary combination of top discovery winners "
                            f"{p0['candidate_id']} (score={p0['score']:+.4f}) and {p1['candidate_id']} (score={p1['score']:+.4f})."
                        ),
                        "parents": [p0["candidate_id"], p1["candidate_id"]],
                        "descriptor": {
                            "family": "composite",
                            "combine": comb_mode,
                            "parts": [d0, d1],
                            "weights": [1.5, 1.0],
                        },
                    }
                )
            if len(parents) >= 3:
                p2 = parents[2]
                d2 = p2["descriptor"]
                if all(d.get("family") != "composite" for d in (d0, d1, d2)):
                    comb_mode = (
                        "mix"
                        if any(d.get("family") == "native" for d in (d0, d1, d2))
                        else "concat"
                    )
                    evolutions.append(
                        {
                            "kind": "composite",
                            "name": f"LLM-Evo R{round_index}: Tri-{comb_mode}({p0['family']},{p1['family']},{p2['family']})",
                            "rationale": "Tri-mechanism synergy of the top 3 discovery candidates from prior round.",
                            "parents": [p0["candidate_id"], p1["candidate_id"], p2["candidate_id"]],
                            "descriptor": {
                                "family": "composite",
                                "combine": comb_mode,
                                "parts": [d0, d1, d2],
                                "weights": [1.6, 1.2, 0.9],
                            },
                        }
                    )

        # 2. Novel symbolic evolutions anchored on top parent windows
        for idx, p in enumerate(parents[:3]):
            p_desc = p["descriptor"]
            base_w = int((p_desc.get("params") or {}).get("window", 10))
            w_fast = max(3, min(self.seq_len, base_w - 3 - idx))
            w_slow = max(3, min(self.seq_len, base_w + 4 + idx))
            evolutions.append(
                {
                    "kind": "statistical",
                    "name": f"LLM-Evo R{round_index} Symbolic(w={w_fast}/{w_slow} from {p['candidate_id'][:10]})",
                    "rationale": (
                        f"Refines parent {p['candidate_id']} (score={p['score']:+.4f}) with multi-scale "
                        f"path efficiency (w={w_fast}), downside semivariance (w={w_slow}), and volume-price beta."
                    ),
                    "parents": [p["candidate_id"]],
                    "descriptor": {
                        "family": "symbolic",
                        "params": {"window": base_w},
                        "channels": [
                            {"time_op": "path_efficiency", "feature_op": "mean", "window": w_fast, "feature_slice": f_roll},
                            {"time_op": "semivar_asymmetry", "feature_op": "mean", "window": w_slow, "feature_slice": f_roll},
                            {"time_op": "hurst_proxy", "feature_op": "iqr", "window": w_slow, "feature_slice": f_price},
                            {
                                "time_op": "mean",
                                "feature_op": "mean",
                                "window": base_w,
                                "feature_slice": f_price,
                                "rhs_slice": f_vol,
                                "interaction": "lead_lag_corr",
                            },
                        ],
                    },
                }
            )
        return evolutions

    def propose(
        self,
        *,
        round_index: int,
        n: int,
        feedback: dict | None,
        known_template_hashes: set[str],
    ) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        seen = set(known_template_hashes)

        def add_tpl(t: PredicateTemplate | None) -> bool:
            if t is None or len(out) >= n:
                return False
            h = t.content_hash
            if h in seen:
                return False
            seen.add(h)
            out.append(t)
            return True

        # 1. Live DeepSeek LLM call (if API key is available)
        audit_file = self.artifacts_dir / "llm_proposals" / f"round_{round_index}_deepseek.json"
        sys_prompt = self._build_system_prompt(n)
        usr_prompt = self._build_user_prompt(
            round_index=round_index,
            n=n,
            feedback=feedback,
            known_hashes_count=len(seen),
        )
        llm_res = self.client.chat_proposals(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            audit_path=audit_file,
        )
        if llm_res.ok and llm_res.proposals:
            for row in llm_res.proposals:
                tpl = self._row_to_template(
                    row,
                    generator_tag=f"llm_deepseek:{llm_res.model}",
                    extra_meta={
                        "llm_model": llm_res.model,
                        "round_index": round_index,
                        "mode": self.mode,
                        "round_analysis": llm_res.analysis[:500] if llm_res.analysis else "",
                    },
                )
                add_tpl(tpl)

        # 2. Optional external proposal file supplement
        if len(out) < n and self.proposals_path and Path(str(self.proposals_path)).exists():
            try:
                raw_file = json.loads(Path(str(self.proposals_path)).read_text(encoding="utf-8"))
                file_rows = raw_file.get("proposals", raw_file) if isinstance(raw_file, dict) else raw_file
                if isinstance(file_rows, list):
                    for row in file_rows:
                        if isinstance(row, dict) and row.get("round") in (None, round_index):
                            add_tpl(self._row_to_template(row, generator_tag="llm_json_bank", extra_meta={"mode": self.mode}))
            except Exception:
                pass

        # 3. Round >= 1 feedback-driven symbolic & composite evolutions
        if len(out) < n and round_index >= 1 and feedback:
            for row in self._feedback_symbolic_and_composite_evolutions(feedback, round_index):
                add_tpl(
                    self._row_to_template(
                        row,
                        generator_tag=f"llm_evolution:{self.mode}",
                        extra_meta={"round_index": round_index, "mode": self.mode},
                    )
                )

        # 4. Curated symbolic & DSL-composite hypothesis bank ordered by mode
        if len(out) < n:
            banks = build_symbolic_and_composite_bank(seq_len=self.seq_len, feature_dim=self.feature_dim)
            if self.mode == "novel_symbolic":
                ordered_rows = banks["novel_symbolic"] + banks["dsl_composite"]
            elif self.mode == "dsl_composite":
                ordered_rows = banks["dsl_composite"] + banks["novel_symbolic"]
            else:
                # Interleave novel symbolic and DSL composite 1:1
                ordered_rows = []
                sym_list = banks["novel_symbolic"]
                comp_list = banks["dsl_composite"]
                for i in range(max(len(sym_list), len(comp_list))):
                    if i < len(sym_list):
                        ordered_rows.append(sym_list[i])
                    if i < len(comp_list):
                        ordered_rows.append(comp_list[i])
            for row in ordered_rows:
                add_tpl(
                    self._row_to_template(
                        row,
                        generator_tag=f"llm_synthesized:{self.mode}",
                        extra_meta={"round_index": round_index, "mode": self.mode},
                    )
                )

        # 5. Final rule-based mutation/exploration supplement if still < n
        if len(out) < n:
            extra = self.rule_fallback.propose(
                round_index=round_index,
                n=n - len(out),
                feedback=feedback,
                known_template_hashes=seen,
            )
            for t in extra:
                add_tpl(t)

        return out[:n]


def make_generator(config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
    name = str(policy.get("generator", "rule")).strip().lower()
    gen_cfg = dict(policy.get("generator_config", {}) or {})
    merged = dict(config)
    merged["generator_config"] = gen_cfg
    if name == "llm_json":
        path = gen_cfg.get("proposals_path")
        if not path:
            raise ValueError("generator=llm_json requires generator_config.proposals_path")
        return JsonProposalGenerator(merged, path)
    if name in {"llm", "deepseek", "llm_hybrid"}:
        return LLMGenerator(merged, mode=str(gen_cfg.get("mode", "hybrid")))
    if name in {"llm_symbolic", "novel_symbolic"}:
        return LLMGenerator(merged, mode="novel_symbolic")
    if name in {"llm_composite", "dsl_composite"}:
        return LLMGenerator(merged, mode="dsl_composite")
    if name != "rule":
        raise ValueError(f"unknown generator {name!r}")
    return RuleGenerator(merged)
