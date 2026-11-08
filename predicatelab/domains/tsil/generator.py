"""Candidate generators for the TSIL domain.

Two generators are provided:

``rule``
    Deterministic seed + feedback-driven mutation.  It reads only evidence
    produced on validation folds (never the holdout), so it cannot leak.

``llm_json``
    Reads a JSON file of proposals written by an external LLM/human.  The file
    is data: every descriptor still goes through static validation, probing and
    the trusted compiler.  If no proposals are supplied for a round, the
    generator falls back to rule-based mutations.

The LLM never gets an execution path.  This is the separation required by the
protocol: proposal (language) vs. validation/execution (deterministic code).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predicatelab.core.adapter import CandidateGenerator
from predicatelab.core.schema import CandidateKind, PredicateTemplate, Provenance
from predicatelab.domains.tsil.seeds import seed_templates


KIND_MAP = {k.value: k for k in CandidateKind}


def _tpl(kind: CandidateKind, name: str, descriptor: dict, rationale: str,
         generator: str, parents: list[str] | None = None) -> PredicateTemplate:
    return PredicateTemplate(
        domain="tsil",
        kind=kind,
        name=name,
        descriptor=descriptor,
        rationale=rationale,
        provenance=Provenance(generator=generator, parents=parents or [], rationale=rationale),
    )


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
    def propose(self, *, round_index: int, n: int, feedback: dict | None,
                known_template_hashes: set[str]) -> list[PredicateTemplate]:
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
                rows.append({"descriptor": desc, "score": float(score),
                             "candidate_id": r.get("candidate_id", ""), "family": desc.get("family")})
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
                out.append(_tpl(
                    CandidateKind.COMPOSITE,
                    f"Composite({a.get('family')},{b.get('family')}) from round feedback",
                    {"family": "composite", "combine": "concat", "parts": [a, b], "weights": [1.0, 1.0]},
                    "combine the two strongest DSL families from the previous round",
                    "rule_mutation", [parents[0]["candidate_id"], parents[1]["candidate_id"]],
                ))
            else:
                out.append(_tpl(
                    CandidateKind.COMPOSITE,
                    f"P-mix({a.get('family')},{b.get('family')}) from round feedback",
                    {"family": "composite", "combine": "mix", "parts": [a, b], "weights": [1.0, 1.0]},
                    "mix the projection matrices of the two strongest candidates",
                    "rule_mutation", [parents[0]["candidate_id"], parents[1]["candidate_id"]],
                ))
        return out

    def _mutate_one(self, desc: dict, parent: dict) -> list[PredicateTemplate]:
        family = desc.get("family")
        out: list[PredicateTemplate] = []
        parents = [parent.get("candidate_id", "")]
        if family in {"momentum", "volatility", "dispersion", "autocorr", "drawdown", "tail", "trend"}:
            base = int(desc.get("params", {}).get("window", 10))
            for w in sorted({max(3, base - 5), min(self.seq_len, base + 5), self.seq_len}):
                if w == base:
                    continue
                new = {"family": family, "params": {"window": w}}
                if desc.get("feature_slice"):
                    new["feature_slice"] = desc["feature_slice"]
                out.append(_tpl(
                    CandidateKind.SOFT if family in {"momentum", "trend"} else CandidateKind.STATISTICAL,
                    f"{family} window={w} (mutated)",
                    new, f"window mutation of {parent.get('family')} (base {base})",
                    "rule_mutation", parents,
                ))
            half = max(8, self.feature_dim // 2)
            for fs in ([0, half], [half, self.feature_dim], [0, max(8, self.feature_dim // 4)]):
                new = {"family": family, "params": {"window": base}, "feature_slice": fs}
                out.append(_tpl(
                    CandidateKind.SOFT if family in {"momentum", "trend"} else CandidateKind.STATISTICAL,
                    f"{family} window={base} features={fs[0]}:{fs[1]} (mutated)",
                    new, f"feature-subset mutation of {family}",
                    "rule_mutation", parents,
                ))
            # add to a native partner via P-mix (keeps upstream math untouched)
            partner = {"family": "native", "predicate": "phi_volatility", "params": {}}
            out.append(_tpl(
                CandidateKind.COMPOSITE,
                f"P-mix({family}, native volatility)",
                {"family": "composite", "combine": "mix",
                 "parts": [{"family": family, "params": {"window": base}}, partner],
                 "weights": [1.0, 1.0]},
                "mix a discovered descriptor with the native volatility predicate",
                "rule_mutation", parents,
            ))
        elif family == "native":
            pred = desc.get("predicate")
            if pred in {"phi_momentum", "phi_volatility", "phi_multiscale"}:
                key = {"phi_momentum": "lookback_windows", "phi_volatility": "window_sizes",
                       "phi_multiscale": "scales"}[pred]
                for windows in ([3, 10], [5, 10, 20], [2, 5, 10, 20]):
                    new = {"family": "native", "predicate": pred, "params": {key: windows}}
                    out.append(_tpl(
                        CandidateKind.NATIVE,
                        f"{pred} {key}={windows} (mutated)",
                        new, f"parameter mutation of native {pred}",
                        "rule_mutation", parents,
                    ))
            else:
                out.append(_tpl(
                    CandidateKind.COMPOSITE,
                    f"P-mix({pred}, momentum window=5)",
                    {"family": "composite", "combine": "mix",
                     "parts": [desc, {"family": "momentum", "params": {"window": 5}}],
                     "weights": [1.0, 1.0]},
                    f"mix native {pred} with a discovered momentum descriptor",
                    "rule_mutation", parents,
                ))
        elif family == "composite":
            # perturb composite weights / depths
            parts = desc.get("parts", [])
            if parts:
                out.append(_tpl(
                    CandidateKind.COMPOSITE,
                    "Composite reweighted (mutated)",
                    {**desc, "weights": [2.0] + [1.0] * (len(parts) - 1)},
                    "re-weight the first component of a composite",
                    "rule_mutation", parents,
                ))
        return out

    # ------------------------------------------------------------ exploration
    def _exploration(self, n: int, seen: set[str], round_index: int) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        families = ["momentum", "volatility", "dispersion", "autocorr", "drawdown", "tail", "trend"]
        windows = [3, 5, 8, 12, 15, 20]
        i = 0
        for fam in families:
            for w in windows:
                desc = {"family": fam, "params": {"window": w}}
                t = _tpl(
                    CandidateKind.SOFT if fam in {"momentum", "trend"} else CandidateKind.STATISTICAL,
                    f"{fam} window={w}",
                    desc, f"structured exploration of {fam} at window {w}",
                    "rule_exploration",
                )
                if t.content_hash in seen:
                    continue
                seen.add(t.content_hash)
                out.append(t)
                i += 1
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

    def propose(self, *, round_index: int, n: int, feedback: dict | None,
                known_template_hashes: set[str]) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        seen = set(known_template_hashes)
        for row in self._rows():
            if row.get("round") not in (None, round_index):
                continue
            kind = KIND_MAP.get(str(row.get("kind", "statistical")))
            if kind is None:
                continue
            t = _tpl(kind, str(row.get("name", "llm proposal")), dict(row.get("descriptor", {})),
                     str(row.get("rationale", "")), "llm_json",
                     [str(x) for x in row.get("parents", [])])
            if t.content_hash in seen:
                continue
            seen.add(t.content_hash)
            out.append(t)
            if len(out) >= n:
                return out
        if len(out) < n:
            out.extend(self.fallback.propose(round_index=round_index, n=n - len(out),
                                             feedback=feedback, known_template_hashes=seen))
        return out


def make_generator(config: dict[str, Any], policy: dict[str, Any]) -> CandidateGenerator:
    name = policy.get("generator", "rule")
    gen_cfg = dict(policy.get("generator_config", {}) or {})
    merged = dict(config)
    merged["generator_config"] = gen_cfg
    if name == "llm_json":
        path = gen_cfg.get("proposals_path")
        if not path:
            raise ValueError("generator=llm_json requires generator_config.proposals_path")
        return JsonProposalGenerator(merged, path)
    if name != "rule":
        raise ValueError(f"unknown generator {name!r}")
    return RuleGenerator(merged)
