"""Candidate generator for the tabular domain (deterministic, feedback-driven)."""

from __future__ import annotations

from typing import Any

from predicatelab.core.adapter import CandidateGenerator
from predicatelab.core.schema import CandidateKind, PredicateTemplate, Provenance


def _tpl(kind: CandidateKind, name: str, descriptor: dict, rationale: str,
         generator: str = "rule", parents: list[str] | None = None) -> PredicateTemplate:
    return PredicateTemplate(
        domain="tabular", kind=kind, name=name, descriptor=descriptor,
        rationale=rationale,
        provenance=Provenance(generator=generator, parents=parents or [], rationale=rationale),
    )


KIND = {
    "ratio": CandidateKind.STATISTICAL,
    "product": CandidateKind.STATISTICAL,
    "difference": CandidateKind.STATISTICAL,
    "sigmoid": CandidateKind.SOFT,
    "threshold": CandidateKind.CONDITIONAL,
    "interaction": CandidateKind.RELATIONAL,
}


class TabularRuleGenerator(CandidateGenerator):
    def __init__(self, config: dict[str, Any], feature_names: list[str], n_features: int):
        self.feature_names = feature_names
        self.n_features = n_features
        self.seed_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (0, 13), (6, 7), (20, 21)]

    def _seed_templates(self) -> list[PredicateTemplate]:
        out = []
        for a, b in self.seed_pairs:
            for op in ("ratio", "product", "difference"):
                out.append(_tpl(
                    KIND[op], f"{op}({self.feature_names[a]}, {self.feature_names[b]})",
                    {"op": op, "a": a, "b": b},
                    f"pairwise {op} interaction between two measurements",
                ))
        for a in (0, 1, 2):
            out.append(_tpl(KIND["sigmoid"], f"sigmoid({self.feature_names[a]})",
                            {"op": "sigmoid", "a": a, "scale": 1.0},
                            "graded (soft) deviation from the fold-local median"))
            out.append(_tpl(KIND["threshold"], f"threshold({self.feature_names[a]}, q=0.5)",
                            {"op": "threshold", "a": a, "quantile": 0.5},
                            "conditional indicator above the fold-local median"))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            out.append(_tpl(KIND["interaction"], f"sign_agreement({self.feature_names[a]}, {self.feature_names[b]})",
                            {"op": "interaction", "a": a, "b": b},
                            "relational rule: whether two standardised measurements agree in sign"))
        return out

    def propose(self, *, round_index: int, n: int, feedback: dict | None,
                known_template_hashes: set[str]) -> list[PredicateTemplate]:
        seen = set(known_template_hashes)
        out: list[PredicateTemplate] = []

        def add(t: PredicateTemplate) -> None:
            if t.content_hash in seen or len(out) >= n:
                return
            seen.add(t.content_hash)
            out.append(t)

        if round_index == 0 or not feedback:
            for t in self._seed_templates():
                add(t)
            return out

        parents = []
        for key in ("validated", "insufficient", "screened_out"):
            for r in feedback.get(key, []) or []:
                if isinstance(r.get("descriptor"), dict):
                    score = r.get("effect") if r.get("effect") is not None else r.get("screen_effect") or 0.0
                    parents.append((float(score), r))
        parents.sort(key=lambda x: -x[0])
        for _score, row in parents[:4]:
            desc = row["descriptor"]
            op, a = desc.get("op"), desc.get("a")
            if not isinstance(op, str) or not isinstance(a, int):
                continue
            parents_ids = [row.get("candidate_id", "")]
            for a2 in {max(0, a - 1), min(self.n_features - 1, a + 1)}:
                if a2 == a:
                    continue
                new = {**desc, "a": a2}
                add(_tpl(KIND[op], f"{op}(feature {a2}) mutated", new,
                         f"neighbour-feature mutation of {op} on feature {a}",
                         parents=parents_ids))
            if op in {"ratio", "product", "difference"} and isinstance(desc.get("b"), int):
                for op2 in ("ratio", "product", "difference"):
                    if op2 != op:
                        new = {**desc, "op": op2}
                        add(_tpl(KIND[op2], f"{op2}(features {a},{desc['b']}) mutated", new,
                                 f"operator mutation from {op}", parents=parents_ids))
            if op == "threshold":
                for q in (0.25, 0.5, 0.75):
                    if q == desc.get("quantile"):
                        continue
                    add(_tpl(KIND[op], f"threshold(feature {a}, q={q}) mutated",
                             {**desc, "quantile": q}, "quantile mutation", parents=parents_ids))
            if op == "sigmoid":
                for s in (0.5, 2.0):
                    add(_tpl(KIND[op], f"sigmoid(feature {a}, scale={s}) mutated",
                             {**desc, "scale": s}, "scale mutation", parents=parents_ids))
            if len(out) >= n:
                return out
        # fill with fresh seeds if feedback was sparse
        for t in self._seed_templates():
            add(t)
            if len(out) >= n:
                break
        return out
