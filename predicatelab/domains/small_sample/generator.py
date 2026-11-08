"""Candidate generator for the small-sample scenario (deterministic)."""

from __future__ import annotations

from typing import Any

from predicatelab.core.adapter import CandidateGenerator
from predicatelab.core.schema import CandidateKind, PredicateTemplate, Provenance

KIND = {
    "ratio": CandidateKind.STATISTICAL,
    "difference": CandidateKind.STATISTICAL,
    "product": CandidateKind.STATISTICAL,
    "ratio_threshold": CandidateKind.CONDITIONAL,
    "threshold": CandidateKind.CONDITIONAL,
    "sigmoid": CandidateKind.SOFT,
    "interaction": CandidateKind.RELATIONAL,
}


def _tpl(kind: CandidateKind, name: str, descriptor: dict, rationale: str,
         generator: str = "rule", parents: list[str] | None = None) -> PredicateTemplate:
    return PredicateTemplate(
        domain="small_sample", kind=kind, name=name, descriptor=descriptor,
        rationale=rationale,
        provenance=Provenance(generator=generator, parents=parents or [], rationale=rationale),
    )


class SmallSampleRuleGenerator(CandidateGenerator):
    """Enumerates a small, closed hypothesis space and mutates good parents.

    The hypothesis space is deliberately fixed and generic: every pairwise
    operator is proposed for a small set of feature pairs, and single-feature
    threshold/sigmoid rules follow.  The generator never sees the ground truth.
    """

    def __init__(self, config: dict[str, Any], feature_names: list[str], n_features: int,
                 direction: str = "higher"):
        self.feature_names = list(feature_names)
        self.n_features = int(n_features)
        self.direction = direction
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.pairs = [(a, b) for a, b in pairs if a < self.n_features and b < self.n_features]
        self.single = list(range(min(4, self.n_features)))
        self.interactions = [(a, b) for a, b in [(1, 3), (0, 2), (2, 3)]
                             if a < self.n_features and b < self.n_features]

    # ------------------------------------------------------------------ seeds
    def _seed_templates(self) -> list[PredicateTemplate]:
        out: list[PredicateTemplate] = []
        for a, b in self.pairs:
            out.append(_tpl(
                KIND["ratio_threshold"],
                f"ratio_threshold({self.feature_names[a]}, {self.feature_names[b]})",
                {"op": "ratio_threshold", "a": a, "b": b, "quantile": 0.75},
                "conditional rule: upper-quartile threshold on a ratio of two measurements",
            ))
            out.append(_tpl(
                KIND["ratio"], f"ratio({self.feature_names[a]}, {self.feature_names[b]})",
                {"op": "ratio", "a": a, "b": b},
                "pairwise scale-free ratio of two measurements",
            ))
            out.append(_tpl(
                KIND["difference"],
                f"difference({self.feature_names[a]}, {self.feature_names[b]})",
                {"op": "difference", "a": a, "b": b},
                "pairwise additive contrast of two measurements",
            ))
        for a, b in self.interactions:
            out.append(_tpl(
                KIND["interaction"],
                f"sign_agreement({self.feature_names[a]}, {self.feature_names[b]})",
                {"op": "interaction", "a": a, "b": b},
                "relational rule: whether two standardised measurements agree in sign",
            ))
        for a in self.single:
            for q in (0.3, 0.5, 0.7):
                out.append(_tpl(
                    KIND["threshold"],
                    f"threshold({self.feature_names[a]}, q={q})",
                    {"op": "threshold", "a": a, "quantile": q},
                    "conditional indicator above a fold-local quantile",
                ))
            out.append(_tpl(
                KIND["sigmoid"], f"sigmoid({self.feature_names[a]})",
                {"op": "sigmoid", "a": a, "scale": 1.0},
                "graded (soft) deviation from the fold-local median",
            ))
        return out

    # ---------------------------------------------------------------- propose
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

        parents: list[tuple[float, dict]] = []
        for key in ("validated", "insufficient", "rejected", "screened_out"):
            for row in feedback.get(key, []) or []:
                desc = row.get("descriptor")
                if not isinstance(desc, dict):
                    continue
                score = row.get("screen_effect")
                if score is None:
                    score = row.get("effect") or 0.0
                parents.append((float(score), row))
        sign = 1.0 if self.direction == "higher" else -1.0
        parents.sort(key=lambda x: -sign * x[0])

        for _score, row in parents[:4]:
            desc = row["descriptor"]
            op, a, b = desc.get("op"), desc.get("a"), desc.get("b")
            if not isinstance(op, str) or not isinstance(a, int):
                continue
            parent_ids = [str(row.get("candidate_id", ""))]
            # Feature-index neighbourhood.
            for a2 in {max(0, a - 1), min(self.n_features - 1, a + 1)}:
                if a2 == a:
                    continue
                if op in {"ratio", "difference", "product", "ratio_threshold", "interaction"}:
                    if a2 == b:
                        continue
                    add(_tpl(KIND[op], f"{op}(feature {a2}) mutated", {**desc, "a": a2},
                             f"neighbour-feature mutation of {op} on feature {a}", parents=parent_ids))
            # Partner swap for pairwise operators.
            if isinstance(b, int) and op in {"ratio", "difference", "product",
                                             "ratio_threshold", "interaction"}:
                for b2 in {max(0, b - 1), min(self.n_features - 1, b + 1)}:
                    if b2 in {a, b}:
                        continue
                    add(_tpl(KIND[op], f"{op}(partner {b2}) mutated", {**desc, "b": b2},
                             f"neighbour-partner mutation of {op}", parents=parent_ids))
            # Operator swap among the ratio family.
            if op in {"ratio", "difference", "product", "ratio_threshold"} and isinstance(b, int):
                for op2 in ("ratio", "difference", "product", "ratio_threshold"):
                    if op2 == op:
                        continue
                    new = {**desc, "op": op2}
                    if op2 == "ratio_threshold":
                        new.setdefault("quantile", 0.75)
                    else:
                        new.pop("quantile", None)
                    add(_tpl(KIND[op2], f"{op2}(features {a},{b}) mutated", new,
                             f"operator mutation from {op}", parents=parent_ids))
            # Parameter mutations.
            if op == "threshold":
                for q in (0.25, 0.5, 0.75):
                    if q == desc.get("quantile"):
                        continue
                    add(_tpl(KIND[op], f"threshold(feature {a}, q={q}) mutated",
                             {**desc, "quantile": q}, "quantile mutation", parents=parent_ids))
            if op == "ratio_threshold":
                for q in (0.6, 0.7, 0.8):
                    if q == desc.get("quantile"):
                        continue
                    add(_tpl(KIND[op], f"ratio_threshold(feature {a}, q={q}) mutated",
                             {**desc, "quantile": q}, "ratio-quantile mutation",
                             parents=parent_ids))
            if op == "sigmoid":
                for s in (0.5, 2.0):
                    if s == desc.get("scale"):
                        continue
                    add(_tpl(KIND[op], f"sigmoid(feature {a}, scale={s}) mutated",
                             {**desc, "scale": s}, "scale mutation", parents=parent_ids))
            if len(out) >= n:
                return out

        for t in self._seed_templates():
            add(t)
            if len(out) >= n:
                break
        return out
