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
    def __init__(self, config: dict[str, Any], feature_names: list[str], n_features: int,
                 direction: str = "higher"):
        self.feature_names = feature_names
        self.n_features = n_features
        self.direction = direction
        raw_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (0, 13), (6, 7), (20, 21)]
        self.seed_pairs = [(a, b) for a, b in raw_pairs if a < n_features and b < n_features]
        if not self.seed_pairs and n_features >= 2:
            self.seed_pairs = [(0, 1)]

    def _seed_templates(self) -> list[PredicateTemplate]:
        out = []
        for a, b in self.seed_pairs:
            for op in ("ratio", "product", "difference"):
                out.append(_tpl(
                    KIND[op], f"{op}({self.feature_names[a]}, {self.feature_names[b]})",
                    {"op": op, "a": a, "b": b},
                    f"pairwise {op} interaction between two measurements",
                ))
        for a in range(min(3, self.n_features)):
            out.append(_tpl(KIND["sigmoid"], f"sigmoid({self.feature_names[a]})",
                            {"op": "sigmoid", "a": a, "scale": 1.0},
                            "graded (soft) deviation from the fold-local median"))
            out.append(_tpl(KIND["threshold"], f"threshold({self.feature_names[a]}, q=0.5)",
                            {"op": "threshold", "a": a, "quantile": 0.5},
                            "conditional indicator above the fold-local median"))
        for a, b in ((0, 1), (0, 2), (1, 2)):
            if a < self.n_features and b < self.n_features:
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
        for key in ("validated", "insufficient", "rejected", "screened_out"):
            for r in feedback.get(key, []) or []:
                if isinstance(r.get("descriptor"), dict):
                    score = r.get("screen_effect") if r.get("screen_effect") is not None else (r.get("effect") or 0.0)
                    parents.append((float(score), r))
        # Rank by improvement in the registered direction (lower-is-better uses
        # the most negative effects as parents).
        sign = 1.0 if self.direction == "higher" else -1.0
        parents.sort(key=lambda x: -sign * x[0])
        for _score, row in parents[:4]:
            desc = row["descriptor"]
            op, a = desc.get("op"), desc.get("a")
            if not isinstance(op, str) or not isinstance(a, int):
                continue
            parents_ids = [row.get("candidate_id", "")]
            for a2 in {max(0, a - 1), min(self.n_features - 1, a + 1)}:
                if a2 == a:
                    continue
                if op in {"ratio", "product", "difference", "interaction"} and a2 == desc.get("b"):
                    # a and b must stay distinct for pairwise operators
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


class TabularLLMGenerator(CandidateGenerator):
    """DeepSeek LLM-driven predicate generator for the Tabular domain."""

    def __init__(self, config: dict[str, Any], feature_names: list[str], n_features: int,
                 direction: str = "higher", gen_cfg: dict[str, Any] | None = None):
        from pathlib import Path
        from predicatelab.core.llm import DeepSeekLLMClient

        self.config = config
        self.feature_names = feature_names
        self.n_features = n_features
        self.direction = direction
        self.client = DeepSeekLLMClient(gen_cfg or config.get("search", {}).get("generator_config", {}))
        self.fallback = TabularRuleGenerator(config, feature_names, n_features, direction=direction)
        self.artifacts_dir = Path(config.get("run", {}).get("artifacts_dir", "artifacts"))

    def propose(self, *, round_index: int, n: int, feedback: dict | None,
                known_template_hashes: set[str]) -> list[PredicateTemplate]:
        import json

        seen = set(known_template_hashes)
        out: list[PredicateTemplate] = []

        def add(t: PredicateTemplate) -> None:
            if t.content_hash in seen or len(out) >= n:
                return
            seen.add(t.content_hash)
            out.append(t)

        f_map = {i: name for i, name in enumerate(self.feature_names[:40])}
        sys_prompt = (
            "You are an AI Scientist constructing tabular feature predicates for PredicateLab.\n"
            f"Available features (0..{self.n_features - 1}): {json.dumps(f_map)}\n"
            "Allowed operators (`descriptor.op`):\n"
            "- `ratio`, `product`, `difference`: requires integer `a` and `b` (0 <= a != b < n_features), kind='statistical'\n"
            "- `interaction`: requires integer `a` and `b` (a != b), kind='relational'\n"
            "- `sigmoid`: requires integer `a` and float `scale` in [0.1, 10.0], kind='soft'\n"
            "- `threshold`: requires integer `a` and float `quantile` in [0.05, 0.95], kind='conditional'\n"
            "Return JSON: {\"round_analysis\": \"...\", \"proposals\": [{\"kind\": \"...\", \"name\": \"...\", "
            "\"rationale\": \"...\", \"parents\": [], \"descriptor\": {...}}]}"
        )
        usr_prompt = f"Round {round_index}: Propose {n + 4} candidates. Feedback: {json.dumps(feedback or {})[:3000]}"
        audit_file = self.artifacts_dir / "llm_proposals" / f"tabular_round_{round_index}.json"
        res = self.client.chat_proposals(system_prompt=sys_prompt, user_prompt=usr_prompt, audit_path=audit_file)
        if res.ok and res.proposals:
            for row in res.proposals:
                desc = row.get("descriptor")
                if not isinstance(desc, dict):
                    continue
                op = str(desc.get("op") or "")
                if op not in KIND:
                    continue
                try:
                    a = int(desc.get("a", 0))
                    if not (0 <= a < self.n_features):
                        continue
                    clean_desc: dict[str, Any] = {"op": op, "a": a}
                    if op in {"ratio", "product", "difference", "interaction"}:
                        b = int(desc.get("b", (a + 1) % self.n_features))
                        if not (0 <= b < self.n_features) or a == b:
                            continue
                        clean_desc["b"] = b
                    elif op == "sigmoid":
                        clean_desc["scale"] = max(0.1, min(10.0, float(desc.get("scale", 1.0))))
                    elif op == "threshold":
                        clean_desc["quantile"] = max(0.05, min(0.95, float(desc.get("quantile", 0.5))))
                    add(_tpl(
                        KIND[op],
                        str(row.get("name") or f"LLM {op}({a})"),
                        clean_desc,
                        str(row.get("rationale") or "LLM tabular hypothesis"),
                        generator=f"llm_deepseek:{res.model}",
                        parents=[str(p) for p in (row.get("parents") or []) if p],
                    ))
                except Exception:
                    continue

        if len(out) < n:
            for t in self.fallback.propose(round_index=round_index, n=n - len(out),
                                           feedback=feedback, known_template_hashes=seen):
                add(t)
        return out

