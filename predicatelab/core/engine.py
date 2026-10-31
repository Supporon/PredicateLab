"""Generic discovery → validation → evidence → feedback engine.

The engine owns the scientific protocol; adapters own execution details.
A round runs:

1. **propose**   adapter generator emits structured templates (data, not code);
2. **bind**      templates are deduplicated by content and bound to the snapshot;
3. **probe**     numeric smoke test on training-fold data + behavioural dedup;
4. **screen**    one cheap validation fold, promotion by paired effect;
5. **validate**  rolling folds with full training budget;
6. **evidence**  paired effect, bootstrap CI, Newey-West p, BH q, stability;
7. **feedback**  persist successes, failures and inconclusive candidates for the
   next round.

The holdout fold is not in the search fold list at all.  Only
:func:`finalize_run` can access it, and only after a frozen manifest is stored.
"""

from __future__ import annotations

import json
import math
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from predicatelab.core.adapter import DomainAdapter, TrialRequest
from predicatelab.core.hashing import sha256_obj, short_id
from predicatelab.core.protocol import HoldoutGuard, ProtocolViolation, validate_protocol
from predicatelab.core.schema import (
    BoundCandidate,
    EvidenceRecord,
    EvidenceStatus,
    FinalEvaluation,
    FittedInstance,
    PredicateTemplate,
    ResearchTask,
    RoundRecord,
    SearchPolicy,
    TrialRecord,
    utcnow,
)
from predicatelab.core.stats import (
    benjamini_hochberg,
    paired_effect,
    pool_fold_series,
    stability,
)
from predicatelab.core.store import StateStore


def _now_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


@dataclass
class RunContext:
    run_id: str
    config: dict[str, Any]
    adapter: DomainAdapter
    store: StateStore
    task: ResearchTask
    protocol: Any
    policy: SearchPolicy
    snapshot: Any
    artifacts: Path
    guard: HoldoutGuard
    protocol_hash: str
    round_logs: list[dict[str, Any]] = field(default_factory=list)


class PredicateLabEngine:
    def __init__(self, config: dict[str, Any], *, db_path: str | None = None, run_id: str | None = None):
        self.config = config
        self.run_cfg = config.get("run", {})
        self.artifacts_root = Path(self.run_cfg.get("artifacts_dir", "artifacts"))
        self.db_path = db_path or self.run_cfg.get("db", str(self.artifacts_root / "predicatelab.sqlite"))
        self.store = StateStore(self.db_path)
        self.run_id = run_id or _now_id("run")
        self.artifacts = self.artifacts_root / self.run_id
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._ctx: RunContext | None = None

    # ------------------------------------------------------------------ setup
    def _build_context(self) -> RunContext:
        from predicatelab.domains import create_adapter

        if self._ctx is not None:
            return self._ctx
        adapter = create_adapter(self.config)
        protocol = adapter.build_protocol(self.config)
        validate_protocol(protocol.folds, protocol.holdout_fold)
        policy = SearchPolicy.model_validate(self.config.get("search", {}))
        if not policy.screen_folds:
            policy.screen_folds = [protocol.folds[-1].fold_id]
        if not policy.validation_folds:
            policy.validation_folds = [f.fold_id for f in protocol.folds]

        contract = adapter.data_contract(self.config)
        snapshot = adapter.snapshot(self.config)
        suitability = adapter.suitability(snapshot, self.config)
        evaluation = adapter.evaluation
        task = ResearchTask(
            title=self.config.get("task", {}).get("title", self.run_cfg.get("name", "untitled")),
            question=self.config.get("task", {}).get("question", ""),
            domain=adapter.name,
            hypothesis=self.config.get("task", {}).get("hypothesis", ""),
            evaluation=protocol.evaluation or evaluation,
            data_contract=contract,
            metadata={"suitability": suitability},
        ).ensure_id()
        self.store.save_task(task)
        self.store.save_snapshot(snapshot)

        protocol_hash = adapter.protocol_hash(protocol, self.config)
        protocol.ensure_id()
        guard = HoldoutGuard.from_folds(protocol.folds, protocol.holdout_fold)
        self._ctx = RunContext(
            run_id=self.run_id,
            config=self.config,
            adapter=adapter,
            store=self.store,
            task=task,
            protocol=protocol,
            policy=policy,
            snapshot=snapshot,
            artifacts=self.artifacts,
            guard=guard,
            protocol_hash=protocol_hash,
        )
        self._persist_context()
        return self._ctx

    def _persist_context(self) -> None:
        ctx = self._ctx
        assert ctx is not None
        meta = {
            "run_id": self.run_id,
            "task_id": ctx.task.task_id,
            "domain": ctx.adapter.name,
            "protocol": ctx.protocol.model_dump(mode="json"),
            "protocol_hash": ctx.protocol_hash,
            "snapshot": ctx.snapshot.model_dump(mode="json"),
            "evaluation": ctx.adapter.evaluation.model_dump(mode="json"),
            "data_contract": ctx.task.data_contract.model_dump(mode="json"),
            "policy": ctx.policy.model_dump(mode="json"),
        }
        (self.artifacts / "context.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        if not self.store.get_run(self.run_id):
            self.store.create_run(self.run_id, ctx.task.task_id, self.config, str(self.artifacts))

    # --------------------------------------------------------------- training
    def _training_config(self, phase: str) -> dict[str, Any]:
        tr = dict(self.config.get("training", {}))
        profile_key = "full" if phase == "final" else phase
        profile = dict(tr.pop(profile_key, {}) or {})
        merged = {**tr, **profile}
        merged.pop("screen", None)
        merged.pop("full", None)
        return merged

    def _max_workers(self) -> int:
        return int(self.config.get("execution", {}).get("workers", 1))

    # ----------------------------------------------------------------- trials
    def _trial_config(self, candidate_id: str, variant: str, descriptor: dict, fold, phase: str, seed: int):
        training = self._training_config(phase)
        req = TrialRequest(
            candidate_id=candidate_id,
            variant=variant,
            descriptor=descriptor,
            fold=fold,
            seed=seed,
            training_config=training,
            workdir="",
            protocol_hash=self._ctx.protocol_hash,
            config_hash="",
            save_predictions=self._ctx.policy.save_predictions,
        )
        cfg_hash = self._ctx.adapter.trial_config_hash(req)
        workdir = self.artifacts / "trials" / fold.fold_id / (candidate_id.replace(":", "_"))
        workdir.mkdir(parents=True, exist_ok=True)
        req.config_hash = cfg_hash
        req.workdir = str(workdir)
        return req

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        """Run one trial with cache reuse (resume) and failure capture."""
        ctx = self._ctx
        assert ctx is not None
        cached = self.store.get_cached_trial(
            request.candidate_id, request.fold.fold_id, request.protocol_hash, request.config_hash
        )
        if cached:
            return cached
        started = datetime.now(timezone.utc)
        try:
            result = ctx.adapter.run_trial(request)
        except Exception as exc:  # adapters should catch; this is a last resort
            result = {
                "status": "failed",
                "metrics": {},
                "metric_series": {},
                "fitted_params": {},
                "predictions_path": "",
                "duration_s": 0.0,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "worker": {},
            }
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        trial = TrialRecord(
            candidate_id=request.candidate_id,
            fold_id=request.fold.fold_id,
            protocol_hash=request.protocol_hash,
            config_hash=request.config_hash,
            status=result.get("status", "failed"),
            metrics=result.get("metrics", {}),
            metric_series=result.get("metric_series", {}),
            fitted_params=result.get("fitted_params", {}),
            predictions_path=result.get("predictions_path", ""),
            duration_s=float(result.get("duration_s", duration)),
            error=result.get("error", ""),
            worker=result.get("worker", {}),
        )
        self.store.record_trial(trial, run_id=self.run_id)
        if trial.status == "success" and trial.fitted_params:
            self.store.add_fitted_instance(FittedInstance(
                candidate_id=request.candidate_id,
                fold_id=request.fold.fold_id,
                fitted_params=trial.fitted_params,
                fit_data_hash=result.get("fit_data_hash", sha256_obj(
                    {"fold": request.fold.model_dump(mode="json"), "candidate": request.candidate_id}
                )),
            ))
        if trial.status != "success":
            print(f"[trial-failed] {request.candidate_id} {request.fold.fold_id}: {trial.error.splitlines()[0] if trial.error else 'unknown'}")
        return self.store.get_trial_by_key(
            request.candidate_id, request.fold.fold_id, request.protocol_hash, request.config_hash
        ) or trial.model_dump(mode="json")

    def _ensure_variant(self, variant_id: str, variant: str, fold, phase: str, seed: int) -> dict[str, Any]:
        req = self._trial_config(variant_id, variant, {}, fold, phase, seed)
        return self.run_trial(req)

    # ------------------------------------------------------------------ probe
    def _probe(self, descriptor: dict, fold, workdir: str) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        try:
            return ctx.adapter.probe(descriptor, fold, ctx.snapshot, self.config, workdir)
        except Exception as exc:
            return {"ok": False, "reason": f"probe error: {type(exc).__name__}: {exc}", "severity": "invalid"}

    @staticmethod
    def _signature_corr(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return float("nan")
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return float("nan")
        x, y = x[mask], y[mask]
        if x.std() == 0 or y.std() == 0:
            return 1.0 if np.allclose(x, y) else float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # -------------------------------------------------------------- evidence
    def _aggregate(self, cand_series, base_series, fold_metrics, seed: int) -> dict[str, Any]:
        policy = self._ctx.policy
        try:
            pooled_c = pool_fold_series(cand_series)
            pooled_b = pool_fold_series(base_series)
        except ValueError as exc:
            return {"error": str(exc)}
        eff = paired_effect(
            pooled_c, pooled_b,
            nw_lag=policy.nw_lag,
            n_boot=policy.bootstrap_samples,
            block=policy.bootstrap_block,
            seed=seed,
        )
        fold_effects = {k: v["effect"] for k, v in fold_metrics.items()}
        return {
            "effect": eff["effect"],
            "ci_low": eff["ci_low"],
            "ci_high": eff["ci_high"],
            "p_value": eff["p_value"],
            "se": eff["se"],
            "cohen_dz": eff["cohen_dz"],
            "n_dates": eff["n_dates"],
            "n_folds": len(fold_metrics),
            "stability": stability(fold_effects, self._ctx.protocol.direction),
            "fold_effects": fold_effects,
            "folds": fold_metrics,
            "mean_candidate": float(np.mean(list(pooled_c.values()))),
            "mean_baseline": float(np.mean(list(pooled_b.values()))),
        }

    # ------------------------------------------------------------------ round
    def run(self, rounds: int | None = None) -> str:
        ctx = self._build_context()
        rounds = int(rounds if rounds is not None else self.run_cfg.get("rounds", 1))
        self._persist_context()
        print(f"[run] {self.run_id} domain={ctx.adapter.name} snapshot={ctx.snapshot.snapshot_id}")

        completed = {r["round_index"] for r in self.store.list_rounds(self.run_id)}
        feedback = self.store.latest_feedback(self.run_id)
        for idx in range(rounds):
            if idx in completed:
                print(f"[round {idx}] already complete, reusing")
                feedback = self.store.latest_feedback(self.run_id)
                continue
            feedback = self._run_round(idx, feedback)
        self._write_report()
        self.store.finish_run(self.run_id, "search_complete")
        return self.run_id

    def _run_round(self, round_index: int, feedback: dict[str, Any] | None) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        policy = ctx.policy
        known = self.store.list_candidates(ctx.snapshot.snapshot_hash)
        known_hashes = {c["template_hash"] for c in known}
        generator = ctx.adapter.make_generator(self.config, policy.model_dump())
        templates = generator.propose(
            round_index=round_index,
            n=policy.max_candidates_per_round,
            feedback=feedback,
            known_template_hashes=known_hashes,
        )
        print(f"[round {round_index}] proposed {len(templates)} templates")

        round_rec = RoundRecord(
            task_id=ctx.task.task_id,
            run_id=self.run_id,
            round_index=round_index,
            generator=policy.generator,
            parent_round_id=(feedback or {}).get("round_id"),
            n_proposed=len(templates),
        ).ensure_id()
        self.store.save_round(round_rec)

        candidates: list[tuple[dict, dict]] = []  # (candidate_row, probe_result)
        duplicates = 0
        invalid = 0
        probe_ok_signatures = dict(self.store.list_probe_signatures(ctx.snapshot.snapshot_hash))

        for tpl in templates:
            tpl.provenance.round_id = round_rec.round_id
            try:
                ctx.adapter.validate_descriptor(tpl.kind.value, tpl.descriptor)
            except Exception as exc:
                invalid += 1
                self._record_early_evidence(
                    round_rec, tpl, EvidenceStatus.INVALID_STRUCTURE,
                    reason=f"static validation: {type(exc).__name__}: {exc}",
                )
                continue
            tpl_id, created = self.store.add_template(tpl)
            bound = BoundCandidate(
                template_id=tpl_id,
                template_hash=tpl.content_hash,
                domain=tpl.domain,
                kind=tpl.kind,
                name=tpl.name,
                descriptor=tpl.descriptor,
                dataset_id=ctx.snapshot.dataset_id,
                snapshot_hash=ctx.snapshot.snapshot_hash,
            ).ensure_id()
            cand_id, cand_created = self.store.add_candidate(bound)
            if not cand_created:
                duplicates += 1
                prior = [e for e in self.store.list_evidence() if e["candidate_id"] == cand_id]
                self._record_early_evidence(
                    round_rec, tpl, EvidenceStatus.DUPLICATE,
                    candidate_id=cand_id,
                    reason="semantically identical to an existing bound candidate"
                    + (f"; prior status={prior[-1]['status']}" if prior else ""),
                )
                continue
            cand_row = self.store.get_candidate(cand_id)
            screen_fold = self._fold(policy.screen_folds[0])
            probe = self._probe(tpl.descriptor, screen_fold, str(self.artifacts / "probe" / cand_id))
            if not probe.get("ok", False):
                sev = probe.get("severity", "invalid")
                status = EvidenceStatus.DUPLICATE if sev == "duplicate" else EvidenceStatus.INVALID_STRUCTURE
                self._record_early_evidence(round_rec, tpl, status, candidate_id=cand_id,
                                            reason=probe.get("reason", "probe failed"))
                continue
            sig = probe.get("signature") or []
            dup_of = None
            for other_id, other_sig in probe_ok_signatures.items():
                if self._signature_corr(sig, other_sig) >= float(self.config.get("search", {}).get("behavioural_dedup_corr", 0.995)):
                    dup_of = other_id
                    break
            if dup_of:
                duplicates += 1
                self._record_early_evidence(
                    round_rec, tpl, EvidenceStatus.DUPLICATE, candidate_id=cand_id,
                    reason=f"behavioural duplicate of {dup_of} (probe signature corr ≥ threshold)",
                )
                continue
            if sig:
                self.store.set_probe_signature(cand_id, sig)
                probe_ok_signatures[cand_id] = sig
            candidates.append((cand_row, probe))

        # ------------------------------------------------ screening
        seed = policy.seed
        screen_fold = self._fold(policy.screen_folds[0])
        baseline_screen = self._ensure_variant(ctx.protocol.baseline_variant, "baseline", screen_fold, "screen", seed)
        if baseline_screen["status"] != "success":
            raise RuntimeError(f"baseline failed on screen fold {screen_fold.fold_id}: {baseline_screen.get('error')}")
        base_screen_series = baseline_screen.get("metric_series", {}).get(ctx.protocol.primary_metric, {})

        screen_results = []
        for cand_row, _probe_res in candidates:
            req = self._trial_config(cand_row["candidate_id"], "candidate", cand_row["descriptor"], screen_fold, "screen", seed)
            res = self.run_trial(req)
            if res["status"] != "success":
                ev = self._make_evidence(cand_row, round_rec, EvidenceStatus.EXECUTION_FAILED,
                                         reason=(res.get("error") or "screen trial failed")[:1000])
                self.store.record_evidence(ev, run_id=self.run_id)
                continue
            cs = res.get("metric_series", {}).get(ctx.protocol.primary_metric, {})
            eff = paired_effect(cs, base_screen_series, nw_lag=policy.nw_lag,
                                n_boot=min(200, policy.bootstrap_samples),
                                block=policy.bootstrap_block, seed=seed)
            screen_results.append({
                "candidate": cand_row,
                "trial": res,
                "screen_effect": eff["effect"],
                "screen_p": eff["p_value"],
                "screen_metric": res.get("metrics", {}).get(ctx.protocol.primary_metric),
            })
        direction = 1.0 if ctx.protocol.direction == "higher" else -1.0

        def _screen_score(row):
            v = row["screen_effect"]
            return direction * v if math.isfinite(v) else -math.inf

        screen_results.sort(key=lambda x: -_screen_score(x))
        promoted = [r for r in screen_results
                    if math.isfinite(r["screen_effect"])
                    and direction * r["screen_effect"] > policy.min_screen_effect][: policy.full_validation_top_k]
        promoted_ids = {r["candidate"]["candidate_id"] for r in promoted}

        # screened-out candidates still get evidence
        for r in screen_results:
            if r["candidate"]["candidate_id"] in promoted_ids:
                continue
            ev = self._make_evidence(
                r["candidate"], round_rec, EvidenceStatus.SCREENED_OUT,
                effect=r["screen_effect"], p_value=r["screen_p"],
                metrics={"screen": {"fold": screen_fold.fold_id,
                                     "effect": r["screen_effect"],
                                     "candidate_metric": r["screen_metric"],
                                     "baseline_metric": baseline_screen.get("metrics", {}).get(ctx.protocol.primary_metric)}},
                reason="not promoted: screen effect ≤ min_screen_effect or outside top-k",
            )
            self.store.record_evidence(ev, run_id=self.run_id)

        # ------------------------------------------------ full validation
        validation_folds = [self._fold(fid) for fid in policy.validation_folds]
        evidence_rows = []
        for r in promoted:
            cand = r["candidate"]
            # the candidate descriptor must be passed to the trial; _paired_folds rebuilds requests
            cand_series, base_series, fold_metrics, failed = self._paired_folds_descriptor(
                cand, validation_folds, seed
            )
            if failed and not fold_metrics:
                ev = self._make_evidence(cand, round_rec, EvidenceStatus.EXECUTION_FAILED,
                                         reason="; ".join(failed)[:1500])
                self.store.record_evidence(ev, run_id=self.run_id)
                evidence_rows.append(ev)
                continue
            agg = self._aggregate(cand_series, base_series, fold_metrics, seed)
            if "error" in agg:
                ev = self._make_evidence(cand, round_rec, EvidenceStatus.EXECUTION_FAILED, reason=agg["error"])
                self.store.record_evidence(ev, run_id=self.run_id)
                evidence_rows.append(ev)
                continue
            status = self._classify(agg, policy)
            ev = self._make_evidence(
                cand, round_rec, status,
                effect=agg["effect"], ci_low=agg["ci_low"], ci_high=agg["ci_high"],
                p_value=agg["p_value"], n_folds=agg["n_folds"], stability=agg["stability"],
                metrics={k: v for k, v in agg.items() if k not in {"effect", "ci_low", "ci_high", "p_value"}},
                reason=self._reason_for(status, agg, policy),
            )
            ev.metrics["n_validation_dates"] = sum(len(s) for s in cand_series.values())
            evidence_rows.append(ev)

        # Benjamini-Hochberg correction across the promoted candidates.  A
        # candidate that looked validated by its raw p-value but does not
        # survive FDR control is downgraded to insufficient evidence.
        pvals = [e.p_value if e.p_value is not None else float("nan") for e in evidence_rows]
        qvals = benjamini_hochberg(pvals)
        direction = 1 if self._ctx.protocol.direction == "higher" else -1
        for ev, q in zip(evidence_rows, qvals):
            ev.q_value = q if q is not None and math.isfinite(q) else None
            if ev.q_value is None:
                continue
            effect = direction * (ev.effect or 0.0)
            if ev.status == EvidenceStatus.VALIDATED and ev.q_value > policy.fdr_alpha:
                ev.status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                ev.reason = (ev.reason + f"; fails BH FDR (q={ev.q_value:.3g} > {policy.fdr_alpha})").strip("; ")
            elif ev.status == EvidenceStatus.INSUFFICIENT_EVIDENCE and ev.q_value <= policy.fdr_alpha and effect < 0:
                ev.status = EvidenceStatus.REJECTED
                ev.reason = (ev.reason + "; BH-significant negative effect").strip("; ")
            self.store.record_evidence(ev, run_id=self.run_id)

        summary = self._round_summary(round_rec, screen_results, evidence_rows)
        if ctx.protocol.control_variant:
            summary["control"] = self._control_summary(screen_fold, validation_folds, seed)
        self.store.update_round(
            round_rec.round_id,
            n_proposed=len(templates),
            n_bound=len(candidates),
            n_validated=sum(1 for e in evidence_rows if e.status == EvidenceStatus.VALIDATED),
            n_rejected=sum(1 for e in evidence_rows if e.status == EvidenceStatus.REJECTED),
            n_insufficient=sum(1 for e in evidence_rows if e.status == EvidenceStatus.INSUFFICIENT_EVIDENCE)
            + sum(1 for e in screen_results if e["candidate"]["candidate_id"] not in promoted_ids),
            summary=summary,
        )
        feedback_payload = self._feedback_payload(round_rec, screen_results, evidence_rows)
        self.store.add_feedback(round_rec.round_id, feedback_payload, run_id=self.run_id)
        print(f"[round {round_index}] screen={len(screen_results)} promoted={len(promoted)} "
              f"validated={summary['n_validated']} rejected={summary['n_rejected']} "
              f"insufficient={summary['n_insufficient']} duplicates={duplicates} invalid={invalid}")
        return feedback_payload

    def _control_summary(self, screen_fold, validation_folds, seed: int) -> dict[str, Any]:
        """Φ_ones control metrics under the exact same training conditions."""
        ctx = self._ctx
        out: dict[str, Any] = {}
        for phase, folds in (("screen", [screen_fold]), ("full", validation_folds)):
            rows = {}
            for fold in folds:
                res = self._ensure_variant(ctx.protocol.control_variant, "control", fold, phase, seed)
                rows[fold.fold_id] = {
                    "status": res.get("status"),
                    "metrics": {k: res.get("metrics", {}).get(k)
                                for k in (ctx.protocol.primary_metric, "ic", "rank_ic", "icir", "mse", "mae")},
                }
            out[phase] = rows
        return out

    # ---------------------------------------------------------- round helpers
    def _fold(self, fold_id: str):
        ctx = self._ctx
        for f in ctx.protocol.folds:
            if f.fold_id == fold_id:
                ctx.guard.assert_fold_accessible(f.fold_id, "search")
                return f
        raise ProtocolViolation(f"unknown search fold {fold_id}")

    def _paired_folds_descriptor(self, cand, folds, seed):
        cand_series, base_series, fold_metrics, failed = {}, {}, {}, []
        metric = self._ctx.protocol.primary_metric
        for fold in folds:
            self._ctx.guard.assert_fold_accessible(fold.fold_id, "search")
            cand_req = self._trial_config(cand["candidate_id"], "candidate", cand["descriptor"], fold, "full", seed)
            cand_res = self.run_trial(cand_req)
            if cand_res["status"] != "success":
                failed.append(f"{fold.fold_id}: {(cand_res.get('error') or '')[:300]}")
                continue
            base_res = self._ensure_variant(self._ctx.protocol.baseline_variant, "baseline", fold, "full", seed)
            if base_res["status"] != "success":
                failed.append(f"baseline {fold.fold_id}: {(base_res.get('error') or '')[:300]}")
                continue
            cs = cand_res.get("metric_series", {}).get(metric, {})
            bs = base_res.get("metric_series", {}).get(metric, {})
            if not cs or not bs:
                failed.append(f"{fold.fold_id}: empty metric series")
                continue
            cand_series[fold.fold_id] = cs
            base_series[fold.fold_id] = bs
            eff = paired_effect(cs, bs, nw_lag=self._ctx.policy.nw_lag, n_boot=200,
                                block=self._ctx.policy.bootstrap_block, seed=seed)
            fold_metrics[fold.fold_id] = {
                "candidate": {k: cand_res["metrics"].get(k) for k in
                              (metric, "ic", "rank_ic", "icir", "mse", "mae", "n_dates")},
                "baseline": {k: base_res["metrics"].get(k) for k in
                             (metric, "ic", "rank_ic", "icir", "mse", "mae", "n_dates")},
                "effect": eff["effect"],
                "p_value": eff["p_value"],
            }
        return cand_series, base_series, fold_metrics, failed

    def _classify(self, agg: dict, policy: SearchPolicy) -> EvidenceStatus:
        if agg["n_folds"] < int(self.config.get("search", {}).get("min_folds_for_validation", 1)):
            return EvidenceStatus.INSUFFICIENT_EVIDENCE
        direction = 1 if self._ctx.protocol.direction == "higher" else -1
        effect = direction * agg["effect"]
        if effect >= policy.min_effect and agg["stability"] >= policy.min_stability and agg["p_value"] <= policy.fdr_alpha:
            return EvidenceStatus.VALIDATED
        if effect <= -policy.min_effect and agg["p_value"] <= policy.fdr_alpha:
            return EvidenceStatus.REJECTED
        return EvidenceStatus.INSUFFICIENT_EVIDENCE

    @staticmethod
    def _reason_for(status: EvidenceStatus, agg: dict, policy: SearchPolicy) -> str:
        if status == EvidenceStatus.VALIDATED:
            return (f"effect {agg['effect']:+.4f} (CI {agg['ci_low']:+.4f},{agg['ci_high']:+.4f}), "
                    f"p={agg['p_value']:.4g}, stability={agg['stability']:.2f} ≥ {policy.min_stability}, "
                    f"effect ≥ {policy.min_effect}")
        if status == EvidenceStatus.REJECTED:
            return f"significant negative effect {agg['effect']:+.4f}, p={agg['p_value']:.4g}"
        return (f"inconclusive: effect {agg['effect']:+.4f}, p={agg['p_value']:.4g}, "
                f"stability={agg['stability']:.2f}, thresholds effect≥{policy.min_effect} & stability≥{policy.min_stability}")

    def _make_evidence(self, cand: dict, round_rec: RoundRecord, status: EvidenceStatus, **kw) -> EvidenceRecord:
        ctx = self._ctx
        return EvidenceRecord(
            candidate_id=cand["candidate_id"],
            round_id=round_rec.round_id,
            protocol_hash=ctx.protocol_hash,
            status=status,
            **{k: v for k, v in kw.items() if k in EvidenceRecord.model_fields},
        ).ensure_id()

    def _record_early_evidence(self, round_rec, tpl: PredicateTemplate, status: EvidenceStatus,
                               candidate_id: str = "", reason: str = "") -> None:
        cid = candidate_id or short_id("unbound", sha256_obj([tpl.domain, tpl.kind.value, tpl.descriptor]))
        ev = EvidenceRecord(
            candidate_id=cid,
            round_id=round_rec.round_id,
            protocol_hash=self._ctx.protocol_hash,
            status=status,
            reason=f"{tpl.name}: {reason}",
            metrics={"template": {"name": tpl.name, "kind": tpl.kind.value, "descriptor": tpl.descriptor}},
        ).ensure_id()
        self.store.record_evidence(ev, run_id=self.run_id)

    def _round_summary(self, round_rec, screen_results, evidence_rows) -> dict:
        def row(r):
            c = r["candidate"]
            return {"candidate_id": c["candidate_id"], "name": c["name"], "kind": c["kind"],
                    "descriptor": c["descriptor"], "screen_effect": r["screen_effect"],
                    "screen_metric": r["screen_metric"]}
        return {
            "round_id": round_rec.round_id,
            "n_screened": len(screen_results),
            "n_validated": sum(1 for e in evidence_rows if e.status == EvidenceStatus.VALIDATED),
            "n_rejected": sum(1 for e in evidence_rows if e.status == EvidenceStatus.REJECTED),
            "n_insufficient": sum(1 for e in evidence_rows if e.status == EvidenceStatus.INSUFFICIENT_EVIDENCE),
            "screen_table": [row(r) for r in screen_results],
            "validation": [
                {"candidate_id": e.candidate_id, "status": e.status.value, "effect": e.effect,
                 "ci": [e.ci_low, e.ci_high], "p": e.p_value, "q": e.q_value,
                 "stability": e.stability, "n_folds": e.n_folds, "reason": e.reason}
                for e in evidence_rows
            ],
        }

    def _feedback_payload(self, round_rec, screen_results, evidence_rows) -> dict:
        cand_by_id = {r["candidate"]["candidate_id"]: r["candidate"] for r in screen_results}
        validated, rejected, insufficient = [], [], []
        for e in evidence_rows:
            base = {
                "candidate_id": e.candidate_id,
                "effect": e.effect, "q_value": e.q_value, "p_value": e.p_value,
                "status": e.status.value,
                "descriptor": cand_by_id.get(e.candidate_id, {}).get("descriptor"),
                "name": cand_by_id.get(e.candidate_id, {}).get("name"),
                "family": (cand_by_id.get(e.candidate_id, {}).get("descriptor") or {}).get("family"),
            }
            (validated if e.status == EvidenceStatus.VALIDATED else
             rejected if e.status == EvidenceStatus.REJECTED else
             insufficient).append(base)
        screen_only = [
            {"candidate_id": r["candidate"]["candidate_id"],
             "descriptor": r["candidate"]["descriptor"],
             "family": (r["candidate"]["descriptor"] or {}).get("family"),
             "screen_effect": r["screen_effect"], "status": "screened_out"}
            for r in screen_results
            if r["candidate"]["candidate_id"] not in {e.candidate_id for e in evidence_rows}
        ]
        return {
            "round_id": round_rec.round_id,
            "round_index": round_rec.round_index,
            "validated": validated,
            "rejected": rejected,
            "insufficient": insufficient,
            "screened_out": screen_only,
            "known_effects": {e.candidate_id: e.effect for e in evidence_rows if e.effect is not None},
        }

    # ---------------------------------------------------------------- finalize
    def finalize(self, candidate_id: str) -> dict[str, Any]:
        """Frozen, independent holdout evaluation.  Separate entry, no search."""
        ctx = self._build_context()
        protocol = ctx.protocol
        if protocol.holdout_fold is None:
            raise ProtocolViolation("protocol has no holdout fold")
        cand = self.store.get_candidate(candidate_id)
        if not cand:
            raise ValueError(f"unknown candidate {candidate_id}")
        evidence = [e for e in self.store.get_evidence(candidate_id) if e["protocol_hash"] == ctx.protocol_hash]
        if not evidence or evidence[-1]["status"] != EvidenceStatus.VALIDATED.value:
            raise ProtocolViolation(
                f"candidate {candidate_id} is not validated; refusing holdout access "
                f"(status={evidence[-1]['status'] if evidence else 'none'})"
            )
        config_hash = self._trial_config(candidate_id, "candidate", cand["descriptor"],
                                         protocol.holdout_fold, "final", ctx.policy.seed).config_hash
        manifest = {
            "candidate_id": candidate_id,
            "template_hash": cand["template_hash"],
            "protocol_hash": ctx.protocol_hash,
            "config_hash": config_hash,
            "seed": ctx.policy.seed,
            "frozen_at": utcnow().isoformat(),
        }
        manifest_hash = sha256_obj(manifest)
        (self.artifacts / "frozen_manifest.json").write_text(
            json.dumps({**manifest, "manifest_hash": manifest_hash}, indent=2), encoding="utf-8"
        )
        self.store.add_final_eval(FinalEvaluation(
            task_id=ctx.task.task_id, candidate_id=candidate_id,
            protocol_hash=ctx.protocol_hash, config_hash=config_hash,
            manifest_hash=manifest_hash, status="frozen",
            notes="candidate and protocol frozen before holdout access",
        ))
        ctx.guard = ctx.guard.unlock(manifest_hash)
        ctx.guard.assert_fold_accessible(protocol.holdout_fold.fold_id, "finalize")

        req = self._trial_config(candidate_id, "candidate", cand["descriptor"],
                                 protocol.holdout_fold, "final", ctx.policy.seed)
        trial = self.run_trial(req)
        if trial["status"] != "success":
            self.store.add_final_eval(FinalEvaluation(
                task_id=ctx.task.task_id, candidate_id=candidate_id, protocol_hash=ctx.protocol_hash,
                config_hash=config_hash, manifest_hash=manifest_hash, status="failed",
                notes=(trial.get("error") or "")[:1500],
            ))
            raise RuntimeError(f"final evaluation failed: {trial.get('error')}")
        base = self._ensure_variant(protocol.baseline_variant, "baseline", protocol.holdout_fold, "final", ctx.policy.seed)
        metric = protocol.primary_metric
        summary: dict[str, Any] = {"candidate": trial["metrics"], "baseline": base.get("metrics", {})}
        if base["status"] == "success":
            cs = trial.get("metric_series", {}).get(metric, {})
            bs = base.get("metric_series", {}).get(metric, {})
            summary["paired"] = paired_effect(
                cs, bs, nw_lag=ctx.policy.nw_lag, n_boot=ctx.policy.bootstrap_samples,
                block=ctx.policy.bootstrap_block, seed=ctx.policy.seed,
            )
        self.store.add_final_eval(FinalEvaluation(
            task_id=ctx.task.task_id, candidate_id=candidate_id, protocol_hash=ctx.protocol_hash,
            config_hash=config_hash, manifest_hash=manifest_hash, status="success",
            metrics=summary, notes="holdout evaluated once after freezing",
        ))
        (self.artifacts / "final_evaluation.json").write_text(json.dumps(
            {"manifest": manifest, "manifest_hash": manifest_hash, "result": summary},
            indent=2, default=str), encoding="utf-8")
        return summary

    # ------------------------------------------------------------------ report
    def _write_report(self) -> None:
        rounds = self.store.list_rounds(self.run_id)
        evidence = self.store.list_evidence(run_id=self.run_id)
        counts: dict[str, int] = {}
        for e in evidence:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        report = {
            "run_id": self.run_id,
            "domain": self._ctx.adapter.name,
            "protocol_hash": self._ctx.protocol_hash,
            "snapshot": self._ctx.snapshot.model_dump(mode="json"),
            "rounds": rounds,
            "evidence_counts": counts,
            "evidence": evidence,
        }
        (self.artifacts / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.store.export_evidence_jsonl(self.artifacts / "evidence.jsonl", run_id=self.run_id)

        lines = [
            f"# PredicateLab report — {self.run_id}",
            "",
            f"- domain: `{self._ctx.adapter.name}`",
            f"- protocol hash: `{self._ctx.protocol_hash[:16]}`",
            f"- snapshot: `{self._ctx.snapshot.snapshot_id}`",
            f"- data suitability: **{self._ctx.task.metadata.get('suitability', {}).get('status', 'unknown')}**",
            "",
            "## Evidence counts",
            "",
            "| status | count |",
            "|---|---|",
        ]
        for k, v in sorted(counts.items()):
            lines.append(f"| {k} | {v} |")
        lines += ["", "## Rounds", ""]
        for r in rounds:
            s = r["summary"] or {}
            lines.append(f"### Round {r['round_index']} ({r['round_id']})")
            lines.append("")
            lines.append(f"- proposed: {r['n_proposed']}, bound: {r['n_bound']}, validated: {r['n_validated']}, "
                         f"rejected: {r['n_rejected']}, insufficient: {r['n_insufficient']}")
            if s.get("validation"):
                lines += ["", "| candidate | status | effect | CI | p | q | stability |", "|---|---|---|---|---|---|---|"]
                for v in s["validation"]:
                    ci = v.get("ci") or [None, None]
                    ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if all(isinstance(x, (int, float)) for x in ci) else "—"
                    lines.append(f"| {v['candidate_id'][:12]} | {v['status']} | "
                                 f"{v['effect']:+.4f} | {ci_s} | {v['p']:.4g} | "
                                 f"{v['q'] if v['q'] is not None else float('nan'):.4g} | {v['stability']:.2f} |")
            lines.append("")
        (self.artifacts / "report.md").write_text("\n".join(lines), encoding="utf-8")

    # --------------------------------------------------------------- utilities
    def export(self, out_dir: str | Path) -> Path:
        """Copy the immutable evidence bundle to another directory."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("context.json", "report.json", "report.md", "evidence.jsonl", "final_evaluation.json"):
            src = self.artifacts / name
            if src.exists():
                shutil.copy2(src, out / name)
        return out


def finalize_run(config: dict[str, Any], run_id: str, candidate_id: str,
                 db_path: str | None = None) -> dict[str, Any]:
    engine = PredicateLabEngine(config, db_path=db_path, run_id=run_id)
    try:
        return engine.finalize(candidate_id)
    finally:
        engine.store.close()
