"""Generic discovery → confirmation → final engine.

The engine owns the scientific protocol; adapters own execution details.
V1.1 separates three stages explicitly:

``discovery``     propose → bind → probe → screen → validate on search folds;
``confirmation``  frozen candidate family evaluated on later, selection-free
                  folds (or disjoint resamples) -- results never feed the
                  generator;
``final``         the one-time holdout evaluation, guarded by a global
                  data-identity ledger.

Every cached execution is keyed by the complete experiment identity (data
manifest, label spec, fold dates, phase, seed, model/training budget, code and
environment).  A round row is only ``completed`` after evidence, feedback and
the status itself are committed; a crash therefore leaves it resumable instead
of silently "done".
"""

from __future__ import annotations

import json
import math
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from predicatelab.core.adapter import DomainAdapter, TrialRequest
from predicatelab.core.hashing import sha256_file, sha256_obj, short_id
from predicatelab.core.holdout import ConfirmationConsumed, HoldoutConsumed
from predicatelab.core.identity import (
    atomic_write_text,
    code_version,
    environment_versions,
    label_spec_hash,
    sanitise_json,
    strict_json_dumps,
)
from predicatelab.core.label_window import (
    LabelCalendar,
    LabelDependencyError,
    parse_future_offset,
    resolve_future_offset,
    validate_and_isolate,
)
from predicatelab.core.protocol import HoldoutGuard, ProtocolViolation, validate_protocol
from predicatelab.core.schema import (
    BoundCandidate,
    CandidateState,
    EvidenceRecord,
    EvidenceStatus,
    FinalEvaluation,
    FittedInstance,
    FoldSpec,
    PredicateTemplate,
    ResearchTask,
    RoundRecord,
    SearchPolicy,
    TrialRecord,
    utcnow,
)
from predicatelab.core.stats import (
    benjamini_hochberg,
    effective_num_tests,
    paired_effect,
    pool_fold_series,
    romano_wolf_stepdown,
    stability,
)
from predicatelab.core.store import StateStore, result_digest


def _now_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _std(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if _finite(v)]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    return float(np.std(vals, ddof=1))


def _selection_score(direction: str, effect: float | None) -> float:
    """Higher is always better, regardless of the metric's direction."""
    sign = 1.0 if direction == "higher" else -1.0
    return sign * (effect or 0.0)


_STATE_FOR_STATUS = {
    EvidenceStatus.VALIDATED: CandidateState.EXPLORATORY_POSITIVE,
    EvidenceStatus.REJECTED: CandidateState.REJECTED,
    EvidenceStatus.INSUFFICIENT_EVIDENCE: CandidateState.INSUFFICIENT,
    EvidenceStatus.SCREENED_OUT: CandidateState.EXECUTABLE,
    EvidenceStatus.DUPLICATE: CandidateState.EXECUTABLE,
    EvidenceStatus.INVALID_STRUCTURE: CandidateState.INSUFFICIENT,
    EvidenceStatus.EXECUTION_FAILED: CandidateState.INSUFFICIENT,
    EvidenceStatus.BLOCKED: CandidateState.INSUFFICIENT,
}


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
    # ---- V1.1 identity and effective (label-trimmed) folds -----------------
    code: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    data_manifest_hash: str = ""
    label_spec: str = ""
    label_expression: str = ""
    label_offset: int = 0
    label_report: dict[str, Any] = field(default_factory=dict)
    effective_folds: list[FoldSpec] = field(default_factory=list)
    effective_confirmation: list[FoldSpec] = field(default_factory=list)
    effective_holdout: FoldSpec | None = None
    context_hash: str = ""
    frozen_family: dict[str, Any] | None = None


class PredicateLabEngine:
    def __init__(self, config: dict[str, Any], *, db_path: str | None = None,
                 run_id: str | None = None, allow_new_run: bool = True):
        self.config = config
        self.run_cfg = config.get("run", {})
        self.artifacts_root = Path(self.run_cfg.get("artifacts_dir", "artifacts"))
        self.db_path = db_path or self.run_cfg.get("db", str(self.artifacts_root / "predicatelab.sqlite"))
        self.store = StateStore(
            self.db_path,
            global_ledger_path=self.run_cfg.get("global_ledger_db"),
        )
        self.run_id = run_id or _now_id("run")
        self._allow_new_run = allow_new_run
        self.artifacts = self.artifacts_root / self.run_id
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._ctx: RunContext | None = None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "PredicateLabEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ setup
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _external_code_roots(self) -> list[str]:
        roots: list[str] = []
        data_cfg = self.config.get("data", {}) if isinstance(self.config.get("data"), dict) else {}
        tsil_root = data_cfg.get("tsil_root")
        if tsil_root:
            roots.append(str(tsil_root))
        extra = data_cfg.get("external_code_roots")
        if isinstance(extra, (list, tuple)):
            roots.extend(str(p) for p in extra if p)
        return roots

    def _check_suitability(self, suitability: dict[str, Any]) -> None:
        """Fail closed on blockers or unacknowledged warnings."""
        status = str(suitability.get("status", "ok")).lower()
        blockers = suitability.get("blockers") or []
        warnings = suitability.get("warnings") or []
        if blockers or status in {"blocked", "fail", "failed", "unsuitable"}:
            raise RuntimeError(
                f"data suitability is {status!r}; blockers={blockers or suitability.get('checks')}; "
                "refusing to run without an explicit, auditable decision"
            )
        if not warnings:
            return
        ack = self.run_cfg.get("warning_acknowledgements")
        reason = str(self.run_cfg.get("warning_ack_reason", "")).strip()
        hashes = [sha256_obj(w)[:16] for w in warnings]
        accepted = False
        if ack in (True, "all") and reason:
            accepted = True
        elif isinstance(ack, (list, tuple, set)) and reason:
            accepted = all(h in set(ack) for h in hashes)
        if not accepted:
            raise RuntimeError(
                "data suitability produced unacknowledged warnings "
                f"{hashes}; add them to run.warning_acknowledgements with "
                "run.warning_ack_reason to proceed (fail closed by design)"
            )

    def _resolve_label_dependency(self, contract) -> tuple[str, dict[str, Any], int]:
        expression = contract.label_definition or ""
        window = dict(contract.label_window or {})
        dep = dict(contract.label_dependency or {})
        for key in ("expression", "label_expression"):
            if dep.get(key):
                expression = str(dep[key])
        for key in ("window", "label_window"):
            if isinstance(dep.get(key), dict):
                window.update(dep[key])
        if not expression:
            # No declared label expression: only same-sample labels are provable.
            declared = window.get("uses_future_days", window.get("future_days"))
            if declared not in (0, "0"):
                raise LabelDependencyError(
                    "data contract does not declare a label expression or an explicit "
                    "same-sample label window; cannot prove time isolation"
                )
            return expression, window, 0
        try:
            offset = resolve_future_offset(expression, window)
        except LabelDependencyError:
            parsed = parse_future_offset(expression)
            if parsed is None and not window:
                offset = 0
            else:
                raise
        return expression, window, int(offset)

    def _build_context(self) -> RunContext:
        from predicatelab.domains import create_adapter

        if self._ctx is not None:
            return self._ctx
        adapter = create_adapter(self.config)
        protocol = adapter.build_protocol(self.config)
        contract = adapter.data_contract(self.config)
        snapshot = adapter.snapshot(self.config)
        suitability = adapter.suitability(snapshot, self.config)
        self._check_suitability(suitability)

        expression, window, offset = self._resolve_label_dependency(contract)
        validate_protocol(
            protocol.folds, protocol.holdout_fold, protocol.confirmation_folds,
            label_future_offset=offset if offset > 0 else None,
        )
        calendar = adapter.trading_calendar(self.config) if offset > 0 else None
        if offset > 0 and not calendar:
            raise LabelDependencyError(
                f"label {expression!r} depends on {offset} future trading days but the adapter "
                "supplied no trading calendar; natural-day arithmetic is not accepted"
            )
        if offset > 0:
            adjusted, adjusted_holdout, label_report = validate_and_isolate(
                [*protocol.folds, *protocol.confirmation_folds], protocol.holdout_fold,
                calendar, expression, window,
            )
        else:
            adjusted = [*protocol.folds, *protocol.confirmation_folds]
            adjusted_holdout = protocol.holdout_fold
            label_report = {
                "label_expression": expression,
                "future_offset_trading_days": 0,
                "same_sample_label": True,
                "folds": {f.fold_id: {"ok": True, "applicable": False,
                                      "reason": "declared same-sample label"} for f in adjusted},
                "ok": True,
            }
        by_id = {f.fold_id: f for f in adjusted}
        effective_folds = [by_id[f.fold_id] for f in protocol.folds]
        effective_confirmation = [by_id[f.fold_id] for f in protocol.confirmation_folds]

        policy = SearchPolicy.model_validate(self.config.get("search", {}))
        if not policy.screen_folds:
            policy.screen_folds = [effective_folds[-1].fold_id]
        if not policy.validation_folds:
            # Default to the discovery folds *not* used for screening.
            policy.validation_folds = [f.fold_id for f in effective_folds
                                       if f.fold_id not in set(policy.screen_folds)]
        known = {f.fold_id for f in effective_folds}
        bad = (set(policy.screen_folds) | set(policy.validation_folds)) - known
        if bad:
            raise ProtocolViolation(f"policy references unknown discovery folds: {sorted(bad)}")
        overlap = sorted(set(policy.screen_folds) & set(policy.validation_folds))
        if overlap:
            raise ProtocolViolation(
                f"screen_folds and validation_folds must be disjoint; overlap={overlap}. "
                "A candidate promoted on a fold may not also receive its 'independent' "
                "validation effect from that same fold (selection leakage)."
            )
        if not policy.validation_folds:
            raise ProtocolViolation(
                "no validation folds remain after removing the screening folds; "
                "declare at least one disjoint validation fold"
            )

        task = ResearchTask(
            title=self.config.get("task", {}).get("title", self.run_cfg.get("name", "untitled")),
            question=self.config.get("task", {}).get("question", ""),
            domain=adapter.name,
            hypothesis=self.config.get("task", {}).get("hypothesis", ""),
            evaluation=protocol.evaluation or adapter.evaluation,
            data_contract=contract,
            metadata={"suitability": suitability},
        ).ensure_id()
        self.store.save_task(task)
        self.store.save_snapshot(snapshot)

        protocol_hash = adapter.protocol_hash(
            protocol, self.config, policy=policy.model_dump(mode="json"),
        )
        protocol.ensure_id()
        code = code_version(self._repo_root(), extra_roots=self._external_code_roots())
        environment = environment_versions()
        data_manifest_hash = str(
            snapshot.details.get("data_manifest_hash")
            or snapshot.details.get("merkle_root")
            or snapshot.manifest_hash
        )
        label_spec = label_spec_hash(contract)
        context_hash = sha256_obj({
            "task": task.task_id,
            "domain": adapter.name,
            "protocol_hash": protocol_hash,
            "snapshot_hash": snapshot.manifest_hash,
            "data_manifest_hash": data_manifest_hash,
            "label_spec": label_spec,
            "label_offset": offset,
            "effective_folds": [f.model_dump(mode="json") for f in effective_folds],
            "effective_confirmation": [f.model_dump(mode="json") for f in effective_confirmation],
            "effective_holdout": (adjusted_holdout.model_dump(mode="json")
                                  if adjusted_holdout else None),
            "code": code,
            "environment": environment,
            "model_spec": protocol.model_spec,
        })
        guard = HoldoutGuard.from_folds(
            effective_folds, adjusted_holdout, confirmation_folds=effective_confirmation,
        )
        ctx = RunContext(
            run_id=self.run_id, config=self.config, adapter=adapter, store=self.store,
            task=task, protocol=protocol, policy=policy, snapshot=snapshot,
            artifacts=self.artifacts, guard=guard, protocol_hash=protocol_hash,
            code=code, environment=environment, data_manifest_hash=data_manifest_hash,
            label_spec=label_spec, label_expression=expression, label_offset=offset,
            label_report=label_report, effective_folds=effective_folds,
            effective_confirmation=effective_confirmation, effective_holdout=adjusted_holdout,
            context_hash=context_hash,
        )

        existing = self.store.get_run(self.run_id)
        if existing:
            if not self._allow_new_run:
                pass
            stored_protocol = existing.get("protocol_hash")
            stored_snapshot = existing.get("snapshot_hash")
            stored_code = existing.get("code_fingerprint")
            current_code = code.get("dirty_fingerprint", "")
            if stored_protocol and stored_protocol != protocol_hash:
                raise ProtocolViolation(
                    f"run {self.run_id} was created with protocol {stored_protocol[:16]}, "
                    f"current config hashes to {protocol_hash[:16]}; refusing to mix runs"
                )
            if stored_snapshot and stored_snapshot != snapshot.manifest_hash:
                raise ProtocolViolation(
                    f"run {self.run_id} snapshot changed ({stored_snapshot[:16]} -> "
                    f"{snapshot.manifest_hash[:16]}); cached baselines are invalid"
                )
            if stored_code and current_code and stored_code != current_code:
                raise ProtocolViolation(
                    f"run {self.run_id} code_fingerprint changed ({stored_code[:16]} -> "
                    f"{current_code[:16]}); refusing to mix code versions within a run"
                )
        else:
            self.store.create_run(
                self.run_id, task.task_id, self.config, str(self.artifacts),
                snapshot_hash=snapshot.manifest_hash, protocol_hash=protocol_hash,
                code_fingerprint=code.get("dirty_fingerprint", ""),
            )
        # Cache only after the stored identity has been checked.  Caching before
        # the ProtocolViolation left a mismatched context available to a caller
        # that caught the exception and called run() again.
        self._ctx = ctx
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
            "effective_protocol": {
                "folds": [f.model_dump(mode="json") for f in ctx.effective_folds],
                "confirmation_folds": [f.model_dump(mode="json") for f in ctx.effective_confirmation],
                "holdout": ctx.effective_holdout.model_dump(mode="json") if ctx.effective_holdout else None,
            },
            "protocol_hash": ctx.protocol_hash,
            "snapshot": ctx.snapshot.model_dump(mode="json"),
            "data_manifest_hash": ctx.data_manifest_hash,
            "label_spec_hash": ctx.label_spec,
            "label_expression": ctx.label_expression,
            "label_offset_trading_days": ctx.label_offset,
            "label_isolation": ctx.label_report,
            "evaluation": ctx.adapter.evaluation.model_dump(mode="json"),
            "data_contract": ctx.task.data_contract.model_dump(mode="json"),
            "policy": ctx.policy.model_dump(mode="json"),
            "code": ctx.code,
            "environment": ctx.environment,
            "identity_hash": ctx.context_hash,
        }
        atomic_write_text(self.artifacts / "context.json", strict_json_dumps(meta))
        if self.store.get_run(self.run_id) is None:
            self.store.create_run(
                self.run_id, ctx.task.task_id, self.config, str(self.artifacts),
                snapshot_hash=ctx.snapshot.manifest_hash, protocol_hash=ctx.protocol_hash,
                code_fingerprint=ctx.code.get("dirty_fingerprint", ""),
            )

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

    def _seeds(self) -> list[int]:
        ctx = self._ctx
        assert ctx is not None
        seeds = [int(s) for s in (ctx.protocol.seeds or [])]
        if not seeds:
            seeds = [int(self.config.get("search", {}).get("seed", 2025))]
        return seeds

    # ----------------------------------------------------------------- trials
    def _trial_config(self, candidate_id: str, variant: str, descriptor: dict,
                      fold: FoldSpec, phase: str, seed: int, *,
                      control_kind: str = "") -> TrialRequest:
        ctx = self._ctx
        assert ctx is not None
        training = self._training_config(phase)
        req = TrialRequest(
            candidate_id=candidate_id,
            variant=variant,
            descriptor=descriptor,
            fold=fold,
            seed=seed,
            training_config=training,
            workdir="",
            protocol_hash=ctx.protocol_hash,
            config_hash="",
            save_predictions=ctx.policy.save_predictions,
            phase=phase,
            snapshot_hash=ctx.snapshot.manifest_hash,
            data_manifest_hash=ctx.data_manifest_hash,
            label_spec_hash=ctx.label_spec,
            preprocessing_version=ctx.task.data_contract.preprocessing_version,
            model_spec=ctx.protocol.model_spec,
            code_identity=ctx.code,
            environment=ctx.environment,
            run_id=self.run_id,
            label_expression=ctx.label_expression,
            label_offset=ctx.label_offset,
            control_kind=control_kind,
        )
        cfg_hash = ctx.adapter.trial_config_hash(req)
        # Phase-, fold-, variant- and seed-specific directories: two stages can
        # never overwrite each other's artifacts or result.json.
        safe_variant = variant.replace(":", "_").replace("/", "_")
        workdir = (
            self.artifacts / "trials" / phase / fold.fold_id
            / candidate_id.replace(":", "_") / safe_variant / f"seed_{seed}" / cfg_hash[:12]
        )
        workdir.mkdir(parents=True, exist_ok=True)
        req.config_hash = cfg_hash
        req.workdir = str(workdir)
        return req

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        """Run one trial with verified cache reuse and attempt logging."""
        ctx = self._ctx
        assert ctx is not None
        cached = self.store.get_cached_trial(
            request.candidate_id, request.fold.fold_id, request.protocol_hash, request.config_hash
        )
        if cached:
            return cached

        prior = self.store.get_trial_by_key(
            request.candidate_id, request.fold.fold_id, request.protocol_hash, request.config_hash
        )
        attempt = int(prior.get("attempt", 0)) + 1 if prior else 1
        started = datetime.now(timezone.utc)
        running = TrialRecord(
            candidate_id=request.candidate_id, fold_id=request.fold.fold_id,
            protocol_hash=request.protocol_hash, config_hash=request.config_hash,
            status="running", phase=request.phase, seed=request.seed,
            snapshot_hash=request.snapshot_hash,
            identity_hash=request.config_hash, attempt=attempt,
            started_at=started.isoformat(),
        )
        self.store.record_attempt(running, run_id=self.run_id)

        # Stale result isolation: never let a previous attempt's result.json
        # masquerade as this attempt's output.
        workdir = Path(request.workdir)
        for name in ("result.json", "trial_result.json", "worker_result.json"):
            stale = workdir / name
            if stale.exists():
                stale.unlink()

        try:
            result = ctx.adapter.run_trial(request) or {}
        except Exception as exc:  # adapters should catch; last-resort containment
            result = {
                "status": "failed", "metrics": {}, "metric_series": {},
                "fitted_params": {}, "predictions_path": "", "duration_s": 0.0,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "worker": {},
            }
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        finished = datetime.now(timezone.utc).isoformat()

        metrics, null_metrics = sanitise_json(result.get("metrics", {}) or {})
        series, null_series = sanitise_json(result.get("metric_series", {}) or {})
        fitted, null_fitted = sanitise_json(result.get("fitted_params", {}) or {})
        null_paths = [*null_metrics, *null_series, *null_fitted]
        status = str(result.get("status", "failed"))
        error = str(result.get("error", "") or "")
        if status == "success" and null_paths:
            status = "failed"
            error = (f"non-finite result rejected at {null_paths[:5]}; NaN/Inf cannot be a success; "
                     + error).strip("; ")
        if status == "success" and not metrics and not series:
            status = "failed"
            error = "worker returned success with empty metrics and metric_series"
        if status not in {"success", "failed", "interrupted", "skipped"}:
            status = "failed"
            error = f"unknown worker status {result.get('status')!r}; " + error
        if status != "success" and null_paths:
            error = (error + f"; null paths: {null_paths[:20]}").strip("; ")

        pred_path = str(result.get("predictions_path", "") or "")
        pred_sha = str(result.get("predictions_sha256", "") or "")
        if pred_path and Path(pred_path).exists():
            pred_sha = pred_sha or sha256_file(Path(pred_path))

        # Trained weights are optional adapter output (worker ``models/``
        # directory).  Keep them in the worker metadata rather than adding a
        # schema column: the result hash stays stable for existing databases.
        worker_meta = dict(result.get("worker", {}) or {})
        for key in ("model_path", "model_sha256"):
            value = str(result.get(key, "") or "")
            if value:
                worker_meta.setdefault(key, value)

        payload = {
            "status": status, "metrics": metrics, "metric_series": series,
            "fitted_params": fitted, "predictions_path": pred_path,
            "predictions_sha256": pred_sha, "null_paths": null_paths,
        }
        trial = TrialRecord(
            candidate_id=request.candidate_id, fold_id=request.fold.fold_id,
            protocol_hash=request.protocol_hash, config_hash=request.config_hash,
            status=status, phase=request.phase, seed=request.seed,
            snapshot_hash=request.snapshot_hash,
            identity_hash=request.config_hash, attempt=attempt,
            result_hash=result_digest(payload),
            predictions_sha256=pred_sha, null_paths=null_paths,
            metrics=metrics, metric_series=series, fitted_params=fitted,
            predictions_path=pred_path,
            duration_s=float(result.get("duration_s", duration) or duration),
            error=error, worker=worker_meta,
            started_at=started.isoformat(), finished_at=finished,
        )
        self.store.record_attempt(trial, run_id=self.run_id)
        self.store.record_trial(trial, run_id=self.run_id)
        if status == "success" and fitted:
            self.store.add_fitted_instance(FittedInstance(
                candidate_id=request.candidate_id, fold_id=request.fold.fold_id,
                fitted_params=fitted,
                fit_data_hash=result.get("fit_data_hash") or sha256_obj({
                    "fold": request.fold.model_dump(mode="json"),
                    "candidate": request.candidate_id,
                    "snapshot": request.snapshot_hash,
                    "config": request.config_hash,
                }),
            ))
        if status != "success":
            print(f"[trial-failed] {request.candidate_id} {request.fold.fold_id} "
                  f"seed={request.seed} phase={request.phase}: "
                  f"{error.splitlines()[0] if error else 'unknown'}")
        return self.store.get_trial_by_key(
            request.candidate_id, request.fold.fold_id,
            request.protocol_hash, request.config_hash,
        ) or trial.model_dump(mode="json")

    def run_trials_batch(self, requests: list[TrialRequest]) -> list[dict[str, Any]]:
        """Execute a batch of TrialRequests using up to ``execution.workers`` parallel worker threads
        while keeping all SQLite store reads and writes strictly on the main thread."""
        max_w = self._max_workers()
        if max_w <= 1 or len(requests) <= 1:
            return [self.run_trial(req) for req in requests]

        from concurrent.futures import ThreadPoolExecutor

        ctx = self._ctx
        assert ctx is not None

        # Deduplicate requests by (candidate_id, fold_id, protocol_hash, config_hash)
        unique_pending: list[tuple[TrialRequest, int, datetime]] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for req in requests:
            key = (req.candidate_id, req.fold.fold_id, req.protocol_hash, req.config_hash)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cached = self.store.get_cached_trial(*key)
            if cached:
                continue
            prior = self.store.get_trial_by_key(*key)
            attempt = int(prior.get("attempt", 0)) + 1 if prior else 1
            req.attempt = attempt
            started = datetime.now(timezone.utc)
            running = TrialRecord(
                candidate_id=req.candidate_id,
                fold_id=req.fold.fold_id,
                protocol_hash=req.protocol_hash,
                config_hash=req.config_hash,
                status="running",
                phase=req.phase,
                seed=req.seed,
                snapshot_hash=req.snapshot_hash,
                identity_hash=req.config_hash,
                attempt=attempt,
                started_at=started.isoformat(),
            )
            self.store.record_attempt(running, run_id=self.run_id)
            workdir = Path(req.workdir)
            for name in ("result.json", "trial_result.json", "worker_result.json"):
                stale = workdir / name
                if stale.exists():
                    stale.unlink()
            unique_pending.append((req, attempt, started))

        def _exec_adapter(req: TrialRequest) -> dict[str, Any]:
            try:
                return ctx.adapter.run_trial(req) or {}
            except Exception as exc:
                return {
                    "status": "failed",
                    "metrics": {},
                    "metric_series": {},
                    "fitted_params": {},
                    "predictions_path": "",
                    "duration_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    "worker": {},
                }

        if unique_pending:
            with ThreadPoolExecutor(max_workers=min(max_w, len(unique_pending))) as pool:
                raw_results = list(pool.map(_exec_adapter, [item[0] for item in unique_pending]))

            for (req, attempt, started), result in zip(unique_pending, raw_results):
                duration = (datetime.now(timezone.utc) - started).total_seconds()
                finished = datetime.now(timezone.utc).isoformat()
                metrics, null_metrics = sanitise_json(result.get("metrics", {}) or {})
                series, null_series = sanitise_json(result.get("metric_series", {}) or {})
                fitted, null_fitted = sanitise_json(result.get("fitted_params", {}) or {})
                null_paths = [*null_metrics, *null_series, *null_fitted]
                status = str(result.get("status", "failed"))
                error = str(result.get("error", "") or "")
                if status == "success" and null_paths:
                    status = "failed"
                    error = (
                        f"non-finite result rejected at {null_paths[:5]}; NaN/Inf cannot be a success; "
                        + error
                    ).strip("; ")
                if status == "success" and not metrics and not series:
                    status = "failed"
                    error = "worker returned success with empty metrics and metric_series"
                if status not in {"success", "failed", "interrupted", "skipped"}:
                    status = "failed"
                    error = f"unknown worker status {result.get('status')!r}; " + error
                if status != "success" and null_paths:
                    error = (error + f"; null paths: {null_paths[:20]}").strip("; ")

                pred_path = str(result.get("predictions_path", "") or "")
                pred_sha = str(result.get("predictions_sha256", "") or "")
                if pred_path and Path(pred_path).exists():
                    pred_sha = pred_sha or sha256_file(Path(pred_path))

                payload = {
                    "status": status,
                    "metrics": metrics,
                    "metric_series": series,
                    "fitted_params": fitted,
                    "predictions_path": pred_path,
                    "predictions_sha256": pred_sha,
                    "null_paths": null_paths,
                }
                trial = TrialRecord(
                    candidate_id=req.candidate_id,
                    fold_id=req.fold.fold_id,
                    protocol_hash=req.protocol_hash,
                    config_hash=req.config_hash,
                    status=status,
                    phase=req.phase,
                    seed=req.seed,
                    snapshot_hash=req.snapshot_hash,
                    identity_hash=req.config_hash,
                    attempt=attempt,
                    result_hash=result_digest(payload),
                    predictions_sha256=pred_sha,
                    null_paths=null_paths,
                    metrics=metrics,
                    metric_series=series,
                    fitted_params=fitted,
                    predictions_path=pred_path,
                    duration_s=float(result.get("duration_s", duration) or duration),
                    error=error,
                    worker=result.get("worker", {}) or {},
                    started_at=started.isoformat(),
                    finished_at=finished,
                )
                self.store.record_attempt(trial, run_id=self.run_id)
                self.store.record_trial(trial, run_id=self.run_id)
                if status == "success" and fitted:
                    self.store.add_fitted_instance(
                        FittedInstance(
                            candidate_id=req.candidate_id,
                            fold_id=req.fold.fold_id,
                            fitted_params=fitted,
                            fit_data_hash=result.get("fit_data_hash")
                            or sha256_obj(
                                {
                                    "fold": req.fold.model_dump(mode="json"),
                                    "candidate": req.candidate_id,
                                    "snapshot": req.snapshot_hash,
                                    "config": req.config_hash,
                                }
                            ),
                        )
                    )
                if status != "success":
                    print(
                        f"[trial-failed] {req.candidate_id} {req.fold.fold_id} "
                        f"seed={req.seed} phase={req.phase}: "
                        f"{error.splitlines()[0] if error else 'unknown'}"
                    )

        return [self.run_trial(req) for req in requests]

    def _ensure_variant(self, variant_id: str, variant: str, fold: FoldSpec,
                        phase: str, seed: int, *, control_kind: str = "") -> dict[str, Any]:
        req = self._trial_config(variant_id, variant, {}, fold, phase, seed,
                                 control_kind=control_kind)
        return self.run_trial(req)

    # ------------------------------------------------------------------ probe
    def _probe(self, descriptor: dict, fold: FoldSpec, workdir: str) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        # Probe reads the discovery training fold only; identity is part of the
        # directory so a data/fold change cannot reuse an old probe.
        probe_dir = Path(workdir) / sha256_obj({
            "snapshot": ctx.data_manifest_hash, "fold": fold.model_dump(mode="json"),
            "label": ctx.label_spec, "descriptor": descriptor,
        })[:16]
        try:
            return ctx.adapter.probe(descriptor, fold, ctx.snapshot, self.config, str(probe_dir))
        except RuntimeError as exc:
            if str(exc).startswith("probe batch failed"):
                raise
            return {"ok": False, "reason": f"probe error: {type(exc).__name__}: {exc}",
                    "severity": "invalid"}
        except Exception as exc:
            return {"ok": False, "reason": f"probe error: {type(exc).__name__}: {exc}",
                    "severity": "invalid"}

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
    def _seed_aggregate(self, seed: int, cand_series: Mapping[str, Mapping[str, float]],
                        base_series: Mapping[str, Mapping[str, float]],
                        fold_metrics: Mapping[str, dict],
                        ctrl_series: Mapping[str, Mapping[str, float]] | None = None,
                        ctrl_variant: str = "") -> dict[str, Any]:
        policy = self._ctx.policy
        try:
            pooled_c = pool_fold_series(cand_series)
            pooled_b = pool_fold_series(base_series)
        except ValueError as exc:
            return {"error": str(exc)}
        eff = paired_effect(
            pooled_c, pooled_b, nw_lag=policy.nw_lag,
            n_boot=policy.bootstrap_samples, block=policy.bootstrap_block, seed=seed,
        )
        fold_effects = {k: v["effect"] for k, v in fold_metrics.items()}
        n_dates = int(eff["n_dates"])
        # A single pooled observation (static holdout) can produce an effect but
        # not a time-series p-value/CI.  Report that explicitly as ``None`` plus
        # ``inferential=False`` instead of a NaN that could sneak into success.
        inferential = n_dates >= 2
        common_keys = sorted(set(pooled_c) & set(pooled_b))
        delta_series = {
            str(k): float(pooled_c[k] - pooled_b[k])
            for k in common_keys
            if _finite(pooled_c[k]) and _finite(pooled_b[k])
        }
        out: dict[str, Any] = {
            "seed": seed,
            "effect": eff["effect"],
            "ci_low": eff["ci_low"] if inferential else None,
            "ci_high": eff["ci_high"] if inferential else None,
            "p_value": eff["p_value"] if inferential else None,
            "se": eff["se"] if inferential else None,
            "cohen_dz": eff["cohen_dz"] if inferential else None,
            "n_dates": n_dates, "n_folds": len(fold_metrics),
            "inferential": inferential,
            "stability": stability(fold_effects, self._ctx.protocol.direction),
            "fold_effects": fold_effects, "folds": dict(fold_metrics),
            "mean_candidate": float(np.mean(list(pooled_c.values()))) if pooled_c else None,
            "mean_baseline": float(np.mean(list(pooled_b.values()))) if pooled_b else None,
            "delta_series": delta_series,
        }
        if ctrl_series:
            try:
                pooled_ctrl = pool_fold_series(ctrl_series)
                ctrl_eff = paired_effect(
                    pooled_c, pooled_ctrl, nw_lag=policy.nw_lag,
                    n_boot=policy.bootstrap_samples, block=policy.bootstrap_block, seed=seed,
                )
                ctrl_inferential = int(ctrl_eff["n_dates"]) >= 2
                out["control_variant"] = ctrl_variant
                out["control_effect"] = ctrl_eff["effect"]
                out["control_ci_low"] = ctrl_eff["ci_low"] if ctrl_inferential else None
                out["control_ci_high"] = ctrl_eff["ci_high"] if ctrl_inferential else None
                out["control_p_value"] = ctrl_eff["p_value"] if ctrl_inferential else None
                out["mean_control"] = float(np.mean(list(pooled_ctrl.values()))) if pooled_ctrl else None
            except ValueError as exc:
                return {"error": f"control series pooling error: {exc}"}
        return out

    def _aggregate(self, per_seed: Mapping[int, dict[str, Any]], *,
                   n_expected_folds: int, failures: list[str]) -> dict[str, Any]:
        """Combine seeds while keeping the three uncertainty sources separate."""
        policy = self._ctx.policy
        direction = 1 if self._ctx.protocol.direction == "higher" else -1
        usable = {s: a for s, a in per_seed.items() if "error" not in a and _finite(a.get("effect"))}
        if not usable:
            return {"error": "no finite seed aggregate: " + "; ".join(failures[:5])}
        seeds = sorted(usable)
        effects = [usable[s]["effect"] for s in seeds]
        pvals = [usable[s]["p_value"] for s in seeds if _finite(usable[s].get("p_value"))]
        lo = [usable[s]["ci_low"] for s in seeds if _finite(usable[s].get("ci_low"))]
        hi = [usable[s]["ci_high"] for s in seeds if _finite(usable[s].get("ci_high"))]
        stabs = [usable[s]["stability"] for s in seeds if _finite(usable[s].get("stability"))]
        n_folds = min(int(usable[s]["n_folds"]) for s in seeds)
        coverage = (n_folds * len(usable)) / max(1, n_expected_folds * len(per_seed))
        primary = usable[seeds[0]]
        fold_effects = primary.get("fold_effects", {})
        eff = float(np.mean(effects))
        # Conservative combination: union of seed CIs, worst (largest) p-value.
        inferential = bool(pvals and primary.get("inferential", True))
        p_value = max(pvals) if inferential else None
        ci_low = min(lo) if (inferential and lo) else None
        ci_high = max(hi) if (inferential and hi) else None
        stab = float(np.mean(stabs)) if stabs else float("nan")
        seed_effects = {str(s): usable[s]["effect"] for s in seeds}
        agreement = float(np.mean([(direction * e) > 0 for e in effects])) if effects else float("nan")
        # Combine delta_series across seeds (mean paired difference per date/sample key)
        all_delta_keys = sorted({k for s in seeds for k in (usable[s].get("delta_series") or {})})
        pooled_delta: dict[str, float] = {}
        for k in all_delta_keys:
            vals = [usable[s]["delta_series"][k] for s in seeds if k in (usable[s].get("delta_series") or {})]
            if vals:
                pooled_delta[k] = float(np.mean(vals))
        out = {
            "effect": eff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "se": primary.get("se"),
            "cohen_dz": primary.get("cohen_dz"),
            "n_dates": min(int(usable[s].get("n_dates", 0)) for s in seeds),
            "n_seed_dates_total": sum(int(usable[s].get("n_dates", 0)) for s in seeds),
            "n_folds": n_folds,
            "n_expected_folds": n_expected_folds,
            "fold_coverage": coverage,
            "stability": stab,
            "fold_effects": fold_effects,
            "folds": primary.get("folds", {}),
            "seed_effects": seed_effects,
            "per_seed": {
                str(s): {k: v for k, v in usable[s].items() if k not in {"folds", "delta_series"}}
                for s in seeds
            },
            "seed_effect_sd": _std(effects),
            "fold_effect_sd": _std(list(fold_effects.values())),
            "time_series_ci": [primary.get("ci_low"), primary.get("ci_high")],
            "seed_agreement": agreement,
            "n_seeds": len(seeds),
            "inferential": inferential,
            "failed_folds": failures,
            "mean_candidate": primary.get("mean_candidate"),
            "mean_baseline": primary.get("mean_baseline"),
            "delta_series": pooled_delta,
        }
        ctrl_effects = [
            usable[s]["control_effect"] for s in seeds if _finite(usable[s].get("control_effect"))
        ]
        if ctrl_effects or primary.get("control_variant"):
            ctrl_pvals = [
                usable[s]["control_p_value"] for s in seeds if _finite(usable[s].get("control_p_value"))
            ]
            ctrl_lo = [
                usable[s]["control_ci_low"] for s in seeds if _finite(usable[s].get("control_ci_low"))
            ]
            ctrl_hi = [
                usable[s]["control_ci_high"] for s in seeds if _finite(usable[s].get("control_ci_high"))
            ]
            out["control_variant"] = primary.get("control_variant") or (
                self._ctx.protocol.control_variant if self._ctx else ""
            )
            out["control_effect"] = float(np.mean(ctrl_effects)) if ctrl_effects else None
            out["control_p_value"] = max(ctrl_pvals) if ctrl_pvals else None
            out["control_ci_low"] = min(ctrl_lo) if ctrl_lo else None
            out["control_ci_high"] = max(ctrl_hi) if ctrl_hi else None
            out["mean_control"] = primary.get("mean_control")
        for key in ("effect", "stability", "fold_coverage"):
            if not _finite(out.get(key)):
                return {"error": f"non-finite aggregate {key}={out.get(key)!r}; " + "; ".join(failures[:3])}
        for key in ("ci_low", "ci_high", "p_value"):
            if out[key] is not None and not _finite(out[key]):
                return {"error": f"non-finite aggregate {key}={out[key]!r}; " + "; ".join(failures[:3])}
        return out

    # ------------------------------------------------------------------ round
    def run(self, rounds: int | None = None) -> str:
        ctx = self._build_context()
        rounds = int(rounds if rounds is not None else self.run_cfg.get("rounds", 1))
        self._persist_context()
        run_row = self.store.get_run(self.run_id) or {}
        frozen_family = self.store.get_frozen_family(self.run_id)
        existing = {r["round_index"] for r in self.store.list_rounds(self.run_id)
                    if r["status"] == "completed"}
        if frozen_family is not None or run_row.get("status") in {
            "confirmation_complete", "final_complete", "final_failed"
        }:
            if rounds > len(existing):
                raise ProtocolViolation(
                    f"run {self.run_id} is frozen / post-discovery "
                    f"(status={run_row.get('status')!r}, frozen_family={bool(frozen_family)}); "
                    f"refusing to append new discovery rounds ({len(existing)} -> {rounds})"
                )
            print(f"[run] {self.run_id} already post-freeze, reusing")
            self._write_report()
            return self.run_id
        if run_row.get("status") == "search_complete":
            if len(existing) >= rounds:
                print(f"[run] {self.run_id} already search-complete, reusing")
                self._write_report()
                return self.run_id
        self.store.set_run_status(self.run_id, "running")
        print(f"[run] {self.run_id} domain={ctx.adapter.name} snapshot={ctx.snapshot.snapshot_id} "
              f"seeds={self._seeds()}")

        feedback = self.store.latest_feedback(self.run_id)
        for idx in range(rounds):
            row = self.store.get_round(self.run_id, idx)
            if row and row["status"] == "completed":
                print(f"[round {idx}] already completed, reusing evidence")
                feedback = self.store.latest_feedback(self.run_id)
                continue
            feedback = self._run_round(idx, feedback)
        self._write_report()
        self.store.finish_run(self.run_id, "search_complete", stage="discovery")
        # Stable task identity check: regenerate and compare to the stored task.
        stored = self.store.get_task(ctx.task.task_id)
        if stored is None:
            raise RuntimeError("task row missing after run")
        return self.run_id

    def _run_round(self, round_index: int, feedback: dict[str, Any] | None) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        policy = ctx.policy
        # A resumed round inherits evidence already committed for this round;
        # the generator must never see confirmation/final evidence.
        discovery_evidence = [e for e in self.store.list_evidence(run_id=self.run_id, stage="discovery")]
        run_candidate_ids = {e["candidate_id"] for e in discovery_evidence
                             if not e["candidate_id"].startswith("unbound")}
        existing_round = self.store.get_round(self.run_id, round_index)
        own_round_ids = ({e["candidate_id"] for e in discovery_evidence
                          if e["round_id"] == existing_round["round_id"]}
                         if existing_round else set())
        saved_proposals = ((existing_round or {}).get("summary") or {}).get("proposals")
        if saved_proposals:
            # Resume the exact proposal set, so an interrupted round cannot
            # silently replace its candidates or add a new post-hoc batch.
            templates = [PredicateTemplate.model_validate(p) for p in saved_proposals]
            print(f"[round {round_index}] resuming {len(templates)} saved proposals")
        else:
            known_hashes = set()
            for cid in run_candidate_ids - own_round_ids:
                row = self.store.get_candidate(cid)
                if row:
                    known_hashes.add(row["template_hash"])
            generator = ctx.adapter.make_generator(self.config, policy.model_dump())
            templates = generator.propose(
                round_index=round_index, n=policy.max_candidates_per_round,
                feedback=feedback, known_template_hashes=known_hashes,
            )
            print(f"[round {round_index}] proposed {len(templates)} templates")

        round_rec = RoundRecord(
            task_id=ctx.task.task_id, run_id=self.run_id, round_index=round_index,
            generator=policy.generator, parent_round_id=(feedback or {}).get("round_id"),
            n_proposed=len(templates), status="pending",
        ).ensure_id()
        if existing_round is None:
            self.store.save_round(round_rec)
        else:
            round_rec.round_id = existing_round["round_id"]
            self.store.update_round(round_rec.round_id, n_proposed=len(templates))
        if not saved_proposals:
            self.store.update_round(round_rec.round_id, summary={
                "proposals": [t.model_dump(mode="json") for t in templates],
            })
        self.store.start_round(round_rec.round_id)

        try:
            feedback_payload = self._run_round_body(round_rec, templates, run_candidate_ids, feedback)
        except KeyboardInterrupt:
            self.store.fail_round(round_rec.round_id, "interrupted", "KeyboardInterrupt")
            self.store.finish_run(self.run_id, "interrupted", stage="discovery")
            raise
        except Exception as exc:
            self.store.fail_round(
                round_rec.round_id, "failed", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            self.store.finish_run(self.run_id, "failed", stage="discovery")
            raise
        return feedback_payload

    def _run_round_body(self, round_rec: RoundRecord, templates: list[PredicateTemplate],
                        run_candidate_ids: set[str], feedback: dict[str, Any] | None) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        policy = ctx.policy
        candidates: list[tuple[dict, dict]] = []
        seen_this_round: set[str] = set()
        bound_count = 0
        duplicates = 0
        invalid = 0
        # Candidates already owned by this (resumed) round are reprocessed, not
        # treated as duplicates of themselves.  Their committed evidence is
        # deterministically rewritten and must never become "duplicate".
        own_round_ids = {e["candidate_id"] for e in
                         self.store.list_evidence(run_id=self.run_id,
                                                  round_id=round_rec.round_id)}
        stored_signatures = self.store.list_probe_signatures(ctx.snapshot.snapshot_hash)
        probe_ok_signatures = {cid: sig for cid, sig in stored_signatures.items()
                               if cid in run_candidate_ids and cid not in own_round_ids}
        screen_fold = self._fold(policy.screen_folds[0], "search")

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
            tpl_id, _created = self.store.add_template(tpl)
            bound = BoundCandidate(
                template_id=tpl_id, template_hash=tpl.content_hash, domain=tpl.domain,
                kind=tpl.kind, name=tpl.name, descriptor=tpl.descriptor,
                dataset_id=ctx.snapshot.dataset_id, snapshot_hash=ctx.snapshot.snapshot_hash,
            ).ensure_id()
            cand_id, _ = self.store.add_candidate(bound)
            if (cand_id in run_candidate_ids and cand_id not in own_round_ids) \
                    or cand_id in seen_this_round:
                duplicates += 1
                prior = self.store.get_evidence(cand_id)
                self._record_early_evidence(
                    round_rec, tpl, EvidenceStatus.DUPLICATE, candidate_id=cand_id,
                    reason="semantically identical to a candidate already used in this run"
                    + (f"; prior state={prior[-1]['state']}" if prior else ""),
                )
                continue
            seen_this_round.add(cand_id)
            bound_count += 1
            cand_row = self.store.get_candidate(cand_id)
            probe = self._probe(tpl.descriptor, screen_fold, str(self.artifacts / "probe" / cand_id))
            if not probe.get("ok", False):
                sev = probe.get("severity", "invalid")
                status = EvidenceStatus.DUPLICATE if sev == "duplicate" else EvidenceStatus.INVALID_STRUCTURE
                self._record_early_evidence(round_rec, tpl, status, candidate_id=cand_id,
                                            reason=probe.get("reason", "probe failed"))
                continue
            sig = probe.get("signature") or []
            dup_of = None
            threshold = float(self.config.get("search", {}).get("behavioural_dedup_corr", 0.995))
            for other_id, other_sig in probe_ok_signatures.items():
                if self._signature_corr(sig, other_sig) >= threshold:
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
        seed = self._seeds()[0]
        baseline_screen = self._ensure_variant(
            ctx.protocol.baseline_variant, "baseline", screen_fold, "screen", seed
        )
        if baseline_screen["status"] != "success":
            raise RuntimeError(
                f"baseline failed on screen fold {screen_fold.fold_id}: "
                f"{baseline_screen.get('error')}"
            )
        base_screen_series = baseline_screen.get("metric_series", {}).get(
            ctx.protocol.primary_metric, {}
        )
        if not base_screen_series:
            raise RuntimeError(f"baseline screen series is empty on {screen_fold.fold_id}")

        if self._max_workers() > 1 and len(candidates) > 1:
            screen_reqs = [
                self._trial_config(
                    c_row["candidate_id"], "candidate", c_row["descriptor"], screen_fold, "screen", seed
                )
                for c_row, _ in candidates
            ]
            self.run_trials_batch(screen_reqs)

        screen_results = []
        for cand_row, _probe_res in candidates:
            req = self._trial_config(cand_row["candidate_id"], "candidate",
                                     cand_row["descriptor"], screen_fold, "screen", seed)
            res = self.run_trial(req)
            if res["status"] != "success":
                ev = self._make_evidence(
                    cand_row, round_rec, EvidenceStatus.EXECUTION_FAILED,
                    reason=(res.get("error") or "screen trial failed")[:1000],
                )
                self.store.record_evidence(ev, run_id=self.run_id)
                continue
            cs = res.get("metric_series", {}).get(ctx.protocol.primary_metric, {})
            if not cs:
                ev = self._make_evidence(
                    cand_row, round_rec, EvidenceStatus.EXECUTION_FAILED,
                    reason="screen trial produced no primary metric series",
                )
                self.store.record_evidence(ev, run_id=self.run_id)
                continue
            eff = paired_effect(cs, base_screen_series, nw_lag=policy.nw_lag,
                                n_boot=min(200, policy.bootstrap_samples),
                                block=policy.bootstrap_block, seed=seed)
            if not _finite(eff["effect"]):
                ev = self._make_evidence(
                    cand_row, round_rec, EvidenceStatus.INSUFFICIENT_EVIDENCE,
                    reason="screen effect is non-finite; no promotion",
                )
                self.store.record_evidence(ev, run_id=self.run_id)
                continue
            screen_results.append({
                "candidate": cand_row, "trial": res, "screen_effect": eff["effect"],
                "screen_p": eff["p_value"] if _finite(eff["p_value"]) else None,
                "screen_metric": res.get("metrics", {}).get(ctx.protocol.primary_metric),
            })
        direction = 1.0 if ctx.protocol.direction == "higher" else -1.0

        def _screen_score(row):
            v = row["screen_effect"]
            return direction * v if math.isfinite(v) else -math.inf

        screen_results.sort(key=lambda x: -_screen_score(x))
        promoted = [r for r in screen_results
                    if math.isfinite(r["screen_effect"])
                    and direction * r["screen_effect"] > policy.min_screen_effect
                    ][: policy.full_validation_top_k]
        promoted_ids = {r["candidate"]["candidate_id"] for r in promoted}

        for r in screen_results:
            if r["candidate"]["candidate_id"] in promoted_ids:
                continue
            ev = self._make_evidence(
                r["candidate"], round_rec, EvidenceStatus.SCREENED_OUT,
                effect=r["screen_effect"], p_value=r["screen_p"],
                metrics={"screen": {
                    "fold": screen_fold.fold_id, "effect": r["screen_effect"],
                    "candidate_metric": r["screen_metric"],
                    "baseline_metric": baseline_screen.get("metrics", {}).get(ctx.protocol.primary_metric),
                }},
                reason="not promoted: screen effect ≤ min_screen_effect or outside top-k",
            )
            self.store.record_evidence(ev, run_id=self.run_id)

        # ------------------------------------------------ full validation (multi-seed)
        validation_folds = [self._fold(fid, "search") for fid in policy.validation_folds]
        seeds = self._seeds()
        evidence_rows = []
        for r in promoted:
            cand = r["candidate"]
            per_seed, failures, n_expected = self._paired_folds_descriptor(
                cand, validation_folds, seeds, phase="full", purpose="search",
            )
            agg = self._aggregate(per_seed, n_expected_folds=n_expected, failures=failures)
            if "error" in agg:
                ev = self._make_evidence(cand, round_rec, EvidenceStatus.EXECUTION_FAILED,
                                         reason=agg["error"][:1500])
                ev.metrics["full_validation"] = True
                self.store.record_evidence(ev, run_id=self.run_id)
                evidence_rows.append(ev)
                continue
            status = self._classify(agg, policy)
            ev = self._make_evidence(
                cand, round_rec, status,
                effect=agg["effect"], ci_low=agg["ci_low"], ci_high=agg["ci_high"],
                p_value=agg["p_value"], n_folds=agg["n_folds"], stability=agg["stability"],
                metrics={k: v for k, v in agg.items()
                         if k not in {"effect", "ci_low", "ci_high", "p_value"}},
                reason=self._reason_for(status, agg, policy),
            )
            ev.metrics["n_validation_dates"] = agg["n_dates"]
            ev.metrics["full_validation"] = True
            evidence_rows.append(ev)

        # R07: Benjamini-Hochberg across the cumulative discovery validation
        # family of the entire run (all rounds evaluated on validation_folds so
        # far), rather than resetting the hypothesis family counter each round.
        prior_full_rows = [
            r for r in self.store.list_evidence(run_id=self.run_id, stage="discovery")
            if r["round_id"] != round_rec.round_id
            and (
                (r.get("metrics") or {}).get("full_validation")
                or (
                    r["status"] not in {"screened_out", "duplicate", "invalid_structure"}
                    and "screen" not in (r.get("metrics") or {})
                )
            )
        ]
        all_pvals = [
            *(r["p_value"] if r.get("p_value") is not None else float("nan") for r in prior_full_rows),
            *(e.p_value if e.p_value is not None else float("nan") for e in evidence_rows),
        ]
        all_qvals = benjamini_hochberg(all_pvals)
        prior_qvals = all_qvals[: len(prior_full_rows)]
        qvals = all_qvals[len(prior_full_rows) :]

        all_metrics_list = [
            *(dict(r.get("metrics") or {}) for r in prior_full_rows),
            *(dict(e.metrics or {}) for e in evidence_rows),
        ]
        m_eff, all_rw_pvals = self._compute_family_multiplicity(
            all_metrics_list,
            nw_lag=policy.nw_lag,
            n_boot=min(500, policy.bootstrap_samples),
            block=policy.bootstrap_block,
            seed=policy.seed,
        )
        prior_rw_pvals = all_rw_pvals[: len(prior_full_rows)]
        rw_pvals = all_rw_pvals[len(prior_full_rows) :]

        affected_prior_rounds: set[str] = set()
        for r_dict, q, rw_p in zip(prior_full_rows, prior_qvals, prior_rw_pvals):
            q_clean = q if q is not None and math.isfinite(q) else None
            if q_clean is None:
                continue
            prior_metrics = dict(r_dict.get("metrics") or {})
            prior_metrics["effective_num_tests"] = m_eff
            prior_metrics["rw_p_value"] = rw_p
            prior_ev = EvidenceRecord(
                evidence_id=r_dict["evidence_id"],
                candidate_id=r_dict["candidate_id"],
                round_id=r_dict["round_id"],
                protocol_hash=r_dict["protocol_hash"],
                stage=r_dict["stage"],
                state=CandidateState(r_dict["state"]),
                seed=r_dict["seed"],
                status=EvidenceStatus(r_dict["status"]),
                effect=r_dict["effect"],
                ci_low=r_dict["ci_low"],
                ci_high=r_dict["ci_high"],
                p_value=r_dict["p_value"],
                q_value=q_clean,
                n_folds=r_dict["n_folds"],
                stability=r_dict["stability"],
                metrics=prior_metrics,
                reason=r_dict["reason"] or "",
            )
            eff_prior = direction * (prior_ev.effect or 0.0)
            changed = (
                r_dict.get("q_value") != q_clean
                or (r_dict.get("metrics") or {}).get("rw_p_value") != rw_p
                or (r_dict.get("metrics") or {}).get("effective_num_tests") != m_eff
            )
            if prior_ev.status == EvidenceStatus.VALIDATED and q_clean > policy.fdr_alpha:
                prior_ev.status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                prior_ev.state = CandidateState.INSUFFICIENT
                if "fails BH FDR" not in prior_ev.reason:
                    prior_ev.reason = (
                        prior_ev.reason + f"; fails run-wide BH FDR (q={q_clean:.3g} > {policy.fdr_alpha})"
                    ).strip("; ")
                changed = True
            elif (
                prior_ev.status == EvidenceStatus.INSUFFICIENT_EVIDENCE
                and q_clean <= policy.fdr_alpha
                and eff_prior < 0
            ):
                prior_ev.status = EvidenceStatus.REJECTED
                prior_ev.state = CandidateState.REJECTED
                if "BH-significant negative effect" not in prior_ev.reason:
                    prior_ev.reason = (prior_ev.reason + "; BH-significant negative effect").strip("; ")
                changed = True
            if changed:
                self.store.record_evidence(prior_ev, run_id=self.run_id)
                affected_prior_rounds.add(prior_ev.round_id)

        for pr_id in affected_prior_rounds:
            pr_ev = [
                r for r in self.store.list_evidence(run_id=self.run_id, round_id=pr_id, stage="discovery")
                if (r.get("metrics") or {}).get("full_validation")
                or (
                    r["status"] not in {"screened_out", "duplicate", "invalid_structure"}
                    and "screen" not in (r.get("metrics") or {})
                )
            ]
            self.store.update_round(
                pr_id,
                n_validated=sum(1 for r in pr_ev if r["status"] == EvidenceStatus.VALIDATED.value),
                n_rejected=sum(1 for r in pr_ev if r["status"] == EvidenceStatus.REJECTED.value),
                n_insufficient=sum(
                    1 for r in pr_ev if r["status"] == EvidenceStatus.INSUFFICIENT_EVIDENCE.value
                ),
            )

        for ev, q, rw_p in zip(evidence_rows, qvals, rw_pvals):
            ev.q_value = q if q is not None and math.isfinite(q) else None
            ev.metrics["effective_num_tests"] = m_eff
            ev.metrics["rw_p_value"] = rw_p
            if ev.q_value is None:
                continue
            effect = direction * (ev.effect or 0.0)
            if ev.status == EvidenceStatus.VALIDATED and ev.q_value > policy.fdr_alpha:
                ev.status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                ev.state = CandidateState.INSUFFICIENT
                ev.reason = (ev.reason + f"; fails BH FDR (q={ev.q_value:.3g} > {policy.fdr_alpha})").strip("; ")
            elif ev.status == EvidenceStatus.INSUFFICIENT_EVIDENCE and ev.q_value <= policy.fdr_alpha and effect < 0:
                ev.status = EvidenceStatus.REJECTED
                ev.state = CandidateState.REJECTED
                ev.reason = (ev.reason + "; BH-significant negative effect").strip("; ")
            self.store.record_evidence(ev, run_id=self.run_id)

        summary = self._round_summary(round_rec, screen_results, evidence_rows)
        summary["effective_num_tests"] = m_eff
        summary["rw_p_values"] = {e.candidate_id: e.metrics.get("rw_p_value") for e in evidence_rows}
        summary["proposals"] = [t.model_dump(mode="json") for t in templates]
        summary["identity_hash"] = ctx.context_hash
        summary["label_isolation"] = ctx.label_report
        controls = list(dict.fromkeys(
            [v for v in [ctx.protocol.control_variant, *ctx.protocol.control_variants] if v]
        ))
        if controls:
            summary["control"] = self._control_summary(
                screen_fold, validation_folds, seed, controls
            )
        # Status is committed only after evidence + summary + feedback.
        self.store.update_round(
            round_rec.round_id, n_proposed=round_rec.n_proposed, n_bound=bound_count,
            n_validated=sum(1 for e in evidence_rows if e.status == EvidenceStatus.VALIDATED),
            n_rejected=sum(1 for e in evidence_rows if e.status == EvidenceStatus.REJECTED),
            n_insufficient=sum(1 for e in evidence_rows if e.status == EvidenceStatus.INSUFFICIENT_EVIDENCE),
            summary=summary,
        )
        feedback_payload = self._feedback_payload(round_rec, screen_results, evidence_rows)
        self.store.add_feedback(round_rec.round_id, feedback_payload, run_id=self.run_id)
        self.store.update_round(
            round_rec.round_id, status="completed",
            ended_at=utcnow().isoformat(),
        )
        print(f"[round {round_rec.round_index}] screen={len(screen_results)} "
              f"promoted={len(promoted)} validated={summary['n_validated']} "
              f"rejected={summary['n_rejected']} insufficient={summary['n_insufficient']} "
              f"duplicates={duplicates} invalid={invalid}")
        return feedback_payload

    def _control_summary(self, screen_fold: FoldSpec, validation_folds: list[FoldSpec],
                         seed: int, variants: list[str]) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        out: dict[str, Any] = {}
        for variant in variants:
            per_variant: dict[str, Any] = {}
            for phase, folds in (("screen", [screen_fold]), ("full", validation_folds)):
                rows = {}
                for fold in folds:
                    res = self._ensure_variant(
                        variant, "control", fold, phase, seed, control_kind=variant,
                    )
                    rows[fold.fold_id] = {
                        "status": res.get("status"),
                        "config_hash": res.get("config_hash"),
                        "metrics": {k: res.get("metrics", {}).get(k)
                                    for k in (ctx.protocol.primary_metric, "ic", "rank_ic",
                                              "icir", "mse", "mae")},
                        "error": (res.get("error") or "")[:500] or None,
                    }
                per_variant[phase] = rows
            out[variant] = per_variant
        return out

    # ---------------------------------------------------------- round helpers
    def _fold(self, fold_id: str, purpose: str) -> FoldSpec:
        ctx = self._ctx
        assert ctx is not None
        pools = {
            "search": ctx.effective_folds,
            "confirmation": ctx.effective_confirmation,
        }
        if purpose == "finalize" and ctx.effective_holdout is not None:
            if ctx.effective_holdout.fold_id == fold_id:
                ctx.guard.assert_fold_accessible(fold_id, "finalize")
                return ctx.effective_holdout
        for fold in pools.get(purpose, []):
            if fold.fold_id == fold_id:
                ctx.guard.assert_fold_accessible(fold.fold_id, purpose)
                return fold
        raise ProtocolViolation(f"unknown {purpose} fold {fold_id}")

    def _paired_folds_descriptor(self, cand: dict, folds: list[FoldSpec], seeds: list[int],
                                 *, phase: str, purpose: str):
        ctx = self._ctx
        assert ctx is not None
        metric = ctx.protocol.primary_metric
        ctrl_variant = str(ctx.protocol.control_variant or "").strip()
        per_seed: dict[int, dict[str, Any]] = {}
        failures: list[str] = []
        n_expected = len(folds)
        if self._max_workers() > 1:
            batch_reqs: list[TrialRequest] = []
            for seed in seeds:
                for fold in folds:
                    ctx.guard.assert_fold_accessible(fold.fold_id, purpose)
                    batch_reqs.append(
                        self._trial_config(
                            cand["candidate_id"], "candidate", cand["descriptor"], fold, phase, seed
                        )
                    )
                    batch_reqs.append(
                        self._trial_config(
                            ctx.protocol.baseline_variant, "baseline", {}, fold, phase, seed
                        )
                    )
                    if ctrl_variant:
                        batch_reqs.append(
                            self._trial_config(
                                ctrl_variant, "control", {}, fold, phase, seed, control_kind=ctrl_variant
                            )
                        )
            self.run_trials_batch(batch_reqs)

        for seed in seeds:
            cand_series: dict[str, dict] = {}
            base_series: dict[str, dict] = {}
            ctrl_series: dict[str, dict] = {}
            fold_metrics: dict[str, dict] = {}
            for fold in folds:
                ctx.guard.assert_fold_accessible(fold.fold_id, purpose)
                cand_req = self._trial_config(
                    cand["candidate_id"], "candidate", cand["descriptor"], fold, phase, seed
                )
                cand_res = self.run_trial(cand_req)
                if cand_res["status"] != "success":
                    failures.append(f"seed {seed} {fold.fold_id}: "
                                    f"{(cand_res.get('error') or '')[:300]}")
                    continue
                base_res = self._ensure_variant(
                    ctx.protocol.baseline_variant, "baseline", fold, phase, seed
                )
                if base_res["status"] != "success":
                    failures.append(f"seed {seed} baseline {fold.fold_id}: "
                                    f"{(base_res.get('error') or '')[:300]}")
                    continue
                cs = cand_res.get("metric_series", {}).get(metric, {})
                bs = base_res.get("metric_series", {}).get(metric, {})
                if not cs or not bs:
                    failures.append(f"seed {seed} {fold.fold_id}: empty metric series")
                    continue
                cand_series[fold.fold_id] = cs
                base_series[fold.fold_id] = bs
                eff = paired_effect(cs, bs, nw_lag=ctx.policy.nw_lag, n_boot=200,
                                    block=ctx.policy.bootstrap_block, seed=seed)
                fm_entry: dict[str, Any] = {
                    "candidate": {k: cand_res["metrics"].get(k) for k in
                                  (metric, "ic", "rank_ic", "icir", "mse", "mae", "n_dates")},
                    "baseline": {k: base_res["metrics"].get(k) for k in
                                 (metric, "ic", "rank_ic", "icir", "mse", "mae", "n_dates")},
                    "effect": eff["effect"], "p_value": eff["p_value"],
                    "candidate_config_hash": cand_res.get("config_hash"),
                    "baseline_config_hash": base_res.get("config_hash"),
                }
                if ctrl_variant:
                    ctrl_res = self._ensure_variant(
                        ctrl_variant, "control", fold, phase, seed, control_kind=ctrl_variant
                    )
                    if ctrl_res["status"] == "success":
                        ctrl_s = ctrl_res.get("metric_series", {}).get(metric, {})
                        if ctrl_s:
                            ctrl_series[fold.fold_id] = ctrl_s
                            ctrl_eff = paired_effect(
                                cs, ctrl_s, nw_lag=ctx.policy.nw_lag, n_boot=200,
                                block=ctx.policy.bootstrap_block, seed=seed
                            )
                            fm_entry["control_variant"] = ctrl_variant
                            fm_entry["control"] = {
                                k: ctrl_res["metrics"].get(k) for k in
                                (metric, "ic", "rank_ic", "icir", "mse", "mae", "n_dates")
                            }
                            fm_entry["control_effect"] = ctrl_eff["effect"]
                            fm_entry["control_p_value"] = ctrl_eff["p_value"]
                            fm_entry["control_config_hash"] = ctrl_res.get("config_hash")
                fold_metrics[fold.fold_id] = fm_entry
            per_seed[seed] = self._seed_aggregate(
                seed, cand_series, base_series, fold_metrics,
                ctrl_series=ctrl_series if ctrl_series else None,
                ctrl_variant=ctrl_variant,
            )
        return per_seed, failures, n_expected

    def _classify(self, agg: dict[str, Any], policy: SearchPolicy) -> EvidenceStatus:
        coverage = float(agg.get("fold_coverage") or 0.0)
        if coverage < policy.min_fold_coverage:
            return EvidenceStatus.INSUFFICIENT_EVIDENCE
        if int(agg.get("n_folds") or 0) < int(
            self.config.get("search", {}).get("min_folds_for_validation", 1)
        ):
            return EvidenceStatus.INSUFFICIENT_EVIDENCE
        if float(agg.get("seed_agreement") or 0.0) < policy.min_seed_agreement:
            return EvidenceStatus.INSUFFICIENT_EVIDENCE
        if not agg.get("inferential", True) or agg.get("p_value") is None:
            return EvidenceStatus.INSUFFICIENT_EVIDENCE
        direction = 1 if self._ctx.protocol.direction == "higher" else -1
        effect = direction * agg["effect"]
        ci_ok = True
        explicit_pos_ci = "require_positive_ci" in (self.config.get("search") or {})
        if policy.require_positive_ci and (policy.min_effect >= 0.0 or explicit_pos_ci):
            if direction > 0:
                ci_low = agg.get("ci_low")
                ci_ok = _finite(ci_low) and float(ci_low) > 0.0
            else:
                ci_high = agg.get("ci_high")
                ci_ok = _finite(ci_high) and float(ci_high) < 0.0
        ctrl_ok = True
        ctrl_variant = agg.get("control_variant") or (
            self._ctx.protocol.control_variant if self._ctx else ""
        )
        if ctrl_variant and policy.require_beat_control and (
            "control_effect" in agg or agg.get("control_variant")
        ):
            ctrl_eff_raw = agg.get("control_effect")
            if not _finite(ctrl_eff_raw):
                ctrl_ok = False
            else:
                ctrl_impr = direction * float(ctrl_eff_raw)
                min_delta = float(policy.min_control_delta)
                ctrl_ok = (ctrl_impr > 0.0) if min_delta == 0.0 else (ctrl_impr >= min_delta)
        if (
            effect >= policy.min_effect
            and agg["stability"] >= policy.min_stability
            and agg["p_value"] <= policy.fdr_alpha
            and ci_ok
            and ctrl_ok
        ):
            return EvidenceStatus.VALIDATED
        if effect < 0.0 and effect <= -abs(policy.min_effect) and agg["p_value"] <= policy.fdr_alpha:
            return EvidenceStatus.REJECTED
        return EvidenceStatus.INSUFFICIENT_EVIDENCE

    def _reason_for(self, status: EvidenceStatus, agg: dict, policy: SearchPolicy) -> str:
        direction = 1 if (self._ctx and self._ctx.protocol.direction == "higher") else -1
        p = agg.get("p_value")
        p_s = f"{p:.4g}" if _finite(p) else "not estimable"
        lo, hi = agg.get("ci_low"), agg.get("ci_high")
        ci_s = (f"({lo:+.4f},{hi:+.4f})" if _finite(lo) and _finite(hi)
                else "not estimable")
        if status == EvidenceStatus.VALIDATED:
            msg = (f"effect {agg['effect']:+.4f} (CI {ci_s}), p={p_s}, "
                   f"stability={agg['stability']:.2f} ≥ {policy.min_stability}, "
                   f"coverage={agg['fold_coverage']:.2f}, "
                   f"seeds={agg.get('n_seeds')} agreement={agg.get('seed_agreement'):.2f}")
            if _finite(agg.get("control_effect")):
                cv = agg.get("control_variant") or "control:ones"
                msg += f", vs {cv}={agg['control_effect']:+.4f}"
            return msg
        if status == EvidenceStatus.REJECTED:
            return f"significant negative effect {agg['effect']:+.4f}, p={p_s}"
        extra_reasons: list[str] = []
        if policy.require_positive_ci and _finite(lo) and _finite(hi):
            if (direction > 0 and float(lo) <= 0.0) or (direction < 0 and float(hi) >= 0.0):
                extra_reasons.append(f"paired CI {ci_s} crosses zero")
        ctrl_variant = agg.get("control_variant") or (
            self._ctx.protocol.control_variant if self._ctx else ""
        )
        if ctrl_variant and policy.require_beat_control and (
            "control_effect" in agg or agg.get("control_variant")
        ):
            ctrl_eff_raw = agg.get("control_effect")
            if not _finite(ctrl_eff_raw):
                extra_reasons.append(f"fails to beat {ctrl_variant} (control_delta not estimable)")
            else:
                ctrl_impr = direction * float(ctrl_eff_raw)
                min_delta = float(policy.min_control_delta)
                ctrl_ok = (ctrl_impr > 0.0) if min_delta == 0.0 else (ctrl_impr >= min_delta)
                if not ctrl_ok:
                    extra_reasons.append(
                        f"fails to beat {ctrl_variant} (control_delta={float(ctrl_eff_raw):+.4f}, "
                        f"min_control_delta={min_delta})"
                    )
        base_msg = (
            f"inconclusive: effect {agg['effect']:+.4f} (CI {ci_s}), p={p_s}, "
            f"stability={agg['stability']:.2f}, coverage={agg.get('fold_coverage')}, "
            f"seed_agreement={agg.get('seed_agreement')}, thresholds effect≥"
            f"{policy.min_effect} & stability≥{policy.min_stability}"
        )
        if extra_reasons:
            base_msg += "; " + "; ".join(extra_reasons)
        return base_msg

    def _make_evidence(self, cand: dict, round_rec: RoundRecord, status: EvidenceStatus,
                       **kw) -> EvidenceRecord:
        ctx = self._ctx
        assert ctx is not None
        stage = str(kw.pop("stage", "discovery"))
        seed = kw.pop("seed", None)
        state = kw.pop("state", _STATE_FOR_STATUS.get(status, CandidateState.INSUFFICIENT))
        return EvidenceRecord(
            candidate_id=cand["candidate_id"], round_id=round_rec.round_id,
            protocol_hash=ctx.protocol_hash, status=status, stage=stage, state=state,
            seed=seed, **{k: v for k, v in kw.items() if k in EvidenceRecord.model_fields},
        ).ensure_id()

    def _record_early_evidence(self, round_rec, tpl: PredicateTemplate,
                               status: EvidenceStatus, candidate_id: str = "",
                               reason: str = "") -> None:
        cid = candidate_id or short_id(
            "unbound", sha256_obj([tpl.domain, tpl.kind.value, tpl.descriptor])
        )
        ev = EvidenceRecord(
            candidate_id=cid, round_id=round_rec.round_id,
            protocol_hash=self._ctx.protocol_hash, status=status, stage="discovery",
            state=_STATE_FOR_STATUS.get(status, CandidateState.INSUFFICIENT),
            reason=f"{tpl.name}: {reason}",
            metrics={"template": {"name": tpl.name, "kind": tpl.kind.value,
                                  "descriptor": tpl.descriptor}},
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
            "status": "completed",
            "n_screened": len(screen_results),
            "n_validated": sum(1 for e in evidence_rows if e.status == EvidenceStatus.VALIDATED),
            "n_rejected": sum(1 for e in evidence_rows if e.status == EvidenceStatus.REJECTED),
            "n_insufficient": sum(1 for e in evidence_rows if e.status == EvidenceStatus.INSUFFICIENT_EVIDENCE),
            "screen_table": [row(r) for r in screen_results],
            "validation": [
                {"candidate_id": e.candidate_id, "status": e.status.value,
                 "state": e.state.value, "stage": e.stage,
                 "effect": e.effect, "ci": [e.ci_low, e.ci_high], "p": e.p_value,
                 "q": e.q_value, "stability": e.stability, "n_folds": e.n_folds,
                 "seed_effects": e.metrics.get("seed_effects"),
                 "seed_effect_sd": e.metrics.get("seed_effect_sd"),
                 "fold_effect_sd": e.metrics.get("fold_effect_sd"),
                 "fold_coverage": e.metrics.get("fold_coverage"),
                 "reason": e.reason}
                for e in evidence_rows
            ],
        }

    def _feedback_payload(self, round_rec, screen_results, evidence_rows) -> dict:
        cand_by_id = {r["candidate"]["candidate_id"]: r["candidate"] for r in screen_results}
        screen_by_id = {r["candidate"]["candidate_id"]: r.get("screen_effect") for r in screen_results}
        validated, rejected, insufficient = [], [], []
        for e in evidence_rows:
            base = {
                "candidate_id": e.candidate_id,
                "screen_effect": screen_by_id.get(e.candidate_id),
                "effect": e.effect,
                "q_value": e.q_value, "p_value": e.p_value,
                "status": e.status.value, "state": e.state.value, "stage": e.stage,
                "descriptor": cand_by_id.get(e.candidate_id, {}).get("descriptor"),
                "name": cand_by_id.get(e.candidate_id, {}).get("name"),
                "family": (cand_by_id.get(e.candidate_id, {}).get("descriptor") or {}).get("family"),
            }
            (validated if e.status == EvidenceStatus.VALIDATED else
             rejected if e.status == EvidenceStatus.REJECTED else insufficient).append(base)
        screen_only = [
            {"candidate_id": r["candidate"]["candidate_id"],
             "descriptor": r["candidate"]["descriptor"],
             "family": (r["candidate"]["descriptor"] or {}).get("family"),
             "screen_effect": r["screen_effect"], "status": "screened_out"}
            for r in screen_results
            if r["candidate"]["candidate_id"] not in {e.candidate_id for e in evidence_rows}
        ]
        return {
            "round_id": round_rec.round_id, "round_index": round_rec.round_index,
            "stage": "discovery",
            "validated": validated, "rejected": rejected,
            "insufficient": insufficient, "screened_out": screen_only,
            "known_effects": {e.candidate_id: e.effect for e in evidence_rows
                              if e.effect is not None},
        }

    # ---------------------------------------------------------- confirmation
    def freeze_family(self, candidate_ids: list[str] | None = None,
                      top_k: int | None = None) -> dict[str, Any]:
        """Freeze the discovery family before any confirmation/holdout access."""
        ctx = self._build_context()
        run = self.store.get_run(self.run_id)
        if not run or run.get("status") not in {
            "search_complete", "confirmation_complete", "final_complete"
        }:
            raise ProtocolViolation(
                f"run {self.run_id} is {run.get('status') if run else 'missing'}; "
                "only a completed discovery run can be frozen"
            )
        existing_family = self.store.get_frozen_family(self.run_id)
        if existing_family:
            # freeze-family happens before any confirmation result is visible.
            # A second call (e.g. after seeing confirmation) must never replace
            # the family; `confirm` reads the frozen row instead.
            raise ProtocolViolation(
                f"run {self.run_id} already has frozen family "
                f"{existing_family.get('family_id', '')[:16]} "
                f"(manifest {existing_family.get('manifest_hash', '')[:16]}); "
                "a frozen family is immutable"
            )
        rounds = self.store.list_rounds(self.run_id)
        if any(r["status"] != "completed" for r in rounds):
            raise ProtocolViolation("cannot freeze: a discovery round is not completed")
        latest: dict[str, dict] = {}
        for e in self.store.list_evidence(run_id=self.run_id, stage="discovery"):
            latest[e["candidate_id"]] = e
        positive = [e for e in latest.values() if e["status"] == EvidenceStatus.VALIDATED.value]
        if candidate_ids:
            selected = []
            for cid in candidate_ids:
                ev = latest.get(cid)
                if ev is None:
                    raise ProtocolViolation(f"candidate {cid} has no discovery evidence in this run")
                if ev["status"] != EvidenceStatus.VALIDATED.value:
                    raise ProtocolViolation(
                        f"candidate {cid} is {ev['status']} in discovery; only exploratory_positive "
                        "candidates may enter confirmation"
                    )
                selected.append(ev)
        else:
            # Sort by improvement, not raw effect: for direction='lower' a
            # negative effect is better.  Sorting on raw effect picked the
            # *worst* candidate when top_k was set.
            positive.sort(key=lambda e: -_selection_score(
                ctx.protocol.direction, e.get("effect")))
            selected = positive[: int(top_k)] if top_k else positive
        if not selected:
            raise ProtocolViolation("no discovery-positive candidate to freeze")
        ids = [e["candidate_id"] for e in selected]
        if len(set(ids)) != len(ids):
            raise ProtocolViolation("duplicate candidate in frozen family")
        family_id = sha256_obj({"run": self.run_id, "candidates": ids, "protocol": ctx.protocol_hash})
        members = []
        for e in selected:
            cand = self.store.get_candidate(e["candidate_id"])
            if not cand:
                raise ProtocolViolation(f"candidate row missing for {e['candidate_id']}")
            members.append({
                "candidate_id": e["candidate_id"], "template_hash": cand["template_hash"],
                "name": cand["name"], "descriptor": cand["descriptor"],
                "discovery_evidence_id": e["evidence_id"], "effect": e["effect"],
                "q_value": e["q_value"], "p_value": e["p_value"],
                "state": e["state"],
            })
        manifest = {
            "version": "1.1",
            "family_id": family_id,
            "task_id": ctx.task.task_id,
            "run_id": self.run_id,
            "protocol_hash": ctx.protocol_hash,
            "snapshot_hash": ctx.snapshot.manifest_hash,
            "data_manifest_hash": ctx.data_manifest_hash,
            "label_spec_hash": ctx.label_spec,
            "label_expression": ctx.label_expression,
            "label_offset_trading_days": ctx.label_offset,
            "code": ctx.code,
            "environment": ctx.environment,
            "model_spec": ctx.protocol.model_spec,
            "seeds": self._seeds(),
            "discovery_folds": [f.model_dump(mode="json") for f in ctx.effective_folds],
            "confirmation_folds": [f.model_dump(mode="json") for f in ctx.effective_confirmation],
            "search_policy": ctx.policy.model_dump(mode="json"),
            "policy_hash": sha256_obj(ctx.policy.model_dump(mode="json")),
            "selection_rule": {
                "mode": "explicit_list" if candidate_ids else ("top_k" if top_k else "all_validated"),
                "candidate_ids": list(candidate_ids) if candidate_ids else None,
                "top_k": int(top_k) if top_k else None,
                "score": "direction * effect (higher is better after direction)",
                "eligible_state": EvidenceStatus.VALIDATED.value,
            },
            "members": members,
            "multiple_testing": {
                "family": ids, "method": "benjamini-hochberg",
                "alpha": ctx.policy.fdr_alpha,
                "assumptions": "frozen, disjoint, independently screened candidate family",
            },
            "frozen_at": utcnow().isoformat(),
        }
        manifest_hash = sha256_obj(manifest)
        manifest["manifest_hash"] = manifest_hash
        self.store.save_frozen_family(self.run_id, manifest)
        atomic_write_text(self.artifacts / "frozen_family.json", strict_json_dumps(manifest))
        ctx.frozen_family = manifest
        print(f"[freeze] family {family_id[:16]} with {len(members)} candidate(s)")
        return manifest

    def _compute_family_multiplicity(
        self,
        metrics_list: list[dict[str, Any]],
        *,
        nw_lag: int,
        n_boot: int,
        block: int,
        seed: int,
    ) -> tuple[float, list[float | None]]:
        if not metrics_list:
            return 0.0, []
        valid_indices: list[int] = []
        series_list: list[dict[str, float]] = []
        for idx, m in enumerate(metrics_list):
            ds = (m or {}).get("delta_series")
            if isinstance(ds, dict) and len(ds) >= 2:
                valid_indices.append(idx)
                series_list.append({str(k): float(v) for k, v in ds.items() if _finite(v)})
        out_rw: list[float | None] = [None] * len(metrics_list)
        if not series_list:
            return float(len(metrics_list)), out_rw
        all_dates = sorted({d for s in series_list for d in s.keys()})
        if len(all_dates) < 2:
            return float(len(series_list)), out_rw
        mat = np.full((len(all_dates), len(series_list)), np.nan, dtype=float)
        for j, s in enumerate(series_list):
            for i, d in enumerate(all_dates):
                if d in s:
                    mat[i, j] = s[d]
        m_eff = effective_num_tests(mat)
        rw_raw = romano_wolf_stepdown(
            mat, n_boot=max(50, int(n_boot)), block=block, seed=seed, nw_lag=nw_lag
        )
        for j, orig_idx in enumerate(valid_indices):
            val = rw_raw[j] if j < len(rw_raw) else float("nan")
            out_rw[orig_idx] = float(val) if _finite(val) else None
        return float(m_eff), out_rw

    def confirm(self, candidate_ids: list[str] | None = None,
                top_k: int | None = None) -> dict[str, Any]:
        """Independent confirmation on later data; never feeds the generator."""
        ctx = self._build_context()
        if not ctx.effective_confirmation:
            raise ProtocolViolation(
                "protocol has no confirmation folds; declare disjoint later data or "
                "explicitly run an exploratory final"
            )
        latest_code = code_version(self._repo_root(), extra_roots=self._external_code_roots())
        ctx.code = latest_code
        run_row = self.store.get_run(self.run_id) or {}
        if run_row.get("code_fingerprint") and latest_code.get("dirty_fingerprint") != run_row.get("code_fingerprint"):
            raise ProtocolViolation(
                f"run {self.run_id} code_fingerprint changed since run creation "
                f"({run_row['code_fingerprint'][:16]} -> {latest_code.get('dirty_fingerprint', '')[:16]}); "
                "refusing confirmation"
            )
        frozen = self.store.get_frozen_family(self.run_id)
        if not frozen:
            frozen = self.freeze_family(candidate_ids, top_k)
        frozen_code_fp = (frozen.get("code") or {}).get("dirty_fingerprint")
        if (
            frozen["protocol_hash"] != ctx.protocol_hash
            or frozen["snapshot_hash"] != ctx.snapshot.manifest_hash
            or frozen["data_manifest_hash"] != ctx.data_manifest_hash
            or (frozen_code_fp and frozen_code_fp != latest_code.get("dirty_fingerprint"))
        ):
            raise ProtocolViolation("frozen family identity no longer matches the run's data or code_fingerprint")
        ctx.frozen_family = frozen
        ctx.guard = ctx.guard.unlock(frozen["manifest_hash"])
        confirmation_claims: list[dict[str, Any]] = []
        prior_exposures: list[dict[str, Any]] = []
        for f in ctx.effective_confirmation:
            ctx.guard.assert_fold_accessible(f.fold_id, "confirmation")
            valid_start = str(f.test_start or f.valid_start or "")
            valid_end = str(f.test_end or f.valid_end or "")
            ckey = self.store.confirmation_key(
                dataset_id=ctx.snapshot.dataset_id,
                snapshot_hash=ctx.snapshot.manifest_hash,
                data_manifest_hash=ctx.data_manifest_hash,
                label_spec_hash=ctx.label_spec,
                fold_id=f.fold_id,
                valid_start=valid_start,
                valid_end=valid_end,
            )
            claim = self.store.claim_confirmation(
                key=ckey,
                study_id=ctx.task.task_id,
                run_id=self.run_id,
                family_id=frozen["family_id"],
                manifest_hash=frozen["manifest_hash"],
                protocol_hash=ctx.protocol_hash,
                snapshot_hash=ctx.snapshot.manifest_hash,
                dataset_id=ctx.snapshot.dataset_id,
                label_spec_hash=ctx.label_spec,
                fold_id=f.fold_id,
                valid_start=valid_start,
                valid_end=valid_end,
                code_fingerprint=ctx.code.get("dirty_fingerprint", ""),
                details={"confirmation_kind": ctx.protocol.confirmation_kind},
            )
            confirmation_claims.append(claim)
            if claim.get("reused") and claim.get("prior_exposure"):
                prior_exposures.append(claim["prior_exposure"])
        if prior_exposures and bool(self.run_cfg.get("strict_confirmation_ledger", False)):
            first_prior = prior_exposures[0]
            raise ConfirmationConsumed(
                f"confirmation fold {first_prior['fold_id']} ({first_prior['valid_start']}.."
                f"{first_prior['valid_end']}) already consumed by run={first_prior['run_id']}"
            )
        seeds = self._seeds()
        round_id = "confirm_" + frozen["manifest_hash"][:16]
        results: list[EvidenceRecord] = []
        for member in frozen["members"]:
            cand = self.store.get_candidate(member["candidate_id"])
            if not cand:
                raise ProtocolViolation(f"frozen candidate vanished: {member['candidate_id']}")
            per_seed, failures, n_expected = self._paired_folds_descriptor(
                cand, ctx.effective_confirmation, seeds,
                phase="confirmation", purpose="confirmation",
            )
            agg = self._aggregate(per_seed, n_expected_folds=n_expected, failures=failures)
            if "error" in agg:
                ev = EvidenceRecord(
                    candidate_id=cand["candidate_id"], round_id=round_id,
                    protocol_hash=ctx.protocol_hash, status=EvidenceStatus.EXECUTION_FAILED,
                    stage="confirmation", state=CandidateState.INSUFFICIENT,
                    reason=agg["error"][:1500],
                ).ensure_id()
            else:
                status = self._classify(agg, ctx.policy)
                confirmed_state = (
                    CandidateState.INDEPENDENTLY_CONFIRMED
                    if ctx.protocol.confirmation_kind == "time_forward"
                    else CandidateState.RESAMPLING_CONFIRMED
                )
                state = (confirmed_state
                         if status == EvidenceStatus.VALIDATED else
                         CandidateState.REJECTED if status == EvidenceStatus.REJECTED else
                         CandidateState.INSUFFICIENT)
                ev = EvidenceRecord(
                    candidate_id=cand["candidate_id"], round_id=round_id,
                    protocol_hash=ctx.protocol_hash, status=status,
                    stage="confirmation", state=state,
                    effect=agg["effect"], ci_low=agg["ci_low"], ci_high=agg["ci_high"],
                    p_value=agg["p_value"], n_folds=agg["n_folds"],
                    stability=agg["stability"],
                    metrics={k: v for k, v in agg.items()
                             if k not in {"effect", "ci_low", "ci_high", "p_value"}},
                    reason=self._reason_for(status, agg, ctx.policy),
                ).ensure_id()
            results.append(ev)
        qvals = benjamini_hochberg([e.p_value if e.p_value is not None else float("nan")
                                    for e in results])
        conf_m_eff, conf_rw_pvals = self._compute_family_multiplicity(
            [dict(e.metrics or {}) for e in results],
            nw_lag=ctx.policy.nw_lag,
            n_boot=min(500, ctx.policy.bootstrap_samples),
            block=ctx.policy.bootstrap_block,
            seed=ctx.policy.seed,
        )
        direction = 1 if ctx.protocol.direction == "higher" else -1
        for ev, q, rw_p in zip(results, qvals, conf_rw_pvals):
            ev.q_value = q if q is not None and math.isfinite(q) else None
            ev.metrics["effective_num_tests"] = conf_m_eff
            ev.metrics["rw_p_value"] = rw_p
            if ev.state in {CandidateState.INDEPENDENTLY_CONFIRMED,
                            CandidateState.RESAMPLING_CONFIRMED} and (
                ev.q_value is None or ev.q_value > ctx.policy.fdr_alpha
            ):
                ev.state = CandidateState.INSUFFICIENT
                ev.reason = (ev.reason + "; independent confirmation fails BH FDR").strip("; ")
            if ev.state in {CandidateState.INDEPENDENTLY_CONFIRMED,
                            CandidateState.RESAMPLING_CONFIRMED} and prior_exposures:
                first_prior = prior_exposures[0]
                ev.state = CandidateState.EXPLORATORY_POSITIVE
                ev.reason = (
                    ev.reason
                    + f"; confirmation fold {first_prior['fold_id']} already consumed by "
                      f"run={first_prior['run_id']} (downgraded to exploratory_positive)"
                ).strip("; ")
            if ev.effect is not None and direction * ev.effect < 0 and ev.p_value is not None \
                    and ev.p_value <= ctx.policy.fdr_alpha:
                ev.state = CandidateState.REJECTED
            self.store.record_evidence(ev, run_id=self.run_id)
        for claim in confirmation_claims:
            if not claim.get("reused"):
                self.store.mark_confirmation(
                    claim["access_id"], "completed",
                    run_id=self.run_id, family_id=frozen["family_id"],
                    note="confirmation completed",
                )
        summary = {
            "family_id": frozen["family_id"],
            "manifest_hash": frozen["manifest_hash"],
            "stage": "confirmation",
            "effective_num_tests": conf_m_eff,
            "rw_p_values": {e.candidate_id: e.metrics.get("rw_p_value") for e in results},
            "confirmation_folds": [f.model_dump(mode="json") for f in ctx.effective_confirmation],
            "confirmation_reused": bool(prior_exposures),
            "prior_confirmation_exposures": [
                {
                    "run_id": p.get("run_id"),
                    "family_id": p.get("family_id"),
                    "fold_id": p.get("fold_id"),
                    "valid_start": p.get("valid_start"),
                    "valid_end": p.get("valid_end"),
                    "status": p.get("status"),
                }
                for p in prior_exposures
            ],
            "results": [
                {"candidate_id": e.candidate_id, "status": e.status.value,
                 "state": e.state.value, "effect": e.effect, "ci": [e.ci_low, e.ci_high],
                 "p": e.p_value, "q": e.q_value, "rw_p_value": e.metrics.get("rw_p_value"),
                 "n_folds": e.n_folds, "stability": e.stability, "reason": e.reason}
                for e in results
            ],
            "note": "confirmation evidence is not visible to the candidate generator",
        }
        atomic_write_text(self.artifacts / "confirmation.json", strict_json_dumps(summary))
        self.store.finish_run(self.run_id, "confirmation_complete", stage="confirmation")
        return summary

    # ---------------------------------------------------------------- finalize
    def _final_candidate_state(self, candidate_id: str) -> tuple[EvidenceRecord | None, bool]:
        """Return (confirmation evidence, independently confirmed)."""
        rows = [e for e in self.store.list_evidence(run_id=self.run_id, stage="confirmation")
                if e["candidate_id"] == candidate_id]
        if not rows:
            return None, False
        ev = rows[-1]
        return ev, ev["state"] in {
            CandidateState.INDEPENDENTLY_CONFIRMED.value,
            CandidateState.RESAMPLING_CONFIRMED.value,
        }

    def _build_final_manifest(self, candidate_id: str, cand: dict,
                              confirmation: dict | None, exploratory: bool) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        assert ctx.effective_holdout is not None
        frozen = self.store.get_frozen_family(self.run_id)
        manifest = {
            "version": "1.1",
            "task_id": ctx.task.task_id,
            "run_id": self.run_id,
            "candidate_id": candidate_id,
            "template_hash": cand["template_hash"],
            "protocol_hash": ctx.protocol_hash,
            "snapshot_hash": ctx.snapshot.manifest_hash,
            "data_manifest_hash": ctx.data_manifest_hash,
            "label_spec_hash": ctx.label_spec,
            "label_expression": ctx.label_expression,
            "label_offset_trading_days": ctx.label_offset,
            "preprocessing_version": ctx.task.data_contract.preprocessing_version,
            "model_spec": ctx.protocol.model_spec,
            "seeds": self._seeds(),
            "training_config": self._training_config("final"),
            "holdout_fold": ctx.effective_holdout.model_dump(mode="json"),
            "discovery_folds": [f.model_dump(mode="json") for f in ctx.effective_folds],
            "confirmation_folds": [f.model_dump(mode="json") for f in ctx.effective_confirmation],
            "code": ctx.code,
            "environment": ctx.environment,
            "family_manifest_hash": frozen["manifest_hash"] if frozen else None,
            "confirmation_evidence_id": confirmation["evidence_id"] if confirmation else None,
            "confirmation_state": confirmation["state"] if confirmation else None,
            "exploratory": bool(exploratory),
            "purpose": "exploratory_reuse" if exploratory else "confirmatory_final",
        }
        manifest["holdout_key"] = self.store.holdout_key(
            dataset_id=ctx.snapshot.dataset_id,
            snapshot_hash=ctx.snapshot.manifest_hash,
            data_manifest_hash=ctx.data_manifest_hash,
            label_spec_hash=ctx.label_spec,
            test_start=str(ctx.effective_holdout.test_start or ""),
            test_end=str(ctx.effective_holdout.test_end or ""),
        )
        # The hash covers scientific identity only.  ``frozen_at`` is run
        # metadata; hashing it made every retry produce a different manifest
        # hash, so a crashed finalize could never resume its own claim.
        manifest_hash = sha256_obj(manifest)
        manifest["frozen_at"] = utcnow().isoformat()
        manifest["manifest_hash"] = manifest_hash
        return manifest

    def finalize(self, candidate_id: str) -> dict[str, Any]:
        """One-time, ledger-guarded holdout evaluation."""
        ctx = self._build_context()
        protocol = ctx.protocol
        holdout = ctx.effective_holdout
        if holdout is None:
            raise ProtocolViolation("protocol has no holdout fold")
        run = self.store.get_run(self.run_id)
        if not run:
            raise ProtocolViolation(f"run {self.run_id} does not exist")
        # Idempotent retry: a completed finalize returns saved results.
        existing = self.store.get_final_eval_for(self.run_id, candidate_id)
        if existing and existing["status"] == "success":
            saved = self.artifacts / "final_evaluation.json"
            if saved.exists():
                data = json.loads(saved.read_text(encoding="utf-8"))
            else:
                data = {"manifest_hash": existing["manifest_hash"],
                        "result": existing["metrics"], "state": existing["state"]}
            data["retry"] = True
            data["already_completed"] = True
            return data
        if any(f["candidate_id"] != candidate_id for f in self.store.list_final_evals(self.run_id)):
            raise ProtocolViolation(
                f"run {self.run_id} already selected a different final candidate; "
                "one final candidate per run"
            )
        allowed_status = {"search_complete", "confirmation_complete", "final_complete"}
        retry_status = {"final_failed", "failed", "interrupted"}
        retryable = (
            existing is not None
            and existing.get("status") in {"frozen", "failed", "interrupted"}
            and run.get("status") in retry_status
        )
        if run.get("status") not in allowed_status and not retryable:
            raise ProtocolViolation(
                f"run {self.run_id} is {run.get('status')!r}; an unfinished run cannot access holdout"
            )
        if any(r["status"] != "completed" for r in self.store.list_rounds(self.run_id)):
            raise ProtocolViolation("a discovery round is not completed; refusing holdout")
        cand = self.store.get_candidate(candidate_id)
        if not cand:
            raise ValueError(f"unknown candidate {candidate_id}")
        if cand["snapshot_hash"] != ctx.snapshot.manifest_hash:
            raise ProtocolViolation("candidate was bound to a different snapshot")
        discovery = [e for e in self.store.list_evidence(run_id=self.run_id, stage="discovery")
                     if e["candidate_id"] == candidate_id]
        if not discovery or discovery[-1]["status"] != EvidenceStatus.VALIDATED.value:
            raise ProtocolViolation(
                f"candidate {candidate_id} is not validated; refusing holdout access "
                f"(status={discovery[-1]['status'] if discovery else 'none'})"
            )
        frozen = self.store.get_frozen_family(self.run_id)
        if not frozen:
            raise ProtocolViolation("candidate family is not frozen; refusing holdout access")
        if candidate_id not in {m["candidate_id"] for m in frozen["members"]}:
            raise ProtocolViolation(
                f"candidate {candidate_id} is not in the frozen family; refusing holdout access"
            )
        latest_code = code_version(self._repo_root(), extra_roots=self._external_code_roots())
        ctx.code = latest_code
        if run.get("code_fingerprint") and latest_code.get("dirty_fingerprint") != run.get("code_fingerprint"):
            raise ProtocolViolation(
                f"run {self.run_id} code_fingerprint changed since run creation "
                f"({run['code_fingerprint'][:16]} -> {latest_code.get('dirty_fingerprint', '')[:16]}); "
                "refusing holdout access"
            )
        frozen_code_fp = (frozen.get("code") or {}).get("dirty_fingerprint")
        if (
            frozen["protocol_hash"] != ctx.protocol_hash
            or frozen["data_manifest_hash"] != ctx.data_manifest_hash
            or (frozen_code_fp and frozen_code_fp != latest_code.get("dirty_fingerprint"))
        ):
            raise ProtocolViolation("frozen family does not match the current snapshot/protocol/code_fingerprint")
        confirmation, confirmed = self._final_candidate_state(candidate_id)
        require_confirmation = bool(self.run_cfg.get("require_confirmation", True))
        if require_confirmation:
            hkey = self.store.holdout_key(
                dataset_id=ctx.snapshot.dataset_id,
                snapshot_hash=ctx.snapshot.manifest_hash,
                data_manifest_hash=ctx.data_manifest_hash,
                label_spec_hash=ctx.label_spec,
                test_start=str(holdout.test_start or ""),
                test_end=str(holdout.test_end or ""),
            )
            self.store.assert_holdout_available(
                key=hkey,
                dataset_id=ctx.snapshot.dataset_id,
                test_start=str(holdout.test_start or ""),
                test_end=str(holdout.test_end or ""),
                run_id=self.run_id,
                candidate_id=candidate_id,
            )
        if not confirmed and require_confirmation:
            raise ProtocolViolation(
                f"candidate {candidate_id} is not independently confirmed "
                f"(state={confirmation['state'] if confirmation else 'none'}); "
                "run confirmation or set run.require_confirmation=false to perform an "
                "*exploratory* final that cannot be reported as confirmatory"
            )
        exploratory = not confirmed

        manifest = self._build_final_manifest(candidate_id, cand, confirmation, exploratory)
        manifest_hash = manifest["manifest_hash"]
        atomic_write_text(self.artifacts / "frozen_manifest.json", strict_json_dumps(manifest))
        if existing and existing["status"] in {"frozen", "failed", "interrupted"}:
            if existing["manifest_hash"] != manifest_hash:
                raise ProtocolViolation(
                    "an earlier frozen manifest differs from the current one; "
                    "the holdout has already been consumed by this run"
                )
        else:
            self.store.add_final_eval(FinalEvaluation(
                task_id=ctx.task.task_id, run_id=self.run_id, candidate_id=candidate_id,
                protocol_hash=ctx.protocol_hash, config_hash=manifest["holdout_key"],
                manifest_hash=manifest_hash, holdout_key=manifest["holdout_key"],
                status="frozen", stage="final",
                state=CandidateState.EXPLORATORY_POSITIVE if exploratory else CandidateState.EXECUTABLE,
                exploratory=exploratory,
                notes="candidate family frozen before holdout access",
            ))
        ctx.guard = ctx.guard.unlock(manifest_hash)
        ctx.guard.assert_fold_accessible(holdout.fold_id, "finalize")

        reuse_used = False
        try:
            claim = self.store.claim_holdout(
                key=manifest["holdout_key"],
                study_id=ctx.task.task_id, run_id=self.run_id,
                candidate_id=candidate_id, protocol_hash=ctx.protocol_hash,
                snapshot_hash=ctx.snapshot.manifest_hash,
                dataset_id=ctx.snapshot.dataset_id,
                label_spec_hash=ctx.label_spec,
                test_start=str(holdout.test_start or ""),
                test_end=str(holdout.test_end or ""),
                purpose=manifest["purpose"], manifest_hash=manifest_hash,
                code_fingerprint=ctx.code.get("dirty_fingerprint", ""),
                details={"label_expression": ctx.label_expression,
                         "label_offset": ctx.label_offset,
                         "label_isolated_test_end": holdout.test_end,
                         "exploratory": exploratory},
            )
        except HoldoutConsumed as exc:
            if not exploratory:
                self.store.add_final_eval(FinalEvaluation(
                    task_id=ctx.task.task_id, run_id=self.run_id, candidate_id=candidate_id,
                    protocol_hash=ctx.protocol_hash, config_hash=manifest["holdout_key"],
                    manifest_hash=manifest_hash, holdout_key=manifest["holdout_key"],
                    status="failed", stage="final", state=CandidateState.INSUFFICIENT,
                    exploratory=exploratory, notes=f"holdout consumed: {exc}",
                ))
                raise
            # Explicit exploratory reuse: allowed only because the caller has
            # already accepted that this is not fresh confirmatory evidence.
            reuse_id = self.store.record_exploratory_reuse(
                key=manifest["holdout_key"], study_id=ctx.task.task_id,
                run_id=self.run_id, candidate_id=candidate_id,
                protocol_hash=ctx.protocol_hash,
                snapshot_hash=ctx.snapshot.manifest_hash,
                dataset_id=ctx.snapshot.dataset_id,
                label_spec_hash=ctx.label_spec,
                test_start=str(holdout.test_start or ""),
                test_end=str(holdout.test_end or ""),
                manifest_hash=manifest_hash,
                code_fingerprint=ctx.code.get("dirty_fingerprint", ""),
                details={"blocked_by": str(exc), "exploratory": True},
            )
            claim = {"access_id": reuse_id, "already_completed": False}
            reuse_used = True
        if claim.get("already_completed"):
            saved = self.artifacts / "final_evaluation.json"
            if saved.exists():
                data = json.loads(saved.read_text(encoding="utf-8"))
                data["retry"] = True
                data["already_completed"] = True
                return data
            return {"manifest_hash": manifest_hash, "result": {},
                    "retry": True, "already_completed": True}
        access_id = claim["access_id"]
        if not reuse_used:
            self.store.mark_holdout(access_id, "running", run_id=self.run_id,
                                    candidate_id=candidate_id,
                                    note="final worker started")
        self.store.set_run_status(self.run_id, "running")
        try:
            result = self._execute_final(candidate_id, cand, holdout, manifest)
            # Statistics such as the SD of a single fold are undefined; they are
            # stored as null and listed, never as NaN or 0.
            result, result_nulls = sanitise_json(result)
            if result_nulls:
                result["null_paths"] = result_nulls
        except KeyboardInterrupt:
            if not reuse_used:
                self.store.mark_holdout(access_id, "interrupted", run_id=self.run_id,
                                        candidate_id=candidate_id, note="KeyboardInterrupt")
            self.store.add_final_eval(FinalEvaluation(
                task_id=ctx.task.task_id, run_id=self.run_id, candidate_id=candidate_id,
                protocol_hash=ctx.protocol_hash, config_hash=manifest["holdout_key"],
                manifest_hash=manifest_hash, holdout_key=manifest["holdout_key"],
                status="interrupted", stage="final", state=CandidateState.INSUFFICIENT,
                exploratory=exploratory, notes="interrupted after holdout access",
            ))
            self.store.finish_run(self.run_id, "interrupted", stage="final")
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            if not reuse_used:
                self.store.mark_holdout(access_id, "failed", run_id=self.run_id,
                                        candidate_id=candidate_id, note=reason[:1000])
            self.store.add_final_eval(FinalEvaluation(
                task_id=ctx.task.task_id, run_id=self.run_id, candidate_id=candidate_id,
                protocol_hash=ctx.protocol_hash, config_hash=manifest["holdout_key"],
                manifest_hash=manifest_hash, holdout_key=manifest["holdout_key"],
                status="failed", stage="final", state=CandidateState.INSUFFICIENT,
                exploratory=exploratory, notes=reason[:1500],
            ))
            atomic_write_text(self.artifacts / "final_evaluation.json", strict_json_dumps({
                "manifest_hash": manifest_hash, "manifest": manifest,
                "status": "failed", "exploratory": exploratory,
                "error": reason[:2000], "result": None,
            }))
            self.store.finish_run(self.run_id, "final_failed", stage="final")
            raise
        if not reuse_used:
            self.store.mark_holdout(access_id, "completed", run_id=self.run_id,
                                    candidate_id=candidate_id,
                                    note="final worker completed; holdout permanently consumed")
        self.store.add_final_eval(FinalEvaluation(
            task_id=ctx.task.task_id, run_id=self.run_id, candidate_id=candidate_id,
            protocol_hash=ctx.protocol_hash, config_hash=manifest["holdout_key"],
            manifest_hash=manifest_hash, holdout_key=manifest["holdout_key"],
            access_id=access_id, status="success", stage="final",
            state=result["state"], exploratory=exploratory,
            result_hash=sha256_obj(result), metrics=result,
            notes=("exploratory reuse of consumed holdout; not confirmatory"
                   if exploratory else "holdout evaluated once after freezing"),
        ))
        bundle = {"manifest_hash": manifest_hash, "manifest": manifest,
                  "access_id": access_id, "status": "success",
                  "exploratory": exploratory, "result": result}
        atomic_write_text(self.artifacts / "final_evaluation.json", strict_json_dumps(bundle))
        self.store.finish_run(self.run_id, "final_complete", stage="final")
        self._write_report()
        return bundle

    def _execute_final(self, candidate_id: str, cand: dict, holdout: FoldSpec,
                       manifest: dict[str, Any]) -> dict[str, Any]:
        ctx = self._ctx
        assert ctx is not None
        seeds = self._seeds()
        metric = ctx.protocol.primary_metric
        per_seed, failures, n_expected = self._paired_folds_descriptor(
            cand, [holdout], seeds, phase="final", purpose="finalize",
        )
        agg = self._aggregate(per_seed, n_expected_folds=n_expected, failures=failures)
        if "error" in agg:
            raise RuntimeError("final evaluation failed: " + agg["error"])
        direction = 1 if ctx.protocol.direction == "higher" else -1
        effect = direction * agg["effect"]
        p = agg.get("p_value")
        classified = self._classify(agg, ctx.policy)
        if not _finite(p):
            state = CandidateState.INSUFFICIENT
        elif classified == EvidenceStatus.VALIDATED:
            state = (CandidateState.EXPLORATORY_POSITIVE if manifest.get("exploratory")
                     else CandidateState.INDEPENDENTLY_CONFIRMED)
        elif classified == EvidenceStatus.REJECTED or (
            effect <= -ctx.policy.min_effect and p <= ctx.policy.fdr_alpha
        ):
            state = CandidateState.REJECTED
        else:
            state = CandidateState.INSUFFICIENT
        control_rows = {}
        for variant in dict.fromkeys([v for v in [ctx.protocol.control_variant,
                                                  *ctx.protocol.control_variants] if v]):
            res = self._ensure_variant(variant, "control", holdout, "final", seeds[0],
                                       control_kind=variant)
            control_rows[variant] = {
                "status": res.get("status"), "config_hash": res.get("config_hash"),
                "metrics": res.get("metrics", {}), "error": res.get("error"),
            }
        trials = []
        for seed in seeds:
            req = self._trial_config(candidate_id, "candidate", cand["descriptor"],
                                     holdout, "final", seed)
            row = self.store.get_trial_by_key(candidate_id, holdout.fold_id,
                                              ctx.protocol_hash, req.config_hash)
            if row:
                trials.append({k: row.get(k) for k in
                               ("trial_id", "seed", "fold_id", "config_hash", "status",
                                "result_hash", "predictions_path", "predictions_sha256")}
                              | {"seed": seed})
            base_req = self._trial_config(ctx.protocol.baseline_variant, "baseline", {},
                                          holdout, "final", seed)
            brow = self.store.get_trial_by_key(ctx.protocol.baseline_variant, holdout.fold_id,
                                               ctx.protocol_hash, base_req.config_hash)
            if brow:
                trials.append({"trial_id": brow.get("trial_id"), "seed": seed,
                               "fold_id": holdout.fold_id,
                               "config_hash": brow.get("config_hash"),
                               "status": brow.get("status"),
                               "result_hash": brow.get("result_hash"),
                               "predictions_path": brow.get("predictions_path"),
                               "predictions_sha256": brow.get("predictions_sha256")})
        return {
            "state": state.value,
            "effect": agg["effect"], "ci_low": agg["ci_low"], "ci_high": agg["ci_high"],
            "p_value": agg["p_value"], "stability": agg["stability"],
            "fold_coverage": agg["fold_coverage"], "n_dates": agg["n_dates"],
            "seed_effects": agg["seed_effects"], "seed_effect_sd": agg["seed_effect_sd"],
            "fold_effect_sd": agg["fold_effect_sd"],
            "time_series_ci": agg["time_series_ci"],
            "per_seed": agg["per_seed"], "folds": agg["folds"],
            "candidate": {"candidate_id": candidate_id, "metrics": per_seed},
            "controls": control_rows,
            "trials": trials,
            "label_isolation": ctx.label_report.get("folds", {}).get(holdout.fold_id),
            "notes": "holdout labels whose prices do not yet exist are excluded and recorded",
        }

    # ------------------------------------------------------------------ report
    def _write_report(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        rounds = self.store.list_rounds(self.run_id)
        evidence = self.store.list_evidence(run_id=self.run_id)
        counts: dict[str, int] = {}
        state_counts: dict[str, int] = {}
        stage_counts: dict[str, int] = {}
        round_counts: dict[str, dict[str, int]] = {}
        for e in evidence:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
            state_counts[e["state"]] = state_counts.get(e["state"], 0) + 1
            stage_counts[e["stage"]] = stage_counts.get(e["stage"], 0) + 1
            per = round_counts.setdefault(e.get("round_id", ""), {})
            per[e["status"]] = per.get(e["status"], 0) + 1
        report = {
            "run_id": self.run_id,
            "run_status": (self.store.get_run(self.run_id) or {}).get("status"),
            "domain": ctx.adapter.name,
            "protocol_hash": ctx.protocol_hash,
            "identity_hash": ctx.context_hash,
            "code": ctx.code,
            "environment": ctx.environment,
            "snapshot": ctx.snapshot.model_dump(mode="json"),
            "data_manifest_hash": ctx.data_manifest_hash,
            "label_expression": ctx.label_expression,
            "label_offset_trading_days": ctx.label_offset,
            "label_isolation": ctx.label_report,
            "seeds": self._seeds(),
            "rounds": rounds,
            "evidence_counts": counts,
            "state_counts": state_counts,
            "stage_counts": stage_counts,
            "evidence": evidence,
            "frozen_family": self.store.get_frozen_family(self.run_id),
            "final_evals": self.store.list_final_evals(self.run_id),
        }
        atomic_write_text(self.artifacts / "report.json", strict_json_dumps(report))
        self.store.export_evidence_jsonl(self.artifacts / "evidence.jsonl", run_id=self.run_id)

        lines = [
            f"# PredicateLab report — {self.run_id}",
            "",
            f"- domain: `{ctx.adapter.name}`",
            f"- protocol hash: `{ctx.protocol_hash[:16]}`",
            f"- identity hash: `{ctx.context_hash[:16]}`",
            f"- snapshot: `{ctx.snapshot.snapshot_id}`",
            f"- data suitability: **{ctx.task.metadata.get('suitability', {}).get('status', 'unknown')}**",
            f"- label expression: `{ctx.label_expression or 'same-sample'}` "
            f"({ctx.label_offset} future trading days)",
            f"- seeds: `{self._seeds()}`",
            "",
            "## Evidence counts (status / state / stage)",
            "",
            "| status | count |",
            "|---|---|",
        ]
        for k, v in sorted(counts.items()):
            lines.append(f"| {k} | {v} |")
        lines += ["", "| state | count |", "|---|---|"]
        for k, v in sorted(state_counts.items()):
            lines.append(f"| {k} | {v} |")
        lines += ["", "| stage | count |", "|---|---|"]
        for k, v in sorted(stage_counts.items()):
            lines.append(f"| {k} | {v} |")
        lines += ["", "> Discovery-positive means `exploratory_positive`, never "
                  "`independently_confirmed`. Confirmation and final results are "
                  "listed separately.", "", "## Rounds", ""]
        for r in rounds:
            s = r["summary"] or {}
            rc = round_counts.get(r["round_id"], {})
            n_validated = rc.get("validated", r["n_validated"])
            n_rejected = rc.get("rejected", r["n_rejected"])
            n_insufficient = rc.get("insufficient_evidence", r["n_insufficient"])
            n_screened = rc.get("screened_out", 0)
            lines.append(f"### Round {r['round_index']} ({r['round_id']}) — `{r['status']}`")
            lines.append("")
            lines.append(f"- proposed: {r['n_proposed']}, bound: {r['n_bound']}, "
                         f"validated: {n_validated}, rejected: {n_rejected}, "
                         f"insufficient: {n_insufficient}, screened_out: {n_screened}")
            if s.get("validation"):
                lines += ["", "| candidate | status | state | effect | CI | p | q | stability |",
                          "|---|---|---|---|---|---|---|---|"]
                for v in s["validation"]:
                    ci = v.get("ci") or [None, None]
                    ci_s = (f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
                            if all(isinstance(x, (int, float)) for x in ci) else "—")
                    lines.append(
                        f"| {v['candidate_id'][:12]} | {v['status']} | {v['state']} | "
                        f"{v['effect']:+.4f} | {ci_s} | {v['p']:.4g} | "
                        f"{v['q'] if v['q'] is not None else float('nan'):.4g} | "
                        f"{v['stability']:.2f} |"
                    )
            lines.append("")
        fin = self.store.list_final_evals(self.run_id)
        if fin:
            lines += ["## Final evaluation", "",
                      "| candidate | status | state | exploratory | manifest |",
                      "|---|---|---|---|---|"]
            for f in fin:
                lines.append(f"| {f['candidate_id'][:12]} | {f['status']} | {f['state']} | "
                             f"{bool(f['exploratory'])} | `{(f['manifest_hash'] or '')[:16]}` |")
        atomic_write_text(self.artifacts / "report.md", "\n".join(lines))

    # --------------------------------------------------------------- utilities
    def export(self, out_dir: str | Path) -> Path:
        """Immutable evidence bundle with predictions and checksums.

        Trials whose adapter did not emit predictions are counted as missing in
        ``bundle_manifest.json``.  An empty path (``Path("")`` is the current
        directory) or any non-file path is skipped rather than copied.
        """
        if self._ctx is None:
            self._build_context()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        names = (
            "context.json", "report.json", "report.md", "evidence.jsonl",
            "frozen_manifest.json", "frozen_family.json", "confirmation.json",
            "final_evaluation.json",
        )
        for name in names:
            src = self.artifacts / name
            if src.exists():
                dst = out / name
                shutil.copy2(src, dst)
                files.append(dst)
        trials = self.store.list_trials(run_id=self.run_id)
        atomic_write_text(out / "trials.json", strict_json_dumps(trials))
        files.append(out / "trials.json")
        evidence = self.store.list_evidence(run_id=self.run_id)
        atomic_write_text(out / "evidence.json", strict_json_dumps(evidence))
        files.append(out / "evidence.json")
        holdout = [r for r in self.store.list_holdout_access()
                   if r["run_id"] == self.run_id]
        atomic_write_text(out / "holdout_access.json", strict_json_dumps(holdout))
        files.append(out / "holdout_access.json")
        pred_dir = out / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        prediction_files: list[str] = []
        n_missing_predictions = 0
        for trial in trials:
            raw = str(trial.get("predictions_path") or "").strip()
            src = Path(raw) if raw else None
            if src is None or not src.is_file():
                n_missing_predictions += 1
                continue
            dst = pred_dir / f"{trial['trial_id']}_{src.name}"
            shutil.copy2(src, dst)
            files.append(dst)
            prediction_files.append(dst.relative_to(out).as_posix())
        checksums = []
        for f in sorted(files):
            rel = f.relative_to(out).as_posix()
            checksums.append(f"{sha256_file(f)}  {rel}")
        atomic_write_text(out / "checksums.sha256", "\n".join(checksums) + "\n")
        manifest = {
            "run_id": self.run_id,
            "protocol_hash": self._ctx.protocol_hash if self._ctx else "",
            "identity_hash": self._ctx.context_hash if self._ctx else "",
            "predictions": {
                "n_copied": len(prediction_files),
                "n_missing": n_missing_predictions,
                "files": sorted(prediction_files),
            },
            "files": [f.relative_to(out).as_posix() for f in sorted(files)]
                     + ["checksums.sha256"],
            "checksums_sha256": sha256_file(out / "checksums.sha256"),
        }
        manifest["bundle_hash"] = sha256_obj(manifest)
        atomic_write_text(out / "bundle_manifest.json", strict_json_dumps(manifest))
        return out

    def publish_snapshot(self, out_dir: str | Path, *,
                         include_models: bool = False,
                         include_data: bool = False,
                         include_trial_details: bool = False,
                         config_path: str | Path | None = None,
                         repo_root: str | Path | None = None) -> Path:
        """Build a lightweight, portable snapshot of this run.

        Only ``results``, ``logs``, ``figures`` and ``code`` are copied by
        default; trained weights (``models``) and bulk arrays/databases
        (``data``) require an explicit opt-in.  The raw run directory is never
        modified.  See :mod:`predicatelab.core.artifacts`.
        """
        from predicatelab.core.artifacts import build_run_snapshot

        return build_run_snapshot(
            self.artifacts, out_dir,
            include_models=include_models,
            include_data=include_data,
            include_trial_details=include_trial_details,
            config=self.config,
            config_path=config_path,
            repo_root=repo_root if repo_root is not None else self._repo_root(),
        )


def finalize_run(config: dict[str, Any], run_id: str, candidate_id: str,
                 db_path: str | None = None) -> dict[str, Any]:
    engine = PredicateLabEngine(config, db_path=db_path, run_id=run_id)
    try:
        return engine.finalize(candidate_id)
    finally:
        engine.store.close()
