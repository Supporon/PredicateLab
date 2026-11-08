"""Synthetic time-series domain adapter for auditing the *real* PredicateLab engine.

Everything except the trial outcome is the production code path:
  * TSIL RuleGenerator (real candidate proposals, feedback-driven mutations);
  * TSIL DSL static validation;
  * real trading calendar + label-isolation + protocol checks from the engine.

Trial outcomes are drawn from an explicit ground-truth model so the
search -> screen -> validate -> freeze -> confirm -> final pipeline can be
calibrated with negative and positive controls:

  daily paired IC diff (candidate - baseline) on date t in year y, seed s:
      delta(c) + u[family(c), y] + e[c, s, fold] + v[c, s, t]
  delta(c)          true effect (0 under the null; >0 for "true" families)
  u[family, year]   family x year heterogeneity (shared by all windows of a family)
  e[c, seed, fold]  training-randomness offset for one fitted model
  v                 iid daily noise

All randomness is derived from (replicate_seed, keys) so results are exact
functions of the scenario parameters.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from predicatelab.core.adapter import DomainAdapter, TrialRequest
from predicatelab.core.hashing import sha256_obj
from predicatelab.core.protocol import effective_time_fold
from predicatelab.core.schema import (
    DataContract, DataSnapshot, EvaluationSpec, ExperimentProtocol, FoldSpec,
)
from predicatelab.domains.tsil import dsl
from predicatelab.domains.tsil.generator import RuleGenerator

LABEL = "Ref($close, -2) / Ref($close, -1) - 1"


def _rng(*keys: Any) -> np.random.Generator:
    h = hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def family_of(desc: dict) -> str:
    fam = desc.get("family")
    if fam == "native":
        return "native:" + str(desc.get("predicate"))
    if fam == "composite":
        parts = desc.get("parts") or []
        return "composite:" + "+".join(sorted(family_of(p) for p in parts))
    return str(fam)


class SynthTSAdapter(DomainAdapter):
    name = "synth_ts"
    TRIAL_LOG: list[dict] = []        # every executed (non-cached) trial, for audit

    def __init__(self, config: dict[str, Any]):
        self.config = config
        s = config.get("synth", {})
        self.rep = int(s.get("replicate_seed", 0))
        self.sd_v = float(s.get("sd_daily", 0.035))
        self.sd_u = float(s.get("sd_family_year", 0.0))
        self.sd_e = float(s.get("sd_seed_fold", 0.0))
        self.true_effects = dict(s.get("true_family_effects", {}))   # family -> delta
        self.calendar = [ln.strip() for ln in Path(s["calendar_path"]).read_text().splitlines() if ln.strip()]
        self._eval = EvaluationSpec(name="synth_ic", primary_metric="ic", direction="higher",
                                    paired=True, multiple_testing="benjamini-hochberg")

    # ---- data ------------------------------------------------------------
    def data_contract(self, config):
        return DataContract(dataset_id="synth-ts", domain=self.name, description="synthetic",
                            label_definition=LABEL,
                            label_window={"uses_future_days": 2, "unit": "trading_day"},
                            time_range=(self.calendar[0], self.calendar[-1]))

    def snapshot(self, config):
        return DataSnapshot(dataset_id="synth-ts", domain=self.name, manifest_method="synthetic",
                            manifest_hash=sha256_obj({"synth": "v1", "cal": len(self.calendar)}),
                            details={})

    def trading_calendar(self, config):
        return list(self.calendar)

    # ---- protocol (same construction as TSILAdapter.build_protocol) --------
    def build_protocol(self, config):
        p = config["protocol"]

        def fold(row, role):
            f = FoldSpec(fold_id=row["id"], role=role, train_start=row["train"][0],
                         train_end=row["train"][1], valid_start=row["valid"][0],
                         valid_end=row["valid"][1],
                         test_start=(row.get("test") or [None, None])[0],
                         test_end=(row.get("test") or [None, None])[1],
                         purge_days=int(p.get("purge_days", 2)),
                         embargo_days=int(p.get("embargo_days", 3)))
            return effective_time_fold(f, self.calendar)

        return ExperimentProtocol(
            task_id="task-synth", domain=self.name, primary_metric="ic", direction="higher",
            folds=[fold(r, "search") for r in p["folds"]],
            confirmation_folds=[fold(r, "confirmation") for r in p.get("confirmation_folds", [])],
            confirmation_kind=p.get("confirmation_kind", "time_forward"),
            holdout_fold=fold(p["holdout"], "holdout") if p.get("holdout") else None,
            seeds=[int(x) for x in config.get("seeds", [2025])],
            model_spec={"synthetic": True}, evaluation=self._eval)

    @property
    def evaluation(self):
        return self._eval

    # ---- candidates --------------------------------------------------------
    def validate_descriptor(self, kind, descriptor):
        dsl.validate_descriptor(descriptor, seq_len=20, feature_dim=158)

    def make_generator(self, config, policy):
        merged = dict(config)
        merged["generator_config"] = dict(policy.get("generator_config", {}) or {})
        return RuleGenerator(merged)

    def probe(self, descriptor, fold, snapshot, config, workdir):
        r = _rng("probe", self.rep, sha256_obj(descriptor))
        return {"ok": True, "reason": "synthetic", "signature": r.normal(size=64).tolist()}

    # ---- trials ------------------------------------------------------------
    def _dates(self, fold: FoldSpec) -> list[str]:
        if fold.test_start:
            s, e = fold.test_start, fold.test_end
        else:
            s, e = fold.valid_start, fold.valid_end
        return [d for d in self.calendar if s <= d <= e]

    def run_trial(self, request: TrialRequest) -> dict[str, Any]:
        dates = self._dates(request.fold)
        cid = request.candidate_id
        base_r = _rng("base", self.rep, request.fold.fold_id)
        base = 0.04 + 0.08 * base_r.normal(size=len(dates))          # shared market-level daily IC
        if request.variant == "baseline":
            # the baseline is itself a trained model: it has its own seed x fold offset,
            # shared by every candidate it is paired with (same seed, same fold)
            base = base + _rng("eb", self.rep, request.seed, request.fold.fold_id).normal() * self.sd_e
        if request.variant == "candidate":
            fam = family_of(request.descriptor)
            delta = float(self.true_effects.get(fam, 0.0))
            u_by_yr: dict[str, float] = {}
            u = np.array([
                u_by_yr.setdefault(d[:4], _rng("u", self.rep, fam, d[:4]).normal() * self.sd_u)
                for d in dates
            ])
            e = _rng("e", self.rep, cid, request.seed, request.fold.fold_id).normal() * self.sd_e
            v = _rng("v", self.rep, cid, request.seed, request.fold.fold_id).normal(size=len(dates)) * self.sd_v
            series = base + delta + u + e + v
        elif request.variant == "control":
            v = _rng("ctl", self.rep, request.seed, request.fold.fold_id).normal(size=len(dates)) * self.sd_v
            series = base + v
        else:
            series = base
        SynthTSAdapter.TRIAL_LOG.append({"candidate_id": cid, "variant": request.variant,
                                         "fold": request.fold.fold_id, "phase": request.phase,
                                         "seed": request.seed, "first_date": dates[0] if dates else None,
                                         "last_date": dates[-1] if dates else None})
        ser = {d: float(x) for d, x in zip(dates, series)}
        return {"status": "success", "metrics": {"ic": float(np.mean(series)), "n_dates": len(dates)},
                "metric_series": {"ic": ser}, "fitted_params": {}, "predictions_path": "",
                "duration_s": 0.0, "error": "", "worker": {"synthetic": True}}


def base_config(tmp: Path, calendar_path: str, **synth) -> dict:
    return {
        "run": {"name": "synth", "domain": "synth_ts", "artifacts_dir": str(tmp / "artifacts"),
                "db": str(tmp / "db.sqlite"), "rounds": 2, "require_confirmation": True},
        "seeds": [2025, 2026, 2027],
        "synth": {"calendar_path": calendar_path, **synth},
        "training": {"screen": {"n_epochs": 1}, "full": {"n_epochs": 1}},
        "protocol": {
            "purge_days": 2, "embargo_days": 3,
            "folds": [
                {"id": "f2020", "train": ["2018-01-01", "2019-12-31"], "valid": ["2020-01-01", "2020-12-31"]},
                {"id": "f2021", "train": ["2018-01-01", "2020-12-31"], "valid": ["2021-01-01", "2021-12-31"]},
                {"id": "f2022", "train": ["2018-01-01", "2021-12-31"], "valid": ["2022-01-01", "2022-12-31"]},
            ],
            "confirmation_folds": [
                {"id": "f2023", "train": ["2018-01-01", "2022-12-31"], "valid": ["2023-01-01", "2023-12-31"]}],
            "confirmation_kind": "time_forward",
            "holdout": {"id": "holdout_2024_2025", "train": ["2018-01-01", "2022-12-31"],
                        "valid": ["2023-01-01", "2023-12-31"], "test": ["2024-01-01", "2025-08-31"]},
        },
        # identical to configs/tsil_lstm_full.yaml search block
        "search": {"generator": "rule", "save_predictions": False,
                   "generator_config": {"seed_pool": "paper6+dsl", "mutation_top_k": 4,
                                        "max_composite_depth": 2},
                   "max_candidates_per_round": 24, "screen_folds": ["f2022"],
                   "validation_folds": ["f2020", "f2021"], "full_validation_top_k": 4,
                   "min_screen_effect": 0.0, "fdr_alpha": 0.1, "min_effect": 0.002,
                   "min_stability": 0.5, "bootstrap_samples": 1000, "bootstrap_block": 5,
                   "nw_lag": 5, "seed": 2025},
    }


def install():
    """Route domain 'synth_ts' to this adapter inside the real engine."""
    import predicatelab.domains as domains
    orig = domains.create_adapter

    def create(config):
        if config.get("run", {}).get("domain") == "synth_ts":
            return SynthTSAdapter(config)
        return orig(config)

    domains.create_adapter = create
