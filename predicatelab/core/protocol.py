"""Experiment protocol helpers: fold construction, purge/embargo, leak guards.

Reproducibility rules enforced here:

* validation folds never overlap the holdout period;
* the last ``purge_days`` trading days of a training period are dropped so that
  the label window of the last training sample ends before validation starts;
* the first ``embargo_days`` trading days of a validation period are dropped to
  break serial dependence with the training block;
* a holdout fold is only reachable through :class:`HoldoutGuard` after a frozen
  manifest has been stored, i.e. after candidate selection is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

from predicatelab.core.schema import FoldSpec


class ProtocolViolation(RuntimeError):
    """Raised when search code tries to touch a held-out segment."""


def _as_date(x: str | date | datetime) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return datetime.strptime(x[:10], "%Y-%m-%d").date()


def trading_days_between(calendar: Sequence[date | str], start, end) -> list[date]:
    s, e = _as_date(start), _as_date(end)
    days = [d for d in (_as_date(x) for x in calendar) if s <= d <= e]
    return sorted(days)


def effective_time_fold(fold: FoldSpec, calendar: Sequence[date | str]) -> FoldSpec:
    """Apply purge to the train tail and embargo to the validation head."""
    if fold.kind != "time" or not fold.train_end or not fold.valid_start:
        return fold
    train_days = trading_days_between(calendar, fold.train_start, fold.train_end)
    valid_days = trading_days_between(calendar, fold.valid_start, fold.valid_end)
    if not train_days or not valid_days:
        raise ValueError(f"fold {fold.fold_id}: empty train or validation period")
    if fold.purge_days > 0:
        if fold.purge_days >= len(train_days):
            raise ValueError(f"fold {fold.fold_id}: purge removes the whole training period")
        train_end = train_days[-(fold.purge_days + 1)]
    else:
        train_end = train_days[-1]
    if fold.embargo_days > 0:
        if fold.embargo_days >= len(valid_days):
            raise ValueError(f"fold {fold.fold_id}: embargo removes the whole validation period")
        valid_start = valid_days[fold.embargo_days]
    else:
        valid_start = valid_days[0]
    return fold.model_copy(
        update={
            "train_end": train_end.isoformat(),
            "valid_start": valid_start.isoformat(),
            "notes": (fold.notes + f"; effective train_end={train_end}, valid_start={valid_start}").strip("; "),
        }
    )


def validate_protocol(folds: Sequence[FoldSpec], holdout: FoldSpec | None) -> None:
    """Static checks that catch accidental leakage in a protocol definition."""
    ids = [f.fold_id for f in folds]
    if len(ids) != len(set(ids)):
        raise ProtocolViolation("duplicate fold_id in protocol")
    for f in folds:
        if f.role != "search":
            raise ProtocolViolation(f"fold {f.fold_id} in search list has role={f.role!r}")
    if holdout is not None:
        if holdout.fold_id in ids:
            raise ProtocolViolation("holdout fold must not appear in search folds")
        if holdout.role != "holdout":
            raise ProtocolViolation("holdout fold must have role='holdout'")
    time_folds = [f for f in folds if f.kind == "time"]
    for f in time_folds:
        if not (f.train_start and f.train_end and f.valid_start and f.valid_end):
            raise ProtocolViolation(f"fold {f.fold_id}: incomplete time period")
        if _as_date(f.train_end) >= _as_date(f.valid_start):
            raise ProtocolViolation(f"fold {f.fold_id}: train_end must precede valid_start")
        if _as_date(f.train_start) > _as_date(f.train_end):
            raise ProtocolViolation(f"fold {f.fold_id}: train_start after train_end")
        if _as_date(f.valid_start) > _as_date(f.valid_end):
            raise ProtocolViolation(f"fold {f.fold_id}: valid_start after valid_end")
    # search validation periods must not overlap each other
    ordered = sorted(time_folds, key=lambda f: _as_date(f.valid_start))
    for prev, cur in zip(ordered, ordered[1:]):
        if _as_date(cur.valid_start) <= _as_date(prev.valid_end):
            raise ProtocolViolation(
                f"validation periods overlap: {prev.fold_id} {prev.valid_start}..{prev.valid_end} "
                f"and {cur.fold_id} {cur.valid_start}..{cur.valid_end}"
            )
    if holdout is not None and holdout.test_start:
        for f in time_folds:
            if _as_date(f.valid_end) >= _as_date(holdout.test_start):
                raise ProtocolViolation(
                    f"fold {f.fold_id} validation reaches into holdout (starts {holdout.test_start})"
                )


@dataclass
class HoldoutGuard:
    """Only the finalize step may unlock the holdout.

    Search code holds an instance with ``frozen_manifest=None``; every attempt
    to read a holdout fold raises :class:`ProtocolViolation`, which the engine
    records as a BLOCKED evidence item rather than crashing the whole run.
    """

    search_fold_ids: set[str]
    holdout_fold_id: str | None
    frozen_manifest_hash: str | None = None

    @classmethod
    def from_folds(
        cls, search_folds: Iterable[FoldSpec], holdout: FoldSpec | None, frozen_manifest_hash: str | None = None
    ) -> "HoldoutGuard":
        return cls(
            search_fold_ids={f.fold_id for f in search_folds},
            holdout_fold_id=holdout.fold_id if holdout else None,
            frozen_manifest_hash=frozen_manifest_hash,
        )

    def assert_fold_accessible(self, fold_id: str, purpose: str) -> None:
        if purpose not in {"search", "finalize"}:
            raise ValueError(f"unknown purpose {purpose!r}")
        if fold_id in self.search_fold_ids:
            if purpose != "search":
                raise ProtocolViolation(f"finalize tried to use search fold {fold_id}")
            return
        if fold_id == self.holdout_fold_id:
            if purpose != "finalize":
                raise ProtocolViolation(
                    f"search attempted to access holdout fold {fold_id}; "
                    "freeze candidates and run finalize instead"
                )
            if not self.frozen_manifest_hash:
                raise ProtocolViolation(
                    f"holdout fold {fold_id} requested without a frozen candidate manifest"
                )
            return
        raise ProtocolViolation(f"unknown fold {fold_id!r}")

    def unlock(self, frozen_manifest_hash: str) -> "HoldoutGuard":
        if not frozen_manifest_hash:
            raise ProtocolViolation("empty frozen manifest")
        return HoldoutGuard(self.search_fold_ids, self.holdout_fold_id, frozen_manifest_hash)
