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


def validate_protocol(
    folds: Sequence[FoldSpec],
    holdout: FoldSpec | None,
    confirmation_folds: Sequence[FoldSpec] = (),
    *,
    label_future_offset: int | None = None,
) -> None:
    """Static checks that catch accidental leakage in a protocol definition.

    V1.1 makes two previously silent leaks hard failures:

    * ``purge_days=0`` on a temporal fold is rejected outright -- the later
      label-window check cannot repair a protocol that deliberately keeps the
      train tail;
    * confirmation folds must be temporally later than every discovery fold;
      the same dates may never serve as both screen and independent check.
    """
    folds = list(folds)
    confirmation_folds = list(confirmation_folds)
    ids = [f.fold_id for f in [*folds, *confirmation_folds]]
    if len(ids) != len(set(ids)):
        raise ProtocolViolation("duplicate fold_id in protocol")
    for f in folds:
        if f.role != "search":
            raise ProtocolViolation(f"fold {f.fold_id} in search list has role={f.role!r}")
    for f in confirmation_folds:
        if f.role != "confirmation":
            raise ProtocolViolation(
                f"confirmation fold {f.fold_id} has role={f.role!r}; expected 'confirmation'"
            )
    if holdout is not None:
        if holdout.fold_id in ids:
            raise ProtocolViolation("holdout fold must not appear in search/confirmation folds")
        if holdout.role != "holdout":
            raise ProtocolViolation("holdout fold must have role='holdout'")
    if not folds:
        raise ProtocolViolation("protocol has no discovery folds")

    all_folds = [*folds, *confirmation_folds]
    time_folds = [f for f in all_folds if f.kind == "time"]
    for f in time_folds:
        if not (f.train_start and f.train_end and f.valid_start and f.valid_end):
            raise ProtocolViolation(f"fold {f.fold_id}: incomplete time period")
        if f.purge_days < 1:
            raise ProtocolViolation(
                f"fold {f.fold_id}: purge_days={f.purge_days} is not allowed; a forward-looking "
                "label needs at least one purged trading day (no natural-day substitution)"
            )
        if label_future_offset is not None and f.purge_days < int(label_future_offset):
            raise ProtocolViolation(
                f"fold {f.fold_id}: purge_days={f.purge_days} is smaller than the label's "
                f"{label_future_offset} future trading-day dependency"
            )
        if f.embargo_days < 0:
            raise ProtocolViolation(f"fold {f.fold_id}: negative embargo_days")
        if _as_date(f.train_end) >= _as_date(f.valid_start):
            raise ProtocolViolation(f"fold {f.fold_id}: train_end must precede valid_start")
        if _as_date(f.train_start) > _as_date(f.train_end):
            raise ProtocolViolation(f"fold {f.fold_id}: train_start after train_end")
        if _as_date(f.valid_start) > _as_date(f.valid_end):
            raise ProtocolViolation(f"fold {f.fold_id}: valid_start after valid_end")
    # no validation period may overlap another (discovery or confirmation)
    ordered = sorted(time_folds, key=lambda f: _as_date(f.valid_start))
    for prev, cur in zip(ordered, ordered[1:]):
        if _as_date(cur.valid_start) <= _as_date(prev.valid_end):
            raise ProtocolViolation(
                f"validation periods overlap: {prev.fold_id} {prev.valid_start}..{prev.valid_end} "
                f"and {cur.fold_id} {cur.valid_start}..{cur.valid_end}"
            )
    # confirmation must use later data than every discovery fold
    search_time = [f for f in folds if f.kind == "time" and f.valid_end]
    if confirmation_folds and search_time:
        last_search_end = max(_as_date(f.valid_end) for f in search_time)
        for f in confirmation_folds:
            if not f.valid_start:
                raise ProtocolViolation(f"confirmation fold {f.fold_id}: missing valid_start")
            if _as_date(f.valid_start) <= last_search_end:
                raise ProtocolViolation(
                    f"confirmation fold {f.fold_id} starts {f.valid_start} on/before the last "
                    f"discovery validation end {last_search_end}; it would not be independent"
                )
    if holdout is not None and holdout.test_start:
        for f in time_folds:
            if _as_date(f.valid_end) >= _as_date(holdout.test_start):
                raise ProtocolViolation(
                    f"fold {f.fold_id} validation reaches into holdout (starts {holdout.test_start})"
                )
        if holdout.kind == "time" and holdout.purge_days < 1:
            raise ProtocolViolation(
                f"holdout {holdout.fold_id}: purge_days={holdout.purge_days} is not allowed"
            )


@dataclass
class HoldoutGuard:
    """Stage-aware fold access control.

    Search/discovery code may only read discovery folds.  Confirmation folds are
    readable only with ``purpose='confirmation'`` and only after the candidate
    family was frozen; the holdout is readable only through ``finalize`` after
    both a frozen manifest and an explicit ledger claim.
    """

    search_fold_ids: set[str]
    holdout_fold_id: str | None
    frozen_manifest_hash: str | None = None
    confirmation_fold_ids: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.confirmation_fold_ids is None:
            self.confirmation_fold_ids = set()

    @classmethod
    def from_folds(
        cls,
        search_folds: Iterable[FoldSpec],
        holdout: FoldSpec | None,
        frozen_manifest_hash: str | None = None,
        confirmation_folds: Iterable[FoldSpec] = (),
    ) -> "HoldoutGuard":
        return cls(
            search_fold_ids={f.fold_id for f in search_folds},
            holdout_fold_id=holdout.fold_id if holdout else None,
            frozen_manifest_hash=frozen_manifest_hash,
            confirmation_fold_ids={f.fold_id for f in confirmation_folds},
        )

    def assert_fold_accessible(self, fold_id: str, purpose: str) -> None:
        if purpose not in {"search", "confirmation", "finalize"}:
            raise ValueError(f"unknown purpose {purpose!r}")
        if fold_id in self.search_fold_ids:
            if purpose != "search":
                raise ProtocolViolation(
                    f"{purpose} tried to use discovery/search fold {fold_id}"
                )
            return
        if fold_id in self.confirmation_fold_ids:
            if purpose != "confirmation":
                raise ProtocolViolation(
                    f"purpose={purpose!r} attempted to use confirmation fold {fold_id}; "
                    "screen data and independent confirmation data must stay disjoint"
                )
            if not self.frozen_manifest_hash:
                raise ProtocolViolation(
                    f"confirmation fold {fold_id} requested before the candidate family was frozen"
                )
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
        return HoldoutGuard(
            self.search_fold_ids, self.holdout_fold_id, frozen_manifest_hash,
            set(self.confirmation_fold_ids),
        )


def validate_label_isolation(
    folds: Sequence[FoldSpec],
    confirmation_folds: Sequence[FoldSpec],
    holdout: FoldSpec | None,
    calendar: Sequence[date | str],
    label_expression: str,
    label_window: dict,
):
    """Trim every partition to its label-availability boundary.

    The returned folds are the *effective* folds; callers must use them instead
    of the nominal protocol dates.  Raises ``LabelDependencyError`` on an
    unresolved leak.  A discovery/confirmation fold ending on the same day the
    holdout starts is rejected by the underlying module.
    """
    from predicatelab.core.label_window import validate_and_isolate

    combined = [*folds, *confirmation_folds]
    return validate_and_isolate(combined, holdout, calendar, label_expression, label_window)
