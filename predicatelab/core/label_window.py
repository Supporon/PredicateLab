"""Label-dependency intervals on the real trading calendar.

A label such as ``Ref($close, -2) / Ref($close, -1) - 1`` is *not* available on
its sample date: it needs the close prices of the next one and two trading
days.  Subtracting natural days is therefore wrong (weekends, holidays, the
New Year boundary).  This module:

* parses the maximum future offset from the label expression and cross-checks
  it against the declared ``label_window``;
* computes each sample's label availability date on the supplied trading
  calendar;
* trims a train/validation/test segment so that **no label in the segment
  depends on prices outside that segment** (except the holdout, where the
  dependency is recorded explicitly);
* reports the exact leaked-date and leaked-sample counts instead of hiding
  them.

The rules are deliberately conservative (fail closed):

* ``train`` labels must be available strictly before ``valid_start``;
* ``valid`` (and confirmation) labels must be available on or before the
  nominal ``valid_end`` -- otherwise the last samples are trimmed;
* ``holdout.valid`` (early stopping) labels must not reach into the test
  segment; ``holdout.test`` tail labels whose prices do not exist yet are
  dropped and recorded as ``tail_labels_unavailable``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Sequence

from predicatelab.core.schema import FoldSpec


class LabelDependencyError(RuntimeError):
    """Raised when a label window cannot be isolated from a later partition."""


_REF_FUTURE = re.compile(r"Ref\s*\(\s*[^,]+,\s*-\s*(\d+)\s*\)")


def as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def normalise_calendar(calendar: Sequence[str | date | datetime]) -> list[date]:
    days = sorted({as_date(d) for d in calendar})
    if not days:
        raise LabelDependencyError("empty trading calendar")
    return days


def parse_future_offset(expression: str) -> int | None:
    """Return the largest ``-N`` Ref offset in a qlib label expression."""
    if not expression:
        return None
    offsets = [int(m.group(1)) for m in _REF_FUTURE.finditer(expression)]
    return max(offsets) if offsets else None


def declared_future_offset(label_window: dict[str, Any]) -> int | None:
    for key in ("uses_future_days", "future_days", "offset_days"):
        if key in label_window and label_window[key] is not None:
            try:
                return int(label_window[key])
            except (TypeError, ValueError):
                return None
    return None


def resolve_future_offset(expression: str, label_window: dict[str, Any]) -> int:
    """Reconcile the expression with the declared contract (fail closed)."""
    parsed = parse_future_offset(expression)
    declared = declared_future_offset(label_window)
    if parsed is None and declared is None:
        raise LabelDependencyError(
            f"cannot determine label future dependency from expression={expression!r} "
            f"and label_window={label_window!r}"
        )
    if parsed is not None and declared is not None and parsed != declared:
        raise LabelDependencyError(
            f"label expression implies {parsed} future trading days but the data contract "
            f"declares {declared}; refusing to guess"
        )
    return int(parsed if parsed is not None else declared)


@dataclass
class LabelCalendar:
    calendar: list[date]
    future_offset: int
    expression: str = ""

    def __post_init__(self) -> None:
        self._index = {d: i for i, d in enumerate(self.calendar)}

    def availability(self, sample_date: str | date | datetime) -> date | None:
        d = as_date(sample_date)
        i = self._index.get(d)
        if i is None:
            return None
        j = i + self.future_offset
        return self.calendar[j] if j < len(self.calendar) else None

    def days_between(self, start, end) -> list[date]:
        s, e = as_date(start), as_date(end)
        return [d for d in self.calendar if s <= d <= e]

    # ------------------------------------------------------------------ trims
    def last_fully_available(
        self, start, nominal_end
    ) -> tuple[date | None, list[date], list[date]]:
        """Last sample date in [start, nominal_end] whose label is available by nominal_end.

        Returns ``(last_date, trimmed_dates, unavailable_tail)``.  ``trimmed_dates``
        are in-range dates removed because their label needs later prices;
        ``unavailable_tail`` are dates whose label cannot be computed at all
        because the calendar ends.
        """
        days = self.days_between(start, nominal_end)
        kept: list[date] = []
        trimmed: list[date] = []
        unavailable: list[date] = []
        end = as_date(nominal_end)
        for d in days:
            av = self.availability(d)
            if av is None:
                unavailable.append(d)
            elif av <= end:
                kept.append(d)
            else:
                trimmed.append(d)
        return (kept[-1] if kept else None), trimmed, unavailable


def _fmt(d: date | None) -> str | None:
    return d.isoformat() if d else None


def isolation_report(
    fold: FoldSpec,
    lab: LabelCalendar,
    *,
    partition: str,
) -> dict[str, Any]:
    """Describe (and trim) one fold.  Returns the report dict with ``fold``."""
    out: dict[str, Any] = {
        "fold_id": fold.fold_id,
        "partition": partition,
        "label_expression": lab.expression,
        "future_offset_trading_days": lab.future_offset,
        "nominal": {
            "train_start": fold.train_start, "train_end": fold.train_end,
            "valid_start": fold.valid_start, "valid_end": fold.valid_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
        },
    }
    updates: dict[str, Any] = {}

    if fold.kind != "time":
        out.update({"ok": True, "applicable": False, "reason": "non-temporal fold"})
        return out

    # ---- train tail: every training label must be available before validation
    if fold.train_start and fold.train_end and fold.valid_start:
        train_last = lab.last_fully_available(fold.train_start, fold.train_end)[0]
        if train_last is None:
            raise LabelDependencyError(f"fold {fold.fold_id}: no training label is available")
        av = lab.availability(train_last)
        if av is None or av >= as_date(fold.valid_start):
            raise LabelDependencyError(
                f"fold {fold.fold_id}: training label at {train_last} becomes available at "
                f"{_fmt(av)} which is not before valid_start {fold.valid_start}; "
                "increase purge_days / purge on the trading calendar"
            )
        out["train"] = {"effective_train_end": train_last.isoformat(),
                        "last_label_available": _fmt(av),
                        "trimmed": train_last != as_date(fold.train_end)}
        if train_last != as_date(fold.train_end):
            updates["train_end"] = train_last.isoformat()
            note = (f"label isolation trimmed train_end {fold.train_end}->"
                    f"{train_last.isoformat()} (tail labels need prices at/after valid_start)")
            updates["notes"] = (fold.notes + "; " + note).strip("; ")

    # ---- validation tail: no label may depend on data after the segment
    if fold.valid_start and fold.valid_end:
        last, trimmed, unavailable = lab.last_fully_available(fold.valid_start, fold.valid_end)
        if last is None:
            raise LabelDependencyError(f"fold {fold.fold_id}: no validation label is available")
        av = lab.availability(last)
        out["valid"] = {
            "nominal_valid_end": fold.valid_end,
            "effective_valid_end": last.isoformat(),
            "last_label_available": _fmt(av),
            "trimmed_dates": [d.isoformat() for d in trimmed],
            "n_trimmed_dates": len(trimmed),
            "label_unavailable_dates": [d.isoformat() for d in unavailable],
        }
        if trimmed or unavailable:
            updates["valid_end"] = last.isoformat()
            note = (f"label isolation trimmed valid_end {fold.valid_end}->{last.isoformat()} "
                    f"({len(trimmed)} dates need later prices)")
            updates["notes"] = (updates.get("notes", fold.notes) + "; " + note).strip("; ")

    # ---- test tail (final holdout only): record actual dependency
    if fold.test_start and fold.test_end:
        last_t, trimmed_t, unavailable_t = lab.last_fully_available(fold.test_start, fold.test_end)
        if last_t is None:
            raise LabelDependencyError(f"fold {fold.fold_id}: no test label is available")
        av_t = lab.availability(last_t)
        # actual dependency end of the last *usable* label
        dep_end = av_t
        out["test"] = {
            "nominal_test_end": fold.test_end,
            "effective_test_end": last_t.isoformat(),
            "last_label_available": _fmt(dep_end),
            "depends_beyond_nominal_test_end": bool(dep_end and dep_end > as_date(fold.test_end)),
            "trimmed_dates": [d.isoformat() for d in trimmed_t],
            "tail_labels_unavailable": [d.isoformat() for d in unavailable_t],
            "n_tail_labels_unavailable_dates": len(unavailable_t),
        }
        if trimmed_t or unavailable_t:
            updates["test_end"] = last_t.isoformat()
            note = (f"label isolation trimmed test_end {fold.test_end}->{last_t.isoformat()} "
                    f"({len(trimmed_t)} dates need later prices, "
                    f"{len(unavailable_t)} unavailable at calendar end)")
            updates["notes"] = (updates.get("notes", fold.notes) + "; " + note).strip("; ")

    out["ok"] = True
    out["trimmed_fold"] = fold.model_copy(update=updates) if updates else fold
    return out


def validate_and_isolate(
    folds: Sequence[FoldSpec],
    holdout: FoldSpec | None,
    calendar: Sequence[str | date | datetime],
    label_expression: str,
    label_window: dict[str, Any],
) -> tuple[list[FoldSpec], FoldSpec | None, dict[str, Any]]:
    """Trim every label segment and verify cross-partition ordering.

    Raises :class:`LabelDependencyError` on any unresolved leak.  Returns
    ``(adjusted_folds, adjusted_holdout, report)``.
    """
    cal = normalise_calendar(calendar)
    offset = resolve_future_offset(label_expression, label_window)
    if offset < 1:
        raise LabelDependencyError(f"label future offset must be >= 1 trading day, got {offset}")
    lab = LabelCalendar(cal, offset, label_expression)

    report: dict[str, Any] = {
        "label_expression": label_expression,
        "future_offset_trading_days": offset,
        "calendar_start": cal[0].isoformat(),
        "calendar_end": cal[-1].isoformat(),
        "folds": {},
    }
    adjusted: list[FoldSpec] = []
    for f in folds:
        part = "confirmation" if getattr(f, "role", "search") == "confirmation" else "discovery"
        r = isolation_report(f, lab, partition=part)
        report["folds"][f.fold_id] = {k: v for k, v in r.items() if k != "trimmed_fold"}
        adjusted.append(r.get("trimmed_fold", f))

    adjusted_holdout = None
    if holdout is not None:
        r = isolation_report(holdout, lab, partition="holdout")
        report["folds"][holdout.fold_id] = {k: v for k, v in r.items() if k != "trimmed_fold"}
        adjusted_holdout = r.get("trimmed_fold", holdout)
        # holdout.valid (early stopping) labels must not reach into test
        hv = adjusted_holdout
        if hv.valid_end and hv.test_start:
            av = lab.availability(hv.valid_end)
            if av is None or av >= as_date(hv.test_start):
                raise LabelDependencyError(
                    f"holdout {hv.fold_id}: early-stopping label at {hv.valid_end} becomes "
                    f"available at {_fmt(av)}, on/after test_start {hv.test_start}"
                )

    # ---- cross-partition ordering with label-aware dates
    search_valid = [(f.fold_id, as_date(f.valid_end)) for f in adjusted if f.valid_end]
    if adjusted_holdout and adjusted_holdout.test_start:
        test_start = as_date(adjusted_holdout.test_start)
        for fid, vend in search_valid:
            if vend >= test_start:
                raise LabelDependencyError(
                    f"fold {fid} validation ends {vend} on/after holdout test_start {test_start}"
                )
    # discovery must precede confirmation (no later screen -> earlier "confirmation")
    discovery = [(f.fold_id, as_date(f.valid_end)) for f in adjusted
                 if getattr(f, "role", "search") == "search" and f.valid_end]
    confirmation = [(f.fold_id, as_date(f.valid_start)) for f in adjusted
                    if getattr(f, "role", "search") == "confirmation" and f.valid_start]
    if confirmation and discovery:
        last_discovery_end = max(d for _, d in discovery)
        for fid, vstart in confirmation:
            if vstart <= last_discovery_end:
                raise LabelDependencyError(
                    f"confirmation fold {fid} starts {vstart} before discovery ends "
                    f"{last_discovery_end}; independent confirmation must be later data"
                )
    report["ok"] = True
    return adjusted, adjusted_holdout, report


def count_label_leak_samples(
    calendar: Sequence[str | date | datetime],
    instruments_path: str,
    start,
    end,
    label_expression: str,
    label_window: dict[str, Any],
    *,
    boundary: str | date | None = None,
) -> dict[str, Any]:
    """Count (instrument, sample-date) labels in [start,end] that need data after ``boundary``.

    Instrument membership is read from a qlib ``instruments/<market>.txt`` file
    (tab-separated symbol, start, end).  This is the concrete number the V1.1
    audit asks for (e.g. the 2023 year-end boundary), not a natural-day guess.
    """
    cal = normalise_calendar(calendar)
    offset = resolve_future_offset(label_expression, label_window)
    lab = LabelCalendar(cal, offset, label_expression)
    b = as_date(boundary) if boundary else as_date(end)
    days = lab.days_between(start, end)
    leaked_days = [d for d in days if (av := lab.availability(d)) is not None and av > b]
    members: dict[str, set[str]] = {d.isoformat(): set() for d in days}
    path = None
    try:
        from pathlib import Path
        path = Path(instruments_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                sym, s, e = parts[0], parts[1], parts[2]
                for d in days:
                    ds = d.isoformat()
                    if s <= ds <= e:
                        members[ds].add(sym)
    except Exception:
        path = None
    per_date = {}
    total = 0
    for d in leaked_days:
        ds = d.isoformat()
        n = len(members.get(ds, ()))
        per_date[ds] = {"n_instruments": n, "label_available": _fmt(lab.availability(d))}
        total += n
    return {
        "start": as_date(start).isoformat(),
        "end": as_date(end).isoformat(),
        "boundary": b.isoformat(),
        "future_offset_trading_days": offset,
        "leaked_dates": [d.isoformat() for d in leaked_days],
        "per_date": per_date,
        "n_leaked_sample_dates": total,
        "instruments_source": str(path) if path else None,
        "instrument_membership_known": bool(path and path.exists()),
    }
