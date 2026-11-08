"""Label-dependency isolation on the real trading calendar.

The V1.0 code treated ``Ref($close, -2)`` as a two-natural-day offset and kept
the train/validation tails, so labels at 2023-12-28/29 depended on prices from
2024-01-02/03 -- i.e. on data that belongs to the frozen holdout.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from predicatelab.core.label_window import (
    LabelDependencyError,
    LabelCalendar,
    count_label_leak_samples,
    resolve_future_offset,
    validate_and_isolate,
)
from predicatelab.core.protocol import validate_label_isolation
from predicatelab.core.schema import FoldSpec

LABEL = "Ref($close, -2) / Ref($close, -1) - 1"
WINDOW = {"uses_future_days": 2}


def _trading_days(start: date, end: date) -> list[str]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d != date(2024, 1, 1):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


CAL = _trading_days(date(2023, 12, 1), date(2024, 1, 31))


def _fold(**kw):
    base = dict(
        fold_id="f", kind="time", train_start="2023-12-01", train_end="2023-12-26",
        valid_start="2023-12-27", valid_end="2023-12-29", purge_days=2, embargo_days=0,
    )
    base.update(kw)
    return FoldSpec(**base)


def test_year_end_validation_tail_is_trimmed_on_calendar_not_natural_days():
    adjusted, holdout, report = validate_and_isolate(
        [_fold()], None, CAL, LABEL, WINDOW,
    )
    v = report["folds"]["f"]["valid"]
    # 2023-12-28 needs 2024-01-02 and 2023-12-29 needs 2024-01-03.
    assert v["trimmed_dates"] == ["2023-12-28", "2023-12-29"]
    assert adjusted[0].valid_end == "2023-12-27"
    assert adjusted[0].valid_end < "2023-12-29"


def test_train_tail_is_trimmed_to_its_last_available_label():
    fold = _fold(train_end="2023-12-27", valid_start="2023-12-29",
                 valid_end="2024-01-05")
    adjusted, _, report = validate_and_isolate([fold], None, CAL, LABEL, WINDOW)
    # 2023-12-25 is the last date whose label (available 2023-12-27) is
    # strictly before valid_start.  The *effective* fold must carry that date,
    # not only report it.
    assert report["folds"]["f"]["train"]["effective_train_end"] == "2023-12-25"
    assert adjusted[0].train_end == "2023-12-25"


def test_holdout_early_stopping_tail_is_trimmed_before_test():
    holdout = FoldSpec(
        fold_id="h", kind="time", role="holdout",
        train_start="2023-12-01", train_end="2023-12-15",
        valid_start="2023-12-18", valid_end="2023-12-29",
        test_start="2024-01-02", test_end="2024-01-31", purge_days=2,
    )
    adjusted, adjusted_holdout, report = validate_and_isolate(
        [], holdout, CAL, LABEL, WINDOW,
    )
    assert adjusted_holdout.valid_end == "2023-12-27"
    assert report["folds"]["h"]["valid"]["n_trimmed_dates"] >= 2
    # Every early-stopping label is available strictly before test_start.
    cal = LabelCalendar([date.fromisoformat(d) for d in CAL], 2, LABEL)
    av = cal.availability(adjusted_holdout.valid_end)
    assert av is not None and av < date(2024, 1, 2)


def test_label_expression_and_declared_window_must_agree():
    with pytest.raises(LabelDependencyError, match="declares 1"):
        resolve_future_offset(LABEL, {"uses_future_days": 1})
    with pytest.raises(LabelDependencyError, match="cannot determine"):
        resolve_future_offset("close", {})
    assert resolve_future_offset(LABEL, {}) == 2


def test_validation_that_reaches_holdout_is_a_hard_error():
    fold = _fold(valid_end="2024-01-05")
    holdout = FoldSpec(
        fold_id="h", kind="time", role="holdout",
        train_start="2023-12-01", train_end="2023-12-15",
        valid_start="2023-12-18", valid_end="2023-12-31",
        test_start="2024-01-02", test_end="2024-01-31", purge_days=2,
    )
    with pytest.raises(LabelDependencyError, match="on/after holdout test_start"):
        validate_label_isolation([fold], [], holdout, CAL, LABEL, WINDOW)


QLIB_ROOT = next(
    (
        p for p in (
            Path("/home/zhengzengrong/.qlib/qlib_data/cn_data"),
            Path("/home/dmz-ai/zhengzengrong/FinTSB-main/qlib_data/cn_data"),
        )
        if (p / "calendars" / "day.txt").exists()
        and (p / "instruments" / "csi300.txt").exists()
    ),
    Path("/home/zhengzengrong/.qlib/qlib_data/cn_data"),
)


@pytest.mark.skipif(
    not (QLIB_ROOT / "calendars" / "day.txt").exists()
    or not (QLIB_ROOT / "instruments" / "csi300.txt").exists(),
    reason="real qlib calendar/instruments not available",
)
def test_2023_year_end_boundary_leak_count_is_exact():
    calendar = [ln.strip() for ln in
                (QLIB_ROOT / "calendars" / "day.txt").read_text().splitlines() if ln.strip()]
    report = count_label_leak_samples(
        calendar, str(QLIB_ROOT / "instruments" / "csi300.txt"),
        "2023-12-25", "2023-12-29", LABEL, WINDOW, boundary="2024-01-01",
    )
    assert report["instrument_membership_known"] is True
    assert report["leaked_dates"] == ["2023-12-28", "2023-12-29"]
    assert report["per_date"]["2023-12-28"]["n_instruments"] == 300
    assert report["per_date"]["2023-12-29"]["n_instruments"] == 300
    assert report["n_leaked_sample_dates"] == 600
