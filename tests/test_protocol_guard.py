"""Time-fold construction, overlap validation and the holdout leak guard."""

import pytest

from predicatelab.core.protocol import (
    HoldoutGuard,
    ProtocolViolation,
    effective_time_fold,
    validate_protocol,
)
from predicatelab.core.schema import FoldSpec

CAL = [f"2020-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15)]


def _fold(**kw):
    base = dict(
        fold_id="f",
        train_start="2020-01-01",
        train_end="2020-06-30",
        valid_start="2020-07-01",
        valid_end="2020-09-30",
        purge_days=2,
        embargo_days=3,
    )
    base.update(kw)
    return FoldSpec(**base)


def test_purge_and_embargo_shrink_the_effective_windows():
    fold = effective_time_fold(_fold(), CAL)
    train_days = [d for d in CAL if "2020-01-01" <= d <= "2020-06-30"]
    valid_days = [d for d in CAL if "2020-07-01" <= d <= "2020-09-30"]
    assert fold.train_end == train_days[-3]
    assert fold.valid_start == valid_days[3]


def test_validate_protocol_rejects_overlapping_validation_periods():
    a = _fold(fold_id="a", valid_start="2020-07-01", valid_end="2020-12-31")
    b = _fold(fold_id="b", train_start="2020-01-01", train_end="2020-09-30",
              valid_start="2020-10-01", valid_end="2020-12-31")
    with pytest.raises(ProtocolViolation, match="overlap"):
        validate_protocol([a, b], None)


def test_validate_protocol_rejects_validation_reaching_into_holdout():
    fold = _fold(valid_end="2024-06-30")
    holdout = FoldSpec(
        fold_id="h", role="holdout",
        train_start="2018-01-01", train_end="2022-12-31",
        valid_start="2023-01-01", valid_end="2023-12-31",
        test_start="2024-01-01", test_end="2025-08-31",
    )
    with pytest.raises(ProtocolViolation, match="holdout"):
        validate_protocol([fold], holdout)


def test_holdout_guard_blocks_search_until_frozen():
    search = _fold(fold_id="search")
    holdout = FoldSpec(
        fold_id="holdout", role="holdout",
        train_start="2018-01-01", train_end="2022-12-31",
        valid_start="2023-01-01", valid_end="2023-12-31",
        test_start="2024-01-01", test_end="2025-08-31",
    )
    guard = HoldoutGuard.from_folds([search], holdout)

    # search folds are accessible to search
    guard.assert_fold_accessible("search", "search")

    # the holdout is not accessible to search ...
    with pytest.raises(ProtocolViolation, match="holdout"):
        guard.assert_fold_accessible("holdout", "search")

    # ... nor to finalize before a manifest exists
    with pytest.raises(ProtocolViolation, match="without a frozen"):
        guard.assert_fold_accessible("holdout", "finalize")

    unlocked = guard.unlock("manifest-hash")
    unlocked.assert_fold_accessible("holdout", "finalize")

    # finalize must not silently read a search fold
    with pytest.raises(ProtocolViolation, match="search fold"):
        unlocked.assert_fold_accessible("search", "finalize")
