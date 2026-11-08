"""Adversarial tests for the global, one-time holdout ledger.

These reproduce the V1.0 failure mode in which a holdout could be read again by
changing the run id or the candidate.  The ledger must be keyed by data
identity, claims must be atomic, a crash must count as consumption, and only an
exact retry of the same frozen manifest may read the same holdout again.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from predicatelab.core.holdout import HoldoutConsumed
from predicatelab.core.store import StateStore


def _kwargs(**over):
    kw = dict(
        key="holdout-key-1",
        study_id="task-a",
        run_id="run-a",
        candidate_id="cand-a",
        protocol_hash="protocol-a",
        snapshot_hash="snapshot-a",
        dataset_id="dataset-a",
        label_spec_hash="label-a",
        test_start="2024-01-01",
        test_end="2025-08-31",
        manifest_hash="manifest-a",
    )
    kw.update(over)
    return kw


def test_holdout_claim_is_global_across_run_and_candidate(tmp_path):
    db = tmp_path / "s.sqlite"
    with StateStore(db) as first, StateStore(db) as second:
        row = first.claim_holdout(**_kwargs())
        assert row["status"] == "claimed"

        # A different run/candidate for the same data identity is refused even
        # through a second connection.
        with pytest.raises(HoldoutConsumed):
            second.claim_holdout(**_kwargs(run_id="run-b", candidate_id="cand-b"))
        rows = second.list_holdout_access("holdout-key-1")
        assert len(rows) == 1
        assert rows[0]["candidate_id"] == "cand-a"


def test_completed_retry_returns_saved_result_without_new_claim(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        row = store.claim_holdout(**_kwargs())
        store.mark_holdout(row["access_id"], "completed", run_id="run-a",
                           candidate_id="cand-a", note="done")
        again = store.claim_holdout(**_kwargs())
        assert again["already_completed"] is True
        assert again["access_id"] == row["access_id"]
        assert len(store.list_holdout_access("holdout-key-1")) == 1
        assert [e["event"] for e in store.holdout_events(row["access_id"])] == [
            "claimed", "completed",
        ]


def test_crash_counts_as_consumed_and_only_same_retry_is_allowed(tmp_path):
    db = tmp_path / "s.sqlite"
    with StateStore(db) as store:
        row = store.claim_holdout(**_kwargs())
        store.mark_holdout(row["access_id"], "running", run_id="run-a",
                           candidate_id="cand-a", note="worker started")
        # Simulate the process dying here; a new store must see it as consumed.
    with StateStore(db) as store:
        with pytest.raises(HoldoutConsumed):
            store.claim_holdout(**_kwargs(run_id="run-c", candidate_id="cand-c"))
        retry = store.claim_holdout(**_kwargs())
        assert retry["retry"] is True
        assert retry["access_id"] == row["access_id"]
        assert store.get_holdout_access(row["access_id"])["status"] == "claimed"


def test_failed_holdout_is_not_reclaimable(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        row = store.claim_holdout(**_kwargs())
        store.mark_holdout(row["access_id"], "failed", note="worker error")
        with pytest.raises(HoldoutConsumed):
            store.claim_holdout(**_kwargs(run_id="run-d", candidate_id="cand-d"))


def test_exploratory_reuse_is_recorded_but_never_confirmatory(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        row = store.claim_holdout(**_kwargs())
        store.mark_holdout(row["access_id"], "completed", note="done")
        reuse_id = store.record_exploratory_reuse(
            key="holdout-key-1", study_id="task-b", run_id="run-b",
            candidate_id="cand-b", protocol_hash="protocol-b",
            snapshot_hash="snapshot-a", dataset_id="dataset-a",
            label_spec_hash="label-a", test_start="2024-01-01",
            test_end="2025-08-31", manifest_hash="manifest-b",
        )
        assert reuse_id.startswith("hoa_expl_")
        reuse = store.get_holdout_access(reuse_id)
        assert reuse["purpose"] == "exploratory_reuse"
        assert reuse["status"] == "exploratory_reuse"
        # The exploratory row does not consume the confirmatory slot, but the
        # confirmatory slot is already taken, so a fresh study is still blocked.
        with pytest.raises(HoldoutConsumed):
            store.claim_holdout(**_kwargs(run_id="run-e", candidate_id="cand-e"))
        assert len(store.list_holdout_access("holdout-key-1")) == 2


def test_concurrent_claims_have_exactly_one_winner(tmp_path):
    db = tmp_path / "s.sqlite"
    StateStore(db).close()  # create schema before racing

    results: list[str] = []

    def attempt(i: int) -> None:
        with StateStore(db) as store:
            try:
                store.claim_holdout(**_kwargs(run_id=f"run-{i}", candidate_id=f"cand-{i}"))
                results.append("won")
            except HoldoutConsumed:
                results.append("blocked")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(attempt, (1, 2)))

    assert sorted(results) == ["blocked", "won"]
    with StateStore(db) as store:
        assert len(store.list_holdout_access("holdout-key-1")) == 1


def _insert_legacy_row(store, *, dataset_id, test_start, test_end, origin="backfill_legacy"):
    with store.transaction() as c:
        c.execute(
            """INSERT INTO holdout_access(access_id, holdout_key, study_id, run_id,
                   candidate_id, protocol_hash, snapshot_hash, dataset_id, label_spec_hash,
                   test_start, test_end, purpose, status, origin, manifest_hash, details,
                   created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"hoa_{origin}_{dataset_id}", "legacy-key-" + dataset_id, "t", "r", "c",
             "p", "snap", dataset_id, "label", test_start, test_end, "legacy_final",
             "completed", origin, "m", "{}", "2026-01-01", "2026-01-01"),
        )


def test_overlapping_legacy_window_blocks_a_new_identity(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        # V1.0 used a sampled snapshot manifest and had no data_manifest_hash,
        # so a V1.1 key can never equal the historical key.  The date window
        # guard must still stop the same holdout from being read again.
        _insert_legacy_row(store, dataset_id="qlib-cn-csi300",
                           test_start="2024-01-01", test_end="2025-08-31")
        with pytest.raises(HoldoutConsumed, match="overlaps"):
            store.claim_holdout(**_kwargs(
                dataset_id="qlib-cn-csi300", key="brand-new-key",
                test_start="2024-01-01", test_end="2025-08-31",
                run_id="new-run", candidate_id="new-cand"))


def test_legacy_row_with_unknown_window_fails_closed(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        _insert_legacy_row(store, dataset_id="qlib-cn-csi300",
                           test_start="legacy_unknown", test_end="legacy_unknown")
        with pytest.raises(HoldoutConsumed, match="overlaps"):
            store.claim_holdout(**_kwargs(
                dataset_id="qlib-cn-csi300", key="another-key",
                run_id="new-run", candidate_id="new-cand"))


def test_non_overlapping_window_on_the_same_dataset_is_allowed(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        _insert_legacy_row(store, dataset_id="qlib-cn-csi300",
                           test_start="2022-01-01", test_end="2022-12-31")
        row = store.claim_holdout(**_kwargs(
            dataset_id="qlib-cn-csi300", key="fresh-key",
            test_start="2026-01-01", test_end="2026-12-31",
            run_id="new-run", candidate_id="new-cand"))
        assert row["status"] == "claimed"


def test_holdout_key_ignores_study_metadata_but_tracks_data_window(tmp_path):
    from predicatelab.core.holdout import holdout_key

    base = dict(dataset_id="d", snapshot_hash="s", data_manifest_hash="m",
                label_spec_hash="l", test_start="2024-01-01", test_end="2025-08-31")
    assert holdout_key(**base) == holdout_key(**base)
    assert holdout_key(**base) != holdout_key(**{**base, "test_end": "2025-08-30"})
    assert holdout_key(**base) != holdout_key(**{**base, "label_spec_hash": "l2"})
    assert holdout_key(**base) != holdout_key(**{**base, "data_manifest_hash": "m2"})


def test_cross_db_concurrent_claims_have_exactly_one_winner(tmp_path):
    """P0-2: Two distinct SQLite DB files sharing a global ledger must never both win."""
    from predicatelab.core.holdout import ConfirmationConsumed

    ledger = tmp_path / "shared_ledger.sqlite"
    db_a = tmp_path / "subdir_a" / "study_a.sqlite"
    db_b = tmp_path / "subdir_b" / "study_b.sqlite"
    StateStore(db_a, global_ledger_path=ledger).close()
    StateStore(db_b, global_ledger_path=ledger).close()

    holdout_results: list[str] = []

    def attempt_holdout(item: tuple[int, str]) -> None:
        idx, db_file = item
        with StateStore(db_file, global_ledger_path=ledger) as store:
            try:
                store.claim_holdout(
                    **_kwargs(
                        key="cross-db-holdout-key",
                        run_id=f"run-{idx}",
                        candidate_id=f"cand-{idx}",
                        manifest_hash=f"manifest-{idx}",
                    )
                )
                holdout_results.append("won")
            except HoldoutConsumed:
                holdout_results.append("blocked")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(attempt_holdout, [(1, str(db_a)), (2, str(db_b))]))

    assert sorted(holdout_results) == ["blocked", "won"]

    confirm_results: list[str] = []

    def attempt_confirm(item: tuple[int, str]) -> None:
        idx, db_file = item
        with StateStore(db_file, global_ledger_path=ledger) as store:
            res = store.claim_confirmation(
                key="cross-db-confirm-key",
                study_id=f"task-{idx}",
                run_id=f"run-{idx}",
                family_id=f"fam-{idx}",
                manifest_hash=f"manifest-{idx}",
                protocol_hash="protocol-a",
                snapshot_hash="snapshot-a",
                dataset_id="dataset-a",
                label_spec_hash="label-a",
                fold_id="f_conf",
                valid_start="2023-01-01",
                valid_end="2023-12-31",
            )
            confirm_results.append("reused" if res.get("reused") else "won")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(attempt_confirm, [(1, str(db_a)), (2, str(db_b))]))

    assert sorted(confirm_results) == ["reused", "won"]


def test_default_ledger_shared_across_subdirectories_and_builtin_csi300_blocked(tmp_path):
    """P0-2: Stores in nested subdirectories share the root ledger and block pre-consumed CSI300 windows."""
    db_sub1 = tmp_path / "artifacts" / "study1" / "state.sqlite"
    db_sub2 = tmp_path / "artifacts_other" / "nested" / "state.sqlite"
    with StateStore(db_sub1) as s1, StateStore(db_sub2) as s2:
        assert s1.global_ledger_path == s2.global_ledger_path
        row = s1.claim_holdout(
            **_kwargs(
                key="sub-shared-key",
                dataset_id="custom-ds",
                test_start="2026-01-01",
                test_end="2026-06-30",
            )
        )
        assert row["status"] == "claimed"
        with pytest.raises(HoldoutConsumed):
            s2.claim_holdout(
                **_kwargs(
                    key="sub-shared-key",
                    dataset_id="custom-ds",
                    test_start="2026-01-01",
                    test_end="2026-06-30",
                    run_id="run-sub2",
                    candidate_id="cand-sub2",
                )
            )
        # Historical CSI300 2024-01-01..2025-08-31 holdout is pre-seeded and blocked even in a brand-new DB
        with pytest.raises(HoldoutConsumed, match="overlaps|already consumed"):
            s2.claim_holdout(
                **_kwargs(
                    key="fresh-csi300-key",
                    dataset_id="qlib-cn-csi300",
                    test_start="2024-01-01",
                    test_end="2025-08-31",
                    run_id="brand-new-csi300-run",
                    candidate_id="brand-new-csi300-cand",
                )
            )
        # Historical CSI300 2023-01-01..2023-12-31 confirmation window is pre-seeded and marked reused
        conf = s2.claim_confirmation(
            key="fresh-csi300-conf-key",
            study_id="study-csi",
            run_id="brand-new-csi300-run",
            family_id="brand-new-family-id",
            manifest_hash="brand-new-manifest-hash",
            protocol_hash="prot-csi",
            snapshot_hash="snap-csi",
            dataset_id="qlib-cn-csi300",
            label_spec_hash="lbl-csi",
            fold_id="f2023",
            valid_start="2023-01-01",
            valid_end="2023-12-31",
        )
        assert conf["reused"] is True

