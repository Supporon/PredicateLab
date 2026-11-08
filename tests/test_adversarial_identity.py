"""Cache identity, content manifests and corrupt-cache rejection.

V1.0 cached a trial on ``(candidate, fold, protocol, config)`` only, so a
changed data snapshot, label, phase or code revision could silently reuse
numbers computed under different conditions.  These tests pin the V1.1
identity fields and the "reject, never reuse" rule.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from predicatelab.core.adapter import TrialRequest
from predicatelab.core.hashing import sha256_file
from predicatelab.core.identity import (
    fold_content_hash,
    full_content_manifest,
    label_spec_hash,
)
from predicatelab.core.schema import DataContract, FoldSpec
from predicatelab.core.store import StateStore, result_digest
from predicatelab.domains.tabular.adapter import TabularAdapter


def _adapter():
    return TabularAdapter({
        "training": {"seed": 2025},
        "protocol": {"cv_splits": 5, "cv_repeats": 2},
        "seeds": [2025],
    })


def _request(adapter):
    fold = adapter.build_protocol({}).folds[0]
    return TrialRequest(
        candidate_id="cand", variant="candidate", descriptor={"op": "product", "a": 0, "b": 1},
        fold=fold, seed=2025, training_config={"lr": 0.1}, workdir="",
        protocol_hash="ph", config_hash="",
        phase="discovery", snapshot_hash="snap-1", data_manifest_hash="data-1",
        label_spec_hash="label-1", preprocessing_version="prep-1",
        model_spec={"model": "lr"}, code_identity={"commit": "abc", "dirty_fingerprint": "d1"},
        environment={"numpy": "1.0"}, label_expression="y", label_offset=0,
    )


@pytest.mark.parametrize("field,value", [
    ("phase", "confirmation"),
    ("seed", 2026),
    ("snapshot_hash", "snap-2"),
    ("data_manifest_hash", "data-2"),
    ("label_spec_hash", "label-2"),
    ("label_expression", "y2"),
    ("preprocessing_version", "prep-2"),
    ("model_spec", {"model": "other"}),
    ("code_identity", {"commit": "def", "dirty_fingerprint": "d2"}),
    ("environment", {"numpy": "2.0"}),
])
def test_trial_hash_covers_every_scientific_identity_field(field, value):
    adapter = _adapter()
    base = _request(adapter)
    assert adapter.trial_config_hash(base) != adapter.trial_config_hash(
        replace(base, **{field: value})
    )


def test_trial_hash_ignores_attempt_and_workdir_only():
    adapter = _adapter()
    base = _request(adapter)
    h = adapter.trial_config_hash(base)
    assert adapter.trial_config_hash(replace(base, attempt=7, workdir="/tmp/retry")) == h


def test_fold_content_hash_tracks_dates_and_role():
    fold = FoldSpec(fold_id="f", kind="time", train_start="2020-01-01",
                    train_end="2020-12-31", valid_start="2021-01-01",
                    valid_end="2021-12-31", purge_days=2)
    assert fold_content_hash(fold) == fold_content_hash(fold.model_copy())
    assert fold_content_hash(fold) != fold_content_hash(fold.model_copy(update={"valid_end": "2021-11-30"}))
    assert fold_content_hash(fold) != fold_content_hash(fold.model_copy(update={"role": "confirmation"}))


def test_label_spec_hash_tracks_expression_and_window():
    a = DataContract(dataset_id="d", domain="tsil", description="d",
                     label_definition="Ref($close,-2)",
                     label_window={"uses_future_days": 2}, entity_key="i", time_key="dt")
    b = a.model_copy(update={"label_window": {"uses_future_days": 1}})
    assert label_spec_hash(a) != label_spec_hash(b)


def test_content_manifest_detects_changed_file(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.bin").write_bytes(b"aaaa")
    (root / "b.bin").write_bytes(b"bbbb")
    cache = tmp_path / "manifest.json"
    first = full_content_manifest(root, ["a.bin", "b.bin"], cache_path=cache)
    second = full_content_manifest(root, ["a.bin", "b.bin"], cache_path=cache)
    assert first["manifest_hash"] == second["manifest_hash"]
    assert second["n_rehashed"] == 0

    (root / "a.bin").write_bytes(b"AAAAA")  # different content and size
    third = full_content_manifest(root, ["a.bin", "b.bin"], cache_path=cache)
    assert third["manifest_hash"] != first["manifest_hash"]
    assert third["n_rehashed"] == 1


def test_tampered_content_manifest_cache_is_rejected(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.bin").write_bytes(b"aaaa")
    cache = tmp_path / "manifest.json"
    first = full_content_manifest(root, ["a.bin"], cache_path=cache)

    payload = json.loads(cache.read_text())
    payload["files"][0]["sha256"] = "0" * 64  # corrupt leaf, keep size/mtime
    cache.write_text(json.dumps(payload))

    rebuilt = full_content_manifest(root, ["a.bin"], cache_path=cache)
    good = sha256_file(root / "a.bin")
    assert rebuilt["files"][0]["sha256"] == good
    assert rebuilt["manifest_hash"] == first["manifest_hash"]
    assert rebuilt["n_rehashed"] == 1


def _trial_with(tmp_path, status="success", metrics=None, series=None, result_hash=None):
    from predicatelab.core.schema import TrialRecord

    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps({"idx": [1], "pred": [0.5]}), encoding="utf-8")
    trial = TrialRecord(
        candidate_id="cand", fold_id="f", protocol_hash="ph", config_hash="ch",
        status=status, metrics=metrics if metrics is not None else {"ic": 0.1},
        metric_series=series if series is not None else {"ic": {"2020-01-02": 0.1}},
        predictions_path=str(pred), predictions_sha256=sha256_file(pred),
    )
    trial.result_hash = result_hash if result_hash is not None else result_digest({
        "status": trial.status, "metrics": trial.metrics,
        "metric_series": trial.metric_series, "fitted_params": trial.fitted_params,
        "predictions_path": trial.predictions_path,
        "predictions_sha256": trial.predictions_sha256, "null_paths": trial.null_paths,
    })
    return trial


def test_corrupt_cached_trial_is_marked_corrupt_and_not_returned(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        store.record_trial(_trial_with(tmp_path), run_id="r")
        with store.transaction() as c:
            c.execute("UPDATE trials SET metrics='{\"ic\": 999}' WHERE candidate_id='cand'")
        assert store.get_cached_trial("cand", "f", "ph", "ch") is None
        row = store.get_trial_by_key("cand", "f", "ph", "ch")
        assert row["status"] == "corrupt"
        assert "hash mismatch" in row["error"]


def test_missing_predictions_file_invalidates_cache(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        trial = _trial_with(tmp_path)
        store.record_trial(trial, run_id="r")
        Path(trial.predictions_path).unlink()
        assert store.get_cached_trial("cand", "f", "ph", "ch") is None


def test_failed_trial_is_never_returned_as_cached_success(tmp_path):
    with StateStore(tmp_path / "s.sqlite") as store:
        store.record_trial(_trial_with(tmp_path, status="failed"), run_id="r")
        assert store.get_cached_trial("cand", "f", "ph", "ch") is None
