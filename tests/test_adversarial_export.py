"""Evidence-bundle export must never crash on missing predictions.

V1.1 dev end-to-end run found that a trial with an empty ``predictions_path``
made ``export`` copy ``Path("")`` -- the current directory -- and abort.  The
bundle must instead skip unusable prediction paths and record how many were
missing, while still copying and checksumming every prediction file that does
exist.
"""

import hashlib
import json

from predicatelab.core.engine import PredicateLabEngine


def _config(tmp_path, *, save_predictions=False):
    return {
        "run": {
            "name": "export_adversarial",
            "domain": "tabular",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "db": str(tmp_path / "artifacts" / "state.sqlite"),
            "rounds": 1,
            "require_confirmation": False,
        },
        "seeds": [2025],
        "training": {"seed": 2025},
        "protocol": {"cv_splits": 5, "cv_repeats": 1, "discovery_repeats": 1,
                     "confirmation_kind": "resampling"},
        "evaluation": {
            "name": "tabular_cv_logloss", "primary_metric": "log_loss",
            "direction": "lower", "secondary_metrics": ["auc", "accuracy"],
            "paired": True, "multiple_testing": "benjamini-hochberg",
        },
        "search": {
            "generator": "rule",
            "max_candidates_per_round": 2,
            "screen_folds": ["cv_r0_f0"],
            # Screening and validation folds must be disjoint (D-10).
            "validation_folds": ["cv_r0_f1", "cv_r0_f2",
                                 "cv_r0_f3", "cv_r0_f4"],
            "full_validation_top_k": 1,
            "min_screen_effect": -1e9,
            "min_effect": -1.0,
            "min_stability": 0.0,
            "fdr_alpha": 1.0,
            "bootstrap_samples": 100,
            "bootstrap_block": 1,
            "nw_lag": 0,
            "seed": 2025,
            "save_predictions": save_predictions,
        },
    }


def _verify_checksums(bundle):
    for line in (bundle / "checksums.sha256").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        actual = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
        assert actual == digest, rel


def test_export_without_prediction_files_succeeds_and_reports_them(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    engine.run(rounds=1)
    bundle = engine.export(tmp_path / "bundle")

    assert (bundle / "checksums.sha256").is_file()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["predictions"]["n_copied"] == 0
    # Tabular trials currently carry no prediction path; they must be counted
    # as missing rather than crashing the export or being silently ignored.
    assert manifest["predictions"]["n_missing"] > 0
    _verify_checksums(bundle)
    assert not list((bundle / "predictions").iterdir())
    engine.store.close()


def test_export_includes_predictions_with_sample_index_and_labels(tmp_path):
    import numpy as np

    engine = PredicateLabEngine(_config(tmp_path, save_predictions=True))
    run_id = engine.run(rounds=1)
    bundle = engine.export(tmp_path / "bundle")
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())

    n_rows = len(engine.store.list_trials(run_id=run_id))
    assert manifest["predictions"]["n_copied"] == n_rows
    assert manifest["predictions"]["n_missing"] == 0
    assert len(manifest["predictions"]["files"]) == n_rows
    for rel in manifest["predictions"]["files"]:
        with np.load(bundle / rel) as data:
            assert {"sample", "score", "label"} <= set(data.files)
            assert len(data["sample"]) == len(data["score"]) == len(data["label"]) > 0
    _verify_checksums(bundle)
    engine.store.close()


def test_export_skips_directory_paths_and_copies_real_files(tmp_path):
    engine = PredicateLabEngine(_config(tmp_path))
    run_id = engine.run(rounds=1)
    trials = engine.store.list_trials(run_id=run_id)
    assert len(trials) >= 2
    prediction = tmp_path / "predictions.npz"
    prediction.write_bytes(b"fake-npz-bytes")
    with engine.store.transaction() as c:
        c.execute("UPDATE trials SET predictions_path='.' WHERE trial_id=?",
                  (trials[0]["trial_id"],))
        c.execute("UPDATE trials SET predictions_path=? WHERE trial_id=?",
                  (str(prediction), trials[1]["trial_id"]))
        c.execute("UPDATE trials SET predictions_path='' WHERE trial_id=?",
                  (trials[2]["trial_id"],))

    bundle = engine.export(tmp_path / "bundle")
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["predictions"]["n_copied"] == 1
    copied = list((bundle / "predictions").iterdir())
    assert len(copied) == 1
    assert copied[0].read_bytes() == b"fake-npz-bytes"
    # The unusable "." path is skipped, not copied as a directory.
    assert all(p.is_file() for p in copied)
    _verify_checksums(bundle)
    engine.store.close()
