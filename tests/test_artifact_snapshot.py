"""Artifact taxonomy and lightweight snapshot packaging.

The snapshot must separate small, citable output (results/logs/figures/code)
from heavy rebuildable output (models/data), never touch the raw run, and
carry enough change information to reproduce the state.
"""

import hashlib
import json
from pathlib import Path

from predicatelab.core.artifacts import (
    build_run_snapshot,
    build_workspace_snapshot,
    classify_artifact,
    collect_artifacts,
    run_facts,
    summarize_entries,
)


def _write(path: Path, content: str | bytes = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _make_run(root: Path, run_id: str = "run_20260101_000000") -> Path:
    run = root / "exp_a" / run_id
    _write(run / "context.json", '{"run_id": "x"}')
    _write(run / "report.json", json.dumps({
        "run_id": run_id, "domain": "tsil", "run_status": "search_complete",
        "evidence_counts": {"validated": 0, "insufficient_evidence": 2},
        "rounds": [{"round_index": 0, "status": "completed", "n_proposed": 2,
                    "n_validated": 0, "n_insufficient": 2}],
        "final_evals": [],
    }))
    _write(run / "report.md", "# report\n")
    _write(run / "evidence.jsonl", '{"evidence_id": "ev_1"}\n')
    trial = run / "trials/full/f2020/cand_x/candidate/seed_2025/abcd1234"
    _write(trial / "result.json", json.dumps({"status": "success", "metrics": {}}))
    _write(trial / "predictions.npz", b"PK\x03\x04not-really-npz")
    _write(trial / "worker.log", "epoch 1 ic=0.01\n")
    _write(trial / "models/model.pt", b"weights")
    _write(run / "_manifests/data.json", json.dumps({"payload": "x" * 100}))
    _write(run / "figures/effects_forest.png", b"\x89PNG\r\n\x1a\n")
    _write(run / "analysis.py", "print('analysis')\n")
    _write(run / "predicatelab.sqlite", b"SQLite format 3\x00")
    _write(run / "predicatelab.sqlite-shm", b"shm")
    return run


def _verify_checksums(snapshot: Path) -> None:
    lines = (snapshot / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        digest, rel = line.split("  ", 1)
        target = snapshot / rel
        assert target.is_file(), rel
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, rel


def test_classification_by_directory_and_suffix():
    assert classify_artifact("run/trials/x/worker.log") == "logs"
    assert classify_artifact("run/trials/x/models/model.pt") == "models"
    assert classify_artifact("run/trials/x/predictions.npz") == "data"
    assert classify_artifact("run/_manifests/content.json") == "data"
    assert classify_artifact("run/report.json") == "results"
    assert classify_artifact("run/figures/plot.png") == "figures"
    assert classify_artifact("run/analysis.py") == "code"
    assert classify_artifact("run/predicatelab.sqlite-shm") is None
    assert classify_artifact("run/__pycache__/x.pyc") is None


def test_collect_skips_transient_files(tmp_path):
    run = _make_run(tmp_path)
    entries = collect_artifacts(run)
    relpaths = {entry.relpath for entry in entries}
    assert "predicatelab.sqlite-shm" not in relpaths
    kinds = {entry.relpath: entry.kind for entry in entries}
    assert kinds["predicatelab.sqlite"] == "data"
    assert kinds["trials/full/f2020/cand_x/candidate/seed_2025/abcd1234/models/model.pt"] == "models"


def test_default_snapshot_is_lightweight_and_checksummed(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "published" / "exp_a" / run.name
    build_run_snapshot(run, out)

    # Included categories.
    assert (out / "results" / "report.json").is_file()
    assert (out / "results" / "evidence.jsonl").is_file()
    assert (out / "logs" / "trials/full/f2020/cand_x/candidate/seed_2025/abcd1234/worker.log").is_file()
    assert (out / "figures" / "effects_forest.png").is_file()
    assert (out / "code" / "analysis.py").is_file()
    assert (out / "results" / "trials_summary.csv").is_file()
    # Per-trial detail JSON is summarised, not copied.
    assert not (out / "results" / "trials").exists()
    # Heavy categories excluded by default.
    assert not (out / "models").exists()
    assert not (out / "data").exists()

    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "predicatelab.snapshot/1"
    assert manifest["scope"] == "run"
    assert manifest["kinds"]["results"]["files"] >= 3
    assert manifest["excluded_kinds"]["models"]["files"] == 1
    assert manifest["excluded_kinds"]["data"]["files"] >= 2
    assert manifest["excluded_trial_details"]["files"] == 1
    assert manifest["run_facts"]["run_status"] == "search_complete"

    assert (out / "changes" / "git_status.txt").is_file() or manifest["change_info"]["dir"] == "changes"
    assert (out / "changes" / "environment.json").is_file()
    assert (out / "README.md").is_file()
    _verify_checksums(out)

    # The raw run is untouched.
    assert (run / "predicatelab.sqlite").is_file()
    assert (run / "trials/full/f2020/cand_x/candidate/seed_2025/abcd1234/models/model.pt").is_file()


def test_snapshot_can_opt_in_to_trial_details(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "detailed"
    build_run_snapshot(run, out, include_trial_details=True)
    assert (out / "results/trials/full/f2020/cand_x/candidate/seed_2025/abcd1234/result.json").is_file()
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["excluded_trial_details"] == {}
    _verify_checksums(out)


def test_snapshot_can_opt_in_to_models_and_data(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "full"
    build_run_snapshot(run, out, include_models=True, include_data=True)
    assert (out / "models/trials/full/f2020/cand_x/candidate/seed_2025/abcd1234/model.pt").is_file()
    assert (out / "data/trials/full/f2020/cand_x/candidate/seed_2025/abcd1234/predictions.npz").is_file()
    assert (out / "data/predicatelab.sqlite").is_file()
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["excluded_kinds"] == {}
    assert manifest["kinds"]["models"]["files"] == 1
    _verify_checksums(out)


def test_workspace_snapshot_groups_experiments_and_loose_files(tmp_path):
    root = tmp_path / "artifacts"
    _make_run(root, "run_1")
    _make_run(root, "run_2")
    _write(root / "exp_b" / "run_3" / "report.json", json.dumps({"run_id": "run_3"}))
    _write(root / "probe_bank_report.json", json.dumps({"ok": True}))
    _write(root / "_probe_batches" / "x.npz", b"data")

    out = build_workspace_snapshot(root, tmp_path / "published")
    assert (out / "INDEX.md").is_file()
    assert (out / "exp_a" / "README.md").is_file()
    assert (out / "exp_a" / "results" / "run_1" / "report.json").is_file()
    assert (out / "exp_b" / "results" / "run_3" / "report.json").is_file()
    assert (out / "exp_a" / "results" / "trials_summary.csv").is_file()
    assert (out / "_root" / "results" / "probe_bank_report.json").is_file()
    # Underscored scratch dirs are data-only and skipped by default.
    assert not (out / "_probe_batches").exists()
    assert "exp_a" in (out / "INDEX.md").read_text(encoding="utf-8")
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["scope"] == "workspace"
    assert {row["experiment"] for row in manifest["experiments"]} >= {"exp_a", "exp_b", "_root"}
    # Excluded kinds must not be advertised as published in the index/README.
    exp_a = next(row for row in manifest["experiments"] if row["experiment"] == "exp_a")
    assert set(exp_a["kinds"]) <= {"results", "logs", "figures", "code"}
    assert "data" not in exp_a["kinds"]
    readme = (out / "exp_a" / "README.md").read_text(encoding="utf-8")
    assert "| `data` |" in readme and "| 0 |" in readme
    _verify_checksums(out)


def test_run_facts_tolerates_missing_or_broken_report(tmp_path):
    run = tmp_path / "run_x"
    _write(run / "report.json", "{not json")
    facts = run_facts(run)
    assert facts["run_id"] == "run_x"
    assert "run_status" not in facts


def test_summarize_counts_every_kind(tmp_path):
    run = _make_run(tmp_path)
    summary = summarize_entries(collect_artifacts(run))
    for kind in ("results", "logs", "figures", "models", "data", "code"):
        assert summary[kind]["files"] >= 1, kind
