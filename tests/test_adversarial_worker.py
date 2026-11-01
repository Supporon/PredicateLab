"""Worker-boundary tests: abnormal exits and non-finite results.

A stale ``result.json`` from an earlier attempt must never be read as the
current attempt's output, and a nonzero exit code is authoritative even if the
worker wrote a success-looking file.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from predicatelab.domains.tsil import adapter as tsil_adapter
from predicatelab.domains.tsil import worker


def _job(tmp_path):
    return {
        "mode": "trial",
        "result_path": str(tmp_path / "result.json"),
    }


def test_nonzero_exit_cannot_be_promoted_to_success(tmp_path, monkeypatch):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    job = _job(tmp_path)
    result_path = Path(job["result_path"])
    # An earlier attempt left a success result behind.
    result_path.write_text(json.dumps({"status": "success", "metrics": {"ic": 0.9}}))

    def fake_run(cmd, **kwargs):
        assert not result_path.exists(), "stale result must be removed before the run"
        result_path.write_text(json.dumps({"status": "success", "metrics": {"ic": 0.9}}))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(tsil_adapter.subprocess, "run", fake_run)
    out = tsil_adapter.TSILAdapter._run_worker(None, job, workdir, timeout=5)
    assert out["status"] == "failed"
    assert "rc=1" in out["error"]


def test_missing_result_with_zero_exit_is_failed(tmp_path, monkeypatch):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    monkeypatch.setattr(tsil_adapter.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0))
    out = tsil_adapter.TSILAdapter._run_worker(None, _job(tmp_path), workdir, timeout=5)
    assert out["status"] == "failed"
    assert "no result" in out["error"]


def test_malformed_result_is_failed(tmp_path, monkeypatch):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    job = _job(tmp_path)

    def fake_run(cmd, **kwargs):
        Path(job["result_path"]).write_text("{not json")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tsil_adapter.subprocess, "run", fake_run)
    out = tsil_adapter.TSILAdapter._run_worker(None, job, workdir, timeout=5)
    assert out["status"] == "failed"
    assert "malformed" in out["error"]


def test_nan_result_is_failed(tmp_path, monkeypatch):
    workdir = tmp_path / "wd"
    workdir.mkdir()
    job = _job(tmp_path)

    def fake_run(cmd, **kwargs):
        Path(job["result_path"]).write_text(
            '{"status": "success", "metrics": {"ic": NaN}}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tsil_adapter.subprocess, "run", fake_run)
    out = tsil_adapter.TSILAdapter._run_worker(None, job, workdir, timeout=5)
    assert out["status"] == "failed"
    assert "malformed" in out["error"]


def test_worker_write_never_emits_nan_and_downgrades_success(tmp_path):
    path = tmp_path / "result.json"
    worker._write(str(path), {"status": "success", "metrics": {"ic": float("nan")},
                              "note": "ok"})
    text = path.read_text()
    assert "NaN" not in text
    payload = json.loads(text)
    assert payload["status"] == "failed"
    assert payload["null_paths"] == ["metrics.ic"]
    assert payload["metrics"]["ic"] is None


def test_worker_write_is_atomic_and_readable(tmp_path):
    path = tmp_path / "result.json"
    worker._write(str(path), {"status": "success", "metrics": {"ic": 0.1}})
    payload = json.loads(path.read_text())
    assert payload["status"] == "success"
    assert payload["metrics"]["ic"] == 0.1
    assert not list(tmp_path.glob("*.tmp*"))
