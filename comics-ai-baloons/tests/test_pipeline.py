import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pipeline


def test_stages_list_well_formed():
    names = [s[0] for s in pipeline.STAGES]
    assert len(names) == len(set(names))  # no duplicate stage names
    for name, script, _output in pipeline.STAGES:
        assert (pipeline.SCRIPTS_DIR / script).exists(), f"{script} for stage {name} missing"


def test_run_stage_skips_when_output_exists(tmp_path, monkeypatch):
    output = tmp_path / "out.jsonl"
    output.write_text("existing")

    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a) or subprocess.CompletedProcess(a, 0))

    pipeline.run_stage("fake", "discover.py", output, force=False)
    assert called == []


def test_run_stage_runs_when_output_missing(tmp_path, monkeypatch):
    output = tmp_path / "missing.jsonl"

    called = []

    def fake_run(*a, **k):
        called.append(a)
        return subprocess.CompletedProcess(a, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    pipeline.run_stage("fake", "discover.py", output, force=False)
    assert len(called) == 1


def test_run_stage_force_reruns_even_if_output_exists(tmp_path, monkeypatch):
    output = tmp_path / "out.jsonl"
    output.write_text("existing")

    called = []
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: called.append(a) or subprocess.CompletedProcess(a, 0)
    )
    pipeline.run_stage("fake", "discover.py", output, force=True)
    assert len(called) == 1


def test_run_stage_exits_on_failure(tmp_path, monkeypatch):
    import pytest

    output = tmp_path / "missing.jsonl"
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1)
    )
    with pytest.raises(SystemExit):
        pipeline.run_stage("fake", "discover.py", output, force=False)
