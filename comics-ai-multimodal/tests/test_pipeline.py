import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pipeline as pl  # noqa: E402


def test_run_stage_skips_when_output_exists(tmp_path):
    marker = tmp_path / "already_here.txt"
    marker.write_text("x")
    calls = []
    pl.run_stage("stage", marker, lambda: calls.append(1), force=False)
    assert calls == []


def test_run_stage_runs_when_output_missing(tmp_path):
    marker = tmp_path / "not_here.txt"
    calls = []
    pl.run_stage("stage", marker, lambda: calls.append(1), force=False)
    assert calls == [1]


def test_run_stage_force_reruns_even_if_output_exists(tmp_path):
    marker = tmp_path / "already_here.txt"
    marker.write_text("x")
    calls = []
    pl.run_stage("stage", marker, lambda: calls.append(1), force=True)
    assert calls == [1]


def test_full_pipeline_resumes_by_skipping_everything_if_present(capsys):
    # Real-data resumability check: since every stage's output already exists from prior manual
    # runs this session, invoking main() (default args, no --force) must skip every stage and
    # complete quickly, not silently re-run expensive work (training, OCR, etc.).
    import pytest

    required = [
        pl.WORK_DIR / "canvas",
        pl.WORK_DIR / "train_pairs" / "manifest.jsonl",
        pl.WORK_DIR / "models" / "unet_baseline.pt",
        pl.WORK_DIR / "alignment.jsonl",
        pl.WORK_DIR / "regions.jsonl",
        pl.WORK_DIR / "eval_report.jsonl",
        pl.WORK_DIR / "balloon_handoff.jsonl",
        pl.WORK_DIR / "library",
        pl.WORK_DIR / "output",
    ]
    if not all(p.exists() for p in required):
        pytest.skip("not all pipeline stage outputs are present -- run the pipeline stages first")

    sys.argv = ["pipeline.py"]
    pl.main()
    out = capsys.readouterr().out
    assert out.count("[skip]") >= 9
    assert "[run ]" not in out
    assert "pipeline complete." in out
