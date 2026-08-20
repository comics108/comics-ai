import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models import CanonicalChapter, SlokaSource
from pipeline import ChapterLockedError, can_reuse_chapter, chapter_lock, run_lottie_source
from report import compute_file_sha256


def _chapter(order=1) -> CanonicalChapter:
    sloka = SlokaSource(
        id=100, chapter_id=order, order=1, name="x", sanskrit="s", transcription="t",
        translation_ru="tr", comment_ru="", audio_ref="", sanskrit_audio_ref="",
    )
    return CanonicalChapter(book_id=1, chapter_id=order, order=order, title="x", slokas=(sloka,))


def test_can_reuse_is_false_with_no_previous_manifest(tmp_path):
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"x")
    assert can_reuse_chapter(None, _chapter(), output_path, "sha256:d", "sha256:c") is False


def test_can_reuse_is_false_when_dataset_fingerprint_differs(tmp_path):
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"x")
    previous = {
        "dataset_fingerprint": "sha256:OLD", "config_fingerprint": "sha256:c",
        "chapters": [{"order": 1, "status": "valid", "sha256": compute_file_sha256(output_path)}],
    }
    assert can_reuse_chapter(previous, _chapter(), output_path, "sha256:NEW", "sha256:c") is False


def test_can_reuse_is_false_when_output_file_is_missing(tmp_path):
    output_path = tmp_path / "chapter_01.comics"  # never created
    previous = {
        "dataset_fingerprint": "sha256:d", "config_fingerprint": "sha256:c",
        "chapters": [{"order": 1, "status": "valid", "sha256": "sha256:whatever"}],
    }
    assert can_reuse_chapter(previous, _chapter(), output_path, "sha256:d", "sha256:c") is False


def test_can_reuse_is_false_when_sha256_mismatches(tmp_path):
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"current bytes")
    previous = {
        "dataset_fingerprint": "sha256:d", "config_fingerprint": "sha256:c",
        "chapters": [{"order": 1, "status": "valid", "sha256": "sha256:stale-hash-from-before"}],
    }
    assert can_reuse_chapter(previous, _chapter(), output_path, "sha256:d", "sha256:c") is False


def test_can_reuse_is_false_when_chapter_status_is_failed(tmp_path):
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"x")
    previous = {
        "dataset_fingerprint": "sha256:d", "config_fingerprint": "sha256:c",
        "chapters": [{"order": 1, "status": "failed", "sha256": compute_file_sha256(output_path)}],
    }
    assert can_reuse_chapter(previous, _chapter(), output_path, "sha256:d", "sha256:c") is False


def test_can_reuse_is_true_when_everything_matches(tmp_path):
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"real matching bytes")
    previous = {
        "dataset_fingerprint": "sha256:d", "config_fingerprint": "sha256:c",
        "chapters": [{"order": 1, "status": "valid", "sha256": compute_file_sha256(output_path)}],
    }
    assert can_reuse_chapter(previous, _chapter(), output_path, "sha256:d", "sha256:c") is True


def test_chapter_lock_prevents_concurrent_acquisition_and_releases_after(tmp_path):
    chapter = _chapter()
    with chapter_lock(tmp_path, chapter):
        try:
            with chapter_lock(tmp_path, chapter):
                raise AssertionError("second lock acquisition should have raised")
        except ChapterLockedError:
            pass
    # released after the first `with` exits -- re-acquiring now must succeed
    with chapter_lock(tmp_path, chapter):
        pass


def test_real_lottie_source_run_is_standalone_and_writes_camera_depth_report(tmp_path):
    manifest = run_lottie_source(tmp_path)
    output = tmp_path / manifest["output_file"]
    assert output.exists()
    assert manifest["counts_toward_chapters"] is False
    assert manifest["scene_count"] == 3
    assert manifest["image_layer_count"] == 519
    assert manifest["camera_point_count"] == 19
    assert manifest["distinct_nonzero_z_depth_count"] > 2
    assert manifest["parallax_rendered_by_current_viewers"] is False
    report = (tmp_path / "lottie_report.md").read_text(encoding="utf-8")
    assert "do not yet render" in report
    assert "does not claim visible parallax today" in report


def test_real_smoke_run_chapter_one_no_ai_no_psd_produces_a_valid_archive(tmp_path):
    """Real integration test: the exact smoke path the Plan calls for --
    `--chapter 1 --no-ai --no-psd` -- run as a real subprocess against the real dataset."""
    import subprocess

    script = str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline.py")
    result = subprocess.run(
        [sys.executable, script, "--chapter", "1", "--no-ai", "--no-psd", "--output-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr

    output_path = tmp_path / "chapter_01.comics"
    assert output_path.exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = next(c for c in manifest["chapters"] if c["order"] == 1)
    assert entry["status"] == "valid"
    assert entry["source_sloka_count"] == 37  # real, previously-verified chapter 1 sloka count
    assert entry["psd_inputs"] == []  # --no-psd
    assert (tmp_path / "report.md").exists()


def test_real_rerun_without_force_reuses_the_chapter_unchanged(tmp_path):
    """Real integration test for the idempotency contract: a second run with identical inputs and
    no --force must reuse the existing valid output rather than re-rendering (observable via the
    file's mtime/bytes staying byte-identical and the run being much faster)."""
    import subprocess
    import time

    script = str(Path(__file__).resolve().parent.parent / "scripts" / "pipeline.py")
    args = [sys.executable, script, "--chapter", "1", "--no-ai", "--no-psd", "--output-dir", str(tmp_path)]

    first = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stderr
    output_path = tmp_path / "chapter_01.comics"
    first_bytes = output_path.read_bytes()
    first_mtime = output_path.stat().st_mtime_ns

    time.sleep(0.05)
    second_start = time.monotonic()
    second = subprocess.run(args, capture_output=True, text=True, timeout=120)
    second_elapsed = time.monotonic() - second_start
    assert second.returncode == 0, second.stderr

    assert output_path.read_bytes() == first_bytes
    assert output_path.stat().st_mtime_ns == first_mtime  # never rewritten
    assert "reused" in second.stderr
    assert second_elapsed < 5.0  # reuse must be fast -- no real re-render happened
