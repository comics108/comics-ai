import base64
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import segment_image as seg  # noqa: E402
import infer_segmenter as inf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_PHOTO_DIR = (
    REPO_ROOT / "dataset" / "boranko" / "mahabharata" / "book1" / "comics_book_lowcamera"
)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "segment_image.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _parse_ndjson(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.strip().splitlines() if line.strip()]


def test_run_emits_failure_for_missing_checkpoint(tmp_path):
    image_path = tmp_path / "fake.png"
    PILImage.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(image_path)
    events = []
    seg.emit = events.append  # capture instead of printing
    try:
        exit_code = seg.run(image_path, tmp_path / "does_not_exist.pt")
    finally:
        seg.emit = print
    assert exit_code == 0
    kinds = [e["event"] for e in events]
    assert kinds == ["routing", "failure"]
    assert events[-1]["reason"] == "model_checkpoint_not_found"
    assert events[-1]["retryable"] is False


def test_run_emits_failure_for_unreadable_image(tmp_path):
    if not inf.DEFAULT_CHECKPOINT.is_file():
        pytest.skip("work/models/unet_baseline.pt not present")
    bogus_image = tmp_path / "not_an_image.png"
    bogus_image.write_bytes(b"this is not image data")
    events = []
    seg.emit = events.append
    try:
        exit_code = seg.run(bogus_image, inf.DEFAULT_CHECKPOINT)
    finally:
        seg.emit = print
    assert exit_code == 0
    assert [e["event"] for e in events] == ["routing", "failure"]
    assert events[-1]["reason"] == "image_not_readable"
    assert events[-1]["retryable"] is False


def test_run_success_path_emits_regions_with_decodable_crops(tmp_path):
    if not inf.DEFAULT_CHECKPOINT.is_file():
        pytest.skip("work/models/unet_baseline.pt not present")
    image_path = tmp_path / "fake.png"
    PILImage.fromarray(np.random.randint(0, 255, (300, 220, 3), dtype=np.uint8)).save(image_path)
    events = []
    seg.emit = events.append
    try:
        exit_code = seg.run(image_path, inf.DEFAULT_CHECKPOINT, device="cpu")
    finally:
        seg.emit = print
    assert exit_code == 0
    event_names = [e["event"] for e in events]
    assert event_names[0] == "routing"
    assert event_names[-1] in ("success", "failure")
    success_events = [e for e in events if e["event"] == "success"]
    if success_events:
        for region in success_events[0]["regions"]:
            assert region["kind"] in {"art", "background", "character", "balloon"}
            x0, y0, x1, y1 = region["bbox"]
            assert x1 > x0 and y1 > y0
            decoded = PILImage.open(BytesIO(base64.b64decode(region["crop_png_base64"])))
            assert decoded.size == (x1 - x0, y1 - y0)


def test_cli_subprocess_produces_valid_ndjson_on_a_real_photo():
    # Real end-to-end smoke test of the actual wire protocol (Task 1.2's verification calls for
    # invoking the real CLI, not just calling run() in-process) -- confirms argument parsing,
    # stdout framing, and that nothing print()s anything else onto stdout ahead of the NDJSON.
    if not inf.DEFAULT_CHECKPOINT.is_file():
        pytest.skip("work/models/unet_baseline.pt not present")
    photos = sorted(REAL_PHOTO_DIR.glob("*.jpg")) if REAL_PHOTO_DIR.is_dir() else []
    if not photos:
        pytest.skip("no real lowcamera photos present")

    result = _run_cli(["--image", str(photos[0]), "--device", "cpu"])
    assert result.returncode == 0, result.stderr
    events = _parse_ndjson(result.stdout)
    assert events[0] == {"event": "routing", "on_device": True, "reason": None}
    assert events[-1]["event"] in ("success", "failure")
    if events[-1]["event"] == "success":
        assert isinstance(events[-1]["regions"], list)


def test_cli_missing_required_image_arg_exits_nonzero_with_no_stdout_event():
    # Argparse failure (missing --image) is exactly the "unanticipated"/crash-shaped case --
    # non-zero exit, nothing resembling a structured event on stdout. Confirms the Dart client's
    # process_error fallback path has something real to catch.
    result = _run_cli(["--checkpoint", "/nonexistent.pt"])
    assert result.returncode != 0
    assert _parse_ndjson(result.stdout) == []
