import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import extract

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_FILE = "8a89f7d689fb441ea280cd782276bd7a.comics"


def _known_b1_balloon():
    balloons = discover.discover_all(DATASET_DIR)
    for b in balloons:
        if b.source_file == SAMPLE_FILE and 0 in b.slots and b.slots[0].file_template.startswith("b1_eng_"):
            return b
    raise AssertionError("expected balloon not found")


def test_extract_balloon_produces_expected_size(tmp_path):
    balloon = _known_b1_balloon()
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)

    assert set(paths.keys()) == {0, 1}
    en_path = tmp_path / paths[0]
    assert en_path.exists()
    img = Image.open(en_path)
    assert img.size == (balloon.slots[0].width, balloon.slots[0].height)


def test_extract_balloon_is_cached(tmp_path):
    balloon = _known_b1_balloon()
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths1 = extract.extract_balloon(balloon, archive, tmp_path)
    mtime1 = (tmp_path / paths1[0]).stat().st_mtime_ns
    paths2 = extract.extract_balloon(balloon, archive, tmp_path)
    mtime2 = (tmp_path / paths2[0]).stat().st_mtime_ns
    assert paths1 == paths2
    assert mtime1 == mtime2  # not re-stitched the second time


def test_extract_all_and_manifest_round_trip(tmp_path):
    balloons = discover.discover_all(DATASET_DIR)[:5]
    manifest = extract.extract_all(balloons, DATASET_DIR, tmp_path)
    assert len(manifest) == 5
    for entry, balloon in zip(manifest, balloons):
        assert entry["source_file"] == balloon.source_file
        assert entry["layer_index"] == balloon.layer_index
        assert len(entry["paths"]) == len(balloon.slots)

    manifest_path = tmp_path / "manifest.jsonl"
    extract.write_manifest(manifest, manifest_path)
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
