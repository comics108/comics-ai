import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import discover
from models import BalloonLayer

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"


def test_discover_matches_survey_count():
    # Independently established during Requirements investigation via a filename-based survey,
    # then re-confirmed structurally during Specifications: exactly 825 multi-language balloon
    # layers across all 27 files.
    balloons = discover.discover_all(DATASET_DIR)
    assert len(balloons) == 825


def test_every_balloon_has_at_least_two_slots():
    balloons = discover.discover_all(DATASET_DIR)
    for b in balloons:
        assert len(b.slots) >= 2


def test_known_sample_balloon_present():
    balloons = discover.discover_all(DATASET_DIR)
    matches = [
        b
        for b in balloons
        if b.source_file == "8a89f7d689fb441ea280cd782276bd7a.comics"
        and 0 in b.slots
        and b.slots[0].file_template.startswith("b1_eng_")
    ]
    assert len(matches) == 1
    b = matches[0]
    assert b.slots[0].width == 648
    assert b.slots[0].height == 152
    assert 1 in b.slots and b.slots[1].file_template.startswith("b1_ru_")


def test_jsonable_round_trip():
    balloons = discover.discover_all(DATASET_DIR)
    sample = balloons[0]
    d = sample.to_jsonable()
    json.dumps(d)  # must be JSON-serializable
    restored = BalloonLayer.from_jsonable(d)
    assert restored == sample


def test_write_jsonl_line_count(tmp_path):
    balloons = discover.discover_all(DATASET_DIR)[:10]
    out_path = tmp_path / "balloons.jsonl"
    discover.write_jsonl(balloons, out_path)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 10
    for line in lines:
        json.loads(line)
