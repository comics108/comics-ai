import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baloons_bridge  # noqa: E402
import kind_heuristic as kh  # noqa: E402


def _layer(images, animations=None):
    return {"images": images, "animations": animations or []}


def _slot(width, height, file="x_{0}_{1}_{2}.png"):
    return {"file": file, "width": width, "height": height}


def test_two_populated_slots_is_balloon():
    data = {"width": 1080, "layers": [_layer([_slot(600, 150), _slot(600, 150), {}])]}
    assert kh.infer_kinds_for_file(data) == ["balloon"]


def test_wide_tall_bottom_layer_is_background():
    data = {
        "width": 1080,
        "layers": [
            _layer([_slot(1080, 2000), {}, {}]),  # full-width, tall, first in stack -> background
            _layer(
                [_slot(300, 500), {}, {}],
                animations=[{"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "y": 100}],
            ),  # portrait character near same y, later in stack
        ],
    }
    kinds = kh.infer_kinds_for_file(data)
    assert kinds[0] == "background"
    assert kinds[1] == "character"


def test_local_bottom_of_stack_not_global():
    # Two separate background+character clusters far apart in y -- each background should be
    # recognized as "bottom of its own local stack", not just the very first layer globally.
    far_translate = [{"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "y": 20000}]
    data = {
        "width": 1080,
        "layers": [
            _layer([_slot(1080, 2000), {}, {}]),  # cluster 1 background, y=0
            _layer([_slot(1080, 2000), {}, {}], animations=far_translate),  # cluster 2 background, y=20000
            _layer([_slot(300, 500), {}, {}], animations=far_translate),  # cluster 2 character, y=20000
        ],
    }
    kinds = kh.infer_kinds_for_file(data)
    assert kinds[0] == "background"
    assert kinds[1] == "background"  # local bottom-of-stack within its own y-neighborhood
    assert kinds[2] == "character"


def test_tiny_irregular_single_slot_falls_back_to_art():
    data = {"width": 1080, "layers": [_layer([_slot(30, 20), {}, {}])]}
    assert kh.infer_kinds_for_file(data) == ["art"]


def test_runs_without_crashing_across_all_real_dataset_files_and_reports_distribution():
    from collections import Counter

    totals = Counter()
    for f in baloons_bridge.find_comics_files():
        archive = baloons_bridge.ComicsArchive(f)
        data = archive.read_data_json()
        kinds = kh.infer_kinds_for_file(data)
        assert len(kinds) == len(data["layers"])
        totals.update(kinds)

    assert sum(totals.values()) == 4594
    # Informational -- printed for the Task 2.2 manual spot-check step, not asserted precisely
    # since this is a heuristic (Specifications explicitly expects imperfection here).
    print("kind distribution across all 27 dataset files:", dict(totals))
