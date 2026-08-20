import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compose_production import AcceptedAsset, build_fail_closed_summary, compose_candidates
from package_comics import PackagingAsset, build_data_json


SHA = "a" * 64


def asset(order: int, state="accepted") -> AcceptedAsset:
    return AcceptedAsset(
        f"asset-{order}:v1", f"ch01-beat{order:02d}", order, f"a{order}.png", f"a{order}.mask.png",
        900 + order * 10, 600 + order * 20, "character" if order == 2 else "art", SHA, state,
    )


def test_rule_and_learned_candidates_are_vertical_editable_and_proposed():
    candidates = compose_candidates(1, [asset(2), asset(1)], learned_positioner=lambda _: {"asset-2:v1": (12, 18)})
    assert [item.method for item in candidates] == [
        "deterministic_vertical_stack_v1", "learned_positioner_candidate_v1"
    ]
    for candidate in candidates:
        assert candidate.review_state == "proposed"
        assert candidate.scroll_type == "vertical"
        assert candidate.preferred_orientation == "portrait"
        assert [item.beat_order for item in candidate.placements] == [1, 2]
        assert all(item.bitmap_mask_file for item in candidate.placements)
        assert candidate.quality["bounds_valid"]
        assert candidate.camera_path[0]["position"] == 0
        assert candidate.camera_path[-1]["position"] == 1000
        assert candidate.lineage["intent_claim"] == "candidate_only_not_artist_intent"

    # The proposal values pass the existing shared `.comics` writer contract; this is deliberately
    # only a structural adapter proof, not promotion of synthetic assets or heuristic intent.
    learned = candidates[1]
    packaged = build_data_json(
        learned.canvas_width, learned.canvas_height,
        [PackagingAsset(
            kind="art", image=Image.new("RGBA", (item.width, item.height)), x=item.x, y=item.y,
            stem=f"beat_{item.beat_order}", contains_russian_text=False, z_depth=item.z_depth,
        ) for item in learned.placements],
        camera_path=list(learned.camera_path),
        preferred_viewport_width=learned.viewport_width,
        preferred_viewport_height=learned.viewport_height,
    )
    assert packaged["cameraPath"] == list(learned.camera_path)
    assert packaged["preferredViewportWidth"] == 1080
    assert packaged["layers"][1]["zDepth"] == .25


def test_unaccepted_or_duplicate_beat_input_is_rejected():
    with pytest.raises(ValueError, match="not accepted"):
        compose_candidates(1, [asset(1, "proposed")])
    with pytest.raises(ValueError, match="one accepted"):
        compose_candidates(1, [asset(1), asset(1)])


def test_real_coverage_stays_fail_closed_without_accepted_assets():
    root = Path(__file__).resolve().parents[4]
    coverage = json.loads((root / "work/bhagavadgita/production/story-coverage/coverage-v1.json").read_text())
    summary = build_fail_closed_summary(coverage)
    assert summary["release_state"] == "blocked"
    assert summary["scroll_type"] == "vertical"
    assert [item["accepted_asset_count"] for item in summary["chapters"]] == [0, 0]
    assert [len(item["missing_beat_ids"]) for item in summary["chapters"]] == [6, 6]
