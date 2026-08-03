import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from full_pipeline_demo import run_demo


def test_full_pipeline_on_a_real_newly_covered_page():
    # Real, previously-zero-coverage episode (2a5e3303...) recovered by
    # sdd-comics-ai-transformations' criterion 3 re-matching refinement -- exactly the kind of
    # "newly covered content" criterion 5 asks this flow to demonstrate end to end.
    report = run_demo("20260731_153814.jpg", 0)

    assert report["episode_file"] == "2a5e3303ba8c42e3ba395dad794164a7.comics"
    assert report["region_count"] == 15
    assert len(report["regions"]) == 15

    kinds = {r["kind"] for r in report["regions"]}
    assert kinds <= {"art", "character", "balloon"}

    # Every region got a real position proposal (positioning stage ran for real).
    for r in report["regions"]:
        assert isinstance(r["proposed_position"]["x"], int)
        assert isinstance(r["proposed_position"]["y"], int)

    # Every region got a real reveal proposal for all 4 properties (transformation stage ran).
    for r in report["regions"]:
        assert set(r["proposed_reveal"].keys()) == {"translate", "scale", "rotate", "alpha"}

    # Balloon regions get the calibrated alpha+scale reveal; this episode's balloons should too.
    balloon_regions = [r for r in report["regions"] if r["kind"] == "balloon"]
    assert balloon_regions
    for r in balloon_regions:
        assert r["proposed_reveal"]["alpha"]["occurs"] is True
        assert r["proposed_reveal"]["scale"]["occurs"] is True

    # Real script-context cross-check is present for this episode (criterion 2's OCR-dialogue
    # fallback covers it, since it never had a spiritual_text match).
    assert report["script_context"] is not None
    assert report["script_context"]["text_source"] == "ocr_dialogue"
    assert len(report["script_context"]["characters"]) > 0
