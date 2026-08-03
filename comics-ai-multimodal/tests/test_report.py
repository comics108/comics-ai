import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report as rpt  # noqa: E402


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_build_report_combines_all_sources(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "p1_p0.comics").touch()

    _write_jsonl(
        tmp_path / "alignment.jsonl",
        [
            {"photo_file": "p1.jpg", "page_index": 0, "episode_file": "ep1.comics", "matched_layer_indexes": [1], "ground_truth_cluster": [1, 2], "confidence": 0.9, "status": "matched", "reason": ""},
            {"photo_file": "p2.jpg", "page_index": 0, "episode_file": None, "matched_layer_indexes": [], "ground_truth_cluster": [], "confidence": 0.0, "status": "skipped_no_match", "reason": "no OCR text extracted from page"},
        ],
    )
    _write_jsonl(
        tmp_path / "regions.jsonl",
        [
            {"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "character", "confidence": 0.8, "bbox": [0, 0, 1, 1]},
            {"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "balloon", "confidence": 0.7, "bbox": [0, 0, 1, 1]},
        ],
    )
    _write_jsonl(
        tmp_path / "eval_report.jsonl",
        [
            {"photo_file": "p1.jpg", "page_index": 0, "episode_file": "ep1.comics", "predicted_counts": {}, "ground_truth_counts": {}, "per_kind_agreement": {}, "mean_agreement": 0.6},
        ],
    )
    _write_jsonl(
        tmp_path / "balloon_handoff.jsonl",
        [
            {"photo_file": "p1.jpg", "page_index": 0, "episode_file": "ep1.comics", "real_balloon_layer_indexes": [1, 2], "predicted_balloon_region_count": 1, "translated_layer_indexes": [1], "rendered_languages_by_layer": {}, "packaged_output_available": True},
        ],
    )

    entries = rpt.build_report(tmp_path)
    assert len(entries) == 2

    p1 = next(e for e in entries if e["photo_file"] == "p1.jpg")
    assert p1["status"] == "matched"
    assert p1["regions_cut"] == 2
    assert p1["regions_by_kind"] == {"character": 1, "balloon": 1}
    assert p1["packaged"] is True
    assert p1["mean_kind_count_agreement"] == 0.6
    assert p1["real_balloon_layers"] == 2
    assert p1["translated_balloon_layers"] == 1

    p2 = next(e for e in entries if e["photo_file"] == "p2.jpg")
    assert p2["status"] == "skipped_no_match"
    assert p2["regions_cut"] == 0
    assert p2["packaged"] is False
    assert "mean_kind_count_agreement" not in p2


def test_render_markdown_includes_summary_and_skip_reasons(tmp_path):
    entries = [
        {"photo_file": "p1.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics", "match_confidence": 0.9, "reason": "", "regions_cut": 3, "regions_by_kind": {}, "packaged": True, "mean_kind_count_agreement": 0.5},
        {"photo_file": "p2.jpg", "page_index": 0, "status": "skipped_no_match", "episode_file": None, "match_confidence": 0.0, "reason": "no OCR text extracted from page", "regions_cut": 0, "regions_by_kind": {}, "packaged": False},
    ]
    md = rpt.render_markdown(entries)
    assert "Total photo/pages processed: 2" in md
    assert "Matched: 1" in md
    assert "no OCR text extracted from page" in md
    assert "p1.jpg" in md


def test_render_markdown_handles_zero_entries_without_crashing():
    md = rpt.render_markdown([])
    assert "Total photo/pages processed: 0" in md


def test_build_report_real_data_if_present():
    if not (rpt.WORK_DIR / "alignment.jsonl").is_file():
        pytest.skip("work/alignment.jsonl not present -- run the pipeline first")
    entries = rpt.build_report()
    assert len(entries) > 0
    md = rpt.render_markdown(entries)
    assert "Total photo/pages processed" in md
