import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report


def test_every_balloon_accounted_for_exactly_once():
    balloons = [
        {"source_file": "f.comics", "layer_index": 0},
        {"source_file": "f.comics", "layer_index": 1},
        {"source_file": "f.comics", "layer_index": 2},
    ]
    matches = [
        {
            "source_file": "f.comics",
            "layer_index": 0,
            "csv_row_id": "P1_001",
            "match_score": 1.0,
            "matched_on": "en",
            "status": "matched",
        },
        {
            "source_file": "f.comics",
            "layer_index": 1,
            "csv_row_id": None,
            "match_score": 0.5,
            "matched_on": "en",
            "status": "skipped_low_confidence",
        },
        # layer_index 2 has no match entry at all -> not_matched_no_data
    ]
    rep = report.build_report(balloons, [], matches, [], [])
    assert len(rep) == 3
    statuses = {r["layer_index"]: r["status"] for r in rep}
    assert statuses[1] == "skipped_low_confidence"
    assert statuses[2] == "not_matched_no_data"


def test_hand_lettered_status():
    balloons = [{"source_file": "f.comics", "layer_index": 0}]
    matches = [
        {
            "source_file": "f.comics",
            "layer_index": 0,
            "csv_row_id": "P1_001",
            "match_score": 1.0,
            "matched_on": "en",
            "status": "matched",
        }
    ]
    lettering = [{"source_file": "f.comics", "layer_index": 0, "label": "hand_lettered"}]
    renders = [
        {
            "source_file": "f.comics",
            "layer_index": 0,
            "lang_code": "uk",
            "rendered": False,
            "reason": "hand_lettered: flagged for manual artist review",
        }
    ]
    rep = report.build_report(balloons, [], matches, lettering, renders)
    assert rep[0]["status"] == "hand_lettered_flagged"
    assert rep[0]["languages_rendered"] == []
    assert rep[0]["languages_skipped"][0]["lang"] == "uk"


def test_rendered_status_and_language_lists():
    balloons = [{"source_file": "f.comics", "layer_index": 0}]
    matches = [
        {
            "source_file": "f.comics",
            "layer_index": 0,
            "csv_row_id": "P1_001",
            "match_score": 1.0,
            "matched_on": "en",
            "status": "matched",
        }
    ]
    renders = [
        {"source_file": "f.comics", "layer_index": 0, "lang_code": "uk", "rendered": True, "reason": ""},
        {
            "source_file": "f.comics",
            "layer_index": 0,
            "lang_code": "th",
            "rendered": False,
            "reason": "text_overflow",
        },
    ]
    rep = report.build_report(balloons, [], matches, [], renders)
    assert rep[0]["status"] == "rendered"
    assert rep[0]["languages_rendered"] == ["uk"]
    assert rep[0]["languages_skipped"] == [{"lang": "th", "reason": "text_overflow"}]


def test_markdown_renders_without_error():
    rep = [
        {
            "source_file": "f.comics",
            "layer_index": 0,
            "ocr_text_en": "hello",
            "match": {"csv_row_id": "P1_001", "score": 1.0, "matched_on": "en"},
            "lettering_class": "machine_set",
            "languages_rendered": ["uk"],
            "languages_skipped": [],
            "status": "rendered",
        }
    ]
    md = report.render_markdown(rep)
    assert "Total balloons discovered**: 1" in md
    assert "f.comics" in md
