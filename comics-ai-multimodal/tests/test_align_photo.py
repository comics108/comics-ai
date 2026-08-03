import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import align_photo as ap  # noqa: E402
import baloons_bridge  # noqa: E402

LOWCAMERA_DIR = (
    baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)


def _corpus_entry(source_file, layer_index, text):
    return {"source_file": source_file, "layer_index": layer_index, "lang_index": 0, "text": text}


def test_match_page_to_episode_finds_confident_multi_phrase_match():
    corpus = [
        _corpus_entry("ep1.comics", 10, "I am listening to your words princes"),
        _corpus_entry("ep1.comics", 11, "You cannot resist all of us"),
        _corpus_entry("ep2.comics", 20, "completely unrelated other episode text"),
    ]
    page_text = (
        "sOmE OcR NoIsE i am listening to your words, princes!! more junk "
        "you cannot resist all of us --- trailing garbage 123"
    )
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode(page_text, corpus)
    assert episode == "ep1.comics"
    assert layer_indexes == [10, 11]
    assert confidence > 0.8
    assert reason == ""


def test_match_page_to_episode_accepts_single_confident_hit_when_no_competitor():
    # Real-data finding (2026-08-01, flows/sdd-comics-ai-transformations/02-specifications.md):
    # a single confident hit with no competing episode is a trustworthy match (21/24 real pages
    # investigated), not something to reject outright just because it's the only hit on the page.
    corpus = [_corpus_entry("ep1.comics", 10, "a fairly distinctive phrase here")]
    page_text = "a fairly distinctive phrase here and nothing else matches"
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode(page_text, corpus)
    assert episode == "ep1.comics"
    assert layer_indexes == [10]
    assert reason == ""


def test_match_page_to_episode_rejects_ambiguous_single_hit_tie():
    # Real-data finding: when 2+ episodes each have exactly 1 hit at a near-identical score
    # (margin < MARGIN_FOR_SINGLE_HIT), that's genuinely ambiguous (3/24 real pages found this way)
    # -- must not guess between them.
    corpus = [
        _corpus_entry("ep1.comics", 10, "bhishma gangeya this very regent of hastinapur"),
        _corpus_entry("ep2.comics", 20, "bhishma gangeya this very regent of hastinapore"),
    ]
    page_text = "bhishma gangeya - the regent of hastinapura?"
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode(page_text, corpus)
    assert episode is None
    assert "competing episodes" in reason


def test_match_page_to_episode_accepts_single_hit_with_clear_margin_over_competitor():
    # A weaker competing single-hit episode (score well below the top one) must not block an
    # otherwise-clean, high-confidence match -- the margin rule only rejects genuine ties.
    corpus = [
        _corpus_entry("ep1.comics", 10, "this phrase matches almost perfectly right here for real"),
        _corpus_entry("ep2.comics", 20, "this phrase matches almost badly right here"),
    ]
    page_text = (
        "some ocr noise this phrase matches almost perfectly right here for real "
        "more junk around it 123"
    )
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode(page_text, corpus)
    assert episode == "ep1.comics"
    assert reason == ""


def test_match_page_to_episode_no_ocr_text():
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode("", [_corpus_entry("ep1.comics", 1, "x")])
    assert episode is None
    assert "no OCR text" in reason


def test_match_page_to_episode_ignores_short_generic_phrases():
    # Regression test for a real false-positive found on real data: short generic corpus phrases
    # ("NO", "NO.") trivially partial_ratio-match almost any OCR'd text by substring containment,
    # producing spurious "confident" multi-hit matches that aren't really independent evidence.
    corpus = [
        _corpus_entry("wrong_episode.comics", 1, "NO"),
        _corpus_entry("wrong_episode.comics", 2, "NO."),
        _corpus_entry("wrong_episode.comics", 3, "no"),
    ]
    # This page text contains "no" as a substring many times (in "know", "north", "cannot") purely
    # incidentally -- must not be treated as 3 confident independent phrase matches.
    page_text = "i know the truth from the north but i cannot say more about it right now"
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode(page_text, corpus)
    assert episode is None
    assert "no balloon phrase matched" in reason


def test_match_page_to_episode_no_candidates_match():
    corpus = [_corpus_entry("ep1.comics", 10, "totally different content")]
    episode, layer_indexes, confidence, reason = ap.match_page_to_episode("garbled nonsense zzz", corpus)
    assert episode is None
    assert "no balloon phrase matched" in reason


def test_ground_truth_cluster_expands_via_scene_clustering(tmp_path):
    gt_data = {
        "episode_file": "ep1.comics",
        "width": 100,
        "height": 500,
        "composite_png": "unused.png",
        "regions": [
            {"layer_index": 0, "kind": "background", "kind_source": "explicit", "bbox": [0, 0, 100, 50]},
            {"layer_index": 1, "kind": "character", "kind_source": "explicit", "bbox": [0, 10, 50, 40]},
            {"layer_index": 2, "kind": "balloon", "kind_source": "explicit", "bbox": [10, 15, 40, 25]},
            {"layer_index": 3, "kind": "background", "kind_source": "explicit", "bbox": [0, 400, 100, 450]},
            {"layer_index": 4, "kind": "character", "kind_source": "explicit", "bbox": [0, 410, 50, 440]},
        ],
    }
    canvas_dir = tmp_path
    (canvas_dir / "ep1.gt.json").write_text(json.dumps(gt_data))

    # Matched only balloon layer_index=2 (first scene) -- must expand to its scene siblings 0,1,
    # not the second scene's 3,4.
    cluster = ap.ground_truth_cluster_for("ep1.comics", [2], canvas_dir=canvas_dir)
    assert cluster == [0, 1, 2]


def test_ground_truth_cluster_falls_back_when_no_gt_file(tmp_path):
    cluster = ap.ground_truth_cluster_for("missing.comics", [5, 7], canvas_dir=tmp_path)
    assert cluster == [5, 7]


def test_real_photo_smoke_no_crash_on_a_few_samples():
    corpus = ap.load_ocr_corpus()
    files = sorted(LOWCAMERA_DIR.glob("*.jpg"))[:3]
    for f in files:
        results = ap.align_photo(f, corpus)
        assert len(results) >= 1
        for r in results:
            assert r.status in ("matched", "skipped_no_match")
            if r.status == "matched":
                assert r.episode_file is not None
                assert len(r.ground_truth_cluster) >= len(r.matched_layer_indexes)
