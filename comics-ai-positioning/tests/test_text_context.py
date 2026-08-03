import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import text_context
from build_pairs import build_pairs_for_row, _load_regions_by_page


def test_verified_episode_has_real_excerpt():
    amba = text_context.get("8a89f7d689fb441ea280cd782276bd7a.comics")
    assert amba is not None
    assert "Amba" in amba.excerpt
    assert "Saubha" in amba.excerpt


def test_unverified_episode_returns_none_not_a_guess():
    assert text_context.get("does-not-exist.comics") is None


def test_build_pairs_source_narrative_context_only_for_hand_verified_episode():
    row = {
        "photo_file": "20260731_153228.jpg",
        "page_index": 0,
        "episode_file": "d00c610a6f4647dcbd8116014674d255.comics",  # NOT in text_context.VERIFIED
        "ground_truth_cluster": [2, 6, 14, 20, 25, 28, 29, 30, 31, 32, 67, 68, 87, 95, 97, 108, 109],
        "confidence": 0.8,
    }
    regions_by_page = _load_regions_by_page()
    canvas_cache: dict = {}
    pairs = build_pairs_for_row(row, regions_by_page, canvas_cache)
    assert pairs
    assert all(
        p.source_narrative_context is None for p in pairs
    ), "episode with no hand-verified spiritual_text match must not get a guessed excerpt"


def test_build_pairs_text_context_is_the_scene_s_own_real_ocr_text():
    # 08_king_arjun_kartavirya: known real dialogue includes "Kartavirya" per ocr.jsonl (grep-verified)
    row = {
        "photo_file": "dummy.jpg",
        "page_index": 0,
        "episode_file": "54e9d4bbf0864460b9ff06271b215bd0.comics",
        "ground_truth_cluster": [80, 82],  # real layer_indexes with OCR'd Kartavirya dialogue
        "confidence": 0.8,
    }
    import scene_text

    text = scene_text.text_for_cluster(row["episode_file"], row["ground_truth_cluster"])
    assert text is not None
    assert "KARTAVIRYA" in text.upper()
