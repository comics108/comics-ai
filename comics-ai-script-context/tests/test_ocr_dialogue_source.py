import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ocr_dialogue_source import build_excerpt_for_episode, load_ocr_entries, BALOONS_OCR_JSONL


def test_build_excerpt_orders_by_layer_index_and_joins():
    entries = [
        {"source_file": "ep1.comics", "layer_index": 20, "lang_index": 0, "text": "second line"},
        {"source_file": "ep1.comics", "layer_index": 5, "lang_index": 0, "text": "first line"},
        {"source_file": "ep2.comics", "layer_index": 1, "lang_index": 0, "text": "other episode"},
    ]
    excerpt = build_excerpt_for_episode("ep1.comics", entries)
    assert excerpt == "first line second line"


def test_build_excerpt_ignores_non_english_and_empty_text():
    entries = [
        {"source_file": "ep1.comics", "layer_index": 1, "lang_index": 1, "text": "русский текст"},
        {"source_file": "ep1.comics", "layer_index": 2, "lang_index": 0, "text": "   "},
        {"source_file": "ep1.comics", "layer_index": 3, "lang_index": 0, "text": "real text"},
    ]
    excerpt = build_excerpt_for_episode("ep1.comics", entries)
    assert excerpt == "real text"


def test_build_excerpt_returns_none_when_episode_has_no_real_text():
    entries = [{"source_file": "other.comics", "layer_index": 1, "lang_index": 0, "text": "hi"}]
    assert build_excerpt_for_episode("ep1.comics", entries) is None


def test_build_excerpt_truncates_to_max_chars():
    entries = [{"source_file": "ep1.comics", "layer_index": 1, "lang_index": 0, "text": "x" * 100}]
    excerpt = build_excerpt_for_episode("ep1.comics", entries, max_chars=10)
    assert excerpt == "x" * 10


def test_load_ocr_entries_reads_the_real_corpus():
    entries = load_ocr_entries()
    assert len(entries) > 0
    assert all("source_file" in e and "text" in e for e in entries[:5])


def test_build_excerpt_on_real_data_covers_all_27_episodes():
    # Real, checked fact (2026-08-02): comics-ai-baloons' structural discover.py scans the whole
    # dataset independent of photo-matching, so all 27 episodes have real OCR'd dialogue.
    entries = load_ocr_entries()
    dataset_dir = (
        BALOONS_OCR_JSONL.parents[4]
        / "dataset"
        / "boranko"
        / "mahabharata"
        / "book1"
        / "comics_interactive"
    )
    episode_files = sorted(p.name for p in dataset_dir.glob("*.comics"))
    assert len(episode_files) == 27
    missing = [ep for ep in episode_files if build_excerpt_for_episode(ep, entries) is None]
    assert missing == []
