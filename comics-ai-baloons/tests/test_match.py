import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import match
from csv_loader import CsvRow
from models import OcrResult


def _row(row_id, en=None, ru=None):
    texts = {}
    if en is not None:
        texts["en"] = en
    if ru is not None:
        texts["ru"] = ru
    return CsvRow(row_id=row_id, bubble_type="speech", chapter_label="", texts=texts)


def test_normalize_strips_punctuation_and_case():
    assert match.normalize("Hello, World!!!") == "hello world"
    assert match.normalize("  multiple   spaces  ") == "multiple spaces"


def test_exact_match_scores_100():
    rows = [_row("P1_001", en="War's end."), _row("P1_002", en="Something else.")]
    result = match.match_balloon("f.comics", 0, "WAR'S END.", "", rows)
    assert result.status == "matched"
    assert result.csv_row_id == "P1_001"
    assert result.match_score == 1.0


def test_no_ocr_text_skips_no_match():
    rows = [_row("P1_001", en="War's end.")]
    result = match.match_balloon("f.comics", 0, "", "", rows)
    assert result.status == "skipped_no_match"
    assert result.csv_row_id is None


def test_unrelated_text_skips_low_confidence():
    rows = [_row("P1_001", en="War's end."), _row("P1_002", en="The gods have cursed me.")]
    result = match.match_balloon(
        "f.comics", 0, "COMPLETELY UNRELATED SENTENCE ABOUT SOMETHING ELSE", "", rows
    )
    assert result.status == "skipped_low_confidence"
    assert result.csv_row_id is None


def test_tied_candidates_skip_ambiguous_without_tiebreak():
    rows = [_row("P1_001", en="Why?"), _row("P2_001", en="Why?")]
    result = match.match_balloon("f.comics", 0, "WHY?", "", rows)
    assert result.status == "skipped_ambiguous"


def test_ru_resolves_tie_when_en_is_ambiguous():
    rows = [
        _row("P1_001", en="Why?", ru="Почему ты здесь?"),
        _row("P2_001", en="Why?", ru="Совсем другое предложение."),
    ]
    result = match.match_balloon("f.comics", 0, "WHY?", "ПОЧЕМУ ТЫ ЗДЕСЬ?", rows)
    assert result.status == "matched"
    assert result.csv_row_id == "P1_001"
    assert result.matched_on == "en+ru_tiebreak"


def test_falls_back_to_ru_when_en_ocr_missing():
    rows = [_row("P1_001", ru="Битва закончена.")]
    result = match.match_balloon("f.comics", 0, "", "БИТВА ЗАКОНЧЕНА.", rows)
    assert result.status == "matched"
    assert result.matched_on == "ru"


def test_match_all_groups_by_balloon():
    ocr = [
        OcrResult("f.comics", 0, 0, "War's end.", 0.9),
        OcrResult("f.comics", 0, 1, "Битва закончена.", 0.9),
        OcrResult("f.comics", 1, 0, "no such text anywhere", 0.9),
    ]
    rows = [_row("P1_001", en="War's end.", ru="Битва закончена.")]
    results = match.match_all(ocr, rows)
    assert len(results) == 2
    r0 = next(r for r in results if r.layer_index == 0)
    assert r0.status == "matched"
    assert r0.csv_row_id == "P1_001"
