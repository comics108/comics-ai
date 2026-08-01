import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import csv_loader


def test_loads_expected_row_count():
    rows = csv_loader.load_csv()
    assert len(rows) == 563


def test_known_row_p1_001():
    rows = csv_loader.load_csv()
    row = next(r for r in rows if r.row_id == "P1_001")
    assert row.bubble_type == "caption"
    assert row.texts["en"] == "War's end."
    assert row.texts["ru"] == "Битва закончена."
    assert row.chapter_label.startswith("01")


def test_sparse_languages_not_fabricated():
    rows = csv_loader.load_csv()
    # Confirmed sparse during Requirements investigation -- not every row has every language.
    missing_some_lang = [r for r in rows if len(r.texts) < 20]
    assert missing_some_lang, "expected at least some rows with sparse language coverage"
    row = missing_some_lang[0]
    assert row.text_for("xx-not-a-lang") is None


def test_all_row_ids_match_pattern():
    rows = csv_loader.load_csv()
    for r in rows:
        assert csv_loader.ROW_ID_PATTERN.match(r.row_id)


def test_iso_column_set_matches_languages_module():
    import languages

    rows = csv_loader.load_csv()
    all_langs_seen = set()
    for r in rows:
        all_langs_seen.update(r.texts.keys())
    assert all_langs_seen <= set(languages.LANGUAGES)
