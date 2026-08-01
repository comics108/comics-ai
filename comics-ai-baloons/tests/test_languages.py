import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import languages as L


def test_existing_cultures_order_matches_editor_enum():
    assert L.LANGUAGES[0] == "en"
    assert L.LANGUAGES[1] == "ru"
    assert L.LANGUAGES[2] == "hi"


def test_round_trip_index_and_code():
    for i, code in enumerate(L.LANGUAGES):
        assert L.index_to_lang(i) == code
        assert L.lang_to_index(code) == i


def test_twenty_languages():
    assert len(L.LANGUAGES) == 20


def test_unknown_language():
    assert not L.is_known_language("xx")
    assert L.is_known_language("ar")
