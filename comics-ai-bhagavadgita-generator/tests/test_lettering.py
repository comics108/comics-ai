import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lettering import (LanguageSlotRegistry, build_golden_lettering_corpus,
                       exact_readback_match, normalize_authoritative, validate_glyph_layout)


def test_authoritative_normalization_and_readback_are_unicode_exact():
    assert normalize_authoritative("  cafe\u0301\r\n") == "café"
    assert exact_readback_match("श्री भगवान", "श्री   भगवान\n")
    assert not exact_readback_match("Бхагавад Гита", "Бхагавад-Гита")


def test_language_slots_are_dynamic_unique_and_sanskrit_need_not_claim_hi_slot():
    registry = LanguageSlotRegistry((("en", 0), ("ru", 1), ("th", 4)))
    assert registry.slot_for("th") == 4
    with pytest.raises(KeyError):
        registry.slot_for("sa")
    with pytest.raises(ValueError, match="unique"):
        LanguageSlotRegistry((("en", 0), ("ru", 0)))


def test_glyph_mask_gate_rejects_collision_and_empty_text():
    region = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(region).rectangle((2, 2, 17, 17), fill=255)
    glyph = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(glyph).rectangle((5, 5, 10, 10), fill=255)
    assert validate_glyph_layout(glyph, region).fit
    ImageDraw.Draw(glyph).point((0, 0), fill=255)
    gate = validate_glyph_layout(glyph, region)
    assert not gate.fit and gate.collision_pixels == 1
    assert not validate_glyph_layout(Image.new("L", (20, 20), 0), region).fit


def test_real_golden_corpus_has_ru_en_and_unfixed_slot_sanskrit_for_every_sloka():
    entries = build_golden_lettering_corpus()
    assert len(entries) == (37 + 52) * 3
    grouped = {}
    for entry in entries:
        grouped.setdefault((entry.chapter_order, entry.sloka_order), {})[entry.language_code] = entry
    assert len(grouped) == 89
    assert all(set(values) == {"ru", "en", "sa"} for values in grouped.values())
    assert all(values["en"].runtime_slot == 0 and values["ru"].runtime_slot == 1
               and values["sa"].runtime_slot is None for values in grouped.values())
