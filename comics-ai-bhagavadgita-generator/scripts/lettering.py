"""Exact multilingual lettering contracts, shaping, masks, and OCR/readback gates."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from load_dataset import load_book_one


FONTS_ROOT = Path(__file__).resolve().parents[1] / "fonts/Noto"
LATIN_FONT = FONTS_ROOT / "NotoSans-Regular.ttf"
DEVANAGARI_FONT = FONTS_ROOT / "NotoSansDevanagari[wdth,wght].ttf"


def normalize_authoritative(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in normalized.strip().split("\n"))


def normalize_readback(text: str) -> str:
    return " ".join(normalize_authoritative(text).split())


@dataclass(frozen=True)
class LanguageSlotRegistry:
    slots: tuple[tuple[str, int], ...]

    def __post_init__(self):
        codes = [code for code, _ in self.slots]
        indices = [slot for _, slot in self.slots]
        if len(codes) != len(set(codes)) or len(indices) != len(set(indices)):
            raise ValueError("language codes and populated slot indices must be unique")
        if any(not code or slot < 0 for code, slot in self.slots):
            raise ValueError("language slot mapping is invalid")

    def slot_for(self, language_code: str) -> int:
        mapping = dict(self.slots)
        if language_code not in mapping:
            raise KeyError(f"language has no runtime image slot: {language_code}")
        return mapping[language_code]


@dataclass(frozen=True)
class AuthoritativeLettering:
    id: str
    chapter_order: int
    sloka_order: int
    source_sloka_id: int
    language_code: str
    content_role: str
    runtime_slot: int | None
    text: str
    normalized_sha256: str


@dataclass(frozen=True)
class LayoutGate:
    glyph_pixels: int
    collision_pixels: int
    foreground_fraction: float
    fit: bool


def make_authoritative_entry(
    *, entry_id: str, chapter_order: int, sloka_order: int, source_sloka_id: int,
    language_code: str, content_role: str, runtime_slot: int | None, text: str,
) -> AuthoritativeLettering:
    normalized = normalize_authoritative(text)
    if not normalized:
        raise ValueError("authoritative lettering text is empty")
    return AuthoritativeLettering(
        entry_id, chapter_order, sloka_order, source_sloka_id, language_code, content_role,
        runtime_slot, normalized, hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def build_golden_lettering_corpus() -> tuple[AuthoritativeLettering, ...]:
    registry = LanguageSlotRegistry((('en', 0), ('ru', 1)))
    russian = {chapter.order: chapter for chapter in load_book_one(book_id=1)}
    english = {chapter.order: chapter for chapter in load_book_one(book_id=2)}
    entries = []
    for chapter_order in (1, 11):
        ru_by_order = {item.order: item for item in russian[chapter_order].slokas}
        en_by_order = {item.order: item for item in english[chapter_order].slokas}
        if set(ru_by_order) != set(en_by_order):
            raise ValueError(f"RU/EN sloka order mismatch in chapter {chapter_order}")
        for order in sorted(ru_by_order):
            ru, en = ru_by_order[order], en_by_order[order]
            entries.extend((
                make_authoritative_entry(
                    entry_id=f"ch{chapter_order:02d}-sloka{order:03d}-ru",
                    chapter_order=chapter_order, sloka_order=order, source_sloka_id=ru.id,
                    language_code="ru", content_role="translation", runtime_slot=registry.slot_for("ru"),
                    text=ru.translation_ru,
                ),
                make_authoritative_entry(
                    entry_id=f"ch{chapter_order:02d}-sloka{order:03d}-en",
                    chapter_order=chapter_order, sloka_order=order, source_sloka_id=en.id,
                    language_code="en", content_role="translation", runtime_slot=registry.slot_for("en"),
                    text=en.translation_ru,
                ),
                make_authoritative_entry(
                    entry_id=f"ch{chapter_order:02d}-sloka{order:03d}-sa",
                    chapter_order=chapter_order, sloka_order=order, source_sloka_id=ru.id,
                    language_code="sa", content_role="sanskrit", runtime_slot=None,
                    text=ru.sanskrit,
                ),
            ))
    return tuple(entries)


def validate_glyph_layout(glyph_mask: Image.Image, region_mask: Image.Image) -> LayoutGate:
    glyph = np.asarray(glyph_mask.convert("L")) > 0
    region = np.asarray(region_mask.convert("L")) > 0
    if glyph.shape != region.shape:
        raise ValueError("glyph and region masks must have identical geometry")
    glyph_pixels = int(glyph.sum())
    collisions = int((glyph & ~region).sum())
    foreground_fraction = glyph_pixels / max(1, int(region.sum()))
    fit = glyph_pixels > 0 and collisions == 0 and .001 <= foreground_fraction <= .65
    return LayoutGate(glyph_pixels, collisions, foreground_fraction, fit)


def exact_readback_match(authoritative: str, readback: str) -> bool:
    return normalize_readback(authoritative) == normalize_readback(readback)


def _font_for(entry: AuthoritativeLettering) -> Path:
    return DEVANAGARI_FONT if entry.language_code == "sa" else LATIN_FONT


def _ocr_language(entry: AuthoritativeLettering) -> str:
    return {"ru": "rus", "en": "eng", "sa": "san"}[entry.language_code]


def render_lettering(
    entry: AuthoritativeLettering, region_mask: Image.Image, output_root: Path,
    *, min_font_size: int = 16, max_font_size: int = 56, font_weight: int = 400,
) -> dict:
    from playwright.sync_api import sync_playwright

    mask = region_mask.convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("lettering region mask is empty")
    width, height = mask.size
    x0, y0, x1, y1 = bbox
    inset = 12
    box_width, box_height = max(1, x1 - x0 - 2 * inset), max(1, y1 - y0 - 2 * inset)
    font = _font_for(entry)
    escaped = html.escape(entry.text).replace("\n", "<br>")
    document = f"""<!doctype html><style>
      @font-face {{ font-family: Exact; src: url('{font.resolve().as_uri()}'); font-weight:100 900; }}
      html,body {{ margin:0; width:{width}px; height:{height}px; background:transparent; overflow:hidden; }}
      #box {{ position:absolute; left:{x0 + inset}px; top:{y0 + inset}px; width:{box_width}px;
              height:{box_height}px; display:flex; align-items:center; justify-content:center;
              overflow:hidden; }}
      #text {{ font-family:Exact,sans-serif; color:#000; text-align:center; line-height:1.18;
               font-weight:{font_weight}; overflow-wrap:anywhere; white-space:normal; }}
    </style><div id='box'><div id='text'>{escaped}</div></div>"""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.set_content(document)
        page.wait_for_timeout(50)
        def overflow(size: int) -> bool:
            page.eval_on_selector("#text", "(node,size)=>node.style.fontSize=size+'px'", size)
            return bool(page.eval_on_selector(
                "#text", "node=>node.scrollWidth>node.parentElement.clientWidth+1||"
                "node.scrollHeight>node.parentElement.clientHeight+1"
            ))
        fit_at_minimum = not overflow(min_font_size)
        selected = min_font_size
        if fit_at_minimum:
            low, high = min_font_size, max_font_size
            while low <= high:
                middle = (low + high) // 2
                if overflow(middle):
                    high = middle - 1
                else:
                    selected, low = middle, middle + 1
        overflow(selected)
        screenshot = page.screenshot(omit_background=True)
        browser.close()
    rgba = Image.open(io.BytesIO(screenshot)).convert("RGBA")
    alpha = rgba.getchannel("A")
    gate = validate_glyph_layout(alpha, mask)
    output_root.mkdir(parents=True, exist_ok=True)
    rgba_path = output_root / f"{entry.id}.rgba.png"
    glyph_path = output_root / f"{entry.id}.glyph-mask.png"
    region_path = output_root / f"{entry.id}.region-mask.png"
    rgba.save(rgba_path)
    alpha.save(glyph_path)
    mask.save(region_path)
    with tempfile.TemporaryDirectory() as temporary:
        ocr_input = Path(temporary) / "ocr.png"
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba.convert("RGB"), mask=alpha)
        background.save(ocr_input)
        completed = subprocess.run(
            ["tesseract", str(ocr_input), "stdout", "-l", _ocr_language(entry), "--psm", "6"],
            check=False, capture_output=True, text=True,
        )
    readback = completed.stdout.strip()
    readback_exact = completed.returncode == 0 and exact_readback_match(entry.text, readback)
    accepted = fit_at_minimum and selected >= min_font_size and gate.fit and readback_exact
    return {
        "id": entry.id, "language_code": entry.language_code, "content_role": entry.content_role,
        "runtime_slot": entry.runtime_slot, "authoritative_sha256": entry.normalized_sha256,
        "region_mask_sha256": hashlib.sha256(region_path.read_bytes()).hexdigest(),
        "glyph_mask_sha256": hashlib.sha256(glyph_path.read_bytes()).hexdigest(),
        "rgba_sha256": hashlib.sha256(rgba_path.read_bytes()).hexdigest(),
        "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
        "font_size": selected, "font_weight": font_weight, "fit_at_minimum": fit_at_minimum,
        "layout_gate": asdict(gate), "ocr_engine": "tesseract-5.5.1",
        "ocr_language": _ocr_language(entry), "ocr_readback": readback,
        "exact_readback": readback_exact, "style_stage": "none_after_verified_glyph_mask",
        "decision": "accepted" if accepted else "rejected",
        "failures": [name for condition, name in (
            (not fit_at_minimum, "text_overflow"), (not gate.fit, "glyph_collision_or_readability"),
            (not readback_exact, "ocr_exact_string_mismatch"),
        ) if condition],
        "files": {"rgba": str(rgba_path), "glyph_mask": str(glyph_path),
                  "region_mask": str(region_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    args = parser.parse_args()
    entries = build_golden_lettering_corpus()
    manifest = {
        "schema_version": 1, "normalization": "Unicode NFC + line-ending normalization",
        "runtime_language_slots": {"en": 0, "ru": 1},
        "sanskrit_storage": "content_role_without_fixed_runtime_slot",
        "entry_count": len(entries), "entries": [asdict(item) for item in entries],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    if args.fixtures:
        if not args.fixture_root:
            raise ValueError("--fixture-root is required with --fixtures")
        selected = []
        for chapter in (1, 11):
            for language in ("ru", "en", "sa"):
                selected.append(next(item for item in entries
                                     if item.chapter_order == chapter and item.language_code == language))
        mask = Image.new("L", (1400, 520), 0)
        ImageDraw.Draw(mask).rounded_rectangle((10, 10, 1390, 510), radius=60, fill=255)
        results = [render_lettering(item, mask, args.fixture_root) for item in selected]
        report = {"schema_version": 1, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "fixture_count": len(results), "accepted_count": sum(item["decision"] == "accepted" for item in results),
                  "release_state": "accepted" if all(item["decision"] == "accepted" for item in results) else "blocked",
                  "results": results}
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        with args.fixtures.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps({"entries": len(entries), "fixtures": len(results),
                          "accepted": report["accepted_count"], "release_state": report["release_state"]}))
    else:
        print(json.dumps({"entries": len(entries)}))


if __name__ == "__main__":
    main()
