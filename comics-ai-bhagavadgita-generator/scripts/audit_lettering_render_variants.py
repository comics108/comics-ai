#!/usr/bin/env python3
"""Render reproducible shipped-font variants and require exact Tesseract readback."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from lettering import exact_readback_match


VARIANTS = tuple(
    (f"w{weight}-s{size}-{align}", weight, size, 0.0, align)
    for weight in (400, 500, 600, 700, 800, 900)
    for size in (44, 48, 52, 56, 60, 64)
    for align in ("center", "left")
) + (("w400-s56-spaced-center", 400, 56, 0.35, "center"),)
OCR_LANGUAGES = {
    "en": ("eng", "script/Latin"),
    "sa": ("san", "san+hin", "script/Devanagari"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(authoritative_manifest: Path, fixture_manifest: Path, output_root: Path) -> dict:
    from playwright.sync_api import sync_playwright

    authoritative = json.loads(authoritative_manifest.read_text(encoding="utf-8"))
    fixtures = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    entries = {item["id"]: item for item in authoritative["entries"]}
    rejected = [item for item in fixtures["results"] if item["decision"] == "rejected"]
    fonts = Path(__file__).resolve().parents[1] / "fonts/Noto"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 520})
        for fixture in rejected:
            entry = entries[fixture["id"]]
            escaped = html.escape(entry["text"]).replace("\n", "<br>")
            for variant_id, weight, size, spacing, align in VARIANTS:
                font = (fonts / "NotoSansDevanagari[wdth,wght].ttf" if fixture["language_code"] == "sa"
                        else fonts / ("NotoSans-Bold.ttf" if weight >= 600 else "NotoSans-Regular.ttf"))
                document = f"""<!doctype html><style>
                @font-face {{font-family:Exact;src:url('{font.resolve().as_uri()}')}}
                html,body{{margin:0;width:1400px;height:520px;background:white;overflow:hidden}}
                #text{{position:absolute;inset:22px;display:flex;align-items:center;justify-content:center;
                text-align:{align};font-family:Exact,sans-serif;font-size:{size}px;line-height:1.18;
                font-weight:{weight};letter-spacing:{spacing}px;white-space:normal}}
                </style><div id='text'><span>{escaped}</span></div>"""
                page.set_content(document)
                page.wait_for_timeout(30)
                image_path = output_root / f"{fixture['id']}-{variant_id}.png"
                image_path.write_bytes(page.screenshot())
                for language in OCR_LANGUAGES[fixture["language_code"]]:
                    for psm in (6, 11):
                        completed = subprocess.run(
                            ["tesseract", str(image_path), "stdout", "-l", language,
                             "--psm", str(psm)], capture_output=True, text=True, check=False,
                        )
                        readback = completed.stdout.strip()
                        exact = completed.returncode == 0 and exact_readback_match(entry["text"], readback)
                        rows.append({
                            "id": fixture["id"], "variant": variant_id, "font_sha256": _sha256(font),
                            "font_weight": weight, "font_size": size, "letter_spacing_px": spacing,
                            "text_align": align,
                            "ocr_language": language, "psm": psm, "readback": readback,
                            "returncode": completed.returncode, "exact": exact,
                            "render_file": str(image_path), "render_sha256": _sha256(image_path),
                        })
        browser.close()
    exact = [row for row in rows if row["exact"]]
    covered = {row["id"] for row in exact}
    return {
        "schema_version": 1,
        "authoritative_manifest_sha256": _sha256(authoritative_manifest),
        "fixture_manifest_sha256": _sha256(fixture_manifest),
        "constraints": ["shipped_fonts_only", "no_custom_words", "no_fuzzy_matching",
                        "no_authoritative_postprocessing"],
        "variant_count": len(rows), "exact_match_count": len(exact),
        "fixture_exact_coverage": sorted(covered),
        "decision": "accepted" if len(covered) == len(rejected) else "rejected",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.authoritative, args.fixtures, args.render_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("decision", "variant_count", "exact_match_count", "fixture_exact_coverage")}))


if __name__ == "__main__":
    main()
