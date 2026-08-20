#!/usr/bin/env python3
"""Audit exact Sanskrit OCR across deterministic word-preserving line breaks."""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import subprocess
from pathlib import Path

from lettering import exact_readback_match


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_variants(text: str) -> tuple[tuple[int, ...], ...]:
    words = text.split()
    boundaries = range(1, len(words))
    return tuple(itertools.chain(itertools.combinations(boundaries, 2),
                                 itertools.combinations(boundaries, 3)))


def _with_breaks(words: list[str], breaks: tuple[int, ...]) -> str:
    pieces, start = [], 0
    for end in breaks + (len(words),):
        pieces.append(" ".join(words[start:end]))
        start = end
    return "\n".join(pieces)


def audit(authoritative_path: Path, candidate_id: str, output_root: Path) -> dict:
    from playwright.sync_api import sync_playwright

    payload = json.loads(authoritative_path.read_text(encoding="utf-8"))
    entry = next(item for item in payload["entries"] if item["id"] == candidate_id)
    if entry["language_code"] != "sa":
        raise ValueError("line-break audit is restricted to Sanskrit")
    words = entry["text"].split()
    font = Path(__file__).resolve().parents[1] / "fonts/Noto/NotoSansDevanagari[wdth,wght].ttf"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 520})
        for index, breaks in enumerate(line_variants(entry["text"])):
            rendered_text = _with_breaks(words, breaks)
            escaped = html.escape(rendered_text).replace("\n", "<br>")
            document = f"""<!doctype html><style>
            @font-face{{font-family:Exact;src:url('{font.resolve().as_uri()}');font-weight:100 900}}
            html,body{{margin:0;width:1400px;height:520px;background:white;overflow:hidden}}
            #text{{position:absolute;inset:22px;display:flex;align-items:center;justify-content:center;
            text-align:center;font-family:Exact,sans-serif;font-size:48px;line-height:1.32;
            font-weight:600;white-space:nowrap}}</style><div id='text'><span>{escaped}</span></div>"""
            page.set_content(document)
            page.wait_for_timeout(20)
            image_path = output_root / f"{candidate_id}-{index:03}.png"
            image_path.write_bytes(page.screenshot())
            for language in ("san", "san+hin", "script/Devanagari"):
                for psm in (6, 11):
                    completed = subprocess.run(
                        ["tesseract", str(image_path), "stdout", "-l", language,
                         "--psm", str(psm)], capture_output=True, text=True, check=False,
                    )
                    readback = completed.stdout.strip()
                    rows.append({
                        "breaks_after_words": list(breaks), "rendered_text": rendered_text,
                        "ocr_language": language, "psm": psm, "readback": readback,
                        "returncode": completed.returncode,
                        "exact": completed.returncode == 0 and exact_readback_match(entry["text"], readback),
                        "render_file": str(image_path), "render_sha256": _sha256(image_path),
                    })
        browser.close()
    exact = [row for row in rows if row["exact"]]
    return {
        "schema_version": 1, "candidate_id": candidate_id,
        "authoritative_manifest_sha256": _sha256(authoritative_path),
        "font_sha256": _sha256(font), "font_weight": 600, "font_size": 48,
        "constraints": ["word_order_preserved", "unicode_text_unchanged", "no_custom_words",
                        "no_fuzzy_matching", "no_ocr_postprocessing"],
        "layout_count": len(line_variants(entry["text"])), "attempt_count": len(rows),
        "exact_match_count": len(exact), "decision": "accepted" if exact else "rejected",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.authoritative, args.candidate_id, args.render_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("decision", "layout_count", "attempt_count", "exact_match_count")}))


if __name__ == "__main__":
    main()
