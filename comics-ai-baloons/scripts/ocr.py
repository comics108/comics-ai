#!/usr/bin/env python3
"""Stage 3: OCR the extracted en/ru balloon renders with Tesseract.

Only en/ru are OCR'd -- those are the only baked-in languages that actually exist in the dataset
(Hindi's reserved slot is empty in all 825 layers, per Specifications). OCR text + confidence feed
both the CSV matcher (stage 4) and, as a free extra signal, the hand-lettering classifier
(stage 5).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output

import languages
from imaging import flatten_to_white
from models import OcrResult

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_MANIFEST = WORK_DIR / "extracted.jsonl"
DEFAULT_EXTRACTED_ROOT = WORK_DIR / "extracted"
DEFAULT_OUT = WORK_DIR / "ocr.jsonl"

# Only these two currently exist as baked-in languages in the dataset.
LANG_TO_TESSERACT = {"en": "eng", "ru": "rus"}


def read_manifest(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


TESSERACT_CONFIG = "--psm 6"  # "assume a single uniform block of text"
#
# Default automatic page-segmentation (psm 3) was found empirically to return empty text on a
# meaningful fraction of otherwise perfectly legible balloons -- the balloon's drawn outline
# confuses Tesseract's layout analysis into discarding the region entirely. Balloon crops are, by
# construction, a single block of text with no multi-column layout to detect, so psm 6 is the
# correct mode here, not a workaround for a one-off image.


def _flatten_to_white(img: Image.Image) -> Image.Image:
    return flatten_to_white(img)


def strip_outline_component(img: Image.Image) -> Image.Image | None:
    """Crop out the balloon's drawn outline before OCR, keeping only the text glyphs.

    Empirically, Tesseract's layout analysis (even under --psm 6) sometimes discards a balloon's
    text entirely on short/tightly-cropped balloons, because the outline stroke -- a single large
    connected component spanning most of the image -- gets mistaken for page structure. This finds
    connected dark components, drops whichever ones span >85% of the image width or height (the
    outline/tail stroke), and crops to the union bounding box of what's left (presumed glyphs).
    Returns None if nothing is left after filtering (no text found at all).
    """
    arr = np.array(img.convert("L"))
    _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h, w = arr.shape

    keep_boxes = []
    for i in range(1, n):  # label 0 is background
        x, y, cw, ch, area = stats[i]
        if cw > 0.85 * w or ch > 0.85 * h:
            continue  # outline/tail stroke
        if area < 3:
            continue  # noise speck
        keep_boxes.append((x, y, x + cw, y + ch))

    if not keep_boxes:
        return None

    pad = 4
    x0 = max(0, min(b[0] for b in keep_boxes) - pad)
    y0 = max(0, min(b[1] for b in keep_boxes) - pad)
    x1 = min(w, max(b[2] for b in keep_boxes) + pad)
    y1 = min(h, max(b[3] for b in keep_boxes) + pad)
    return img.crop((x0, y0, x1, y1))


def _ocr_once(img: Image.Image, tess_lang: str) -> tuple[str, float]:
    data = pytesseract.image_to_data(
        img, lang=tess_lang, config=TESSERACT_CONFIG, output_type=Output.DICT
    )
    words: list[str] = []
    confs: list[float] = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        words.append(text)
        try:
            c = float(conf)
        except ValueError:
            c = -1.0
        if c >= 0:
            confs.append(c)
    full_text = " ".join(words)
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return full_text, avg_conf


def ocr_image(path: Path, tess_lang: str) -> tuple[str, float, bool]:
    """Returns (text, confidence, needed_crop_fallback)."""
    img = _flatten_to_white(Image.open(path))
    text, conf = _ocr_once(img, tess_lang)
    if text:
        return text, conf, False

    cropped = strip_outline_component(img)
    if cropped is None or cropped.size[0] == 0 or cropped.size[1] == 0:
        return text, conf, False

    text2, conf2 = _ocr_once(cropped, tess_lang)
    if text2:
        return text2, conf2, True
    return text, conf, False


def run_ocr(manifest_entries: list[dict], extracted_root: Path) -> list[OcrResult]:
    results: list[OcrResult] = []
    for entry in manifest_entries:
        for lang_idx_str, rel_path in entry["paths"].items():
            lang_idx = int(lang_idx_str)
            lang_code = (
                languages.index_to_lang(lang_idx)
                if lang_idx < len(languages.LANGUAGES)
                else None
            )
            tess_lang = LANG_TO_TESSERACT.get(lang_code) if lang_code else None
            if tess_lang is None:
                continue
            text, conf, needed_fallback = ocr_image(extracted_root / rel_path, tess_lang)
            results.append(
                OcrResult(
                    source_file=entry["source_file"],
                    layer_index=entry["layer_index"],
                    lang_index=lang_idx,
                    text=text,
                    confidence=conf,
                    needed_crop_fallback=needed_fallback,
                )
            )
    return results


def write_jsonl(results: list[OcrResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--extracted-root", default=str(DEFAULT_EXTRACTED_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    entries = read_manifest(Path(args.manifest))
    results = run_ocr(entries, Path(args.extracted_root))
    write_jsonl(results, Path(args.out))

    n_empty = sum(1 for r in results if not r.text)
    n_fallback = sum(1 for r in results if r.needed_crop_fallback)
    avg_conf = sum(r.confidence for r in results) / len(results) if results else 0.0
    print(
        f"OCR'd {len(results)} slots -> {args.out} "
        f"({n_empty} empty, {n_fallback} recovered via outline-strip fallback, "
        f"avg confidence {avg_conf:.2f})"
    )


if __name__ == "__main__":
    main()
