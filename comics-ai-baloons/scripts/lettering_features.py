#!/usr/bin/env python3
"""Stage 5.1: hand-lettering signal extraction.

Manual review of a 39-balloon stratified sample (Task 5.2) found only two distinct *uniform*
digitally-set fonts across every file/era in the dataset -- no obvious hand lettering. This module
exists to check that conclusion far more broadly than manual sampling can: it computes cheap,
interpretable outlier signals across *all* balloons so the small number of highest-scoring
candidates can be visually spot-checked directly, rather than trusting a random sample to have
found rare cases.

Two signals, both computed on the outline-stripped text region (reusing ocr.strip_outline_component
so the balloon's drawn border doesn't dominate the measurement):

1. stroke_width_cv: coefficient of variation (std/mean) of stroke widths, estimated via a distance
   transform on the binarized glyph mask. A uniform font has consistent stroke width; hand lettering
   varies more, including within a single word.
2. baseline_wobble: for each detected text line, the standard deviation of connected-component
   vertical centers, normalized by median glyph height. Set type sits on a straight baseline; hand
   lettering wobbles.

Plus two free signals already computed in stage 3 (OCR): `ocr_confidence` (lower on hand lettering,
empirically, since Tesseract is trained on printed text) and `needed_crop_fallback` (the outline
confused Tesseract enough to need the fallback crop -- could correlate with irregular lettering).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import ocr as ocr_module

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_EXTRACTED_ROOT = WORK_DIR / "extracted"
DEFAULT_OCR_JSONL = WORK_DIR / "ocr.jsonl"
DEFAULT_OUT = WORK_DIR / "lettering_features.jsonl"


@dataclass
class LetteringFeatures:
    source_file: str
    layer_index: int
    lang_index: int
    stroke_width_cv: float
    baseline_wobble: float
    ocr_confidence: float
    needed_crop_fallback: bool


def _text_region(img: Image.Image) -> Image.Image:
    flat = ocr_module._flatten_to_white(img)
    cropped = ocr_module.strip_outline_component(flat)
    return cropped if cropped is not None else flat


def stroke_width_cv(binary_mask: np.ndarray) -> float:
    """Coefficient of variation of stroke width, via distance transform on the glyph mask.
    `binary_mask` is uint8, 255 = foreground (ink), 0 = background.
    """
    if binary_mask.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    # Local maxima of the distance transform approximate half the local stroke width. Skeletonizing
    # properly is overkill for a ranking signal -- just take the ridge via a max filter comparison.
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(dist, kernel)
    ridge = (dist == dilated) & (dist > 0.5)
    widths = dist[ridge] * 2.0
    if widths.size < 4:
        return 0.0
    mean = float(widths.mean())
    if mean <= 0:
        return 0.0
    return float(widths.std() / mean)


def baseline_wobble(binary_mask: np.ndarray) -> float:
    """Std of connected-component vertical centers within each text line, normalized by median
    glyph height, averaged across lines.
    """
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if n <= 1:
        return 0.0
    comps = [
        (centroids[i][1], stats[i][3])  # (cy, height)
        for i in range(1, n)
        if stats[i][4] >= 3  # drop noise specks
    ]
    if len(comps) < 3:
        return 0.0

    # Cluster components into text lines via 1D agglomeration on cy (gap > median height -> new line)
    comps.sort(key=lambda c: c[0])
    heights = [h for _, h in comps]
    median_h = float(np.median(heights)) or 1.0

    lines: list[list[tuple[float, float]]] = [[comps[0]]]
    for cy, h in comps[1:]:
        if cy - lines[-1][-1][0] > median_h * 0.8:
            lines.append([])
        lines[-1].append((cy, h))

    wobbles = []
    for line in lines:
        if len(line) < 3:
            continue
        cys = np.array([c[0] for c in line])
        wobbles.append(float(cys.std()) / median_h)
    return float(np.mean(wobbles)) if wobbles else 0.0


def compute_features_for_image(path: Path) -> tuple[float, float]:
    img = Image.open(path)
    region = _text_region(img)
    arr = np.array(region.convert("L"))
    _, binary = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY_INV)
    return stroke_width_cv(binary), baseline_wobble(binary)


def run(
    ocr_results_path: Path, extracted_root: Path
) -> list[LetteringFeatures]:
    out = []
    with ocr_results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            file_stem = Path(r["source_file"]).stem
            lang_label = ocr_module.LANG_TO_TESSERACT
            # reconstruct the extracted path the same way extract.py named it
            import languages as languages_module

            lang_code = (
                languages_module.index_to_lang(r["lang_index"])
                if r["lang_index"] < len(languages_module.LANGUAGES)
                else f"idx{r['lang_index']}"
            )
            img_path = extracted_root / file_stem / f"layer_{r['layer_index']}_{lang_code}.png"
            if not img_path.exists():
                continue
            swcv, wobble = compute_features_for_image(img_path)
            out.append(
                LetteringFeatures(
                    source_file=r["source_file"],
                    layer_index=r["layer_index"],
                    lang_index=r["lang_index"],
                    stroke_width_cv=swcv,
                    baseline_wobble=wobble,
                    ocr_confidence=r["confidence"],
                    needed_crop_fallback=r["needed_crop_fallback"],
                )
            )
    return out


def write_jsonl(features: list[LetteringFeatures], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(asdict(feat), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocr", default=str(DEFAULT_OCR_JSONL))
    ap.add_argument("--extracted-root", default=str(DEFAULT_EXTRACTED_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    features = run(Path(args.ocr), Path(args.extracted_root))
    write_jsonl(features, Path(args.out))
    print(f"Computed lettering features for {len(features)} slots -> {args.out}")


if __name__ == "__main__":
    main()
