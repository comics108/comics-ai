#!/usr/bin/env python3
"""Task 3.1 (Checkpoint B): measure real comics_book_lowcamera/*.jpg characteristics so
augment.py's synthetic degradation is calibrated from real data, not guessed.

Metrics:
- sharpness: variance of the Laplacian (higher = sharper; standard, fast proxy -- not a physical
  blur-kernel-radius measurement, but sufficient to rank/calibrate relative blur amounts)
- noise sigma: Immerkær's fast single-image noise estimator (a fixed 3x3 mask convolution;
  standard technique, doesn't require a second reference image)
- resolution / aspect ratio: sanity-check these really are all the same camera/setup
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
LOWCAMERA_DIR = (
    REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)

# Immerkær (1996) fast noise estimation mask.
_NOISE_MASK = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def noise_sigma(gray: np.ndarray) -> float:
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float64), -1, _NOISE_MASK)
    sigma = math.sqrt(math.pi / 2) * (1.0 / (6 * (w - 2) * (h - 2))) * np.sum(np.abs(conv))
    return float(sigma)


def analyze_file(path: Path) -> dict:
    im = cv2.imread(str(path))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return {
        "file": path.name,
        "width": w,
        "height": h,
        "sharpness": sharpness_score(gray),
        "noise_sigma": noise_sigma(gray),
    }


def analyze_all(lowcamera_dir: Path = LOWCAMERA_DIR) -> list[dict]:
    files = sorted(lowcamera_dir.glob("*.jpg"))
    return [analyze_file(f) for f in files]


def summarize(results: list[dict]) -> dict:
    sharpness = sorted(r["sharpness"] for r in results)
    noise = sorted(r["noise_sigma"] for r in results)

    def pct(sorted_vals, p):
        idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p))
        return sorted_vals[idx]

    return {
        "count": len(results),
        "sharpness": {
            "min": sharpness[0],
            "p25": pct(sharpness, 0.25),
            "median": pct(sharpness, 0.5),
            "p75": pct(sharpness, 0.75),
            "max": sharpness[-1],
        },
        "noise_sigma": {
            "min": noise[0],
            "p25": pct(noise, 0.25),
            "median": pct(noise, 0.5),
            "p75": pct(noise, 0.75),
            "max": noise[-1],
        },
        "resolutions": sorted({(r["width"], r["height"]) for r in results}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "work" / "photo_analysis.json")
    args = parser.parse_args()

    results = analyze_all()
    summary = summarize(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_file": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
