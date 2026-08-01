#!/usr/bin/env python3
"""vdd-comics-editor-ai-uiux, Task 1.2: single-image entry point for the comics-editor's new
Cutting mode. Unlike every other script in this pipeline, this one segments one ad hoc image
handed to it by the editor -- not a page from a known, already-matched book (see
flows/vdd-comics-editor-ai-uiux/03-specifications.md, Finding 5: `pipeline.py` has no equivalent
single-image entry point). Prints NDJSON events to stdout so a Dart subprocess client
(`ProcessCuttingClient`) can stream routing/progress/results the same way `CoreClient` already
does for the native editor core.

Protocol (see 03-specifications.md's Interfaces section):
  {"event": "routing", "on_device": true, "reason": null}
  {"event": "progress", "stage": "loading_model" | "segmenting" | "extracting_regions"}
  {"event": "success", "regions": [{"kind", "confidence", "bbox": [x0,y0,x1,y1], "crop_png_base64"}]}
  -- or, for an anticipated failure --
  {"event": "failure", "reason": "...", "retryable": bool}

Anticipated failures (missing checkpoint, unreadable image) emit a "failure" event and exit 0 --
a clean, expected outcome, not a crash. An unanticipated exception is deliberately left
uncaught: it prints a Python traceback to stderr and exits non-zero with no "success"/"failure"
line on stdout, which the Dart client maps to `Failure(reason: "process_error", retryable: true)`
-- there is no value in this script trying to re-describe an error it didn't anticipate.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image as PILImage

from infer_segmenter import DEFAULT_CHECKPOINT, infer_regions_with_crops, load_model


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def encode_crop_png(crop_rgb) -> str:
    buf = BytesIO()
    PILImage.fromarray(crop_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _default_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(image_path: Path, checkpoint_path: Path, device: str | None = None) -> int:
    # Always on-device this iteration -- there is no server path (Requirements Won't Have), but
    # the event still fires so the Dart contract/UI don't special-case "no routing event" as a
    # distinct state (see 03-specifications.md's Interfaces note on this).
    emit({"event": "routing", "on_device": True, "reason": None})

    if not checkpoint_path.is_file():
        emit({"event": "failure", "reason": "model_checkpoint_not_found", "retryable": False})
        return 0

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        emit({"event": "failure", "reason": "image_not_readable", "retryable": False})
        return 0

    emit({"event": "progress", "stage": "loading_model"})
    resolved_device = device or _default_device()
    model = load_model(checkpoint_path, resolved_device)

    emit({"event": "progress", "stage": "segmenting"})
    emit({"event": "progress", "stage": "extracting_regions"})
    results = infer_regions_with_crops(model, image_bgr, resolved_device)

    regions = [
        {
            "kind": kind,
            "confidence": confidence,
            "bbox": list(bbox),
            "crop_png_base64": encode_crop_png(crop),
        }
        for kind, confidence, bbox, crop in results
    ]
    emit({"event": "success", "regions": regions})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    return run(args.image, args.checkpoint, args.device)


if __name__ == "__main__":
    sys.exit(main())
