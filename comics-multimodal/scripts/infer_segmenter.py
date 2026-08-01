#!/usr/bin/env python3
"""Task 6.1: run the trained segmentation model (U-Net baseline, per Phase 4's working-baseline
conclusion -- see flows/sdd-comics-ai-multimodal/04-implementation-log.md) on real, matched photos
from work/alignment.jsonl, deriving discrete predicted regions via connected-components per
predicted class (Plan Task 4.2's stated instance-derivation approach for the semantic-segmentation
baseline).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import baloons_bridge
from dataset import LABEL_TO_KIND
from detect_panels import detect_pages
from rectify import rectify_page
from segmenter_models.unet_baseline import UNetBaseline

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_CHECKPOINT = WORK_DIR / "models" / "unet_baseline.pt"
DEFAULT_ALIGNMENT = WORK_DIR / "alignment.jsonl"
DEFAULT_LOWCAMERA_DIR = (
    baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)
DEFAULT_OUT = WORK_DIR / "regions.jsonl"

TRAIN_SIZE = (256, 256)  # (H, W), matches train_segmenter.py's fixed inference resolution
MIN_REGION_AREA = 200  # pixels at TRAIN_SIZE resolution -- filters speckle/noise blobs


@dataclass
class CutRegion:
    photo_file: str
    page_index: int
    predicted_kind: str
    confidence: float
    bbox: tuple[int, int, int, int]  # in TRAIN_SIZE-resolution coordinates (256x256)


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cpu") -> UNetBaseline:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNetBaseline(num_classes=len(LABEL_TO_KIND))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def infer_regions(
    model: UNetBaseline, image_bgr: np.ndarray, device: str = "cpu"
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Returns (kind, confidence, bbox) tuples in TRAIN_SIZE-resolution coordinates."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    th, tw = TRAIN_SIZE
    resized = cv2.resize(rgb, (tw, th))  # cv2.resize wants (width, height)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()  # (C, H, W) -- move off MPS before any .numpy()
        pred = probs.argmax(dim=0).numpy()

    regions = []
    for class_idx, kind in LABEL_TO_KIND.items():
        mask = (pred == class_idx).astype(np.uint8)
        if mask.sum() == 0:
            continue
        num_labels, labels = cv2.connectedComponents(mask)
        for label_id in range(1, num_labels):
            component = labels == label_id
            area = int(component.sum())
            if area < MIN_REGION_AREA:
                continue
            ys, xs = np.where(component)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            conf = float(probs[class_idx].numpy()[component].mean())
            regions.append((kind, conf, (x0, y0, x1, y1)))
    return regions


def infer_regions_with_crops(
    model: UNetBaseline, image_bgr: np.ndarray, device: str = "cpu"
) -> list[tuple[str, float, tuple[int, int, int, int], np.ndarray]]:
    """vdd-comics-editor-ai-uiux, Task 1.1: like infer_regions, but for a single ad hoc image (not
    a known book page) the caller needs region geometry usable directly against image_bgr -- not
    TRAIN_SIZE-resized coordinates, which is all infer_regions returns. Rescales each bbox back to
    image_bgr's real pixel dimensions and slices out that region's rectangular RGB crop, since
    nothing upstream (regions.jsonl, build_library.py) has ever needed to do either of these for a
    region in isolation before (both only ever re-derive the same TRAIN_SIZE-resolution crop
    internally, for a *known* book photo/page/bbox triple looked up from alignment.jsonl).

    Returns (kind, confidence, bbox, crop) tuples; bbox is (x0, y0, x1, y1) in image_bgr's own
    coordinate space; crop is RGB (not BGR) since that's what gets PNG-encoded downstream.
    """
    raw_regions = infer_regions(model, image_bgr, device)
    orig_h, orig_w = image_bgr.shape[:2]
    train_h, train_w = TRAIN_SIZE
    scale_x = orig_w / train_w
    scale_y = orig_h / train_h
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = []
    for kind, conf, (x0, y0, x1, y1) in raw_regions:
        rx0 = int(round(x0 * scale_x))
        ry0 = int(round(y0 * scale_y))
        rx1 = int(round(x1 * scale_x))
        ry1 = int(round(y1 * scale_y))
        # Rounding can push the far edge a hair past the real image bounds -- clip, then make sure
        # the near edge still leaves at least one pixel of crop rather than producing an empty slice.
        rx1 = min(rx1, orig_w)
        ry1 = min(ry1, orig_h)
        rx0 = max(0, min(rx0, rx1 - 1))
        ry0 = max(0, min(ry0, ry1 - 1))
        crop = rgb[ry0:ry1, rx0:rx1]
        results.append((kind, conf, (rx0, ry0, rx1, ry1), crop))
    return results


def infer_photo_page(
    model: UNetBaseline,
    photo_path: Path,
    page_index: int,
    device: str = "cpu",
) -> list[CutRegion]:
    """Re-detects and re-crops the same page from the photo that align_photo.py matched (alignment
    results store the match, not the pixel crop, so this recomputes the crop deterministically the
    same way align_photo.py did).
    """
    image = cv2.imread(str(photo_path))
    if image is None:
        return []
    pages = detect_pages(image)
    if page_index >= len(pages):
        return []
    x0, y0, x1, y1 = pages[page_index].bbox
    crop = image[y0:y1, x0:x1]
    rect_result = rectify_page(crop)

    raw_regions = infer_regions(model, rect_result.rectified, device)
    return [
        CutRegion(photo_file=photo_path.name, page_index=page_index, predicted_kind=k, confidence=c, bbox=b)
        for k, c, b in raw_regions
    ]


def infer_all(
    alignment_path: Path = DEFAULT_ALIGNMENT,
    lowcamera_dir: Path = DEFAULT_LOWCAMERA_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    out_path: Path = DEFAULT_OUT,
    device: str | None = None,
) -> list[CutRegion]:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    all_regions: list[CutRegion] = []
    with alignment_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["status"] != "matched":
                continue
            photo_path = lowcamera_dir / entry["photo_file"]
            regions = infer_photo_page(model, photo_path, entry["page_index"], device)
            all_regions.extend(regions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in all_regions:
            f.write(json.dumps(asdict(r)) + "\n")
    return all_regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    regions = infer_all(args.alignment, checkpoint_path=args.checkpoint, out_path=args.out, device=args.device)
    print(f"{len(regions)} regions inferred -> {args.out}")


if __name__ == "__main__":
    main()
