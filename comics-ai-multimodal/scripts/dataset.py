"""Task 3.3: torch Dataset wrapping augment.py's training pairs (work/train_pairs/manifest.jsonl),
yielding (degraded_image_tensor, target) pairs for the segmentation model (Task 4.x).

`target` is a dict, not a single tensor, since a training pair's ground truth is a variable-length
list of regions (one per layer in the cluster) each with its own box/kind -- the shape a
Mask-R-CNN-style instance-segmentation model expects (torchvision's detection models take exactly
this "list of per-image target dicts" convention), and adaptable for the semantic-segmentation
baseline (Task 4.2) by rasterizing `boxes`+`labels` into a single label map at train time instead
(see `models/unet_baseline.py`'s `rasterize_label_map`). Ground truth is rectangle-only (no
per-pixel masks) throughout this pipeline -- a documented simplification, not an oversight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_MANIFEST = WORK_DIR / "train_pairs" / "manifest.jsonl"

# Kept in sync with kind_heuristic.py's vocabulary (plus "art" as the catch-all fallback).
KIND_TO_LABEL = {"art": 0, "background": 1, "character": 2, "balloon": 3}
LABEL_TO_KIND = {v: k for k, v in KIND_TO_LABEL.items()}


def _load_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class TrainingPairDataset(Dataset):
    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"{self.manifest_path} not found -- run augment.py first (Task 3.2)"
            )
        self.entries = _load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        entry = self.entries[idx]

        degraded = Image.open(entry["degraded_png"]).convert("RGB")
        image_tensor = torch.from_numpy(np.array(degraded)).permute(2, 0, 1).float() / 255.0

        x0, y0, x1, y1 = entry["bbox"]
        width, height = x1 - x0, y1 - y0

        kinds = entry["kinds"]
        labels = torch.tensor(
            [KIND_TO_LABEL.get(k, KIND_TO_LABEL["art"]) for k in kinds], dtype=torch.int64
        )
        # region_bboxes are crop-local, POST-degradation coordinates (augment.py's
        # degrade_with_boxes carries them through the same rotation+perspective transform applied
        # to the pixels), parallel to layer_indexes/labels -- already in the coordinate space of
        # `image_tensor`, no further transform needed here.
        boxes = torch.tensor(entry["region_bboxes"], dtype=torch.float32)
        target = {
            "episode_file": entry["episode_file"],
            "cluster_index": entry["cluster_index"],
            "layer_indexes": torch.tensor(entry["layer_indexes"], dtype=torch.int64),
            "labels": labels,
            "boxes": boxes,
            "crop_size": (width, height),
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]):
    """Instance-segmentation-style batches can't be stacked into one tensor (variable image sizes,
    variable region counts) -- return parallel lists, matching torchvision detection models'
    expected input shape (`list[Tensor]`, `list[dict]`).
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
