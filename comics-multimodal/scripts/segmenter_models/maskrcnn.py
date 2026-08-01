"""Task 4.3: instance-segmentation model (Plan Option A) -- fine-tunes torchvision's
maskrcnn_resnet50_fpn_v2 (COCO-pretrained) on the same synthetic training pairs as the Task 4.2
baseline, replacing the classification head for our Kind taxonomy. Chosen over the baseline for
native support of overlapping instance masks (two characters, or a character over a background,
don't collapse into one blob the way the U-Net's flat label map does) and because a pretrained
backbone meaningfully de-risks learning from only ~750 real-photo-adjacent synthetic samples
(Specifications' resolved "trained from scratch" scope: pretrained weights are permitted, Plan
Task 4.1).

torchvision's detection models reserve class 0 for "no object" internally -- our own
`dataset.KIND_TO_LABEL` (art=0, background=1, character=2, balloon=3) is shifted by +1 here
(`NUM_DETECTION_CLASSES = 5`) so as not to conflate our "art" category with that reserved class.

Ground truth is rectangle-only everywhere in this pipeline (no per-pixel masks -- see dataset.py),
so `to_detection_target` builds box-shaped binary masks: a documented approximation, not an
oversight.
"""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import KIND_TO_LABEL

NUM_KINDS = len(KIND_TO_LABEL)
NUM_DETECTION_CLASSES = NUM_KINDS + 1  # +1 for torchvision's reserved "no object" class 0


def build_model(pretrained: bool = True):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_DETECTION_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, NUM_DETECTION_CLASSES
    )

    return model


def to_detection_target(target: dict, crop_size: tuple[int, int]) -> dict:
    """Convert a dataset.py target dict (0-indexed Kind `labels`, rectangle `boxes`) into
    torchvision detection's expected shape: labels shifted +1, plus box-shaped binary masks.
    Boxes with zero/negative area after clipping are dropped (torchvision's training loop asserts
    all boxes have positive area).
    """
    w, h = crop_size
    raw_boxes = target["boxes"]
    raw_labels = target["labels"] + 1

    kept_boxes, kept_labels, masks = [], [], []
    for box, label in zip(raw_boxes, raw_labels):
        x0, y0, x1, y1 = (int(round(v)) for v in box.tolist())
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = torch.zeros((h, w), dtype=torch.uint8)
        mask[y0:y1, x0:x1] = 1
        kept_boxes.append([x0, y0, x1, y1])
        kept_labels.append(int(label))
        masks.append(mask)

    boxes_t = torch.tensor(kept_boxes, dtype=torch.float32) if kept_boxes else torch.zeros((0, 4))
    labels_t = torch.tensor(kept_labels, dtype=torch.int64)
    masks_t = torch.stack(masks) if masks else torch.zeros((0, h, w), dtype=torch.uint8)
    area = (
        (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        if len(boxes_t)
        else torch.zeros((0,))
    )

    return {
        "boxes": boxes_t,
        "labels": labels_t,
        "masks": masks_t,
        "area": area,
        "iscrowd": torch.zeros(len(kept_boxes), dtype=torch.int64),
    }
