"""Task 5.1 (flows/sdd-comics-ai-positioning/03-plan.md): shared feature engineering for the
learned positioner, used identically by train_positioner.py and infer_positioner.py so train/serve
skew isn't possible by construction.

Kinds fixed to the 4 the segmenter actually predicts (infer_segmenter.py's real output vocabulary,
confirmed in Task 2.2) plus "art" as the untyped fallback -- an unseen kind still gets a valid
(all-zero) one-hot encoding rather than crashing, same defensive posture as baseline_position.py's
_height_for_kind fallback.

`text_context_length` (chars of the page-cluster's own real OCR'd dialogue, scene_text.py) replaces
an earlier `has_text_context` boolean: once scene_text.py's broad-coverage rework made real text
context present on effectively every training pair (392/392, not the original 127/392 from the
narrower spiritual_text-only signal), a presence flag stopped carrying any information (it's true
almost everywhere) -- length is a real, still-crude but non-constant proxy for "how much dialogue is
happening in this scene."
"""

from __future__ import annotations

TRAIN_SIZE = 256
KNOWN_KINDS = ["background", "character", "balloon", "art"]


def _kind_one_hot(kind: str) -> list[float]:
    return [1.0 if kind == k else 0.0 for k in KNOWN_KINDS]


def feature_names() -> list[str]:
    return [
        *[f"kind_{k}" for k in KNOWN_KINDS],
        "local_width",
        "local_height",
        "local_center_x",
        "local_center_y",
        "reading_order_index",
        "page_region_count",
        "reading_order_fraction",
        "match_confidence",
        "text_context_length",
    ]


def build_features(
    kind: str,
    local_bbox: tuple[int, int, int, int],
    reading_order_index: int,
    page_region_count: int,
    match_confidence: float,
    text_context_length: int = 0,
) -> list[float]:
    x0, y0, x1, y1 = local_bbox
    width, height = x1 - x0, y1 - y0
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    fraction = reading_order_index / page_region_count if page_region_count > 0 else 0.0

    return [
        *_kind_one_hot(kind),
        float(width),
        float(height),
        float(center_x),
        float(center_y),
        float(reading_order_index),
        float(page_region_count),
        float(fraction),
        float(match_confidence),
        float(text_context_length),
    ]
