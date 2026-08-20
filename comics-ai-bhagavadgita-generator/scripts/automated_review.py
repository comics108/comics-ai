"""Fail-closed automated mask reviewer for autonomous Gold v1 acceptance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AutomatedMaskReview:
    source_kind: Literal["psd", "panorama"]
    method_families: tuple[str, ...]
    agreement_iou: float
    boundary_f1: float
    foreground_coverage: float
    rectangularity: float
    source_sha256: str
    mask_sha256: str
    pipeline_version: str = "automated-mask-review-v1"


@dataclass(frozen=True)
class AutomatedReviewResult:
    reviewer_id: str
    evidence: tuple[str, ...]


def validate_automated_mask_review(review: AutomatedMaskReview) -> AutomatedReviewResult:
    metrics = (
        review.agreement_iou,
        review.boundary_f1,
        review.foreground_coverage,
        review.rectangularity,
    )
    if any(not math.isfinite(value) for value in metrics):
        raise ValueError("automated mask review metrics must be finite")
    if len(review.source_sha256) != 64 or len(review.mask_sha256) != 64:
        raise ValueError("automated mask review requires source and mask checksums")
    families = frozenset(review.method_families)
    if review.source_kind == "psd":
        if "native_alpha" not in families:
            raise ValueError("PSD automated acceptance requires native alpha evidence")
    elif len(families) < 2:
        raise ValueError("panorama automated acceptance requires two independent method families")
    if review.source_kind == "panorama" and review.agreement_iou < .85:
        raise ValueError("panorama mask consensus IoU is below 0.85")
    if review.boundary_f1 < .75:
        raise ValueError("mask boundary F1 is below 0.75")
    if not .01 < review.foreground_coverage < .95:
        raise ValueError("foreground coverage must be between 0.01 and 0.95")
    if review.rectangularity >= .98:
        raise ValueError("near-rectangular masks are not autonomous gold evidence")
    evidence = (
        review.pipeline_version,
        f"families:{','.join(sorted(families))}",
        f"agreement_iou:{review.agreement_iou:.6f}",
        f"boundary_f1:{review.boundary_f1:.6f}",
        f"coverage:{review.foreground_coverage:.6f}",
        f"rectangularity:{review.rectangularity:.6f}",
        f"source_sha256:{review.source_sha256}",
        f"mask_sha256:{review.mask_sha256}",
    )
    return AutomatedReviewResult(
        reviewer_id=f"auto:{review.pipeline_version}",
        evidence=evidence,
    )
