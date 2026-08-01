#!/usr/bin/env python3
"""Task 5.1 (part 2): detect whole PAGE regions in a photo -- not individual panels.

Revision 1.2 (2026-07-31, during Implementation): the original plan was to detect individual
comic panels within a page. Classical-CV attempts (brightness thresholding, then local-activity/
Laplacian thresholding at several kernel sizes) only cleanly isolated roughly half the panels on a
densely-packed page -- panels there are separated by thin gutters that are hard to distinguish
reliably from busy in-panel art at the edges. The user confirmed a simpler, more robust pivot:
detect whole PAGE regions instead (splitting a two-page spread at the spine gutter, which is a much
wider and more reliably low-detail gap than inter-panel gutters), and OCR/match at page granularity
instead of panel granularity. This trades matching precision (`ground_truth_cluster` spans more
layers per match) for reliability -- verified visually on real photos: clean left/right page boxes
recovered correctly on legible interior-page photos; a blurry cover-shot photo's split was
meaningless, but that photo has no balloon text to match anyway, so it will simply fail to match at
the OCR stage (skip + log) regardless of how the page-split falls.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PageBox:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in the source photo


def _activity_mask(gray: np.ndarray, blur_sigma: float = 15.0, close_kernel: int = 25) -> np.ndarray:
    """A page/page-half reads as one large, roughly-uniform blob of "high edge activity" (busy art,
    balloon outlines, text) against gutters/table background, which are comparatively flat. Uses
    Laplacian magnitude (not raw brightness -- some pages are very dark scenes, see Task 2.4's
    real-photo spot-checks) heavily blurred and thresholded, then closed to solidify each page into
    one blob and opened to break any thin bridge across the spine gutter.
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    activity = cv2.convertScaleAbs(lap)
    activity_blur = cv2.GaussianBlur(activity, (0, 0), sigmaX=blur_sigma)
    _, mask = cv2.threshold(activity_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def detect_pages(image: np.ndarray, min_area_ratio: float = 0.05, max_pages: int = 2) -> list[PageBox]:
    """Return up to `max_pages` page-region bounding boxes, in left-to-right reading order."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _activity_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = image.shape[0] * image.shape[1]

    candidates = [c for c in contours if cv2.contourArea(c) > min_area_ratio * frame_area]
    candidates.sort(key=cv2.contourArea, reverse=True)

    boxes = []
    for c in candidates[:max_pages]:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(PageBox(bbox=(x, y, x + w, y + h)))

    boxes.sort(key=lambda b: b.bbox[0])  # left-to-right reading order, not area order
    return boxes
