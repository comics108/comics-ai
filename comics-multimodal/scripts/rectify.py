#!/usr/bin/env python3
"""Task 5.1 (part 1): detect the printed page's physical boundary in a raw
comics_book_lowcamera/*.jpg photo and perspective-correct it to a top-down rectangle.

Classical "document scanner" approach (OpenCV): the pages in this dataset's photos are
photographed against a darker table/background, so the page(s) show up as the largest bright,
roughly-quadrilateral contour in the frame. Not a trained model -- consistent with Specifications'
"page rectification is a classical CV step" framing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RectifyResult:
    rectified: np.ndarray  # BGR image, top-down
    quad: np.ndarray | None  # the 4 source corners found in the original photo, or None if not found
    status: str  # "rectified" | "fallback_full_frame"
    reason: str


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_page_quad(image: np.ndarray, min_area_ratio: float = 0.2) -> np.ndarray | None:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold rather than a fixed value -- real photos vary a lot in lighting/exposure
    # (Task 3.1's measured sharpness/noise range already showed this), so a fixed brightness cutoff
    # would be fragile across the real dataset.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small gaps (e.g. text creating dark specks inside the bright page) so the page reads
    # as one solid bright region.
    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = h * w
    best_quad = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx
            best_area = area

    if best_quad is None:
        return None
    return _order_corners(best_quad.astype(np.float32))


def rectify_page(image: np.ndarray, out_size: tuple[int, int] | None = None) -> RectifyResult:
    """Returns a top-down rectified page. Falls back to the original full frame (status
    "fallback_full_frame") when no confident quadrilateral is found -- callers (panel detection)
    must handle a possibly-unrectified, still-perspective-distorted image rather than assuming
    success, per the skip+log discipline used throughout this pipeline.
    """
    quad = find_page_quad(image)
    if quad is None:
        return RectifyResult(
            rectified=image, quad=None, status="fallback_full_frame", reason="no confident page quad found"
        )

    tl, tr, br, bl = quad
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))
    if max_width < 10 or max_height < 10:
        return RectifyResult(
            rectified=image, quad=None, status="fallback_full_frame", reason="degenerate quad size"
        )

    if out_size:
        max_width, max_height = out_size

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    rectified = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return RectifyResult(rectified=rectified, quad=quad, status="rectified", reason="")
