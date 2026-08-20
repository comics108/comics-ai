"""Automatically match and register coloured panoramas to B&W source pages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairEvidence:
    colour_page: int
    bw_page: int
    ratio_matches: int
    ransac_inliers: int
    inlier_ratio: float
    homography: tuple[float, ...]


def select_confident_pairs(evidence: list[PairEvidence]) -> tuple[PairEvidence, ...]:
    """Require a unique, high-margin geometric match for every colour page."""
    by_colour: dict[int, list[PairEvidence]] = {}
    for item in evidence:
        by_colour.setdefault(item.colour_page, []).append(item)
    selected = []
    for colour_page, candidates in sorted(by_colour.items()):
        candidates.sort(key=lambda item: (-item.ransac_inliers, -item.inlier_ratio, item.bw_page))
        best = candidates[0]
        runner_up = candidates[1].ransac_inliers if len(candidates) > 1 else 0
        margin = best.ransac_inliers / max(1, runner_up)
        if best.ransac_inliers < 50 or best.inlier_ratio < .75 or margin < 3:
            raise ValueError(f"ambiguous panorama registration for colour page {colour_page}")
        selected.append(best)
    bw_pages = [item.bw_page for item in selected]
    if len(bw_pages) != len(set(bw_pages)):
        raise ValueError("multiple colour pages selected the same B&W composition")
    return tuple(selected)


def _page_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"preview has no page number: {path}")
    return int(match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_pair(colour_path: Path, bw_path: Path) -> PairEvidence:
    import cv2
    import numpy as np

    colour = cv2.imread(str(colour_path), cv2.IMREAD_GRAYSCALE)
    bw = cv2.imread(str(bw_path), cv2.IMREAD_GRAYSCALE)
    if colour is None or bw is None:
        raise ValueError("cannot decode panorama preview")
    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=10)
    colour_points, colour_descriptors = orb.detectAndCompute(colour, None)
    bw_points, bw_descriptors = orb.detectAndCompute(bw, None)
    if colour_descriptors is None or bw_descriptors is None:
        return PairEvidence(_page_number(colour_path), _page_number(bw_path), 0, 0, 0., ())
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        colour_descriptors, bw_descriptors, k=2
    )
    good = [left for left, right in pairs if left.distance < .72 * right.distance]
    if len(good) < 8:
        return PairEvidence(
            _page_number(colour_path), _page_number(bw_path), len(good), 0, 0., ()
        )
    source = np.float32([colour_points[item.queryIdx].pt for item in good])
    target = np.float32([bw_points[item.trainIdx].pt for item in good])
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return PairEvidence(
        colour_page=_page_number(colour_path),
        bw_page=_page_number(bw_path),
        ratio_matches=len(good),
        ransac_inliers=inliers,
        inlier_ratio=inliers / max(1, len(good)),
        homography=tuple(float(value) for value in homography.reshape(-1)) if homography is not None else (),
    )


def _boundary_f1(left, right) -> float:
    import cv2
    import numpy as np

    kernel = np.ones((5, 5), np.uint8)
    left_near = cv2.dilate(left.astype(np.uint8), kernel) > 0
    right_near = cv2.dilate(right.astype(np.uint8), kernel) > 0
    precision = float((right & left_near).sum()) / max(1, int(right.sum()))
    recall = float((left & right_near).sum()) / max(1, int(left.sum()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def register_pairs(
    preview_root: Path,
    output_root: Path,
    *,
    bw_document: Path,
    colour_document: Path,
    crop_width: int = 512,
) -> dict:
    import cv2
    import numpy as np

    colour_paths = sorted(preview_root.glob("colour-*.jpg"), key=_page_number)
    bw_paths = sorted(preview_root.glob("bw-*.jpg"), key=_page_number)
    evidence = [measure_pair(colour, bw) for colour in colour_paths for bw in bw_paths]
    selected = select_confident_pairs(evidence)
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = []
    for selection in selected:
        colour_path = next(path for path in colour_paths if _page_number(path) == selection.colour_page)
        bw_path = next(path for path in bw_paths if _page_number(path) == selection.bw_page)
        colour = cv2.imread(str(colour_path), cv2.IMREAD_COLOR)
        bw = cv2.imread(str(bw_path), cv2.IMREAD_COLOR)
        height, width = bw.shape[:2]
        homography = np.asarray(selection.homography, dtype=np.float64).reshape(3, 3)
        aligned = cv2.warpPerspective(
            colour, homography, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        valid = cv2.warpPerspective(
            np.full(colour.shape[:2], 255, np.uint8), homography, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        invalid = np.where(valid > 0, 0, 255).astype(np.uint8)
        bw_edges = cv2.Canny(cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY), 60, 160) > 0
        colour_edges = cv2.Canny(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), 60, 160) > 0
        colour_edges &= valid > 0
        edge_f1 = _boundary_f1(bw_edges, colour_edges)
        pair_id = f"colour-{selection.colour_page:02}-bw-{selection.bw_page:02}"
        pair_root = output_root / pair_id
        pair_root.mkdir(parents=True, exist_ok=True)
        aligned_path = pair_root / "aligned-colour.jpg"
        invalid_path = pair_root / "invalid-mask.png"
        cv2.imwrite(str(aligned_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(invalid_path), invalid)
        crops = []
        for index, x0 in enumerate(range(0, width, crop_width)):
            x1 = min(width, x0 + crop_width)
            crop_invalid = invalid[:, x0:x1]
            if float((crop_invalid > 0).mean()) > .10:
                continue
            crop_root = pair_root / f"crop-{index:03}"
            bw_crop_path = crop_root.with_suffix(".bw.jpg")
            colour_crop_path = crop_root.with_suffix(".colour.jpg")
            invalid_crop_path = crop_root.with_suffix(".invalid.png")
            cv2.imwrite(str(bw_crop_path), bw[:, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(colour_crop_path), aligned[:, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(invalid_crop_path), crop_invalid)
            crops.append({
                "bbox": [x0, 0, x1, height],
                "bw_file": str(bw_crop_path),
                "colour_file": str(colour_crop_path),
                "invalid_mask_file": str(invalid_crop_path),
                "bw_sha256": _sha256(bw_crop_path),
                "colour_sha256": _sha256(colour_crop_path),
                "invalid_mask_sha256": _sha256(invalid_crop_path),
            })
        pairs.append({
            "id": pair_id,
            "colour_page": selection.colour_page,
            "bw_page": selection.bw_page,
            "ratio_matches": selection.ratio_matches,
            "ransac_inliers": selection.ransac_inliers,
            "inlier_ratio": selection.inlier_ratio,
            "homography_colour_to_bw": list(selection.homography),
            "preview_resolution": [width, height],
            "valid_coverage": float((valid > 0).mean()),
            "registered_ink_edge_f1": edge_f1,
            "aligned_colour_file": str(aligned_path),
            "aligned_colour_sha256": _sha256(aligned_path),
            "invalid_mask_file": str(invalid_path),
            "invalid_mask_sha256": _sha256(invalid_path),
            "crops": crops,
        })
    return {
        "schema_version": 1,
        "reviewer": "auto:orb-homography-registration-v1",
        "bw_document_sha256": _sha256(bw_document),
        "colour_document_sha256": _sha256(colour_document),
        "pair_count": len(pairs),
        "pairs": pairs,
        "all_pair_evidence": [asdict(item) for item in evidence],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    drawing = repository_root / "dataset/bhagavadgita/vaishnav/drawing"
    report = register_pairs(
        args.previews, args.out_root,
        bw_document=drawing / "All_Black-n-White.pdf",
        colour_document=drawing / "All_Coloured.pdf",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        "manifest": str(args.manifest), "pairs": report["pair_count"],
        "crops": sum(len(item["crops"]) for item in report["pairs"]),
    }))


if __name__ == "__main__":
    main()
