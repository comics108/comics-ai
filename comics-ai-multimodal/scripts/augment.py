#!/usr/bin/env python3
"""Stage 2 (revised per Specifications Revision 1.1): crop panel-shaped local content clusters
from each canvas reference and apply camera-realism degradation, producing
(degraded panel-like crop -> ground truth) training pairs.

Clusters, not arbitrary rectangular windows: Checkpoint A found real photos show individually
composed print panels, not raw crops of the tall scrolling canvas -- so synthetic training data
crops around each `kind_heuristic.py` y-window neighborhood (the same local-cluster grouping used
for "bottom of stack" background detection), which is a much closer approximation of what a real
panel actually contains (a background + its characters + its balloon(s), tightly grouped) than an
arbitrary slice of scroll would be.

Degradation parameters are calibrated from `analyze_photos.py`'s measurement of all 80 real
`comics_book_lowcamera/*.jpg` photos (Task 3.1 / Checkpoint B), not guessed:
sharpness (Laplacian var): min 67.7, p25 348, median 640, p75 950, max 1514
noise sigma (Immerkær estimator): min 0.72, p25 1.33, median 2.58, p75 3.20, max 5.93

Verified against the real dataset (all 27 canvases, 2026-07-31): 753 training pairs produced;
cluster height median 2653px (cap 3000px); of the 145/753 clusters exceeding the cap, all are a
single unsplittable oversized background-art layer (verified: 0 multi-layer clusters exceed the
cap) -- the documented, acceptable edge case, not a bug.

Revised again in Phase 6 (`build_page_groups`): real-photo evaluation found the segmentation model
(trained on the single-scene crops described above) performed poorly at actual inference time,
because Phase 5's page-level matching (Revision 1.2) means inference always runs on whole
multi-panel pages, not single scenes -- a real train/inference scale mismatch. Training crops now
group several consecutive scene clusters together (`PAGE_GROUP_SIZE` scenes) to match what the
model will actually see.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # canvases are legitimately large (up to ~100M px); trusted local data

from render_canvas import GroundTruthRegion

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_CANVAS_DIR = WORK_DIR / "canvas"
DEFAULT_OUT_DIR = WORK_DIR / "train_pairs"

REAL_SHARPNESS_RANGE = (67.7, 1514.0)
REAL_NOISE_SIGMA_RANGE = (0.72, 5.93)

Y_WINDOW = 1500  # matches kind_heuristic.py's default neighborhood size (used only as the
                  # defensive re-split cap below, not as the primary clustering rule -- see note)
MAX_CLUSTER_HEIGHT = 3000  # defensive cap so no single crop balloons past a plausible panel size


def cluster_layers_by_y(
    regions: list[GroundTruthRegion], y_window: int = Y_WINDOW
) -> list[list[GroundTruthRegion]]:
    """DEPRECATED clustering rule, kept only for the transitive-chaining unit tests below (it's a
    real, useful primitive) -- NOT used by build_training_pairs anymore. A real run against all 27
    canvases showed this simple "chain while gap <= y_window" rule transitively merges nearly an
    entire file's regions into 1-3 giant clusters on densely-packed content (median cluster height
    came out to 15,300px, up to 206 layers in one cluster) -- nothing like a printed panel. See
    `cluster_layers_by_scene` for the fix actually used.
    """
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)
    clusters: list[list[GroundTruthRegion]] = [[ordered[0]]]
    last_y = (ordered[0].bbox[1] + ordered[0].bbox[3]) / 2
    for r in ordered[1:]:
        y = (r.bbox[1] + r.bbox[3]) / 2
        if y - last_y <= y_window:
            clusters[-1].append(r)
        else:
            clusters.append([r])
        last_y = y
    return clusters


def cluster_layers_by_scene(
    regions: list[GroundTruthRegion], max_cluster_height: int = MAX_CLUSTER_HEIGHT
) -> list[list[GroundTruthRegion]]:
    """Group ground-truth regions into panel-like clusters anchored at each `kind == "background"`
    layer -- in this dataset's structure a new background layer marks a new scene, which is a much
    closer proxy for "one printed panel's worth of content" than a fixed-radius y-window chain
    (see the deprecation note on `cluster_layers_by_y` above for why that approach failed in
    practice). A defensive `max_cluster_height` re-splits any background-to-background span that's
    still too tall (background anchors can be sparse in some stretches of some files), so no crop
    ever balloons back up to a multi-scene mega-cluster.

    Each non-background region is assigned to its **nearest background by y-center distance**, not
    by a sequential "flush on background encounter" scan over y-sorted regions -- an earlier
    version did the latter and had a real bug (caught by a Phase 5 test, not Phase 3's original
    verification): a non-background region whose y-center happens to be *smaller* than its own
    scene's background center (e.g. a balloon placed near the top of a tall panel) would sort
    before that background in the y-ordering and get flushed into its own spurious single-region
    cluster instead of joining its actual scene. Nearest-background assignment is immune to this
    since it's a distance comparison, not an encounter-order artifact.
    """
    if not regions:
        return []

    backgrounds = [r for r in regions if r.kind == "background"]
    non_backgrounds = [r for r in regions if r.kind != "background"]

    if not backgrounds:
        # No anchor at all -- one cluster for everything (matches the prior behavior's fallback).
        scene_clusters: list[list[GroundTruthRegion]] = [list(regions)]
    else:
        groups: dict[int, list[GroundTruthRegion]] = {id(bg): [bg] for bg in backgrounds}

        def _center(r: GroundTruthRegion) -> float:
            return (r.bbox[1] + r.bbox[3]) / 2

        for r in non_backgrounds:
            nearest_bg = min(backgrounds, key=lambda bg: abs(_center(bg) - _center(r)))
            groups[id(nearest_bg)].append(r)

        scene_clusters = list(groups.values())

    final: list[list[GroundTruthRegion]] = []
    for cluster in scene_clusters:
        bbox = cluster_bbox(cluster)
        if bbox[3] - bbox[1] <= max_cluster_height:
            final.append(cluster)
            continue
        # Still too tall (sparse backgrounds in this stretch) -- fall back to re-chaining within
        # just this cluster, checking the *actual resulting union bbox* on each tentative add
        # (not just center-to-center distance, which under-counts when individual regions have
        # real height of their own -- e.g. a single tall background layer can by itself push a
        # chunk's true bbox well past max_cluster_height even while every center-distance looks
        # fine; verified this was happening on the real dataset before this fix).
        sub = sorted(cluster, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)
        chunk = [sub[0]]
        for r in sub[1:]:
            candidate_bbox = cluster_bbox(chunk + [r])
            if candidate_bbox[3] - candidate_bbox[1] > max_cluster_height:
                final.append(chunk)
                chunk = [r]
            else:
                chunk.append(r)
        if chunk:
            final.append(chunk)
    return final


PAGE_GROUP_SIZE = 4  # approx how many panels/scenes a real printed page shows -- a visual estimate
# from Task 5.1's real photo inspection (e.g. the "28/29" page spread had ~5 panels on one page,
# ~4 on the other).


def build_page_groups(
    scene_clusters: list[list[GroundTruthRegion]], group_size: int = PAGE_GROUP_SIZE
) -> list[list[GroundTruthRegion]]:
    """Group consecutive scene clusters (already correctly bounded per-scene by
    cluster_layers_by_scene) into page-scale groups, approximating a real printed page's
    multi-panel content.

    Added in Phase 6 to fix a real train/inference scale mismatch found via real-photo evaluation:
    the segmentation model was originally trained on single-scene (panel-scale) crops (this
    module's original design), but Phase 5's page-level matching (Revision 1.2) means inference
    always sees whole multi-panel pages -- a real photo's OCR-matched content spans several scene
    clusters at once, not one. Training data must resemble what the model will actually see at
    inference time.
    """
    groups = []
    for i in range(0, len(scene_clusters), group_size):
        chunk = scene_clusters[i : i + group_size]
        merged = [r for cluster in chunk for r in cluster]
        if merged:
            groups.append(merged)
    return groups


def cluster_bbox(cluster: list[GroundTruthRegion]) -> tuple[int, int, int, int]:
    x0 = min(r.bbox[0] for r in cluster)
    y0 = min(r.bbox[1] for r in cluster)
    x1 = max(r.bbox[2] for r in cluster)
    y1 = max(r.bbox[3] for r in cluster)
    return (x0, y0, x1, y1)


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))


@dataclass
class TrainingPair:
    episode_file: str
    cluster_index: int
    degraded_png: str
    clean_png: str
    bbox: tuple[int, int, int, int]
    layer_indexes: list[int]
    kinds: list[str]
    region_bboxes: list[tuple[float, float, float, float]]  # crop-local, POST-degradation coords


def _box_corners(box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32).reshape(-1, 1, 2)


def degrade_with_boxes(
    im: Image.Image, boxes: list[tuple[float, float, float, float]], rng: random.Random
) -> tuple[Image.Image, list[tuple[float, float, float, float]]]:
    """Camera-realism augmentation: mild rotation, perspective warp, vignette, blur (targeting the
    real sharpness distribution), sensor noise (targeting the real noise-sigma distribution), JPEG
    re-compression -- see module docstring for the measured ranges this is calibrated from.

    `boxes` (crop-local, pre-degradation coordinates) are carried through the *same* rotation +
    perspective matrices applied to the pixels, so ground-truth region positions stay correct in
    the degraded image -- rotation/perspective would otherwise silently shift content out from
    under untransformed boxes, injecting real label noise scaled with the distortion magnitude.
    Photometric-only steps (vignette/blur/noise/JPEG) don't move content, so boxes are untouched by
    those. Perspective-warped corners are collapsed back to an axis-aligned enclosing box (a
    documented simplification -- true quadrilateral/polygon ground truth isn't tracked anywhere in
    this pipeline, matching `GroundTruthRegion`'s own rectangle-only representation).
    """
    arr = np.array(im.convert("RGB"))
    h, w = arr.shape[:2]
    corners = (
        np.concatenate([_box_corners(b) for b in boxes], axis=0)
        if boxes
        else np.zeros((0, 1, 2), dtype=np.float32)
    )

    angle = rng.uniform(-8, 8)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    arr = cv2.warpAffine(arr, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if len(corners):
        corners = cv2.transform(corners, rot_mat)

    jitter_x, jitter_y = w * 0.04, h * 0.04
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
            [w - rng.uniform(0, jitter_x), rng.uniform(0, jitter_y)],
            [w - rng.uniform(0, jitter_x), h - rng.uniform(0, jitter_y)],
            [rng.uniform(0, jitter_x), h - rng.uniform(0, jitter_y)],
        ]
    )
    persp = cv2.getPerspectiveTransform(src, dst)
    arr = cv2.warpPerspective(arr, persp, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if len(corners):
        corners = cv2.perspectiveTransform(corners, persp)

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    vignette = 1.0 - rng.uniform(0.1, 0.35) * np.clip(dist, 0, 1)
    arr = (arr.astype(np.float64) * vignette[..., None]).clip(0, 255).astype(np.uint8)

    target_sharpness = rng.uniform(*REAL_SHARPNESS_RANGE)
    blur_strength = 1.0 - (target_sharpness - REAL_SHARPNESS_RANGE[0]) / (
        REAL_SHARPNESS_RANGE[1] - REAL_SHARPNESS_RANGE[0]
    )
    ksize = max(1, int(blur_strength * 15) | 1)
    if ksize > 1:
        arr = cv2.GaussianBlur(arr, (ksize, ksize), 0)

    sigma = rng.uniform(*REAL_NOISE_SIGMA_RANGE)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    out = Image.fromarray(arr)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=rng.randint(60, 92))
    buf.seek(0)
    degraded = Image.open(buf).convert("RGB")

    transformed_boxes = []
    for i in range(len(boxes)):
        c = corners[i * 4 : i * 4 + 4].reshape(4, 2)
        nx0, ny0 = float(c[:, 0].min()), float(c[:, 1].min())
        nx1, ny1 = float(c[:, 0].max()), float(c[:, 1].max())
        transformed_boxes.append((max(0.0, nx0), max(0.0, ny0), min(float(w), nx1), min(float(h), ny1)))

    return degraded, transformed_boxes


def degrade(im: Image.Image, rng: random.Random) -> Image.Image:
    """Back-compat wrapper (no boxes to transform) -- kept for the existing unit tests."""
    degraded, _ = degrade_with_boxes(im, [], rng)
    return degraded


def build_training_pairs(
    canvas_dir: Path = DEFAULT_CANVAS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 0,
) -> list[TrainingPair]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[TrainingPair] = []

    for gt_path in sorted(canvas_dir.glob("*.gt.json")):
        ref_data = json.loads(gt_path.read_text())
        episode_file = ref_data["episode_file"]
        width, height = ref_data["width"], ref_data["height"]
        regions = [
            GroundTruthRegion(
                layer_index=r["layer_index"],
                kind=r["kind"],
                kind_source=r["kind_source"],
                bbox=tuple(r["bbox"]),
            )
            for r in ref_data["regions"]
        ]
        composite_path = Path(ref_data["composite_png"])
        if not composite_path.is_file():
            continue
        canvas = Image.open(composite_path).convert("RGB")

        page_groups = build_page_groups(cluster_layers_by_scene(regions))
        for ci, cluster in enumerate(page_groups):
            bbox = _clip_bbox(cluster_bbox(cluster), width, height)
            x0, y0, x1, y1 = bbox
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue  # degenerate cluster, skip

            # Ascending layer_index = bottom-to-top compositing order (established in
            # render_canvas.py / Editor Schema Ground Truth) -- painting label maps in this order
            # at train time reproduces correct z-order occlusion (a balloon/character painted over
            # its background), regardless of the y-center sort cluster_layers_by_scene used.
            cluster = sorted(cluster, key=lambda r: r.layer_index)

            clean_crop = canvas.crop(bbox)
            local_boxes = [
                (r.bbox[0] - x0, r.bbox[1] - y0, r.bbox[2] - x0, r.bbox[3] - y0) for r in cluster
            ]
            degraded_crop, transformed_boxes = degrade_with_boxes(clean_crop, local_boxes, rng)

            stem = f"{Path(episode_file).stem}_{ci:04d}"
            clean_path = out_dir / f"{stem}_clean.png"
            degraded_path = out_dir / f"{stem}_degraded.jpg"
            clean_crop.save(clean_path)
            degraded_crop.save(degraded_path, quality=90)

            pairs.append(
                TrainingPair(
                    episode_file=episode_file,
                    cluster_index=ci,
                    degraded_png=str(degraded_path),
                    clean_png=str(clean_path),
                    bbox=bbox,
                    layer_indexes=[r.layer_index for r in cluster],
                    kinds=[r.kind for r in cluster],
                    region_bboxes=transformed_boxes,
                )
            )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__) + "\n")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas-dir", type=Path, default=DEFAULT_CANVAS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pairs = build_training_pairs(args.canvas_dir, args.out, args.seed)
    print(f"Generated {len(pairs)} training pairs -> {args.out}")


if __name__ == "__main__":
    main()
