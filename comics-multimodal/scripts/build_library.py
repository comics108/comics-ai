#!/usr/bin/env python3
"""Task 8.2: cluster character/environment CutRegions (Phase 6) into a gallery, per Plan Task 8.1's
decision -- classical grouping (by episode, a weak prior) validated first, then a pretrained
embedding (frozen ResNet-18, ImageNet weights -- Specifications' resolved "trained from scratch"
scope permits pretrained backbones) layered in for pose/lighting invariance within and across
episodes.

Identity names are seeded from `Comics_Episodes.csv`'s episode-name token (e.g. "21_ambas_plea" ->
candidate "amba") -- a weak, best-effort label per Specifications, not authoritative: one episode
can contain multiple characters, and many episode titles aren't character names at all (e.g.
"03_the_chase"). Ambiguous/ungrouped crops land in `unclustered/`, never force-merged.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torchvision.models import ResNet18_Weights, resnet18

import baloons_bridge
from detect_panels import detect_pages
from infer_segmenter import TRAIN_SIZE
from rectify import rectify_page

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_REGIONS = WORK_DIR / "regions.jsonl"
DEFAULT_ALIGNMENT = WORK_DIR / "alignment.jsonl"
DEFAULT_LOWCAMERA_DIR = (
    baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)
DEFAULT_EPISODES_CSV = (
    baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_interactive"
    / "Comics_Episodes.csv"
)
DEFAULT_LIBRARY_DIR = WORK_DIR / "library"

KIND_TO_LIBRARY_DIR = {"character": "characters", "background": "environments"}
MIN_CONFIDENCE = 0.4  # below this, a region is too uncertain to trust for library membership
WITHIN_EPISODE_DISTANCE_THRESHOLD = 0.5  # cosine distance; splits an episode's crops into
# multiple sub-identities if they look very different from each other
CROSS_EPISODE_MERGE_THRESHOLD = 0.25  # cosine distance; conservative -- only merge across
# episodes when embeddings are very close, never force it


def load_episode_seed_names(csv_path: Path = DEFAULT_EPISODES_CSV) -> dict[str, str]:
    """episode_file (basename) -> best-effort seed identity name from the episode's Product token."""
    seeds: dict[str, str] = {}
    if not csv_path.is_file():
        return seeds
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_field = row.get("File", "")
            product = row.get("Product", "")
            if not file_field or not product or product == "NULL":
                continue
            episode_file = Path(file_field).name
            seeds[episode_file] = seed_name_from_episode_token(product)
    return seeds


def seed_name_from_episode_token(product_token: str) -> str:
    parts = product_token.split("_", 1)
    rest = parts[1] if len(parts) > 1 and parts[0].isdigit() else product_token
    first_word = rest.split("_")[0]
    # Crude possessive-strip heuristic: CSV-safe titles drop apostrophes ("Amba's" -> "ambas"),
    # so a trailing "s" on a long-enough first word is often a possessive marker, not a plural.
    if first_word.endswith("s") and len(first_word) > 3:
        first_word = first_word[:-1]
    return first_word.lower()


_embedding_model: nn.Module | None = None


def _get_embedding_model() -> nn.Module:
    global _embedding_model
    if _embedding_model is None:
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        _embedding_model = nn.Sequential(*list(base.children())[:-1])  # drop final FC -> 512-d
        _embedding_model.eval()
    return _embedding_model


def compute_embedding(crop_rgb: np.ndarray) -> np.ndarray:
    model = _get_embedding_model()
    resized = cv2.resize(crop_rgb, (224, 224))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.no_grad():
        emb = model(tensor).flatten().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def extract_crop_image(
    photo_file: str, page_index: int, bbox: tuple[int, int, int, int], lowcamera_dir: Path
) -> np.ndarray | None:
    """Re-derives the exact same page crop infer_segmenter.py used (detect -> rectify -> resize to
    TRAIN_SIZE), then crops `bbox` from it -- bbox is already in that resolution's coordinate space.
    """
    image = cv2.imread(str(lowcamera_dir / photo_file))
    if image is None:
        return None
    pages = detect_pages(image)
    if page_index >= len(pages):
        return None
    x0, y0, x1, y1 = pages[page_index].bbox
    page_crop = image[y0:y1, x0:x1]
    rect_result = rectify_page(page_crop)

    th, tw = TRAIN_SIZE
    resized = cv2.resize(cv2.cvtColor(rect_result.rectified, cv2.COLOR_BGR2RGB), (tw, th))
    bx0, by0, bx1, by1 = bbox
    region = resized[by0:by1, bx0:bx1]
    return region if region.size > 0 else None


def build_library(
    regions_path: Path = DEFAULT_REGIONS,
    alignment_path: Path = DEFAULT_ALIGNMENT,
    lowcamera_dir: Path = DEFAULT_LOWCAMERA_DIR,
    episodes_csv: Path = DEFAULT_EPISODES_CSV,
    out_dir: Path = DEFAULT_LIBRARY_DIR,
) -> dict[str, list[str]]:
    """Returns {kind: [identity_names]} for reporting/testing. Writes
    out_dir/{characters,environments}/<name>/*.png + unclustered/*.png.
    """
    alignment: dict[tuple[str, int], str] = {}
    with alignment_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d["status"] == "matched":
                alignment[(d["photo_file"], d["page_index"])] = d["episode_file"]

    seed_names = load_episode_seed_names(episodes_csv)

    regions_by_kind_episode: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with regions_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["predicted_kind"] not in KIND_TO_LIBRARY_DIR:
                continue
            key = (r["photo_file"], r["page_index"])
            episode = alignment.get(key)
            if episode is None:
                continue
            regions_by_kind_episode[r["predicted_kind"]][episode].append(r)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    result: dict[str, list[str]] = defaultdict(list)

    for kind, lib_subdir in KIND_TO_LIBRARY_DIR.items():
        kind_dir = out_dir / lib_subdir
        kind_dir.mkdir(parents=True, exist_ok=True)
        unclustered_dir = kind_dir / "unclustered"

        # Per-episode sub-clusters: (episode, sub_cluster_id) -> list[(region, crop_rgb, embedding)]
        episode_subclusters: dict[tuple[str, int], list[tuple[dict, np.ndarray, np.ndarray]]] = {}

        for episode, regions in regions_by_kind_episode[kind].items():
            crops_and_embeddings = []
            for r in regions:
                crop = extract_crop_image(r["photo_file"], r["page_index"], tuple(r["bbox"]), lowcamera_dir)
                if crop is None or crop.size == 0:
                    continue  # nothing to write anywhere -- genuinely no image data
                if r["confidence"] < MIN_CONFIDENCE:
                    # Too uncertain to trust for clustering, but still saved for human review --
                    # never silently dropped (Plan Task 8.1: ambiguous crops land in
                    # unclustered/, not force-assigned and not disappeared).
                    unclustered_dir.mkdir(parents=True, exist_ok=True)
                    out_path = (
                        unclustered_dir
                        / f"{Path(r['photo_file']).stem}_p{r['page_index']}_{episode[:8]}_low_confidence.png"
                    )
                    Image.fromarray(crop).save(out_path)
                    continue
                emb = compute_embedding(crop)
                crops_and_embeddings.append((r, crop, emb))

            if not crops_and_embeddings:
                continue

            if len(crops_and_embeddings) == 1:
                episode_subclusters[(episode, 0)] = crops_and_embeddings
                continue

            embeddings = np.stack([e for _, _, e in crops_and_embeddings])
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=WITHIN_EPISODE_DISTANCE_THRESHOLD,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)
            for sub_id in set(labels):
                members = [ce for ce, lbl in zip(crops_and_embeddings, labels) if lbl == sub_id]
                episode_subclusters[(episode, int(sub_id))] = members

        # Cross-episode merge: conservative, centroid-distance based.
        keys = list(episode_subclusters.keys())
        centroids = {
            k: np.mean([e for _, _, e in episode_subclusters[k]], axis=0) for k in keys
        }
        parent = {k: k for k in keys}

        def find(k):
            while parent[k] != k:
                k = parent[k]
            return k

        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1 :]:
                if k1[0] == k2[0]:
                    continue  # same episode, already split intentionally -- don't re-merge
                dist = 1 - float(np.dot(centroids[k1], centroids[k2]))
                if dist <= CROSS_EPISODE_MERGE_THRESHOLD:
                    parent[find(k1)] = find(k2)

        merged_groups: dict[tuple, list] = defaultdict(list)
        for k in keys:
            merged_groups[find(k)].extend(episode_subclusters[k])

        # Name and write each merged group.
        used_names: dict[str, int] = defaultdict(int)
        for root_key, members in merged_groups.items():
            episode_of_root = root_key[0]
            base_name = seed_names.get(episode_of_root, Path(episode_of_root).stem[:8])
            used_names[base_name] += 1
            name = base_name if used_names[base_name] == 1 else f"{base_name}-{used_names[base_name]}"

            identity_dir = kind_dir / name
            identity_dir.mkdir(parents=True, exist_ok=True)
            for i, (r, crop, _emb) in enumerate(members):
                out_path = identity_dir / f"{Path(r['photo_file']).stem}_p{r['page_index']}_{i}.png"
                Image.fromarray(crop).save(out_path)
            result[kind].append(name)

    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_LIBRARY_DIR)
    args = parser.parse_args()
    result = build_library(out_dir=args.out)
    for kind, names in result.items():
        print(f"{kind}: {len(names)} identities -> {sorted(names)}")


if __name__ == "__main__":
    main()
