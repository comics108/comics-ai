"""Autonomous visual descriptors and fail-closed classification proposals for Gold assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from build_gold_dataset import GoldAnnotation
from gold_segmenter_data import load_gold_dataset, load_review_pair


@dataclass(frozen=True)
class VisualDescriptor:
    vector: tuple[float, ...]
    palette: tuple[str, ...]
    style_tags: tuple[str, ...]
    foreground_coverage: float
    rectangularity: float


@dataclass(frozen=True)
class ClassificationProposal:
    asset_version_id: str
    source_annotation_id: str
    semantic_kind_proposal: str
    semantic_kind_confidence: None
    semantic_kind_state: str
    art_stage_proposal: str
    style_tags: tuple[str, ...]
    palette: tuple[str, ...]
    pose: None
    expression: None
    costume: None
    canonical_entity_id: None
    identity_state: str
    identity_confidence: None
    method: str
    evidence: tuple[str, ...]
    descriptor: tuple[float, ...]


def _rectangularity(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if not len(xs):
        return 1.0
    area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return float(mask.sum()) / area


def describe_visual(image: Image.Image, mask_image: Image.Image) -> VisualDescriptor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask_image.convert("L")) > 0
    if rgb.shape[:2] != mask.shape or not bool(mask.any()):
        raise ValueError("descriptor requires a non-empty aligned foreground mask")
    pixels = rgb[mask]
    quantized = np.minimum(pixels // 64, 3)
    bins = quantized[:, 0] * 16 + quantized[:, 1] * 4 + quantized[:, 2]
    histogram = np.bincount(bins, minlength=64).astype(np.float64)
    histogram /= histogram.sum()
    luminance = (
        pixels[:, 0].astype(np.float64) * .2126
        + pixels[:, 1].astype(np.float64) * .7152
        + pixels[:, 2].astype(np.float64) * .0722
    )
    luminance_histogram = np.histogram(luminance, bins=8, range=(0, 256))[0].astype(np.float64)
    luminance_histogram /= luminance_histogram.sum()
    color_range = pixels.max(axis=1) - pixels.min(axis=1)
    monochrome_fraction = float((color_range < 16).mean())
    dark_fraction = float((luminance < 96).mean())
    light_fraction = float((luminance > 224).mean())
    coverage = float(mask.mean())
    rectangularity = _rectangularity(mask)
    aspect = math.log2(max(1, image.width) / max(1, image.height))
    shape_features = np.asarray((coverage, rectangularity, math.tanh(aspect / 3)), dtype=np.float64)
    vector = np.concatenate((histogram, luminance_histogram, shape_features))
    norm = float(np.linalg.norm(vector))
    vector = vector / max(norm, 1e-12)

    color_counts = np.bincount(bins, minlength=64)
    palette = []
    for index in np.argsort(color_counts)[::-1]:
        if color_counts[index] == 0 or len(palette) >= 5:
            break
        red, rest = divmod(int(index), 16)
        green, blue = divmod(rest, 4)
        palette.append(f"#{red * 64 + 32:02x}{green * 64 + 32:02x}{blue * 64 + 32:02x}")
    tags = []
    if monochrome_fraction >= .85:
        tags.append("monochrome")
    if monochrome_fraction >= .75 and dark_fraction >= .08 and light_fraction >= .25:
        tags.append("ink-drawing")
    if coverage < .65:
        tags.append("separable-cutout")
    if not tags:
        tags.append("colour-art")
    return VisualDescriptor(
        vector=tuple(float(value) for value in vector),
        palette=tuple(palette),
        style_tags=tuple(tags),
        foreground_coverage=coverage,
        rectangularity=rectangularity,
    )


def classify_annotation(
    annotation: GoldAnnotation,
    image: Image.Image,
    mask: Image.Image,
) -> ClassificationProposal:
    descriptor = describe_visual(image, mask)
    is_native = annotation.source_kind == "psd"
    evidence = (
        f"source_label:{annotation.semantic_kind}",
        f"source_label_origin:{annotation.label_origin}",
        f"mask_sha256:{annotation.mask_sha256}",
        "identity_abstention:no_independent_canonical_reference",
    )
    return ClassificationProposal(
        asset_version_id=annotation.asset_version_id,
        source_annotation_id=annotation.id,
        semantic_kind_proposal=annotation.semantic_kind,
        semantic_kind_confidence=None,
        semantic_kind_state="seed_label_unscored",
        art_stage_proposal="final" if is_native else "ink",
        style_tags=descriptor.style_tags,
        palette=descriptor.palette,
        pose=None,
        expression=None,
        costume=None,
        canonical_entity_id=None,
        identity_state="abstained",
        identity_confidence=None,
        method="visual-descriptor-v1+gold-seed-label",
        evidence=evidence,
        descriptor=descriptor.vector,
    )


def evaluation_coverage(annotations: tuple[GoldAnnotation, ...]) -> dict:
    accepted = [item for item in annotations if item.accepted]
    train_kinds = sorted({item.semantic_kind for item in accepted if item.split == "train"})
    test_kinds = sorted({item.semantic_kind for item in accepted if item.split == "test"})
    principal = [item for item in accepted if item.principal_character]
    identities = sorted({item.canonical_entity_id for item in principal if item.canonical_entity_id})
    reasons = []
    if len(train_kinds) < 2 or not set(test_kinds).issubset(train_kinds):
        reasons.append("semantic_train_split_has_insufficient_class_coverage")
    if len(identities) < 2:
        reasons.append("principal_canonical_identity_coverage_is_insufficient")
    return {
        "train_semantic_kinds": train_kinds,
        "test_semantic_kinds": test_kinds,
        "principal_instances": len(principal),
        "canonical_identities": identities,
        "semantic_macro_f1": None,
        "identity_top1": None,
        "decision": "abstained" if reasons else "evaluation_ready",
        "reasons": reasons,
    }


def build_catalog(manifest: Path, repository_root: Path) -> dict:
    dataset = load_gold_dataset(manifest, repository_root)
    proposals = []
    for annotation in dataset.annotations:
        if not annotation.accepted:
            continue
        image, mask = load_review_pair(annotation, repository_root)
        proposals.append(asdict(classify_annotation(annotation, image, mask)))
    return {
        "schema_version": 1,
        "dataset_version": dataset.version,
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "classifier": "visual-descriptor-v1+gold-seed-label",
        "proposal_count": len(proposals),
        "evaluation": evaluation_coverage(dataset.annotations),
        "proposals": proposals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    manifest = repository_root / "work/bhagavadgita/production/gold-v1/manifest.json"
    catalog = build_catalog(manifest, repository_root)
    output = args.out if args.out.is_absolute() else repository_root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(catalog, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        "catalog": str(output), "proposals": catalog["proposal_count"],
        "evaluation": catalog["evaluation"]["decision"],
    }))


if __name__ == "__main__":
    main()
