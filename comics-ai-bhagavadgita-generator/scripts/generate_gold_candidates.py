"""Generate automatically reviewed Gold v1 candidates from native source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from adapters.psd import recover_psd_layer, recover_psd_structure
from automated_review import AutomatedMaskReview, validate_automated_mask_review
from build_gold_dataset import GoldAnnotation, GoldDataset, validate_gold_dataset, write_gold_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_metrics(mask: Image.Image) -> tuple[float, float]:
    """Return nonzero coverage and bbox rectangularity for an L/alpha mask."""
    grayscale = mask.convert("L")
    histogram = grayscale.histogram()
    nonzero = sum(histogram[1:])
    total = grayscale.width * grayscale.height
    coverage = nonzero / total if total else 0.0
    bounds = grayscale.getbbox()
    if bounds is None:
        return coverage, 1.0
    bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
    rectangularity = nonzero / bbox_area if bbox_area else 1.0
    return coverage, rectangularity


def generate_psd_gold_candidates(
    source: Path,
    *,
    composition_id: str,
    output_root: Path,
    limit: int,
) -> tuple[GoldAnnotation, ...]:
    """Recover and auto-accept bounded native-alpha candidates; never flatten the whole PSD."""
    document = recover_psd_structure(source)
    source_checksum = sha256_file(source)
    pixel_nodes = [
        node
        for node in document.nodes
        if node.kind != "group"
        and (node.bbox[2] - node.bbox[0]) * (node.bbox[3] - node.bbox[1]) >= 512
    ]
    # Prefer substantial layers, then stabilize ties by native hierarchy path.
    pixel_nodes.sort(
        key=lambda node: (
            -((node.bbox[2] - node.bbox[0]) * (node.bbox[3] - node.bbox[1])),
            node.native_path,
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    accepted: list[GoldAnnotation] = []
    for node in pixel_nodes:
        if len(accepted) >= limit:
            break
        try:
            recovered = recover_psd_layer(source, node.native_path)
        except (KeyError, ValueError):
            continue
        coverage, rectangularity = mask_metrics(recovered.bitmap_mask)
        if not .01 < coverage < .95 or rectangularity >= .98:
            continue
        candidate_id = f"{composition_id}-{node.native_path.replace('/', '-') }"
        mask_path = output_root / f"{candidate_id}.mask.png"
        rgba_path = output_root / f"{candidate_id}.rgba.png"
        recovered.bitmap_mask.save(mask_path, format="PNG")
        recovered.rgba.save(rgba_path, format="PNG")
        mask_checksum = sha256_file(mask_path)
        result = validate_automated_mask_review(AutomatedMaskReview(
            source_kind="psd",
            method_families=("native_alpha",),
            agreement_iou=1.0,
            boundary_f1=1.0,
            foreground_coverage=coverage,
            rectangularity=rectangularity,
            source_sha256=source_checksum,
            mask_sha256=mask_checksum,
        ))
        x0, y0, x1, y1 = recovered.bbox
        accepted.append(GoldAnnotation(
            id=candidate_id,
            asset_version_id=f"asset:{composition_id}:{node.native_path}:v1",
            source_composition_id=composition_id,
            source_kind="psd",
            split="train",
            semantic_kind="art",
            canonical_entity_id=None,
            principal_character=False,
            bitmap_mask_file=mask_path.as_posix(),
            mask_sha256=mask_checksum,
            source_region=(x0, y0, x1 - x0, y1 - y0),
            review_resolution=recovered.bitmap_mask.size,
            source_to_review_scale=(1.0, 1.0),
            label_origin="psd_alpha_reviewed",
            reviewer=result.reviewer_id,
            accepted_at="2026-08-11T00:00:00Z",
            accepted=True,
            review_mode="automated",
            review_evidence=result.evidence,
        ))
    return tuple(accepted)


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def generate_panorama_gold_candidates(
    candidate_manifest: Path,
    *,
    source_document: Path,
    rendered_page: Path,
    source_composition_id: str,
    source_resolution: tuple[int, int],
    repository_root: Path,
) -> tuple[GoldAnnotation, ...]:
    """Convert independently generated consensus masks to reversible held-out Gold records."""
    payload: Any = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("panorama candidate manifest must be a JSON array")
    source_checksum = sha256_file(source_document)
    render_checksum = sha256_file(rendered_page)
    with Image.open(rendered_page) as page:
        review_page_resolution = page.size
    scale_x = review_page_resolution[0] / source_resolution[0]
    scale_y = review_page_resolution[1] / source_resolution[1]
    accepted: list[GoldAnnotation] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("panorama candidate must be an object")
        mask_path = Path(str(raw["mask_file"]))
        mask_checksum = sha256_file(mask_path)
        if mask_checksum != raw["mask_sha256"]:
            raise ValueError(f"panorama mask checksum mismatch: {mask_path}")
        if tuple(raw["page_resolution"]) != review_page_resolution:
            raise ValueError("panorama candidate page resolution does not match rendered evidence")
        result = validate_automated_mask_review(AutomatedMaskReview(
            source_kind="panorama",
            method_families=tuple(raw["method_families"]),
            agreement_iou=float(raw["agreement_iou"]),
            boundary_f1=float(raw["boundary_f1"]),
            foreground_coverage=float(raw["coverage"]),
            rectangularity=float(raw["rectangularity"]),
            source_sha256=source_checksum,
            mask_sha256=mask_checksum,
        ))
        x0, y0, x1, y1 = (int(value) for value in raw["bbox"])
        source_x0 = round(x0 / scale_x)
        source_y0 = round(y0 / scale_y)
        source_x1 = round(x1 / scale_x)
        source_y1 = round(y1 / scale_y)
        coco_label = int(raw["coco_label"])
        evidence = result.evidence + (
            f"render_sha256:{render_checksum}",
            f"render_bbox:{x0},{y0},{x1},{y1}",
            f"coco_proposal:{coco_label}@{float(raw['coco_score']):.6f}",
            "identity_status:unresolved_nonprincipal",
        )
        accepted.append(GoldAnnotation(
            id=f"{source_composition_id}-{raw['id']}",
            asset_version_id=f"asset:{source_composition_id}:{raw['id']}:v1",
            source_composition_id=source_composition_id,
            source_kind="panorama",
            split="test",
            semantic_kind="character" if coco_label == 1 else "art",
            canonical_entity_id=None,
            principal_character=False,
            bitmap_mask_file=_portable_path(mask_path, repository_root),
            mask_sha256=mask_checksum,
            source_region=(
                source_x0,
                source_y0,
                max(1, source_x1 - source_x0),
                max(1, source_y1 - source_y0),
            ),
            review_resolution=tuple(int(value) for value in raw["review_resolution"]),
            source_to_review_scale=(scale_x, scale_y),
            label_origin="automated_consensus",
            reviewer=result.reviewer_id,
            accepted_at="2026-08-11T00:00:00Z",
            accepted=True,
            review_mode="automated",
            review_evidence=evidence,
        ))
    return tuple(accepted)


def build_real_gold_v1(repository_root: Path, output_root: Path) -> GoldDataset:
    """Build the fixed five-composition Gold v1 release from read-only source evidence."""
    drawing_root = repository_root / "dataset/bhagavadgita/vaishnav/drawing"
    candidate_root = output_root / "candidates"
    annotations: list[GoldAnnotation] = []
    for filename, composition_id, limit in (
        ("5_2.psd", "psd-5-2", 26),
        ("app_BG._chiba5.psd", "psd-app-bg-chiba5", 60),
        ("5_1.psd", "psd-5-1", 4),
    ):
        annotations.extend(generate_psd_gold_candidates(
            drawing_root / filename,
            composition_id=composition_id,
            output_root=candidate_root,
            limit=limit,
        ))
    panorama_source = output_root / "panorama-source"
    source_document = drawing_root / "All_Black-n-White.pdf"
    for page, source_resolution in (("02", (21767, 2913)), ("12", (65433, 8976))):
        annotations.extend(generate_panorama_gold_candidates(
            output_root / f"panorama-page-{page}/candidates.json",
            source_document=source_document,
            rendered_page=panorama_source / f"bw-page-{page}.jpg",
            source_composition_id=f"panorama-bw-page-{page}",
            source_resolution=source_resolution,
            repository_root=repository_root,
        ))
    dataset = GoldDataset(version="gold-v1-2026-08-11", annotations=tuple(annotations))
    validate_gold_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/bhagavadgita/production/gold-v1"),
    )
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    dataset = build_real_gold_v1(repository_root, output_root)
    manifest = output_root / "manifest.json"
    write_gold_manifest(dataset, manifest)
    result = validate_gold_dataset(dataset)
    print(json.dumps({
        "manifest": _portable_path(manifest, repository_root),
        "accepted": result.accepted_count,
        "held_out": result.held_out_count,
        "compositions": result.source_composition_count,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
