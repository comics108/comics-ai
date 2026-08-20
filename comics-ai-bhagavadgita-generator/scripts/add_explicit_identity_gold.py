#!/usr/bin/env python3
"""Add canonical identity only from an explicitly named native PSD hierarchy group."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from adapters.psd import recover_psd_layer
from automated_review import AutomatedMaskReview, validate_automated_mask_review
from build_gold_dataset import GoldAnnotation, GoldDataset, validate_gold_dataset, write_gold_manifest
from generate_gold_candidates import mask_metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def augment(source_manifest: Path, psd: Path, output_root: Path) -> GoldDataset:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    annotations = [GoldAnnotation(**item) for item in payload["annotations"]]
    native_path = "8/13/1"  # parent 8/13 is explicitly named `krishna` in the source PSD
    recovered = recover_psd_layer(psd, native_path)
    coverage, rectangularity = mask_metrics(recovered.bitmap_mask)
    output_root.mkdir(parents=True, exist_ok=True)
    mask_path = output_root / "psd-app-bg-chiba5-krishna-8-13-1.mask.png"
    rgba_path = output_root / "psd-app-bg-chiba5-krishna-8-13-1.rgba.png"
    recovered.bitmap_mask.save(mask_path)
    recovered.rgba.save(rgba_path)
    mask_hash = sha256(mask_path)
    review = validate_automated_mask_review(AutomatedMaskReview(
        source_kind="psd", method_families=("native_alpha",), agreement_iou=1,
        boundary_f1=1, foreground_coverage=coverage, rectangularity=rectangularity,
        source_sha256=sha256(psd), mask_sha256=mask_hash,
    ))
    x0, y0, x1, y1 = recovered.bbox
    identity = GoldAnnotation(
        id="psd-app-bg-chiba5-krishna-8-13-1",
        asset_version_id="asset:psd-app-bg-chiba5:8/13/1:v2",
        source_composition_id="psd-app-bg-chiba5", source_kind="psd", split="test",
        semantic_kind="character", canonical_entity_id="krishna", principal_character=True,
        bitmap_mask_file=mask_path.as_posix(), mask_sha256=mask_hash,
        source_region=(x0, y0, x1 - x0, y1 - y0), review_resolution=recovered.bitmap_mask.size,
        source_to_review_scale=(1, 1), label_origin="psd_alpha_reviewed",
        reviewer=review.reviewer_id, accepted_at="2026-08-12T00:00:00Z", accepted=True,
        review_mode="automated", review_evidence=review.evidence + (
            "canonical_identity:explicit_psd_parent_group:8/13:krishna",
            "identity_scope:this_native_layer_only",
        ),
    )
    annotations = [item for item in annotations if item.id != identity.id] + [identity]
    dataset = GoldDataset("gold-v2.1-explicit-identity-2026-08-12", tuple(annotations))
    validate_gold_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--psd", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = augment(args.source, args.psd, args.candidate_root)
    write_gold_manifest(dataset, args.out)
    print(json.dumps({"dataset_version": dataset.version, "annotations": len(dataset.annotations)}))


if __name__ == "__main__":
    main()
