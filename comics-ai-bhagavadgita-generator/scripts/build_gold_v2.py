#!/usr/bin/env python3
"""Derive a non-circular, source-disjoint Gold v2 split without altering accepted masks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from build_gold_dataset import GoldAnnotation, GoldDataset, validate_gold_dataset, write_gold_manifest


SPLITS = {
    "psd-app-bg-chiba5": "test",       # 60 independent native-alpha instances
    "psd-5-1": "validation",           # separate native-alpha composition
    "psd-5-2": "train",
    "panorama-bw-page-02": "train",    # consensus may train, never evaluate participating families
    "panorama-bw-page-12": "train",
}

EXPLICIT_HIERARCHY_KINDS = {
    "6/5": "animal",   # source group name: animals
    "7/2": "animal",   # source group name: bierd
    "8/14": "character",  # source group name: воины
    "6/1": "fx",       # source group name: planets
    "6/2": "fx",       # source group name: planet
}


def _explicit_semantic(annotation: GoldAnnotation) -> GoldAnnotation:
    if annotation.source_composition_id != "psd-app-bg-chiba5":
        return annotation
    native_path = annotation.asset_version_id.split(":")[-2]
    matches = [(prefix, kind) for prefix, kind in EXPLICIT_HIERARCHY_KINDS.items()
               if native_path == prefix or native_path.startswith(prefix + "/")]
    if not matches:
        return annotation
    prefix, kind = max(matches, key=lambda item: len(item[0]))
    return replace(
        annotation, semantic_kind=kind,
        review_evidence=tuple(annotation.review_evidence) + (f"semantic_label:explicit_psd_group:{prefix}:{kind}",),
    )


def derive(source_manifest: Path) -> GoldDataset:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    annotations = []
    for raw in payload["annotations"]:
        annotation = GoldAnnotation(**raw)
        if annotation.source_composition_id not in SPLITS:
            raise ValueError(f"unassigned source composition: {annotation.source_composition_id}")
        annotations.append(_explicit_semantic(replace(annotation, split=SPLITS[annotation.source_composition_id])))
    dataset = GoldDataset("gold-v2-independent-split-2026-08-12", tuple(annotations))
    validate_gold_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = derive(args.source)
    write_gold_manifest(dataset, args.out)
    counts = {split: sum(item.split == split for item in dataset.annotations) for split in ("train", "validation", "test")}
    print(json.dumps({"dataset_version": dataset.version, "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
