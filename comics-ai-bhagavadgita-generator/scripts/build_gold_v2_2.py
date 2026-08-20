#!/usr/bin/env python3
"""Gold v2.2 split with source-disjoint native-alpha test and train semantic coverage."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from build_gold_dataset import GoldAnnotation, GoldDataset, validate_gold_dataset, write_gold_manifest


SPLITS = {
    "psd-5-1": "test", "psd-5-2": "test", "psd-app-bg-chiba5": "train",
    "panorama-bw-page-02": "train", "panorama-bw-page-12": "validation",
}


def derive(source: Path) -> GoldDataset:
    payload = json.loads(source.read_text(encoding="utf-8"))
    annotations = []
    for raw in payload["annotations"]:
        item = GoldAnnotation(**raw)
        item = replace(item, split=SPLITS[item.source_composition_id])
        if item.source_composition_id == "psd-5-2":
            native_path = item.asset_version_id.split(":")[-2]
            if native_path == "0/0/5" or native_path.startswith("0/0/5/"):
                item = replace(
                    item, semantic_kind="animal",
                    review_evidence=tuple(item.review_evidence) + (
                        "semantic_label:explicit_psd_group:0/0/5:animal:name=cat",
                    ),
                )
        annotations.append(item)
    dataset = GoldDataset("gold-v2.2-semantic-split-2026-08-12", tuple(annotations))
    validate_gold_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dataset = derive(args.source)
    write_gold_manifest(dataset, args.out)
    print(json.dumps({split: sum(item.split == split for item in dataset.annotations)
                      for split in ("train", "validation", "test")}, sort_keys=True))


if __name__ == "__main__":
    main()
