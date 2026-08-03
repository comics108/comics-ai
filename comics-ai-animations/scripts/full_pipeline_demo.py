#!/usr/bin/env python3
"""Criterion 5 (flows/sdd-comics-ai-transformations/01-requirements.md v0.3): a real, end-to-end
run of cut -> position -> transform for one real page, to report actual per-page completeness --
not a new pipeline, a real invocation of three already-built stages (cutting =
comics-multimodal's infer_segmenter.py, already run into work/regions.jsonl; positioning =
comics-ai-positioning's baseline_position.py; transformation = this flow's own
baseline_transform.py) plus a check of what `sdd-comics-ai-script-context` already knows about the
episode's real narrative content, for a plausibility cross-check.

Balloon *text* rendering itself (comics-ai-baloons' matching/rendering stage) is intentionally not
re-invoked here -- per Requirements' Won't-Have, "no new balloon translation work" is out of this
flow's scope; this script reports which regions are balloon-kind and leaves their actual text
content to that flow's own established pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_POSITIONING_SCRIPTS = (
    Path(__file__).resolve().parents[3] / "comics-ai" / "comics-positioning" / "scripts"
)
if str(_POSITIONING_SCRIPTS) not in sys.path:
    sys.path.append(str(_POSITIONING_SCRIPTS))

import transforms_bridge as tb  # noqa: E402
from baseline_position import load_stats, position_page  # noqa: E402
from baseline_transform import propose_reveal  # noqa: E402
from positioning_models import RegionFeatures  # noqa: E402
from transform_stats import compute_stats  # noqa: E402

MULTIMODAL_WORK_DIR = (
    Path(__file__).resolve().parents[3] / "comics-ai" / "comics-multimodal" / "work"
)
REGIONS_JSONL = MULTIMODAL_WORK_DIR / "regions.jsonl"
ALIGNMENT_JSONL = MULTIMODAL_WORK_DIR / "alignment.jsonl"
SCRIPT_CONTEXT_SCENES_DIR = (
    Path(__file__).resolve().parents[3] / "comics-ai" / "comics-script-context" / "work" / "scenes"
)


def load_regions_for_page(photo_file: str, page_index: int) -> list[dict]:
    regions = []
    with REGIONS_JSONL.open() as f:
        for line in f:
            d = json.loads(line)
            if d["photo_file"] == photo_file and d["page_index"] == page_index:
                regions.append(d)
    return regions


def find_episode_for_page(photo_file: str, page_index: int) -> str | None:
    with ALIGNMENT_JSONL.open() as f:
        for line in f:
            d = json.loads(line)
            if d["photo_file"] == photo_file and d["page_index"] == page_index and d["status"] == "matched":
                return d["episode_file"]
    return None


def load_script_context(episode_file: str) -> dict | None:
    path = SCRIPT_CONTEXT_SCENES_DIR / f"{episode_file}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def run_demo(photo_file: str, page_index: int) -> dict:
    regions = load_regions_for_page(photo_file, page_index)
    if not regions:
        raise ValueError(f"No regions found for {photo_file} page {page_index}")

    episode_file = find_episode_for_page(photo_file, page_index)

    # --- Positioning stage (real baseline_position.py) ---
    ordered = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    region_features = [
        RegionFeatures(
            kind=r["predicted_kind"],
            kind_source="predicted",
            local_bbox=tuple(r["bbox"]),
            page_index=page_index,
            reading_order_index=i,
        )
        for i, r in enumerate(ordered)
    ]
    region_ids = [str(i) for i in range(len(region_features))]
    position_stats = load_stats()
    position_proposals = {
        p.region_id: p for p in position_page(region_features, position_stats, region_ids=region_ids)
    }

    # --- Transformation stage (this flow's own baseline_transform.py) ---
    transform_stats = compute_stats()

    # --- Script-context cross-check (real narrative content for this episode, if any) ---
    scene = load_script_context(episode_file) if episode_file else None

    page_report = []
    for i, r in enumerate(ordered):
        rid = str(i)
        pos = position_proposals[rid]
        reveal = propose_reveal(r["predicted_kind"], transform_stats)
        page_report.append(
            {
                "region_id": rid,
                "kind": r["predicted_kind"],
                "kind_confidence": round(r["confidence"], 2),
                "local_bbox": r["bbox"],
                "proposed_position": {"x": pos.proposed_x, "y": pos.proposed_y},
                "proposed_reveal": {
                    prop: {"occurs": rv.occurs, "end": rv.end} for prop, rv in reveal.items()
                },
            }
        )

    return {
        "photo_file": photo_file,
        "page_index": page_index,
        "episode_file": episode_file,
        "region_count": len(regions),
        "regions": page_report,
        "script_context": (
            {"characters": scene["characters"], "text_source": scene["text_source"]}
            if scene
            else None
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", required=True)
    parser.add_argument("--page", type=int, default=0)
    args = parser.parse_args()

    report = run_demo(args.photo, args.page)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
