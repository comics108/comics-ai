#!/usr/bin/env python3
"""Task 7.1: hand off balloon-kind content to comics-ai-baloons -- as a lookup/cross-reference
against its already-completed work, not by re-invoking its pipeline.

Revised understanding vs. Specifications' original framing: comics-ai-baloons' discover/extract/
OCR/match/classify/render chain operates on `BalloonLayer` records that reference real tile data
inside a `dataset/*.comics` zip archive -- it has no notion of an arbitrary photo-extracted pixel
crop (this project's own `CutRegion`, Phase 6) as a "layer" to feed in. There is also no need to
re-run it: `comics-ai-baloons` has **already processed the entire dataset** (verified: 825 balloons
discovered/matched, 1586 per-language render records, 22 packaged output `.comics` files in its
`work/output/`). The meaningful handoff for a matched photo (Phase 5, which already tells us
`episode_file` + `ground_truth_cluster`) is therefore to **look up** that pipeline's existing
per-layer results for the balloon layers in the matched cluster, and to **cross-check** them
against this project's own photo-predicted balloon `CutRegion`s (Phase 6) as a sanity signal --
not to reconstruct or duplicate comics-ai-baloons' own logic.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import baloons_bridge
from render_canvas import GroundTruthRegion

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_ALIGNMENT = WORK_DIR / "alignment.jsonl"
DEFAULT_REGIONS = WORK_DIR / "regions.jsonl"
DEFAULT_CANVAS_DIR = WORK_DIR / "canvas"
DEFAULT_OUT = WORK_DIR / "balloon_handoff.jsonl"

BALOONS_WORK_DIR = baloons_bridge.BALOONS_APP_DIR / "work"
BALOONS_BALLOONS_JSONL = BALOONS_WORK_DIR / "balloons.jsonl"
BALOONS_MATCHES_JSONL = BALOONS_WORK_DIR / "matches.jsonl"
BALOONS_RENDERS_JSONL = BALOONS_WORK_DIR / "renders.jsonl"
BALOONS_OUTPUT_DIR = BALOONS_WORK_DIR / "output"


@dataclass
class BalloonHandoffResult:
    photo_file: str
    page_index: int
    episode_file: str
    real_balloon_layer_indexes: list[int]  # ground-truth balloon layers in the matched cluster
    predicted_balloon_region_count: int  # this project's own photo-based prediction (Phase 6)
    translated_layer_indexes: list[int]  # subset with a confident comics-ai-baloons CSV match
    rendered_languages_by_layer: dict[int, list[str]]
    packaged_output_available: bool  # comics-ai-baloons already produced work/output/<episode>


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _index_by_episode(entries: list[dict], key: str = "source_file") -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        grouped[e[key]].append(e)
    return grouped


def real_balloon_layers_for_cluster(
    episode_file: str, ground_truth_cluster: list[int], canvas_dir: Path = DEFAULT_CANVAS_DIR
) -> list[int]:
    gt_path = canvas_dir / f"{Path(episode_file).stem}.gt.json"
    if not gt_path.is_file():
        return []
    ref_data = json.loads(gt_path.read_text())
    wanted = set(ground_truth_cluster)
    return sorted(
        r["layer_index"]
        for r in ref_data["regions"]
        if r["layer_index"] in wanted and r["kind"] == "balloon"
    )


def route_all(
    alignment_path: Path = DEFAULT_ALIGNMENT,
    regions_path: Path = DEFAULT_REGIONS,
    canvas_dir: Path = DEFAULT_CANVAS_DIR,
    out_path: Path = DEFAULT_OUT,
) -> list[BalloonHandoffResult]:
    balloons_by_episode = _index_by_episode(_load_jsonl(BALOONS_BALLOONS_JSONL))
    matches_by_episode = _index_by_episode(_load_jsonl(BALOONS_MATCHES_JSONL))
    renders_by_episode = _index_by_episode(_load_jsonl(BALOONS_RENDERS_JSONL))

    predicted_balloon_counts: dict[tuple[str, int], int] = defaultdict(int)
    for r in _load_jsonl(regions_path):
        if r["predicted_kind"] == "balloon":
            predicted_balloon_counts[(r["photo_file"], r["page_index"])] += 1

    results = []
    for entry in _load_jsonl(alignment_path):
        if entry["status"] != "matched":
            continue
        episode_file = entry["episode_file"]
        cluster = entry["ground_truth_cluster"]

        real_balloon_layers = real_balloon_layers_for_cluster(episode_file, cluster, canvas_dir)
        real_set = set(real_balloon_layers)

        matched_layers = {
            m["layer_index"]
            for m in matches_by_episode.get(episode_file, [])
            if m["status"] == "matched" and m["layer_index"] in real_set
        }

        rendered_by_layer: dict[int, list[str]] = defaultdict(list)
        for r in renders_by_episode.get(episode_file, []):
            if r["layer_index"] in real_set and r.get("rendered"):
                rendered_by_layer[r["layer_index"]].append(r["lang_code"])

        results.append(
            BalloonHandoffResult(
                photo_file=entry["photo_file"],
                page_index=entry["page_index"],
                episode_file=episode_file,
                real_balloon_layer_indexes=real_balloon_layers,
                predicted_balloon_region_count=predicted_balloon_counts.get(
                    (entry["photo_file"], entry["page_index"]), 0
                ),
                translated_layer_indexes=sorted(matched_layers),
                rendered_languages_by_layer={k: sorted(v) for k, v in rendered_by_layer.items()},
                packaged_output_available=(BALOONS_OUTPUT_DIR / episode_file).is_file(),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    results = route_all(out_path=args.out)
    with_package = sum(1 for r in results if r.packaged_output_available)
    print(f"{len(results)} matched pages routed -> {args.out}")
    print(f"{with_package}/{len(results)} have a comics-ai-baloons packaged output already available")


if __name__ == "__main__":
    main()
