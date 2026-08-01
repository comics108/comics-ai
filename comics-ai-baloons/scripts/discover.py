#!/usr/bin/env python3
"""Stage 1: structural balloon-layer discovery.

A Layer qualifies as a balloon/text layer iff >=2 of its images[] entries are non-empty (have a
"file"). This is deliberately NOT filename-based -- see flows/sdd-comics-ai-baloons/
02-specifications.md "Editor Schema Ground Truth" for why (>=8 divergent filename conventions
found in the survey; the structural rule is what actually holds across all of them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import comics_io
from models import BalloonLayer, ImageSlot

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "work" / "balloons.jsonl"


def find_translate_offset(animations: list[dict]) -> tuple[int, int]:
    """Best-effort resting position of the layer, from its base TranslateAnim (the one with no
    "start" key -- animations with "start" are scroll-triggered follow-up moves, not the resting
    position). Used only as a soft matching tie-breaker later, precision doesn't matter much.
    """
    base = [
        a
        for a in animations
        if "TranslateAnim" in a.get("$type", "") and "start" not in a
    ]
    if base:
        a = base[0]
        return (a.get("x", 0), a.get("y", 0))
    any_translate = [a for a in animations if "TranslateAnim" in a.get("$type", "")]
    if any_translate:
        a = any_translate[0]
        return (a.get("x", 0), a.get("y", 0))
    return (0, 0)


def discover_file(path: Path) -> list[BalloonLayer]:
    archive = comics_io.ComicsArchive(path)
    data = archive.read_data_json()
    out: list[BalloonLayer] = []
    for idx, layer in enumerate(data.get("layers", [])):
        images = layer.get("images", [])
        populated = [(i, im) for i, im in enumerate(images) if im.get("file")]
        if len(populated) < 2:
            continue
        slots = {
            i: ImageSlot(
                lang_index=i,
                file_template=im["file"],
                width=im.get("width", 0),
                height=im.get("height", 0),
            )
            for i, im in populated
        }
        offset = find_translate_offset(layer.get("animations", []))
        out.append(
            BalloonLayer(
                source_file=path.name,
                layer_index=idx,
                slots=slots,
                canvas_offset=offset,
            )
        )
    return out


def discover_all(dataset_dir: Path) -> list[BalloonLayer]:
    balloons: list[BalloonLayer] = []
    for p in sorted(dataset_dir.glob("*.comics")):
        balloons.extend(discover_file(p))
    return balloons


def write_jsonl(balloons: list[BalloonLayer], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for b in balloons:
            f.write(json.dumps(b.to_jsonable(), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_dir", nargs="?", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    balloons = discover_all(Path(args.dataset_dir))
    out_path = Path(args.out)
    write_jsonl(balloons, out_path)
    print(f"Discovered {len(balloons)} balloons -> {out_path}")


if __name__ == "__main__":
    main()
