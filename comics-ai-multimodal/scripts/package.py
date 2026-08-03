#!/usr/bin/env python3
"""Task 9.1: package predicted regions (Phase 6, work/regions.jsonl) for each matched photo/page
into a new, valid .comics file.

Design note: each output file's canvas IS that photo's own rectified page crop (the 256x256-
resolution coordinate space `CutRegion` bboxes and pixel data already live in, per
infer_segmenter.py/build_library.py) -- not an attempt to reposition content into the matched
episode's *canvas* coordinates. Revision 1.1/1.2 already established no pixel-level mapping from a
real photo into canvas space exists or is practically obtainable; packaging output at the photo's
own coordinate space is the self-consistent, honestly-computable interpretation of Requirements'
acceptance criterion ("a new, valid .comics file... produced... from the cut regions").

Reuses comics-ai-baloons' tiling convention (512px tiles, same filename template shape) via the
bridge -- but writes a fresh zip archive directly (not `comics_io.write_comics`, which copies from
an existing source archive; there is no "source" here, this is new content built from scratch).
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

import baloons_bridge
from build_library import extract_crop_image
from infer_segmenter import TRAIN_SIZE

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
DEFAULT_OUT_DIR = WORK_DIR / "output"

MIN_REGION_DIM = 2  # px; degenerate boxes below this can't be tiled/rendered meaningfully


@dataclass
class PackageResult:
    photo_file: str
    page_index: int
    episode_file: str
    output_path: str | None
    layer_count: int
    status: str  # "packaged" | "skipped_no_regions"


def package_photo_page(
    photo_file: str,
    page_index: int,
    episode_file: str,
    regions: list[dict],
    lowcamera_dir: Path,
    out_dir: Path,
) -> PackageResult:
    th, tw = TRAIN_SIZE
    layers = []
    tile_entries: dict[str, bytes] = {}

    for i, r in enumerate(regions):
        x0, y0, x1, y1 = r["bbox"]
        w, h = x1 - x0, y1 - y0
        if w < MIN_REGION_DIM or h < MIN_REGION_DIM:
            continue
        crop = extract_crop_image(photo_file, page_index, (x0, y0, x1, y1), lowcamera_dir)
        if crop is None:
            continue

        file_template = f"r{i}_{{0}}_{{1}}_{{2}}.png"
        image = Image.fromarray(crop).convert("RGBA")
        tiles = baloons_bridge.retile_image(image, file_template)
        for name, data in tiles.items():
            tile_entries[f"layers/{name}"] = data

        layers.append(
            {
                "images": [{"file": file_template, "width": w, "height": h}, {}, {}],
                "animations": [
                    {
                        "$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor",
                        "x": x0,
                        "y": y0,
                    }
                ],
                "kind": r["predicted_kind"],
            }
        )

    stem = f"{Path(photo_file).stem}_p{page_index}"
    if not layers:
        return PackageResult(photo_file, page_index, episode_file, None, 0, "skipped_no_regions")

    data_json = {"width": tw, "height": th, "layers": layers, "sounds": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.comics"
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.json", json.dumps(data_json, ensure_ascii=False, indent=2))
        for name, data in tile_entries.items():
            zf.writestr(name, data)

    return PackageResult(photo_file, page_index, episode_file, str(out_path), len(layers), "packaged")


def package_all(
    alignment_path: Path = DEFAULT_ALIGNMENT,
    regions_path: Path = DEFAULT_REGIONS,
    lowcamera_dir: Path = DEFAULT_LOWCAMERA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[PackageResult]:
    regions_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with regions_path.open() as f:
        for line in f:
            r = json.loads(line)
            regions_by_page[(r["photo_file"], r["page_index"])].append(r)

    results = []
    with alignment_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["status"] != "matched":
                continue
            key = (entry["photo_file"], entry["page_index"])
            regions = regions_by_page.get(key, [])
            result = package_photo_page(
                entry["photo_file"], entry["page_index"], entry["episode_file"], regions, lowcamera_dir, out_dir
            )
            results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    results = package_all(out_dir=args.out)
    packaged = sum(1 for r in results if r.status == "packaged")
    print(f"{packaged}/{len(results)} photo/pages packaged -> {args.out}")


if __name__ == "__main__":
    main()
