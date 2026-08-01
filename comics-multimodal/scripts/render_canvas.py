#!/usr/bin/env python3
"""Stage 1: composite each dataset .comics file's full canvas at resting layer positions, and emit
a parallel ground-truth region map (flows/sdd-comics-ai-multimodal/02-specifications.md
"CanvasReference").

Known, documented simplifications for this v1 compositor (not silently glossed over):
- Rotation (`RotateAnim`) is not applied -- only 1146/4594 layers in the real dataset have any
  RotateAnim at all, and resting angles seen during Task 2.1's investigation were small (~3deg).
  Tracked as a limitation, not a correctness bug for the common case.
- Compositing uses `Image.paste(stitched, (x, y), stitched)` (mask = the image's own alpha),
  not true "over" alpha blending -- correct for opaque-over-opaque and simple partial-alpha
  layers, but does not accumulate translucency the way a real renderer's alpha-over would for
  stacked semi-transparent layers. Acceptable for synthesizing *training data* (Specifications'
  stated purpose); not claimed to be pixel-identical to the real editor's renderer.
- A layer whose resting alpha resolves to 0 (fully invisible once settled) is skipped entirely --
  it contributes nothing to the final composite and has no ground-truth region.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # canvases are legitimately large (up to ~100M px); trusted local data

import baloons_bridge
from kind_heuristic import infer_kinds_for_file
from resting_position import resolve_resting_transform

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_CANVAS_DIR = WORK_DIR / "canvas"


@dataclass
class GroundTruthRegion:
    layer_index: int
    kind: str
    kind_source: str  # "explicit" | "inferred_heuristic"
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in canvas coordinates (resting position)


@dataclass
class CanvasReference:
    episode_file: str
    width: int
    height: int
    composite_png: str
    regions: list[GroundTruthRegion]

    def to_jsonable(self) -> dict:
        d = asdict(self)
        return d


def _populated_image(layer: dict) -> dict | None:
    for im in layer.get("images", []):
        if im.get("file"):
            return im
    return None


def render_canvas_reference(
    archive: "baloons_bridge.ComicsArchive", out_dir: Path
) -> CanvasReference:
    data = archive.read_data_json()
    width, height = data["width"], data["height"]
    layers = data["layers"]

    kinds = infer_kinds_for_file(data)
    canvas = Image.new("RGBA", (width, height))
    regions: list[GroundTruthRegion] = []

    for i, layer in enumerate(layers):
        explicit_kind = layer.get("kind") or layer.get("Kind")
        kind = explicit_kind if explicit_kind else kinds[i]
        kind_source = "explicit" if explicit_kind else "inferred_heuristic"

        transform = resolve_resting_transform(layer.get("animations", []))
        if transform.alpha <= 0.0:
            continue  # settled state is fully invisible -- nothing to composite or track

        image = _populated_image(layer)
        if image is None:
            continue

        img_w, img_h = image.get("width", 0), image.get("height", 0)
        if not img_w or not img_h:
            continue

        try:
            stitched = baloons_bridge.stitch_image(archive, image["file"], img_w, img_h)
        except FileNotFoundError:
            continue  # missing tiles -- skip this layer, don't crash the whole file

        if transform.scale_x != 1.0 or transform.scale_y != 1.0:
            new_w = max(1, round(img_w * transform.scale_x))
            new_h = max(1, round(img_h * transform.scale_y))
            stitched = stitched.resize((new_w, new_h))
            img_w, img_h = new_w, new_h

        if transform.alpha < 1.0:
            r, g, b, a = stitched.split()
            a = a.point(lambda v: int(v * transform.alpha))
            stitched = Image.merge("RGBA", (r, g, b, a))

        paste_x, paste_y = transform.x, transform.y
        canvas.paste(stitched, (paste_x, paste_y), stitched)

        regions.append(
            GroundTruthRegion(
                layer_index=i,
                kind=kind,
                kind_source=kind_source,
                bbox=(paste_x, paste_y, paste_x + img_w, paste_y + img_h),
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    composite_path = out_dir / f"{archive.path.stem}.png"
    canvas.save(composite_path)

    return CanvasReference(
        episode_file=archive.path.name,
        width=width,
        height=height,
        composite_png=str(composite_path),
        regions=regions,
    )


def render_all(dataset_dir: Path | None = None, out_dir: Path = DEFAULT_CANVAS_DIR) -> list[CanvasReference]:
    files = baloons_bridge.find_comics_files(dataset_dir) if dataset_dir else baloons_bridge.find_comics_files()
    refs = []
    for f in files:
        archive = baloons_bridge.ComicsArchive(f)
        ref = render_canvas_reference(archive, out_dir)
        gt_path = out_dir / f"{f.stem}.gt.json"
        gt_path.write_text(json.dumps(ref.to_jsonable(), indent=2))
        refs.append(ref)
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_CANVAS_DIR)
    args = parser.parse_args()
    refs = render_all(out_dir=args.out)
    for ref in refs:
        print(f"{ref.episode_file}: {ref.width}x{ref.height}, {len(ref.regions)} regions")


if __name__ == "__main__":
    main()
