#!/usr/bin/env python3
"""Render source-context overlays for automated consensus-mask sanity review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(manifest_path: Path, output: Path, *, tile_size: tuple[int, int] = (420, 260),
           reviewer_manifest: Path | None = None) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = {}
    # Source files are stable by naming convention in the reviewer workspace.
    source_root = manifest_path.parents[2] / "gold-v1/panorama-source"
    source_by_page = {}
    if reviewer_manifest:
        reviewer = json.loads(reviewer_manifest.read_text(encoding="utf-8"))
        source_by_page = {item["page_id"]: Path(item["source_file"]) for item in reviewer["pages"]}
    records = payload.get("records", payload.get("candidates", []))
    cols = 3
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1]), "white")
    for index, record in enumerate(records):
        page_id = record["page_id"]
        if page_id not in pages:
            pages[page_id] = Image.open(source_by_page.get(page_id, source_root / f"{page_id}.jpg")).convert("RGB")
        page = pages[page_id]
        x0, y0, x1, y1 = record["page_bbox"]
        pad = max(80, int(max(x1 - x0, y1 - y0) * .6))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(page.width, x1 + pad), min(page.height, y1 + pad)
        crop = page.crop((cx0, cy0, cx1, cy1)).convert("RGBA")
        mask = np.asarray(Image.open(record["mask_file"]).convert("L")) > 0
        wx0 = record["window_index"] * (2048 - 384)
        window_count = max(1, (page.width - 2048 + (2048 - 384) - 1) // (2048 - 384) + 1)
        if record["window_index"] == window_count - 1:
            wx0 = page.width - 2048
        local = mask[cy0:cy1, max(0, cx0 - wx0):max(0, cx1 - wx0)]
        # Correctly align when the context begins before the current window.
        overlay = np.zeros((crop.height, crop.width, 4), dtype=np.uint8)
        target_x0 = max(0, wx0 - cx0)
        source_x0 = max(0, cx0 - wx0)
        usable_w = min(mask.shape[1] - source_x0, crop.width - target_x0)
        source_y0 = cy0
        usable_h = min(mask.shape[0] - source_y0, crop.height)
        if usable_w > 0 and usable_h > 0:
            selected = mask[source_y0:source_y0 + usable_h, source_x0:source_x0 + usable_w]
            overlay[:usable_h, target_x0:target_x0 + usable_w, 0] = selected * 255
            overlay[:usable_h, target_x0:target_x0 + usable_w, 3] = selected * 120
        crop.alpha_composite(Image.fromarray(overlay, "RGBA"))
        crop.thumbnail((tile_size[0], tile_size[1] - 30), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", tile_size, "white")
        cell.paste(crop.convert("RGB"), ((tile_size[0] - crop.width) // 2, 28))
        record_id = record.get("id", record.get("sam_proposal_id", "candidate"))
        metric_name = "IoU" if "agreement_iou" in record else "boundary"
        metric = record.get("agreement_iou", record.get("boundary_support", 0.))
        ImageDraw.Draw(cell).text((6, 6), f"{record_id} {metric_name}={metric:.3f}", fill="black")
        sheet.paste(cell, ((index % cols) * tile_size[0], (index // cols) * tile_size[1]))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewer-manifest", type=Path)
    args = parser.parse_args(); render(args.manifest, args.out, reviewer_manifest=args.reviewer_manifest)


if __name__ == "__main__":
    main()
