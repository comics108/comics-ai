#!/usr/bin/env python3
"""Run independent reviewers over registered author-colour and B&W source renditions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from graph_panorama_reviewer import review_page as graph_review
from sam_panorama_reviewer import review_page as sam_review


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(registration_path: Path, preview_root: Path, checkpoint: Path, output_root: Path,
          *, points_per_side: int = 12, crop_n_layers: int = 0) -> tuple[dict, dict]:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    sam_pages, graph_pages, pair_evidence = [], [], []
    for pair in registration["pairs"]:
        colour = Path(pair["aligned_colour_file"])
        bw = preview_root / f"bw-{pair['bw_page']:02}.jpg"
        if not colour.is_file() or not bw.is_file():
            raise ValueError(f"registered source rendition missing for {pair['id']}")
        with Image.open(colour) as registered_colour:
            if tuple(pair["preview_resolution"]) != registered_colour.size:
                raise ValueError("registered colour resolution drift")
        page_id = pair["id"]
        sam_pages.append(sam_review(colour, checkpoint, output_root / "sam", page_id=page_id,
                                    window=2048, overlap=384, points_per_side=points_per_side,
                                    crop_n_layers=crop_n_layers))
        graph_pages.append(graph_review(bw, output_root / "graph", page_id=page_id,
                                        window=2048, overlap=384))
        pair_evidence.append({
            "page_id": page_id, "bw_page": pair["bw_page"], "colour_page": pair["colour_page"],
            "registered_ink_edge_f1": pair["registered_ink_edge_f1"],
            "valid_coverage": pair["valid_coverage"], "bw_source_sha256": _sha256(bw),
            "aligned_colour_sha256": _sha256(colour),
        })
    common = {"schema_version": 1, "registration_manifest_sha256": _sha256(registration_path),
              "paired_source_evidence": pair_evidence}
    sam = {**common, "reviewer_family": "meta-sam-vit-b-on-author-colour-rendition",
           "checkpoint_sha256": _sha256(checkpoint), "model_license": "Apache-2.0",
           "independence": "author_colour_pixels_not_used_by_bw_graph_reviewer",
           "configuration": {"points_per_side": points_per_side, "crop_n_layers": crop_n_layers},
           "page_count": len(sam_pages), "proposal_count": sum(x["proposal_count"] for x in sam_pages),
           "pages": sam_pages}
    graph = {**common, "reviewer_family": "felzenszwalb-multiscale-on-bw-rendition",
             "implementation_license": "BSD-3-Clause scikit-image",
             "independence": "bw_pixels_no_sam_coco_or_colour_output",
             "page_count": len(graph_pages), "proposal_count": sum(x["proposal_count"] for x in graph_pages),
             "pages": graph_pages}
    return sam, graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam-out", type=Path, required=True)
    parser.add_argument("--graph-out", type=Path, required=True)
    parser.add_argument("--points-per-side", type=int, default=12)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    args = parser.parse_args()
    sam, graph = build(args.registration, args.previews, args.checkpoint, args.output_root,
                       points_per_side=args.points_per_side, crop_n_layers=args.crop_n_layers)
    for path, payload in ((args.sam_out, sam), (args.graph_out, graph)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps({"pages": sam["page_count"], "sam": sam["proposal_count"],
                      "graph": graph["proposal_count"]}))


if __name__ == "__main__":
    main()
