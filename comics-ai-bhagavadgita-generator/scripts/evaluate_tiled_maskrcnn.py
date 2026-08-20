#!/usr/bin/env python3
"""Full-panorama tiled geometry evaluation for the Gold true-mask Mask R-CNN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_tiled_instances import bbox_match, merge_boxes, one_to_one_matches
from train_gold_maskrcnn import build_model


def predict_page(model, image: Image.Image, *, device="cpu", tile_width=512, overlap=128, threshold=.3):
    import torch

    origins = list(range(0, max(1, image.width - tile_width + 1), tile_width - overlap))
    last = max(0, image.width - tile_width)
    if not origins or origins[-1] != last: origins.append(last)
    raw = []
    with torch.no_grad():
        for origin in origins:
            tile = image.crop((origin, 0, min(image.width, origin + tile_width), image.height))
            tensor = torch.from_numpy(np.asarray(tile).copy()).permute(2, 0, 1).float().to(device) / 255
            output = model([tensor])[0]
            for index in torch.where(output["scores"] >= threshold)[0].tolist():
                x0, y0, x1, y1 = (round(float(value)) for value in output["boxes"][index])
                if x1 > x0 and y1 > y0:
                    raw.append((x0 + origin, y0, x1 + origin, y1))
    return raw, merge_boxes(raw)


def evaluate(manifest: Path, repository_root: Path, checkpoint: Path, *, device="cpu"):
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    gold = json.loads(manifest.read_text(encoding="utf-8"))
    panorama = [item for item in gold["annotations"] if item["accepted"] and item["source_kind"] == "panorama"]
    by_source = {}
    for item in panorama: by_source.setdefault(item["source_composition_id"], []).append(item)
    pages=[]; total_matches=total_predictions=total_truth=total_duplicates=total_collapsed=0
    for source, items in sorted(by_source.items()):
        page=source.rsplit("-",1)[-1]
        with Image.open(repository_root/f"work/bhagavadgita/production/gold-v1/panorama-source/bw-page-{page}.jpg") as opened:
            raw, merged=predict_page(model,opened.convert("RGB"),device=device)
        truth=[tuple(int(value) for value in next(e for e in item["review_evidence"] if e.startswith("render_bbox:")).split(":",1)[1].split(",")) for item in items]
        matches=len(one_to_one_matches(merged,truth))
        per_truth=[sum(bbox_match(pred,target) for pred in merged) for target in truth]
        per_pred=[sum(bbox_match(pred,target) for target in truth) for pred in merged]
        duplicates=sum(max(0,value-1) for value in per_truth); collapsed=sum(max(0,value-1) for value in per_pred)
        total_matches+=matches; total_predictions+=len(merged); total_truth+=len(truth); total_duplicates+=duplicates; total_collapsed+=collapsed
        pages.append({"source":source,"raw_instances":len(raw),"merged_instances":len(merged),"truth_instances":len(truth),"one_to_one_matches":matches,"duplicate_matches":duplicates,"collapsed_truth_matches":collapsed})
    recall=total_matches/max(1,total_truth); precision=total_matches/max(1,total_predictions); duplicate=total_duplicates/max(1,total_matches+total_duplicates)
    passed=recall>=.85 and precision>=.85 and duplicate<=.03 and total_collapsed==0
    return {"schema_version":1,"dataset_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"predictor":checkpoint.stem,"pages":pages,"instance_recall_at_bbox_match":recall,"instance_precision_at_bbox_match":precision,"duplicate_instance_rate":duplicate,"collapsed_truth_match_count":total_collapsed,"promotion_gate":"accepted" if passed else "rejected"}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository-root",type=Path,default=Path.cwd()); parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--device",default="cpu")
    args=parser.parse_args(); root=args.repository_root.resolve(); report=evaluate(args.manifest if args.manifest.is_absolute() else root/args.manifest,root,args.checkpoint if args.checkpoint.is_absolute() else root/args.checkpoint,device=args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream: json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2); stream.write("\n")
    print(json.dumps({"gate":report["promotion_gate"],"recall":report["instance_recall_at_bbox_match"],"precision":report["instance_precision_at_bbox_match"]}))


if __name__=="__main__": main()
