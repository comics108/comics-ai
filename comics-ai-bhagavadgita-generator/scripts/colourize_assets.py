"""Deterministic paired palette transfer and autonomous colourization gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def luminance_palette_transfer(bw, reference_colour, invalid_mask, bins: int = 16):
    """Transfer reference chroma by luminance bin while retaining B&W luminance exactly."""
    import cv2

    bw_gray = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    bw_lab_l = cv2.cvtColor(bw, cv2.COLOR_BGR2LAB)[:, :, 0]
    reference_lab = cv2.cvtColor(reference_colour, cv2.COLOR_BGR2LAB)
    reference_l = reference_lab[:, :, 0]
    valid = invalid_mask == 0
    output_lab = np.empty_like(reference_lab)
    output_lab[:, :, 0] = bw_lab_l
    global_ab = np.median(reference_lab[:, :, 1:][valid], axis=0)
    indices = np.minimum((bw_gray.astype(np.int32) * bins) // 256, bins - 1)
    reference_indices = np.minimum((reference_l.astype(np.int32) * bins) // 256, bins - 1)
    for index in range(bins):
        sample = valid & (reference_indices == index)
        ab = np.median(reference_lab[:, :, 1:][sample], axis=0) if sample.any() else global_ab
        output_lab[:, :, 1:][indices == index] = np.rint(ab).astype(np.uint8)
    output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output[~valid] = bw[~valid]
    return output


def _edge_f1(left_gray, right_gray, valid) -> float:
    import cv2

    left = (cv2.Canny(left_gray, 60, 160) > 0) & valid
    right = (cv2.Canny(right_gray, 60, 160) > 0) & valid
    kernel = np.ones((5, 5), np.uint8)
    left_near = cv2.dilate(left.astype(np.uint8), kernel) > 0
    right_near = cv2.dilate(right.astype(np.uint8), kernel) > 0
    precision = float((right & left_near).sum()) / max(1, int(right.sum()))
    recall = float((left & right_near).sum()) / max(1, int(left.sum()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def evaluate_colourization(bw, target_colour, candidate, invalid_mask) -> dict:
    import cv2

    valid = invalid_mask == 0
    bw_gray = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    target_lab = cv2.cvtColor(target_colour, cv2.COLOR_BGR2LAB).astype(np.float32)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_e = np.linalg.norm(target_lab - candidate_lab, axis=2)
    bw_lab_l = cv2.cvtColor(bw, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    luminance_mae = np.abs(candidate_lab[:, :, 0] - bw_lab_l)
    return {
        "ink_edge_f1": _edge_f1(bw_gray, candidate_gray, valid),
        "mean_delta_e76": float(delta_e[valid].mean()),
        "luminance_mae": float(luminance_mae[valid].mean()),
        "valid_fraction": float(valid.mean()),
    }


def decide_colourizer(metrics: list[dict], *, learned: bool) -> dict:
    if not metrics:
        raise ValueError("colourizer decision requires metrics")
    mean_edge = sum(item["ink_edge_f1"] for item in metrics) / len(metrics)
    mean_delta = sum(item["mean_delta_e76"] for item in metrics) / len(metrics)
    max_luminance = max(item["luminance_mae"] for item in metrics)
    failures = []
    if mean_edge < .95:
        failures.append("ink_edge_preservation_below_0.95")
    if max_luminance > 3.0:
        failures.append("luminance_geometry_drift")
    if mean_delta > 30.0:
        failures.append("held_out_palette_error_above_30")
    return {
        "mean_ink_edge_f1": mean_edge,
        "mean_delta_e76": mean_delta,
        "max_luminance_mae": max_luminance,
        "decision": "accepted" if not failures else "rejected",
        "failures": failures,
    }


def run_deterministic(registration_manifest: Path, output_root: Path) -> dict:
    import cv2

    registration = json.loads(registration_manifest.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for pair in registration["pairs"]:
        for index, crop in enumerate(pair["crops"]):
            bw = cv2.imread(crop["bw_file"], cv2.IMREAD_COLOR)
            colour = cv2.imread(crop["colour_file"], cv2.IMREAD_COLOR)
            invalid = cv2.imread(crop["invalid_mask_file"], cv2.IMREAD_GRAYSCALE)
            if bw is None or colour is None or invalid is None:
                raise ValueError("cannot decode registered colourization crop")
            candidate = luminance_palette_transfer(bw, colour, invalid)
            output = output_root / f"{pair['id']}-crop-{index:03}.png"
            cv2.imwrite(str(output), candidate)
            records.append({
                "pair_id": pair["id"],
                "crop_index": index,
                "candidate_file": str(output),
                "candidate_sha256": _sha256(output),
                "metrics": evaluate_colourization(bw, colour, candidate, invalid),
            })
    decision = decide_colourizer([item["metrics"] for item in records], learned=False)
    return {
        "schema_version": 1,
        "method": "luminance-bin-palette-transfer-v1",
        "registration_manifest_sha256": _sha256(registration_manifest),
        "candidate_count": len(records),
        "records": records,
        "aggregate": decision,
        "scope": (
            "paired-reference baseline only; not a general unpaired colourizer and not canonical "
            "identity/palette evidence"
        ),
    }


def run_learned(registration_manifest: Path, checkpoint: Path, output_root: Path, device=None) -> dict:
    import cv2
    import torch
    from compact_colourizer import CompactChromaUNet

    registration = json.loads(registration_manifest.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_family") != "compact_chroma_unet":
        raise ValueError("unsupported learned colourizer checkpoint")
    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CompactChromaUNet().to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    validation_pair = payload["history"]["validation_pair"]
    for pair in registration["pairs"]:
        if pair["id"] != validation_pair:
            continue
        for index, crop in enumerate(pair["crops"]):
            bw = cv2.imread(crop["bw_file"], cv2.IMREAD_COLOR)
            colour = cv2.imread(crop["colour_file"], cv2.IMREAD_COLOR)
            invalid = cv2.imread(crop["invalid_mask_file"], cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(cv2.resize(bw, (256, 256)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0)
            tensor = tensor.to(resolved_device) / 255.
            with torch.no_grad():
                chroma = model(tensor)[0].cpu().numpy().transpose(1, 2, 0)
            chroma = cv2.resize(chroma, (bw.shape[1], bw.shape[0]), interpolation=cv2.INTER_LINEAR)
            lab = cv2.cvtColor(bw, cv2.COLOR_BGR2LAB)
            lab[:, :, 1:] = np.clip(chroma * 128. + 128., 0, 255).astype(np.uint8)
            candidate = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            candidate[invalid > 0] = bw[invalid > 0]
            output = output_root / f"{pair['id']}-crop-{index:03}.png"
            cv2.imwrite(str(output), candidate)
            records.append({"pair_id": pair["id"], "crop_index": index,
                            "candidate_file": str(output), "candidate_sha256": _sha256(output),
                            "metrics": evaluate_colourization(bw, colour, candidate, invalid)})
    return {
        "schema_version": 1, "method": "compact-chroma-unet-v1",
        "registration_manifest_sha256": _sha256(registration_manifest),
        "checkpoint_sha256": _sha256(checkpoint), "held_out_pair": validation_pair,
        "candidate_count": len(records), "records": records,
        "aggregate": decide_colourizer([item["metrics"] for item in records], learned=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("registration_manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = (
        run_learned(args.registration_manifest, args.checkpoint, args.output_root, args.device)
        if args.checkpoint
        else run_deterministic(args.registration_manifest, args.output_root)
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(args.report), **report["aggregate"]}))


if __name__ == "__main__":
    main()
