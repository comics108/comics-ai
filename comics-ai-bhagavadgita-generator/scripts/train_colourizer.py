"""Train the compact chroma-only candidate on source-disjoint registered pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from compact_colourizer import CompactChromaUNet


class PairedColourDataset(Dataset):
    def __init__(self, crops: list[dict]):
        self.crops = crops

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        crop = self.crops[index]
        bw = cv2.imread(crop["bw_file"], cv2.IMREAD_COLOR)
        colour = cv2.imread(crop["colour_file"], cv2.IMREAD_COLOR)
        invalid = cv2.imread(crop["invalid_mask_file"], cv2.IMREAD_GRAYSCALE)
        bw = cv2.resize(bw, (256, 256), interpolation=cv2.INTER_AREA)
        colour = cv2.resize(colour, (256, 256), interpolation=cv2.INTER_AREA)
        invalid = cv2.resize(invalid, (256, 256), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bw, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(colour, cv2.COLOR_BGR2LAB)
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.
        chroma = torch.from_numpy(lab[:, :, 1:].copy()).permute(2, 0, 1).float()
        chroma = (chroma - 128.) / 128.
        valid = torch.from_numpy((invalid == 0).copy()).float().unsqueeze(0)
        return image, chroma, valid


def train(manifest: Path, checkpoint: Path, *, epochs=30, device=None, seed=20260811):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    validation_pair = "colour-06-bw-07"
    train_crops, validation_crops = [], []
    for pair in payload["pairs"]:
        target = validation_crops if pair["id"] == validation_pair else train_crops
        target.extend(pair["crops"])
    if not train_crops or not validation_crops:
        raise ValueError("colourizer source-disjoint split is empty")
    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader = DataLoader(PairedColourDataset(train_crops), batch_size=4, shuffle=True)
    validation_loader = DataLoader(PairedColourDataset(validation_crops), batch_size=4)
    model = CompactChromaUNet().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {
        "seed": seed, "device": resolved_device, "train_count": len(train_crops),
        "validation_count": len(validation_crops), "validation_pair": validation_pair,
        "train_loss": [], "validation_l1": [],
    }
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        total = 0.
        for images, chroma, valid in train_loader:
            images, chroma, valid = (
                images.to(resolved_device), chroma.to(resolved_device), valid.to(resolved_device)
            )
            prediction = model(images)
            loss = (functional.smooth_l1_loss(prediction, chroma, reduction="none") * valid).sum()
            loss = loss / max(1, int(valid.sum().item()) * 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(images)
        history["train_loss"].append(total / len(train_crops))
        model.eval()
        validation_total = pixels = 0.
        with torch.no_grad():
            for images, chroma, valid in validation_loader:
                images, chroma, valid = (
                    images.to(resolved_device), chroma.to(resolved_device), valid.to(resolved_device)
                )
                error = (model(images) - chroma).abs() * valid
                validation_total += float(error.sum().item())
                pixels += int(valid.sum().item()) * 2
        validation_l1 = validation_total / max(1, pixels)
        history["validation_l1"].append(validation_l1)
        if validation_l1 < best_loss:
            best_loss = validation_l1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "train_loss": history["train_loss"][-1],
                          "validation_l1": validation_l1}))
    history["best_validation_l1"] = best_loss
    history["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    if checkpoint.exists() or checkpoint.with_suffix(".history.json").exists():
        raise FileExistsError(f"immutable colourizer output exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "model_family": "compact_chroma_unet",
                "model_state": best_state, "history": history}, checkpoint)
    checkpoint.with_suffix(".history.json").write_text(
        json.dumps(history, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("registration_manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.registration_manifest, args.out, epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
