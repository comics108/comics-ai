"""Train a compact binary segmenter on source-disjoint Gold v1 PSD masks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from compact_segmenter import CompactBinaryUNet
from gold_segmenter_data import load_gold_dataset, load_review_pair


TRAIN_SIZE = (256, 256)


class GoldMaskDataset(Dataset):
    def __init__(self, annotations, repository_root: Path):
        self.annotations = tuple(annotations)
        self.repository_root = repository_root

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        image, mask = load_review_pair(self.annotations[index], self.repository_root)
        image = image.resize(TRAIN_SIZE, Image.Resampling.BILINEAR)
        mask = mask.resize(TRAIN_SIZE, Image.Resampling.NEAREST)
        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((np.asarray(mask) > 0).copy()).float().unsqueeze(0)
        return image_tensor, mask_tensor


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def train(
    manifest: Path,
    repository_root: Path,
    checkpoint: Path,
    *,
    epochs: int = 20,
    batch_size: int = 8,
    seed: int = 20260811,
    device: str | None = None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_gold_dataset(manifest, repository_root)
    training = [item for item in dataset.annotations if item.accepted and item.split == "train"]
    declared_validation = [
        item for item in dataset.annotations if item.accepted and item.split == "validation"
    ]
    validation_source = "declared_validation_split" if declared_validation else "psd-5-1"
    train_items = training if declared_validation else [
        item for item in training if item.source_composition_id != validation_source
    ]
    validation_items = declared_validation or [
        item for item in training if item.source_composition_id == validation_source
    ]
    if not train_items or not validation_items:
        raise ValueError("source-disjoint compact segmenter train/validation split is empty")
    train_loader = DataLoader(
        GoldMaskDataset(train_items, repository_root), batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        GoldMaskDataset(validation_items, repository_root), batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CompactBinaryUNet().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {
        "dataset_version": dataset.version,
        "seed": seed,
        "device": resolved_device,
        "train_sources": sorted({item.source_composition_id for item in train_items}),
        "validation_sources": sorted({item.source_composition_id for item in validation_items}),
        "train_count": len(train_items),
        "validation_count": len(validation_items),
        "train_loss": [],
        "validation_iou": [],
    }
    best_iou = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(resolved_device), masks.to(resolved_device)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
        history["train_loss"].append(total_loss / len(train_items))
        model.eval()
        intersection = union = 0
        with torch.no_grad():
            for images, masks in validation_loader:
                predictions = torch.sigmoid(model(images.to(resolved_device))) >= .5
                truth = masks.to(resolved_device) > 0
                intersection += int((predictions & truth).sum().item())
                union += int((predictions | truth).sum().item())
        validation_iou = intersection / max(1, union)
        history["validation_iou"].append(validation_iou)
        if validation_iou > best_iou:
            best_iou = validation_iou
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss": history["train_loss"][-1],
            "validation_iou": validation_iou,
        }))
    if checkpoint.exists() or checkpoint.with_suffix(".history.json").exists():
        raise FileExistsError(f"immutable compact segmenter output already exists: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "model_family": "compact_gold_unet",
        "train_size": TRAIN_SIZE,
        "threshold": .5,
        "model_state": best_state,
        "history": history,
    }, checkpoint)
    checkpoint.with_suffix(".history.json").write_text(
        json.dumps(history, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("work/bhagavadgita/production/gold-v1/manifest.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("work/bhagavadgita/production/segmenter-competition/compact-gold-unet.pt"),
    )
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repository_root / args.manifest
    checkpoint = args.out if args.out.is_absolute() else repository_root / args.out
    train(
        manifest,
        repository_root,
        checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
