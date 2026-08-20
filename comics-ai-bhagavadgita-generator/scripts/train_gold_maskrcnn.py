#!/usr/bin/env python3
"""Train Mask R-CNN on true Gold bitmap masks with source-explicit semantic kinds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from gold_segmenter_data import load_gold_dataset, load_review_pair


KINDS = ("art", "animal", "character", "fx")
KIND_TO_LABEL = {kind: index + 1 for index, kind in enumerate(KINDS)}


class GoldInstanceDataset:
    def __init__(self, annotations, repository_root: Path):
        self.annotations = tuple(annotations)
        self.repository_root = repository_root

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        import torch

        annotation = self.annotations[index]
        image, mask_image = load_review_pair(annotation, self.repository_root)
        mask = np.asarray(mask_image) > 0
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError(f"empty Gold mask: {annotation.id}")
        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        mask_tensor = torch.from_numpy(mask.copy()).to(torch.uint8).unsqueeze(0)
        box = torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)
        label = torch.tensor([KIND_TO_LABEL[annotation.semantic_kind]], dtype=torch.int64)
        return image_tensor, {
            "boxes": box, "labels": label, "masks": mask_tensor,
            "area": torch.tensor([int(mask.sum())], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64), "image_id": torch.tensor([index]),
        }


def collate(batch):
    return tuple(zip(*batch))


def build_model(pretrained: bool = True):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=256, max_size=768)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(KINDS) + 1)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, len(KINDS) + 1)
    return model


def train(manifest: Path, repository_root: Path, output: Path, *, epochs=3, batch_size=1, seed=20260812, device=None, resume: Path | None = None, balanced=False):
    import torch
    from torch.utils.data import DataLoader

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gold = load_gold_dataset(manifest, repository_root)
    training = [item for item in gold.annotations if item.accepted and item.split == "train" and item.semantic_kind in KIND_TO_LABEL]
    validation = [item for item in gold.annotations if item.accepted and item.split == "validation" and item.semantic_kind in KIND_TO_LABEL]
    if not training or not validation:
        raise ValueError("declared source-disjoint train/validation sets are required")
    resolved = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    sampler = None
    if balanced:
        counts = {kind: sum(item.semantic_kind == kind for item in training) for kind in KINDS}
        weights = [1 / counts[item.semantic_kind] for item in training]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(training), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    train_loader = DataLoader(
        GoldInstanceDataset(training, repository_root), batch_size=batch_size,
        shuffle=sampler is None, sampler=sampler, collate_fn=collate,
    )
    model = build_model(pretrained=True).to(resolved)
    previous_history = None
    if resume is not None:
        payload = torch.load(resume, map_location=resolved, weights_only=False)
        if payload.get("model_family") != "gold_true_mask_maskrcnn_v1":
            raise ValueError("resume checkpoint family mismatch")
        model.load_state_dict(payload["model_state"])
        previous_history = payload["history"]
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.0025, momentum=.9, weight_decay=.0005)
    history = {"dataset_version": gold.version, "device": resolved, "seed": seed,
               "train_sources": sorted({x.source_composition_id for x in training}),
               "validation_sources": sorted({x.source_composition_id for x in validation}),
               "train_count": len(training), "validation_count": len(validation),
               "train_loss": list((previous_history or {}).get("train_loss", [])),
               "balanced_sampler": balanced}
    for epoch in range(epochs):
        model.train(); total = 0
        for images, targets in train_loader:
            images = [image.to(resolved) for image in images]
            targets = [{key: value.to(resolved) for key, value in target.items()} for target in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss.item()) * len(images)
        history["train_loss"].append(total / len(training))
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema_version": 1, "model_family": "gold_true_mask_maskrcnn_v1", "kinds": KINDS,
                    "model_state": model.state_dict(), "history": history}, output)
        output.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"epoch": epoch + 1, "train_loss": history["train_loss"][-1]}), flush=True)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args(); root = args.repository_root.resolve()
    train(args.manifest if args.manifest.is_absolute() else root / args.manifest, root,
          args.out if args.out.is_absolute() else root / args.out,
          epochs=args.epochs, batch_size=args.batch_size, device=args.device,
          resume=None if args.resume is None else (args.resume if args.resume.is_absolute() else root / args.resume),
          balanced=args.balanced)


if __name__ == "__main__":
    main()
