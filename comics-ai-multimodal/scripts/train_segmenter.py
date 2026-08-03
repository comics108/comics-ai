#!/usr/bin/env python3
"""Task 4.2/4.3 entrypoint: train a segmentation model (baseline U-Net, and later Mask R-CNN) on
augment.py's synthetic training pairs (work/train_pairs/manifest.jsonl).

Uses Apple Silicon MPS if available (torch.backends.mps), else CPU -- no CUDA in this environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import KIND_TO_LABEL, TrainingPairDataset
from segmenter_models.unet_baseline import UNetBaseline, rasterize_label_map

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_CHECKPOINT = WORK_DIR / "models" / "unet_baseline.pt"

TRAIN_SIZE = (256, 256)  # (H, W); must be divisible by 4 for UNetBaseline's two pooling steps
NUM_CLASSES = len(KIND_TO_LABEL)


def resize_sample(
    image: torch.Tensor, target: dict, size: tuple[int, int] = TRAIN_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize one (image, target) pair to a fixed training resolution, scaling boxes to match, then
    rasterize the per-pixel label map at that resolution. Fixed-size resizing (rather than
    variable-size per-sample training) is what makes efficient batched training possible here.
    """
    _, h, w = image.shape
    th, tw = size
    resized = F.interpolate(
        image.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0)

    scale_x, scale_y = tw / w, th / h
    boxes = target["boxes"].clone()
    if len(boxes):
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
    label_map = rasterize_label_map(boxes, target["labels"], size)
    return resized, label_map


def unet_collate(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, torch.Tensor]:
    images, label_maps = [], []
    for image, target in batch:
        img, lm = resize_sample(image, target)
        images.append(img)
        label_maps.append(lm)
    return torch.stack(images), torch.stack(label_maps)


def compute_class_weights(
    dataset: TrainingPairDataset,
    indices: list[int],
    size: tuple[int, int] = TRAIN_SIZE,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Inverse-frequency class weights for the training-loss cross-entropy, computed from real
    pixel counts in the (resized) rasterized label maps -- not guessed. A first training run
    without this (equal-weighted cross-entropy) collapsed to predicting the dominant class
    ("art", the default/background label for any pixel not covered by a region box) everywhere:
    val IoU came out {'art': 0.65, 'background': 0.0, 'character': 0.0, 'balloon': 0.10}. `art`
    dominates pixel count heavily (most of any crop's area isn't inside a tagged region box), so
    plain cross-entropy has little incentive to get the minority classes right at all.
    """
    # Verified on the real dataset (2026-07-31, 10 epochs, batch_size=8): unweighted val IoU was
    # {'art': 0.654, 'background': 0.0, 'character': 0.0, 'balloon': 0.100} -- collapsed to the
    # dominant class. With these weights, val IoU became {'art': 0.353, 'background': 0.023,
    # 'character': 0.129, 'balloon': 0.401} -- a real trade-off (art dropped as expected from
    # de-weighting it) but a much more balanced, actually-discriminating baseline. `background`
    # remains weak (0.023) even after weighting -- worth revisiting (more epochs? a higher weight
    # specifically for background? investigate whether background regions are often clipped at
    # crop edges) but not blocking for a first baseline tier.
    th, tw = size
    counts = torch.zeros(num_classes)
    for idx in indices:
        _, target = dataset[idx]
        w, h = target["crop_size"]
        boxes = target["boxes"].clone()
        if len(boxes):
            boxes[:, [0, 2]] *= tw / w
            boxes[:, [1, 3]] *= th / h
        label_map = rasterize_label_map(boxes, target["labels"], size)
        for c in range(num_classes):
            counts[c] += (label_map == c).sum().item()

    freq = counts / counts.sum().clamp(min=1)
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes  # normalize so mean weight ~= 1
    return weights


def compute_iou(
    model: UNetBaseline, loader: DataLoader, device: str, num_classes: int = NUM_CLASSES
) -> dict[str, float]:
    from dataset import LABEL_TO_KIND

    model.eval()
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    with torch.no_grad():
        for images, label_maps in loader:
            images, label_maps = images.to(device), label_maps.to(device)
            preds = model(images).argmax(dim=1)
            for c in range(num_classes):
                pred_c = preds == c
                true_c = label_maps == c
                intersection[c] += (pred_c & true_c).sum().item()
                union[c] += (pred_c | true_c).sum().item()
    result = {}
    for c in range(num_classes):
        result[LABEL_TO_KIND[c]] = (intersection[c] / union[c]).item() if union[c] > 0 else float("nan")
    return result


def train(
    manifest_path: Path | None = None,
    checkpoint_out: Path = DEFAULT_CHECKPOINT,
    epochs: int = 8,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    full_dataset = TrainingPairDataset(manifest_path) if manifest_path else TrainingPairDataset()

    generator = torch.Generator().manual_seed(seed)
    n_val = max(1, int(len(full_dataset) * val_fraction))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=unet_collate
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=unet_collate)

    class_weights = compute_class_weights(full_dataset, list(train_set.indices)).to(device)
    print("class weights (inverse pixel-frequency):", class_weights.tolist())

    model = UNetBaseline(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict = {"train_loss": [], "val_loss": [], "class_weights": class_weights.tolist()}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, label_maps in train_loader:
            images, label_maps = images.to(device), label_maps.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, label_maps, weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        train_loss = total_loss / len(train_set)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for images, label_maps in val_loader:
                images, label_maps = images.to(device), label_maps.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, label_maps)
                val_loss_total += loss.item() * images.size(0)
        val_loss = val_loss_total / len(val_set)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    history["val_iou"] = compute_iou(model, val_loader, device)
    print("val per-class IoU:", history["val_iou"])

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_out)
    history_path = checkpoint_out.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2))
    return history


DEFAULT_MASKRCNN_CHECKPOINT = WORK_DIR / "models" / "maskrcnn.pt"


def maskrcnn_collate(batch: list[tuple[torch.Tensor, dict]]):
    from segmenter_models.maskrcnn import to_detection_target

    images, targets = [], []
    for image, target in batch:
        images.append(image)
        targets.append(to_detection_target(target, target["crop_size"]))
    return images, targets


def compute_maskrcnn_iou(model, loader, device: str) -> dict[str, float]:
    """Per-class IoU comparable to the U-Net baseline's `compute_iou`: rasterize both ground truth
    and predictions into a flat per-pixel label map per image (predictions painted in ascending
    score order so the highest-confidence instance wins overlapping pixels) and accumulate
    intersection/union -- reuses `rasterize_label_map` so the metric definition matches Task 4.2's,
    making the Checkpoint D comparison apples-to-apples.
    """
    from dataset import LABEL_TO_KIND
    from segmenter_models.maskrcnn import NUM_DETECTION_CLASSES
    from segmenter_models.unet_baseline import rasterize_label_map

    model.eval()
    intersection = torch.zeros(NUM_DETECTION_CLASSES)
    union = torch.zeros(NUM_DETECTION_CLASSES)
    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(device) for img in images]
            preds = model(images_dev)
            for image, target, pred in zip(images, targets, preds):
                _, h, w = image.shape
                gt_map = rasterize_label_map(target["boxes"], target["labels"], (h, w))

                scores = pred["scores"].cpu()
                order = torch.argsort(scores)  # ascending -- highest score painted last, wins ties
                pred_boxes = pred["boxes"].cpu()[order]
                pred_labels = pred["labels"].cpu()[order]
                pred_map = rasterize_label_map(pred_boxes, pred_labels, (h, w))

                for c in range(NUM_DETECTION_CLASSES):
                    pred_c = pred_map == c
                    true_c = gt_map == c
                    intersection[c] += (pred_c & true_c).sum().item()
                    union[c] += (pred_c | true_c).sum().item()

    result = {}
    for c in range(NUM_DETECTION_CLASSES):
        name = "no_object_reserved" if c == 0 else LABEL_TO_KIND[c - 1]
        result[name] = (intersection[c] / union[c]).item() if union[c] > 0 else float("nan")
    return result


def train_maskrcnn(
    manifest_path: Path | None = None,
    checkpoint_out: Path = DEFAULT_MASKRCNN_CHECKPOINT,
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 0.005,
    val_fraction: float = 0.15,
    seed: int = 0,
    device: str | None = None,
    max_samples: int | None = None,
) -> dict:
    """Fine-tune Mask R-CNN. Defaults are deliberately modest (2 epochs, batch_size=2) given this
    is a much heavier model than the U-Net baseline -- see Plan Task 4.3's "High" complexity
    rating and Checkpoint D's purpose (compare against the baseline before committing to a longer
    training budget), not a claim that 2 epochs is enough for a production-quality model.
    """
    from segmenter_models.maskrcnn import build_model

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    full_dataset = TrainingPairDataset(manifest_path) if manifest_path else TrainingPairDataset()

    indices = list(range(len(full_dataset)))
    if max_samples is not None and max_samples < len(indices):
        rng_local = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(indices), generator=rng_local)[:max_samples].tolist()
        indices = sorted(perm)
    subset = torch.utils.data.Subset(full_dataset, indices)

    generator = torch.Generator().manual_seed(seed)
    n_val = max(1, int(len(subset) * val_fraction))
    n_train = len(subset) - n_val
    train_set, val_set = random_split(subset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=maskrcnn_collate
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=maskrcnn_collate)

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    history: dict = {"train_loss": [], "n_train": len(train_set), "n_val": len(val_set)}
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            n += len(images)
        train_loss = total_loss / n
        history["train_loss"].append(train_loss)
        print(f"[maskrcnn] epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}")

        # Long MPS/CPU runs can span hours; checkpoint after every epoch so an interruption
        # doesn't lose all training progress, not just at the very end.
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_out)
        checkpoint_out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))

    history["val_iou"] = compute_maskrcnn_iou(model, val_loader, device)
    print("[maskrcnn] val per-class IoU:", history["val_iou"])

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "history": history}, checkpoint_out)
    checkpoint_out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["unet", "maskrcnn"], default="unet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Force 'cpu' or 'mps'; defaults to auto-detect (mps if available, else cpu). Mask "
        "R-CNN's first MPS call on a given machine pays a one-time Metal shader compilation cost "
        "(observed: several minutes, looks like a hang -- 0%% CPU, uninterruptible-sleep state, no "
        "progress) before settling into normal per-batch timing; this is a one-time-per-machine "
        "cost; pass --device cpu to opt back into the slower but compilation-free path.",
    )
    args = parser.parse_args()

    if args.model == "unet":
        train(
            epochs=args.epochs or 8,
            batch_size=args.batch_size or 8,
            lr=args.lr or 1e-3,
            checkpoint_out=args.out or DEFAULT_CHECKPOINT,
            device=args.device,
        )
    else:
        train_maskrcnn(
            epochs=args.epochs or 2,
            batch_size=args.batch_size or 2,
            lr=args.lr or 0.005,
            checkpoint_out=args.out or DEFAULT_MASKRCNN_CHECKPOINT,
            max_samples=args.max_samples,
            device=args.device,
        )


if __name__ == "__main__":
    main()
