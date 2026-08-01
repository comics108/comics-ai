"""Task 4.2: baseline semantic-segmentation model (Plan Option C, Task 4.1's decision) -- a small
U-Net predicting a per-pixel Kind label (art/background/character/balloon, per dataset.py's
KIND_TO_LABEL). Chosen as the cheap, fast-to-train first tier, to unblock Phase 5-9 integration
testing before the heavier Task 4.3 Mask R-CNN model. Does not natively separate overlapping
same-kind instances (connected-components on the predicted label map is the intended follow-up
for instance separation, per Task 4.2's plan description) -- a documented limitation, not a bug.

Ground truth is rectangle-only (no per-pixel masks anywhere in this pipeline -- see dataset.py) so
`rasterize_label_map` paints axis-aligned boxes, not precise shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBaseline(nn.Module):
    """A small 3-level U-Net. Deliberately compact (channels default 16/32/64) given only ~750
    real training samples -- a deeper/wider net would be more prone to overfitting this small a
    dataset than this baseline is meant to risk. Input H/W must be divisible by 4 (two 2x pooling
    steps) for the skip connections to align; `train_segmenter.py` fixes training resolution to
    256x256 for exactly this reason.
    """

    def __init__(self, num_classes: int = 4, channels: tuple[int, int, int] = (16, 32, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.enc1 = DoubleConv(3, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


def rasterize_label_map(
    boxes: torch.Tensor, labels: torch.Tensor, size: tuple[int, int], background_label: int = 0
) -> torch.Tensor:
    """Paint each (box, label) into a per-pixel label map of shape (H, W). `boxes`/`labels` must
    already be in ascending z-order (background first, later entries overwrite) -- augment.py
    sorts each training pair's regions by ascending layer_index before writing the manifest, which
    is the established bottom-to-top compositing order (Editor Schema Ground Truth), so iterating
    the manifest's own order here reproduces correct occlusion (a character/balloon painted over
    its background) without needing to re-sort.
    """
    h, w = size
    label_map = torch.full((h, w), background_label, dtype=torch.int64)
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = (int(round(v)) for v in box.tolist())
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            label_map[y0:y1, x0:x1] = int(label)
    return label_map
