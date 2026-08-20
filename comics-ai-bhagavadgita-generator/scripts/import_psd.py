#!/usr/bin/env python3
"""Task 4.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): read-only
compositing of the three real chapter-5 PSD sources via `psd-tools`, per
02-specifications.md's "Chapter-5 PSD adapter" section. Never blocks the 18-chapter Must-Have
run: any failure (package missing, decode error, out-of-memory) is caught and turned into a
warning string, and the caller falls back to the deterministic baseline for that chapter.

Real, measured cost of compositing the three actual files this session (`/opt/homebrew/bin/
python3.14`, this app's venv): `5_1.psd` (9449x7087, 1 layer) ~2.2s / ~2.9GB peak RSS;
`5_2.psd` (9977x8101, 1 layer) ~2.4s / ~3.6GB peak RSS; `app_BG._chiba5.psd` (4127x26421,
33 layers) ~6.9s / ~4.8GB peak RSS. All three composited successfully on this machine -- real
evidence the "excessive memory" failure mode is a genuine possibility on smaller machines
(hence the graceful-fallback path), not a hypothetical one avoided here by luck.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class PsdImportResult:
    path: Path
    image: Image.Image | None
    warning: str | None


def resize_to_content_width(img: Image.Image, content_width: int) -> Image.Image:
    """Resizes preserving aspect ratio; a no-op if already at content_width."""
    if img.width == content_width:
        return img
    scale = content_width / img.width
    new_height = max(1, round(img.height * scale))
    return img.resize((content_width, new_height), Image.LANCZOS)


def import_psd_panel(path: Path, content_width: int) -> PsdImportResult:
    """Attempts a real composite of `path` via psd-tools, resized to `content_width`. On any
    failure -- psd-tools not installed, a decode error, or anything else -- returns a warning
    instead of raising, so the pipeline can continue on the deterministic baseline."""
    try:
        from psd_tools import PSDImage

        psd = PSDImage.open(path)
        composited = psd.composite()
        if composited is None:
            return PsdImportResult(
                path=path, image=None, warning=f"PSD composite returned no pixels for {path.name}"
            )
        image = resize_to_content_width(composited.convert("RGBA"), content_width)
        return PsdImportResult(path=path, image=image, warning=None)
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure degrades gracefully
        return PsdImportResult(
            path=path, image=None, warning=f"PSD import failed for {path.name}: {exc!r}"
        )
