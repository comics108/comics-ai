"""Read immutable Gold v1 and reconstruct the exact review image for each mask."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from build_gold_dataset import GoldAnnotation, GoldDataset, verify_gold_artifacts


def load_gold_dataset(manifest: Path, repository_root: Path) -> GoldDataset:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Gold manifest schema")
    dataset = GoldDataset(
        version=str(payload["dataset_version"]),
        annotations=tuple(GoldAnnotation(**item) for item in payload["annotations"]),
    )
    verify_gold_artifacts(dataset, repository_root)
    return dataset


def _resolved(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repository_root / path


def _evidence_value(annotation: GoldAnnotation, prefix: str) -> str:
    values = [item[len(prefix):] for item in annotation.review_evidence if item.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"{annotation.id} requires exactly one {prefix} evidence item")
    return values[0]


def load_review_pair(
    annotation: GoldAnnotation,
    repository_root: Path,
) -> tuple[Image.Image, Image.Image]:
    """Return RGB review input and L target mask at the annotation's documented resolution."""
    mask_path = _resolved(annotation.bitmap_mask_file, repository_root)
    mask = Image.open(mask_path).convert("L")
    if annotation.source_kind == "psd":
        suffix = ".mask.png"
        if not mask_path.name.endswith(suffix):
            raise ValueError(f"cannot resolve PSD RGBA evidence for {annotation.id}")
        rgba_path = mask_path.with_name(mask_path.name.removesuffix(suffix) + ".rgba.png")
        rgba = Image.open(rgba_path).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        image = background.convert("RGB")
    else:
        page = annotation.source_composition_id.rsplit("-", 1)[-1]
        rendered_page = (
            repository_root
            / "work/bhagavadgita/production/gold-v1/panorama-source"
            / f"bw-page-{page}.jpg"
        )
        x0, y0, x1, y1 = (
            int(value) for value in _evidence_value(annotation, "render_bbox:").split(",")
        )
        with Image.open(rendered_page) as panorama:
            image = panorama.convert("RGB").crop((x0, y0, x1, y1))
    if image.size != mask.size or image.size != tuple(annotation.review_resolution):
        raise ValueError(f"review image/mask geometry mismatch for {annotation.id}")
    return image, mask

