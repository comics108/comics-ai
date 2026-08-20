"""Native PSD hierarchy recovery without flattening or eager pixel allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class RecoveredPsdNode:
    native_path: str
    parent_path: str | None
    name: str
    kind: str
    visible: bool
    blend_mode: str
    opacity: int
    bbox: tuple[int, int, int, int]
    child_count: int


@dataclass(frozen=True)
class RecoveredPsdDocument:
    source_path: Path
    width: int
    height: int
    color_mode: str
    depth: int
    nodes: tuple[RecoveredPsdNode, ...]
    text_group_names: tuple[str, ...]


@dataclass(frozen=True)
class RecoveredPsdLayer:
    native_path: str
    name: str
    bbox: tuple[int, int, int, int]
    rgba: Image.Image
    bitmap_mask: Image.Image


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _bbox_tuple(layer: Any) -> tuple[int, int, int, int]:
    bbox = layer.bbox
    if isinstance(bbox, tuple):
        return tuple(map(int, bbox))
    return int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)


def recover_psd_structure(path: Path) -> RecoveredPsdDocument:
    """Recover hierarchy and native layer facts; never composite the document."""
    from psd_tools import PSDImage

    source = path.resolve(strict=True)
    psd = PSDImage.open(source)
    nodes: list[RecoveredPsdNode] = []

    def walk(container: Any, parent_indices: tuple[int, ...] = ()) -> None:
        parent_path = "/".join(map(str, parent_indices)) or None
        for index, layer in enumerate(container):
            indices = (*parent_indices, index)
            native_path = "/".join(map(str, indices))
            is_group = bool(layer.is_group())
            nodes.append(RecoveredPsdNode(
                native_path=native_path,
                parent_path=parent_path,
                name=str(layer.name or f"unnamed-{native_path}"),
                kind=str(layer.kind),
                visible=bool(layer.visible),
                blend_mode=_enum_value(layer.blend_mode),
                opacity=int(layer.opacity),
                bbox=_bbox_tuple(layer),
                child_count=len(layer) if is_group else 0,
            ))
            if is_group:
                walk(layer, indices)

    walk(psd)
    text_groups = [node.name for node in nodes if node.kind == "group" and node.name.startswith("text")]
    text_groups.sort(key=lambda name: int(name[4:]) if name[4:].isdigit() else 10**9)
    return RecoveredPsdDocument(
        source_path=source,
        width=int(psd.width),
        height=int(psd.height),
        color_mode=_enum_value(psd.color_mode),
        depth=int(psd.depth),
        nodes=tuple(nodes),
        text_group_names=tuple(text_groups),
    )


def recover_psd_layer(path: Path, native_path: str) -> RecoveredPsdLayer:
    """Lazily recover one selected native layer and its true alpha bitmap mask."""
    from psd_tools import PSDImage

    parts = native_path.split("/")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid PSD native path: {native_path!r}")
    layer: Any = PSDImage.open(path.resolve(strict=True))
    for part in parts:
        index = int(part)
        if index >= len(layer):
            raise KeyError(f"PSD native path does not exist: {native_path}")
        layer = layer[index]
    if layer.is_group():
        raise ValueError("group recovery must be explicitly flattened; select a pixel/type layer")
    pixels = layer.topil()
    if pixels is None:
        raise ValueError(f"PSD layer has no recoverable pixels: {native_path}")
    rgba = pixels.convert("RGBA")
    mask = rgba.getchannel("A").copy()
    return RecoveredPsdLayer(
        native_path=native_path,
        name=str(layer.name or f"unnamed-{native_path}"),
        bbox=_bbox_tuple(layer),
        rgba=rgba,
        bitmap_mask=mask,
    )
