"""Native Lottie package recovery with translation and audio provenance."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecoveredLottieLayer:
    native_path: str
    index: int
    name: str
    layer_type: int
    parent_index: int | None
    reference_id: str | None
    in_frame: float
    out_frame: float
    start_frame: float
    transform: dict[str, Any]


@dataclass(frozen=True)
class RecoveredLottiePackage:
    package_root: Path
    content_file: Path
    width: int
    height: int
    frame_rate: float
    in_frame: float
    out_frame: float
    root_layer_count: int
    precomposition_count: int
    referenced_image_count: int
    layers: tuple[RecoveredLottieLayer, ...]
    translation_files: tuple[Path, ...]
    translation_image_counts: tuple[tuple[str, int], ...]
    audio_files: tuple[Path, ...]
    semantic_scope_id: str
    camera_depth_authority: str


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _native_layers(document: dict[str, Any]) -> tuple[RecoveredLottieLayer, ...]:
    recovered: list[RecoveredLottieLayer] = []

    def append_layers(container: str, layers: list[dict[str, Any]]) -> None:
        for ordinal, layer in enumerate(layers):
            index = int(layer.get("ind", ordinal))
            parent = layer.get("parent")
            recovered.append(RecoveredLottieLayer(
                native_path=f"{container}/{ordinal}",
                index=index,
                name=str(layer.get("nm") or f"layer-{index}"),
                layer_type=int(layer.get("ty", -1)),
                parent_index=int(parent) if isinstance(parent, (int, float)) else None,
                reference_id=str(layer["refId"]) if layer.get("refId") is not None else None,
                in_frame=float(layer.get("ip", document.get("ip", 0))),
                out_frame=float(layer.get("op", document.get("op", 0))),
                start_frame=float(layer.get("st", 0)),
                transform=dict(layer.get("ks") or {}),
            ))

    append_layers("root", list(document.get("layers") or ()))
    for asset in document.get("assets") or ():
        if isinstance(asset.get("layers"), list):
            append_layers(f"asset:{asset.get('id', 'unknown')}", asset["layers"])
    return tuple(recovered)


def recover_lottie_package(package_root: Path) -> RecoveredLottiePackage:
    root = package_root.resolve(strict=True)
    content_candidates = sorted(root.glob("*_content/*.json"))
    if len(content_candidates) != 1:
        raise ValueError(f"expected one Lottie content JSON below {root}, found {len(content_candidates)}")
    content_file = content_candidates[0]
    document = _load(content_file)
    assets = list(document.get("assets") or ())

    translations: list[tuple[str, int, Path]] = []
    for path in sorted(root.glob("*_translations/*.json")):
        match = re.search(r"_([a-z]{2,3})\.json$", path.name)
        language = match.group(1) if match else "und"
        translation = _load(path)
        image_count = sum(1 for asset in translation.get("assets") or () if asset.get("p"))
        translations.append((language, image_count, path.resolve()))

    audio_suffixes = {".aac", ".mp3", ".wav", ".ogg", ".m4a"}
    audio_files = tuple(sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in audio_suffixes),
        key=lambda path: path.as_posix(),
    ))
    return RecoveredLottiePackage(
        package_root=root,
        content_file=content_file.resolve(),
        width=int(document["w"]),
        height=int(document["h"]),
        frame_rate=float(document["fr"]),
        in_frame=float(document["ip"]),
        out_frame=float(document["op"]),
        root_layer_count=len(document.get("layers") or ()),
        precomposition_count=sum(1 for asset in assets if isinstance(asset.get("layers"), list)),
        referenced_image_count=sum(1 for asset in assets if asset.get("p")),
        layers=_native_layers(document),
        translation_files=tuple(item[2] for item in translations),
        translation_image_counts=tuple((item[0], item[1]) for item in translations),
        audio_files=audio_files,
        semantic_scope_id="scope-gita-dhyanam-nine-stanzas",
        camera_depth_authority="derived_evidence_not_gold",
    )
