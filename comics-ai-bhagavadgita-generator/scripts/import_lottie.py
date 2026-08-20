#!/usr/bin/env python3
"""Import the real Bhagavad Gita Lottie into canonical `.comics` camera/depth data.

The source has no Lottie camera (ddd=0, no ty=13). Its three root precomps are vertical sweeps;
the perceived camera curve is reconstructed from the richest animated image layer in each scene.
No external Lottie renderer is used.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from package_comics import PackagingAsset

TRANSLATE = "Comics.Editor.Models.TranslateAnim, Comics.Editor"
SCALE = "Comics.Editor.Models.ScaleAnim, Comics.Editor"
ROTATE = "Comics.Editor.Models.RotateAnim, Comics.Editor"


@dataclass(frozen=True)
class PointSample:
    frame: float
    position: int
    x: float
    y: float
    scale_x: float
    scale_y: float
    angle: float


@dataclass(frozen=True)
class ImportedLottie:
    width: int
    height: int
    assets: tuple[PackagingAsset, ...]
    camera_path: tuple[dict, ...]
    scene_count: int
    image_layer_count: int
    animated_layer_count: int
    reference_layers: tuple[str, ...]


def load_lottie(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _numbers(value: Any, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        return tuple(float(v) for v in value)
    return fallback


def _keyframes(prop: dict | None) -> list[dict]:
    if not isinstance(prop, dict) or prop.get("a") != 1 or not isinstance(prop.get("k"), list):
        return []
    return [k for k in prop["k"] if isinstance(k, dict) and isinstance(k.get("t"), (int, float))]


def _property_frames(prop: dict | None) -> list[float]:
    return [float(k["t"]) for k in _keyframes(prop)]


def _property_value(prop: dict | None, frame: float, fallback: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(prop, dict):
        return fallback
    keys = _keyframes(prop)
    if not keys:
        return _numbers(prop.get("k"), fallback)
    if frame <= float(keys[0]["t"]):
        return _numbers(keys[0].get("s"), fallback)
    if frame >= float(keys[-1]["t"]):
        return _numbers(keys[-1].get("s"), fallback)
    for index, left in enumerate(keys[:-1]):
        right = keys[index + 1]
        start, end = float(left["t"]), float(right["t"])
        if start <= frame <= end:
            a = _numbers(left.get("s"), fallback)
            b = _numbers(right.get("s"), a)
            if left.get("h") == 1 or end == start:
                return a
            t = (frame - start) / (end - start)
            size = max(len(a), len(b))
            return tuple(
                (a[i] if i < len(a) else a[-1])
                + ((b[i] if i < len(b) else b[-1]) - (a[i] if i < len(a) else a[-1])) * t
                for i in range(size)
            )
    return fallback


# Affine tuple (a,b,c,d,tx,ty): x'=a*x+c*y+tx, y'=b*x+d*y+ty.
def _mul(m1: tuple[float, ...], m2: tuple[float, ...]) -> tuple[float, ...]:
    a, b, c, d, tx, ty = m1
    e, f, g, h, ux, uy = m2
    return (
        a * e + c * f,
        b * e + d * f,
        a * g + c * h,
        b * g + d * h,
        a * ux + c * uy + tx,
        b * ux + d * uy + ty,
    )


def _transform(layer: dict, frame: float) -> tuple[float, ...]:
    ks = layer.get("ks") or {}
    anchor = _property_value(ks.get("a"), frame, (0.0, 0.0))
    position = _property_value(ks.get("p"), frame, (0.0, 0.0))
    scale = _property_value(ks.get("s"), frame, (100.0, 100.0))
    rotation = _property_value(ks.get("r"), frame, (0.0,))[0]
    ax, ay = anchor[0], anchor[1] if len(anchor) > 1 else 0.0
    px, py = position[0], position[1] if len(position) > 1 else 0.0
    sx, sy = scale[0] / 100.0, (scale[1] if len(scale) > 1 else scale[0]) / 100.0
    radians = math.radians(rotation)
    cos, sin = math.cos(radians), math.sin(radians)
    return (cos * sx, sin * sx, -sin * sy, cos * sy, px - cos * sx * ax + sin * sy * ay,
            py - sin * sx * ax - cos * sy * ay)


def _world_matrix(layer: dict, layers_by_index: dict[int, dict], frame: float) -> tuple[float, ...]:
    chain: list[dict] = []
    seen: set[int] = set()
    current: dict | None = layer
    while current is not None:
        index = int(current.get("ind", -1))
        if index in seen:
            raise ValueError(f"Lottie parent cycle at layer index {index}")
        seen.add(index)
        chain.append(current)
        parent = current.get("parent")
        current = layers_by_index.get(int(parent)) if isinstance(parent, (int, float)) else None
    result = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for item in reversed(chain):
        result = _mul(result, _transform(item, frame))
    return result


def _animation_frames(layer: dict, layers_by_index: dict[int, dict]) -> list[float]:
    frames: set[float] = set()
    seen: set[int] = set()
    current: dict | None = layer
    while current is not None:
        index = int(current.get("ind", -1))
        if index in seen:
            break
        seen.add(index)
        ks = current.get("ks") or {}
        for key in ("p", "s", "r"):
            frames.update(_property_frames(ks.get(key)))
        parent = current.get("parent")
        current = layers_by_index.get(int(parent)) if isinstance(parent, (int, float)) else None
    return sorted(frames)


def _root_frame(root: dict, local_frame: float) -> float:
    return float(root.get("st", 0)) + local_frame * float(root.get("sr", 1) or 1)


def _scene_scroll(root: dict, local_frame: float) -> float:
    position = root["ks"]["p"]
    first_frame = float(root.get("ip", root.get("st", 0)))
    start_y = _property_value(position, first_frame, (0.0, 0.0))[1]
    current_y = _property_value(position, _root_frame(root, local_frame), (0.0, 0.0))[1]
    return start_y - current_y


def _sample_layer(
    layer: dict,
    layers_by_index: dict[int, dict],
    root: dict,
    local_frame: float,
    scene_offset: int,
) -> PointSample:
    local = _world_matrix(layer, layers_by_index, local_frame)
    root_matrix = _transform(root, _root_frame(root, local_frame))
    world = _mul(root_matrix, local)
    scroll = _scene_scroll(root, local_frame)
    document_position = int(round(scene_offset + scroll))
    scale_x = math.hypot(world[0], world[1])
    scale_y = math.hypot(world[2], world[3])
    angle = math.degrees(math.atan2(world[1], world[0]))
    return PointSample(
        frame=local_frame,
        position=document_position,
        x=world[4],
        y=world[5] + scene_offset + scroll,
        scale_x=scale_x,
        scale_y=scale_y,
        angle=angle,
    )


def _samples_for_layer(layer: dict, layers_by_index: dict[int, dict], root: dict,
                       scene_offset: int) -> list[PointSample]:
    frames = _animation_frames(layer, layers_by_index)
    if not frames:
        frames = [0.0]
    samples = [_sample_layer(layer, layers_by_index, root, frame, scene_offset) for frame in frames]
    by_position: dict[int, PointSample] = {sample.position: sample for sample in samples}
    return [by_position[p] for p in sorted(by_position)]


def select_camera_reference_layer(layers: list[dict]) -> dict | None:
    candidates = []
    for layer in layers:
        if layer.get("ty") != 2:
            continue
        ks = layer.get("ks") or {}
        position_keys = _keyframes(ks.get("p"))
        if len(position_keys) < 2:
            continue
        values = [_numbers(k.get("s"), (0.0, 0.0)) for k in position_keys]
        displacement = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(values, values[1:])
        )
        candidates.append((bool(_keyframes(ks.get("s"))), len(position_keys), displacement, layer))
    return max(candidates, key=lambda item: item[:3])[3] if candidates else None


def build_camera_path(samples: list[PointSample]) -> list[dict]:
    return [
        {"position": sample.position, "x": sample.x, "y": sample.y}
        for sample in samples
    ]


def _distance(samples: list[PointSample]) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(samples, samples[1:]))


def derive_z_depth(samples: list[PointSample], camera_samples: list[PointSample],
                   is_camera_reference: bool) -> float:
    if is_camera_reference or len(samples) < 2:
        return 0.0
    initial_scale = (abs(samples[0].scale_x) + abs(samples[0].scale_y)) / 2
    final_scale = (abs(samples[-1].scale_x) + abs(samples[-1].scale_y)) / 2
    if initial_scale > 1e-9 and not math.isclose(initial_scale, final_scale, rel_tol=1e-6):
        growth = final_scale / initial_scale
        value = 1 / growth - 1 if growth > 0 else 0.0
    else:
        camera_amplitude = _distance(camera_samples)
        layer_amplitude = _distance(samples)
        if camera_amplitude <= 1e-9 or layer_amplitude <= 1e-9:
            return 0.0
        ratio = layer_amplitude / camera_amplitude
        value = 1 / ratio - 1
    return round(value, 3) if math.isfinite(value) and value > -1 else 0.0


def _seeded_anims(samples: list[PointSample]) -> tuple[dict, ...]:
    if not samples:
        return ()
    anims: list[dict] = [{"$type": TRANSLATE, "start": samples[0].position,
                          "end": samples[0].position, "x": round(samples[0].x, 3),
                          "y": round(samples[0].y, 3)}]
    for previous, sample in zip(samples, samples[1:]):
        anims.append({"$type": TRANSLATE, "start": previous.position, "end": sample.position,
                      "x": round(sample.x, 3), "y": round(sample.y, 3)})
    if any(not math.isclose(s.scale_x, 1) or not math.isclose(s.scale_y, 1) for s in samples):
        for index, sample in enumerate(samples):
            start = samples[index - 1].position if index else sample.position
            anims.append({"$type": SCALE, "start": start, "end": sample.position,
                          "scaleX": round(sample.scale_x, 6), "scaleY": round(sample.scale_y, 6),
                          "pivotX": 0.0, "pivotY": 0.0})
    if any(not math.isclose(s.angle, 0, abs_tol=1e-6) for s in samples):
        for index, sample in enumerate(samples):
            start = samples[index - 1].position if index else sample.position
            anims.append({"$type": ROTATE, "start": start, "end": sample.position,
                          "angle": round(sample.angle, 3), "pivotX": 0.0, "pivotY": 0.0})
    return tuple(anims)


def _decode_image(asset: dict) -> Image.Image:
    payload = asset.get("p", "")
    if not isinstance(payload, str) or not payload.startswith("data:") or "," not in payload:
        raise ValueError(f"image asset {asset.get('id')!r} is not embedded")
    raw = base64.b64decode(payload.split(",", 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    image.load()
    return image


def import_lottie_document(document: dict) -> ImportedLottie:
    assets_by_id = {asset.get("id"): asset for asset in document.get("assets", [])}
    image_assets = {key: value for key, value in assets_by_id.items() if value.get("p")}
    roots = sorted(
        (layer for layer in document.get("layers", []) if layer.get("ty") == 0 and layer.get("refId")),
        key=lambda layer: float(layer.get("ip", layer.get("st", 0))),
    )
    packaged: list[PackagingAsset] = []
    camera_path: list[dict] = []
    references: list[str] = []
    scene_offset = 0
    animated_count = 0

    for scene_index, root in enumerate(roots):
        scene = assets_by_id.get(root.get("refId"))
        if not scene or not isinstance(scene.get("layers"), list):
            continue
        layers = scene["layers"]
        layers_by_index = {int(layer["ind"]): layer for layer in layers if "ind" in layer}
        reference = select_camera_reference_layer(layers)
        if reference is None:
            continue
        references.append(f"{root.get('nm')}/{reference.get('nm')} (ind={reference.get('ind')})")
        reference_samples = _samples_for_layer(reference, layers_by_index, root, scene_offset)
        scene_camera = build_camera_path(reference_samples)
        if camera_path and scene_camera:
            # Preserve the scene's relative curve while making the document-level path continuous.
            dx = camera_path[-1]["x"] - scene_camera[0]["x"]
            dy = camera_path[-1]["y"] - scene_camera[0]["y"]
            scene_camera = [{**p, "x": p["x"] + dx, "y": p["y"] + dy} for p in scene_camera]
        for point in scene_camera:
            if camera_path and point["position"] <= camera_path[-1]["position"]:
                continue
            camera_path.append(point)

        for layer_index, layer in enumerate(layers):
            image_asset = image_assets.get(layer.get("refId"))
            if layer.get("ty") != 2 or image_asset is None:
                continue
            samples = _samples_for_layer(layer, layers_by_index, root, scene_offset)
            if len(samples) > 1:
                animated_count += 1
            depth = derive_z_depth(samples, reference_samples, layer is reference)
            packaged.append(PackagingAsset(
                kind="art",
                image=_decode_image(image_asset),
                x=round(samples[0].x),
                y=round(samples[0].y),
                stem=f"lottie_{scene_index + 1}_{layer_index:03d}",
                contains_russian_text=False,
                animations=_seeded_anims(samples),
                z_depth=depth,
            ))

        root_position = root.get("ks", {}).get("p")
        root_keys = _keyframes(root_position)
        if root_keys:
            start_y = _numbers(root_keys[0].get("s"), (0.0, 0.0))[1]
            end_y = _numbers(root_keys[-1].get("s"), (0.0, 0.0))[1]
            scene_offset += int(round(abs(start_y - end_y)))

    return ImportedLottie(
        width=int(document.get("w", 720)),
        height=scene_offset + int(document.get("h", 1600)),
        assets=tuple(packaged),
        camera_path=tuple(camera_path),
        scene_count=len(roots),
        image_layer_count=len(packaged),
        animated_layer_count=animated_count,
        reference_layers=tuple(references),
    )


def import_lottie_file(path: Path) -> ImportedLottie:
    return import_lottie_document(load_lottie(path))
