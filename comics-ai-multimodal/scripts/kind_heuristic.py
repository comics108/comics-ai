"""Legacy (untagged) layer -> Kind inference heuristic.

flows/sdd-comics-ai-multimodal/02-specifications.md "CanvasReference": nearly every real layer in
the dataset has no explicit `Kind` (that field was only ever populated for balloons, going forward,
by the lettering flow -- confirmed empirically: 0 of 4594 real layers currently carry a `kind`/
`Kind` field). This module's inferred kind is used as `GroundTruthRegion.kind_source =
"inferred_heuristic"`, distinct from an eventual explicit `kind_source = "explicit"` -- it is a
deliberate, honestly-labeled approximation (see Specifications' edge case table), not a claim of
ground truth.

Rule, in order:
1. **balloon**: >=2 populated `images[]` slots -- reuses `comics-ai-baloons`'s own structural rule
   verbatim via `baloons_bridge.is_balloon_layer` (single source of truth, not a second copy).
2. **background**: a single-slot layer whose image spans most of the canvas width, is tall enough
   to be full-bleed art, AND sits at the bottom of its *local* stack -- "local" because layer_index
   is global across the whole tall (~33000px) scrolling canvas, which repeats
   background/character/balloon clusters many times down the page; "bottom of stack" is therefore
   evaluated only against other layers whose resting y-position is nearby (`y_window`), using the
   already-verified compositing order (data.json layer array order == bottom-to-top).
3. **character**: a single-slot layer with a portrait-ish aspect ratio (taller than wide) within a
   plausible character-crop size range (not a tiny icon, not a full-canvas background).
4. **art**: fallback for anything not matched above.
"""

from __future__ import annotations

import baloons_bridge
from resting_position import resolve_resting_transform


def _populated_image(layer: dict) -> dict | None:
    for im in layer.get("images", []):
        if im.get("file"):
            return im
    return None


def infer_kinds_for_file(
    data: dict,
    *,
    y_window: int = 1500,
    background_min_width_ratio: float = 0.9,
    background_min_height: int = 600,
    character_min_width: int = 150,
    character_max_width: int = 950,
    character_min_height: int = 150,
    character_max_height: int = 3200,
) -> list[str]:
    """Infer a `kind` per layer in `data` (a parsed data.json dict). Returns a list parallel to
    `data["layers"]`.
    """
    layers = data["layers"]
    canvas_width = data.get("width", 0)
    resting_y = [resolve_resting_transform(l.get("animations", [])).y for l in layers]

    kinds: list[str] = []
    for i, layer in enumerate(layers):
        if baloons_bridge.is_balloon_layer(layer):
            kinds.append("balloon")
            continue

        image = _populated_image(layer)
        if image is None:
            kinds.append("art")  # no image slot at all -- shouldn't normally happen, be safe
            continue

        width, height = image.get("width", 0), image.get("height", 0)
        if not width or not height:
            kinds.append("art")
            continue

        this_y = resting_y[i]
        neighborhood = [j for j in range(len(layers)) if abs(resting_y[j] - this_y) <= y_window]
        is_bottom_of_local_stack = i == min(neighborhood)

        width_ratio = (width / canvas_width) if canvas_width else 0.0
        if (
            width_ratio >= background_min_width_ratio
            and height >= background_min_height
            and is_bottom_of_local_stack
        ):
            kinds.append("background")
            continue

        is_portrait = height > width * 1.15
        in_character_size_range = (
            character_min_width <= width <= character_max_width
            and character_min_height <= height <= character_max_height
        )
        if is_portrait and in_character_size_range:
            kinds.append("character")
            continue

        kinds.append("art")

    return kinds
