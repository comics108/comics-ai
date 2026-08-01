"""Resolve a layer's fully-settled ("resting") transform from its animations[] keyframes.

Ground truth verified directly against the editor's C# source (not guessed):
apps/comics-editor/native/Comics.Editor/Models/{Anim,TranslateAnim,ScaleAnim,RotateAnim,AlphaAnim,
PivotAnim}.cs.

This is a *scroll-driven* reveal-animation system: each Anim's Start/End is a scroll-position
range (not wall-clock time) -- the tall canvas (up to ~33000px) is read by scrolling down through
it, and elements animate into place as the reader's scroll position passes their own y-region, then
stay fixed. Anim.FindNearest (Anim.cs) resolves the value at any scroll position to either an
in-progress interpolation, or -- once scroll has passed every keyframe's End -- the raw target
field values of the *last* keyframe (ordered by Start). With no further keyframes ever added,
that terminal state is permanent: this is exactly what flows/sdd-comics-ai-multimodal/
02-specifications.md's "resting position" assumption needs (Checkpoint A: whether a printed page
reflects this per-layer settled state).

Per-type defaults when a layer has *no* keyframes of that type at all come from each class's own
Init() (used by FindNearest to synthesize a "before the first keyframe" value):
- TranslateAnim: no Init() override -> C# default int 0,0
- AlphaAnim.Init(): alpha = 1 (fully visible)
- ScaleAnim.Init(): scaleX = scaleY = 1 (no scaling)
- PivotAnim.Init() (base of Scale/Rotate): pivotX = pivotY = 0.5 (normalized center pivot)
- RotateAnim: no Init() override beyond PivotAnim's -> angle default 0

These Init() defaults are NOT the same thing as an omitted field *within* an existing keyframe
entry (which really does mean 0, e.g. a fade-in's first AlphaAnim keyframe omitting "alpha" means
alpha=0 at scroll-start, the "invisible" starting point of a fade-in) -- this module only ever
reads the *last* keyframe of each type (using its own explicit fields, defaulting missing fields to
each type's normal serialization default of 0/0.0), and only falls back to the Init() values when
that type's keyframe list is empty altogether.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RestingTransform:
    x: int = 0
    y: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_pivot_x: float = 0.5
    scale_pivot_y: float = 0.5
    angle: float = 0.0
    rotate_pivot_x: float = 0.5
    rotate_pivot_y: float = 0.5
    alpha: float = 1.0


def _short_type(anim: dict) -> str:
    # "$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor" -> "TranslateAnim"
    return anim.get("$type", "").split(",")[0].split(".")[-1]


def _group_by_type(animations: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for a in animations:
        groups.setdefault(_short_type(a), []).append(a)
    return groups


def _last_by_start(entries: list[dict]) -> dict | None:
    if not entries:
        return None
    # Mirrors Anim.FindNearest's own `.OrderBy(x => x.Start)` -- raw array order is not trusted.
    return sorted(entries, key=lambda a: a.get("start", 0))[-1]


def resolve_resting_transform(animations: list[dict]) -> RestingTransform:
    result = RestingTransform()
    groups = _group_by_type(animations)

    translate = _last_by_start(groups.get("TranslateAnim", []))
    if translate is not None:
        result.x = translate.get("x", 0)
        result.y = translate.get("y", 0)

    scale = _last_by_start(groups.get("ScaleAnim", []))
    if scale is not None:
        result.scale_x = scale.get("scaleX", 0.0)
        result.scale_y = scale.get("scaleY", 0.0)
        result.scale_pivot_x = scale.get("pivotX", 0.0)
        result.scale_pivot_y = scale.get("pivotY", 0.0)

    rotate = _last_by_start(groups.get("RotateAnim", []))
    if rotate is not None:
        result.angle = rotate.get("angle", 0.0)
        result.rotate_pivot_x = rotate.get("pivotX", 0.0)
        result.rotate_pivot_y = rotate.get("pivotY", 0.0)

    alpha = _last_by_start(groups.get("AlphaAnim", []))
    if alpha is not None:
        result.alpha = alpha.get("alpha", 0.0)

    return result
