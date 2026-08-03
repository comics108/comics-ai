#!/usr/bin/env python3
"""Criterion 1: rule-based transformation baseline, calibrated from real per-kind occurrence
rates and real per-property duration/value statistics (`transform_stats.py`) -- same "real
calibration, not guessed constants" precedent as `sdd-comics-ai-positioning`'s
`baseline_position.py`.

Proposes, per region kind and property (translate/scale/rotate/alpha), whether it reveals at all
(occurrence rate > OCCURRENCE_THRESHOLD) and, if so, its (start, end, from, to) -- using the
confident real conventions found for alpha (fade-in 0->1) and scale (grow-in 0.6->1.0). Translate
and rotate get a real, calibrated occurrence + duration, but an honestly-disclosed zero-delta value
(no confident direction/magnitude exists in the real data -- see `transform_stats.py`'s docstring)
rather than a fabricated default.
"""

from __future__ import annotations

from dataclasses import dataclass

OCCURRENCE_THRESHOLD = 0.5


@dataclass
class ProposedReveal:
    occurs: bool
    start: int = 0
    end: int = 0
    from_value: dict | None = None
    to_value: dict | None = None


def propose_property(
    kind: str, prop: str, stats: dict
) -> ProposedReveal:
    rate = stats["occurrence_rate"].get(kind, {}).get(prop, 0.0)
    if rate <= OCCURRENCE_THRESHOLD:
        return ProposedReveal(occurs=False)

    duration = stats["median_duration"][prop]
    if prop == "alpha":
        from_v, to_v = stats["alpha_from_to"]
        return ProposedReveal(
            occurs=True, start=0, end=round(duration), from_value={"alpha": from_v}, to_value={"alpha": to_v}
        )
    if prop == "scale":
        from_v, to_v = stats["scale_from_to"]
        return ProposedReveal(
            occurs=True,
            start=0,
            end=round(duration),
            from_value={"scaleX": from_v, "scaleY": from_v},
            to_value={"scaleX": to_v, "scaleY": to_v},
        )
    # translate/rotate: occurrence + duration are real and calibrated; direction/magnitude is not
    # confidently predictable from real data (see transform_stats.py docstring) -- disclosed zero
    # delta rather than a fabricated default.
    return ProposedReveal(occurs=True, start=0, end=round(duration), from_value={}, to_value={})


def propose_reveal(kind: str, stats: dict) -> dict[str, ProposedReveal]:
    return {prop: propose_property(kind, prop, stats) for prop in ("translate", "scale", "rotate", "alpha")}
