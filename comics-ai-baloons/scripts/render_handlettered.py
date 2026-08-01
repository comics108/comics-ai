#!/usr/bin/env python3
"""Track 6b: hand-lettered balloons -- flag for manual review, do not auto-render.

Decision made at Checkpoint B (flows/sdd-comics-ai-baloons/04-implementation-log.md), confirmed
with the user: the real hand-lettered count in this dataset is 2 of 825 balloons (one genuine
hand-drawn sound-effect panel, one non-text false positive), and both already fail CSV matching --
so there is neither enough data to train a dedicated model nor any actual balloon that would use
it in this run. Rather than build unused machinery, this stage is intentionally a flag-only no-op:
every hand-lettered balloon gets an explicit "not rendered, needs manual review" result per
language, which flows into the report (stage 7.2) so these balloons stay visible rather than
silently disappearing.

If a future dataset/run turns up meaningfully more hand-lettered examples, revisit this decision
rather than extending this stub in place.
"""

from __future__ import annotations

from models import RenderResult

FLAG_REASON = (
    "hand_lettered: flagged for manual artist review, no automated render "
    "(Checkpoint B decision -- insufficient training data and no matched balloon needs it)"
)


def handle_hand_lettered_balloon(
    source_file: str, layer_index: int, target_langs: list[str]
) -> list[RenderResult]:
    return [
        RenderResult(
            source_file=source_file,
            layer_index=layer_index,
            lang_code=lang,
            rendered=False,
            image_path=None,
            reason=FLAG_REASON,
        )
        for lang in target_langs
    ]
