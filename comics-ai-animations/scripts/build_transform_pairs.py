#!/usr/bin/env python3
"""Criterion 1 (flows/sdd-comics-ai-transformations/01-requirements.md v0.3): extract real
(region kind -> RevealAnimation) ground-truth pairs from all 27 real `.comics` files.

Unlike `sdd-comics-ai-positioning` (which needs a matched photo to have a *predicted* region to
pair against ground truth), this flow's Must-Have only needs the ground truth itself -- "does this
kind of layer typically animate, and how" is a property of the 27 real authored files, not of the
(much smaller) matched-photo subset. All 27 files contribute training signal here, not just the 19
with matched photos.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import transforms_bridge as tb

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_OUT = WORK_DIR / "transform_pairs.jsonl"


@dataclass
class TransformPair:
    episode_stem: str
    layer_index: int
    kind: str
    kind_source: str  # "explicit" | "inferred_heuristic" -- same provenance tag comics-multimodal uses
    reveal: dict  # RevealAnimation.to_dict()-shaped (see build_pairs_for_file)


def _reveal_to_dict(reveal: "tb.RevealAnimation") -> dict:
    def pr(p):
        return None if p is None else asdict(p)

    return {
        "translate": pr(reveal.translate),
        "scale": pr(reveal.scale),
        "rotate": pr(reveal.rotate),
        "alpha": pr(reveal.alpha),
    }


def build_pairs_for_file(path: Path) -> list[TransformPair]:
    archive = tb.ComicsArchive(path)
    data = archive.read_data_json()
    kinds = infer_kinds(data)
    stem = path.stem

    pairs = []
    for i, layer in enumerate(data["layers"]):
        explicit_kind = layer.get("kind") or layer.get("Kind")
        kind = explicit_kind if explicit_kind else kinds[i]
        kind_source = "explicit" if explicit_kind else "inferred_heuristic"
        reveal = tb.resolve_reveal_animation(layer.get("animations", []))
        pairs.append(
            TransformPair(
                episode_stem=stem,
                layer_index=i,
                kind=kind,
                kind_source=kind_source,
                reveal=_reveal_to_dict(reveal),
            )
        )
    return pairs


def infer_kinds(data: dict) -> list[str]:
    from kind_heuristic import infer_kinds_for_file

    return infer_kinds_for_file(data)


def build_all(out_path: Path = DEFAULT_OUT) -> dict[str, int]:
    counts: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for path in tb.find_comics_files():
            pairs = build_pairs_for_file(path)
            counts[path.stem] = len(pairs)
            for pair in pairs:
                f.write(json.dumps(asdict(pair)) + "\n")
    return counts


def main() -> None:
    counts = build_all()
    total = sum(counts.values())
    print(f"Built {total} real transform pairs across {len(counts)} episodes -> {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
