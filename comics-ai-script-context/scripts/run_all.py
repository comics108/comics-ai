#!/usr/bin/env python3
"""Tasks 2.1-2.2 (flows/sdd-comics-ai-script-context/03-plan.md): full-coverage run across all
27 real episode files, writing one JSON per successfully-extracted episode plus an honest
report.md with three status categories -- extracted / failed / no-source-text -- never silently
collapsing "we had no text" into "we tried and got nothing"."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "comics-positioning" / "scripts")
)

from extract_scene import ExtractionFailed, extract
from ocr_dialogue_source import build_excerpt_for_episode, load_ocr_entries
from scene_models import SceneExtraction
from text_context import VERIFIED

DATASET_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_interactive"
)
WORK_DIR = Path(__file__).resolve().parent.parent / "work"
SCENES_DIR = WORK_DIR / "scenes"

PLACEHOLDER_NAME_SUBSTRINGS = ("heroine", "the speaker", "protagonist", "hero ")


def all_episode_files() -> list[str]:
    return sorted(p.name for p in DATASET_DIR.glob("*.comics"))


def has_placeholder_name(extraction: SceneExtraction) -> bool:
    for c in extraction.characters:
        lowered = c.name.lower()
        if any(sub in lowered for sub in PLACEHOLDER_NAME_SUBSTRINGS):
            return True
    return False


def run_all() -> dict:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    ocr_entries = load_ocr_entries()

    extracted: list[SceneExtraction] = []
    failed: list[tuple[str, str]] = []  # (episode_file, reason)
    no_source_text: list[str] = []

    for episode_file in all_episode_files():
        verified = VERIFIED.get(episode_file)
        if verified is not None:
            excerpt, text_source = verified.excerpt, "spiritual_text"
        else:
            ocr_excerpt = build_excerpt_for_episode(episode_file, ocr_entries)
            if ocr_excerpt is None:
                no_source_text.append(episode_file)
                continue
            excerpt, text_source = ocr_excerpt, "ocr_dialogue"

        try:
            result = extract(excerpt, episode_file, text_source=text_source)
        except ExtractionFailed as e:
            failed.append((episode_file, e.reason))
            continue

        extracted.append(result)
        (SCENES_DIR / f"{episode_file}.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        )

    write_report(extracted, failed, no_source_text)
    return {
        "extracted": extracted,
        "failed": failed,
        "no_source_text": no_source_text,
    }


def write_report(
    extracted: list[SceneExtraction],
    failed: list[tuple[str, str]],
    no_source_text: list[str],
) -> None:
    total = len(extracted) + len(failed) + len(no_source_text)
    lines = [
        "# comics-ai-script-context: coverage report",
        "",
        f"Total episodes: {total}",
        f"- Extracted: {len(extracted)}",
        f"- Failed: {len(failed)}",
        f"- No source text: {len(no_source_text)}",
        "",
        "## Extracted",
        "",
    ]
    for e in extracted:
        names = ", ".join(c.name for c in e.characters) or "(none found)"
        flag = " **[PLACEHOLDER NAME FLAGGED]**" if has_placeholder_name(e) else ""
        flag += " **[ZERO CHARACTERS]**" if not e.characters else ""
        lines.append(f"- `{e.episode_file}` ({e.text_source}): {names}{flag}")

    lines += ["", "## Failed", ""]
    for episode_file, reason in failed:
        lines.append(f"- `{episode_file}`: {reason}")

    lines += ["", "## No source text", ""]
    for episode_file in no_source_text:
        lines.append(f"- `{episode_file}`")

    (WORK_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    summary = run_all()
    print(
        f"Extracted: {len(summary['extracted'])}, "
        f"Failed: {len(summary['failed'])}, "
        f"No source text: {len(summary['no_source_text'])}"
    )
