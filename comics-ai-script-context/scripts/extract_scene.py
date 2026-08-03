#!/usr/bin/env python3
"""Tasks 1.2-1.3 (flows/sdd-comics-ai-script-context/03-plan.md): prompt building, model-output
parsing, and the live ollama call. Task 1.2's parsing path is deliberately separable from Task
1.3's subprocess call so it can be unit-tested against canned strings, no live model needed."""

from __future__ import annotations

import json
import subprocess

from scene_models import CharacterMention, SceneExtraction

DEFAULT_MODEL = "qwen2.5-coder:32b"
OLLAMA_TIMEOUT_SECONDS = 180

PROMPT_TEMPLATE = """You are extracting structured scene information from a narrative excerpt for a comic-book production pipeline.

Given the TEXT below, respond with ONLY a valid JSON object (no other text, no markdown fences) with this exact shape:

{{
  "characters": [{{"name": "...", "action_or_state": "..."}}],
  "props": ["..."],
  "locations": ["..."]
}}

Rules:
- "characters": every named person mentioned, with a short (3-8 word) description of what they are doing or feeling in this excerpt.
- "props": notable physical objects mentioned (empty list if none).
- "locations": notable places mentioned (empty list if none).
- Do not invent characters, props, or locations not in the text.

TEXT:
"{excerpt}"
"""


class ExtractionFailed(Exception):
    def __init__(self, episode_file: str, reason: str, raw_output: str | None):
        self.episode_file = episode_file
        self.reason = reason
        self.raw_output = raw_output
        super().__init__(f"{episode_file}: {reason}")


def build_prompt(excerpt: str) -> str:
    return PROMPT_TEMPLATE.format(excerpt=excerpt)


def _extract_first_json_object(text: str) -> str | None:
    """Brace-matching fallback for models that wrap JSON in prose or markdown fences."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_model_output(
    raw_output: str,
    episode_file: str,
    excerpt: str,
    model_name: str = DEFAULT_MODEL,
    text_source: str = "spiritual_text",
) -> SceneExtraction:
    parsed = None
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        candidate = _extract_first_json_object(raw_output)
        if candidate is not None:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        raise ExtractionFailed(episode_file, "malformed JSON", raw_output)

    for required_key in ("characters", "props", "locations"):
        if required_key not in parsed:
            raise ExtractionFailed(episode_file, f"missing required field '{required_key}'", raw_output)

    seen_names: set[str] = set()
    characters: list[CharacterMention] = []
    for entry in parsed["characters"]:
        name = entry.get("name", "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        characters.append(
            CharacterMention(name=name, action_or_state=entry.get("action_or_state", "").strip())
        )

    return SceneExtraction(
        episode_file=episode_file,
        source_excerpt=excerpt,
        characters=tuple(characters),
        props=tuple(parsed["props"]),
        locations=tuple(parsed["locations"]),
        raw_model_output=raw_output,
        model_name=model_name,
        text_source=text_source,
    )


def extract(
    excerpt: str,
    episode_file: str,
    model: str = DEFAULT_MODEL,
    text_source: str = "spiritual_text",
) -> SceneExtraction:
    prompt = build_prompt(excerpt)
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise ExtractionFailed(episode_file, "ollama call timed out", None) from e
    except FileNotFoundError as e:
        raise ExtractionFailed(episode_file, "ollama not found on PATH", None) from e

    if result.returncode != 0:
        raise ExtractionFailed(episode_file, f"ollama exited {result.returncode}", result.stderr)

    return parse_model_output(
        result.stdout, episode_file, excerpt, model_name=model, text_source=text_source
    )
