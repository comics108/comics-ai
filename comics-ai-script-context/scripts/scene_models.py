#!/usr/bin/env python3
"""Task 1.1 (flows/sdd-comics-ai-script-context/03-plan.md): data types for structured
scene-extraction output. Pure data, no ollama/subprocess dependency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterMention:
    name: str
    action_or_state: str

    def to_dict(self) -> dict:
        return {"name": self.name, "action_or_state": self.action_or_state}

    @staticmethod
    def from_dict(d: dict) -> "CharacterMention":
        return CharacterMention(name=d["name"], action_or_state=d["action_or_state"])


@dataclass(frozen=True)
class SceneExtraction:
    episode_file: str
    source_excerpt: str
    characters: tuple[CharacterMention, ...]
    props: tuple[str, ...]
    locations: tuple[str, ...]
    raw_model_output: str
    model_name: str = "qwen2.5-coder:32b"
    # "spiritual_text" (hand-verified narrative prose, high trust) | "ocr_dialogue" (the episode's
    # own balloon dialogue, broad coverage, lower narrative context per Requirements v0.2's
    # criterion 2 -- both are real text, provenance is kept explicit, never merged silently).
    text_source: str = "spiritual_text"

    def to_dict(self) -> dict:
        return {
            "episode_file": self.episode_file,
            "source_excerpt": self.source_excerpt,
            "characters": [c.to_dict() for c in self.characters],
            "props": list(self.props),
            "locations": list(self.locations),
            "raw_model_output": self.raw_model_output,
            "model_name": self.model_name,
            "text_source": self.text_source,
        }

    @staticmethod
    def from_dict(d: dict) -> "SceneExtraction":
        return SceneExtraction(
            episode_file=d["episode_file"],
            source_excerpt=d["source_excerpt"],
            characters=tuple(CharacterMention.from_dict(c) for c in d["characters"]),
            props=tuple(d["props"]),
            locations=tuple(d["locations"]),
            raw_model_output=d["raw_model_output"],
            model_name=d.get("model_name", "qwen2.5-coder:32b"),
            text_source=d.get("text_source", "spiritual_text"),
        )
