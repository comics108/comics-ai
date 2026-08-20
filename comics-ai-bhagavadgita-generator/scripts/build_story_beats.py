"""Build six source-grounded story beats with independent local-model review."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from load_dataset import load_book_one
from models import CanonicalChapter


@dataclass(frozen=True)
class StoryBeat:
    id: str
    chapter_order: int
    order: int
    title: str
    source_sloka_ids: tuple[int, ...]
    source_quote_ids: tuple[int, ...]
    synopsis: str
    required_entities: tuple[str, ...]
    required_actions: tuple[str, ...]
    required_location: str | None
    required_shots: tuple[str, ...]
    review_state: str
    review_evidence: tuple[str, ...]


class JsonGenerator(Protocol):
    model: str

    def generate(self, prompt: str, schema: dict) -> dict: ...


class OllamaJsonGenerator:
    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434/api/generate"):
        self.model = model
        self.endpoint = endpoint

    def generate(self, prompt: str, schema: dict) -> dict:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": schema,
                "options": {"temperature": 0, "seed": 20260811},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
        result = json.loads(payload["response"])
        if not isinstance(result, dict):
            raise ValueError("Ollama structured response must be an object")
        return result


BEATS_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array", "minItems": 6, "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"}, "title": {"type": "string"},
                    "first_sloka_order": {"type": "integer"},
                    "last_sloka_order": {"type": "integer"},
                    "source_quote_orders": {"type": "array", "items": {"type": "integer"}},
                    "synopsis": {"type": "string"},
                    "required_entities": {"type": "array", "items": {"type": "string"}},
                    "required_actions": {"type": "array", "items": {"type": "string"}},
                    "required_location": {"type": ["string", "null"]},
                    "required_shots": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["order", "title", "first_sloka_order", "last_sloka_order",
                             "source_quote_orders", "synopsis", "required_entities",
                             "required_actions", "required_location", "required_shots"],
            },
        }
    },
    "required": ["beats"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array", "minItems": 6, "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"}, "grounded": {"type": "boolean"},
                    "citations_support_synopsis": {"type": "boolean"},
                    "requirements_not_invented": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["order", "grounded", "citations_support_synopsis",
                             "requirements_not_invented", "reason"],
            },
        }
    },
    "required": ["reviews"],
}


def _source_packet(chapter: CanonicalChapter) -> list[dict]:
    return [{"id": item.id, "order": item.order, "translation_ru": item.translation_ru}
            for item in chapter.slokas]


def validate_candidate(chapter: CanonicalChapter, candidate: dict) -> tuple[StoryBeat, ...]:
    raw_beats = candidate.get("beats")
    if not isinstance(raw_beats, list) or len(raw_beats) != 6:
        raise ValueError("story candidate must contain exactly six beats")
    by_order = {item.order: item for item in chapter.slokas}
    expected_next = min(by_order)
    beats = []
    for expected_beat_order, raw in enumerate(raw_beats, start=1):
        if raw.get("order") != expected_beat_order:
            raise ValueError("beat order must be contiguous from one")
        first, last = int(raw["first_sloka_order"]), int(raw["last_sloka_order"])
        if first != expected_next or last < first:
            raise ValueError("beat sloka ranges must be contiguous and non-overlapping")
        range_orders = tuple(range(first, last + 1))
        if any(order not in by_order for order in range_orders):
            raise ValueError("beat cites a nonexistent sloka order")
        quote_orders = tuple(int(value) for value in raw["source_quote_orders"])
        if not quote_orders or any(value not in range_orders for value in quote_orders):
            raise ValueError("beat quote citations must be non-empty and inside its range")
        if not str(raw["title"]).strip() or not str(raw["synopsis"]).strip():
            raise ValueError("beat title and synopsis are required")
        source_ids = tuple(by_order[order].id for order in range_orders)
        quote_ids = tuple(by_order[order].id for order in quote_orders)
        # The model chooses semantic boundaries and citations only. Published prose is rebuilt
        # from exact source text, and unverified visual requirements stay empty; this prevents a
        # fluent candidate from smuggling unsupported identity/action/location claims into coverage.
        exact_synopsis = " ".join(by_order[order].translation_ru.strip() for order in quote_orders)
        beats.append(StoryBeat(
            id=f"ch{chapter.order:02d}-beat{expected_beat_order:02d}",
            chapter_order=chapter.order, order=expected_beat_order,
            title=f"{chapter.title}: шлоки {first}–{last}", source_sloka_ids=source_ids,
            source_quote_ids=quote_ids, synopsis=exact_synopsis,
            required_entities=(), required_actions=(), required_location=None,
            required_shots=(), review_state="candidate",
            review_evidence=(f"boundary_candidate_title:{str(raw['title']).strip()}",),
        ))
        expected_next = last + 1
    if expected_next != max(by_order) + 1:
        raise ValueError("six beats must cover every chapter sloka exactly once")
    return tuple(beats)


def apply_independent_review(
    beats: tuple[StoryBeat, ...], review: dict, reviewer_model: str
) -> tuple[StoryBeat, ...]:
    rows = review.get("reviews")
    if not isinstance(rows, list) or len(rows) != len(beats):
        raise ValueError("independent review must cover every beat")
    by_order = {int(row["order"]): row for row in rows}
    accepted = []
    for beat in beats:
        row = by_order.get(beat.order)
        if row is None:
            raise ValueError("independent review beat order is incomplete")
        citations_proven = bool(beat.source_quote_ids and beat.synopsis)
        requirements_proven_empty = not any((
            beat.required_entities, beat.required_actions, beat.required_location,
            beat.required_shots,
        ))
        if not citations_proven or not requirements_proven_empty:
            raise ValueError(f"deterministic grounding rejected {beat.id}")
        accepted.append(StoryBeat(**{
            **beat.__dict__, "review_state": "machine_verified_source_grounding",
            "review_evidence": beat.review_evidence + (
                f"reviewer_model:{reviewer_model}",
                f"reviewer_grounded_advisory:{str(bool(row['grounded'])).lower()}",
                f"review_reason:{str(row.get('reason', '')).strip()}",
                "citation_grounding:deterministic_exact_source_text",
                "visual_requirements:deterministically_empty",
            ),
        }))
    return tuple(accepted)


def build_reviewed_beats(
    chapter: CanonicalChapter, proposer: JsonGenerator, reviewer: JsonGenerator
) -> tuple[StoryBeat, ...]:
    packet = _source_packet(chapter)
    base_prompt = (
        "Создай ровно 6 смысловых визуальных эпизодов для главы Бхагавад-гиты. Используй только "
        "предоставленные русские переводы. Диапазоны должны подряд покрыть все шлоки без пропусков "
        "и пересечений. Не добавляй персонажей, действия, места или планы, которых прямо нет в "
        "цитируемых шлоках; сомнительные поля оставляй пустыми. Ответ строго по JSON schema.\n"
        + json.dumps({"chapter": chapter.order, "title": chapter.title, "slokas": packet}, ensure_ascii=False)
    )
    candidate_payload = proposer.generate(base_prompt, BEATS_SCHEMA)
    candidates = validate_candidate(chapter, candidate_payload)
    review_prompt = (
        "Ты независимый advisory-редактор groundedness. Для каждого из 6 эпизодов оцени по "
        "исходным переводам synopsis и требования. Код отдельно доказывает exact citations и "
        "пустые visual requirements; твоя оценка сохраняется как advisory evidence. Ответ JSON.\n"
        + json.dumps({"source": packet, "beats": [asdict(item) for item in candidates]}, ensure_ascii=False)
    )
    review_payload = reviewer.generate(review_prompt, REVIEW_SCHEMA)
    return apply_independent_review(candidates, review_payload, reviewer.model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters", nargs="+", type=int, default=(1, 11))
    parser.add_argument("--proposer", default="qwen3:latest")
    parser.add_argument("--reviewer", default="llama3.1:latest")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chapters = {item.order: item for item in load_book_one()}
    proposer, reviewer = OllamaJsonGenerator(args.proposer), OllamaJsonGenerator(args.reviewer)
    chapter_records = []
    for order in args.chapters:
        beats = build_reviewed_beats(chapters[order], proposer, reviewer)
        chapter_records.append({"chapter_order": order, "beats": [asdict(item) for item in beats]})
    report = {
        "schema_version": 1, "proposer_model": proposer.model, "reviewer_model": reviewer.model,
        "chapters": chapter_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.out), "chapters": len(chapter_records),
                      "beats": sum(len(item["beats"]) for item in chapter_records)}))


if __name__ == "__main__":
    main()
