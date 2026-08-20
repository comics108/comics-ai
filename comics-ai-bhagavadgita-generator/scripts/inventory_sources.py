#!/usr/bin/env python3
"""Read-only source inventory and reviewed semantic-scope classification."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from production_models import SourceKind, SourceRecord, SourceSemanticScope


_UNCLASSIFIED = SourceSemanticScope(
    id="scope-unclassified",
    work="unclassified",
    scope="unclassified",
    chapter_orders=(),
    verse_ranges=(),
    mapping_state="unmapped",
    evidence=(),
    reviewer=None,
)

_GITA_DHYANAM = SourceSemanticScope(
    id="scope-gita-dhyanam-nine-stanzas",
    work="gita_dhyanam",
    scope="standalone_prologue",
    chapter_orders=(),
    verse_ranges=(),
    mapping_state="confirmed",
    evidence=(
        "reviewed RU/EN overlays contain the complete traditional nine-stanza Gita Dhyanam",
        "directory 1 and layer S3_B1_C1 are production identifiers, not canonical chapter numbers",
    ),
    reviewer="anton",
)

_CHAPTER_FIVE = SourceSemanticScope(
    id="scope-bhagavad-gita-05-14-29",
    work="bhagavad_gita",
    scope="canonical_verse_range",
    chapter_orders=(5,),
    verse_ranges=((5, 14, 29),),
    mapping_state="confirmed",
    evidence=("reviewed 15 sequential PSD balloon/caption groups cover verses 5.14-5.29",),
    reviewer="anton",
)


def _chapter_five_component(name: str) -> SourceSemanticScope:
    return SourceSemanticScope(
        id=f"scope-bhagavad-gita-ch05-component-{name.removesuffix('.psd')}",
        work="bhagavad_gita",
        scope="source_component",
        chapter_orders=(),
        verse_ranges=(),
        mapping_state="not_applicable",
        evidence=(f"reviewed {name} as a production component reproduced inside app_BG._chiba5.psd",),
        reviewer="anton",
    )


@dataclass(frozen=True)
class ScopeRule:
    path: str
    scope: SourceSemanticScope
    prefix: bool = False


class SourceScopeRegistry:
    """Resolve only explicitly reviewed paths; never infer semantics from numbers."""

    def __init__(self, rules: tuple[ScopeRule, ...]) -> None:
        self.rules = rules

    def resolve(self, relative_path: str) -> SourceSemanticScope:
        normalized = relative_path.replace("\\", "/").lstrip("./")
        for rule in self.rules:
            if normalized == rule.path or (rule.prefix and normalized.startswith(f"{rule.path}/")):
                return rule.scope
        return _UNCLASSIFIED

    def scopes(self) -> tuple[SourceSemanticScope, ...]:
        by_id = {_UNCLASSIFIED.id: _UNCLASSIFIED}
        by_id.update((rule.scope.id, rule.scope) for rule in self.rules)
        return tuple(by_id[key] for key in sorted(by_id))


def default_scope_registry() -> SourceScopeRegistry:
    return SourceScopeRegistry((
        ScopeRule(
            "vaishnav/bhagavadgita_lottie/unzip/1",
            _GITA_DHYANAM,
            prefix=True,
        ),
        ScopeRule(
            "vaishnav/bhagavadgita/lottie_unzip/Mediation of the Bhagavat Gita",
            _GITA_DHYANAM,
            prefix=True,
        ),
        ScopeRule("vaishnav/bhagavadgita/lottie/Mediation of the Bhagavat Gita.lottie.zip", _GITA_DHYANAM),
        ScopeRule("vaishnav/drawing/app_BG._chiba5.psd", _CHAPTER_FIVE),
        ScopeRule("vaishnav/drawing/5_1.psd", _chapter_five_component("5_1.psd")),
        ScopeRule("vaishnav/drawing/5_2.psd", _chapter_five_component("5_2.psd")),
    ))


def canonical_chapters(scope: SourceSemanticScope) -> tuple[int, ...]:
    """Return canonical coverage only for confirmed Bhagavad Gita chapter material."""
    if scope.work != "bhagavad_gita" or scope.mapping_state != "confirmed":
        return ()
    if scope.scope not in {"canonical_chapter", "canonical_verse_range"}:
        return ()
    return scope.chapter_orders


_KIND_BY_SUFFIX: dict[str, SourceKind] = {
    ".csv": "structured_text",
    ".json": "structured_text",
    ".txt": "structured_text",
    ".md": "editorial_note",
    ".psd": "psd",
    ".pdf": "pdf",
    ".png": "raster",
    ".jpg": "raster",
    ".jpeg": "raster",
    ".webp": "raster",
    ".lottie": "lottie",
    ".mp3": "audio",
    ".wav": "audio",
    ".ogg": "audio",
    ".comics": "comics",
    ".ttf": "font",
    ".otf": "font",
}

_MEDIA_BY_SUFFIX = {
    ".psd": "image/vnd.adobe.photoshop",
    ".lottie": "application/zip+dotlottie",
    ".comics": "application/vnd.nativemind.comics+zip",
    ".csv": "text/csv",
    ".json": "application/json",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_kind(path: Path) -> SourceKind:
    name = path.name.lower()
    if name.endswith(".lottie.zip"):
        return "lottie"
    return _KIND_BY_SUFFIX.get(path.suffix.lower(), "editorial_note")


def _media_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".lottie.zip"):
        return "application/zip+dotlottie"
    suffix = path.suffix.lower()
    return _MEDIA_BY_SUFFIX.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def inventory_source_root(
    source_root: Path,
    *,
    registry: SourceScopeRegistry | None = None,
) -> tuple[SourceRecord, ...]:
    """Hash every regular file below ``source_root`` without writing to it."""
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    registry = registry or default_scope_registry()
    records: list[SourceRecord] = []
    paths = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"source inventory refuses symbolic-link files: {path}")
        relative_path = path.relative_to(root).as_posix()
        content_sha256 = _sha256_file(path)
        stable_id = hashlib.sha256(f"{relative_path}\0{content_sha256}".encode()).hexdigest()[:24]
        scope = registry.resolve(relative_path)
        records.append(SourceRecord(
            id=f"source-{stable_id}",
            kind=_source_kind(path),
            relative_path=relative_path,
            sha256=content_sha256,
            byte_size=path.stat().st_size,
            media_type=_media_type(path),
            metadata={"suffix": path.suffix.lower()},
            semantic_scope_id=scope.id,
        ))
    return tuple(records)


def write_inventory_json(
    output_path: Path,
    records: tuple[SourceRecord, ...],
    registry: SourceScopeRegistry,
    *,
    source_roots: tuple[Path, ...],
) -> Path:
    """Validate and atomically publish the current inventory outside source roots."""
    output = output_path.resolve()
    for source_root in source_roots:
        resolved_source = source_root.resolve()
        if output == resolved_source or output.is_relative_to(resolved_source):
            raise PermissionError(
                f"inventory output is inside read-only source root {resolved_source}: {output}"
            )
    document = {
        "schema_version": 1,
        "sources": [asdict(record) for record in records],
        "semantic_scopes": [asdict(scope) for scope in registry.scopes()],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output.parent / ".staging"
    staging_root.mkdir(exist_ok=True)
    stage = staging_root / f"inventory-{uuid.uuid4().hex}.json"
    try:
        stage.write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        validated = json.loads(stage.read_text(encoding="utf-8"))
        if validated.get("schema_version") != 1 or len(validated.get("sources", ())) != len(records):
            raise ValueError("staged inventory failed validation")
        os.replace(stage, output)
        return output
    finally:
        stage.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a read-only Bhagavad Gita source inventory")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = default_scope_registry()
    records = inventory_source_root(args.source_root, registry=registry)
    write_inventory_json(args.output, records, registry, source_roots=(args.source_root,))
    classified = sum(record.semantic_scope_id != _UNCLASSIFIED.id for record in records)
    print(f"inventoried {len(records)} sources ({classified} semantically classified): {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
