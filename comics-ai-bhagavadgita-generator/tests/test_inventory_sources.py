import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from inventory_sources import (
    canonical_chapters,
    default_scope_registry,
    inventory_source_root,
    write_inventory_json,
)


def test_gita_dhyanam_package_numbers_never_become_canonical_chapter_one():
    registry = default_scope_registry()
    scope = registry.resolve(
        "vaishnav/bhagavadgita_lottie/unzip/1/"
        "Mediation of the Bhagavat Gita_content/S3_B1_C1.json"
    )

    assert scope.work == "gita_dhyanam"
    assert scope.scope == "standalone_prologue"
    assert canonical_chapters(scope) == ()


def test_chapter_five_psd_and_its_numbered_components_have_distinct_scopes():
    registry = default_scope_registry()
    chapter_psd = registry.resolve("vaishnav/drawing/app_BG._chiba5.psd")
    component_1 = registry.resolve("vaishnav/drawing/5_1.psd")
    component_2 = registry.resolve("vaishnav/drawing/5_2.psd")

    assert canonical_chapters(chapter_psd) == (5,)
    assert chapter_psd.verse_ranges == ((5, 14, 29),)
    assert component_1.scope == component_2.scope == "source_component"
    assert component_1.verse_ranges == component_2.verse_ranges == ()
    assert canonical_chapters(component_1) == canonical_chapters(component_2) == ()


def test_unknown_numbered_filename_stays_unclassified():
    scope = default_scope_registry().resolve("incoming/chapter_11_S3_B1_C1.png")

    assert scope.work == "unclassified"
    assert scope.mapping_state == "unmapped"
    assert canonical_chapters(scope) == ()


def test_inventory_is_deterministic_and_does_not_modify_source_root(tmp_path: Path):
    source_root = tmp_path / "dataset" / "bhagavadgita"
    (source_root / "incoming").mkdir(parents=True)
    known = source_root / "vaishnav" / "drawing" / "app_BG._chiba5.psd"
    known.parent.mkdir(parents=True)
    known.write_bytes(b"psd-source")
    unknown = source_root / "incoming" / "chapter_11.png"
    unknown.write_bytes(b"numbered-but-unreviewed")
    before = {path.relative_to(source_root): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}

    first = inventory_source_root(source_root)
    second = inventory_source_root(source_root)

    after = {path.relative_to(source_root): path.read_bytes() for path in source_root.rglob("*") if path.is_file()}
    assert first == second
    assert [record.relative_path for record in first] == [
        "incoming/chapter_11.png",
        "vaishnav/drawing/app_BG._chiba5.psd",
    ]
    assert first[0].semantic_scope_id == "scope-unclassified"
    assert first[1].semantic_scope_id == "scope-bhagavad-gita-05-14-29"
    assert before == after


def test_inventory_json_is_atomic_and_contains_reviewed_scopes(tmp_path: Path):
    source_root = tmp_path / "dataset"
    source_root.mkdir()
    (source_root / "note.txt").write_text("source", encoding="utf-8")
    records = inventory_source_root(source_root)
    output = tmp_path / "work" / "production" / "inventory.json"

    write_inventory_json(output, records, default_scope_registry(), source_roots=(source_root,))

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["sources"][0]["relative_path"] == "note.txt"
    scopes = {scope["id"]: scope for scope in document["semantic_scopes"]}
    assert scopes["scope-gita-dhyanam-nine-stanzas"]["scope"] == "standalone_prologue"
    assert scopes["scope-bhagavad-gita-05-14-29"]["verse_ranges"] == [[5, 14, 29]]
