import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from production_store import ImmutableVersionStore


def test_published_version_is_immutable(tmp_path: Path):
    store = ImmutableVersionStore(tmp_path / "production")

    published = store.create_version(
        namespace="assets",
        object_id="asset-krishna",
        version=1,
        files={"rgba.png": b"first-version"},
        metadata={"review_state": "proposed"},
    )

    with pytest.raises(FileExistsError, match="immutable version already exists"):
        store.create_version(
            namespace="assets",
            object_id="asset-krishna",
            version=1,
            files={"rgba.png": b"replacement"},
            metadata={"review_state": "accepted"},
        )

    assert published == tmp_path / "production" / "assets" / "asset-krishna" / "1"
    assert (published / "rgba.png").read_bytes() == b"first-version"


def test_write_inside_source_root_is_forbidden_before_staging(tmp_path: Path):
    source_root = tmp_path / "dataset" / "bhagavadgita"
    source_root.mkdir(parents=True)
    original = source_root / "source.psd"
    original.write_bytes(b"original")
    store = ImmutableVersionStore(
        source_root / "production",
        source_roots=(source_root,),
    )

    with pytest.raises(PermissionError, match="read-only source root"):
        store.create_version(
            namespace="assets",
            object_id="asset-arjuna",
            version=1,
            files={"rgba.png": b"derived"},
            metadata={},
        )

    assert original.read_bytes() == b"original"
    assert not (source_root / "production").exists()


def test_validation_failure_never_publishes_and_cleans_staging(tmp_path: Path):
    root = tmp_path / "production"
    store = ImmutableVersionStore(root)

    def reject(_stage: Path) -> None:
        raise ValueError("invalid bitmap mask")

    with pytest.raises(ValueError, match="invalid bitmap mask"):
        store.create_version(
            namespace="assets",
            object_id="asset-invalid",
            version=1,
            files={"mask.png": b"not-a-mask"},
            metadata={},
            validator=reject,
        )

    assert not (root / "assets" / "asset-invalid" / "1").exists()
    assert list((root / ".staging").iterdir()) == []
