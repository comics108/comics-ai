import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
WORK_DIR = Path(__file__).resolve().parents[1] / "work" / "test_tmp"


def _pick_sample_file() -> Path:
    files = sorted(DATASET_DIR.glob("*.comics"))
    assert files, "expected at least one .comics file in dataset/"
    return files[0]


def test_round_trip_content_fidelity():
    """Every entry's decompressed content survives a full read + rewrite unchanged.

    Note: this checks content fidelity per zip entry, not raw container-byte equality --
    the output archive uses ZIP_DEFLATED regardless of the source's original per-entry
    compression method, so the container bytes legitimately differ while every entry's
    content is identical. Content fidelity is the invariant that actually matters (any
    conformant zip reader gets the same bytes back).
    """
    src_path = _pick_sample_file()
    src = comics_io.ComicsArchive(src_path)
    original_entries = src.read_all()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = WORK_DIR / f"roundtrip_{src_path.name}"
    comics_io.write_comics(dest_path, src)

    out = comics_io.ComicsArchive(dest_path)
    out_entries = out.read_all()

    assert set(out_entries.keys()) == set(original_entries.keys())
    for name, content in original_entries.items():
        assert out_entries[name] == content, f"content mismatch for {name}"

    dest_path.unlink()


def test_data_json_override_and_extra_files():
    src_path = _pick_sample_file()
    src = comics_io.ComicsArchive(src_path)
    data = src.read_data_json()
    data["_test_marker"] = True

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = WORK_DIR / f"override_{src_path.name}"
    comics_io.write_comics(
        dest_path,
        src,
        data_json_override=data,
        extra_files={"layers/_test_extra.png": b"not really a png"},
    )

    out = comics_io.ComicsArchive(dest_path)
    out_data = out.read_data_json()
    assert out_data["_test_marker"] is True
    assert out.read_bytes("layers/_test_extra.png") == b"not really a png"

    # source untouched
    assert "_test_marker" not in src.read_data_json()

    dest_path.unlink()


def test_source_never_opened_for_write():
    """dataset/ files must never be modified -- sanity check mtime/size are unchanged."""
    src_path = _pick_sample_file()
    before = src_path.stat()
    src = comics_io.ComicsArchive(src_path)
    src.read_all()
    src.read_data_json()
    after = src_path.stat()
    assert before.st_mtime == after.st_mtime
    assert before.st_size == after.st_size
