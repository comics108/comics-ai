"""Read-only access to dataset/*.comics, and a writer for new output .comics archives.

dataset/ must never be opened in write mode anywhere in this pipeline -- every function here that
touches a source .comics file only ever opens it with mode="r".
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

DATA_JSON_NAME = "data.json"


@dataclass
class ComicsArchive:
    """Read-only view of a single dataset/*.comics file."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def names(self) -> list[str]:
        with zipfile.ZipFile(self.path, mode="r") as zf:
            return zf.namelist()

    def read_bytes(self, name: str) -> bytes:
        with zipfile.ZipFile(self.path, mode="r") as zf:
            return zf.read(name)

    def read_data_json(self) -> dict:
        raw = self.read_bytes(DATA_JSON_NAME).decode("utf-8-sig")
        return json.loads(raw)

    def read_all(self) -> dict[str, bytes]:
        """All entries (name -> bytes), for entries only (not directory markers)."""
        with zipfile.ZipFile(self.path, mode="r") as zf:
            return {
                info.filename: zf.read(info.filename)
                for info in zf.infolist()
                if not info.is_dir()
            }


def write_comics(
    dest_path: str | Path,
    source: ComicsArchive,
    *,
    data_json_override: dict | None = None,
    extra_files: dict[str, bytes] | None = None,
) -> Path:
    """Write a new .comics file to dest_path: every entry from `source`, with data.json optionally
    replaced and extra files added. `source` is only ever read, never written.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = source.read_all()
    if data_json_override is not None:
        entries[DATA_JSON_NAME] = json.dumps(
            data_json_override, ensure_ascii=False, indent=2
        ).encode("utf-8")
    if extra_files:
        entries.update(extra_files)

    with zipfile.ZipFile(dest_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)

    return dest_path
