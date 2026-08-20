#!/usr/bin/env python3
"""Immutable, atomically-published production artifact storage.

The store owns only ``work/bhagavadgita/production``-style roots.  A version is
built below ``.staging``, validated there, and renamed into its final location;
an existing final version is never replaced.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ImmutableVersionStore:
    """Create immutable ``<namespace>/<object-id>/<version>`` directories."""

    def __init__(self, root: Path, *, source_roots: tuple[Path, ...] = ()) -> None:
        self.root = root.resolve()
        self.source_roots = tuple(path.resolve() for path in source_roots)

    def create_version(
        self,
        *,
        namespace: str,
        object_id: str,
        version: int | str,
        files: Mapping[str, bytes],
        metadata: Mapping[str, Any],
        validator: Callable[[Path], None] | None = None,
    ) -> Path:
        """Stage, validate, and atomically publish one new immutable version."""
        self._assert_outside_source_roots(self.root)
        namespace = self._safe_id(namespace, "namespace")
        object_id = self._safe_id(object_id, "object_id")
        version_text = self._safe_id(str(version), "version")
        destination = self.root / namespace / object_id / version_text
        if destination.exists():
            raise FileExistsError(f"immutable version already exists: {destination}")

        staging_root = self.root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        stage = staging_root / f"{namespace}-{object_id}-{version_text}-{uuid.uuid4().hex}"
        stage.mkdir()
        try:
            for relative_name, content in files.items():
                target = stage / self._safe_relative_path(relative_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            (stage / "metadata.json").write_text(
                json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            if validator is not None:
                validator(stage)

            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"immutable version already exists: {destination}")
            os.rename(stage, destination)
            return destination
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _assert_outside_source_roots(self, path: Path) -> None:
        resolved = path.resolve()
        for source_root in self.source_roots:
            if resolved == source_root or resolved.is_relative_to(source_root):
                raise PermissionError(
                    f"write target is inside read-only source root {source_root}: {resolved}"
                )

    @staticmethod
    def _safe_id(value: str, label: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(f"unsafe {label}: {value!r}")
        return value

    @staticmethod
    def _safe_relative_path(value: str) -> Path:
        path = PurePosixPath(value)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"unsafe version file path: {value!r}")
        return Path(*path.parts)
