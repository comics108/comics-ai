"""PDF native-image inventory via Poppler metadata tools, without page rendering."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddedPdfImage:
    page: int
    image_number: int
    image_type: str
    width: int
    height: int
    color_space: str
    components: int
    bits_per_component: int
    encoding: str
    interpolated: bool
    object_id: tuple[int, int]
    x_ppi: int
    y_ppi: int


@dataclass(frozen=True)
class RecoveredPdfDocument:
    source_path: Path
    page_count: int
    page_size_points: tuple[float, float]
    file_size: int
    pdf_version: str
    embedded_images: tuple[EmbeddedPdfImage, ...]


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout


def recover_pdf_structure(path: Path) -> RecoveredPdfDocument:
    """Recover page/media facts and embedded-image objects without rasterizing pages."""
    source = path.resolve(strict=True)
    info_text = _run(["pdfinfo", str(source)])
    info: dict[str, str] = {}
    for line in info_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    page_size_match = re.match(r"([0-9.]+)\s+x\s+([0-9.]+)\s+pts", info.get("Page size", ""))
    if not page_size_match:
        raise ValueError(f"pdfinfo omitted parseable page size for {source}")

    images: list[EmbeddedPdfImage] = []
    for line in _run(["pdfimages", "-list", str(source)]).splitlines():
        fields = line.split()
        if len(fields) < 16 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        images.append(EmbeddedPdfImage(
            page=int(fields[0]),
            image_number=int(fields[1]),
            image_type=fields[2],
            width=int(fields[3]),
            height=int(fields[4]),
            color_space=fields[5],
            components=int(fields[6]),
            bits_per_component=int(fields[7]),
            encoding=fields[8],
            interpolated=fields[9] == "yes",
            object_id=(int(fields[10]), int(fields[11])),
            x_ppi=int(fields[12]),
            y_ppi=int(fields[13]),
        ))
    return RecoveredPdfDocument(
        source_path=source,
        page_count=int(info["Pages"]),
        page_size_points=(float(page_size_match.group(1)), float(page_size_match.group(2))),
        file_size=int(info["File size"].split()[0]),
        pdf_version=info["PDF version"],
        embedded_images=tuple(images),
    )
