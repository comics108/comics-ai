#!/usr/bin/env python3
"""Non-circular OCR engine/configuration audit over immutable lettering glyph fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from lettering import exact_readback_match


CONFIGS = {
    "en": (("eng", 6), ("lat", 6), ("script/Latin", 6), ("script/Latin", 11)),
    "sa": (("san", 6), ("san+hin", 6), ("script/Devanagari", 6), ("script/Devanagari", 11)),
}


def audit(authoritative_manifest: Path, fixture_manifest: Path) -> dict:
    authoritative = json.loads(authoritative_manifest.read_text(encoding="utf-8"))
    fixtures = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    text_by_id = {item["id"]: item["text"] for item in authoritative["entries"]}
    failed = [item for item in fixtures["results"] if item["decision"] == "rejected"]
    rows = []
    for fixture in failed:
        entry_id, language = fixture["id"], fixture["language_code"]
        rgba = Image.open(fixture["files"]["rgba"]).convert("RGBA")
        alpha = rgba.getchannel("A")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba.convert("RGB"), mask=alpha)
        for scale in (1, 2):
            image = background if scale == 1 else background.resize(
                (background.width * scale, background.height * scale), Image.Resampling.LANCZOS
            )
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "ocr.png"; image.save(path)
                for ocr_language, psm in CONFIGS[language]:
                    completed = subprocess.run(
                        ["tesseract", str(path), "stdout", "-l", ocr_language, "--psm", str(psm)],
                        capture_output=True, text=True, check=False,
                    )
                    readback = completed.stdout.strip()
                    rows.append({"id": entry_id, "language_code": language, "ocr_language": ocr_language,
                                 "psm": psm, "scale": scale, "returncode": completed.returncode,
                                 "exact": completed.returncode == 0 and exact_readback_match(text_by_id[entry_id], readback),
                                 "readback": readback})
    exact = sum(item["exact"] for item in rows)
    return {"schema_version": 1,
            "authoritative_manifest_sha256": hashlib.sha256(authoritative_manifest.read_bytes()).hexdigest(),
            "fixture_manifest_sha256": hashlib.sha256(fixture_manifest.read_bytes()).hexdigest(),
            "configuration_count": len(rows), "exact_match_count": exact,
            "decision": "accepted" if failed and exact == len(failed) else "rejected",
            "rows": rows,
            "constraints": ["no_authoritative_wordlist", "no_fuzzy_matching", "no_diacritic_stripping"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--authoritative",type=Path,required=True); parser.add_argument("--fixtures",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); report=audit(args.authoritative,args.fixtures); args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream: json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2); stream.write("\n")
    print(json.dumps({"decision":report["decision"],"exact":report["exact_match_count"],"configurations":report["configuration_count"]}))


if __name__=="__main__": main()
