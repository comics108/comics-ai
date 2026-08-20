#!/usr/bin/env python3
"""Apple Vision exact-readback audit; unsupported scripts abstain explicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from lettering import exact_readback_match


def audit(authoritative_manifest: Path, fixture_manifest: Path, swift_script: Path) -> dict:
    authoritative = json.loads(authoritative_manifest.read_text(encoding="utf-8"))
    fixtures = json.loads(fixture_manifest.read_text(encoding="utf-8"))
    texts = {item["id"]: item["text"] for item in authoritative["entries"]}
    rows = []
    for fixture in fixtures["results"]:
        if fixture["decision"] != "rejected":
            continue
        if fixture["language_code"] != "en":
            rows.append({"id": fixture["id"], "language_code": fixture["language_code"],
                         "state": "abstained", "reason": "vision_language_not_supported"})
            continue
        rgba = Image.open(fixture["files"]["rgba"]).convert("RGBA")
        white = Image.new("RGB", rgba.size, "white"); white.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.png"; white.save(path)
            completed = subprocess.run(["swift", str(swift_script), str(path), "en-US"], capture_output=True, text=True, check=False)
        readback = completed.stdout.strip()
        exact = completed.returncode == 0 and exact_readback_match(texts[fixture["id"]], readback)
        rows.append({"id": fixture["id"], "language_code": "en", "state": "accepted" if exact else "rejected",
                     "reason": None if exact else "ocr_exact_string_mismatch", "readback": readback})
    return {"schema_version": 1,
            "authoritative_manifest_sha256": hashlib.sha256(authoritative_manifest.read_bytes()).hexdigest(),
            "fixture_manifest_sha256": hashlib.sha256(fixture_manifest.read_bytes()).hexdigest(),
            "engine": "apple-vision-accurate-no-language-correction", "rows": rows,
            "decision": "accepted" if rows and all(item["state"] == "accepted" for item in rows) else "rejected",
            "constraints": ["no_custom_words", "no_language_correction", "no_fuzzy_matching"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--authoritative",type=Path,required=True); parser.add_argument("--fixtures",type=Path,required=True); parser.add_argument("--swift-script",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); report=audit(args.authoritative,args.fixtures,args.swift_script); args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream: json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2); stream.write("\n")
    print(json.dumps({"decision":report["decision"],"states":[item["state"] for item in report["rows"]]}))


if __name__=="__main__": main()
