import sys
import json
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_vision_ocr import audit


def test_vision_swift_script_disables_language_correction():
    script = Path(__file__).resolve().parent.parent / "scripts/vision_ocr.swift"
    text = script.read_text()
    assert "recognitionLevel = .accurate" in text
    assert "usesLanguageCorrection = false" in text
    assert "customWords" not in text


def test_audit_rejects_inexact_english_and_abstains_on_sanskrit(tmp_path, monkeypatch):
    rgba_path = tmp_path / "en.png"
    Image.new("RGBA", (20, 10), (0, 0, 0, 255)).save(rgba_path)
    authoritative = tmp_path / "authoritative.json"
    authoritative.write_text(json.dumps({"entries": [
        {"id": "en", "text": "Dhṛtarāṣṭra"},
        {"id": "sa", "text": "धृतराष्ट्र"},
    ]}))
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(json.dumps({"results": [
        {"id": "en", "language_code": "en", "decision": "rejected",
         "files": {"rgba": str(rgba_path)}},
        {"id": "sa", "language_code": "sa", "decision": "rejected",
         "files": {"rgba": str(rgba_path)}},
    ]}))

    class Completed:
        returncode = 0
        stdout = "Dhrtarastra\n"

    monkeypatch.setattr("audit_vision_ocr.subprocess.run", lambda *args, **kwargs: Completed())
    result = audit(authoritative, fixtures, tmp_path / "vision.swift")

    assert result["decision"] == "rejected"
    assert [row["state"] for row in result["rows"]] == ["rejected", "abstained"]
    assert result["constraints"] == [
        "no_custom_words", "no_language_correction", "no_fuzzy_matching",
    ]
