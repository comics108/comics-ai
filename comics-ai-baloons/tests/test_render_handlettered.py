import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import render_handlettered as rhl


def test_flags_every_language_without_rendering():
    results = rhl.handle_hand_lettered_balloon("f.comics", 181, ["hi", "uk", "th"])
    assert len(results) == 3
    for r in results:
        assert r.rendered is False
        assert r.image_path is None
        assert "manual" in r.reason.lower()


def test_empty_target_langs():
    assert rhl.handle_hand_lettered_balloon("f.comics", 181, []) == []
