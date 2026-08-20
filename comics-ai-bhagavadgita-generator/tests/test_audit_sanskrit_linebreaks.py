import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_sanskrit_linebreaks import _with_breaks, line_variants


def test_line_variants_preserve_words_and_use_three_or_four_lines():
    text = "one two three four five"
    variants = line_variants(text)
    assert len(variants) == 10
    for breaks in variants:
        rendered = _with_breaks(text.split(), breaks)
        assert rendered.split() == text.split()
        assert rendered.count("\n") in (2, 3)
