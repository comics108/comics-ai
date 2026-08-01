"""Canonical language <-> array-index table for balloon Image slots.

Indices 0-2 are fixed by the real editor's C# `Cultures` enum
(apps/comics-editor-v2.9/native/Comics.Editor/Models/Cultures.cs: En=0, Ru=1, Hi=2) and must never
change. Indices 3+ are this feature's proposed extension, in the translation CSV's own column
order (excluding en/ru/hi, already placed) -- see flows/sdd-comics-ai-baloons/02-specifications.md.
"""

from __future__ import annotations

# Order matters. Do not reorder existing entries -- only append.
LANGUAGES: list[str] = [
    "en",  # 0 -- Cultures.En
    "ru",  # 1 -- Cultures.Ru
    "hi",  # 2 -- Cultures.Hi
    "uk",  # 3
    "th",  # 4
    "zh",  # 5
    "ko",  # 6
    "kn",  # 7
    "es",  # 8
    "fr",  # 9
    "pt",  # 10
    "ja",  # 11
    "tr",  # 12
    "vi",  # 13
    "ta",  # 14
    "mr",  # 15
    "bn",  # 16
    "ne",  # 17
    "he",  # 18
    "ar",  # 19
]

# The three culture slots the current editor actually reads (Models/Cultures.cs enum order).
EXISTING_CULTURES_ORDER = ["en", "ru", "hi"]

# Scripts that need shaping beyond plain Latin/Cyrillic rendering (see render_shaped.py).
COMPLEX_SCRIPT_LANGUAGES = {
    "th", "zh", "ko", "kn", "ja", "hi", "ta", "mr", "bn", "ne", "he", "ar",
}

RTL_LANGUAGES = {"he", "ar"}

_LANG_TO_INDEX = {code: i for i, code in enumerate(LANGUAGES)}


def index_to_lang(index: int) -> str:
    return LANGUAGES[index]


def lang_to_index(code: str) -> int:
    return _LANG_TO_INDEX[code]


def is_known_language(code: str) -> bool:
    return code in _LANG_TO_INDEX


assert LANGUAGES[:3] == EXISTING_CULTURES_ORDER, (
    "Indices 0-2 must match the real Cultures enum order (En, Ru, Hi) -- "
    "this is load-bearing against the actual dataset, not a convention we can change."
)
assert len(LANGUAGES) == len(set(LANGUAGES)), "Duplicate language code in LANGUAGES"
