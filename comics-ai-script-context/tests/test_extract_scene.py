import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent
        / "comics-positioning"
        / "scripts"
    ),
)

import pytest

from extract_scene import ExtractionFailed, build_prompt, extract, parse_model_output

EXCERPT = "At heart I had chosen the king of Saubha for my husband."


def test_build_prompt_includes_excerpt_and_schema():
    prompt = build_prompt(EXCERPT)
    assert EXCERPT in prompt
    assert '"characters"' in prompt
    assert '"props"' in prompt
    assert '"locations"' in prompt


def test_parse_well_formed_json():
    raw = '{"characters": [{"name": "Amba", "action_or_state": "pleading"}], "props": [], "locations": ["Kasi"]}'
    result = parse_model_output(raw, "ep21.comics", EXCERPT)
    assert result.characters == (
        __import__("scene_models").CharacterMention(name="Amba", action_or_state="pleading"),
    )
    assert result.locations == ("Kasi",)
    assert result.raw_model_output == raw


def test_parse_json_wrapped_in_markdown_fence():
    raw = (
        "Here is the extracted scene:\n\n```json\n"
        '{"characters": [{"name": "Amba", "action_or_state": "pleading"}], "props": [], "locations": []}'
        "\n```\n\nLet me know if you need anything else."
    )
    result = parse_model_output(raw, "ep21.comics", EXCERPT)
    assert len(result.characters) == 1
    assert result.characters[0].name == "Amba"


def test_parse_malformed_output_raises():
    raw = "I cannot extract structured data from this text, sorry."
    with pytest.raises(ExtractionFailed) as exc_info:
        parse_model_output(raw, "ep21.comics", EXCERPT)
    assert exc_info.value.episode_file == "ep21.comics"
    assert exc_info.value.raw_output == raw


def test_parse_empty_characters_list_is_not_an_error():
    raw = '{"characters": [], "props": [], "locations": []}'
    result = parse_model_output(raw, "ep21.comics", EXCERPT)
    assert result.characters == ()


def test_parse_deduplicates_exact_name_matches():
    raw = (
        '{"characters": ['
        '{"name": "Amba", "action_or_state": "pleading"}, '
        '{"name": "Amba", "action_or_state": "reflecting"}'
        '], "props": [], "locations": []}'
    )
    result = parse_model_output(raw, "ep21.comics", EXCERPT)
    assert len(result.characters) == 1
    assert result.characters[0].action_or_state == "pleading"  # first occurrence wins


def test_parse_missing_required_field_raises():
    raw = '{"characters": []}'
    with pytest.raises(ExtractionFailed):
        parse_model_output(raw, "ep21.comics", EXCERPT)


@pytest.mark.slow
def test_live_extraction_finds_amba_in_episode_21():
    """Specifications' integration test: real ollama call, regression guard against a
    model/prompt change silently breaking the one known-good case from the Requirements spike.
    Uses the REAL full verified excerpt (not the short EXCERPT above, which is a first-person
    quote from Amba herself and never names her) -- the narrator's frame around it is what
    actually names her."""
    from text_context import VERIFIED

    episode_file = "8a89f7d689fb441ea280cd782276bd7a.comics"
    real_excerpt = VERIFIED[episode_file].excerpt

    result = extract(real_excerpt, episode_file)
    names = {c.name for c in result.characters}
    assert "Amba" in names
