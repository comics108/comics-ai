import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scene_models import CharacterMention, SceneExtraction


def test_scene_extraction_round_trips_through_dict():
    original = SceneExtraction(
        episode_file="8a89f7d689fb441ea280cd782276bd7a.comics",
        source_excerpt="At heart I had chosen the king of Saubha for my husband.",
        characters=(
            CharacterMention(name="Amba", action_or_state="permitted to choose her own path"),
            CharacterMention(name="king of Saubha", action_or_state="her originally chosen husband"),
        ),
        props=(),
        locations=("Kasi",),
        raw_model_output='{"characters": [...]}',
        model_name="qwen2.5-coder:32b",
    )

    round_tripped = SceneExtraction.from_dict(original.to_dict())

    assert round_tripped == original
