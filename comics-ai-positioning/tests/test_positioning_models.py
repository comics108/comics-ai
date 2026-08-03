import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from positioning_models import PositionProposal, PositionTrainingPair, RegionFeatures


def test_position_training_pair_round_trips_through_json():
    pair = PositionTrainingPair(
        episode_file="a.comics",
        photo_file="p.jpg",
        region=RegionFeatures(
            kind="balloon",
            kind_source="predicted",
            local_bbox=(1, 2, 3, 4),
            page_index=0,
            reading_order_index=2,
        ),
        target_layer_index=5,
        target_bbox=(10, 20, 30, 40),
        target_transform={"x": 10, "y": 20},
        match_confidence=0.8,
    )
    encoded = json.dumps(pair.to_jsonable())
    decoded = json.loads(encoded)
    assert decoded["episode_file"] == "a.comics"
    assert decoded["region"]["kind"] == "balloon"
    assert decoded["target_bbox"] == [10, 20, 30, 40]


def test_position_proposal_defaults():
    proposal = PositionProposal(region_id="r1", proposed_x=5, proposed_y=6)
    assert proposal.source == "baseline"
    assert proposal.confidence is None
    assert proposal.proposed_scale_x == 1.0
