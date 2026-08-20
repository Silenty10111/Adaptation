import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptation.morphology import amputate_legs


STANDARD = Path(__file__).resolve().parents[1] / "robot_assets" / "standard_hexapod" / "robot_description.json"


@pytest.mark.parametrize("removed", [[i] for i in range(6)] + [[0, 1], [1, 4], [3, 5]])
def test_amputation_preserves_a_valid_unique_tree(removed):
    source = json.loads(STANDARD.read_text())
    result = amputate_legs(source, removed)
    expected_legs = 6 - len(removed)
    assert result["num_legs"] == expected_legs
    assert len(result["links"]) == 1 + expected_legs * 6
    assert len(result["joints"]) == expected_legs * 6
    link_names = [x["name"] for x in result["links"]]
    joint_names = [x["name"] for x in result["joints"]]
    assert len(link_names) == len(set(link_names))
    assert len(joint_names) == len(set(joint_names))
    assert {x["leg_id"] for x in result["links"] if x.get("leg_id") is not None} == set(range(expected_legs))
    assert all(j["parent"] in link_names and j["child"] in link_names for j in result["joints"])


def test_amputation_does_not_mutate_source():
    source = json.loads(STANDARD.read_text())
    before = json.dumps(source, sort_keys=True)
    amputate_legs(source, [1])
    assert json.dumps(source, sort_keys=True) == before


def test_amputation_rejects_unknown_leg():
    source = json.loads(STANDARD.read_text())
    with pytest.raises(ValueError, match="unknown leg"):
        amputate_legs(source, [99])
