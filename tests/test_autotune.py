import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptation.autotune import generate_candidates, score_metrics


STANDARD = Path(__file__).resolve().parents[1] / "robot_assets" / "standard_hexapod" / "robot_description.json"


def test_candidates_include_two_controller_waveforms():
    description = json.loads(STANDARD.read_text())
    candidates = generate_candidates(description, max_candidates=4)
    assert candidates[0].plan["cpg"]["mode"] == "tripod"
    assert candidates[1].plan["cpg"]["mode"] == "legacy_sine"


def test_accepted_probe_always_wins():
    passed = {"passed": True, "forward_speed": 0.03}
    fast_but_failed = {"passed": False, "forward_speed": 10.0}
    assert score_metrics(passed) > score_metrics(fast_but_failed)


def test_candidates_keep_opposite_walking_directions():
    description = json.loads(STANDARD.read_text())
    candidates = generate_candidates(description, max_candidates=18)
    axes = [np.asarray(item.plan["final_forward_axis"]) for item in candidates]
    first = axes[0] / np.linalg.norm(axes[0])
    assert any(float(np.dot(first, axis / np.linalg.norm(axis))) < -0.999
               for axis in axes)
