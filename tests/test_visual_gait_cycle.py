import math

import numpy as np
import pytest

from scripts.test_gait import build_gait_targets, gait_wave


def test_gait_wave_has_grounded_stance_and_lifted_return():
    duty = 0.60

    stance_swing, stance_lift = gait_wave(2.0 * math.pi * 0.30, duty)
    return_swing, return_lift = gait_wave(2.0 * math.pi * 0.80, duty)

    assert stance_swing == pytest.approx(0.0, abs=1e-12)
    assert stance_lift == 0.0
    assert return_swing == pytest.approx(0.0, abs=1e-12)
    assert return_lift == pytest.approx(1.0)


def test_gait_wave_is_continuous_at_touchdown_and_liftoff():
    duty = 0.60
    epsilon = 1e-7

    before_liftoff = gait_wave(2.0 * math.pi * (duty - epsilon), duty)
    after_liftoff = gait_wave(2.0 * math.pi * (duty + epsilon), duty)
    before_touchdown = gait_wave(2.0 * math.pi * (1.0 - epsilon), duty)
    after_touchdown = gait_wave(2.0 * math.pi * epsilon, duty)

    assert before_liftoff == pytest.approx(after_liftoff, abs=1e-9)
    assert before_touchdown == pytest.approx(after_touchdown, abs=1e-9)


def test_build_targets_raises_lift_joint_only_during_return_swing():
    description = {
        "links": [
            {"role": "foot", "leg_id": 0, "default_world_origin": [0.0, 0.3, -0.5]},
        ],
    }
    plan = {
        "final_forward_axis": [1.0, 0.0],
        "topology": {
            "groups": {"group_a": [0], "group_b": [], "group_c": []},
        },
    }
    triplets = {
        0: {
            "lift_idx": 0, "lift_lower": 0.0, "lift_upper": 1.0,
            "drop_idx": 1, "drop_lower": 0.0, "drop_upper": 1.0,
            "swing_idx": 2, "swing_lower": -1.0, "swing_upper": 1.0,
        },
    }

    common = dict(
        description=description,
        gait_plan=plan,
        triplets=triplets,
        defaults=np.zeros(3, dtype=np.float32),
        gait_freq=1.0,
        swing_amp=0.3,
        stance_lift=0.2,
        swing_lift=0.7,
        stance_drop=0.1,
        swing_drop=0.1,
        gait_mode="alternating",
        duty_factor=0.60,
    )

    stance = build_gait_targets(sim_time=0.30, **common)
    returning = build_gait_targets(sim_time=0.80, **common)

    assert stance[0] == pytest.approx(0.2)
    assert returning[0] == pytest.approx(0.7)
    assert stance[1] == pytest.approx(0.1)
    assert returning[1] == pytest.approx(0.1)

