import numpy as np

from adaptation.validation import evaluate_trajectory


def _trail(vx=0.10, vy=0.0, yaw=0.0, z=0.50, n=600, dt=1 / 60):
    t = np.arange(n) * dt
    return np.column_stack((vx * t, vy * t, np.full(n, yaw), np.full(n, z)))


def test_accepts_steady_straight_motion():
    result = evaluate_trajectory(_trail(), [1.0, 0.0])
    assert result["passed"]
    assert abs(result["forward_speed"] - 0.1) < 1e-9


def test_rejects_lateral_drift():
    result = evaluate_trajectory(_trail(vy=0.04), [1.0, 0.0])
    assert not result["passed"]
    assert not result["checks"]["lateral_speed"]


def test_rejects_wrong_heading():
    result = evaluate_trajectory(_trail(vx=0.10, vy=0.05), [1.0, 0.0])
    assert not result["passed"]
    assert not result["checks"]["heading_error"]


def test_rejects_stop_go_motion():
    n = 600
    dt = 1 / 60
    # Multi-cycle stop/go behaviour must not be hidden by the one-second gait
    # averaging window used to remove normal contact ripple.
    velocity = np.tile(np.r_[np.zeros(120), np.full(120, 0.2)], 3)[:n]
    x = np.cumsum(velocity) * dt
    samples = np.column_stack((x, np.zeros(n), np.zeros(n), np.full(n, 0.5)))
    result = evaluate_trajectory(samples, [1.0, 0.0])
    assert not result["passed"]
    assert not result["checks"]["speed_cv"]


def test_rotated_forward_axis():
    result = evaluate_trajectory(_trail(vx=0.0, vy=0.1, yaw=np.pi / 2), [0.0, 1.0])
    assert result["passed"]


def test_rejects_fallen_robot():
    result = evaluate_trajectory(_trail(vx=0.0, z=0.12), [1.0, 0.0])
    assert not result["passed"]
    assert not result["checks"]["mean_height"]
