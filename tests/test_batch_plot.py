import numpy as np

from scripts._batch_plot import project_to_path_frame


def test_projection_uses_commanded_forward_and_lateral_axes():
    lateral, forward, _, _ = project_to_path_frame(
        [[5.0, 7.0], [7.0, 8.0]], [1.0, 0.0]
    )
    np.testing.assert_allclose(lateral, [0.0, 1.0])
    np.testing.assert_allclose(forward, [0.0, 2.0])


def test_projection_handles_rotated_axis_and_ignores_world_origin():
    lateral, forward, _, _ = project_to_path_frame(
        [[100.0, -30.0], [101.0, -28.0]], [0.0, 4.0]
    )
    np.testing.assert_allclose(lateral, [0.0, -1.0])
    np.testing.assert_allclose(forward, [0.0, 2.0])


def test_zero_axis_has_deterministic_x_fallback():
    lateral, forward, fwd_axis, _ = project_to_path_frame(
        [[0.0, 0.0], [3.0, 2.0]], [0.0, 0.0]
    )
    np.testing.assert_allclose(fwd_axis, [1.0, 0.0])
    np.testing.assert_allclose(lateral[-1], 2.0)
    np.testing.assert_allclose(forward[-1], 3.0)
