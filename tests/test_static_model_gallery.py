from pathlib import Path

import pytest

from scripts.test_sim import find_models_under, grid_positions


def test_find_models_under_discovers_nested_batch_samples(tmp_path: Path):
    expected = []
    for seed in (7, 19):
        urdf = tmp_path / f"seed_{seed}" / "robot_assets" / "robot.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot/>", encoding="utf-8")
        expected.append((f"seed_{seed}", urdf.resolve()))

    assert find_models_under(tmp_path) == sorted(expected)


def test_grid_positions_centres_eight_models_in_four_by_two_layout():
    positions = grid_positions(8, spacing=2.0, columns=4)

    assert len(positions) == 8
    assert sum(position[0] for position in positions) == pytest.approx(0.0)
    assert sum(position[1] for position in positions) == pytest.approx(0.0)
    assert sorted({position[0] for position in positions}) == [-3.0, -1.0, 1.0, 3.0]
    assert sorted({position[1] for position in positions}) == [-1.0, 1.0]
