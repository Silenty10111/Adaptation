import json
from pathlib import Path

import pytest

from scripts.test_gait import resolve_robot_model


def make_model(directory: Path, urdf_name: str = "robot.urdf") -> tuple[Path, Path]:
    directory.mkdir()
    description = directory / "robot_description.json"
    urdf = directory / urdf_name
    description.write_text(
        json.dumps({"robot_name": "test", "num_legs": 4, "urdf_path": urdf_name}),
        encoding="utf-8",
    )
    urdf.write_text("<robot name='test'/>", encoding="utf-8")
    return description, urdf


@pytest.mark.parametrize("entry_kind", ["directory", "description", "urdf"])
def test_resolve_robot_model_accepts_all_supported_entries(tmp_path, entry_kind):
    description, urdf = make_model(tmp_path / "robot")
    entry = {
        "directory": description.parent,
        "description": description,
        "urdf": urdf,
    }[entry_kind]

    resolved_description, resolved_urdf = resolve_robot_model(entry)

    assert resolved_description == description.resolve()
    assert resolved_urdf == urdf.resolve()


def test_resolve_robot_model_supports_standard_urdf_name(tmp_path):
    description, urdf = make_model(tmp_path / "standard", "generated_robot.urdf")

    assert resolve_robot_model(description.parent) == (
        description.resolve(), urdf.resolve(),
    )


def test_resolve_robot_model_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_robot_model(tmp_path / "missing")


def test_resolve_robot_model_rejects_directory_without_description(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="description not found"):
        resolve_robot_model(empty)
