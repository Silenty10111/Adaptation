#!/usr/bin/env python3
"""Import the generated URDF into Isaac Sim, export USD, and create a simple scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


ASSET_DIR_NAME = "robot_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(ASSET_DIR_NAME))
    parser.add_argument("--description", type=Path, default=Path(ASSET_DIR_NAME) / "robot_description.json")
    parser.add_argument("--urdf", type=Path, default=Path(ASSET_DIR_NAME) / "generated_robot.urdf")
    parser.add_argument("--usd-output", type=Path, default=Path(ASSET_DIR_NAME) / "generated_robot.usd")
    parser.add_argument("--scene-output", type=Path, default=Path(ASSET_DIR_NAME) / "generated_scene.usd")
    parser.add_argument(
        "--headless",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use 1 for SSH/headless execution, 0 for local GUI mode.",
    )
    parser.add_argument("--sim-steps", type=int, default=120, help="Warm-up simulation steps after loading the scene.")
    return parser.parse_args()


def load_description(description_path: Path) -> Dict[str, object]:
    return json.loads(description_path.read_text(encoding="utf-8"))


def safe_setattr(obj: object, attr_name: str, value: object) -> None:
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)


class MacroLegController:
    """Small controller abstraction that maps macro commands onto named URDF joints."""

    def __init__(self, leg_count: int) -> None:
        self.leg_count = leg_count
        self.pending_targets: Dict[str, float] = {}

    def _joint_name(self, leg_id: int, action: str) -> str:
        if leg_id < 0 or leg_id >= self.leg_count:
            raise IndexError(f"leg_id out of range: {leg_id}")
        return f"leg_{leg_id}_{action}"

    def lift_leg(self, leg_id: int, amount: float = 0.60) -> None:
        self.pending_targets[self._joint_name(leg_id, "lift")] = float(amount)

    def swing_leg(self, leg_id: int, target_pos: List[float]) -> None:
        if len(target_pos) != 3:
            raise ValueError("target_pos must be a 3D vector [x, y, z].")
        swing_angle = math.atan2(float(target_pos[1]), max(abs(float(target_pos[0])), 1e-6))
        self.pending_targets[self._joint_name(leg_id, "swing")] = float(np_clip(swing_angle, -0.55, 0.55))

    def drop_leg(self, leg_id: int, extension: float = 0.75) -> None:
        self.pending_targets[self._joint_name(leg_id, "drop")] = float(np_clip(extension, 0.0, 1.10))

    def flush(self, articulation=None) -> Dict[str, float]:
        """Return buffered joint targets, or apply them if an articulation wrapper is supplied.

        In Isaac Lab / Isaac Sim, replace the pseudocode inside this method with the articulation API
        used in your stack, for example an ArticulationView or Isaac Lab Articulation object.
        """
        targets = dict(self.pending_targets)
        if articulation is not None:
            for joint_name, target in targets.items():
                if hasattr(articulation, "set_joint_position_target"):
                    articulation.set_joint_position_target(target, joint_name)
        self.pending_targets.clear()
        return targets


def np_clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def create_scene_prims(stage) -> None:
    from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux, UsdPhysics

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 25.0, Gf.Vec3f(0.0), Gf.Vec3f(0.55))

    light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    light.CreateIntensityAttr(3500.0)


def import_urdf_to_usd(urdf_path: Path, usd_output: Path):
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("Failed to create Isaac URDF import configuration.")

    safe_setattr(import_config, "merge_fixed_joints", False)
    safe_setattr(import_config, "convex_decomp", False)
    safe_setattr(import_config, "import_inertia_tensor", True)
    safe_setattr(import_config, "fix_base", False)
    safe_setattr(import_config, "distance_scale", 1.0)
    safe_setattr(import_config, "create_physics_scene", False)
    safe_setattr(import_config, "make_default_prim", True)
    safe_setattr(import_config, "collision_from_visuals", False)

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path.resolve()),
        import_config=import_config,
        dest_path=str(usd_output.resolve()),
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError("Isaac URDF importer failed.")
    return prim_path


def build_scene(description: Dict[str, object], usd_output: Path, scene_output: Path) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    create_scene_prims(stage)

    robot_prim = stage.DefinePrim("/World/GeneratedRobot", "Xform")
    robot_prim.GetReferences().AddReference(str(usd_output.resolve()).replace("\\", "/"))

    robot_xform = UsdGeom.Xformable(robot_prim)
    robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.32))

    stage.GetRootLayer().Export(str(scene_output.resolve()))

    description["usd_path"] = usd_output.as_posix()
    description["scene_path"] = scene_output.as_posix()


def main() -> None:
    args = parse_args()
    description = load_description(args.description)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless), "renderer": "RayTracedLighting"})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.asset.importer.urdf")

        prim_path = import_urdf_to_usd(args.urdf, args.usd_output)
        build_scene(description, args.usd_output, args.scene_output)

        controller = MacroLegController(int(description["num_legs"]))
        controller.lift_leg(0)
        controller.swing_leg(0, [0.15, 0.05, -0.20])
        controller.drop_leg(0)
        print("Macro controller example joint targets:")
        print(controller.flush())
        print(f"Imported articulation root: {prim_path}")
        print(f"USD asset saved to: {args.usd_output.resolve()}")
        print(f"Scene saved to: {args.scene_output.resolve()}")

        for _ in range(max(args.sim_steps, 0)):
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()