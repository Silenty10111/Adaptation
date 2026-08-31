#!/usr/bin/env python3
"""Batch-generate and inspect multiple robot variants in PyBullet."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple


p: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="需要生成并展示的机器人数量。")
    parser.add_argument("--seed-base", type=int, default=7, help="第一个模型使用的随机种子。")
    parser.add_argument("--seed-step", type=int, default=17, help="相邻模型种子间隔。")
    parser.add_argument("--spacing", type=float, default=2.0, help="模型在 X 轴上的摆放间距。")
    parser.add_argument("--variant-prefix", default="variant", help="模型命名前缀。")
    parser.add_argument("--no-generate", action="store_true", help="不重新生成，直接加载已有变体。")
    parser.add_argument(
        "--models-root", type=Path,
        help="递归加载该目录中的 robot.urdf；适用于 batch_results 中的现有样例。",
    )
    parser.add_argument("--grid-columns", type=int, default=4, help="全部展示时的网格列数。")
    parser.add_argument("--fixed-base", action="store_true", help="固定主体，进行纯静态结构展示。")
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="默认展示所有模型；不传则默认突出显示当前选中模型。",
    )
    return parser.parse_args()


def run_cmd(args: List[str]) -> None:
    subprocess.run(args, check=True)


def run_cmd_ok(args: List[str]) -> bool:
    """Run a command, return True on success, False on non-zero exit (e.g. SSM fail)."""
    result = subprocess.run(args)
    return result.returncode == 0


def clone_generated_assets(variant_dir: Path) -> Path:
    src_urdf = Path("robot_assets/generated_robot.urdf")
    src_mesh_dir = Path("robot_assets/meshes")

    if not src_urdf.exists():
        raise FileNotFoundError(f"找不到 URDF: {src_urdf}")
    if not src_mesh_dir.exists():
        raise FileNotFoundError(f"找不到网格目录: {src_mesh_dir}")

    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    (variant_dir / "meshes").mkdir(parents=True, exist_ok=True)

    dst_urdf = variant_dir / "robot.urdf"
    shutil.copy2(src_urdf, dst_urdf)

    for mesh_file in src_mesh_dir.glob("*.stl"):
        shutil.copy2(mesh_file, variant_dir / "meshes" / mesh_file.name)

    return dst_urdf


def generate_variants(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    """Generate `args.count` unique robot variants, skipping seeds that fail SSM."""
    variants_root = Path("robot_assets/variants")
    variants_root.mkdir(parents=True, exist_ok=True)

    generated: List[Tuple[str, Path]] = []
    index = 0          # variant counter (only increments on success)
    seed_offset = 0    # absolute seed offset (increments every attempt)
    max_attempts = args.count * 20  # safety cap to avoid infinite loop
    attempts = 0

    while len(generated) < args.count and attempts < max_attempts:
        seed = args.seed_base + seed_offset * args.seed_step
        name = f"{args.variant_prefix}_{index:02d}_seed{seed}"
        seed_offset += 1
        attempts += 1

        ok = run_cmd_ok(
            [
                sys.executable,
                "generate_geometry.py",
                "--robot-name",
                name,
                "--seed",
                str(seed),
                "--leg-placement",
                "random",
            ]
        )
        if not ok:
            print(f"[SKIP] seed={seed} 未通过 SSM 预检 (geometry fail)，跳过。")
            continue

        urdf_ok = run_cmd_ok(
            [sys.executable, "generate_urdf.py"]
        )
        if not urdf_ok:
            print(f"[SKIP] seed={seed} 未通过 SSM 构建检 (urdf fail)，跳过。")
            continue

        variant_dir = variants_root / name
        urdf_path = clone_generated_assets(variant_dir)
        generated.append((name, urdf_path))
        index += 1

    if len(generated) < args.count:
        raise RuntimeError(
            f"仅生成了 {len(generated)}/{args.count} 个变体（尝试了 {attempts} 个种子）。"
            " 请减少 --count 或缩小 --body-length/--body-width 让足端更易通过 SSM 预检。"
        )
    return generated


def find_existing_variants(prefix: str) -> List[Tuple[str, Path]]:
    variants_root = Path("robot_assets/variants")
    if not variants_root.exists():
        return []

    found: List[Tuple[str, Path]] = []
    for item in sorted(variants_root.iterdir()):
        if not item.is_dir() or not item.name.startswith(prefix):
            continue
        urdf_path = item / "robot.urdf"
        if urdf_path.exists():
            found.append((item.name, urdf_path))
    return found


def find_models_under(root: Path) -> List[Tuple[str, Path]]:
    """Find generated sample URDFs recursively with stable display labels."""
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"模型根目录不存在: {resolved}")
    found: List[Tuple[str, Path]] = []
    for urdf_path in sorted(resolved.rglob("robot.urdf")):
        relative = urdf_path.relative_to(resolved)
        label = relative.parts[0] if len(relative.parts) > 1 else urdf_path.parent.name
        found.append((label, urdf_path.resolve()))
    return found


def grid_positions(count: int, spacing: float, columns: int) -> List[Tuple[float, float, float]]:
    """Return a centred row-major layout for displaying every model at once."""
    columns = max(1, min(int(columns), max(int(count), 1)))
    rows = max(1, int(math.ceil(count / columns)))
    positions: List[Tuple[float, float, float]] = []
    for index in range(count):
        column = index % columns
        row = index // columns
        x = (column - 0.5 * (columns - 1)) * spacing
        y = (0.5 * (rows - 1) - row) * spacing
        positions.append((float(x), float(y), 0.45))
    return positions


def lock_robot_joints(robot_id: int, hold_force: float = 50.0) -> None:
    num_joints = p.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            force=hold_force,
        )


def set_robot_alpha(robot_id: int, alpha: float) -> None:
    p.changeVisualShape(robot_id, -1, rgbaColor=[1.0, 1.0, 1.0, alpha])
    for joint_index in range(p.getNumJoints(robot_id)):
        p.changeVisualShape(robot_id, joint_index, rgbaColor=[1.0, 1.0, 1.0, alpha])


def focus_camera(
    target_pos: Tuple[float, float, float], distance: float = 2.8,
    pitch: float = -25.0,
) -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=38.0,
        cameraPitch=pitch,
        cameraTargetPosition=list(target_pos),
    )


def main() -> None:
    global p
    import pybullet as pybullet_module
    import pybullet_data

    p = pybullet_module
    args = parse_args()

    if args.models_root is not None:
        variants = find_models_under(args.models_root)
        if not variants:
            raise RuntimeError(f"没有在 {args.models_root} 下找到 robot.urdf。")
        if len(variants) > args.count:
            variants = variants[: args.count]
    elif args.no_generate:
        variants = find_existing_variants(args.variant_prefix)
        if not variants:
            raise RuntimeError("没有找到可加载的变体。请先去掉 --no-generate 生成模型。")
        if len(variants) > args.count:
            variants = variants[: args.count]
    else:
        variants = generate_variants(args)

    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("无法启动 PyBullet GUI。")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    model_ids: List[int] = []
    model_names: List[str] = []
    model_positions: List[Tuple[float, float, float]] = []

    layout = (
        grid_positions(len(variants), args.spacing, args.grid_columns)
        if args.show_all else
        [(index * args.spacing, 0.0, 0.5) for index in range(len(variants))]
    )
    for index, (name, urdf_path) in enumerate(variants):
        start_pos = layout[index]
        start_orientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        robot_id = p.loadURDF(
            str(urdf_path), start_pos, start_orientation,
            useFixedBase=args.fixed_base,
        )
        lock_robot_joints(robot_id)
        p.addUserDebugText(
            name, [start_pos[0], start_pos[1], start_pos[2] + 0.42],
            textColorRGB=[0.05, 0.05, 0.05], textSize=1.15,
        )
        model_ids.append(robot_id)
        model_names.append(name)
        model_positions.append(start_pos)

    active_index = 0
    single_focus = not args.show_all

    def apply_visibility() -> None:
        for idx, rid in enumerate(model_ids):
            if single_focus:
                set_robot_alpha(rid, 1.0 if idx == active_index else 0.12)
            else:
                set_robot_alpha(rid, 1.0)
        if single_focus:
            focus_camera(model_positions[active_index])
        else:
            array_extent = args.spacing * max(
                min(args.grid_columns, len(model_ids)),
                int(math.ceil(len(model_ids) / max(args.grid_columns, 1))),
            )
            focus_camera((0.0, 0.0, 0.15), distance=max(4.0, array_extent * 1.35), pitch=-48.0)
        print(f"当前模型: {active_index + 1}/{len(model_ids)} -> {model_names[active_index]}")

    apply_visibility()
    print("按键说明: [N/右箭头]下一个, [P/左箭头]上一个, [A]全部显示, [S]单模型聚焦, [Q]退出")

    while True:
        keys = p.getKeyboardEvents()

        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            break
        if ord("Q") in keys and keys[ord("Q")] & p.KEY_WAS_TRIGGERED:
            break

        moved = False
        if ord("n") in keys and keys[ord("n")] & p.KEY_WAS_TRIGGERED:
            active_index = (active_index + 1) % len(model_ids)
            moved = True
        if ord("N") in keys and keys[ord("N")] & p.KEY_WAS_TRIGGERED:
            active_index = (active_index + 1) % len(model_ids)
            moved = True
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED:
            active_index = (active_index + 1) % len(model_ids)
            moved = True

        if ord("p") in keys and keys[ord("p")] & p.KEY_WAS_TRIGGERED:
            active_index = (active_index - 1) % len(model_ids)
            moved = True
        if ord("P") in keys and keys[ord("P")] & p.KEY_WAS_TRIGGERED:
            active_index = (active_index - 1) % len(model_ids)
            moved = True
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED:
            active_index = (active_index - 1) % len(model_ids)
            moved = True

        for idx in range(min(9, len(model_ids))):
            keycode = ord(str(idx + 1))
            if keycode in keys and keys[keycode] & p.KEY_WAS_TRIGGERED:
                active_index = idx
                moved = True

        if ord("a") in keys and keys[ord("a")] & p.KEY_WAS_TRIGGERED:
            single_focus = False
            moved = True
        if ord("A") in keys and keys[ord("A")] & p.KEY_WAS_TRIGGERED:
            single_focus = False
            moved = True

        if ord("s") in keys and keys[ord("s")] & p.KEY_WAS_TRIGGERED:
            single_focus = True
            moved = True
        if ord("S") in keys and keys[ord("S")] & p.KEY_WAS_TRIGGERED:
            single_focus = True
            moved = True

        if moved:
            apply_visibility()

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()
