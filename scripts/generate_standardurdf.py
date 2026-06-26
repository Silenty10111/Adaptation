#!/usr/bin/env python3
"""生成标准六足机器人几何体、物理元数据及 URDF 文件。

机器人主体为长方体；六条腿均匀分布在主体较长的两侧（每侧前、中、后各一条，
包含前后端点，间距均等）。

所有文件输出到 robot_assets/standard_hexapod/ 子目录，不会覆盖
robot_assets/generated_robot.urdf（自适应机器人专用路径）。

用法示例
--------
python generate_standardurdf.py
python generate_standardurdf.py --body-length 0.60 --body-width 0.36
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
REPO_ROOT        = Path(__file__).resolve().parent.parent
ASSET_DIR_NAME   = "robot_assets"
STANDARD_SUBDIR  = "standard_hexapod"
MESH_DIR_NAME    = "meshes"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="standard_hexapod")

    # ---- 躯干尺寸 -----------------------------------------------------------
    parser.add_argument("--body-length", type=float, default=0.55,
                        help="躯干长度（前后方向，X 轴），单位 m。")
    parser.add_argument("--body-width", type=float, default=0.36,
                        help="躯干宽度（左右方向，Y 轴），单位 m。")
    parser.add_argument("--body-height", type=float, default=0.08,
                        help="躯干高度（Z 轴），仅影响外观，单位 m。蜘蛛形态推荐 0.06~0.10m。")
    parser.add_argument("--hip-depth", type=float, default=0.04,
                        help="髋关节安装点相对躯干中心的向下偏移（固定，与 body-height 解耦）。")

    # ---- 腿部尺寸 -----------------------------------------------------------
    parser.add_argument("--upper-length", type=float, default=0.38,
                        help="大腿（femur）长度，单位 m。")
    parser.add_argument("--lower-length", type=float, default=0.38,
                        help="小腿（tibia）长度，单位 m。")
    parser.add_argument("--joint-radius", type=float, default=0.035,
                        help="关节球体半径，单位 m。")
    parser.add_argument("--link-radius", type=float, default=0.018,
                        help="肢体圆柱半径，单位 m。")

    # ---- 腿部方向向量权重（蜘蛛化核心参数）------------------------------------
    # 大腿（femur）方向由三个分量线性混合后归一化：
    #   upper_dir ∝ upper_x_bias·sign(x)·X̂ + upper_y_weight·Ŷ_out + upper_z_weight·(-Ẑ)
    # 小腿（tibia）方向：
    #   lower_dir ∝ lower_x_bias·sign(x)·X̂ + lower_y_weight·Ŷ_out + 1.0·(-Ẑ)
    # 这些权重不会直接成为角度，而是在归一化前进行混合（类似 slerp 的线性近似）
    parser.add_argument("--upper-x-bias", type=float, default=0.55,
                        help="大腿 X 轴（前/后）偏置权重。前腿取正、后腿取负；"
                             "越大则前后腿越偏向前后方，X足迹跨度越大。[0.0, 1.0+]")
    parser.add_argument("--upper-y-weight", type=float, default=0.60,
                        help="大腿侧向（Y 轴）展开权重。越大则腿越水平展开。[0.3, 1.0+]")
    parser.add_argument("--upper-z-weight", type=float, default=0.08,
                        help="大腿向下（-Z）倾斜权重。0=水平，>0=略微下倾（降低膝关节高度）。[0.0, 0.4]")
    parser.add_argument("--lower-x-bias", type=float, default=0.30,
                        help="小腿 X 轴偏置权重（延续大腿的前后方向，让足端更靠前/后）。[0.0, 0.6]")
    parser.add_argument("--lower-y-weight", type=float, default=0.08,
                        help="小腿侧向（Y）权重。控制足端相对膝关节的横向偏移量。[0.0, 0.3]")

    # ---- 物理密度 -----------------------------------------------------------
    parser.add_argument("--density", type=float, default=900.0,
                        help="均匀密度，单位 kg/m³。扁平躯干需提高密度以维持 I_roll ≥ 0.12 kg·m²。")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def to_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def normalize(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    n = np.linalg.norm(arr)
    if n < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return arr / n


def _export(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _mass_props(mesh: trimesh.Trimesh, density: float) -> Dict:
    mc = mesh.copy()
    mc.density = density
    mp = mc.mass_properties
    inertia = np.asarray(mp["inertia"], dtype=float)
    return {
        "volume": float(mp["volume"]),
        "mass": float(mp["mass"]),
        "center_mass": to_list(mp["center_mass"]),
        "inertia": [[float(v) for v in row] for row in inertia],
    }


def _link_record(
    name: str,
    mesh: trimesh.Trimesh,
    density: float,
    mesh_rel: str,
    meshes_root: Path,
    leg_id: int | None,
    role: str,
    world_origin: Sequence[float],
) -> Dict:
    _export(mesh, meshes_root / Path(mesh_rel).name)
    return {
        "name": name,
        "leg_id": leg_id,
        "role": role,
        "mesh_path": mesh_rel,
        "mass_properties": _mass_props(mesh, density),
        "default_world_origin": to_list(world_origin),
    }


def _sphere(radius: float) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=2, radius=radius)


def _cylinder(start: Sequence[float], end: Sequence[float], radius: float) -> trimesh.Trimesh:
    s, e = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    d = e - s
    h = np.linalg.norm(d)
    if h < 1e-9:
        raise ValueError("Cylinder length must be > 0.")
    mesh = trimesh.creation.cylinder(radius=radius, height=h, sections=32)
    mesh.apply_transform(trimesh.geometry.align_vectors([0., 0., 1.], d))
    mesh.apply_translation((s + e) / 2.0)
    return mesh


# ---------------------------------------------------------------------------
# URDF 写出（自洽，不依赖 generate_urdf.py）
# ---------------------------------------------------------------------------

def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.9f}" for v in values)


def _color_for_role(role: str) -> Tuple[str, str]:
    return {
        "trunk":      ("trunk_material",      "0.35 0.35 0.38 1.0"),
        "upper_link": ("upper_link_material",  "0.15 0.46 0.65 1.0"),
        "lower_link": ("lower_link_material",  "0.11 0.60 0.44 1.0"),
        "foot":       ("foot_material",        "0.90 0.52 0.14 1.0"),
    }.get(role, ("joint_material", "0.72 0.72 0.72 1.0"))


def _add_link_xml(robot: ET.Element, link_data: Dict) -> None:
    el = ET.SubElement(robot, "link", name=link_data["name"])
    mp = link_data["mass_properties"]
    inertial = ET.SubElement(el, "inertial")
    ET.SubElement(inertial, "origin", xyz=_fmt(mp["center_mass"]), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{float(mp['mass']):.9f}")
    I = mp["inertia"]
    ET.SubElement(inertial, "inertia",
                  ixx=f"{I[0][0]:.9f}", ixy=f"{I[0][1]:.9f}", ixz=f"{I[0][2]:.9f}",
                  iyy=f"{I[1][1]:.9f}", iyz=f"{I[1][2]:.9f}", izz=f"{I[2][2]:.9f}")
    mat_name, rgba = _color_for_role(link_data["role"])
    for tag in ("visual", "collision"):
        sub = ET.SubElement(el, tag)
        ET.SubElement(sub, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(sub, "geometry")
        ET.SubElement(geom, "mesh", filename=link_data["mesh_path"], scale="1 1 1")
        if tag == "visual":
            mat = ET.SubElement(sub, "material", name=mat_name)
            ET.SubElement(mat, "color", rgba=rgba)


def _add_joint_xml(robot: ET.Element, j: Dict) -> None:
    el = ET.SubElement(robot, "joint", name=j["name"], type=j["type"])
    ET.SubElement(el, "parent", link=j["parent"])
    ET.SubElement(el, "child",  link=j["child"])
    ET.SubElement(el, "origin", xyz=_fmt(j["origin"]["xyz"]), rpy=_fmt(j["origin"]["rpy"]))
    if j["type"] != "fixed":
        ET.SubElement(el, "axis", xyz=_fmt(j["axis"]))
        lim = j["limit"]
        ET.SubElement(el, "limit",
                      lower=f"{lim['lower']:.9f}", upper=f"{lim['upper']:.9f}",
                      effort=f"{lim['effort']:.9f}", velocity=f"{lim['velocity']:.9f}")
        dyn = j.get("dynamics")
        if dyn:
            ET.SubElement(el, "dynamics",
                          damping=f"{dyn['damping']:.9f}", friction=f"{dyn['friction']:.9f}")


def write_urdf(metadata: Dict, urdf_path: Path, desc_path: Path) -> None:
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    robot = ET.Element("robot", name=metadata["robot_name"])
    for link_data in metadata["links"]:
        _add_link_xml(robot, link_data)
    for joint_data in metadata["joints"]:
        _add_joint_xml(robot, joint_data)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    # 将 urdf_path 写回 json
    metadata["urdf_path"] = str(urdf_path.relative_to(REPO_ROOT))
    desc_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[URDF] 写出: {urdf_path}")


# ---------------------------------------------------------------------------
# 标准六足几何体生成
# ---------------------------------------------------------------------------

def generate(args: argparse.Namespace) -> Dict:
    """生成几何体、JSON 描述及 URDF，全部输出到 standard_hexapod/ 子目录。

    Returns
    -------
    dict  包含 metadata 及几何评估指标（x_span, y_span, xy_ratio, i_roll, total_mass）。
    """

    # 输出路径
    assets_root = REPO_ROOT / ASSET_DIR_NAME / STANDARD_SUBDIR
    meshes_root = assets_root / MESH_DIR_NAME
    meshes_root.mkdir(parents=True, exist_ok=True)

    BL   = args.body_length    # 躯干长（X）
    BW   = args.body_width     # 躯干宽（Y）
    BH   = args.body_height    # 躯干高（Z，仅影响外观）
    UL   = args.upper_length
    LL   = args.lower_length
    JR   = args.joint_radius
    LR   = args.link_radius
    rho  = args.density

    half_l = BL / 2.0
    half_w = BW / 2.0
    # hip_z 固定为 -hip_depth，与 body_height 解耦，保证腿部运动学不随躯干薄厚改变
    hip_z  = -args.hip_depth   # 默认 -0.075m（旧设计值），不随 BH 变动

    # ------------------------------------------------------------------ 躯干
    # 长方体：以 XY 中心为原点，底面在 -BH/2，顶面在 +BH/2
    trunk_box = trimesh.creation.box(extents=[BL, BW, BH])
    trunk_rel = f"{MESH_DIR_NAME}/trunk.stl"

    links: List[Dict] = []
    joints: List[Dict] = []
    assembled: List[trimesh.Trimesh] = []

    trunk_record = _link_record(
        "base_link", trunk_box, rho, trunk_rel, meshes_root,
        leg_id=None, role="trunk", world_origin=[0.0, 0.0, 0.0],
    )
    links.append(trunk_record)
    assembled.append(trunk_box.copy())

    # ------------------------------------------------------------------ 腿位
    # 6 条腿：左侧（Y = +half_w）3 条，右侧（Y = -half_w）3 条。
    # 沿 X 轴均匀分布：前（+half_l）、中（0）、后（-half_l）。
    # 腿编号：从右前开始，按逆时针排序，最后是左前。
    num_legs = 6
    leg_defs: List[Dict] = []
    ordered_positions = [
        (half_l, -1.0),   # 右前
        (0.0, -1.0),      # 右中
        (-half_l, -1.0),  # 右后
        (-half_l, 1.0),   # 左后
        (0.0, 1.0),       # 左中
        (half_l, 1.0),    # 左前
    ]
    for leg_id, (x, side) in enumerate(ordered_positions):
        leg_defs.append({"leg_id": leg_id, "x": x, "side": side})

    for leg_def in leg_defs:
        leg_index = leg_def["leg_id"]
        x_pos     = leg_def["x"]
        side      = leg_def["side"]    # +1 = 左侧(Y+)，-1 = 右侧(Y-)

        # 髋关节安装点（躯干侧面中点高度）
        attach = np.array([x_pos, side * half_w, hip_z], dtype=float)

        # ---- 腿部方向向量（参数化设计）-----------------------------------------
        # 使用 CLI 参数控制各分量权重，权重意义见 parse_args 注释。
        outward_y = np.array([0.0, side, 0.0], dtype=float)
        down      = np.array([0.0, 0.0, -1.0], dtype=float)
        forward   = np.array([1.0, 0.0, 0.0], dtype=float)

        # 前腿 (x_sign=+1) 偏前，后腿 (x_sign=-1) 偏后，中腿 (x_sign=0) 纯侧向
        x_sign = float(np.sign(x_pos))

        # 大腿方向：X 前后偏置 + Y 侧向展开 + Z 轻微下倾（均由参数控制）
        upper_dir = normalize(
            x_sign * args.upper_x_bias  * forward
            + args.upper_y_weight * outward_y
            + args.upper_z_weight * down
        )
        # 小腿方向：主要向下（权重固定=1.0）+ 参数化的 X/Y 分量
        lower_dir = normalize(
            x_sign * args.lower_x_bias  * forward
            + args.lower_y_weight * outward_y
            + 1.00 * down
        )

        upper_vec   = upper_dir * UL
        lower_vec   = lower_dir * LL
        knee_world  = attach + upper_vec
        foot_world  = knee_world + lower_vec

        # 旋转轴：lift（抬腿）绕 X 轴；swing（前后摆）绕 Z 轴
        lift_axis   = np.array([side, 0.0, 0.0], dtype=float)
        swing_axis  = np.array([0.0, 0.0, 1.0], dtype=float)

        # 命名
        hip_name    = f"leg_{leg_index}_hip"
        swing_name  = f"leg_{leg_index}_swing_node"
        upper_name  = f"leg_{leg_index}_upper"
        knee_name   = f"leg_{leg_index}_knee"
        lower_name  = f"leg_{leg_index}_lower"
        foot_name   = f"leg_{leg_index}_foot"

        # 网格
        hip_mesh    = _sphere(JR)
        swing_mesh  = _sphere(JR * 0.65)
        upper_mesh  = _cylinder([0., 0., 0.], upper_vec, LR)
        knee_mesh   = _sphere(JR * 0.90)
        lower_mesh  = _cylinder([0., 0., 0.], lower_vec, LR * 0.92)
        foot_mesh   = _sphere(JR * 0.75)

        leg_links = [
            (hip_name,   hip_mesh,   "joint_sphere", attach),
            (swing_name, swing_mesh, "joint_sphere", attach),
            (upper_name, upper_mesh, "upper_link",   attach),
            (knee_name,  knee_mesh,  "joint_sphere", knee_world),
            (lower_name, lower_mesh, "lower_link",   knee_world),
            (foot_name,  foot_mesh,  "foot",         foot_world),
        ]
        for link_name, mesh, role, world_origin in leg_links:
            rel = f"{MESH_DIR_NAME}/{link_name}.stl"
            rec = _link_record(link_name, mesh, rho, rel, meshes_root,
                               leg_id=leg_index, role=role, world_origin=world_origin)
            links.append(rec)
            wm = mesh.copy()
            wm.apply_translation(world_origin)
            assembled.append(wm)

        joints.extend([
            {
                "name": f"leg_{leg_index}_mount",
                "type": "fixed",
                "parent": "base_link",
                "child": hip_name,
                "origin": {"xyz": to_list(attach), "rpy": [0., 0., 0.]},
            },
            {
                "name": f"leg_{leg_index}_lift",
                "type": "revolute",
                "parent": hip_name,
                "child": swing_name,
                "origin": {"xyz": [0., 0., 0.], "rpy": [0., 0., 0.]},
                "axis": to_list(lift_axis),
                "limit": {"lower": -0.80, "upper": 0.95, "effort": 80.0, "velocity": 2.5},
                "dynamics": {"damping": 0.2, "friction": 0.05},
            },
            {
                "name": f"leg_{leg_index}_swing",
                "type": "revolute",
                "parent": swing_name,
                "child": upper_name,
                "origin": {"xyz": [0., 0., 0.], "rpy": [0., 0., 0.]},
                "axis": to_list(swing_axis),
                "limit": {"lower": -0.55, "upper": 0.55, "effort": 60.0, "velocity": 2.0},
                "dynamics": {"damping": 0.15, "friction": 0.04},
            },
            {
                "name": f"leg_{leg_index}_knee_mount",
                "type": "fixed",
                "parent": upper_name,
                "child": knee_name,
                "origin": {"xyz": to_list(upper_vec), "rpy": [0., 0., 0.]},
            },
            {
                "name": f"leg_{leg_index}_drop",
                "type": "revolute",
                "parent": knee_name,
                "child": lower_name,
                "origin": {"xyz": [0., 0., 0.], "rpy": [0., 0., 0.]},
                "axis": to_list(lift_axis),
                "limit": {"lower": -0.10, "upper": 1.10, "effort": 50.0, "velocity": 2.3},
                "dynamics": {"damping": 0.12, "friction": 0.03},
            },
            {
                "name": f"leg_{leg_index}_foot_mount",
                "type": "fixed",
                "parent": lower_name,
                "child": foot_name,
                "origin": {"xyz": to_list(lower_vec), "rpy": [0., 0., 0.]},
            },
        ])

    # ----------------------------------------------------------------- 预览网格
    preview = trimesh.util.concatenate(assembled)
    _export(preview, meshes_root / "assembled_preview.stl")

    # ----------------------------------------------------------------- 躯干轮廓（矩形）
    poly_xy = [
        [ half_l, -half_w], [ half_l,  half_w],
        [-half_l,  half_w], [-half_l, -half_w],
    ]

    # ----------------------------------------------------------------- JSON 描述
    metadata: Dict = {
        "robot_name":       args.robot_name,
        "seed":             0,
        "density":          float(rho),
        "num_legs":         num_legs,
        "asset_root":       f"{ASSET_DIR_NAME}/{STANDARD_SUBDIR}",
        "mesh_root":        MESH_DIR_NAME,
        "preview_mesh":     f"{MESH_DIR_NAME}/assembled_preview.stl",
        "trunk_polygon_xy": poly_xy,
        "trunk_bottom_z":   hip_z,
        "links":            links,
        "joints":           joints,
    }

    desc_path = assets_root / "robot_description.json"
    desc_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[GEO] 描述文件写出: {desc_path}")

    # ----------------------------------------------------------------- URDF
    urdf_path = assets_root / "generated_robot.urdf"
    write_urdf(metadata, urdf_path, desc_path)

    print(f"\n生成完毕 → {assets_root}")
    print(f"  leg 数量 : {num_legs}")
    total_mass = sum(lk["mass_properties"]["mass"] for lk in links)
    print(f"  总质量   : {total_mass:.3f} kg")

    # ---- 几何评估指标（供寻优脚本读取）---------------------------------------
    metrics: Dict = {"total_mass": float(total_mass)}
    foot_links = [lk for lk in links if lk["role"] == "foot"]
    if foot_links:
        fxs = [lk["default_world_origin"][0] for lk in foot_links]
        fys = [lk["default_world_origin"][1] for lk in foot_links]
        fzs = [lk["default_world_origin"][2] for lk in foot_links]
        x_span = max(fxs) - min(fxs)
        y_span = max(fys) - min(fys)
        xy_ratio = x_span / max(y_span, 1e-9)
        avg_foot_z = float(np.mean(fzs))
        metrics.update({
            "x_span": float(x_span),
            "y_span": float(y_span),
            "xy_ratio": float(xy_ratio),
            "avg_foot_z": float(avg_foot_z),
        })
        print(f"  足端X跨度: {x_span:.3f} m")
        print(f"  足端Y跨度: {y_span:.3f} m")
        print(f"  X/Y比值  : {xy_ratio:.3f}  ({'≥ 1.0 ✓' if xy_ratio >= 1.0 else '< 1.0 ✗  请加大 upper-x-bias 或缩小 body-width'})")
    trunk_link = next((lk for lk in links if lk["role"] == "trunk"), None)
    if trunk_link:
        I = trunk_link["mass_properties"]["inertia"]
        i_roll = float(I[0][0])
        metrics["i_roll"] = i_roll
        print(f"  I_roll   : {i_roll:.4f} kg·m²  ({'≥ 0.12 ✓' if i_roll >= 0.12 else '< 0.12 ✗ 请增大 density 或 body_width'})")

    metrics["metadata"] = metadata
    return metrics


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def generate_from_dict(params: Dict) -> Dict:
    """程序化接口：传入参数字典（key 对应 CLI 长参数名去掉 '--' 并将 '-' 换成 '_'）。

    适合被 search_morphology.py 批量调用，不依赖 sys.argv。
    缺失参数自动使用默认值。
    """
    import types
    defaults = parse_args.__defaults__  # noqa: not used, just for documentation
    # Build a Namespace by parsing an empty list (gets all defaults) then overwrite
    dummy_argv = []
    for key, val in params.items():
        cli_key = "--" + key.replace("_", "-")
        dummy_argv.extend([cli_key, str(val)])
    import sys as _sys
    _old = _sys.argv
    _sys.argv = ["generate_standardurdf.py"] + dummy_argv
    try:
        ns = parse_args()
    finally:
        _sys.argv = _old
    return generate(ns)


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
