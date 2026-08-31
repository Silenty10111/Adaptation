# pyright: reportGeneralTypeIssues=false
#!/usr/bin/env python3
"""Generate a generalized multi-legged robot geometry and physical metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


ASSET_DIR_NAME = "robot_assets"
MESH_DIR_NAME = "meshes"

# Default random-morphology profile.  Keep these values centralized so batch
# generation, CLI help and regression tests cannot silently drift apart.
DEFAULT_BODY_LENGTH = 0.90
DEFAULT_BODY_WIDTH = 0.32
DEFAULT_BODY_HEIGHT = 0.18
MIN_EXPLICIT_LEGS = 4
MAX_EXPLICIT_LEGS = 14
DEFAULT_LEG_POOL = np.array([8, 9, 10, 11, 12, 13, 14], dtype=int)
DEFAULT_LEG_PROBABILITIES = np.array(
    [0.05, 0.10, 0.20, 0.25, 0.20, 0.12, 0.08], dtype=float,
)
DEFAULT_LEG_PROBABILITIES /= DEFAULT_LEG_PROBABILITIES.sum()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="generated_multileg")
    parser.add_argument(
        "--num-legs", type=int, default=None,
        help=("When omitted, sample the elongated-body profile over [8,14] "
              "(peak at 10-12); explicit values in [4,14] remain supported."),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--leg-placement", choices=("uniform", "random"), default="random")
    parser.add_argument(
        "--leg-style",
        choices=("swing", "pendulum", "mixed"),
        default="mixed",
        help="Swing legs allow larger lateral sweep; pendulum legs stay close to a single plane.",
    )
    parser.add_argument("--body-length", type=float, default=DEFAULT_BODY_LENGTH)
    parser.add_argument("--body-width", type=float, default=DEFAULT_BODY_WIDTH)
    parser.add_argument("--body-height", type=float, default=DEFAULT_BODY_HEIGHT)
    parser.add_argument(
        "--morphology-type",
        choices=("irregular_rigid", "serial_rigid", "serial_flexible"),
        default="irregular_rigid",
        help=("Rigid irregular outline, rigid serial-module outline, or the "
              "reserved (not yet implemented) flexible serial interface."),
    )
    parser.add_argument(
        "--allow-unstable", action="store_true",
        help="Export even when the pre-export SSM gate fails (research/debug only).",
    )
    parser.add_argument("--ssm-threshold", type=float, default=0.03)
    parser.add_argument("--upper-length", type=float, default=0.28)
    parser.add_argument("--lower-length", type=float, default=0.30)
    parser.add_argument("--joint-radius", type=float, default=0.035)
    parser.add_argument("--link-radius", type=float, default=0.018)
    parser.add_argument("--density", type=float, default=300.0, help="Uniform density in kg/m^3.")
    return parser.parse_args()


def to_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def normalize(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return array / norm


def make_assets_root() -> Tuple[Path, Path]:
    assets_root = Path.cwd() / ASSET_DIR_NAME
    meshes_root = assets_root / MESH_DIR_NAME
    meshes_root.mkdir(parents=True, exist_ok=True)
    return assets_root, meshes_root


def polygon_has_concavity(polygon: Polygon, tolerance: float = 1e-9) -> bool:
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=float)
    if len(coords) < 4:
        return False

    for idx in range(len(coords)):
        prev_pt = coords[idx - 1]
        curr_pt = coords[idx]
        next_pt = coords[(idx + 1) % len(coords)]
        edge_a = curr_pt - prev_pt
        edge_b = next_pt - curr_pt
        cross_z = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
        if cross_z < -tolerance:
            return True

    return False


def create_irregular_trunk_polygon(length: float, width: float, rng: np.random.Generator) -> Polygon:
    """Create a noticeably irregular trunk outline with controlled concave notches."""
    half_length = length / 2.0
    half_width = width / 2.0

    for _ in range(24):
        vertex_count = int(rng.integers(13, 20))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=vertex_count))
        base_x = half_length * rng.uniform(0.82, 1.08)
        base_y = half_width * rng.uniform(0.82, 1.08)
        radial_noise = rng.uniform(0.78, 1.24, size=vertex_count)
        wobble = 1.0 + 0.14 * np.sin(3.0 * angles + rng.uniform(0.0, 2.0 * np.pi))
        wobble += 0.08 * np.cos(5.0 * angles + rng.uniform(0.0, 2.0 * np.pi))
        radii = radial_noise * wobble

        # Inject one or more inward notches so the trunk is not purely convex.
        notch_count = int(rng.integers(1, 4))
        notch_indices = rng.choice(vertex_count, size=notch_count, replace=False)
        for notch_idx in np.atleast_1d(notch_indices):
            notch_scale = float(rng.uniform(0.35, 0.62))
            radii[notch_idx] *= notch_scale
            left_idx = (int(notch_idx) - 1) % vertex_count
            right_idx = (int(notch_idx) + 1) % vertex_count
            shoulder_scale = float(rng.uniform(0.72, 0.88))
            radii[left_idx] *= shoulder_scale
            radii[right_idx] *= shoulder_scale

        min_radius = 0.22
        radii = np.clip(radii, min_radius, None)

        vertices = np.column_stack(
            [
                np.cos(angles) * base_x * radii,
                np.sin(angles) * base_y * radii,
            ]
        )
        vertices[:, 0] += rng.uniform(-length * 0.03, length * 0.03)
        vertices[:, 1] += rng.uniform(-width * 0.03, width * 0.03)

        polygon = orient(Polygon(vertices), sign=1.0)
        if (
            polygon.is_valid
            and polygon.area > length * width * 0.18
            and len(polygon.exterior.coords) >= 8
            and polygon_has_concavity(polygon)
        ):
            # The CLI dimensions describe the final trunk envelope, not an
            # ellipse scale that random radial noise may exceed.  Normalize
            # each accepted outline to the requested, centred AABB.
            min_x, min_y, max_x, max_y = polygon.bounds
            vertices[:, 0] = (vertices[:, 0] - 0.5 * (min_x + max_x)) * (
                length / max(max_x - min_x, 1e-12)
            )
            vertices[:, 1] = (vertices[:, 1] - 0.5 * (min_y + max_y)) * (
                width / max(max_y - min_y, 1e-12)
            )
            normalized = orient(Polygon(vertices), sign=1.0)
            if normalized.is_valid:
                return normalized

    raise RuntimeError("Failed to generate a valid trunk polygon.")


def create_serial_rigid_trunk_polygon(length: float, width: float) -> Polygon:
    """Create a single rigid three-module serial outline with an exact AABB.

    This reserves serial morphology semantics without pretending that the
    current rigid mesh has flexible inter-module dynamics.
    """
    half_l = length / 2.0
    half_w = width / 2.0
    neck_x = length / 6.0
    neck_w = width * 0.32
    vertices = [
        (-half_l, -half_w), (-neck_x, -half_w), (-neck_x, -neck_w),
        (neck_x, -neck_w), (neck_x, -half_w), (half_l, -half_w),
        (half_l, half_w), (neck_x, half_w), (neck_x, neck_w),
        (-neck_x, neck_w), (-neck_x, half_w), (-half_l, half_w),
    ]
    return orient(Polygon(vertices), sign=1.0)


def extrude_trunk_mesh(polygon: Polygon, body_height: float) -> trimesh.Trimesh:
    trunk_mesh = trimesh.creation.extrude_polygon(polygon, height=body_height, engine="earcut")
    trunk_mesh.apply_translation([0.0, 0.0, -body_height / 2.0])
    trunk_mesh.process(validate=True)
    return trunk_mesh


def compute_mount_points(
    polygon: Polygon,
    count: int,
    placement: str,
    rng: np.random.Generator,
    upper_length: float = 0.28,
) -> List[Dict[str, List[float]]]:
    coords = np.asarray(polygon.exterior.coords[:-1], dtype=float)
    segments: List[Tuple[np.ndarray, np.ndarray, float, float, float]] = []
    cumulative = 0.0
    for index in range(len(coords)):
        start = coords[index]
        end = coords[(index + 1) % len(coords)]
        length = np.linalg.norm(end - start)
        segments.append((start, end, length, cumulative, cumulative + length))
        cumulative += length

    if placement == "uniform":
        distances = np.linspace(0.0, cumulative, count, endpoint=False) + cumulative / (2.0 * count)
    else:
        # Construct *circular* gaps directly.  The previous forward-only
        # adjustment did not check the wrap-around gap between the last and
        # first hip, so high-leg-count models could contain a near-coincident
        # pair at the perimeter seam.
        min_gap = min(float(upper_length), 0.80 * cumulative / count)
        residual = max(cumulative - count * min_gap, 0.0)
        random_gaps = rng.dirichlet(np.ones(count)) * residual
        gaps = min_gap + random_gaps
        origin = float(rng.uniform(0.0, cumulative))
        distances = np.mod(origin + np.r_[0.0, np.cumsum(gaps[:-1])], cumulative)
        distances.sort()

    centroid = np.array([polygon.centroid.x, polygon.centroid.y], dtype=float)
    mount_data: List[Dict[str, List[float]]] = []

    for distance in distances:
        for start, end, seg_length, seg_begin, seg_end in segments:
            if distance <= seg_end or np.isclose(distance, cumulative):
                ratio = (distance - seg_begin) / seg_length if seg_length > 1e-9 else 0.0
                point = start + ratio * (end - start)
                tangent = normalize(end - start)
                inward = normalize(centroid - point)
                mount_data.append(
                    {
                        "point_xy": to_list(point),
                        "tangent_xy": to_list(tangent),
                        "inward_xy": to_list(inward),
                    }
                )
                break

    return mount_data


def create_sphere_mesh(radius: float) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=2, radius=radius)


def create_cylinder_mesh(start: Sequence[float], end: Sequence[float], radius: float) -> trimesh.Trimesh:
    start_vec = np.asarray(start, dtype=float)
    end_vec = np.asarray(end, dtype=float)
    direction = end_vec - start_vec
    height = np.linalg.norm(direction)
    if height < 1e-9:
        raise ValueError("Cylinder height must be positive.")

    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    mesh.apply_transform(transform)
    mesh.apply_translation((start_vec + end_vec) / 2.0)
    return mesh


def compute_mass_properties(mesh: trimesh.Trimesh, density: float) -> Dict[str, object]:
    mesh_copy = mesh.copy()
    mesh_copy.density = density
    properties = mesh_copy.mass_properties
    inertia = np.asarray(properties["inertia"], dtype=float)
    return {
        "volume": float(properties["volume"]),
        "mass": float(properties["mass"]),
        "center_mass": to_list(properties["center_mass"]),
        "inertia": [[float(value) for value in row] for row in inertia],
    }


def export_mesh(mesh: trimesh.Trimesh, mesh_path: Path) -> None:
    mesh.export(mesh_path)


def link_record(
    name: str,
    mesh: trimesh.Trimesh,
    density: float,
    mesh_relative_path: str,
    meshes_root: Path,
    leg_id: int | None,
    role: str,
    world_origin: Sequence[float],
) -> Dict[str, object]:
    export_mesh(mesh, meshes_root / Path(mesh_relative_path).name)
    return {
        "name": name,
        "leg_id": leg_id,
        "role": role,
        "mesh_path": mesh_relative_path,
        "mass_properties": compute_mass_properties(mesh, density),
        "default_world_origin": to_list(world_origin),
    }


def build_leg_vectors(
    upper_length: float,
    lower_length: float,
    inward_xy: Sequence[float],
    tangent_xy: Sequence[float],
    leg_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    inward = np.array([inward_xy[0], inward_xy[1], 0.0], dtype=float)
    outward = -inward
    tangent = np.array([tangent_xy[0], tangent_xy[1], 0.0], dtype=float)
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    side = rng.choice([-1.0, 1.0])

    if leg_type == "swing":
        upper_direction = normalize(0.90 * outward + 0.30 * side * tangent + 0.12 * down)
        lower_direction = normalize(0.16 * outward - 0.12 * side * tangent + 1.00 * down)
    else:
        upper_direction = normalize(0.94 * outward + 0.18 * side * tangent + 0.08 * down)
        lower_direction = normalize(0.10 * outward + 0.08 * side * tangent + 1.00 * down)

    return upper_direction * upper_length, lower_direction * lower_length


def coplanarize_lower_leg_vectors(
    knee_positions: Sequence[Sequence[float]],
    lower_vectors: Sequence[Sequence[float]],
) -> Tuple[List[np.ndarray], float]:
    """Adjust lower-leg directions so all nominal feet share one Z plane.

    Link lengths and each lower leg's horizontal direction are preserved.  The
    common plane is chosen near the median original foot height and clipped to
    the intersection of every leg's reachable vertical interval.
    """
    knees = [np.asarray(value, dtype=float) for value in knee_positions]
    lowers = [np.asarray(value, dtype=float) for value in lower_vectors]
    if len(knees) != len(lowers) or not knees:
        raise ValueError("knee_positions and lower_vectors must be non-empty and aligned")
    lengths = [float(np.linalg.norm(value)) for value in lowers]
    preferred = float(np.median([
        knee[2] + lower[2] for knee, lower in zip(knees, lowers)
    ]))
    lowest_common = max(knee[2] - length for knee, length in zip(knees, lengths))
    highest_common = min(knee[2] + length for knee, length in zip(knees, lengths))
    if lowest_common > highest_common + 1e-9:
        raise ValueError("lower-leg lengths have no common reachable foot plane")
    target_z = float(np.clip(preferred, lowest_common, highest_common))

    adjusted: List[np.ndarray] = []
    for knee, lower, length in zip(knees, lowers, lengths):
        vertical = float(np.clip(target_z - knee[2], -length, length))
        horizontal_length = math.sqrt(max(length * length - vertical * vertical, 0.0))
        horizontal = lower[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= 1e-12:
            horizontal_direction = np.array([1.0, 0.0], dtype=float)
        else:
            horizontal_direction = horizontal / horizontal_norm
        adjusted.append(np.array([
            horizontal_direction[0] * horizontal_length,
            horizontal_direction[1] * horizontal_length,
            vertical,
        ], dtype=float))
    return adjusted, target_z


def select_leg_type(requested_style: str, leg_index: int, rng: np.random.Generator) -> str:
    if requested_style == "mixed":
        return "swing" if (leg_index % 2 == 0 or rng.random() > 0.5) else "pendulum"
    return requested_style


def validate_foot_layout_ssm(
    foot_positions_xy: List[List[float]],
    trunk_com_xy: Sequence[float] = (0.0, 0.0),
    threshold: float = 0.03,
) -> Dict[str, object]:
    """SSM pre-check using foot XY positions before any mesh file is written.

    Called after all leg vectors are computed but before disk I/O, so a failed
    check can be detected early.  A warning is printed when SSM is below the
    threshold, but generation is **not** aborted — use generate_urdf.py's
    check for a hard gate before URDF export.

    Parameters
    ----------
    foot_positions_xy : list of [x, y]  World-frame foot contact XY positions.
    trunk_com_xy      : [x, y]          Trunk CoM estimate. Default [0,0]; for a
                                        more realistic estimate, use the centroid
                                        of foot positions (legs shift CoM outward).
    threshold         : float           Minimum acceptable SSM in metres.
                                        Default 0.03 to flag unstable layouts early.
    """
    try:
        from adaptation.stability import compute_ssm
    except ImportError:
        print("[SSM] stability 模块未找到，跳过足端布局预检。")
        return {"measured": False, "passed": None, "ssm": None,
                "reason": "adaptation.stability unavailable"}

    def ensure_ccw(polygon_xy: np.ndarray) -> np.ndarray:
        if len(polygon_xy) < 3:
            return polygon_xy
        x = polygon_xy[:, 0]
        y = polygon_xy[:, 1]
        area = 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
        if area < 0.0:
            return polygon_xy[::-1].copy()
        return polygon_xy

    pts = np.array(foot_positions_xy, dtype=float)

    # Better CoM estimate: average of trunk + foot centroid.
    # Pure trunk (0,0) underestimates CoM shift from leg masses.
    trunk_com = np.array(trunk_com_xy, dtype=float)
    foot_centroid = pts.mean(axis=0)
    # 70% trunk + 30% feet — legs typically contribute ~30% of total mass
    com_est = 0.70 * trunk_com + 0.30 * foot_centroid

    # Build CCW convex hull of foot positions
    try:
        from shapely.geometry import MultiPoint
        from shapely.geometry import Polygon as ShapelyPolygon

        hull_geom = MultiPoint([tuple(p) for p in pts]).convex_hull
        if isinstance(hull_geom, ShapelyPolygon):
            polygon_xy = ensure_ccw(np.array(hull_geom.exterior.coords[:-1], dtype=float))
        else:
            polygon_xy = pts
    except ImportError:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        polygon_xy = ensure_ccw(pts[np.argsort(angles)])

    ssm = compute_ssm(polygon_xy, com_est)
    status = "PASS" if ssm >= threshold else "FAIL"
    print(f"[SSM] 足端布局预检（生成前）  SSM = {ssm:.4f} m  [{status}]")
    print(f"[SSM]   足端数量 = {len(pts)}，CoM 估算 = {list(com_est)}")

    if ssm < threshold:
        print(
            f"\n[SSM] 足端布局静态稳定性检验不通过：SSM = {ssm:.4f} m < 阈值 {threshold} m\n"
            "      默认将中止导出；仅显式 --allow-unstable 可生成诊断资产。"
        )
    return {"measured": True, "passed": bool(ssm >= threshold),
            "ssm": float(ssm), "threshold": float(threshold)}


def print_link_summary(link: Dict[str, object]) -> None:
    props = link["mass_properties"]
    inertia = np.asarray(props["inertia"], dtype=float)
    print(f"[{link['name']}] role={link['role']} mesh={link['mesh_path']}")
    print(f"  volume={props['volume']:.6f} m^3")
    print(f"  mass={props['mass']:.6f} kg")
    print(f"  center_of_mass={props['center_mass']}")
    print("  inertia_tensor_kg_m2=")
    for row in inertia:
        print(f"    {row.tolist()}")


def assemble_robot(args: argparse.Namespace) -> Dict[str, object]:
    rng = np.random.default_rng(args.seed)
    # Batch generation now targets elongated many-legged morphologies.  Low
    # counts remain explicitly available for standard/amputation experiments,
    # but are no longer drawn by the default random profile.
    num_legs = (
        int(args.num_legs)
        if args.num_legs is not None
        else int(rng.choice(DEFAULT_LEG_POOL, p=DEFAULT_LEG_PROBABILITIES))
    )
    if num_legs < MIN_EXPLICIT_LEGS or num_legs > MAX_EXPLICIT_LEGS:
        raise ValueError(
            f"num_legs must be within [{MIN_EXPLICIT_LEGS}, {MAX_EXPLICIT_LEGS}]."
        )
    if args.body_length <= 0.0 or args.body_width <= 0.0 or args.body_height <= 0.0:
        raise ValueError("body dimensions must be positive")
    if args.morphology_type == "serial_flexible":
        raise NotImplementedError(
            "serial_flexible is a reserved interface; flexible trunk dynamics "
            "have not been validated in this project"
        )

    assets_root, meshes_root = make_assets_root()
    if args.morphology_type == "serial_rigid":
        polygon = create_serial_rigid_trunk_polygon(args.body_length, args.body_width)
    else:
        polygon = create_irregular_trunk_polygon(args.body_length, args.body_width, rng)
    trunk_mesh = extrude_trunk_mesh(polygon, args.body_height)

    links: List[Dict[str, object]] = []
    joints: List[Dict[str, object]] = []
    assembled_meshes: List[trimesh.Trimesh] = []

    mount_points = compute_mount_points(polygon, num_legs, args.leg_placement, rng, args.upper_length)
    hip_z = -args.body_height / 2.0

    # ---- Phase 1: compute all leg geometry — no disk I/O, rng consumed here ----
    leg_specs: List[Dict[str, object]] = []
    for leg_index, mount in enumerate(mount_points):
        point_xy = mount["point_xy"]
        tangent_xy = mount["tangent_xy"]
        inward_xy = mount["inward_xy"]
        leg_type = select_leg_type(args.leg_style, leg_index, rng)
        upper_vector, lower_vector = build_leg_vectors(
            args.upper_length,
            args.lower_length,
            inward_xy,
            tangent_xy,
            leg_type,
            rng,
        )
        outward_xy = -np.asarray(inward_xy, dtype=float)
        attach_clearance = args.joint_radius * 0.55 + args.link_radius * 0.35
        attach_xy = np.asarray(point_xy, dtype=float) + outward_xy * attach_clearance
        attach = np.array([attach_xy[0], attach_xy[1], hip_z], dtype=float)
        knee_world = attach + upper_vector
        foot_world = knee_world + lower_vector
        lift_axis = normalize([tangent_xy[0], tangent_xy[1], 0.0])
        swing_limits = [-0.55, 0.55] if leg_type == "swing" else [-0.12, 0.12]
        leg_specs.append({
            "leg_index": leg_index,
            "leg_type": leg_type,
            "attach": attach,
            "upper_vector": upper_vector,
            "lower_vector": lower_vector,
            "knee_world": knee_world,
            "foot_world": foot_world,
            "lift_axis": lift_axis,
            "swing_limits": swing_limits,
        })

    # Mixed swing/pendulum directions previously produced nominal foot heights
    # differing by several centimetres.  Make the zero-angle stance physically
    # realizable before evaluating SSM or exporting meshes.
    adjusted_lowers, nominal_foot_plane_z = coplanarize_lower_leg_vectors(
        [spec["knee_world"] for spec in leg_specs],
        [spec["lower_vector"] for spec in leg_specs],
    )
    for spec, adjusted_lower in zip(leg_specs, adjusted_lowers):
        spec["lower_vector"] = adjusted_lower
        spec["foot_world"] = spec["knee_world"] + adjusted_lower

    # ---- Phase 2: SSM pre-check (before any file is written to disk) ----
    foot_positions_xy = [spec["foot_world"][:2].tolist() for spec in leg_specs]  # type: ignore[index]
    layout_ssm = validate_foot_layout_ssm(
        foot_positions_xy, trunk_com_xy=[0.0, 0.0],
        threshold=float(args.ssm_threshold),
    )
    if layout_ssm.get("passed") is False and not args.allow_unstable:
        raise ValueError(
            f"foot layout SSM {layout_ssm['ssm']:.4f} m is below "
            f"threshold {layout_ssm['threshold']:.4f} m; use --allow-unstable "
            "only for an explicitly labelled diagnostic asset"
        )

    # ---- Phase 3: export trunk mesh then all leg meshes ----
    trunk_relative_mesh = f"{MESH_DIR_NAME}/trunk.stl"
    base_link = link_record(
        name="base_link",
        mesh=trunk_mesh,
        density=args.density,
        mesh_relative_path=trunk_relative_mesh,
        meshes_root=meshes_root,
        leg_id=None,
        role="trunk",
        world_origin=[0.0, 0.0, 0.0],
    )
    links.append(base_link)
    assembled_meshes.append(trunk_mesh.copy())

    for spec in leg_specs:
        leg_index = spec["leg_index"]
        attach = spec["attach"]
        upper_vector = spec["upper_vector"]
        lower_vector = spec["lower_vector"]
        knee_world = spec["knee_world"]
        foot_world = spec["foot_world"]
        lift_axis = spec["lift_axis"]
        swing_limits = spec["swing_limits"]
        swing_axis = np.array([0.0, 0.0, 1.0], dtype=float)

        hip_name = f"leg_{leg_index}_hip"
        swing_name = f"leg_{leg_index}_swing_node"
        upper_name = f"leg_{leg_index}_upper"
        knee_name = f"leg_{leg_index}_knee"
        lower_name = f"leg_{leg_index}_lower"
        foot_name = f"leg_{leg_index}_foot"

        hip_mesh = create_sphere_mesh(args.joint_radius)
        swing_mesh = create_sphere_mesh(args.joint_radius * 0.65)
        upper_mesh = create_cylinder_mesh([0.0, 0.0, 0.0], upper_vector, args.link_radius)
        knee_mesh = create_sphere_mesh(args.joint_radius * 0.90)
        lower_mesh = create_cylinder_mesh([0.0, 0.0, 0.0], lower_vector, args.link_radius * 0.92)
        foot_mesh = create_sphere_mesh(args.joint_radius * 0.75)

        leg_links = [
            (hip_name, hip_mesh, "joint_sphere", attach),
            (swing_name, swing_mesh, "joint_sphere", attach),
            (upper_name, upper_mesh, "upper_link", attach),
            (knee_name, knee_mesh, "joint_sphere", knee_world),
            (lower_name, lower_mesh, "lower_link", knee_world),
            (foot_name, foot_mesh, "foot", foot_world),
        ]

        for link_name, mesh, role, world_origin in leg_links:
            relative_mesh = f"{MESH_DIR_NAME}/{link_name}.stl"
            record = link_record(
                name=link_name,
                mesh=mesh,
                density=args.density,
                mesh_relative_path=relative_mesh,
                meshes_root=meshes_root,
                leg_id=leg_index,
                role=role,
                world_origin=world_origin,
            )
            links.append(record)
            world_mesh = mesh.copy()
            world_mesh.apply_translation(world_origin)
            assembled_meshes.append(world_mesh)

        joints.extend(
            [
                {
                    "name": f"leg_{leg_index}_mount",
                    "type": "fixed",
                    "parent": "base_link",
                    "child": hip_name,
                    "origin": {"xyz": to_list(attach), "rpy": [0.0, 0.0, 0.0]},
                },
                {
                    "name": f"leg_{leg_index}_lift",
                    "type": "revolute",
                    "parent": hip_name,
                    "child": swing_name,
                    "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                    "axis": to_list(lift_axis),
                    "limit": {"lower": -0.80, "upper": 0.95, "effort": 80.0, "velocity": 2.5},
                    "dynamics": {"damping": 0.2, "friction": 0.05},
                },
                {
                    "name": f"leg_{leg_index}_swing",
                    "type": "revolute",
                    "parent": swing_name,
                    "child": upper_name,
                    "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                    "axis": to_list(swing_axis),
                    "limit": {
                        "lower": float(swing_limits[0]),
                        "upper": float(swing_limits[1]),
                        "effort": 60.0,
                        "velocity": 2.0,
                    },
                    "dynamics": {"damping": 0.15, "friction": 0.04},
                },
                {
                    "name": f"leg_{leg_index}_knee_mount",
                    "type": "fixed",
                    "parent": upper_name,
                    "child": knee_name,
                    "origin": {"xyz": to_list(upper_vector), "rpy": [0.0, 0.0, 0.0]},
                },
                {
                    "name": f"leg_{leg_index}_drop",
                    "type": "revolute",
                    "parent": knee_name,
                    "child": lower_name,
                    "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
                    "axis": to_list(lift_axis),
                    "limit": {"lower": -0.10, "upper": 1.10, "effort": 50.0, "velocity": 2.3},
                    "dynamics": {"damping": 0.12, "friction": 0.03},
                },
                {
                    "name": f"leg_{leg_index}_foot_mount",
                    "type": "fixed",
                    "parent": lower_name,
                    "child": foot_name,
                    "origin": {"xyz": to_list(lower_vector), "rpy": [0.0, 0.0, 0.0]},
                },
            ]
        )

    preview_mesh = trimesh.util.concatenate(assembled_meshes)
    preview_relative_mesh = f"{MESH_DIR_NAME}/assembled_preview.stl"
    export_mesh(preview_mesh, meshes_root / "assembled_preview.stl")

    metadata = {
        "robot_name": args.robot_name,
        "seed": args.seed,
        "density": float(args.density),
        "num_legs": num_legs,
        "generation_profile": {
            "name": "elongated_many_leg_v2_coplanar_feet",
            "leg_count_source": (
                "explicit" if args.num_legs is not None else "weighted_default_8_to_14"
            ),
            "body_aspect_ratio": float(args.body_length / args.body_width),
            "nominal_foot_plane_z": float(nominal_foot_plane_z),
        },
        "morphology_type": args.morphology_type,
        "requested_body_envelope": {
            "length": float(args.body_length), "width": float(args.body_width),
            "height": float(args.body_height),
        },
        "layout_ssm_precheck": layout_ssm,
        "asset_root": ASSET_DIR_NAME,
        "mesh_root": MESH_DIR_NAME,
        "preview_mesh": preview_relative_mesh,
        "trunk_polygon_xy": [to_list(vertex) for vertex in np.asarray(polygon.exterior.coords[:-1], dtype=float)],
        "trunk_bottom_z": hip_z,
        "links": links,
        "joints": joints,
    }

    description_path = assets_root / "robot_description.json"
    description_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated robot assets under: {assets_root}")
    print(f"Robot name: {args.robot_name}")
    print(f"Leg count: {num_legs}")
    print(f"Description file: {description_path}")
    print()
    for link in links:
        print_link_summary(link)
        print()

    return metadata


def main() -> None:
    args = parse_args()
    assemble_robot(args)


if __name__ == "__main__":
    main()
