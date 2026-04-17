#!/usr/bin/env python3
"""Generate a generalized multi-legged robot geometry and physical metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


ASSET_DIR_NAME = "robot_assets"
MESH_DIR_NAME = "meshes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-name", default="generated_multileg")
    parser.add_argument("--num-legs", type=int, default=None, help="Randomly chosen in [4, 10] when omitted.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--leg-placement", choices=("uniform", "random"), default="uniform")
    parser.add_argument(
        "--leg-style",
        choices=("swing", "pendulum", "mixed"),
        default="mixed",
        help="Swing legs allow larger lateral sweep; pendulum legs stay close to a single plane.",
    )
    parser.add_argument("--body-length", type=float, default=0.72)
    parser.add_argument("--body-width", type=float, default=0.44)
    parser.add_argument("--body-height", type=float, default=0.18)
    parser.add_argument("--upper-length", type=float, default=0.28)
    parser.add_argument("--lower-length", type=float, default=0.30)
    parser.add_argument("--joint-radius", type=float, default=0.035)
    parser.add_argument("--link-radius", type=float, default=0.018)
    parser.add_argument("--density", type=float, default=780.0, help="Uniform density in kg/m^3.")
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


def create_irregular_trunk_polygon(length: float, width: float, rng: np.random.Generator) -> Polygon:
    """Create an irregular convex polygon by trimming the four rectangle corners."""
    trim_x = rng.uniform(length * 0.06, length * 0.18, size=4)
    trim_y = rng.uniform(width * 0.06, width * 0.18, size=4)
    half_length = length / 2.0
    half_width = width / 2.0

    vertices = np.array(
        [
            [-half_length + trim_x[0], -half_width],
            [half_length - trim_x[1], -half_width],
            [half_length, -half_width + trim_y[1]],
            [half_length, half_width - trim_y[2]],
            [half_length - trim_x[2], half_width],
            [-half_length + trim_x[3], half_width],
            [-half_length, half_width - trim_y[3]],
            [-half_length, -half_width + trim_y[0]],
        ],
        dtype=float,
    )
    polygon = orient(Polygon(vertices), sign=1.0)
    if not polygon.is_valid or polygon.area <= 0.0:
        raise RuntimeError("Failed to generate a valid trunk polygon.")
    return polygon


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
        distances = np.sort(rng.uniform(0.0, cumulative, size=count))

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
    tangent = np.array([tangent_xy[0], tangent_xy[1], 0.0], dtype=float)
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    side = rng.choice([-1.0, 1.0])

    if leg_type == "swing":
        upper_direction = normalize(0.24 * inward + 0.12 * side * tangent + 1.0 * down)
        lower_direction = normalize(0.10 * inward - 0.05 * side * tangent + 1.0 * down)
    else:
        upper_direction = normalize(0.22 * inward + 1.0 * down)
        lower_direction = normalize(0.08 * inward + 1.0 * down)

    return upper_direction * upper_length, lower_direction * lower_length


def select_leg_type(requested_style: str, leg_index: int, rng: np.random.Generator) -> str:
    if requested_style == "mixed":
        return "swing" if (leg_index % 2 == 0 or rng.random() > 0.5) else "pendulum"
    return requested_style


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
    num_legs = args.num_legs if args.num_legs is not None else int(rng.integers(4, 11))
    if num_legs < 4 or num_legs > 10:
        raise ValueError("num_legs must be within [4, 10].")

    assets_root, meshes_root = make_assets_root()
    polygon = create_irregular_trunk_polygon(args.body_length, args.body_width, rng)
    trunk_mesh = extrude_trunk_mesh(polygon, args.body_height)

    links: List[Dict[str, object]] = []
    joints: List[Dict[str, object]] = []
    assembled_meshes: List[trimesh.Trimesh] = []

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

    mount_points = compute_mount_points(polygon, num_legs, args.leg_placement, rng)
    hip_z = -args.body_height / 2.0

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
        attach = np.array([point_xy[0], point_xy[1], hip_z], dtype=float)
        knee_world = attach + upper_vector
        foot_world = knee_world + lower_vector
        lift_axis = normalize([tangent_xy[0], tangent_xy[1], 0.0])
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

        swing_limits = [-0.55, 0.55] if leg_type == "swing" else [-0.12, 0.12]
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
                    "limit": {"lower": -0.80, "upper": 0.95, "effort": 25.0, "velocity": 2.5},
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
                        "effort": 20.0,
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
                    "limit": {"lower": -0.10, "upper": 1.10, "effort": 18.0, "velocity": 2.3},
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