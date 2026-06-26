#!/usr/bin/env python3
"""
export_mpc_inputs.py  —  SRB-MPC 输入打包器
===========================================
将 robot_description.json + adaptation.gait 的输出，转换为 MPC 求解器的标准输入 JSON。

输出字段示例：
  {
    "mass": 12.3,
    "com": [x, y, z],
    "inertia": [[...],[...],[...]],
    "foot_positions": {"0": [x,y,z], ...},
    "Q": [[...]],
    "R": [[...]],
    "cpg": {"frequency_hz": 0.85, "duty_factor": 0.6, "phase_offsets": {...}},
    "topology": {...}
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from adaptation.mpc import AdaptiveMPCWeights, RobotPhysicsParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description",
        type=Path,
        default=Path("robot_assets") / "robot_description.json",
        help="robot_description.json 路径",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="可选 gait state JSON（覆盖 phase/CPG 等信息）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 路径（不传则打印到 stdout）",
    )
    return parser.parse_args()


def _to_list(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main() -> None:
    args = parse_args()

    description = json.loads(args.description.read_text(encoding="utf-8"))
    state = json.loads(args.state.read_text(encoding="utf-8")) if args.state else {}

    physics = RobotPhysicsParser.from_json(str(args.description))
    weights = AdaptiveMPCWeights(physics).compute()

    # optional adaptation.gait info
    plan = {}
    try:
        from adaptation.gait import compute_adaptive_plan
        plan = compute_adaptive_plan(description, state)
    except Exception:
        plan = {}

    foot_positions = {str(k): v.tolist() for k, v in physics.foot_positions.items()}

    payload: Dict[str, Any] = {
        "mass": float(physics.total_mass),
        "com": physics.com_position.tolist(),
        "inertia": physics.inertia_tensor.tolist(),
        "foot_positions": foot_positions,
        "Q": _to_list(weights.Q),
        "R": _to_list(weights.R),
        "geometry": {
            "aspect_ratio": float(physics.aspect_ratio),
            "body_height": float(physics.body_height),
            "body_half_span_xy": physics.body_half_span_xy.tolist(),
        },
    }

    if plan:
        payload["cpg"] = plan.get("cpg", {})
        payload["topology"] = plan.get("topology", {})
        payload["support_leg_ids"] = plan.get("support_leg_ids", [])

    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
