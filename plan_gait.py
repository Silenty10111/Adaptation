#!/usr/bin/env python3
"""Plan the virtual forward axis and grouped gait for a generated robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from adaptive_gait import compute_adaptive_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description",
        type=Path,
        default=Path("robot_assets") / "robot_description.json",
        help="机器人描述 JSON 路径。",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="可选状态 JSON，用于提供锁死腿、摆动向量、相位等覆盖信息。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选输出 JSON 路径；不传则仅打印结果。",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    description = load_json(args.description)
    state = load_json(args.state) if args.state else {}
    plan = compute_adaptive_plan(description, state)

    print("[阶段一] 虚拟身体轴线与头尾方向")
    print(f"- 支撑中心 CoS: {plan['support_center_xy']}")
    print(f"- 投影质心 CoM: {plan['projected_com_xy']}")
    print(f"- PCA 初始轴: {plan['initial_virtual_forward_axis']}")
    print(f"- 最终前向轴: {plan['final_forward_axis']}")
    print(f"- 驱动合力: {plan['drive_resultant_xy']}")
    print(f"- 平移代偿: {plan['translational_compensation_xy']}")
    print(f"- 支撑腿: {plan['support_leg_ids']}")
    print(f"- 近支撑腿: {plan['near_stance_leg_ids']}")
    print()
    print("[阶段二] 腿间协调与步态分组")
    print(f"- group_a: {plan['topology']['groups']['group_a']}")
    print(f"- group_b: {plan['topology']['groups']['group_b']}")
    print(f"- 拓扑屏蔽: {plan['topology']['inhibition_rules']}")

    if args.output:
        args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"规划结果已写入: {args.output}")


if __name__ == "__main__":
    main()