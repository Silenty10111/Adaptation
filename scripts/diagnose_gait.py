#!/usr/bin/env python3
"""诊断步态配置和 DOF 映射问题。"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_description(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_urdf_dof_names(urdf_path: Path) -> List[str]:
    """从 URDF 中提取关节名称顺序。"""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    dof_names = []
    for joint in root.findall("joint"):
        jtype = joint.get("type")
        if jtype and jtype != "fixed":
            name = joint.get("name")
            if name:
                dof_names.append(name)
    
    return dof_names


def diagnose_mappings(desc_path: Path, urdf_path: Path) -> None:
    """诊断 triplets 映射和分组。"""
    print("=" * 80)
    print("诊断步态配置")
    print("=" * 80)
    
    # 1. 加载描述和 URDF
    print("\n[1] 加载文件...")
    description = load_description(desc_path)
    dof_names = load_urdf_dof_names(urdf_path)
    
    print(f"  描述文件: {desc_path}")
    print(f"  URDF 文件: {urdf_path}")
    print(f"  DOF 数量: {len(dof_names)}")
    print(f"  机器人腿数: {description.get('num_legs', 0)}")
    
    # 2. DOF 关节顺序
    print("\n[2] DOF 关节顺序:")
    for idx, name in enumerate(dof_names):
        print(f"  [{idx:2d}] {name}")
    
    # 3. 模拟 triplets 映射
    print("\n[3] Triplets 映射 (腿 -> DOF 关键关节):")
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    triplets: Dict[int, dict] = {}
    
    for leg_id in range(int(description.get("num_legs", 0))):
        lift_n = f"leg_{leg_id}_lift"
        swing_n = f"leg_{leg_id}_swing"
        drop_n = f"leg_{leg_id}_drop"
        
        if lift_n in name_to_idx and swing_n in name_to_idx and drop_n in name_to_idx:
            triplets[leg_id] = {
                "lift_idx": name_to_idx[lift_n],
                "swing_idx": name_to_idx[swing_n],
                "drop_idx": name_to_idx[drop_n],
            }
            t = triplets[leg_id]
            print(f"  leg_{leg_id}: lift={t['lift_idx']:2d}({lift_n}), " +
                  f"swing={t['swing_idx']:2d}({swing_n}), " +
                  f"drop={t['drop_idx']:2d}({drop_n})")
        else:
            missing = []
            if lift_n not in name_to_idx:
                missing.append(lift_n)
            if swing_n not in name_to_idx:
                missing.append(swing_n)
            if drop_n not in name_to_idx:
                missing.append(drop_n)
            print(f"  leg_{leg_id}: ❌ 缺失关节 {missing}")
    
    print(f"\n  成功映射: {len(triplets)}/{int(description.get('num_legs', 0))} 条腿")
    
    # 4. 足端信息
    print("\n[4] 足端位置 (机器人框架):")
    for link in description.get("links", []):
        if link.get("role") == "foot" and link.get("leg_id") is not None:
            leg_id = link["leg_id"]
            origin = link.get("default_world_origin", [0, 0, 0])
            print(f"  leg_{leg_id}: pos=({origin[0]:.3f}, {origin[1]:.3f})")
    
    # 5. 腿编号顺序
    print("\n[5] 腿编号顺序:")
    print("  期望顺序: 右前 -> 右中 -> 右后 -> 左后 -> 左中 -> 左前")
    print("           (0) -> (1) -> (2) -> (3) -> (4) -> (5)")
    
    foot_positions = {}
    for link in description.get("links", []):
        if link.get("role") == "foot" and link.get("leg_id") is not None:
            leg_id = int(link["leg_id"])
            origin = np.array(link.get("default_world_origin", [0, 0, 0]))
            foot_positions[leg_id] = origin[:2]
    
    print("\n  实际足端位置（Y 坐标判断左右）:")
    for leg_id in sorted(foot_positions.keys()):
        x, y = foot_positions[leg_id]
        side = "左" if y > 0 else "右"
        x_pos = "前" if x > 0.1 else ("后" if x < -0.1 else "中")
        print(f"  leg_{leg_id}: {side}{x_pos} ({x:.3f}, {y:.3f})")
    
    # 6. 分组信息
    print("\n[6] 分组信息 (来自自适应规划):")
    try:
        from adaptation.gait import compute_adaptive_plan
        plan = compute_adaptive_plan(description, {})
        group_a = plan["topology"]["groups"]["group_a"]
        group_b = plan["topology"]["groups"]["group_b"]
        print(f"  group_a: {group_a}")
        print(f"  group_b: {group_b}")
        
        phi_offsets = plan.get("cpg", {}).get("phase_offsets", {})
        if phi_offsets:
            print(f"\n  CPG 相位偏置:")
            for leg_id_str, phase in phi_offsets.items():
                print(f"    leg_{leg_id_str}: {phase/np.pi:.2f}π")
    except Exception as e:
        print(f"  ⚠️  无法加载自适应规划: {e}")
    
    # 7. 问题诊断
    print("\n[7] 问题诊断:")
    if len(triplets) < int(description.get("num_legs", 0)):
        print(f"  ❌ 腿映射不完整 ({len(triplets)}/{int(description.get('num_legs', 0))})")
    else:
        print(f"  ✓ 所有腿都成功映射")
    
    # 检查 DOF 编号是否连续
    dof_indices = set()
    for triplet in triplets.values():
        dof_indices.update([triplet["lift_idx"], triplet["swing_idx"], triplet["drop_idx"]])
    
    if len(dof_indices) == 3 * len(triplets):
        print(f"  ✓ DOF 索引无重复")
    else:
        print(f"  ❌ DOF 索引有重复或缺失")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    ASSET_DIR = Path("robot_assets/standard_hexapod")
    desc_path = ASSET_DIR / "robot_description.json"
    urdf_path = ASSET_DIR / "generated_robot.urdf"
    
    if not desc_path.exists():
        print(f"❌ 描述文件不存在: {desc_path}")
        exit(1)
    if not urdf_path.exists():
        print(f"❌ URDF 文件不存在: {urdf_path}")
        exit(1)
    
    diagnose_mappings(desc_path, urdf_path)
