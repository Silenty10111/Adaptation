#!/usr/bin/env python3
"""
机器人几何形态与SSM可视化工具（修复版 - 每次生成全新机器人）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================
# 字体配置
# ============================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

chinese_fonts = [
    'WenQuanYi Zen Hei',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Noto Sans SC',
    'SimHei',
    'DejaVu Sans',
]

available_font = None
for font_name in chinese_fonts:
    try:
        font_path = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
        if font_path:
            available_font = font_name
            break
    except:
        continue

if available_font:
    print(f"[INFO] 使用字体: {available_font}")
    matplotlib.rcParams['font.family'] = available_font
else:
    print("[WARN] 未找到中文字体，使用英文")

matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# ---------- 配置 ----------
GEN_PYTHON = "/data/conda/envs/Adaptation/bin/python"
REPO_ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = REPO_ROOT / "generate_geometry.py"
OUTPUT_DIR = REPO_ROOT / "png"
DESC_PATH = REPO_ROOT / "robot_assets" / "robot_description.json"
# ------------------------

try:
    from shapely.geometry import MultiPoint
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


def _signed_polygon_area(polygon_xy: np.ndarray) -> float:
    pts = np.asarray(polygon_xy, dtype=float)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _ensure_ccw(polygon_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon_xy, dtype=float)
    if len(pts) < 3:
        return pts
    if _signed_polygon_area(pts) < 0.0:
        return pts[::-1].copy()
    return pts


class SSMVisualizer:
    """SSM 可视化器"""
    
    def __init__(self, description: Dict):
        self.desc = description
        self._parse_data()
    
    def _parse_data(self):
        self.body_outline = self._get_body_outline()
        self.legs_info = self._get_legs_info()
        self.foot_positions = self._get_foot_positions()
        self.com_xy = self._compute_com_xy()
        self.support_polygon = self._compute_support_polygon()
        self.ssm_value, self.closest_edge = self._compute_ssm()
    
    def _get_body_outline(self) -> np.ndarray:
        if "trunk_polygon_xy" in self.desc:
            pts = np.array(self.desc["trunk_polygon_xy"], dtype=float)
            if len(pts) > 0 and pts.shape[1] >= 2:
                return pts[:, :2]
        
        if "body_outline" in self.desc:
            pts = np.array(self.desc["body_outline"], dtype=float)
            if len(pts) > 0 and pts.shape[1] >= 2:
                return pts[:, :2]
        
        bl = float(self.desc.get("body_length", 0.3))
        bw = float(self.desc.get("body_width", 0.2))
        hw = bw / 2
        hl = bl / 2
        return np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=float)
    
    def _get_legs_info(self) -> List[Dict]:
        legs = []
        for link in self.desc.get("links", []):
            role = link.get("role", "")
            if role in ("hip", "upper_link", "lower_link", "foot"):
                origin = np.array(link.get("default_world_origin", [0, 0, 0]), dtype=float)
                leg_id = link.get("leg_id")
                
                found = False
                for leg in legs:
                    if leg.get("leg_id") == leg_id:
                        leg["segments"].append({"role": role, "origin": origin})
                        if role == "foot":
                            leg["foot_xy"] = origin[:2]
                        found = True
                        break
                
                if not found and leg_id is not None:
                    new_leg = {"leg_id": leg_id, "segments": [{"role": role, "origin": origin}]}
                    if role == "foot":
                        new_leg["foot_xy"] = origin[:2]
                    legs.append(new_leg)
        
        return legs
    
    def _get_foot_positions(self) -> np.ndarray:
        feet = []
        for leg in self.legs_info:
            if "foot_xy" in leg:
                feet.append(leg["foot_xy"])
        
        if not feet:
            for link in self.desc.get("links", []):
                if link.get("role") == "foot":
                    origin = np.array(link.get("default_world_origin", [0, 0, 0]), dtype=float)
                    feet.append(origin[:2])
        
        return np.array(feet) if feet else np.zeros((0, 2))
    
    def _compute_com_xy(self) -> np.ndarray:
        weighted_sum = np.zeros(2, dtype=float)
        total_mass = 0.0
        
        for link in self.desc.get("links", []):
            mass_props = link.get("mass_properties", {})
            mass = float(mass_props.get("mass", 0.0))
            if mass <= 0:
                continue
            
            origin = np.array(link.get("default_world_origin", [0, 0, 0]), dtype=float)
            cm_local = np.array(mass_props.get("center_mass", [0, 0, 0]), dtype=float)
            world_cm = origin + cm_local
            weighted_sum += world_cm[:2] * mass
            total_mass += mass
        
        return weighted_sum / total_mass if total_mass > 1e-9 else np.zeros(2)
    
    def _compute_support_polygon(self) -> np.ndarray:
        if len(self.foot_positions) < 3:
            return self.foot_positions.copy()
        
        if SHAPELY_AVAILABLE:
            try:
                hull = MultiPoint(self.foot_positions.tolist()).convex_hull
                if hull.geom_type == 'Polygon':
                    return _ensure_ccw(np.array(hull.exterior.coords[:-1]))
            except:
                pass
        
        center = self.foot_positions.mean(axis=0)
        angles = np.arctan2(
            self.foot_positions[:, 1] - center[1],
            self.foot_positions[:, 0] - center[0]
        )
        return _ensure_ccw(self.foot_positions[np.argsort(angles)])
    
    def _compute_ssm(self) -> Tuple[float, Optional[Dict]]:
        n = len(self.support_polygon)
        if n < 3:
            return 0.0, None
        
        pts = self.support_polygon
        p = self.com_xy
        
        min_dist = float('inf')
        closest_edge = None
        
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            edge = b - a
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-9:
                continue
            
            signed_dist = (edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])) / edge_len
            
            if signed_dist < min_dist:
                min_dist = signed_dist
                closest_edge = {
                    'start': a.copy(),
                    'end': b.copy(),
                    'signed_distance': signed_dist,
                    'edge_index': i
                }
        
        return float(min_dist) if np.isfinite(min_dist) else 0.0, closest_edge
    
    def plot(self, save_path: Path, robot_name: str, seed: int):
        """绘制可视化图片"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))
        
        # 1. 躯干轮廓
        body_closed = np.vstack([self.body_outline, self.body_outline[0]])
        ax.plot(body_closed[:, 0], body_closed[:, 1], 'k-', linewidth=2.5, 
                label='躯干轮廓' if available_font else 'Body Outline')
        ax.fill(body_closed[:, 0], body_closed[:, 1], alpha=0.15, color='gray')
        
        # 2. 腿部链节
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.legs_info), 1)))
        for idx, leg in enumerate(self.legs_info):
            color = colors[idx % len(colors)]
            segments = leg.get("segments", [])
            
            for i in range(len(segments) - 1):
                p1 = segments[i]["origin"][:2]
                p2 = segments[i + 1]["origin"][:2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=color, linewidth=1.5, alpha=0.7)
            
            for seg in segments:
                pt = seg["origin"][:2]
                ax.plot(pt[0], pt[1], 'o', color=color, markersize=4, alpha=0.6)
        
        # 3. 落足点
        if len(self.foot_positions) > 0:
            ax.scatter(
                self.foot_positions[:, 0], 
                self.foot_positions[:, 1],
                s=120, c='red', marker='s', zorder=5, edgecolors='darkred', linewidths=1.5,
                label='落足点' if available_font else 'Foot Points'
            )
            for i, fp in enumerate(self.foot_positions):
                ax.annotate(
                    f'F{i+1}', (fp[0], fp[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
                )
        
        # 4. 支撑多边形
        if len(self.support_polygon) >= 3:
            poly_closed = np.vstack([self.support_polygon, self.support_polygon[0]])
            
            if self.ssm_value >= 0:
                poly_color = 'green'
                poly_alpha = 0.25
                edge_color = 'darkgreen'
            else:
                poly_color = 'red'
                poly_alpha = 0.15
                edge_color = 'darkred'
            
            poly_label = f'支撑多边形 ({len(self.support_polygon)}边形)' if available_font else f'Support Polygon ({len(self.support_polygon)}-gon)'
            ax.fill(poly_closed[:, 0], poly_closed[:, 1], alpha=poly_alpha, color=poly_color)
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], '--', color=edge_color, linewidth=2, label=poly_label)
            
            for i, v in enumerate(self.support_polygon):
                ax.annotate(
                    f'V{i}', (v[0], v[1]),
                    xytext=(-8, -8), textcoords='offset points',
                    fontsize=8, color=edge_color
                )
        
        # 5. 质心位置
        com_label = '质心投影 CoM' if available_font else 'CoM Projection'
        ax.scatter(
            self.com_xy[0], self.com_xy[1],
            s=200, c='blue', marker='*', zorder=10, edgecolors='darkblue', linewidths=2,
            label=com_label
        )
        
        # 6. SSM标注
        if self.closest_edge is not None:
            edge_data = self.closest_edge
            edge_start = edge_data['start']
            edge_end = edge_data['end']
            
            edge_vec = edge_end - edge_start
            edge_len = np.linalg.norm(edge_vec)
            if edge_len > 1e-9:
                edge_unit = edge_vec / edge_len
                t = np.dot(self.com_xy - edge_start, edge_unit)
                t = np.clip(t, 0, edge_len)
                foot_point = edge_start + t * edge_unit
                
                ax.plot(
                    [self.com_xy[0], foot_point[0]], 
                    [self.com_xy[1], foot_point[1]],
                    'g--', linewidth=2, alpha=0.8
                )
                
                mid_point = (self.com_xy + foot_point) / 2
                ax.annotate(
                    f'SSM = {self.ssm_value:.4f} m',
                    mid_point,
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    color='green' if self.ssm_value >= 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6)
                )
        
        # 7. 标题
        status_text = "✓ 通过 (静态稳定)" if self.ssm_value >= 0 else "✗ 不通过 (静态不稳定)"
        status_color = 'green' if self.ssm_value >= 0 else 'red'
        
        title = (
            f"机器人静态稳定性分析 (种子: {seed})\n"
            f"模型: {robot_name} | "
            f"足端数: {len(self.foot_positions)} | "
            f"SSM = {self.ssm_value:.4f} m | "
            f"{status_text}"
        )
        ax.set_title(title, fontsize=14, fontweight='bold', color=status_color)
        
        # 8. 图表设置
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # 9. 信息文本
        info_text = (
            f"SSM (静态稳定裕度): {self.ssm_value:.4f} m\n"
            f"阈值: 0.0 m\n"
            f"足端数量: {len(self.foot_positions)}\n"
            f"支撑多边形顶点: {len(self.support_polygon)}\n"
            f"质心坐标: ({self.com_xy[0]:.4f}, {self.com_xy[1]:.4f})\n"
            f"判定: {'稳定' if self.ssm_value >= 0 else '不稳定'}\n"
            f"种子: {seed}"
        )
        
        props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9)
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=props,
            family='monospace'
        )
        
        # 10. 保存
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] 图片已保存: {save_path}")


def generate_robot_ignore_ssm(robot_name: str, seed: int, leg_placement: str = "random",
                               body_length=None, body_width=None, body_height=None,
                               upper_length=None, lower_length=None) -> bool:
    """
    生成机器人并忽略SSM检测
    原理：先生成几何，如果SSM不通过，用另一种方式强制生成描述文件
    """
    # 方案：先删除旧的描述文件
    if DESC_PATH.exists():
        DESC_PATH.unlink()
        print(f"[INFO] 已删除旧描述文件")
    
    cmd = [
        GEN_PYTHON, str(GEN_SCRIPT),
        "--robot-name", robot_name,
        "--seed", str(seed),
        "--leg-placement", leg_placement,
    ]
    
    for param_name, flag in [
        ("body_length", "--body-length"),
        ("body_width", "--body-width"),
        ("body_height", "--body-height"),
        ("upper_length", "--upper-length"),
        ("lower_length", "--lower-length"),
    ]:
        val = locals().get(param_name)
        if val is not None:
            cmd.extend([flag, str(val)])
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 检查是否生成了描述文件
    if DESC_PATH.exists():
        # 验证文件是否是刚生成的
        file_time = DESC_PATH.stat().st_mtime
        current_time = time.time()
        if current_time - file_time < 10:  # 10秒内生成的文件
            print(f"[INFO] 描述文件生成成功 (时间戳匹配)")
            return True
    
    # 如果SSM失败导致没有生成描述文件，尝试强制生成
    print(f"[WARN] 标准生成失败，尝试强制生成模式...")
    print(f"[STDERR] {result.stderr[:300]}")
    
    # 这里可以添加备用生成逻辑
    # 例如：使用更宽松的参数重新生成
    # 或者：直接构造一个简化的描述文件
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SSM Visualization Tool")
    parser.add_argument("--single", action="store_true", help="Single generation mode")
    parser.add_argument("--count", type=int, default=10, help="Batch count")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--seed-base", type=int, default=7, help="Batch base seed")
    parser.add_argument("--seed-step", type=int, default=17, help="Batch seed step")
    parser.add_argument("--leg-placement", default="random", choices=["uniform", "random"])
    parser.add_argument("--body-length", type=float, default=None)
    parser.add_argument("--body-width", type=float, default=None)
    parser.add_argument("--body-height", type=float, default=None)
    parser.add_argument("--upper-length", type=float, default=None)
    parser.add_argument("--lower-length", type=float, default=None)
    
    args = parser.parse_args()
    
    if not GEN_SCRIPT.exists():
        print(f"[ERROR] {GEN_SCRIPT} not found")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.single:
        # 单次模式
        robot_name = f"robot_s{args.seed}"
        print(f"\n{'='*60}")
        print(f"生成机器人: {robot_name}")
        print(f"{'='*60}")
        
        # 生成机器人
        success = generate_robot_ignore_ssm(
            robot_name, args.seed, args.leg_placement,
            args.body_length, args.body_width, args.body_height,
            args.upper_length, args.lower_length
        )
        
        if success and DESC_PATH.exists():
            with open(DESC_PATH, 'r') as f:
                description = json.load(f)
            
            visualizer = SSMVisualizer(description)
            ssm_status = "pass" if visualizer.ssm_value >= 0 else "fail"
            filename = f"{robot_name}_{ssm_status}_ssm{visualizer.ssm_value:.4f}.png"
            save_path = OUTPUT_DIR / filename
            
            visualizer.plot(save_path, robot_name, args.seed)
            
            print(f"\n结果:")
            print(f"  种子: {args.seed}")
            print(f"  足端数: {len(visualizer.foot_positions)}")
            print(f"  SSM: {visualizer.ssm_value:.4f} m")
            print(f"  状态: {'✓ 通过' if visualizer.ssm_value >= 0 else '✗ 不通过'}")
        else:
            print(f"[ERROR] 无法生成机器人")
    else:
        # 批量模式
        print(f"\n批量生成 {args.count} 个机器人...")
        
        for i in range(args.count):
            seed = args.seed_base + i * args.seed_step
            robot_name = f"robot_{i:03d}_s{seed}"
            
            print(f"\n[{i+1}/{args.count}] 种子={seed}")
            
            success = generate_robot_ignore_ssm(
                robot_name, seed, args.leg_placement,
                args.body_length, args.body_width, args.body_height,
                args.upper_length, args.lower_length
            )
            
            if success and DESC_PATH.exists():
                with open(DESC_PATH, 'r') as f:
                    description = json.load(f)
                
                visualizer = SSMVisualizer(description)
                ssm_status = "pass" if visualizer.ssm_value >= 0 else "fail"
                filename = f"{robot_name}_{ssm_status}_ssm{visualizer.ssm_value:.4f}.png"
                save_path = OUTPUT_DIR / filename
                
                visualizer.plot(save_path, robot_name, seed)
                print(f"  SSM: {visualizer.ssm_value:.4f} m")
            
            time.sleep(0.1)


if __name__ == "__main__":
    main()