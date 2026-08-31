#!/usr/bin/env python3
"""
拓扑不变性虚拟足端映射器 (Topology-Invariant Virtual Foot Mapper)
=================================================================
关联论文: Multi-Loco: Unifying Multi-Embodiment Legged Locomotion
         via Reinforcement Learning Augmented Diffusion

核心思想
--------
无论机器人有 5 条腿还是 10 条腿、形态多么不规则，
都将其在数学上映射到一个**标准虚拟支撑多边形**上进行高层控制指令生成。

这样，核心控制逻辑彻底与"具体有几条腿"解耦，
实现零样本 (Zero-shot) 的新形态步态生成。

架构
----
1. VirtualSupportPolygon   — 标准虚拟支撑结构（4-节点菱形）
2. TopologyInvariantMapper — 将任意 N 腿形态映射到虚拟多边形
3. VirtualLegController    — 在虚拟空间生成控制指令，再反映射到真实腿
4. zero_shot_gait_plan()   — 零样本步态生成接口

关键不变量
----------
- 虚拟支撑多边形始终为标准化的 4 节点结构：
    · 前节点 (Front)  → 对应前向推进组
    · 后节点 (Rear)   → 对应后向支撑组
    · 左节点 (Left)   → 对应左侧稳定组
    · 右节点 (Right)  → 对应右侧稳定组

- 拓扑不变性矩阵 T ∈ R^{4×N}：
    T[v, i] = 第 i 腿对虚拟节点 v 的贡献权重
    约束: Σ_i T[v, i] = 1 for all v（凸组合）

- 高层控制只与 4 节点虚拟结构交互，
  底层执行通过 T^† (伪逆) 将命令分配到真实腿。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from adaptation.phase import (
    PHASE_STRATEGIES,
    build_phase_offsets,
    circular_mean,
    clip_duty_factor,
    resolve_duty_factors,
    wrap_2pi,
)


# ---------------------------------------------------------------------------
# 虚拟支撑结构常量
# ---------------------------------------------------------------------------

VIRTUAL_NODES = {
    "front": 0,
    "rear":  1,
    "left":  2,
    "right": 3,
}
N_VIRTUAL = 4  # 虚拟节点数（固定，确保拓扑不变性）


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class VirtualSupportPolygon:
    """
    标准虚拟支撑多边形（4 节点）。

    坐标系：机器人质心为原点，前进方向为 +X，左侧为 +Y。
    """
    front_xy: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.0]))
    rear_xy: np.ndarray  = field(default_factory=lambda: np.array([-0.20, 0.0]))
    left_xy: np.ndarray  = field(default_factory=lambda: np.array([0.0, 0.15]))
    right_xy: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.15]))

    def get_node(self, idx: int) -> np.ndarray:
        return [self.front_xy, self.rear_xy, self.left_xy, self.right_xy][idx]

    def as_matrix(self) -> np.ndarray:
        """返回 4×2 矩阵，每行是一个虚拟节点坐标。"""
        return np.stack([
            self.front_xy, self.rear_xy, self.left_xy, self.right_xy
        ], axis=0)


@dataclass
class MorphologyTopology:
    """
    存储形态拓扑信息：真实腿 ↔ 虚拟节点的映射关系。
    """
    n_real_legs: int
    n_virtual_nodes: int = N_VIRTUAL
    assignment_matrix: np.ndarray = field(default=None)  # R^{4×N}
    real_leg_ids: List[int] = field(default_factory=list)
    virtual_polygon: VirtualSupportPolygon = field(
        default_factory=VirtualSupportPolygon
    )
    description: str = ""  # 形态描述（用于调试）


# ---------------------------------------------------------------------------
# 1. 拓扑不变性映射器
# ---------------------------------------------------------------------------

class TopologyInvariantMapper:
    """
    将任意 N 腿形态映射到标准 4 节点虚拟支撑多边形。

    映射算法
    --------
    步骤 1：标准化足端坐标
        将所有足端坐标转换到以质心为原点、前进方向为 +X 的坐标系。

    步骤 2：软分配 (Soft Assignment)
        使用径向基函数 (RBF) 计算每条真实腿对每个虚拟节点的相似度：
            sim(leg_i, node_v) = exp(-||foot_i - virtual_v||² / (2σ²))
        归一化：T[v, i] = sim(leg_i, node_v) / Σ_j sim(leg_j, node_v)

    步骤 3：列归一化（确保每腿权重之和为 1）
        T_norm[v, i] = T[v, i] / (Σ_v T[v, i])

    步骤 4：计算伪逆 T^†，用于从虚拟指令反映射到真实腿。

    拓扑不变性保证
    --------------
    对于任意 N，映射矩阵 T 的形状始终为 4×N，
    高层控制器仅操作 4 维虚拟空间，与 N 无关。
    """

    def __init__(
        self,
        foot_positions_xy: Dict[int, np.ndarray],
        com_xy: np.ndarray,
        forward_axis: np.ndarray,
        sigma: float = 0.0,
    ):
        """
        Parameters
        ----------
        foot_positions_xy : {leg_id: [x, y]} 足端坐标（世界坐标系）
        com_xy            : 质心 XY
        forward_axis      : 前进方向（世界坐标系，单位向量）
        sigma             : RBF 带宽参数。
                            <= 0 时自动从足端分布范围估算。
        """
        self._feet_raw = foot_positions_xy
        self._com = np.asarray(com_xy[:2], dtype=float)
        fwd = np.asarray(forward_axis[:2], dtype=float)
        fwd_n = float(np.linalg.norm(fwd))
        self._fwd = fwd / fwd_n if fwd_n > 1e-9 else np.array([1., 0.])
        self._lat = np.array([-self._fwd[1], self._fwd[0]], dtype=float)

        self._leg_ids: List[int] = sorted(foot_positions_xy.keys())
        self._N = len(self._leg_ids)
        self._feet_local = self._transform_to_local()

        # 自动估计 sigma（若未指定）
        if sigma <= 0:
            self._sigma = self._estimate_sigma()
        else:
            self._sigma = sigma

    def _estimate_sigma(self) -> float:
        """从足端分布范围自动估算 RBF 带宽。

        基于平均足端间距的 0.3 倍，确保 RBF 不会过度集中或过度平滑。
        """
        if self._N < 2:
            return 0.25
        fwd_range = float(np.ptp(self._feet_local[:, 0])) if self._N > 1 else 0.4
        lat_range = float(np.ptp(self._feet_local[:, 1])) if self._N > 1 else 0.3
        # 平均足端间距 ≈ sqrt(range_fwd * range_lat / N)
        mean_spacing = math.sqrt(max(fwd_range * lat_range, 0.01) / max(self._N, 1))
        return max(0.08, min(0.60, mean_spacing * 0.30))

    def _transform_to_local(self) -> np.ndarray:
        """
        将足端坐标转换到以质心为原点、前进方向为 +X 的局部坐标系。

        Returns
        -------
        np.ndarray  shape (N, 2)  每行是一个足端的局部坐标 [fwd, lat]
        """
        local = np.zeros((self._N, 2), dtype=float)
        for i, lid in enumerate(self._leg_ids):
            r = self._feet_raw[lid] - self._com
            local[i, 0] = float(np.dot(r, self._fwd))   # 前向分量
            local[i, 1] = float(np.dot(r, self._lat))   # 横向分量
        return local

    def build_virtual_polygon(self) -> VirtualSupportPolygon:
        """
        根据足端分布自动计算虚拟支撑多边形的尺寸。

        虚拟节点位置 = 实际足端在各方向上的典型位置（加权均值）。
        """
        if self._N == 0:
            return VirtualSupportPolygon()

        feet = self._feet_local  # (N, 2)

        # 前节点：前向分量最大的腿群重心
        # 后节点：前向分量最小的腿群重心
        # 左节点：横向分量最大的腿群重心
        # 右节点：横向分量最小的腿群重心
        fwd_coords = feet[:, 0]
        lat_coords = feet[:, 1]

        fwd_mid = float(np.median(fwd_coords))
        lat_mid = float(np.median(lat_coords))

        def _weighted_centroid(mask: np.ndarray) -> np.ndarray:
            if not np.any(mask):
                return np.zeros(2, dtype=float)
            w = np.exp(3.0 * np.abs(feet[mask, 0] - fwd_mid) +
                       3.0 * np.abs(feet[mask, 1] - lat_mid))
            w = w / w.sum()
            return np.average(feet[mask], axis=0, weights=w)

        front_mask = fwd_coords > fwd_mid
        rear_mask = fwd_coords <= fwd_mid
        left_mask = lat_coords > lat_mid
        right_mask = lat_coords <= lat_mid

        # 自适应 fallback：基于足端分布范围缩放默认值
        fwd_ext = max(abs(float(np.min(fwd_coords))), abs(float(np.max(fwd_coords))), 0.20)
        lat_ext = max(abs(float(np.min(lat_coords))), abs(float(np.max(lat_coords))), 0.15)

        front_xy = _weighted_centroid(front_mask) if np.any(front_mask) else np.array([fwd_ext, 0.0])
        rear_xy = _weighted_centroid(rear_mask) if np.any(rear_mask) else np.array([-fwd_ext, 0.0])
        left_xy = _weighted_centroid(left_mask) if np.any(left_mask) else np.array([0.0, lat_ext])
        right_xy = _weighted_centroid(right_mask) if np.any(right_mask) else np.array([0.0, -lat_ext])

        return VirtualSupportPolygon(
            front_xy=front_xy,
            rear_xy=rear_xy,
            left_xy=left_xy,
            right_xy=right_xy,
        )

    def compute_topology_matrix(
        self, virtual_polygon: Optional[VirtualSupportPolygon] = None
    ) -> np.ndarray:
        """
        计算拓扑不变性矩阵 T ∈ R^{4×N}。

        T[v, i] = 第 i 条真实腿对第 v 个虚拟节点的贡献权重

        满足约束：
            Σ_i T[v, i] = 1  for all v   (每虚拟节点权重归一化)

        Returns
        -------
        np.ndarray  shape (4, N)
        """
        if virtual_polygon is None:
            virtual_polygon = self.build_virtual_polygon()

        vp_mat = virtual_polygon.as_matrix()  # (4, 2)  虚拟节点坐标
        feet = self._feet_local              # (N, 2)  真实足端局部坐标

        sigma2 = 2.0 * self._sigma ** 2

        # RBF 相似度矩阵: sim[v, i] = exp(-||foot_i - node_v||² / sigma²)
        T = np.zeros((N_VIRTUAL, self._N), dtype=float)
        for v in range(N_VIRTUAL):
            for i in range(self._N):
                diff = feet[i] - vp_mat[v]
                T[v, i] = math.exp(-float(np.dot(diff, diff)) / sigma2)

        # 行归一化（每虚拟节点的权重之和为 1）
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        T = T / row_sums

        return T

    def build_morphology_topology(self) -> MorphologyTopology:
        """构建完整的形态拓扑信息。"""
        vp = self.build_virtual_polygon()
        T = self.compute_topology_matrix(vp)

        desc_parts = []
        desc_parts.append(f"{self._N}条腿")
        if self._N <= 10:
            fwd_vals = self._feet_local[:, 0]
            lat_vals = self._feet_local[:, 1]
            n_front = int(np.sum(fwd_vals > np.median(fwd_vals)))
            n_left = int(np.sum(lat_vals > np.median(lat_vals)))
            desc_parts.append(f"前{n_front}后{self._N - n_front}")
            desc_parts.append(f"左{n_left}右{self._N - n_left}")

        return MorphologyTopology(
            n_real_legs=self._N,
            assignment_matrix=T,
            real_leg_ids=self._leg_ids,
            virtual_polygon=vp,
            description="，".join(desc_parts),
        )


# ---------------------------------------------------------------------------
# 2. 虚拟腿控制器
# ---------------------------------------------------------------------------

class VirtualLegController:
    """
    在虚拟 4 节点空间中生成控制指令，然后反映射到真实腿。

    高层控制逻辑（与腿数无关）：
    - 前虚拟节点：摆动相（推进）
    - 后虚拟节点：支撑相（稳定）
    - 左/右虚拟节点：平衡相（防侧翻）

    反映射：
        real_command_i = T^† @ virtual_command
    其中 T^† = T^T (T T^T)^{-1} 是 T 的右伪逆。
    """

    def __init__(self, topology: MorphologyTopology):
        self._topo = topology
        self._T = topology.assignment_matrix  # (4, N)
        # 右伪逆 T^† ∈ R^{N×4}
        try:
            TT_T = self._T @ self._T.T  # (4, 4)
            self._T_pinv = self._T.T @ np.linalg.inv(TT_T + 1e-8 * np.eye(4))  # (N, 4)
        except np.linalg.LinAlgError:
            self._T_pinv = self._T.T  # 退化到转置

    def virtual_to_real_phases(
        self,
        virtual_phase_front: float,
        virtual_phase_rear: float,
        virtual_phase_left: float,
        virtual_phase_right: float,
    ) -> Dict[int, float]:
        """
        将 4 个虚拟节点的相位映射到各真实腿。

        Parameters
        ----------
        virtual_phase_* : 各虚拟节点的当前相位

        Returns
        -------
        Dict[int, float]  leg_id → 真实腿相位
        """
        virtual_phases = np.array([
            virtual_phase_front,
            virtual_phase_rear,
            virtual_phase_left,
            virtual_phase_right,
        ], dtype=float)

        result: Dict[int, float] = {}
        for i, lid in enumerate(self._topo.real_leg_ids):
            # Phase is circular: use the non-negative assignment weights on
            # the unit circle instead of linearly averaging values near 0/2π.
            weights = np.asarray(self._T[:, i], dtype=float)
            result[lid] = circular_mean(
                virtual_phases.tolist(), weights=weights.tolist(),
            )

        return result

    def virtual_to_real_amplitudes(
        self,
        virtual_amplitude: np.ndarray,  # shape (4,) 各虚拟节点步幅
    ) -> Dict[int, float]:
        """
        将虚拟步幅映射到各真实腿的步幅缩放因子。

        Returns
        -------
        Dict[int, float]  leg_id → 步幅缩放因子 [0.1, 0.9]
        """
        real_amps = self._T_pinv @ virtual_amplitude  # (N,)
        result: Dict[int, float] = {}
        for i, lid in enumerate(self._topo.real_leg_ids):
            result[lid] = float(np.clip(abs(real_amps[i]), 0.10, 0.90))
        return result

    def generate_alternating_groups(self) -> Tuple[List[int], List[int], List[int]]:
        """
        基于拓扑矩阵自动生成腿组分配，替代 adaptation.gait 的启发式分组。

        策略：
        - group_a (相位 0)   = 主要受"前"虚拟节点影响的腿
        - group_b (相位 π)   = 主要受"后"虚拟节点影响的腿
        - group_c (被动)     = 左/右节点为主（横向稳定）

        Returns
        -------
        (group_a, group_b, group_c)  各腿 ID 列表
        """
        T = self._T  # (4, N)
        front_weights = T[VIRTUAL_NODES["front"], :]   # (N,)
        rear_weights  = T[VIRTUAL_NODES["rear"], :]    # (N,)
        left_weights  = T[VIRTUAL_NODES["left"], :]    # (N,)
        right_weights = T[VIRTUAL_NODES["right"], :]   # (N,)

        group_a: List[int] = []
        group_b: List[int] = []
        group_c: List[int] = []

        for i, lid in enumerate(self._topo.real_leg_ids):
            fw = front_weights[i]
            rw = rear_weights[i]
            lw = left_weights[i]
            rtw = right_weights[i]

            max_w = max(fw, rw, lw, rtw)

            if max_w < 1e-9:
                group_b.append(lid)
                continue

            # 前节点主导 → group_a
            if fw == max_w:
                group_a.append(lid)
            # 后节点主导 → group_b
            elif rw == max_w:
                group_b.append(lid)
            # 左/右节点主导 → 根据横向位置分配到 a/b（保持数量平衡）
            else:
                if len(group_a) <= len(group_b):
                    group_a.append(lid)
                else:
                    group_b.append(lid)

        # 确保两组数量基本均衡（差距 > 1 时重新平衡）
        while len(group_a) > len(group_b) + 1:
            # 将 group_a 中贡献最小的腿移到 group_b
            min_leg = min(
                group_a,
                key=lambda lid: T[VIRTUAL_NODES["front"], self._topo.real_leg_ids.index(lid)]
            )
            group_a.remove(min_leg)
            group_b.append(min_leg)

        while len(group_b) > len(group_a) + 1:
            min_leg = min(
                group_b,
                key=lambda lid: T[VIRTUAL_NODES["rear"], self._topo.real_leg_ids.index(lid)]
            )
            group_b.remove(min_leg)
            group_a.append(min_leg)

        return group_a, group_b, group_c

    def get_topology_summary(self) -> Dict:
        """返回拓扑摘要（用于诊断）。"""
        T = self._T
        summary = {
            "n_real_legs": self._topo.n_real_legs,
            "morphology": self._topo.description,
            "assignment_matrix_shape": list(T.shape),
            "virtual_polygon": {
                "front": self._topo.virtual_polygon.front_xy.tolist(),
                "rear": self._topo.virtual_polygon.rear_xy.tolist(),
                "left": self._topo.virtual_polygon.left_xy.tolist(),
                "right": self._topo.virtual_polygon.right_xy.tolist(),
            },
            "per_leg_assignments": {},
        }
        node_names = ["front", "rear", "left", "right"]
        for i, lid in enumerate(self._topo.real_leg_ids):
            weights = {node_names[v]: float(T[v, i]) for v in range(N_VIRTUAL)}
            dominant = max(weights, key=weights.get)
            summary["per_leg_assignments"][str(lid)] = {
                "weights": weights,
                "dominant_node": dominant,
            }
        return summary


# ---------------------------------------------------------------------------
# 3. 零样本步态生成接口
# ---------------------------------------------------------------------------

def zero_shot_gait_plan(
    description: Dict,
    desired_heading: Optional[List[float]] = None,
    desired_speed: float = 0.15,
    frequency_hz: float = 0.85,
    duty_factor: float = 0.60,
    sigma: float = 0.0,
    state: Optional[Dict] = None,
) -> Dict:
    """
    零样本步态生成：无需预设任何形态假设，
    仅根据机器人描述 JSON 自动生成完整步态计划。

    与 adaptation.gait.compute_adaptive_plan() 的区别
    ------------------------------------------------
    · 不依赖启发式腿组分组（极坐标轮流分配）
    · 通过拓扑不变性矩阵自动推导最优腿组
    · 相位由虚拟节点反映射计算，而非固定的 0/π 偏移
    · 支持任意腿数（5、7、9、10 等奇偶数）

    Parameters
    ----------
    description    : robot_description.json 内容
    desired_heading: 期望前进方向（2D），None 则自动计算
    desired_speed  : 期望速度 (m/s)，用于调整步幅
    frequency_hz   : 步态频率 (Hz)
    duty_factor    : 基础占空比
    sigma          : 拓扑映射 RBF 带宽

    Returns
    -------
    Dict  兼容 adaptation.gait 输出格式的步态计划
    """
    state = state or {}
    missing_ids = {int(v) for v in state.get("missing_leg_ids", [])}
    locked_ids = {int(v) for v in state.get("locked_leg_ids", [])}

    # 步骤 1：从描述中提取足端位置和质心
    foot_positions_xy: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
        leg_id = int(link["leg_id"])
        if leg_id not in missing_ids:
            foot_positions_xy[leg_id] = origin[:2]

    # 计算质心
    com_xy = np.zeros(2, dtype=float)
    total_mass = 0.0
    for link in description.get("links", []):
        mp = link.get("mass_properties", {})
        mass = float(mp.get("mass", 0.0))
        if mass <= 0:
            continue
        origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
        cm_local = np.asarray(mp.get("center_mass", [0., 0., 0.]), dtype=float)
        com_xy += mass * (origin + cm_local)[:2]
        total_mass += mass
    if total_mass > 1e-9:
        com_xy /= total_mass

    # 步骤 2：确定前进方向（若未指定则使用 PCA）
    if desired_heading is not None:
        fwd = np.asarray(desired_heading[:2], dtype=float)
        fwd_n = float(np.linalg.norm(fwd))
        forward_axis = fwd / fwd_n if fwd_n > 1e-9 else np.array([1., 0.])
    else:
        # 简单 PCA：用足端位置协方差矩阵的主轴
        if len(foot_positions_xy) >= 2:
            pts = np.array(list(foot_positions_xy.values()), dtype=float)
            pts_centered = pts - pts.mean(axis=0)
            cov = pts_centered.T @ pts_centered
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            idx_max = int(np.argmax(eigenvals))
            forward_axis = eigenvecs[:, idx_max].copy()
            # 确保朝 +X 方向
            if forward_axis[0] < 0:
                forward_axis = -forward_axis
        else:
            forward_axis = np.array([1., 0.])

    # 步骤 3：构建拓扑映射器
    if not foot_positions_xy:
        return _empty_gait_plan()

    mapper = TopologyInvariantMapper(
        foot_positions_xy, com_xy, forward_axis, sigma=sigma
    )
    topology = mapper.build_morphology_topology()
    controller = VirtualLegController(topology)

    # 步骤 4：生成腿组分配
    group_a, group_b, group_c = controller.generate_alternating_groups()
    group_a = [lid for lid in group_a if lid not in locked_ids]
    group_b = [lid for lid in group_b if lid not in locked_ids]
    group_c = sorted(set(group_c) | (locked_ids & set(foot_positions_xy)))

    # 步骤 5：生成虚拟相位和步幅
    # 标准交替步态：前虚拟节点相位=0, 后虚拟节点相位=π
    real_phases = controller.virtual_to_real_phases(
        virtual_phase_front=0.0,
        virtual_phase_rear=math.pi,
        virtual_phase_left=math.pi / 2,
        virtual_phase_right=3 * math.pi / 2,
    )

    # 步幅：根据期望速度和频率计算
    stride_length = desired_speed / max(frequency_hz, 0.01)
    # 虚拟步幅分配：前后主驱动，左右稍小
    virtual_amp = np.array([
        min(stride_length * 2.5, 0.75),  # front
        min(stride_length * 2.5, 0.75),  # rear
        min(stride_length * 1.5, 0.50),  # left
        min(stride_length * 1.5, 0.50),  # right
    ], dtype=float)
    real_amps = controller.virtual_to_real_amplitudes(virtual_amp)

    # 步骤 6：构造 per_leg_stride_amplitudes（与 adaptation.gait 格式兼容）
    per_leg_stride_amplitudes: Dict[str, float] = {
        str(lid): real_amps.get(lid, 0.5)
        for lid in foot_positions_xy
    }

    # 步骤 7：支撑多边形
    all_feet = np.array(list(foot_positions_xy.values()), dtype=float)
    try:
        from shapely.geometry import MultiPoint
        hull = MultiPoint(all_feet.tolist()).convex_hull
        if hull.geom_type == "Polygon":
            support_polygon = list(hull.exterior.coords[:-1])
        else:
            support_polygon = all_feet.tolist()
    except Exception:
        support_polygon = all_feet.tolist()

    # 步骤 8：组装步态计划（兼容 adaptation.gait 输出格式）
    cos_xy = all_feet.mean(axis=0) if len(all_feet) > 0 else np.zeros(2)
    lat_axis = np.array([-forward_axis[1], forward_axis[0]], dtype=float)

    topo_summary = controller.get_topology_summary()

    active_phase_ids = sorted(set(group_a) | set(group_b))
    phase_offsets = {
        str(lid): wrap_2pi(float(real_phases.get(lid, 0.0)))
        for lid in active_phase_ids
    }
    cpg_state = state.get("cpg", {}) if isinstance(state.get("cpg", {}), dict) else {}
    configured_strategy = str(cpg_state.get("phase_strategy", "")).strip().lower()
    configured_wave_count = cpg_state.get("wave_count", None)
    selected_wave_count = (
        0.0 if configured_wave_count is None and configured_strategy == "adaptive_wave"
        else float(configured_wave_count if configured_wave_count is not None else 1.0)
    )
    phase_strategy = "topology_invariant"
    if configured_strategy in PHASE_STRATEGIES:
        phase_strategy = str(configured_strategy)
        generated, selected_wave_count = build_phase_offsets(
            strategy=phase_strategy,
            foot_positions=foot_positions_xy,
            forward_axis=forward_axis,
            groups={"group_a": group_a, "group_b": group_b, "group_c": group_c},
            active_leg_ids=active_phase_ids,
            missing_leg_ids=missing_ids,
            locked_leg_ids=locked_ids,
            wave_count=selected_wave_count,
            wave_direction=float(cpg_state.get("wave_direction", 1.0)),
            lateral_phase_lag=float(cpg_state.get("lateral_phase_lag", math.pi)),
        )
        phase_offsets = {str(lid): wrap_2pi(value) for lid, value in generated.items()}
    configured_offsets = cpg_state.get("phase_offsets", {})
    if isinstance(configured_offsets, dict):
        for raw_leg_id, raw_offset in configured_offsets.items():
            try:
                leg_id = int(raw_leg_id)
                if leg_id in active_phase_ids:
                    phase_offsets[str(leg_id)] = wrap_2pi(float(raw_offset))
            except (TypeError, ValueError):
                continue
    requested_duty = float(
        state.get("cpg", {}).get("duty_factor", duty_factor)
        if isinstance(state.get("cpg", {}), dict) else duty_factor
    )
    resolved_duty, duty_clipped = clip_duty_factor(requested_duty)
    per_leg_duties, per_leg_diagnostics = resolve_duty_factors(
        {"cpg": {
            "duty_factor": resolved_duty,
            "per_leg_duty_factors": cpg_state.get("per_leg_duty_factors", {}),
        }},
        active_phase_ids,
    )
    duty_diagnostics = (
        [f"cpg.duty_factor clipped from {requested_duty:.6g} to {resolved_duty:.6g}"]
        if duty_clipped else []
    ) + per_leg_diagnostics
    physical_leg_ids = sorted(foot_positions_xy)
    zeroed_edges = [
        {"leg_i": int(a), "leg_j": int(b), "reason": "locked"}
        for index, a in enumerate(physical_leg_ids)
        for b in physical_leg_ids[index + 1:]
        if a in locked_ids or b in locked_ids
    ]

    plan = {
        # 兼容字段
        "support_center_xy": cos_xy.tolist(),
        "projected_com_xy": com_xy.tolist(),
        "initial_virtual_forward_axis": forward_axis.tolist(),
        "final_forward_axis": forward_axis.tolist(),
        "drive_resultant_xy": (forward_axis * desired_speed).tolist(),
        "direction_scores": {"positive": 1.0, "negative": 0.0},
        "support_polygon_xy": support_polygon,
        "safety_corridor_xy": [],
        "translational_compensation_xy": [0.0, 0.0],
        "planned_swings": {},
        "support_leg_ids": list(foot_positions_xy.keys()),
        "near_stance_leg_ids": [],
        "yaw_balance": {
            "net_yaw_nominal": 0.0,
            "net_yaw_corrected": 0.0,
            "psi_by_leg": {},
            "yaw_levers": {},
        },
        "cpg": {
            "phase_strategy": phase_strategy,
            "active_leg_ids": active_phase_ids,
            "phase_offsets": phase_offsets,
            "per_leg_duty_factors": {
                str(lid): value for lid, value in per_leg_duties.items()
            },
            "frequency_hz": frequency_hz,
            "omega": 2.0 * math.pi * frequency_hz,
            "duty_factor": resolved_duty,
            "duty_factor_requested": requested_duty,
            "duty_factor_diagnostics": duty_diagnostics,
            "wave_count": selected_wave_count,
            "wave_direction": float(cpg_state.get("wave_direction", 1.0)),
            "lateral_phase_lag": float(cpg_state.get("lateral_phase_lag", math.pi)),
        },
        "impedance": {
            "space": "joint",
            "stance_kp": 180.0,
            "stance_kd": 12.0,
            "swing_kp": 80.0,
            "swing_kd": 6.0,
            "max_deflection": 0.05,
        },
        "topology": {
            "groups": {
                "group_a": group_a,
                "group_b": group_b,
                "group_c": group_c,
            },
            "phase_offsets": {"group_a": 0.0, "group_b": math.pi},
            "per_leg_stride_amplitudes": per_leg_stride_amplitudes,
            "swing_projections": {str(lid): 1.0 for lid in foot_positions_xy},
            "inhibition_rules": [
                {"leg_id": lid, "reason": "locked", "in_degree": 0.0, "out_degree": 0.0}
                for lid in sorted(locked_ids & set(foot_positions_xy))
            ] + [
                {"leg_id": lid, "reason": "missing", "in_degree": 0.0, "out_degree": 0.0}
                for lid in sorted(missing_ids)
            ],
            "coupling_matrix_zeroed_edges": zeroed_edges,
        },
        # 拓扑不变性扩展字段
        "topology_invariant": {
            "n_virtual_nodes": N_VIRTUAL,
            "real_leg_ids": list(foot_positions_xy.keys()),
            "topology_matrix": topology.assignment_matrix.tolist(),
            "virtual_polygon": topo_summary["virtual_polygon"],
            "per_leg_topology": topo_summary["per_leg_assignments"],
            "morphology_description": topology.description,
        },
    }

    print(
        f"[TopoMapper] 零样本步态生成完成: {topology.description}  "
        f"group_a={group_a}  group_b={group_b}  group_c={group_c}"
    )

    return plan


def _empty_gait_plan() -> Dict:
    """返回空步态计划（没有腿时的退化情况）。"""
    return {
        "support_center_xy": [0.0, 0.0],
        "projected_com_xy": [0.0, 0.0],
        "initial_virtual_forward_axis": [1.0, 0.0],
        "final_forward_axis": [1.0, 0.0],
        "drive_resultant_xy": [0.0, 0.0],
        "direction_scores": {"positive": 0.0, "negative": 0.0},
        "support_polygon_xy": [],
        "safety_corridor_xy": [],
        "translational_compensation_xy": [0.0, 0.0],
        "planned_swings": {},
        "support_leg_ids": [],
        "near_stance_leg_ids": [],
        "yaw_balance": {"net_yaw_nominal": 0.0, "net_yaw_corrected": 0.0, "psi_by_leg": {}, "yaw_levers": {}},
        "cpg": {"active_leg_ids": [], "phase_offsets": {}, "frequency_hz": 1.0, "omega": 6.28, "duty_factor": 0.6},
        "impedance": {"space": "joint", "stance_kp": 180.0, "stance_kd": 12.0, "swing_kp": 80.0, "swing_kd": 6.0, "max_deflection": 0.05},
        "topology": {"groups": {"group_a": [], "group_b": [], "group_c": []}, "phase_offsets": {"group_a": 0.0, "group_b": 3.14159}, "per_leg_stride_amplitudes": {}, "swing_projections": {}, "inhibition_rules": [], "coupling_matrix_zeroed_edges": []},
        "topology_invariant": {"n_virtual_nodes": N_VIRTUAL, "real_leg_ids": [], "topology_matrix": [], "virtual_polygon": {}, "per_leg_topology": {}, "morphology_description": "空"},
    }


# ---------------------------------------------------------------------------
# 工具：从现有 gait_plan 提取拓扑并可视化
# ---------------------------------------------------------------------------

def analyze_topology(description: Dict, gait_plan: Optional[Dict] = None) -> Dict:
    """
    分析机器人的拓扑不变性表示，返回诊断信息。

    Parameters
    ----------
    description : robot_description.json
    gait_plan   : 可选，若提供则使用其 final_forward_axis

    Returns
    -------
    Dict  诊断信息
    """
    foot_positions_xy: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
        foot_positions_xy[int(link["leg_id"])] = origin[:2]

    com_xy = np.zeros(2, dtype=float)
    total_mass = 0.0
    for link in description.get("links", []):
        mp = link.get("mass_properties", {})
        mass = float(mp.get("mass", 0.0))
        if mass <= 0:
            continue
        origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
        cm_local = np.asarray(mp.get("center_mass", [0., 0., 0.]), dtype=float)
        com_xy += mass * (origin + cm_local)[:2]
        total_mass += mass
    if total_mass > 1e-9:
        com_xy /= total_mass

    if gait_plan:
        forward_axis = np.asarray(
            gait_plan.get("final_forward_axis", [1., 0.]), dtype=float
        )
    else:
        forward_axis = np.array([1., 0.])

    if not foot_positions_xy:
        return {"error": "没有足端数据"}

    mapper = TopologyInvariantMapper(foot_positions_xy, com_xy, forward_axis)
    topology = mapper.build_morphology_topology()
    controller = VirtualLegController(topology)
    group_a, group_b, group_c = controller.generate_alternating_groups()
    summary = controller.get_topology_summary()

    result = {
        "n_real_legs": topology.n_real_legs,
        "morphology": topology.description,
        "forward_axis": forward_axis.tolist(),
        "topology_matrix": topology.assignment_matrix.tolist(),
        "virtual_polygon": summary["virtual_polygon"],
        "generated_groups": {
            "group_a": group_a,
            "group_b": group_b,
            "group_c": group_c,
        },
        "per_leg_assignments": summary["per_leg_assignments"],
    }
    return result


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="拓扑不变性虚拟足端映射器测试"
    )
    parser.add_argument("--description", type=Path,
                        default=Path("robot_assets/robot_description.json"))
    parser.add_argument("--mode",
                        choices=["analyze", "zero-shot"],
                        default="analyze",
                        help="analyze: 分析当前形态拓扑；zero-shot: 零样本生成步态")
    parser.add_argument("--frequency", type=float, default=0.85)
    parser.add_argument("--speed", type=float, default=0.15,
                        help="期望速度 (m/s)，用于零样本模式")
    parser.add_argument("--sigma", type=float, default=0.25,
                        help="RBF 带宽参数")
    args = parser.parse_args()

    desc = json.loads(args.description.read_text(encoding="utf-8"))

    if args.mode == "zero-shot":
        plan = zero_shot_gait_plan(
            desc,
            frequency_hz=args.frequency,
            desired_speed=args.speed,
            sigma=args.sigma,
        )
        print(f"\n[零样本步态计划]")
        print(f"  前进方向: {plan['final_forward_axis']}")
        print(f"  group_a: {plan['topology']['groups']['group_a']}")
        print(f"  group_b: {plan['topology']['groups']['group_b']}")
        print(f"  group_c: {plan['topology']['groups']['group_c']}")
        print(f"  步幅: { {k: round(v,3) for k,v in plan['topology']['per_leg_stride_amplitudes'].items()} }")
        topo_inv = plan.get("topology_invariant", {})
        print(f"  形态描述: {topo_inv.get('morphology_description', 'N/A')}")

    else:  # analyze
        result = analyze_topology(desc)
        print(f"\n[拓扑分析]")
        print(f"  腿数: {result['n_real_legs']}")
        print(f"  形态: {result['morphology']}")
        print(f"  前进轴: {result['forward_axis']}")
        print(f"  虚拟多边形: {result['virtual_polygon']}")
        print(f"  腿组: group_a={result['generated_groups']['group_a']}  "
              f"group_b={result['generated_groups']['group_b']}")
        print("\n  每腿拓扑分配:")
        for lid, info in sorted(result["per_leg_assignments"].items(), key=lambda x: int(x[0])):
            w = info["weights"]
            print(f"    leg {lid} → dominant: {info['dominant_node']:6s}  "
                  f"F={w['front']:.3f} R={w['rear']:.3f} L={w['left']:.3f} Ri={w['right']:.3f}")
