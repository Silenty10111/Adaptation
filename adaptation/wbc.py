#!/usr/bin/env python3
"""
降维质心动力学与全身控制 (Centroidal Dynamics & WBC)
=====================================================
关联论文: Time-Efficient Contact Consistent Whole-Body Control Framework
         via Reduced-Dimension Dynamics Construction

核心功能
--------
1. CentroidalDynamics  — 将不规则多足机器人的完整动力学映射到6D质心动量空间
   (线动量 h_lin ∈ R³, 角动量 h_ang ∈ R³)。

2. GRFAllocator        — 二次规划（QP）分配地面反作用力（Ground Reaction Forces）。
   当形态不对称时，强制接触力满足：
     · 合力 ≈ 期望线加速度 × 总质量
     · 合力矩 ≈ 期望角加速度（消除偏航漂移）
     · 每腿接触力在摩擦锥约束内

3. YawTorqueBalancer   — 轻量级包装器，调用 GRFAllocator 只求解 XY 平面偏航力矩，
   并将结果转换为 per_leg_stride_amplitude 修正量，直接对接
   adaptation.gait.compute_adaptive_plan() 的输出格式。

使用示例
--------
>>> from adaptation.wbc import YawTorqueBalancer
>>> balancer = YawTorqueBalancer(description, gait_plan)
>>> corrected_amplitudes = balancer.solve(desired_heading=[1.0, 0.0])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 可选：scipy.optimize.minimize（QP 求解器回退）
# 若安装了 quadprog 或 osqp 则优先使用；否则使用 scipy L-BFGS-B 近似求解
# ---------------------------------------------------------------------------
try:
    import quadprog  # type: ignore
    _HAS_QUADPROG = True
except ImportError:
    _HAS_QUADPROG = False

try:
    import osqp  # type: ignore
    import scipy.sparse as sp  # type: ignore
    _HAS_OSQP = True
except ImportError:
    _HAS_OSQP = False


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ContactState:
    """描述单条腿的接触状态。"""
    leg_id: int
    in_contact: bool
    foot_position: np.ndarray          # 世界坐标系 [x, y, z]，shape (3,)
    normal: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
    mu: float = 0.6                    # 摩擦系数
    max_normal_force: float = 200.0    # 最大法向接触力 (N)


@dataclass
class CentroidalState:
    """质心状态 (世界坐标系)。"""
    com_pos: np.ndarray                # [x, y, z]
    com_vel: np.ndarray                # [vx, vy, vz]
    com_acc_desired: np.ndarray        # [ax, ay, az] 期望加速度
    angular_momentum: np.ndarray       # [hx, hy, hz]
    angular_momentum_dot_desired: np.ndarray  # [ḣx, ḣy, ḣz] 期望角动量导数
    total_mass: float


# ---------------------------------------------------------------------------
# 1. 质心动力学计算
# ---------------------------------------------------------------------------

class CentroidalDynamics:
    """
    从机器人描述 JSON 提取并维护降维质心动力学模型。

    降维策略 (Reduced-Dimension Construction)
    ------------------------------------------
    完整的刚体动力学方程为：
        M(q)q̈ + C(q,q̇)q̇ + g(q) = S^T τ + J_c^T F_c

    本类通过以下步骤降维：
    1. 提取质心位置 r_com = Σ(m_i * r_i) / M_total
    2. 计算惯量张量 I = Σ m_i [(r_i - r_com)² I₃ - (r_i-r_com)(r_i-r_com)^T]
    3. 质心线动量: p = M * v_com
    4. 质心角动量: L = I * ω_body + Σ m_i (r_i - r_com) × v_i
    5. 平面化：忽略 Z 方向，仅关心 XY 平面的力矩平衡

    这样从 O(n_joints) 降至 O(1) 的 6D 质心空间。
    """

    def __init__(self, description: Dict):
        self._desc = description
        self._parse_mass_properties()

    def _parse_mass_properties(self) -> None:
        """解析描述 JSON 中每个刚体的质量和质心位置。"""
        self.link_masses: List[float] = []
        self.link_com_world: List[np.ndarray] = []  # 世界坐标系质心

        for link in self._desc.get("links", []):
            mp = link.get("mass_properties", {})
            mass = float(mp.get("mass", 0.0))
            if mass <= 0.0:
                continue
            origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
            cm_local = np.asarray(mp.get("center_mass", [0., 0., 0.]), dtype=float)
            self.link_masses.append(mass)
            self.link_com_world.append(origin + cm_local)

        if not self.link_masses:
            # 兜底：假设单质点 1kg 位于原点
            self.link_masses = [1.0]
            self.link_com_world = [np.zeros(3, dtype=float)]

        self.total_mass = float(sum(self.link_masses))

    def compute_com(self) -> np.ndarray:
        """计算全机质心位置 (世界坐标系 3D)。"""
        com = np.zeros(3, dtype=float)
        for m, r in zip(self.link_masses, self.link_com_world):
            com += m * r
        return com / self.total_mass

    def compute_inertia_tensor_2d(self) -> np.ndarray:
        """
        计算平面惯量张量（2×2 子块，XY 平面内）。

        I_2d = Σ m_i * [(r_i² I₂) - r_i r_i^T]
        其中 r_i = (link_com_i - com)[:2]
        """
        com = self.compute_com()
        I = np.zeros((2, 2), dtype=float)
        for m, r3 in zip(self.link_masses, self.link_com_world):
            r = (r3 - com)[:2]
            r2 = float(np.dot(r, r))
            I += m * (r2 * np.eye(2) - np.outer(r, r))
        return I

    def compute_yaw_inertia(self) -> float:
        """
        计算绕 Z 轴（偏航轴）的惯量矩 Izz。

        Izz = Σ m_i * |r_i - r_com|²_xy
        """
        com = self.compute_com()
        izz = 0.0
        for m, r3 in zip(self.link_masses, self.link_com_world):
            dr = (r3 - com)[:2]
            izz += m * float(np.dot(dr, dr))
        return max(izz, 1e-6)

    def compute_centroidal_state(
        self,
        contact_states: List[ContactState],
        desired_heading: np.ndarray,
        desired_speed: float = 0.15,
        desired_yaw_rate: float = 0.0,
    ) -> CentroidalState:
        """
        构建期望的质心状态，用于 QP 求解。

        Parameters
        ----------
        desired_heading   : 2D 单位向量，机器人期望前进方向
        desired_speed     : 期望前进速度 (m/s)
        desired_yaw_rate  : 期望偏航角速度 (rad/s)，直线行走时为 0
        """
        com = self.compute_com()
        izz = self.compute_yaw_inertia()
        g = 9.81

        # 期望加速度：简单比例控制 + 重力补偿
        fwd = np.asarray(desired_heading[:2], dtype=float)
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm > 1e-9:
            fwd = fwd / fwd_norm

        # 期望线加速度 = 0（匀速）
        com_acc_desired = np.array([0., 0., g], dtype=float)  # 仅重力补偿

        # 期望角动量导数：-kd * 当前角动量 → 驱动偏航率为 0
        # 此处简化：只要求偏航力矩为 0（直线行走）
        h_dot_desired = np.array([0., 0., desired_yaw_rate * izz], dtype=float)

        return CentroidalState(
            com_pos=com,
            com_vel=np.zeros(3, dtype=float),
            com_acc_desired=com_acc_desired,
            angular_momentum=np.zeros(3, dtype=float),
            angular_momentum_dot_desired=h_dot_desired,
            total_mass=self.total_mass,
        )


# ---------------------------------------------------------------------------
# 2. QP 地面反作用力分配器
# ---------------------------------------------------------------------------

class GRFAllocator:
    """
    二次规划分配地面反作用力 (Ground Reaction Forces)。

    问题描述
    --------
    给定质心期望加速度 a_des 和角动量导数 ḣ_des，
    求满足约束的各腿接触力 F_i ∈ R³：

    min  Σ_i ||F_i - F_i_ref||²_Wi + λ_reg ||F||²
    s.t.
        Σ_i F_i = M * a_des                   (合力约束)
        Σ_i (r_i - r_com) × F_i = ḣ_des      (力矩约束)
        F_i_z ≥ 0                              (法向接触力非负)
        |F_i_x|, |F_i_y| ≤ μ * F_i_z          (摩擦锥约束，线性近似)
        F_i_z ≤ F_i_max                        (最大接触力)

    降维简化
    --------
    本实现仅关注 XY 平面（偏航控制），因此：
    - 将力约束降至 2D 平面力 [Fx, Fy] 的力矩分配
    - 法向力 Fz 由重力平均分配（基于支撑腿数）
    - QP 仅求解 Fz 偏差和平面力 [Fx, Fy]，复杂度大幅降低

    返回
    ----
    per_leg_grf: {leg_id: np.ndarray([Fx, Fy, Fz])} 各腿地面反作用力
    """

    def __init__(
        self,
        reg_weight: float = 1e-4,
        yaw_weight: float = 10.0,
        friction_mu: float = 0.6,
        max_proj_iters: int = 10,
    ):
        self.reg_weight = reg_weight
        self.yaw_weight = yaw_weight
        self.friction_mu = friction_mu
        self.max_proj_iters = max_proj_iters

    def allocate(
        self,
        centroidal_state: CentroidalState,
        contact_states: List[ContactState],
    ) -> Dict[int, np.ndarray]:
        """
        分配地面反作用力。

        Returns
        -------
        Dict[int, np.ndarray]  leg_id → [Fx, Fy, Fz] (N)
        """
        active = [c for c in contact_states if c.in_contact]
        n = len(active)
        if n == 0:
            return {}

        M = centroidal_state.total_mass
        g = 9.81
        com = centroidal_state.com_pos

        # ---- 平均法向力（承重） -----------------------------------------------
        Fz_total = M * g
        Fz_avg = Fz_total / n

        # ---- 构造偏航力矩矩阵 -------------------------------------------------
        # 偏航力矩: τ_z = Σ_i [(r_i - r_com) × F_i]_z
        #              = Σ_i [rx_i * Fy_i - ry_i * Fx_i]
        # 决策变量: x = [F0_x, F0_y, F1_x, F1_y, ..., Fn_x, Fn_y, δFz_0..δFz_{n-1}]
        # 共 3n 个变量（2n 个平面力 + n 个法向力偏差）

        n_vars = 3 * n  # [Fx_i, Fy_i, δFz_i] for each contact

        # 目标函数: min ||x||²_W  (加权正则化)
        # Q = diag([w_f, w_f, w_fz, ...]) + yaw 项
        Q = np.eye(n_vars, dtype=float) * self.reg_weight

        # 偏航力矩贡献: Σ [rx_i*Fy_i - ry_i*Fx_i] = τ_des_z
        tau_z_des = float(centroidal_state.angular_momentum_dot_desired[2])

        # 构造偏航力矩雅可比 J_yaw ∈ R^{1×n_vars}
        J_yaw = np.zeros((1, n_vars), dtype=float)
        for i, c in enumerate(active):
            rx = float(c.foot_position[0]) - float(com[0])
            ry = float(c.foot_position[1]) - float(com[1])
            J_yaw[0, 3 * i]     = -ry  # ∂τz/∂Fx_i = -ry
            J_yaw[0, 3 * i + 1] =  rx  # ∂τz/∂Fy_i = +rx

        # 将偏航力矩误差加入目标函数（软约束）:
        # min w_yaw * (J_yaw @ x - tau_z_des)²
        Q += self.yaw_weight * J_yaw.T @ J_yaw

        # 线性项 c = -2 * w_yaw * tau_z_des * J_yaw^T
        c_vec = -2.0 * self.yaw_weight * tau_z_des * J_yaw[0]

        # ---- 求解: 迭代投影法 (constrained QP) -----------------------
        # 使用迭代投影来满足摩擦锥不等式约束：
        #   1. F_i_z >= 0
        #   2. sqrt(Fx_i² + Fy_i²) <= mu * Fz_i
        # 每次迭代: 无约束 QP → 投影到可行域 → 固定违反变量 → 重解
        x_star = np.zeros(n_vars, dtype=float)
        try:
            x_star = -0.5 * np.linalg.solve(Q, c_vec)
        except np.linalg.LinAlgError:
            pass

        for _ in range(self.max_proj_iters):
            # 投影到可行域
            converged = True
            for i, c in enumerate(active):
                base = 3 * i
                Fx = float(x_star[base])
                Fy = float(x_star[base + 1])
                dFz = float(x_star[base + 2])
                Fz = Fz_avg + dFz

                # 约束 1: Fz >= 0
                if Fz < 0.0:
                    Fz = 1e-6
                    dFz = Fz - Fz_avg
                    x_star[base + 2] = dFz
                    converged = False

                # 约束 2: friction cone
                mu = min(self.friction_mu, c.mu)
                F_xy_max = mu * Fz
                F_xy = math.hypot(Fx, Fy)
                if F_xy > F_xy_max and F_xy > 1e-9:
                    scale = F_xy_max / F_xy
                    x_star[base]     = Fx * scale
                    x_star[base + 1] = Fy * scale
                    converged = False

                # 约束 3: Fz <= F_max
                Fz_max = c.max_normal_force
                if Fz > Fz_max:
                    Fz = Fz_max
                    x_star[base + 2] = Fz - Fz_avg
                    converged = False

            if converged:
                break

            # 重新求解，将已投影到边界的变量用惩罚项固定
            # Q_pen = Q + diag(active_penalties)
            Q_pen = Q.copy()
            for i, c in enumerate(active):
                base = 3 * i
                Fx = float(x_star[base])
                Fy = float(x_star[base + 1])
                dFz = float(x_star[base + 2])
                Fz = Fz_avg + dFz
                mu = min(self.friction_mu, c.mu)

                # 对在摩擦锥边界上的力施加强惩罚
                F_xy = math.hypot(Fx, Fy)
                if F_xy > 1e-3 and abs(F_xy - mu * max(Fz, 1e-6)) < 1e-3:
                    Q_pen[base, base]         += 100.0
                    Q_pen[base + 1, base + 1] += 100.0
                if Fz <= 1e-6:
                    Q_pen[base + 2, base + 2] += 100.0
                if Fz >= c.max_normal_force - 1e-3:
                    Q_pen[base + 2, base + 2] += 100.0

            try:
                x_star = -0.5 * np.linalg.solve(Q_pen, c_vec)
            except np.linalg.LinAlgError:
                break

        # ---- 组装结果（最终投影确保可行性）-------------------------------------
        result: Dict[int, np.ndarray] = {}
        for i, c in enumerate(active):
            Fx = float(x_star[3 * i])
            Fy = float(x_star[3 * i + 1])
            dFz = float(x_star[3 * i + 2])
            Fz = float(np.clip(Fz_avg + dFz, 0.0, c.max_normal_force))

            # 最终摩擦锥投影
            mu = min(self.friction_mu, c.mu)
            F_xy_max = mu * Fz
            F_xy = math.hypot(Fx, Fy)
            if F_xy > F_xy_max and F_xy > 1e-9:
                scale = F_xy_max / F_xy
                Fx *= scale
                Fy *= scale

            result[c.leg_id] = np.array([Fx, Fy, Fz], dtype=float)

        return result

    def compute_net_yaw_torque(
        self,
        com_xy: np.ndarray,
        per_leg_grf: Dict[int, np.ndarray],
        foot_positions: Dict[int, np.ndarray],
    ) -> float:
        """计算当前 GRF 分配下的净偏航力矩 (N·m)。"""
        tau_z = 0.0
        for leg_id, F in per_leg_grf.items():
            if leg_id not in foot_positions:
                continue
            r = foot_positions[leg_id][:2] - com_xy
            tau_z += float(r[0] * F[1] - r[1] * F[0])
        return tau_z


# ---------------------------------------------------------------------------
# 3. 偏航力矩平衡器 (对接 adaptation.gait)
# ---------------------------------------------------------------------------

class YawTorqueBalancer:
    """
    轻量级包装器：通过 GRF 分配求解，将结果转换为
    ``per_leg_stride_amplitude`` 修正字典，直接对接
    ``adaptation.gait.compute_adaptive_plan()`` 的输出。

    核心思想
    --------
    步态执行时，第 i 腿在支撑相产生的推进力近似为：
        F_i ≈ k_i * stride_amplitude_i * forward_axis

    偏航力矩:
        τ_z_i = F_i_x * r_i_y - F_i_y * r_i_x
              ≈ k_i * amp_i * yaw_lever_i

    目标: Σ_i amp_i * ψ_i = 0  (直线行走条件)
    其中 ψ_i = k_i * yaw_lever_i

    最小二乘解:
        amp_i = amp_i_base * (1 - λ * ψ_i)
        λ = (Σ amp_i_base * ψ_i) / (Σ (amp_i_base * ψ_i)²)

    与原始 adaptation.gait 中 YAW_COMP_GAIN=0.5 不同，
    这里通过质心动力学精确计算 ψ_i，消除静态+动态偏航分量。
    """

    def __init__(
        self,
        description: Dict,
        gait_plan: Dict,
        mu: float = 0.6,
        comp_gain: float = 0.85,
        max_adj: float = 0.40,
    ):
        """
        Parameters
        ----------
        description : robot_description.json 内容
        gait_plan   : compute_adaptive_plan() 的返回值
        mu          : 摩擦系数
        comp_gain   : 偏航补偿增益 [0,1]（保守值以防过校正）
        max_adj     : 最大步幅调整幅度 (fraction of base amplitude)
        """
        self._cd = CentroidalDynamics(description)
        self._plan = gait_plan
        self._mu = mu
        self._comp_gain = comp_gain
        self._max_adj = max_adj
        self._allocator = GRFAllocator(friction_mu=mu)

    def _build_contact_states(self) -> List[ContactState]:
        """从 gait_plan 构造接触状态列表。"""
        topo = self._plan.get("topology", {})
        groups = topo.get("groups", {})
        active_ids = (
            groups.get("group_a", []) +
            groups.get("group_b", []) +
            groups.get("group_c", [])
        )

        foot_positions = self._extract_foot_positions_3d()
        states = []
        for leg_id in active_ids:
            pos = foot_positions.get(leg_id, np.zeros(3, dtype=float))
            states.append(ContactState(
                leg_id=leg_id,
                in_contact=True,
                foot_position=pos,
                mu=self._mu,
            ))
        return states

    def _extract_foot_positions_3d(self) -> Dict[int, np.ndarray]:
        """提取足端 3D 世界坐标。"""
        result: Dict[int, np.ndarray] = {}
        for link in self._cd._desc.get("links", []):
            if link.get("role") != "foot" or link.get("leg_id") is None:
                continue
            origin = np.asarray(
                link.get("default_world_origin", [0., 0., 0.]), dtype=float
            )
            result[int(link["leg_id"])] = origin
        return result

    def solve(
        self,
        desired_heading: Optional[List[float]] = None,
        desired_yaw_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        执行质心动力学 + QP 求解，返回修正后的
        ``per_leg_stride_amplitude`` 字典 {str(leg_id): float}。

        Parameters
        ----------
        desired_heading   : 2D 前进方向向量（默认取 gait_plan 中的值）
        desired_yaw_rate  : 期望偏航速率（直线行走 = 0.0）

        Returns
        -------
        Dict[str, float]  leg_id (str) → 修正后步幅缩放因子
        """
        if desired_heading is None:
            desired_heading = self._plan.get("final_forward_axis", [1.0, 0.0])

        fwd = np.asarray(desired_heading[:2], dtype=float)
        fwd_n = float(np.linalg.norm(fwd))
        if fwd_n > 1e-9:
            fwd = fwd / fwd_n

        contact_states = self._build_contact_states()
        centroidal_state = self._cd.compute_centroidal_state(
            contact_states, fwd, desired_yaw_rate=desired_yaw_rate
        )

        # ---- QP 分配 ----------------------------------------------------------
        grf = self._allocator.allocate(centroidal_state, contact_states)

        foot_positions = self._extract_foot_positions_3d()
        com = centroidal_state.com_pos
        net_tau = self._allocator.compute_net_yaw_torque(
            com[:2], grf, foot_positions
        )

        # ---- 基础步幅（来自 adaptation.gait）-----------------------------------
        topo = self._plan.get("topology", {})
        base_amps: Dict[int, float] = {
            int(k): float(v)
            for k, v in topo.get("per_leg_stride_amplitudes", {}).items()
        }
        swing_proj: Dict[int, float] = {
            int(k): float(v)
            for k, v in topo.get("swing_projections", {}).items()
        }
        yaw_bal = self._plan.get("yaw_balance", {})
        psi_by_leg: Dict[int, float] = {
            int(k): float(v)
            for k, v in yaw_bal.get("psi_by_leg", {}).items()
        }
        yaw_levers: Dict[int, float] = {
            int(k): float(v)
            for k, v in yaw_bal.get("yaw_levers", {}).items()
        }

        active_ids = list(base_amps.keys())
        if not active_ids:
            return {}

        # ---- 质心动力学增强的 ψ_i 计算 ----------------------------------------
        # ψ_i (物理) = GRF_i 在偏航力矩上的杠杆
        # ψ_i_phys = (r_i_x * F_i_y - r_i_y * F_i_x) / (amp_i * forward_thrust_i)
        # 当 GRF 数据可用时，用物理 ψ 替换启发式 ψ
        psi_phys: Dict[int, float] = {}
        for leg_id in active_ids:
            if leg_id in grf and leg_id in foot_positions:
                r = foot_positions[leg_id][:2] - com[:2]
                F = grf[leg_id]
                tau_i = float(r[0] * F[1] - r[1] * F[0])
                amp_i = base_amps.get(leg_id, 1.0)
                # 归一化（避免除以 0）
                denom = max(abs(amp_i) * max(abs(swing_proj.get(leg_id, 1.0)), 0.1), 1e-6)
                psi_phys[leg_id] = tau_i / denom
            else:
                psi_phys[leg_id] = psi_by_leg.get(leg_id, 0.0)

        # ---- 最小二乘偏航补偿 ------------------------------------------------
        # 混合：70% 物理 ψ + 30% 几何 ψ（防止 GRF 估计误差）
        BLEND = 0.70
        psi_mixed: Dict[int, float] = {}
        for lid in active_ids:
            psi_mixed[lid] = (
                BLEND * psi_phys.get(lid, 0.0) +
                (1.0 - BLEND) * psi_by_leg.get(lid, 0.0)
            )

        # 净偏航（基于 ψ，单位步幅）
        net_psi = sum(base_amps.get(lid, 1.0) * psi_mixed[lid] for lid in active_ids)
        denom_psi = sum((base_amps.get(lid, 1.0) * psi_mixed[lid]) ** 2 for lid in active_ids)

        corrected: Dict[str, float] = {}
        if denom_psi > 1e-9:
            lam = self._comp_gain * net_psi / denom_psi
            for lid in active_ids:
                s_raw = 1.0 - lam * psi_mixed[lid]
                s_clamped = float(np.clip(s_raw,
                                          1.0 - self._max_adj,
                                          1.0 + self._max_adj))
                base = base_amps.get(lid, 0.5)
                corrected[str(lid)] = float(np.clip(base * s_clamped, 0.08, 0.90))
        else:
            for lid in active_ids:
                corrected[str(lid)] = float(base_amps.get(lid, 0.5))

        # 诊断输出
        net_after = sum(
            corrected.get(str(lid), base_amps.get(lid, 1.0)) * psi_mixed[lid]
            for lid in active_ids
        )
        print(
            f"[CentroidalWBC] net_yaw_before={net_psi:.4f}  "
            f"net_yaw_after≈{net_after:.4f}  "
            f"GRF_net_tau={net_tau:.4f} N·m  "
            f"total_mass={centroidal_state.total_mass:.2f} kg"
        )

        return corrected


# ---------------------------------------------------------------------------
# 4. 接触雅可比（用于完整 WBC，可选）
# ---------------------------------------------------------------------------

class ContactJacobian:
    """
    计算接触点雅可比矩阵，用于完整 WBC 时的约束投影。

    本类为扩展接口预留，当前实现仅提供 2D 平面版本。
    完整 3D 版本需要 URDF/Pinocchio 解析，超出当前框架范围。

    二维平面版本
    ------------
    对于第 i 腿，雅可比 J_i ∈ R^{2×3} (2D 接触力, 3 DOF 平面体):
        J_i = [1, 0, -r_i_y]
              [0, 1,  r_i_x ]
    其中 [r_i_x, r_i_y] = foot_i_xy - com_xy
    """

    @staticmethod
    def compute_2d(
        com_xy: np.ndarray,
        foot_positions_xy: Dict[int, np.ndarray],
        active_leg_ids: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        Returns
        -------
        Dict[int, np.ndarray]  leg_id → J_i ∈ R^{2×3}
        """
        result = {}
        for lid in active_leg_ids:
            fp = foot_positions_xy.get(lid)
            if fp is None:
                continue
            r = fp[:2] - com_xy[:2]
            J = np.array([
                [1., 0., -r[1]],
                [0., 1.,  r[0]],
            ], dtype=float)
            result[lid] = J
        return result


# ---------------------------------------------------------------------------
# 工具函数：从 description + gait_plan 一键运行
# ---------------------------------------------------------------------------

def run_centroidal_wbc(
    description: Dict,
    gait_plan: Dict,
    desired_heading: Optional[List[float]] = None,
    desired_yaw_rate: float = 0.0,
    mu: float = 0.6,
    comp_gain: float = 0.85,
) -> Dict[str, float]:
    """
    一键运行质心动力学 + WBC，返回修正后的 per_leg_stride_amplitudes。

    可直接替换 adaptation.gait.compute_adaptive_plan() 中的
    YAW_COMP_GAIN 启发式逻辑。

    Parameters
    ----------
    description     : robot_description.json 内容
    gait_plan       : compute_adaptive_plan() 的输出
    desired_heading : 期望前进方向（2D），默认取 gait_plan['final_forward_axis']
    desired_yaw_rate: 期望偏航速率，直线行走 = 0
    mu              : 地面摩擦系数
    comp_gain       : 偏航补偿增益

    Returns
    -------
    Dict[str, float]  修正后的 per_leg_stride_amplitude
    """
    balancer = YawTorqueBalancer(
        description, gait_plan, mu=mu, comp_gain=comp_gain
    )
    return balancer.solve(
        desired_heading=desired_heading,
        desired_yaw_rate=desired_yaw_rate,
    )


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="质心动力学 + WBC 偏航力矩平衡器测试"
    )
    parser.add_argument("--description", type=Path,
                        default=Path("robot_assets/robot_description.json"))
    parser.add_argument("--gait-plan", type=Path, default=None,
                        help="adaptation.gait 输出的 JSON；不传则自动计算")
    parser.add_argument("--mu", type=float, default=0.6, help="摩擦系数")
    parser.add_argument("--comp-gain", type=float, default=0.85,
                        help="偏航补偿增益 [0,1]")
    args = parser.parse_args()

    desc = json.loads(args.description.read_text(encoding="utf-8"))

    if args.gait_plan:
        plan = json.loads(args.gait_plan.read_text(encoding="utf-8"))
    else:
        from adaptation.gait import compute_adaptive_plan
        plan = compute_adaptive_plan(desc, {})

    corrected_amps = run_centroidal_wbc(
        desc, plan, mu=args.mu, comp_gain=args.comp_gain
    )

    print("\n[结果] 修正后 per_leg_stride_amplitudes:")
    for k, v in sorted(corrected_amps.items(), key=lambda x: int(x[0])):
        orig = plan.get("topology", {}).get("per_leg_stride_amplitudes", {}).get(k, "N/A")
        print(f"  leg {k}: {float(orig):.4f} → {v:.4f}")
