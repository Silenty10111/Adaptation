#!/usr/bin/env python3
"""
在线参数估计与快速运动自适应 (Online State Estimator)
=====================================================
关联论文: RMA: Rapid Motor Adaptation for Legged Robots
         (Ashish Kumar et al., RSS 2021)

核心思想
--------
RMA 的关键思想："在线估计外在环境/形态特征 (Extrinsics)"
可以直接融入启发式框架，无需依赖端到端强化学习。

在底层控制循环中加入轻量级观测器：
- 扩展卡尔曼滤波 (EKF)：在线估计质心偏移量 (CoM Offset)
- 递归最小二乘法 (RLS)：在线估计等效摩擦系数 (Effective Friction)
- 步态适应控制器 (GaitAdaptationController)：
  将估计的状态实时反馈给 CPG 节律发生器或 MPC 求解器

与静态重规划的区别
------------------
· 静态方法：检测到断腿 → 停下来 → 重新运行 compute_adaptive_plan() → 继续
· 本模块：连续在线估计，当断腿发生时，EKF 瞬间感受到运动学残差，
  自动更新质心偏移估计，并在 < 1 个步态周期内完成自适应调整。

模块结构
--------
1. EKFComOffsetEstimator    — EKF 估计质心偏移 (ΔCoM_x, ΔCoM_y)
2. RLSFrictionEstimator     — RLS 估计等效摩擦系数
3. LegHealthMonitor         — 腿健康状态检测（基于力矩残差）
4. GaitAdaptationController — 集成控制器，对接 adaptive_gait 输出
5. OnlineStateEstimator     — 顶层封装，一键对接仿真控制循环

关键参数
--------
所有参数均设有合理默认值，可直接用于 Isaac Gym 仿真中的实时控制循环。
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ObservationBundle:
    """
    一步仿真的完整观测量。

    所有字段均可为 None（表示未观测到），估计器会相应处理缺失数据。
    """
    # 机体位姿（世界坐标系）
    body_xy: Optional[np.ndarray] = None       # [x, y]
    body_z: Optional[float] = None             # z 高度
    body_roll: Optional[float] = None          # roll (rad)
    body_pitch: Optional[float] = None         # pitch (rad)
    body_yaw: Optional[float] = None           # yaw (rad)

    # 机体速度（世界坐标系）
    body_vx: Optional[float] = None
    body_vy: Optional[float] = None
    body_vz: Optional[float] = None

    # 关节力矩（每腿 {leg_id: [lift_torque, swing_torque, drop_torque]}）
    joint_torques: Optional[Dict[int, np.ndarray]] = None

    # 足端接触力（估计值或传感器值，每腿 {leg_id: F_normal}）
    contact_forces: Optional[Dict[int, float]] = None

    # 时间戳
    sim_time: float = 0.0
    dt: float = 1.0 / 60.0


@dataclass
class AdaptationState:
    """
    在线估计的机器人状态（用于反馈到 CPG/MPC）。
    """
    # EKF 估计的质心偏移（相对于 URDF 额定质心）
    com_offset_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    com_offset_cov: np.ndarray = field(
        default_factory=lambda: np.eye(2) * 0.01
    )

    # RLS 估计的等效摩擦系数
    effective_mu: float = 0.6
    mu_confidence: float = 0.5   # 0=低置信，1=高置信

    # 腿健康状态 {leg_id: health_score [0,1]}
    # 0=完全失效，1=正常
    leg_health: Dict[int, float] = field(default_factory=dict)

    # 失效腿 ID 列表（health < threshold 的腿）
    detected_failed_legs: List[int] = field(default_factory=list)

    # 步态频率自适应修正因子（当 CoM 偏移大时降速）
    frequency_scale: float = 1.0

    # 步幅自适应修正 {str(leg_id): scale_factor}
    adaptive_stride_scales: Dict[str, float] = field(default_factory=dict)

    # 偏航速率估计 (rad/s)
    estimated_yaw_rate: float = 0.0

    # 上次更新时间
    last_update_time: float = 0.0


# ---------------------------------------------------------------------------
# 1. EKF 质心偏移估计器
# ---------------------------------------------------------------------------

class EKFComOffsetEstimator:
    """
    扩展卡尔曼滤波器：在线估计机器人有效质心偏移。

    状态向量: x = [Δcx, Δcy, ω_yaw_bias]
        · Δcx, Δcy  — 质心偏移量（相对于 URDF 额定值）(m)
        · ω_yaw_bias — 偏航率偏置 (rad/s)

    观测方程: z = Hx + v
        · 观测量：实测偏航率变化 Δω_yaw（由仿真角速度计算）
        · H = [ψ₁/M, ψ₂/M, 1]（偏航杠杆加权质心偏移 → 偏航率）

    动力学方程: x_{t+1} = F x_t + w
        · F = I（质心偏移和偏置视为慢变过程）
        · 过程噪声 Q：量化形态突变时的跳变

    当断腿发生时
    -----------
    足端几何突变 → H 向量突变 → 新息（innovation）大幅增加
    → EKF 快速增大卡尔曼增益 → 快速跟踪新质心位置
    """

    def __init__(
        self,
        gait_plan: Dict,
        description: Dict,
        process_noise: float = 1e-4,
        obs_noise: float = 5e-3,
    ):
        """
        Parameters
        ----------
        gait_plan      : adaptive_gait 输出（用于提取足端几何）
        description    : robot_description.json
        process_noise  : 过程噪声方差（形态突变时需增大）
        obs_noise      : 观测噪声方差
        """
        self._plan = gait_plan
        self._desc = description

        # 状态向量维度: [Δcx, Δcy, ω_yaw_bias]
        self.n_state = 3
        self.x = np.zeros(self.n_state, dtype=float)   # 状态估计
        self.P = np.eye(self.n_state, dtype=float) * 0.01  # 协方差

        # 噪声
        self.Q = np.eye(self.n_state, dtype=float) * process_noise  # 过程噪声
        self.R = float(obs_noise)  # 观测噪声（标量）

        # 先验：额定质心（来自描述 JSON）
        self._nominal_com_xy = self._compute_nominal_com()

        # 足端位置和偏航杠杆
        self._foot_xy = self._extract_foot_positions()
        self._yaw_levers = self._compute_yaw_levers()

        # 历史偏航角（用于计算偏航率）
        self._prev_yaw: Optional[float] = None
        self._prev_time: float = 0.0

    def _compute_nominal_com(self) -> np.ndarray:
        com = np.zeros(2, dtype=float)
        total_mass = 0.0
        for link in self._desc.get("links", []):
            mp = link.get("mass_properties", {})
            mass = float(mp.get("mass", 0.0))
            if mass <= 0:
                continue
            origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
            cm_local = np.asarray(mp.get("center_mass", [0., 0., 0.]), dtype=float)
            com += mass * (origin + cm_local)[:2]
            total_mass += mass
        if total_mass > 1e-9:
            com /= total_mass
        return com

    def _extract_foot_positions(self) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for link in self._desc.get("links", []):
            if link.get("role") != "foot" or link.get("leg_id") is None:
                continue
            origin = np.asarray(link.get("default_world_origin", [0., 0., 0.]), dtype=float)
            result[int(link["leg_id"])] = origin[:2]
        return result

    def _compute_yaw_levers(self) -> Dict[int, float]:
        """
        计算每条腿的偏航杠杆: ψ_i = cross2d(r_i, fwd)
        r_i = foot_i - nominal_com
        """
        fwd = np.asarray(self._plan.get("final_forward_axis", [1., 0.]), dtype=float)
        fwd_n = float(np.linalg.norm(fwd))
        if fwd_n > 1e-9:
            fwd = fwd / fwd_n
        levers: Dict[int, float] = {}
        for lid, foot in self._foot_xy.items():
            r = foot - self._nominal_com_xy
            levers[lid] = float(r[0] * fwd[1] - r[1] * fwd[0])
        return levers

    def _build_observation_matrix(
        self, active_leg_ids: List[int], total_mass: float
    ) -> np.ndarray:
        """
        构造观测矩阵 H ∈ R^{1×3}。

        偏航动力学简化模型：
            ω̇_yaw ≈ (1/Izz) * Σ_i (F_i * ψ_i)  (步态推进偏航力矩)
                   + offset_term  (质心偏移导致的被动偏航)

        观测量 = ω_yaw_measured - ω_yaw_expected_from_plan
               ≈ (ψ_effective / Izz) * Δcx + 0 * Δcy + ω_yaw_bias

        简化：H ≈ [ψ_x_eff/Izz, ψ_y_eff/Izz, 1]
        """
        # 计算活跃腿的平均偏航杠杆（用于估计偏移对偏航的影响）
        psi_x = 0.0
        psi_y = 0.0
        n_active = max(len(active_leg_ids), 1)
        for lid in active_leg_ids:
            lever = self._yaw_levers.get(lid, 0.0)
            psi_x += lever
            psi_y += abs(lever)  # 用绝对值捕获横向效应

        Izz = max(total_mass * 0.04, 1e-3)  # 估计偏航惯量
        H = np.array([
            [psi_x / (Izz * n_active),
             psi_y / (Izz * n_active * 4.0),
             1.0]
        ], dtype=float)
        return H

    def update(
        self,
        obs: ObservationBundle,
        active_leg_ids: List[int],
        total_mass: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行一步 EKF 更新。

        Parameters
        ----------
        obs            : 当前仿真步的观测量
        active_leg_ids : 当前活跃（未失效）的腿 ID 列表
        total_mass     : 机器人总质量 (kg)

        Returns
        -------
        (com_offset_xy, P_2x2)  质心偏移估计和前两维协方差
        """
        dt = obs.dt

        # ---- EKF 预测步 -------------------------------------------------------
        # F = I (慢变过程)
        self.P = self.P + self.Q * dt

        # ---- EKF 更新步 -------------------------------------------------------
        # 计算偏航率（从仿真角度差分）
        yaw = obs.body_yaw
        if yaw is None or self._prev_yaw is None or dt < 1e-9:
            # 无观测：仅预测
            self._prev_yaw = yaw
            self._prev_time = obs.sim_time
            return self.x[:2].copy(), self.P[:2, :2].copy()

        # 偏航率（差分，带角度折叠处理）
        yaw_diff = yaw - self._prev_yaw
        # 处理 ±π 折叠
        if yaw_diff > math.pi:
            yaw_diff -= 2.0 * math.pi
        elif yaw_diff < -math.pi:
            yaw_diff += 2.0 * math.pi
        omega_yaw_measured = yaw_diff / dt

        self._prev_yaw = yaw
        self._prev_time = obs.sim_time

        # 观测矩阵
        H = self._build_observation_matrix(active_leg_ids, total_mass)

        # 新息 (innovation): z - Hx
        z = np.array([omega_yaw_measured], dtype=float)
        innov = z - (H @ self.x)

        # 创新协方差: S = H P H^T + R
        S = float((H @ self.P @ H.T)[0, 0]) + self.R

        # 卡尔曼增益: K = P H^T S^{-1}
        K = (self.P @ H.T) / max(S, 1e-9)  # (3, 1)

        # 状态更新
        self.x = self.x + K[:, 0] * float(innov[0])

        # 协方差更新 (Joseph form for numerical stability)
        I_KH = np.eye(self.n_state) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K * self.R @ K.T

        # 限制质心偏移估计幅度（防止发散）
        self.x[:2] = np.clip(self.x[:2], -0.30, 0.30)

        return self.x[:2].copy(), self.P[:2, :2].copy()

    def inject_morphology_change(self, new_active_leg_ids: List[int]) -> None:
        """
        形态突变注入（断腿时调用）。

        增大过程噪声，使 EKF 快速适应新质心。
        """
        # 临时增大过程噪声（注入不确定性）
        self.Q *= 50.0
        # 重置偏航偏置估计（断腿后偏置改变）
        self.x[2] = 0.0
        self.P[2, 2] = 0.1
        print(f"[EKF] 形态突变: 活跃腿 → {new_active_leg_ids}, 注入过程噪声")

    def decay_process_noise(self, factor: float = 0.98) -> None:
        """逐步恢复过程噪声到稳态值（断腿后调用）。"""
        Q_steady = 1e-4 * np.eye(self.n_state)
        self.Q = Q_steady + (self.Q - Q_steady) * factor


# ---------------------------------------------------------------------------
# 2. RLS 摩擦系数估计器
# ---------------------------------------------------------------------------

class RLSFrictionEstimator:
    """
    递归最小二乘法 (RLS) 在线估计等效摩擦系数。

    模型
    ----
    当腿处于支撑相时，足端水平力 F_xy ≤ μ * F_z（摩擦锥约束）。
    若观测到腿打滑（实际位移 > 预期 → 残差大），等效摩擦系数降低。

    简化在线估计：
        μ_eff = argmin Σ_t λ^(T-t) ||F_z_t - μ * F_xy_t||²
    其中 λ 为遗忘因子（λ < 1 → 近期数据权重更大）。

    当摩擦系数低时
    -----------
    · 减小步幅振幅（避免打滑加剧偏航）
    · 降低步态频率（增加支撑相时间，提高稳定性）
    """

    def __init__(
        self,
        forgetting_factor: float = 0.97,
        init_mu: float = 0.6,
        mu_min: float = 0.15,
        mu_max: float = 1.2,
    ):
        self._lambda = forgetting_factor
        self._mu = init_mu
        self._mu_min = mu_min
        self._mu_max = mu_max

        # RLS 内部状态
        self._P_rls = 1.0 / init_mu  # 初始估计协方差（标量，1D 模型）
        self._confidence = 0.3        # 初始低置信度

        # 历史缓冲（用于平滑）
        self._mu_history: Deque[float] = deque(maxlen=30)
        self._mu_history.append(init_mu)

    def update(
        self,
        contact_forces: Dict[int, float],  # {leg_id: F_normal}
        measured_slippage: Dict[int, float],  # {leg_id: slip_magnitude}
        n_contact_legs: int = 3,
    ) -> Tuple[float, float]:
        """
        RLS 更新一步。

        Parameters
        ----------
        contact_forces   : 每腿法向力（N）
        measured_slippage: 每腿打滑幅度（m，近似为水平力/法向力比）
        n_contact_legs   : 当前接触腿数

        Returns
        -------
        (mu_estimate, confidence)
        """
        if not contact_forces:
            return self._mu, self._confidence

        # 构造 RLS 观测：每腿产生一个样本
        # y = F_z * slip_ratio  (等效水平力)
        # x = F_z              (法向力)
        # 估计: μ = y / x
        for lid in contact_forces:
            F_z = max(float(contact_forces[lid]), 1.0)
            slip = float(measured_slippage.get(lid, 0.0))
            # 水平力 = 正向力 × 打滑比 × 调节系数
            F_xy_est = F_z * slip * 2.0  # 系数 2 是经验校正

            # y: 当前测量值（实际法向力 × 当前估计的摩擦要求）
            # RLS 更新：最小化 (μ * F_z - F_xy_est)²
            phi = F_z  # 回归器
            y = F_xy_est  # 目标

            # RLS 增益
            K_rls = self._P_rls * phi / (self._lambda + phi * self._P_rls * phi)
            # 更新估计
            innov = y - self._mu * phi
            self._mu += K_rls * innov
            self._mu = float(np.clip(self._mu, self._mu_min, self._mu_max))
            # 更新协方差
            self._P_rls = (1.0 / self._lambda) * (1.0 - K_rls * phi) * self._P_rls

            # 更新置信度（接触腿多 + 时间长 → 置信度高）
            self._confidence = float(np.clip(
                self._confidence * 0.995 + 0.005 * (n_contact_legs / 3.0),
                0.0, 1.0
            ))

        self._mu_history.append(self._mu)

        # 平滑输出（防止噪声抖动）
        mu_smooth = float(np.mean(self._mu_history))
        return mu_smooth, self._confidence

    def estimate(self) -> Tuple[float, float]:
        """返回当前估计值（不更新）。"""
        mu_smooth = float(np.mean(self._mu_history)) if self._mu_history else self._mu
        return mu_smooth, self._confidence


# ---------------------------------------------------------------------------
# 3. 腿健康状态监测器
# ---------------------------------------------------------------------------

class LegHealthMonitor:
    """
    通过关节力矩残差检测腿的健康状态。

    失效检测原理 (基于 RMA 外在估计思想)
    -------------------------------------
    正常腿在支撑相会产生大于阈值的关节力矩（承重）。
    失效腿（断裂、脱臼、电机故障）的力矩突降至接近零。

    健康得分:
        health_i(t) = EMA(|torque_i(t)| / τ_max_i)
        其中 EMA 是指数移动平均（时间常数 ~10 仿真步）

    检测阈值:
        若 health_i < FAILURE_THRESHOLD 持续 CONFIRM_STEPS 步 → 判定失效
    """

    FAILURE_THRESHOLD = 0.08  # 健康得分低于此值判为失效
    CONFIRM_STEPS = 5          # 连续失效步数才确认

    def __init__(
        self,
        leg_ids: List[int],
        ema_alpha: float = 0.1,
    ):
        """
        Parameters
        ----------
        leg_ids    : 所有腿的 ID 列表
        ema_alpha  : EMA 平滑系数（越小 = 越平滑，响应越慢）
        """
        self._leg_ids = leg_ids
        self._alpha = ema_alpha
        self._health: Dict[int, float] = {lid: 1.0 for lid in leg_ids}
        self._max_torques: Dict[int, float] = {lid: 1.0 for lid in leg_ids}
        self._failure_counts: Dict[int, int] = {lid: 0 for lid in leg_ids}
        self._confirmed_failures: set = set()
        self._step_count = 0

    def update(
        self,
        joint_torques: Dict[int, np.ndarray],  # {leg_id: [lift, swing, drop]}
        in_stance: Dict[int, bool],            # {leg_id: is_in_stance}
    ) -> Dict[int, float]:
        """
        更新健康得分。

        Returns
        -------
        Dict[int, float]  leg_id → 健康得分 [0, 1]
        """
        self._step_count += 1

        for lid in self._leg_ids:
            torque_vec = joint_torques.get(lid)
            if torque_vec is None:
                # 无数据 → 轻微降低健康
                self._health[lid] = self._health[lid] * (1.0 - self._alpha * 0.1)
                continue

            torque_mag = float(np.linalg.norm(torque_vec))

            # 只在支撑相检测（摆动相力矩小是正常的）
            if in_stance.get(lid, True):
                # 更新最大力矩（用于归一化）
                self._max_torques[lid] = max(self._max_torques[lid], torque_mag, 0.1)
                # 归一化力矩
                normalized = torque_mag / self._max_torques[lid]
                # EMA 更新健康得分
                self._health[lid] = (
                    (1 - self._alpha) * self._health[lid] +
                    self._alpha * normalized
                )
            else:
                # 摆动相：缓慢恢复健康（防止误判）
                self._health[lid] = min(1.0, self._health[lid] + self._alpha * 0.05)

            # 失效检测
            if self._health[lid] < self.FAILURE_THRESHOLD:
                self._failure_counts[lid] = self._failure_counts.get(lid, 0) + 1
                if self._failure_counts[lid] >= self.CONFIRM_STEPS:
                    if lid not in self._confirmed_failures:
                        self._confirmed_failures.add(lid)
                        print(f"[LegHealth] ⚠ leg {lid} 检测到失效! "
                              f"health={self._health[lid]:.3f}")
            else:
                self._failure_counts[lid] = 0
                if lid in self._confirmed_failures and self._health[lid] > 0.5:
                    self._confirmed_failures.discard(lid)
                    print(f"[LegHealth] ✓ leg {lid} 恢复正常, health={self._health[lid]:.3f}")

        return dict(self._health)

    def get_failed_legs(self) -> List[int]:
        """返回已确认失效的腿 ID 列表。"""
        return list(self._confirmed_failures)

    def get_health_scores(self) -> Dict[int, float]:
        """返回所有腿的当前健康得分。"""
        return dict(self._health)


# ---------------------------------------------------------------------------
# 4. 步态自适应控制器
# ---------------------------------------------------------------------------

class GaitAdaptationController:
    """
    集成控制器：将在线估计的状态转换为步态参数修正量。

    功能
    ----
    1. 接收 EKF 估计的质心偏移 → 修正步幅（消除偏航漂移）
    2. 接收 RLS 估计的摩擦系数 → 调整步态频率和步幅幅度
    3. 接收腿健康状态 → 实时触发腿组重规划（无需停止运动）
    4. 输出修正后的 per_leg_stride_amplitudes 和 frequency_scale
    """

    def __init__(
        self,
        gait_plan: Dict,
        # 增益参数
        com_offset_gain: float = 1.5,    # 质心偏移 → 步幅修正增益
        yaw_rate_gain: float = 0.8,      # 偏航率残差 → 步幅修正增益
        friction_freq_gain: float = 0.4, # 摩擦降低 → 频率修正增益
        max_amp_adj: float = 0.35,       # 最大步幅调整幅度
        max_freq_adj: float = 0.30,      # 最大频率调整幅度
    ):
        self._plan = gait_plan
        self._com_gain = com_offset_gain
        self._yaw_gain = yaw_rate_gain
        self._friction_gain = friction_freq_gain
        self._max_amp = max_amp_adj
        self._max_freq = max_freq_adj

        # 基础步幅（来自 gait_plan）
        topo = gait_plan.get("topology", {})
        self._base_amps: Dict[int, float] = {
            int(k): float(v)
            for k, v in topo.get("per_leg_stride_amplitudes", {}).items()
        }
        self._base_freq: float = float(
            gait_plan.get("cpg", {}).get("frequency_hz", 0.85)
        )

        # 前进轴和横向轴
        fwd = np.asarray(gait_plan.get("final_forward_axis", [1., 0.]), dtype=float)
        fwd_n = float(np.linalg.norm(fwd))
        self._fwd = fwd / fwd_n if fwd_n > 1e-9 else np.array([1., 0.])
        self._lat = np.array([-self._fwd[1], self._fwd[0]], dtype=float)

        # 足端位置
        self._foot_xy: Dict[int, np.ndarray] = {}

        # 上次使用的步幅（用于平滑过渡）
        self._current_amps: Dict[int, float] = dict(self._base_amps)
        self._current_freq_scale: float = 1.0

        # 平滑因子（EMA）
        self._smooth_alpha = 0.15

    def _get_foot_positions(self, description: Dict) -> Dict[int, np.ndarray]:
        if not self._foot_xy:
            for link in description.get("links", []):
                if link.get("role") != "foot" or link.get("leg_id") is None:
                    continue
                origin = np.asarray(
                    link.get("default_world_origin", [0., 0., 0.]), dtype=float
                )
                self._foot_xy[int(link["leg_id"])] = origin[:2]
        return self._foot_xy

    def compute_corrections(
        self,
        adaptation_state: AdaptationState,
        description: Dict,
    ) -> Tuple[Dict[str, float], float]:
        """
        根据在线估计状态计算步态参数修正量。

        Parameters
        ----------
        adaptation_state : 当前在线估计状态
        description      : robot_description.json

        Returns
        -------
        (corrected_stride_amps, corrected_frequency_scale)
        """
        foot_xy = self._get_foot_positions(description)
        com_offset = adaptation_state.com_offset_xy  # [Δcx, Δcy]

        # ---- 1. 质心偏移补偿 --------------------------------------------------
        # 若质心向左偏移，左侧腿增大步幅、右侧腿减小步幅（抵消偏航）
        # 偏航杠杆：r_i_cross = cross2d(foot_i - com, fwd)
        # 修正量：Δamp_i = -gain * dot(com_offset, lat_axis) * sign(yaw_lever_i)

        lat_com_offset = float(np.dot(com_offset, self._lat))
        fwd_com_offset = float(np.dot(com_offset, self._fwd))

        target_amps: Dict[int, float] = {}
        for lid, base_amp in self._base_amps.items():
            fp = foot_xy.get(lid, np.zeros(2))
            # 偏航杠杆（足端位置相对于质心的偏航力矩臂）
            # 使用名义质心（不加偏移）
            r = fp - np.zeros(2)  # 近似：质心在原点
            yaw_lever = float(r[0] * self._fwd[1] - r[1] * self._fwd[0])

            # 横向偏移修正：质心偏左 → 左腿（yaw_lever > 0）增大步幅
            correction_lat = (
                -self._com_gain * lat_com_offset * np.sign(yaw_lever)
            )

            # 偏航率修正：当前偏航率 → 对应腿侧步幅调整
            yaw_correction = (
                -self._yaw_gain * adaptation_state.estimated_yaw_rate * yaw_lever
            )

            delta = float(np.clip(correction_lat + yaw_correction,
                                  -self._max_amp, self._max_amp))
            target_amps[lid] = float(np.clip(base_amp + delta, 0.08, 0.92))

        # ---- 2. 摩擦系数补偿 --------------------------------------------------
        mu = adaptation_state.effective_mu
        mu_nominal = 0.6
        if mu < mu_nominal * 0.8:
            # 摩擦明显降低：减小步幅和频率
            friction_deficit = (mu_nominal - mu) / mu_nominal
            freq_scale_target = 1.0 - self._friction_gain * friction_deficit
        else:
            freq_scale_target = 1.0

        # ---- 3. 失效腿处理 ---------------------------------------------------
        for failed_lid in adaptation_state.detected_failed_legs:
            if failed_lid in target_amps:
                target_amps[failed_lid] = 0.0  # 失效腿步幅归零

        # ---- 4. EMA 平滑输出（防止突变冲击）----------------------------------
        alpha = self._smooth_alpha
        for lid in list(self._current_amps.keys()):
            tgt = target_amps.get(lid, self._base_amps.get(lid, 0.5))
            self._current_amps[lid] = (
                (1 - alpha) * self._current_amps[lid] + alpha * tgt
            )

        self._current_freq_scale = (
            (1 - alpha) * self._current_freq_scale + alpha * freq_scale_target
        )
        self._current_freq_scale = float(
            np.clip(self._current_freq_scale, 1.0 - self._max_freq, 1.0 + 0.15)
        )

        corrected = {str(lid): float(amp) for lid, amp in self._current_amps.items()}
        return corrected, self._current_freq_scale


# ---------------------------------------------------------------------------
# 5. 顶层封装：在线状态估计器
# ---------------------------------------------------------------------------

class OnlineStateEstimator:
    """
    顶层封装，一键对接仿真控制循环。

    使用方法
    --------
    1. 在仿真初始化时创建实例：
        estimator = OnlineStateEstimator(description, gait_plan)

    2. 在每个仿真步调用 step()：
        obs = ObservationBundle(body_yaw=yaw, sim_time=t, dt=dt)
        adaptation = estimator.step(obs)

    3. 将 adaptation 中的修正量应用到步态控制：
        per_leg_amps = adaptation.adaptive_stride_scales
        freq_scale = adaptation.frequency_scale

    4. 检测到断腿时调用 notify_leg_failure()：
        estimator.notify_leg_failure(broken_leg_id)
    """

    def __init__(
        self,
        description: Dict,
        gait_plan: Dict,
        total_mass: float = 5.0,
        process_noise: float = 1e-4,
        obs_noise: float = 5e-3,
        forgetting_factor: float = 0.97,
    ):
        topo = gait_plan.get("topology", {})
        groups = topo.get("groups", {})
        self._all_leg_ids = sorted(set(
            groups.get("group_a", []) +
            groups.get("group_b", []) +
            groups.get("group_c", [])
        ))

        self._desc = description
        self._plan = gait_plan
        self._total_mass = total_mass

        self._ekf = EKFComOffsetEstimator(
            gait_plan, description,
            process_noise=process_noise,
            obs_noise=obs_noise,
        )
        self._rls = RLSFrictionEstimator(forgetting_factor=forgetting_factor)
        self._health_monitor = LegHealthMonitor(self._all_leg_ids)
        self._gait_ctrl = GaitAdaptationController(gait_plan)

        self._state = AdaptationState()
        self._state.leg_health = {lid: 1.0 for lid in self._all_leg_ids}
        self._prev_yaw: Optional[float] = None
        self._step_count = 0

        # 活跃腿（未失效）
        self._active_leg_ids = list(self._all_leg_ids)

    def step(self, obs: ObservationBundle) -> AdaptationState:
        """
        执行一步在线估计。

        Parameters
        ----------
        obs : 当前仿真步的观测量

        Returns
        -------
        AdaptationState  最新自适应状态
        """
        self._step_count += 1

        # ---- EKF 更新 -------------------------------------------
        com_offset, cov = self._ekf.update(obs, self._active_leg_ids, self._total_mass)
        self._state.com_offset_xy = com_offset
        self._state.com_offset_cov = cov

        # 衰减过程噪声（断腿后注入的噪声逐步恢复）
        self._ekf.decay_process_noise()

        # ---- 偏航率估计 ------------------------------------------
        if obs.body_yaw is not None and self._prev_yaw is not None and obs.dt > 1e-9:
            yaw_diff = obs.body_yaw - self._prev_yaw
            if yaw_diff > math.pi:
                yaw_diff -= 2.0 * math.pi
            elif yaw_diff < -math.pi:
                yaw_diff += 2.0 * math.pi
            self._state.estimated_yaw_rate = float(yaw_diff / obs.dt)
        self._prev_yaw = obs.body_yaw

        # ---- RLS 更新 -------------------------------------------
        # 简化：用偏航率幅度作为打滑指标（实际应使用足端传感器）
        slippage_proxy: Dict[int, float] = {}
        yaw_rate_abs = abs(self._state.estimated_yaw_rate)
        for lid in self._active_leg_ids:
            slippage_proxy[lid] = min(yaw_rate_abs * 0.1, 0.5)

        contact_forces = obs.contact_forces or {}
        if not contact_forces:
            # 默认：均匀分配重力
            F_per_leg = self._total_mass * 9.81 / max(len(self._active_leg_ids), 1)
            contact_forces = {lid: F_per_leg for lid in self._active_leg_ids}

        mu_est, mu_conf = self._rls.update(
            contact_forces, slippage_proxy, n_contact_legs=len(self._active_leg_ids)
        )
        self._state.effective_mu = mu_est
        self._state.mu_confidence = mu_conf

        # ---- 腿健康监测 ------------------------------------------
        if obs.joint_torques:
            # 从步态相位估计哪些腿在支撑相
            freq = float(self._plan.get("cpg", {}).get("frequency_hz", 0.85))
            phase_now = 2.0 * math.pi * freq * obs.sim_time
            group_a = set(self._plan.get("topology", {}).get("groups", {}).get("group_a", []))
            group_b = set(self._plan.get("topology", {}).get("groups", {}).get("group_b", []))
            in_stance: Dict[int, bool] = {}
            for lid in self._all_leg_ids:
                if lid in group_b:
                    lg_phase = phase_now + math.pi
                else:
                    lg_phase = phase_now
                in_stance[lid] = math.sin(lg_phase) < 0.0  # 支撑相：sin < 0

            health_scores = self._health_monitor.update(obs.joint_torques, in_stance)
            self._state.leg_health = health_scores

            # 检测新的失效腿
            newly_failed = [
                lid for lid in self._health_monitor.get_failed_legs()
                if lid not in self._state.detected_failed_legs
            ]
            if newly_failed:
                for lid in newly_failed:
                    self.notify_leg_failure(lid)

        self._state.detected_failed_legs = self._health_monitor.get_failed_legs()

        # ---- 步态参数修正 -----------------------------------------
        corrected_amps, freq_scale = self._gait_ctrl.compute_corrections(
            self._state, self._desc
        )
        self._state.adaptive_stride_scales = corrected_amps
        self._state.frequency_scale = freq_scale
        self._state.last_update_time = obs.sim_time

        return self._state

    def notify_leg_failure(self, leg_id: int) -> None:
        """
        外部通知：腿 leg_id 已失效（人工注入或传感器确认）。

        自动触发：
        1. EKF 注入形态突变噪声
        2. 从活跃腿列表移除
        3. 打印诊断信息
        """
        if leg_id not in self._all_leg_ids:
            return
        if leg_id in self._active_leg_ids:
            self._active_leg_ids.remove(leg_id)

        self._ekf.inject_morphology_change(self._active_leg_ids)

        print(
            f"[Adaptation] 腿 {leg_id} 失效通知 → 活跃腿: {self._active_leg_ids}  "
            f"质心偏移估计: [{self._state.com_offset_xy[0]:.4f}, "
            f"{self._state.com_offset_xy[1]:.4f}] m"
        )

    def get_state(self) -> AdaptationState:
        """返回最新自适应状态（不更新）。"""
        return self._state

    def print_diagnostic(self) -> None:
        """打印当前诊断信息。"""
        s = self._state
        print(
            f"[OnlineEstimator] step={self._step_count}\n"
            f"  CoM offset:    [{s.com_offset_xy[0]:.4f}, {s.com_offset_xy[1]:.4f}] m\n"
            f"  Eff. friction: {s.effective_mu:.3f} (conf={s.mu_confidence:.2f})\n"
            f"  Yaw rate:      {math.degrees(s.estimated_yaw_rate):.2f} °/s\n"
            f"  Freq scale:    {s.frequency_scale:.3f}\n"
            f"  Failed legs:   {s.detected_failed_legs}\n"
            f"  Leg health:    { {lid: round(h,2) for lid,h in s.leg_health.items()} }\n"
            f"  Stride scales: { {k: round(v,3) for k,v in s.adaptive_stride_scales.items()} }"
        )


# ---------------------------------------------------------------------------
# 工具函数：从仿真状态快速构建 ObservationBundle
# ---------------------------------------------------------------------------

def make_observation(
    body_xy: List[float],
    body_attitude: Tuple[float, float, float],  # (roll, pitch, yaw)
    sim_time: float,
    dt: float = 1.0 / 60.0,
    joint_torques: Optional[Dict[int, np.ndarray]] = None,
    contact_forces: Optional[Dict[int, float]] = None,
) -> ObservationBundle:
    """快速构建 ObservationBundle，适用于 Isaac Gym 仿真循环。"""
    roll, pitch, yaw = body_attitude
    return ObservationBundle(
        body_xy=np.asarray(body_xy[:2], dtype=float),
        body_roll=float(roll),
        body_pitch=float(pitch),
        body_yaw=float(yaw),
        sim_time=float(sim_time),
        dt=float(dt),
        joint_torques=joint_torques,
        contact_forces=contact_forces,
    )


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="在线状态估计器仿真测试")
    parser.add_argument("--description", type=Path,
                        default=Path("robot_assets/robot_description.json"))
    parser.add_argument("--steps", type=int, default=200,
                        help="仿真步数")
    parser.add_argument("--break-leg", type=int, default=None,
                        help="在第 N 步使腿 ID 失效")
    parser.add_argument("--break-step", type=int, default=100,
                        help="断腿发生的仿真步")
    args = parser.parse_args()

    desc = json.loads(args.description.read_text(encoding="utf-8"))
    from adaptive_gait import compute_adaptive_plan
    plan = compute_adaptive_plan(desc, {})

    estimator = OnlineStateEstimator(desc, plan, total_mass=5.0)

    print(f"\n[测试] 运行 {args.steps} 步仿真...")
    dt = 1.0 / 60.0

    for step in range(args.steps):
        t = step * dt

        # 模拟偏航（直线行走时偏航率 ≈ 0，断腿后偏航率增大）
        if args.break_leg is not None and step == args.break_step:
            print(f"\n[测试] 第 {step} 步：注入断腿事件 (leg {args.break_leg})")

        # 模拟偏航角（无断腿时 ≈ 0，断腿后线性增大）
        if args.break_leg is not None and step > args.break_step:
            simulated_yaw = (step - args.break_step) * dt * 0.05
        else:
            simulated_yaw = 0.0

        obs = make_observation(
            body_xy=[t * 0.1, simulated_yaw * 0.5],
            body_attitude=(0.0, 0.0, simulated_yaw),
            sim_time=t,
            dt=dt,
        )

        if args.break_leg is not None and step == args.break_step:
            estimator.notify_leg_failure(args.break_leg)

        state = estimator.step(obs)

        # 每 50 步打印一次
        if step % 50 == 0 or (args.break_leg and step == args.break_step + 1):
            estimator.print_diagnostic()

    print("\n[测试完成]")
