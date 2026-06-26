"""
srb_mpc_demo.py  —  SRB-MPC 端到端控制循环演示
=================================================
将以下模块串联成完整的控制流水线：
  plan_gait / adaptation.gait  →  步态规划（CPG 相位、分组）
  adaptation.mpc                 →  物理参数 + 自适应 Q/R 权重
  SRBMPCController           →  单刚体 MPC 求解（基于 numpy/scipy）

不依赖 Gazebo / Isaac Gym，可在纯 Python 环境中运行仿真。

运行：
    python srb_mpc_demo.py [--json robot_assets/robot_description.json] [--steps 200]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are  # 用于求解离散时间 Riccati 方程

from adaptation.mpc import (
    AdaptiveMPCWeights,
    MPCWeights,
    RobotPhysics,
    RobotPhysicsParser,
    build_mpc_params,
)

# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：旋转矩阵 / 斜对称矩阵
# ──────────────────────────────────────────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """将向量 v (3,) 转为斜对称矩阵 (3,3)。"""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def rz(yaw: float) -> np.ndarray:
    """绕 Z 轴旋转矩阵。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CPG 步态时序生成器
# ──────────────────────────────────────────────────────────────────────────────

class CPGScheduler:
    """
    基于 CPG（中枢模式发生器）相位偏移，在每个控制时步确定
    哪些腿处于支撑相（stance），哪些处于摆动相（swing）。
    """

    def __init__(
        self,
        leg_ids: List[int],
        phase_offsets: Dict[str, float],
        frequency_hz: float = 0.85,
        duty_factor: float = 0.60,
        dt: float = 0.02,
    ):
        self.leg_ids = leg_ids
        self.phase_offsets = {int(k): float(v) for k, v in phase_offsets.items()}
        self.frequency_hz = frequency_hz
        self.duty_factor = duty_factor
        self.dt = dt
        self.t = 0.0

    def step(self) -> Dict[int, str]:
        """
        推进一步，返回 {leg_id: 'stance' | 'swing'}。

        支撑相判断：归一化相位 ∈ [0, duty_factor)
        """
        phase_states: Dict[int, str] = {}
        omega = 2.0 * np.pi * self.frequency_hz
        for lid in self.leg_ids:
            offset = self.phase_offsets.get(lid, 0.0)
            phi = (omega * self.t + offset) % (2.0 * np.pi)
            normalized = phi / (2.0 * np.pi)   # ∈ [0, 1)
            phase_states[lid] = "stance" if normalized < self.duty_factor else "swing"
        self.t += self.dt
        return phase_states

    def reset(self):
        self.t = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. SRB（单刚体）状态空间离散化
# ──────────────────────────────────────────────────────────────────────────────

class SRBDynamics:
    """
    将单刚体动力学线性化并离散化，得到
        x_{k+1} = A_d x_k + B_d(p_feet, contact) u_k
    状态向量 x (13,):
        [roll, pitch, yaw, px, py, pz, droll, dpitch, dyaw, vx, vy, vz, g_placeholder]
    控制向量 u (n_contact * 3,): 每个接触腿的地面反力 [Fx, Fy, Fz]
    """

    def __init__(self, physics: RobotPhysics, dt: float = 0.02):
        self.m = physics.total_mass
        self.I_body = physics.inertia_tensor      # 3×3，body frame
        self.I_inv = np.linalg.inv(self.I_body)
        self.dt = dt
        self.g = 9.81
        self.n_state = 13

    def build_AB(
        self,
        yaw: float,
        foot_positions: np.ndarray,  # (n_contact, 3) 世界坐标
        com_pos: np.ndarray,          # (3,) 当前质心位置
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建连续时间系统矩阵 A_c (13,13) 和 B_c (13, n_contact*3)，
        然后用一阶 Euler 离散化得到 A_d, B_d。

        SRB-MPC 参考：Di Carlo et al., "Dynamic Locomotion in the MIT
        Cheetah 3 Through Convex Model-Predictive Control", IROS 2018.
        """
        n_c = len(foot_positions)
        R_world = rz(yaw)                       # 躯干到世界系的旋转
        I_world = R_world @ self.I_body @ R_world.T
        I_world_inv = np.linalg.inv(I_world)

        # ── 连续时间 A (13×13) ──────────────────────────────────────────────
        A_c = np.zeros((13, 13))
        # d(euler)/dt = R_euler_inv · omega  ≈ I (小角假设)
        A_c[0:3, 6:9] = np.eye(3)   # roll/pitch/yaw → angular velocity
        # d(pos)/dt = vel
        A_c[3:6, 9:12] = np.eye(3)
        # gravity in state
        A_c[11, 12] = 1.0           # vz += g * g_placeholder (g_ph = -g, so vz -= g)

        # ── 连续时间 B (13 × n_c*3) ─────────────────────────────────────────
        B_c = np.zeros((13, n_c * 3))
        for i, p_foot in enumerate(foot_positions):
            r = p_foot - com_pos               # 足端相对质心的向量
            # angular acceleration: I^{-1} (r × F)
            #   = I^{-1} skew(r) F
            col_start = i * 3
            B_c[6:9, col_start:col_start+3] = I_world_inv @ skew(r)
            # linear acceleration: F / m
            B_c[9:12, col_start:col_start+3] = np.eye(3) / self.m

        # ── 一阶 Euler 离散化 ────────────────────────────────────────────────
        A_d = np.eye(13) + A_c * self.dt
        B_d = B_c * self.dt

        return A_d, B_d

    def gravity_bias(self) -> np.ndarray:
        """重力偏置项（添加到状态更新中）。"""
        b = np.zeros(13)
        b[11] = -self.g * self.dt   # vz 受到重力影响
        return b


# ──────────────────────────────────────────────────────────────────────────────
# 3. SRB-MPC 控制器（有限时域 LQR / Receding Horizon）
# ──────────────────────────────────────────────────────────────────────────────

class SRBMPCController:
    """
    有限时域 SRB-MPC。

    每步计算：
      1. 根据当前步态相位确定接触腿集合
      2. 用接触腿足端位置构建 B_d
      3. 求解有限时域 LQR（近似 MPC via Riccati 递推）
      4. 返回第一步最优控制力 u* = [Fx_0, Fy_0, Fz_0, ...]

    注意：这里使用**定常** Riccati 解（DARE）近似滚动时域，
    等价于无限时域 LQR，是实际 MPC 的常见工程近似。
    如需真正有限时域，可替换为 _solve_finite_horizon_lqr()。
    """

    def __init__(
        self,
        physics: RobotPhysics,
        weights: MPCWeights,
        dt: float = 0.02,
        horizon: int = 10,
        mu_friction: float = 0.6,   # 摩擦系数（约束力锥）
        f_min: float = 10.0,        # 最小法向力 (N)
        f_max: float = 500.0,       # 最大法向力 (N)
    ):
        self.physics = physics
        self.weights = weights
        self.dt = dt
        self.horizon = horizon
        self.mu = mu_friction
        self.f_min = f_min
        self.f_max = f_max
        self.dynamics = SRBDynamics(physics, dt)

        # 反馈增益缓存（可选；此 demo 默认每步更新）
        self._K: Optional[np.ndarray] = None

    # ── 参考状态构造 ────────────────────────────────────────────────────────
    @staticmethod
    def make_reference(
        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
        px: float = 0.0,   py: float = 0.0,   pz: float = 0.35,
        vx: float = 0.0,   vy: float = 0.0,
    ) -> np.ndarray:
        """构建 13 维参考状态向量。"""
        x_ref = np.zeros(13)
        x_ref[0:3] = [roll, pitch, yaw]
        x_ref[3:6] = [px, py, pz]
        x_ref[12] = -9.81   # g placeholder（固定）
        return x_ref

    # ── 核心求解 ────────────────────────────────────────────────────────────
    def compute(
        self,
        x_cur: np.ndarray,           # (13,) 当前状态
        x_ref: np.ndarray,           # (13,) 参考状态
        contact_leg_ids: List[int],  # 当前处于支撑相的腿
        foot_positions: np.ndarray,  # (n_legs, 3) 所有腿足端位置（世界系）
    ) -> np.ndarray:
        """
        返回接触腿的最优地面反力向量 u* (n_contact*3,)，
        顺序与 contact_leg_ids 一致。
        """
        if len(contact_leg_ids) == 0:
            return np.zeros(0)

        # 提取接触腿足端位置
        contact_feet = foot_positions[contact_leg_ids]   # (n_c, 3)
        com_pos = x_cur[3:6]
        yaw = float(x_cur[2])

        A_d, B_d = self.dynamics.build_AB(yaw, contact_feet, com_pos)

        # ── 裁剪 R 矩阵至当前接触腿数 ─────────────────────────────────────
        n_c = len(contact_leg_ids)
        # 取 Q 前13维（全状态），R 取前 n_c*3 维
        Q = self.weights.Q
        R_full = self.weights.R
        r_dim = n_c * 3
        if r_dim <= R_full.shape[0]:
            R = R_full[:r_dim, :r_dim]
        else:
            # 腿数超出 R 矩阵维度时：用对角元素均值扩展
            r_scale = float(np.mean(np.diag(R_full)))
            R = np.eye(r_dim) * r_scale

        # ── DARE（近似 MPC 的无限时域 LQR）─────────────────────────────────
        # A_d / B_d 与质心位置、足端位置相关，因此每步更新更稳定。
        try:
            P = solve_discrete_are(A_d, B_d, Q, R)
            self._K = np.linalg.solve(
                R + B_d.T @ P @ B_d,
                B_d.T @ P @ A_d
            )  # (n_c*3, 13)
        except np.linalg.LinAlgError:
            # DARE 不收敛时，退化为零增益
            self._K = np.zeros((r_dim, 13))

        # ── 状态误差 + 重力/扭矩前馈 ──────────────────────────────────────
        e = x_cur - x_ref
        e[12] = 0.0   # g placeholder 不参与误差
        u_star = -self._K @ e

        # 静态平衡前馈：解出满足力/力矩平衡的最小范数足端力
        u_ff = self._solve_static_wrench(contact_feet, com_pos)
        u_star += u_ff

        # ── 约束剪裁（摩擦锥 + 法向力范围）─────────────────────────────────
        u_star = self._clip_forces(u_star, n_c)
        return u_star

    def _solve_static_wrench(
        self,
        contact_feet: np.ndarray,
        com_pos: np.ndarray,
    ) -> np.ndarray:
        """
        求解静态平衡足端力：
            sum(F) = [0, 0, m g]
            sum(r x F) = [0, 0, 0]
        使用最小范数解（least squares）。
        """
        n_c = len(contact_feet)
        if n_c == 0:
            return np.zeros(0)

        A = np.zeros((6, n_c * 3))
        for i, p_foot in enumerate(contact_feet):
            r = p_foot - com_pos
            A[0:3, i*3:i*3+3] = np.eye(3)
            A[3:6, i*3:i*3+3] = skew(r)

        b = np.zeros(6)
        b[2] = self.physics.total_mass * 9.81

        # 最小范数解（带轻微正则）
        reg = 1e-6
        AtA = A.T @ A + reg * np.eye(n_c * 3)
        u = np.linalg.solve(AtA, A.T @ b)
        return u

    def _clip_forces(self, u: np.ndarray, n_c: int) -> np.ndarray:
        """简单摩擦锥约束 + 法向力上下限。"""
        u = u.copy()
        for i in range(n_c):
            fz = np.clip(u[i*3+2], self.f_min, self.f_max)
            fx = np.clip(u[i*3+0], -self.mu * fz, self.mu * fz)
            fy = np.clip(u[i*3+1], -self.mu * fz, self.mu * fz)
            u[i*3:i*3+3] = [fx, fy, fz]
        return u


# ──────────────────────────────────────────────────────────────────────────────
# 4. 简单刚体仿真器（用于闭环验证）
# ──────────────────────────────────────────────────────────────────────────────

class SimpleSRBSimulator:
    """
    极简的单刚体动力学积分器，用于验证 MPC 控制效果。
    状态：[roll, pitch, yaw, px, py, pz, droll, dpitch, dyaw, vx, vy, vz, g_ph=-9.81]
    """

    def __init__(self, physics: RobotPhysics, dt: float = 0.02):
        self.m = physics.total_mass
        self.I_body = physics.inertia_tensor
        self.I_inv = np.linalg.inv(self.I_body)
        self.dt = dt
        self.g = 9.81

    def reset(self, x0: Optional[np.ndarray] = None, height: float = 0.35) -> np.ndarray:
        x = np.zeros(13)
        if x0 is not None:
            x[:] = x0
        else:
            x[5] = height
        x[12] = -self.g
        return x

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,          # (n_c*3,) 地面反力
        contact_feet: np.ndarray,  # (n_c, 3) 足端世界坐标
        com_pos: np.ndarray,    # (3,) 当前质心
    ) -> np.ndarray:
        """Euler 积分一步。"""
        roll, pitch, yaw = x[0], x[1], x[2]
        px, py, pz       = x[3], x[4], x[5]
        wr, wp, wy       = x[6], x[7], x[8]
        vx, vy, vz       = x[9], x[10], x[11]

        R_world = rz(yaw)
        I_w = R_world @ self.I_body @ R_world.T
        I_w_inv = np.linalg.inv(I_w)

        total_force  = np.zeros(3)
        total_torque = np.zeros(3)

        n_c = len(contact_feet)
        for i in range(n_c):
            fi = u[i*3:i*3+3] if len(u) >= (i+1)*3 else np.zeros(3)
            r  = contact_feet[i] - com_pos
            total_force  += fi
            total_torque += np.cross(r, fi)

        # 重力
        total_force[2] -= self.m * self.g

        # 线加速度
        a_lin = total_force / self.m

        # 角加速度（世界系）
        alpha_w = I_w_inv @ total_torque

        x_new = x.copy()
        x_new[0:3] += x[6:9] * self.dt
        x_new[3:6] += x[9:12] * self.dt
        x_new[6:9] += alpha_w * self.dt
        x_new[9:12] += a_lin * self.dt
        x_new[12] = -self.g   # 常数

        return x_new


# ──────────────────────────────────────────────────────────────────────────────
# 5. 端到端演示主函数
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(json_path: str, n_steps: int = 200, dt: float = 0.02):
    print("=" * 65)
    print("  SRB-MPC 端到端控制循环演示")
    print("=" * 65)

    # ── Step 1: 解析物理参数 & 自适应权重 ──────────────────────────────────
    print("\n[1] 加载物理参数与自适应 Q/R 权重...")
    physics, weights = build_mpc_params(json_path)
    print(f"    总质量          : {physics.total_mass:.3f} kg")
    print(f"    腿数            : {physics.num_legs}")
    print(f"    aspect_ratio    : {physics.aspect_ratio:.3f}")
    print(f"    Q[roll,pitch]   : {weights.Q[0,0]:.4f}")
    print(f"    Q[z]            : {weights.Q[5,5]:.4f}")
    print(f"    R shape         : {weights.R.shape}")

    # ── Step 2: 步态规划 ────────────────────────────────────────────────────
    print("\n[2] 调用步态规划器...")
    try:
        from adaptation.gait import compute_adaptive_plan
        with open(json_path) as f:
            description = json.load(f)
        plan = compute_adaptive_plan(description, {})
        cpg_cfg = plan["cpg"]
        phase_offsets = {int(k): float(v) for k, v in cpg_cfg["phase_offsets"].items()}
        freq_hz = cpg_cfg["frequency_hz"]
        duty_factor = cpg_cfg["duty_factor"]
        leg_ids = sorted(physics.foot_positions.keys())
        print(f"    group_a         : {plan['topology']['groups']['group_a']}")
        print(f"    group_b         : {plan['topology']['groups']['group_b']}")
        print(f"    CPG freq        : {freq_hz:.2f} Hz")
        print(f"    duty_factor     : {duty_factor:.2f}")
    except Exception as e:
        print(f"    [警告] 步态规划加载失败: {e}，使用默认参数")
        leg_ids = sorted(physics.foot_positions.keys())
        phase_offsets = {lid: (np.pi if lid % 2 == 1 else 0.0) for lid in leg_ids}
        freq_hz, duty_factor = 0.85, 0.60

    # ── Step 3: 初始化各模块 ────────────────────────────────────────────────
    foot_arr = physics.foot_positions_array          # (n_legs, 3) 默认足端位置
    cpg = CPGScheduler(leg_ids, phase_offsets, freq_hz, duty_factor, dt)
    mpc = SRBMPCController(physics, weights, dt, horizon=10)
    sim = SimpleSRBSimulator(physics, dt)

    # 初始状态：轻微俯仰扰动 5°，测试控制器的稳定能力
    x0 = sim.reset(height=physics.body_height)
    x0[1] = np.deg2rad(5.0)   # pitch 扰动
    x = x0.copy()

    x_ref = SRBMPCController.make_reference(pz=physics.body_height)

    print(f"\n[3] 初始状态: roll={np.rad2deg(x[0]):.2f}°  "
          f"pitch={np.rad2deg(x[1]):.2f}°  pz={x[5]:.3f}m")
    print(f"    参考状态: roll=0°  pitch=0°  pz={physics.body_height:.3f}m")

    # ── Step 4: 控制循环 ────────────────────────────────────────────────────
    print(f"\n[4] 运行 {n_steps} 步控制循环 (dt={dt}s, 总时长={n_steps*dt:.1f}s)...")
    log_t, log_roll, log_pitch, log_pz = [], [], [], []
    log_fz_total = []

    t0 = time.time()
    for step in range(n_steps):
        phase_states = cpg.step()
        contact_ids = [lid for lid, s in phase_states.items() if s == "stance"]

        if len(contact_ids) == 0:
            x = sim.step(x, np.zeros(0), np.zeros((0, 3)), x[3:6])
            fz_total = 0.0
        else:
            contact_feet = foot_arr[contact_ids]
            u = mpc.compute(x, x_ref, contact_ids, foot_arr)
            x = sim.step(x, u, contact_feet, x[3:6])
            fz_total = sum(u[i*3+2] for i in range(len(contact_ids)))

        log_t.append(step * dt)
        log_roll.append(np.rad2deg(x[0]))
        log_pitch.append(np.rad2deg(x[1]))
        log_pz.append(x[5])
        log_fz_total.append(fz_total)

        if step % 50 == 0:
            n_c = len(contact_ids)
            print(f"    t={step*dt:5.2f}s | roll={np.rad2deg(x[0]):+6.2f}°"
                  f"  pitch={np.rad2deg(x[1]):+6.2f}°"
                  f"  pz={x[5]:.3f}m"
                  f"  |contact={n_c}|")

    elapsed = time.time() - t0
    print(f"\n    循环完成，耗时 {elapsed:.3f}s ({elapsed/n_steps*1000:.2f}ms/step)")

    # ── Step 5: 统计结果 ────────────────────────────────────────────────────
    print("\n[5] 控制效果统计:")
    print(f"    Roll  RMSE  : {np.sqrt(np.mean(np.array(log_roll)**2)):.3f}°")
    print(f"    Pitch RMSE  : {np.sqrt(np.mean(np.array(log_pitch)**2)):.3f}°")
    pz_err = np.array(log_pz) - physics.body_height
    print(f"    pz    RMSE  : {np.sqrt(np.mean(pz_err**2))*100:.2f} cm")
    print(f"    末态 roll   : {log_roll[-1]:+.3f}°")
    print(f"    末态 pitch  : {log_pitch[-1]:+.3f}°")
    print(f"    末态 pz     : {log_pz[-1]:.4f} m")

    # ── Step 6: 可选绘图 ────────────────────────────────────────────────────
    _try_plot(log_t, log_roll, log_pitch, log_pz, physics.body_height)

    return {
        "roll_rmse_deg":  float(np.sqrt(np.mean(np.array(log_roll)**2))),
        "pitch_rmse_deg": float(np.sqrt(np.mean(np.array(log_pitch)**2))),
        "pz_rmse_m":      float(np.sqrt(np.mean(pz_err**2))),
        "final_roll_deg": float(log_roll[-1]),
        "final_pitch_deg": float(log_pitch[-1]),
        "final_pz_m":     float(log_pz[-1]),
    }


def _try_plot(t, roll, pitch, pz, pz_ref):
    """可选绘图（若 matplotlib 可用）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, roll,  color="tab:blue",   label="Roll (deg)")
        axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("Roll (°)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, pitch, color="tab:orange", label="Pitch (deg)")
        axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Pitch (°)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, pz,    color="tab:green",  label="pz (m)")
        axes[2].axhline(pz_ref, color="k", linestyle="--", linewidth=0.8, label="ref")
        axes[2].set_ylabel("Height (m)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("SRB-MPC Closed-Loop Response", fontsize=14)
        fig.tight_layout()
        out_path = "png/srb_mpc_response.png"
        import os; os.makedirs("png", exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\n    [绘图] 已保存至 {out_path}")
    except Exception as e:
        print(f"\n    [绘图] 跳过: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", default="robot_assets/robot_description.json",
        help="robot_description.json 路径"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="控制循环步数"
    )
    parser.add_argument(
        "--dt", type=float, default=0.02,
        help="控制时步 (s)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_demo(args.json, n_steps=args.steps, dt=args.dt)
    print("\n[返回结果 JSON]")
    print(json.dumps(result, indent=2))
