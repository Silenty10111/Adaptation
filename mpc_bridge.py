"""
mpc_bridge.py  —  动态物理参数解析与 SRB-MPC 桥接模块
=======================================================
功能：
  1. RobotPhysicsParser  —— 从 robot_description.json 或 URDF 解析
     总质量、质心、惯性张量、各腿足端初始坐标
  2. FootJacobianCalculator —— 使用 Pinocchio 在运行时计算足端雅可比
  3. AdaptiveMPCWeights     —— 根据几何/质量特征自适应生成 Q / R 权重矩阵

依赖：
  pip install numpy pin  (pinocchio 的 pip 包名为 pin)
  或 conda install -c conda-forge pinocchio

用法示例见文件末尾 __main__ 块。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 数据容器
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotPhysics:
    """SRB-MPC 所需的全部物理参数。"""
    total_mass: float                          # kg
    com_position: np.ndarray                   # (3,)  质心在 body 坐标系中的位置
    inertia_tensor: np.ndarray                 # (3,3) 绕质心的惯性张量，body 坐标系
    foot_positions: Dict[int, np.ndarray]      # {leg_id: (3,)} 足端初始世界坐标
    num_legs: int
    # 几何特征（由解析器填充，供权重自适应使用）
    body_half_span_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    body_height: float = 0.0                   # 站立时躯干到地面高度估算

    # ── 派生便利属性 ────────────────────────────────────────────────
    @property
    def aspect_ratio(self) -> float:
        """宽/高 比，值越大越"扁平"。"""
        span = float(np.linalg.norm(self.body_half_span_xy))
        if self.body_height < 1e-6:
            return 1.0
        return span / self.body_height

    @property
    def foot_positions_array(self) -> np.ndarray:
        """(n_legs, 3) 将所有足端坐标堆叠为数组。"""
        keys = sorted(self.foot_positions.keys())
        return np.stack([self.foot_positions[k] for k in keys])


# ──────────────────────────────────────────────────────────────────────────────
# 1. URDF / JSON 解析器
# ──────────────────────────────────────────────────────────────────────────────

class RobotPhysicsParser:
    """
    从 robot_description.json（本项目格式）或 URDF 中提取 RobotPhysics。

    优先使用 JSON（字段更直接；URDF 用 urdf_parser_py 作为后备）。
    """

    # ── JSON 接口 ──────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, json_path: str) -> RobotPhysics:
        """从项目的 robot_description.json 解析物理参数。"""
        with open(json_path, "r") as f:
            desc = json.load(f)

        links: List[dict] = desc["links"]

        # ── 累计全机质量与质心（并联惯性合成）──────────────────────────────
        total_mass = 0.0
        com_world = np.zeros(3)
        for lk in links:
            mp = lk["mass_properties"]
            m = mp["mass"]
            origin = np.array(lk["default_world_origin"], dtype=float)
            # center_mass 是相对于 default_world_origin 的局部偏移
            local_com = np.array(mp["center_mass"], dtype=float)
            total_mass += m
            com_world += m * (origin + local_com)

        if total_mass < 1e-12:
            raise ValueError("解析到的总质量为零，请检查 JSON 文件。")
        com_world /= total_mass

        # ── 并联惯性张量（平行轴定理） ─────────────────────────────────────
        I_total = np.zeros((3, 3))
        for lk in links:
            mp = lk["mass_properties"]
            m = mp["mass"]
            origin = np.array(lk["default_world_origin"], dtype=float)
            local_com = np.array(mp["center_mass"], dtype=float)
            r = (origin + local_com) - com_world          # 到系统质心的向量
            I_local = np.array(mp["inertia"], dtype=float)
            # 平行轴定理：I_sys += I_local + m*(r·r*E - r⊗r)
            I_total += I_local + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

        # ── 足端初始位置 ───────────────────────────────────────────────────
        foot_positions: Dict[int, np.ndarray] = {}
        for lk in links:
            if lk["role"] == "foot":
                leg_id = lk["leg_id"]
                origin = np.array(lk["default_world_origin"], dtype=float)
                local_com = np.array(lk["mass_properties"]["center_mass"], dtype=float)
                foot_positions[leg_id] = origin + local_com

        # ── 几何特征估算 ───────────────────────────────────────────────────
        foot_arr = np.stack(list(foot_positions.values()))  # (n, 3)
        half_span = (foot_arr[:, :2].max(axis=0) - foot_arr[:, :2].min(axis=0)) / 2.0
        # 站立高度 = |足端平均 Z| - 躯干 Z（躯干固定在原点）
        body_height = float(np.abs(foot_arr[:, 2].mean()))

        return RobotPhysics(
            total_mass=total_mass,
            com_position=com_world,
            inertia_tensor=I_total,
            foot_positions=foot_positions,
            num_legs=desc.get("num_legs", len(foot_positions)),
            body_half_span_xy=half_span,
            body_height=body_height,
        )

    # ── URDF 接口（后备） ──────────────────────────────────────────────────────

    @classmethod
    def from_urdf(cls, urdf_path: str) -> RobotPhysics:
        """
        从 URDF 解析物理参数（需要 urdf_parser_py）。
        pip install urdf_parser_py
        """
        try:
            from urdf_parser_py.urdf import URDF
        except ImportError as e:
            raise ImportError("请先安装 urdf_parser_py：pip install urdf_parser_py") from e

        robot = URDF.from_xml_file(urdf_path)

        total_mass = 0.0
        com_world = np.zeros(3)
        link_data = []

        for link in robot.links:
            if link.inertial is None:
                continue
            m = link.inertial.mass
            if m is None or m < 1e-12:
                continue
            # urdf_parser_py 的 origin 相对于 link frame；这里简化为仅用 xyz
            origin = np.zeros(3)
            if link.inertial.origin is not None:
                origin = np.array(link.inertial.origin.xyz, dtype=float)
            total_mass += m
            com_world += m * origin
            inr = link.inertial.inertia
            I_local = np.array([
                [inr.ixx, inr.ixy, inr.ixz],
                [inr.ixy, inr.iyy, inr.iyz],
                [inr.ixz, inr.iyz, inr.izz],
            ], dtype=float)
            link_data.append((m, origin, I_local))

        if total_mass < 1e-12:
            raise ValueError("URDF 中解析到的总质量为零。")
        com_world /= total_mass

        I_total = np.zeros((3, 3))
        for m, origin, I_local in link_data:
            r = origin - com_world
            I_total += I_local + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

        # URDF 中足端需要按命名约定查找（*foot* link 且无子 link）
        foot_positions: Dict[int, np.ndarray] = {}
        child_links = {j.child.link for j in robot.joints}
        leg_id = 0
        for link in robot.links:
            if "foot" in link.name.lower() and link.name not in child_links:
                foot_positions[leg_id] = np.zeros(3)  # URDF 中需要 FK，这里留零
                leg_id += 1

        return RobotPhysics(
            total_mass=total_mass,
            com_position=com_world,
            inertia_tensor=I_total,
            foot_positions=foot_positions,
            num_legs=len(foot_positions),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. 足端雅可比（Pinocchio）
# ──────────────────────────────────────────────────────────────────────────────

class FootJacobianCalculator:
    """
    使用 Pinocchio 实时加载 URDF 并计算足端雅可比矩阵。

    例：
        calc = FootJacobianCalculator("robot_assets/variants/variant_00_seed7/robot.urdf")
        q = calc.neutral_config()
        J = calc.compute_foot_jacobians(q)   # {frame_name: (6, nv) 矩阵}
    """

    def __init__(self, urdf_path: str, foot_keyword: str = "foot"):
        """
        Parameters
        ----------
        urdf_path    : URDF 文件路径
        foot_keyword : 足端 frame/link 名称中包含的关键词（不区分大小写）
        """
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "请先安装 Pinocchio：conda install -c conda-forge pinocchio  或  pip install pin"
            ) from e

        self._pin = pin
        self.urdf_path = urdf_path
        self.foot_keyword = foot_keyword.lower()

        mesh_dir = os.path.join(os.path.dirname(urdf_path), "meshes")
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            package_dirs=[mesh_dir, os.path.dirname(urdf_path)],
        )
        self.data = self.model.createData()

        # 找到所有足端 frame 的索引
        self.foot_frame_ids: Dict[str, int] = {
            self.model.frames[i].name: i
            for i in range(len(self.model.frames))
            if self.foot_keyword in self.model.frames[i].name.lower()
            and self.model.frames[i].type == pin.FrameType.BODY
        }
        if not self.foot_frame_ids:
            # 后备：查找 joint frame
            self.foot_frame_ids = {
                self.model.frames[i].name: i
                for i in range(len(self.model.frames))
                if self.foot_keyword in self.model.frames[i].name.lower()
            }

    def neutral_config(self) -> np.ndarray:
        """返回中性（零）关节配置向量 q。"""
        return self._pin.neutral(self.model)

    def compute_foot_jacobians(
        self,
        q: np.ndarray,
        reference_frame: str = "LOCAL_WORLD_ALIGNED",
    ) -> Dict[str, np.ndarray]:
        """
        给定关节角 q，计算所有足端的 6×nv 空间雅可比矩阵。

        Parameters
        ----------
        q                : 关节配置向量，shape (nq,)
        reference_frame  : "LOCAL_WORLD_ALIGNED" | "LOCAL" | "WORLD"

        Returns
        -------
        {frame_name: J (6, nv)}
        """
        pin = self._pin
        rf_map = {
            "LOCAL_WORLD_ALIGNED": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            "LOCAL": pin.ReferenceFrame.LOCAL,
            "WORLD": pin.ReferenceFrame.WORLD,
        }
        rf = rf_map.get(reference_frame.upper(), pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        result: Dict[str, np.ndarray] = {}
        for name, fid in self.foot_frame_ids.items():
            J = pin.getFrameJacobian(self.model, self.data, fid, rf)
            result[name] = J  # (6, nv)
        return result

    def compute_foot_positions(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """给定关节角 q，返回足端在世界坐标系中的位置 {name: (3,)}。"""
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return {
            name: np.array(self.data.oMf[fid].translation)
            for name, fid in self.foot_frame_ids.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. 自适应 Q / R 权重矩阵
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MPCWeights:
    """
    标准 SRB-MPC 的状态与控制权重。

    状态向量约定（13 维）：
      [roll, pitch, yaw,  px, py, pz,  droll, dpitch, dyaw,  vx, vy, vz,  gravity_placeholder]
    控制向量（每腿 3 维地面反力，共 n_legs*3 维）。
    """
    Q: np.ndarray   # (13, 13) 对角权重，或完整矩阵
    R: np.ndarray   # (n_legs*3, n_legs*3) 对角权重


class AdaptiveMPCWeights:
    """
    根据机器人几何/质量特征启发式地生成 Q 和 R 权重矩阵。

    设计原则
    ---------
    - 宽扁机器人（aspect_ratio 大）：侧滚/俯仰误差天然更容易稳定，
      Roll/Pitch 权重适当降低。
    - 细高机器人（aspect_ratio 小）：Roll/Pitch 权重升高，以防翻倒。
    - 质量越大：控制力权重 R 适当增大，避免过激的力指令。
    - 腿数越多：每腿分配的力权重略降（各腿协同承重）。
    """

    # ── 基准权重（标准六足机器人，aspect_ratio ≈ 1.5） ────────────────────
    _Q_BASE = np.array([
        # roll,  pitch, yaw,   x,     y,     z,
        25.0,  25.0,   1.0,   1.0,   1.0,  50.0,
        # droll, dpitch, dyaw,  vx,    vy,    vz,   gravity
        0.01,   0.01,  0.01,  0.1,   0.1,   0.1,   0.0,
    ])  # (13,) 对角元素

    _R_BASE_PER_LEG = np.array([1e-4, 1e-4, 1e-4])  # Fx, Fy, Fz 每腿基准

    # 非对角惯量耦合门槛：|I_ij| / sqrt(I_ii*I_jj) 超过此值才引入交叉惩罚项
    _INERTIA_COUPLING_THRESHOLD = 0.05

    def __init__(
        self,
        physics: RobotPhysics,
        *,
        # 可选覆盖（None 表示使用自适应计算值）
        q_roll_pitch_override: Optional[float] = None,
        q_z_override: Optional[float] = None,
        r_scale_override: Optional[float] = None,
    ):
        self.physics = physics
        self._q_rp_override = q_roll_pitch_override
        self._q_z_override = q_z_override
        self._r_scale_override = r_scale_override

    # ── 内部启发式函数 ────────────────────────────────────────────────────────

    def _roll_pitch_scale(self) -> Tuple[float, float]:
        """
        分别返回 Roll 和 Pitch 的权重缩放因子，基于 2D 足端轮廓包围盒。
        纵向跨度（X）小 → Pitch 方向更不稳 → Pitch 权重升高
        横向跨度（Y）小 → Roll 方向更不稳  → Roll 权重升高
        """
        foot_arr = self.physics.foot_positions_array  # (n, 3)
        if len(foot_arr) < 2:
            return 1.0, 1.0
        x_span = float(foot_arr[:, 0].max() - foot_arr[:, 0].min())
        y_span = float(foot_arr[:, 1].max() - foot_arr[:, 1].min())
        base_span = 0.6  # 基准跨度（m），约标准六足半跨
        x_span = max(x_span, 1e-3)
        y_span = max(y_span, 1e-3)
        pitch_scale = float(np.clip((base_span / x_span) ** 1.2, 0.2, 5.0))
        roll_scale  = float(np.clip((base_span / y_span) ** 1.2, 0.2, 5.0))
        return roll_scale, pitch_scale

    def _height_weight_scale(self) -> float:
        """
        z 方向位置权重缩放：机器人越高（重心高），z 误差越危险，权重升高。
        """
        h = self.physics.body_height
        base_h = 0.35
        scale = np.clip((h / base_h) ** 0.5, 0.5, 3.0)
        return float(scale)

    def _force_weight_scale(self) -> float:
        """
        控制力权重缩放：质量越大 -> R 整体放大（避免过激）；腿多 -> 每腿 R 降低。
        """
        base_mass = 12.0   # kg，基准
        base_legs = 6
        mass_scale = np.clip(self.physics.total_mass / base_mass, 0.5, 4.0)
        leg_scale = np.clip(base_legs / self.physics.num_legs, 0.5, 2.0)
        return float(mass_scale * leg_scale)

    def _inertia_off_diagonal_coupling(self, q_diag: np.ndarray, Q: np.ndarray) -> None:
        """
        将完整 3×3 惯量张量的非对角元素映射为 Q 矩阵中的交叉惩罚项（in-place）。
        轴序 Ixx→roll(0), Iyy→pitch(1), Izz→yaw(2)。
        若相对耦合强度 |I_ij|/sqrt(I_ii*I_jj) > threshold，
        则在 Q[i,j] 注入正比于两轴权重几何均值的交叉项，
        使 MPC 能预见并抑制因质心偏置引起的横摇-俯仰耦合。
        """
        I = self.physics.inertia_tensor  # (3,3)
        thresh = self._INERTIA_COUPLING_THRESHOLD
        for ai in range(3):
            for aj in range(ai + 1, 3):
                I_ii = float(I[ai, ai])
                I_jj = float(I[aj, aj])
                I_ij = float(I[ai, aj])
                denom = math.sqrt(max(I_ii * I_jj, 1e-20))
                coupling_ratio = abs(I_ij) / denom
                if coupling_ratio > thresh:
                    cross = coupling_ratio * math.sqrt(q_diag[ai] * q_diag[aj]) * 0.5
                    Q[ai, aj] = cross
                    Q[aj, ai] = cross

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def compute(self) -> MPCWeights:
        """计算并返回自适应的 Q / R 权重矩阵（含全惯量张量非对角耦合项）。"""
        q_diag = self._Q_BASE.copy()

        if self._q_rp_override is None:
            roll_scale, pitch_scale = self._roll_pitch_scale()
        else:
            roll_scale = pitch_scale = self._q_rp_override / self._Q_BASE[0]
        q_diag[0] *= roll_scale    # roll
        q_diag[1] *= pitch_scale   # pitch

        z_scale = self._height_weight_scale() if self._q_z_override is None else (
            self._q_z_override / self._Q_BASE[5]
        )
        q_diag[5] *= z_scale    # z

        Q = np.diag(q_diag)
        # 注入全惯量张量非对角耦合项
        self._inertia_off_diagonal_coupling(q_diag, Q)

        r_scale = self._force_weight_scale() if self._r_scale_override is None else self._r_scale_override
        r_per_leg = self._R_BASE_PER_LEG * r_scale
        R = np.diag(np.tile(r_per_leg, self.physics.num_legs))

        return MPCWeights(Q=Q, R=R)

    def summary(self) -> str:
        """返回可读性强的权重摘要字符串。"""
        w = self.compute()
        lines = [
            f"  aspect_ratio   = {self.physics.aspect_ratio:.3f}",
            f"  body_height    = {self.physics.body_height:.3f} m",
            f"  total_mass     = {self.physics.total_mass:.3f} kg",
            f"  num_legs       = {self.physics.num_legs}",
            "",
            f"  Roll/Pitch Q   = {w.Q[0,0]:.4f} / {w.Q[1,1]:.4f}",
            f"  Yaw Q          = {w.Q[2,2]:.4f}",
            f"  z-pos Q        = {w.Q[5,5]:.4f}",
            f"  R (per-leg Fz) = {w.R[-1,-1]:.2e}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 便利函数：一次性构建 MPC 所需的全部输入
# ──────────────────────────────────────────────────────────────────────────────

def build_mpc_params(
    json_path: str,
    urdf_path: Optional[str] = None,
) -> Tuple[RobotPhysics, MPCWeights]:
    """
    给定 robot_description.json（以及可选的 URDF），
    返回 (RobotPhysics, MPCWeights)，可直接传入 MPC 求解器。

    Parameters
    ----------
    json_path : robot_description.json 的路径
    urdf_path : （可选）URDF 路径；若提供则同时验证惯性参数一致性

    Returns
    -------
    physics : RobotPhysics  —— 物理参数
    weights : MPCWeights    —— 自适应 Q/R 矩阵
    """
    physics = RobotPhysicsParser.from_json(json_path)
    weights = AdaptiveMPCWeights(physics).compute()
    return physics, weights


# ──────────────────────────────────────────────────────────────────────────────
# 使用示例
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    json_path = sys.argv[1] if len(sys.argv) > 1 else "robot_assets/robot_description.json"

    print("=" * 60)
    print("【1】解析物理参数")
    print("=" * 60)
    physics = RobotPhysicsParser.from_json(json_path)
    print(f"  总质量         : {physics.total_mass:.4f} kg")
    print(f"  质心位置 (body): {physics.com_position}")
    print(f"  惯性张量:\n{physics.inertia_tensor}")
    print(f"  腿数           : {physics.num_legs}")
    print("  各腿足端初始坐标 (世界系):")
    for leg_id, pos in sorted(physics.foot_positions.items()):
        print(f"    leg_{leg_id}: {pos}")
    print(f"  aspect_ratio   : {physics.aspect_ratio:.3f}")
    print(f"  body_height    : {physics.body_height:.3f} m")

    print()
    print("=" * 60)
    print("【2】自适应 Q / R 权重")
    print("=" * 60)
    adapter = AdaptiveMPCWeights(physics)
    print(adapter.summary())
    weights = adapter.compute()
    print(f"\n  Q 矩阵对角 (前6维): {np.diag(weights.Q)[:6]}")
    print(f"  R 矩阵尺寸        : {weights.R.shape}")

    print()
    print("=" * 60)
    print("【3】Pinocchio 足端雅可比示例（需要 pinocchio + URDF）")
    print("=" * 60)
    urdf_candidates = [
        "robot_assets/standard_hexapod/generated_robot.urdf",
        "robot_assets/variants/variant_00_seed7/robot.urdf",
    ]
    urdf_path = next((p for p in urdf_candidates if os.path.exists(p)), None)
    if urdf_path is None:
        print("  未找到可用的 URDF，跳过雅可比演示。")
    else:
        try:
            calc = FootJacobianCalculator(urdf_path)
            q0 = calc.neutral_config()
            jacobians = calc.compute_foot_jacobians(q0)
            positions = calc.compute_foot_positions(q0)
            print(f"  已加载 URDF: {urdf_path}")
            print(f"  找到足端 frame: {list(jacobians.keys())}")
            for name, J in jacobians.items():
                print(f"    {name}: J.shape={J.shape}  ||J||={np.linalg.norm(J):.4f}")
            print("  足端位置 (世界系):")
            for name, pos in positions.items():
                print(f"    {name}: {pos}")
        except ImportError as e:
            print(f"  [跳过] {e}")
        except Exception as e:
            print(f"  [错误] {e}")

    print()
    print("=" * 60)
    print("【4】build_mpc_params 便利接口")
    print("=" * 60)
    phys2, w2 = build_mpc_params(json_path)
    print(f"  总质量={phys2.total_mass:.3f} kg  Q[0,0]={w2.Q[0,0]:.4f}  R[0,0]={w2.R[0,0]:.2e}")
    print("  ✓ 所有参数已就绪，可注入 MPC 求解器。")
