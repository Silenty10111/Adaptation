#!/usr/bin/env python3
"""共享工具函数 — 被多个模块复用的数学、运动学与环境切换工具。

提供:
  - 数学工具: ratio_to_joint, smoothstep, quat_to_euler
  - 运动学工具: foot_xy_map
  - 环境工具: maybe_reexec (自动重启到指定 Python 环境)
  - 指标工具: compute_metrics (从仿真轨迹计算 yaw/lat/fwd)
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 数学工具
# ═══════════════════════════════════════════════════════════════════════════════

def ratio_to_joint(lower: float, upper: float, ratio: float) -> float:
    """线性插值: ratio ∈ [0, 1] → joint ∈ [lower, upper]."""
    bounded = max(0.0, min(1.0, float(ratio)))
    return float(lower + bounded * (upper - lower))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Smooth Hermite interpolation between 0 and 1, zero-derivative at edges."""
    t = max(0.0, min(1.0, (x - edge0) / max(edge1 - edge0, 1e-9)))
    return t * t * (3.0 - 2.0 * t)


def quat_to_euler(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """四元数 (w, x, y, z) → roll, pitch, yaw (弧度)."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


# ═══════════════════════════════════════════════════════════════════════════════
# 运动学工具
# ═══════════════════════════════════════════════════════════════════════════════

def foot_xy_map(description: dict) -> Dict[int, np.ndarray]:
    """从 robot_description 中提取每条腿的足端 XY 坐标。

    Returns:
        {leg_id: np.ndarray([x, y])} 映射。
    """
    mapping: Dict[int, np.ndarray] = {}
    for link in description.get("links", []):
        if link.get("role") != "foot" or link.get("leg_id") is None:
            continue
        origin = np.asarray(link.get("default_world_origin", [0.0, 0.0, 0.0]), dtype=float)
        mapping[int(link["leg_id"])] = origin[:2]
    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# 环境切换
# ═══════════════════════════════════════════════════════════════════════════════

_REEXEC_ENV_VAR = "BATCH_REEXEC"


def maybe_reexec(target_python: str, target_ld_path: str) -> None:
    """若当前 Python 不是 target_python，则自动 execve 切换到目标环境。

    仅在未设置 BATCH_REEXEC 环境变量且目标解释器存在时执行。
    调用后当前进程被替换；若未切换则无副作用返回。

    Args:
        target_python: 目标 Python 解释器的绝对路径。
        target_ld_path: 目标环境的 LD_LIBRARY_PATH 目录。
    """
    if os.environ.get(_REEXEC_ENV_VAR) == "1":
        return
    if sys.executable == target_python:
        return
    if not Path(target_python).exists():
        print(f"[utils] WARNING: {target_python} not found, continuing in current env")
        return

    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{target_ld_path}:{ld}" if ld else target_ld_path
    env[_REEXEC_ENV_VAR] = "1"
    print(f"[utils] Re-exec into {target_python}")
    os.execve(target_python, [target_python] + sys.argv, env)


# ═══════════════════════════════════════════════════════════════════════════════
# 指标计算
# ═══════════════════════════════════════════════════════════════════════════════

_DT = 1.0 / 60.0


def compute_metrics(
    com_trail: List[List[float]],
    forward_axis: List[float],
    yaw_stats: Optional[Dict],
    n_steps: int,
) -> Dict[str, float]:
    """从仿真轨迹计算偏航、横向漂移、前进速度三个指标。

    Args:
        com_trail:  质心轨迹，形如 [[x0, y0], [x1, y1], ...]。
        forward_axis: 前进方向单位向量 [fx, fy]。
        yaw_stats:   含 "yaw_rate_mean" 的字典或 None。
        n_steps:     仿真步数。

    Returns:
        {"yaw_abs": float, "lat_drift": float, "fwd_vel": float}
    """
    fwd = np.asarray(forward_axis, dtype=float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    lat = np.array([-fwd[1], fwd[0]], dtype=float)

    if len(com_trail) < 2:
        return {"yaw_abs": float("nan"), "lat_drift": float("nan"), "fwd_vel": 0.0}

    arr = np.asarray(com_trail, dtype=float)
    disp = arr[-1] - arr[0]
    elapsed = max(n_steps * _DT, _DT)

    return {
        "yaw_abs": abs(yaw_stats["yaw_rate_mean"]) if yaw_stats else float("nan"),
        "lat_drift": float(np.dot(disp, lat)),
        "fwd_vel": float(np.dot(disp, fwd)) / elapsed,
    }
