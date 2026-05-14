#!/usr/bin/env python3
"""低重心匍匐仿蜘蛛六足机器人 — 形态与步态参数自动寻优脚本。

【工作流程】
  1. 定义形态参数 + 步态控制参数的搜索空间（见 SEARCH_SPACE）。
  2. 使用随机搜索（Random Search）在空间内采样。
  3. 对每组参数：
       a. 调用 generate_standardurdf.py 生成对应几何体 + URDF；
       b. 检查静态约束（X/Y 比值、I_roll）；
       c. 调用 test_standard_gait.py --headless 运行仿真，输出 metrics JSON；
       d. 从 JSON 读取动态指标（roll/pitch RMSE、CoM Z 方差）。
  4. 计算综合适应度（Fitness）并排序。
  5. 将最佳参数写到 search_results/best_params.json，全量结果写到
     search_results/all_results.json，并打印排行榜。

【适应度函数设计】
  F = w_ratio   * clip(xy_ratio,  0, 2.0)          # X/Y 足端跨度比（越接近 1.0 → 越均衡）
    - w_roll    * roll_rmse_deg                     # 翻滚 RMSE（越小越好）
    - w_pitch   * pitch_rmse_deg                    # 俯仰 RMSE（越小越好）
    - w_z_var   * com_z_var * 1000                  # CoM Z 方差（放大以与角度量纲对齐）
    + w_fwd     * fwd_distance                      # 前进距离（越大越好，验证能走动）
    - w_lat     * abs(lat_drift)                    # 侧向漂移惩罚
    - w_heading * abs(heading_error_deg)            # 方向偏差惩罚

  其中 xy_ratio 当 >= 1.0 时不再额外奖励，避免"拼命拉长X"而忽视平稳性：
  使用 ratio_reward = -abs(xy_ratio - 1.0)（离 1.0 越远扣分越多，上下均罚）。

用法示例
--------
  # 标准随机搜索（无界面），每组跑 600 步（约 10 秒仿真时间）
  python search_morphology.py --n-trials 30 --sim-steps 600

  # 快速验证（仅 5 组，200 步）
  python search_morphology.py --n-trials 5 --sim-steps 200 --seed 42

  # 指定 GPU / CPU
  python search_morphology.py --n-trials 20 --cpu-sim --sim-steps 500

注意
----
 - 本脚本会覆盖 robot_assets/standard_hexapod/ 下的文件（每次试验重新生成）。
 - 每个试验需要启动完整的 Isaac Gym 进程（子进程），因此速度受 GPU 驱动初始化影响。
 - 建议先用 --n-trials 5 确认流程可跑通，再执行完整搜索。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
REPO_ROOT      = Path(__file__).resolve().parent
GEN_SCRIPT     = REPO_ROOT / "generate_standardurdf.py"
SIM_SCRIPT     = REPO_ROOT / "test_standard_gait.py"
RESULT_DIR     = REPO_ROOT / "search_results"
DESC_PATH      = REPO_ROOT / "robot_assets" / "standard_hexapod" / "robot_description.json"
URDF_PATH      = REPO_ROOT / "robot_assets" / "standard_hexapod" / "generated_robot.urdf"
METRICS_PATH   = REPO_ROOT / "search_results" / "trial_metrics.json"

# 使用与项目一致的 Python 解释器
GEN_PYTHON     = str(Path(sys.executable))
SIM_PYTHON     = "/data/conda/envs/unitree-rl/bin/python"  # Isaac Gym 环境
SIM_LD_PATH    = "/data/conda/envs/unitree-rl/lib"


# ---------------------------------------------------------------------------
# 搜索空间定义
# ---------------------------------------------------------------------------
# 每个参数三元组：(min, max, type)
# type ∈ {"float", "int"}  — float 会均匀采样，int 会从整数范围均匀采样
SEARCH_SPACE: Dict[str, Tuple[float, float, str]] = {
    # ---- 形态参数 (geometry) ------------------------------------------------
    "upper_x_bias":   (0.30, 0.80, "float"),  # 大腿前后偏置
    "upper_y_weight": (0.45, 0.80, "float"),  # 大腿侧向展开
    "upper_z_weight": (0.00, 0.25, "float"),  # 大腿向下倾角
    "lower_x_bias":   (0.10, 0.55, "float"),  # 小腿前后偏置
    "lower_y_weight": (0.00, 0.22, "float"),  # 小腿侧向
    "upper_length":   (0.32, 0.44, "float"),  # 大腿长度 (m)
    "lower_length":   (0.32, 0.44, "float"),  # 小腿长度 (m)
    "body_length":    (0.45, 0.65, "float"),  # 躯干长度 (m)
    "body_width":     (0.28, 0.46, "float"),  # 躯干宽度 (m)
    "density":        (700,  1200, "float"),  # 密度 (kg/m³)
    # ---- 步态控制参数 (gait control) ----------------------------------------
    "stance_drop_ratio": (0.75, 0.95, "float"),  # 支撑相膝关节下压比例（越大越趴伏）
    "swing_lift_ratio":  (0.55, 0.85, "float"),  # 摆动相抬腿比例
    "gait_frequency":    (0.60, 1.10, "float"),  # 步态频率 (Hz)
}

# 固定不参与搜索的参数（节省搜索时间）
FIXED_PARAMS: Dict[str, Any] = {
    "body_height":  0.08,
    "hip_depth":    0.04,
    "stance_lift_ratio": 0.46,   # 对应 lift≈0 rad（扁平几何下正确支撑角）
    "swing_drop_ratio":  0.38,
}


# ---------------------------------------------------------------------------
# 适应度权重（可在 CLI 调整）
# ---------------------------------------------------------------------------
FITNESS_WEIGHTS = {
    "w_ratio":   2.50,  # X/Y 比值接近 1.0 的奖励
    "w_roll":    0.40,  # roll RMSE（°）的惩罚系数
    "w_pitch":   0.30,  # pitch RMSE（°）的惩罚系数
    "w_z_var":   0.60,  # CoM Z 方差（×1000 缩放后）的惩罚系数
    "w_fwd":     1.00,  # 前进距离（m）的奖励系数
    "w_lat":     0.50,  # 侧向漂移（m）的惩罚系数
    "w_heading": 0.05,  # 方向偏差（°）的惩罚系数
}


# ---------------------------------------------------------------------------
# 静态约束（几何生成后即可判断，无需仿真）
# ---------------------------------------------------------------------------
STATIC_CONSTRAINTS = {
    "xy_ratio_min":  0.80,    # 足端 X/Y 跨度比下限（避免过分细长）
    "i_roll_min":    0.12,    # 翻转惯量下限 (kg·m²)
    "total_mass_min": 8.0,    # 总质量下限 (kg)
    "total_mass_max": 45.0,   # 总质量上限 (kg)
}


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def sample_params(space: Dict[str, Tuple], rng: random.Random) -> Dict[str, Any]:
    """从搜索空间中均匀随机采样一组参数。"""
    params: Dict[str, Any] = {}
    for key, (lo, hi, typ) in space.items():
        if typ == "int":
            params[key] = int(rng.randint(int(lo), int(hi)))
        else:
            params[key] = float(rng.uniform(lo, hi))
    params.update(FIXED_PARAMS)
    return params


# ---------------------------------------------------------------------------
# 几何生成
# ---------------------------------------------------------------------------

def run_generate(params: Dict[str, Any]) -> Optional[Dict]:
    """
    调用 generate_standardurdf.py 生成 URDF，返回几何指标 dict，失败返回 None。

    几何指标包含：x_span, y_span, xy_ratio, i_roll, total_mass。
    """
    cmd = [GEN_PYTHON, str(GEN_SCRIPT)]
    # 将参数字典转为 CLI 参数（仅传几何相关参数）
    geo_keys = {
        "body_length", "body_width", "body_height", "hip_depth",
        "upper_length", "lower_length", "density",
        "upper_x_bias", "upper_y_weight", "upper_z_weight",
        "lower_x_bias", "lower_y_weight",
    }
    for key, val in params.items():
        if key in geo_keys:
            cmd.extend([f"--{key.replace('_', '-')}", str(val)])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  [GEN ERROR] return={result.returncode}")
            print(result.stderr[-400:] if result.stderr else "")
            return None
        # 从标准输出解析几何指标
        return parse_geo_output(result.stdout)
    except subprocess.TimeoutExpired:
        print("  [GEN TIMEOUT]")
        return None
    except Exception as e:
        print(f"  [GEN EXCEPTION] {e}")
        return None


def parse_geo_output(stdout: str) -> Dict:
    """从 generate_standardurdf.py 的标准输出解析几何指标。"""
    geo: Dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "足端X跨度" in line:
            try:
                geo["x_span"] = float(line.split(":")[1].split("m")[0].strip())
            except Exception:
                pass
        elif "足端Y跨度" in line:
            try:
                geo["y_span"] = float(line.split(":")[1].split("m")[0].strip())
            except Exception:
                pass
        elif "X/Y比值" in line or "X/Y\u6bd4\u503c" in line:
            try:
                val_str = line.split(":")[1].strip().split()[0]
                geo["xy_ratio"] = float(val_str)
            except Exception:
                pass
        elif "I_roll" in line:
            try:
                val_str = line.split(":")[1].strip().split()[0]
                geo["i_roll"] = float(val_str)
            except Exception:
                pass
        elif "总质量" in line:
            try:
                geo["total_mass"] = float(line.split(":")[1].split("kg")[0].strip())
            except Exception:
                pass
    # 若解析失败则直接读取 JSON 描述文件（更可靠）
    if "xy_ratio" not in geo and DESC_PATH.exists():
        try:
            desc = json.loads(DESC_PATH.read_text(encoding="utf-8"))
            feet = [lk for lk in desc["links"] if lk["role"] == "foot"]
            if feet:
                fxs = [lk["default_world_origin"][0] for lk in feet]
                fys = [lk["default_world_origin"][1] for lk in feet]
                xs = max(fxs) - min(fxs)
                ys = max(fys) - min(fys)
                geo["x_span"] = xs
                geo["y_span"] = ys
                geo["xy_ratio"] = xs / max(ys, 1e-9)
            trunk = next(lk for lk in desc["links"] if lk["role"] == "trunk")
            geo["i_roll"] = trunk["mass_properties"]["inertia"][0][0]
            geo["total_mass"] = sum(lk["mass_properties"]["mass"] for lk in desc["links"])
        except Exception:
            pass
    return geo


# ---------------------------------------------------------------------------
# 静态约束检查
# ---------------------------------------------------------------------------

def check_static_constraints(geo: Dict) -> Tuple[bool, str]:
    """返回 (passed, reason)。"""
    xy = geo.get("xy_ratio", 0.0)
    if xy < STATIC_CONSTRAINTS["xy_ratio_min"]:
        return False, f"xy_ratio={xy:.3f} < {STATIC_CONSTRAINTS['xy_ratio_min']}"
    i_roll = geo.get("i_roll", 0.0)
    if i_roll < STATIC_CONSTRAINTS["i_roll_min"]:
        return False, f"i_roll={i_roll:.4f} < {STATIC_CONSTRAINTS['i_roll_min']}"
    mass = geo.get("total_mass", 0.0)
    if mass < STATIC_CONSTRAINTS["total_mass_min"]:
        return False, f"total_mass={mass:.2f} < {STATIC_CONSTRAINTS['total_mass_min']}"
    if mass > STATIC_CONSTRAINTS["total_mass_max"]:
        return False, f"total_mass={mass:.2f} > {STATIC_CONSTRAINTS['total_mass_max']}"
    return True, "ok"


# ---------------------------------------------------------------------------
# 仿真运行
# ---------------------------------------------------------------------------

def run_simulation(params: Dict[str, Any], sim_steps: int, trial_idx: int,
                   cpu_sim: bool) -> Optional[Dict]:
    """
    调用 test_standard_gait.py --headless，返回 metrics dict，失败返回 None。
    """
    metrics_path = RESULT_DIR / f"trial_{trial_idx:04d}_metrics.json"
    cmd = [SIM_PYTHON, str(SIM_SCRIPT),
           "--headless",
           "--steps", str(sim_steps),
           "--metrics-out", str(metrics_path),
           "--description", str(DESC_PATH),
           "--urdf", str(URDF_PATH),
    ]
    # 步态参数
    gait_keys = {
        "gait_frequency": "--gait-frequency",
        "stance_drop_ratio": "--stance-drop-ratio",
        "swing_lift_ratio":  "--swing-lift-ratio",
        "swing_drop_ratio":  "--swing-drop-ratio",
        "stance_lift_ratio": "--stance-lift-ratio",
    }
    for pname, flag in gait_keys.items():
        if pname in params:
            cmd.extend([flag, str(params[pname])])
    if cpu_sim:
        cmd.append("--cpu-sim")

    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{SIM_LD_PATH}:{ld_path}" if ld_path else SIM_LD_PATH
    env["TEST_GAIT_REEXEC"] = "1"  # 告知脚本不再重新 exec

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            print(f"  [SIM ERROR] return={result.returncode}")
            # 打印最后几行错误，便于调试
            lines = (result.stderr or result.stdout or "").splitlines()
            for ln in lines[-6:]:
                print(f"    {ln}")
            return None
        if metrics_path.exists():
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        # 兜底：从 stdout 解析
        return parse_sim_output(result.stdout)
    except subprocess.TimeoutExpired:
        print("  [SIM TIMEOUT]")
        return None
    except Exception as e:
        print(f"  [SIM EXCEPTION] {e}")
        return None


def parse_sim_output(stdout: str) -> Dict:
    """从 test_standard_gait.py 标准输出解析指标（兜底方法）。"""
    metrics: Dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "roll_rmse" in line:
            try:
                metrics["roll_rmse_rad"] = math.radians(float(line.split("=")[1].replace("°", "").strip()))
            except Exception:
                pass
        elif "pitch_rmse" in line:
            try:
                metrics["pitch_rmse_rad"] = math.radians(float(line.split("=")[1].replace("°", "").strip()))
            except Exception:
                pass
        elif "com_z_var" in line:
            try:
                metrics["com_z_var"] = float(line.split("=")[1].split()[0])
            except Exception:
                pass
        elif "前进距离" in line:
            try:
                metrics["fwd_distance"] = float(line.split(":")[1].split("m")[0].strip())
            except Exception:
                pass
        elif "侧向漂移" in line:
            try:
                metrics["lat_drift"] = float(line.split(":")[1].split("m")[0].strip())
            except Exception:
                pass
        elif "方向偏差" in line and "°" in line:
            try:
                metrics["heading_error_deg"] = float(line.split(":")[1].replace("°", "").strip().split()[0])
            except Exception:
                pass
    return metrics


# ---------------------------------------------------------------------------
# 适应度计算
# ---------------------------------------------------------------------------

def compute_fitness(geo: Dict, dyn: Dict, weights: Dict) -> float:
    """
    计算综合适应度分数（越高越好）。

    设计逻辑：
    1. X/Y 比值使用"距 1.0 的绝对偏差"作为惩罚（上下均等惩罚）。
       xy_ratio = 1.5 时，|1.5-1.0| = 0.5；xy_ratio = 0.8 时，|0.8-1.0| = 0.2。
       这样鼓励接近正方形足端布局，而不是无限拉长 X。
    2. 角度指标转换成度数，与直觉一致。
    3. CoM Z 方差乘以 1000，使其量纲与度数大致对齐（约 1°² ≈ 0.0003 m²）。
    4. 前进距离奖励 — 验证机器人确实能走起来（而不是原地抖动）。
    """
    xy_ratio = geo.get("xy_ratio", 0.5)
    xy_penalty = abs(xy_ratio - 1.0)   # 理想值 0.0（1:1 布局）

    roll_rmse_deg  = math.degrees(dyn.get("roll_rmse_rad", 10.0))
    pitch_rmse_deg = math.degrees(dyn.get("pitch_rmse_rad", 10.0))
    com_z_var_scaled = dyn.get("com_z_var", 0.01) * 1000.0
    fwd_dist = max(0.0, dyn.get("fwd_distance", 0.0))  # 不奖励倒退
    lat_drift = abs(dyn.get("lat_drift", 0.0))
    heading_err = abs(dyn.get("heading_error_deg", 90.0))

    F = (- weights["w_ratio"]   * xy_penalty
         - weights["w_roll"]    * roll_rmse_deg
         - weights["w_pitch"]   * pitch_rmse_deg
         - weights["w_z_var"]   * com_z_var_scaled
         + weights["w_fwd"]     * fwd_dist
         - weights["w_lat"]     * lat_drift
         - weights["w_heading"] * heading_err)
    return float(F)


# ---------------------------------------------------------------------------
# 主搜索循环
# ---------------------------------------------------------------------------

def run_search(args: argparse.Namespace) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    results: List[Dict] = []
    trial_count = 0
    skip_count  = 0
    sim_fail    = 0

    print("=" * 70)
    print(f"[搜索] 开始随机搜索 — 计划试验数: {args.n_trials}")
    print(f"       仿真步数: {args.sim_steps}  采样种子: {args.seed}")
    print(f"       静态约束: {STATIC_CONSTRAINTS}")
    print("=" * 70)
    t0 = time.time()

    for trial_idx in range(args.n_trials):
        print(f"\n{'─'*60}")
        print(f"[试验 {trial_idx+1}/{args.n_trials}]  耗时: {time.time()-t0:.0f}s")

        # ---- 采样 -----------------------------------------------------------
        params = sample_params(SEARCH_SPACE, rng)
        print(f"  形态: UL={params['upper_length']:.3f}  LL={params['lower_length']:.3f}"
              f"  BL={params['body_length']:.3f}  BW={params['body_width']:.3f}"
              f"  ρ={params['density']:.0f}")
        print(f"  腿向量: uxb={params['upper_x_bias']:.3f}  uyw={params['upper_y_weight']:.3f}"
              f"  uzw={params['upper_z_weight']:.3f}  lxb={params['lower_x_bias']:.3f}"
              f"  lyw={params['lower_y_weight']:.3f}")
        print(f"  步态: freq={params['gait_frequency']:.2f}  "
              f"stance_drop={params['stance_drop_ratio']:.3f}  "
              f"swing_lift={params['swing_lift_ratio']:.3f}")

        # ---- 几何生成 -------------------------------------------------------
        geo = run_generate(params)
        if geo is None:
            print("  [SKIP] 几何生成失败")
            skip_count += 1
            continue

        # ---- 静态约束检查 ---------------------------------------------------
        passed, reason = check_static_constraints(geo)
        print(f"  静态检查: xy_ratio={geo.get('xy_ratio', 0):.3f}  "
              f"i_roll={geo.get('i_roll', 0):.4f}  mass={geo.get('total_mass', 0):.1f}kg")
        if not passed:
            print(f"  [SKIP] 静态约束不满足: {reason}")
            skip_count += 1
            continue

        # ---- 仿真运行 -------------------------------------------------------
        print(f"  ▶ 运行仿真 ({args.sim_steps} steps)...")
        dyn = run_simulation(params, args.sim_steps, trial_idx, args.cpu_sim)
        if dyn is None:
            print("  [SKIP] 仿真失败或超时")
            sim_fail += 1
            continue

        # ---- 适应度计算 -----------------------------------------------------
        fitness = compute_fitness(geo, dyn, FITNESS_WEIGHTS)
        trial_count += 1

        roll_d  = math.degrees(dyn.get("roll_rmse_rad", 0))
        pitch_d = math.degrees(dyn.get("pitch_rmse_rad", 0))
        print(f"  结果: fitness={fitness:+.4f}  "
              f"fwd={dyn.get('fwd_distance', 0):+.4f}m  "
              f"roll={roll_d:.2f}°  pitch={pitch_d:.2f}°  "
              f"z_var={dyn.get('com_z_var', 0)*1e4:.2f}×10⁻⁴")

        record = {
            "trial_idx": trial_idx,
            "fitness":   fitness,
            "params":    params,
            "geo":       {k: v for k, v in geo.items() if k != "metadata"},
            "dyn":       dyn,
        }
        results.append(record)

    # ---- 排序 ---------------------------------------------------------------
    results.sort(key=lambda r: r["fitness"], reverse=True)

    print(f"\n{'='*70}")
    print(f"[搜索完成] 有效试验: {trial_count}  跳过: {skip_count}  仿真失败: {sim_fail}")
    print(f"  总耗时: {time.time()-t0:.1f}s")

    if not results:
        print("[警告] 无有效试验结果，请检查参数范围或仿真环境。")
        return

    # ---- 打印排行榜 ---------------------------------------------------------
    print("\n[排行榜] Top-10 结果")
    print(f"  {'排名':>4}  {'fitness':>9}  {'xy_ratio':>8}  {'roll°':>6}  "
          f"{'pitch°':>7}  {'fwd(m)':>7}  {'z_var×10⁻⁴':>10}")
    print("  " + "─" * 64)
    for rank, r in enumerate(results[:10], 1):
        g, d = r["geo"], r["dyn"]
        print(f"  {rank:>4}  {r['fitness']:>+9.4f}  {g.get('xy_ratio', 0):>8.3f}  "
              f"{math.degrees(d.get('roll_rmse_rad', 0)):>6.2f}  "
              f"{math.degrees(d.get('pitch_rmse_rad', 0)):>7.2f}  "
              f"{d.get('fwd_distance', 0):>7.4f}  "
              f"{d.get('com_z_var', 0)*1e4:>10.2f}")

    # ---- 输出最优参数 -------------------------------------------------------
    best = results[0]
    best_params_path = RESULT_DIR / "best_params.json"
    best_params_path.write_text(
        json.dumps({"fitness": best["fitness"],
                    "params":  best["params"],
                    "geo":     best["geo"],
                    "dyn":     best["dyn"]}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[最优参数] 已保存到: {best_params_path}")
    print(f"  fitness = {best['fitness']:+.4f}")
    for k, v in best["params"].items():
        print(f"    {k:25s} = {v}")

    # ---- 全量结果 -----------------------------------------------------------
    all_path = RESULT_DIR / "all_results.json"
    all_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\n[全量结果] 已保存到: {all_path}")

    # ---- 用最优参数重新生成并提示用户 ----------------------------------------
    print("\n[提示] 要应用最优参数，运行以下命令：")
    best_p = best["params"]
    cmd_parts = [f"python generate_standardurdf.py"]
    geo_keys = {"body_length", "body_width", "body_height", "hip_depth",
                "upper_length", "lower_length", "density",
                "upper_x_bias", "upper_y_weight", "upper_z_weight",
                "lower_x_bias", "lower_y_weight"}
    for k, v in best_p.items():
        if k in geo_keys:
            cmd_parts.append(f"  --{k.replace('_', '-')} {v:.4f}")
    print(" \\\n".join(cmd_parts))
    print()
    gait_keys = {"gait_frequency", "stance_drop_ratio", "swing_lift_ratio"}
    sim_cmd = [f"python test_standard_gait.py"]
    for k, v in best_p.items():
        if k in gait_keys:
            sim_cmd.append(f"  --{k.replace('_', '-')} {v:.4f}")
    print(" \\\n".join(sim_cmd))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-trials",  type=int,   default=20,
                   help="随机搜索试验总数。")
    p.add_argument("--sim-steps", type=int,   default=600,
                   help="每组参数的仿真步数（dt=1/60s，600步≈10秒）。")
    p.add_argument("--seed",      type=int,   default=42,
                   help="随机种子，保证可复现。")
    p.add_argument("--cpu-sim",   action="store_true",
                   help="使用 CPU 物理引擎（速度慢但无 GPU 要求）。")
    p.add_argument("--w-ratio",   type=float, default=FITNESS_WEIGHTS["w_ratio"])
    p.add_argument("--w-roll",    type=float, default=FITNESS_WEIGHTS["w_roll"])
    p.add_argument("--w-pitch",   type=float, default=FITNESS_WEIGHTS["w_pitch"])
    p.add_argument("--w-z-var",   type=float, default=FITNESS_WEIGHTS["w_z_var"])
    p.add_argument("--w-fwd",     type=float, default=FITNESS_WEIGHTS["w_fwd"])
    p.add_argument("--w-lat",     type=float, default=FITNESS_WEIGHTS["w_lat"])
    p.add_argument("--w-heading", type=float, default=FITNESS_WEIGHTS["w_heading"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 将 CLI 权重注入全局 FITNESS_WEIGHTS
    FITNESS_WEIGHTS["w_ratio"]   = args.w_ratio
    FITNESS_WEIGHTS["w_roll"]    = args.w_roll
    FITNESS_WEIGHTS["w_pitch"]   = args.w_pitch
    FITNESS_WEIGHTS["w_z_var"]   = args.w_z_var
    FITNESS_WEIGHTS["w_fwd"]     = args.w_fwd
    FITNESS_WEIGHTS["w_lat"]     = args.w_lat
    FITNESS_WEIGHTS["w_heading"] = args.w_heading
    run_search(args)


if __name__ == "__main__":
    main()
