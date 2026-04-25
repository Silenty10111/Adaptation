#!/usr/bin/env python3
"""
Tune geometry parameters for generate_geometry.py.
新增功能：
  --use-default : 只尝试默认几何（不传任何几何参数），观察成功率
  --ref-json     : 从 robot_description.json 中读取基准几何值，在其附近随机抖动
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------- configuration ----------
GEN_PYTHON = "/data/conda/envs/Adaptation/bin/python"
REPO_ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = REPO_ROOT / "generate_geometry.py"
# -----------------------------------

PARAM_ARGS = [
    "--body-length",
    "--body-width",
    "--body-height",
    "--upper-length",
    "--lower-length",
]

# 映射 JSON 字段到参数名
JSON_KEY_MAP = {
    "body_length": "--body-length",
    "body_width": "--body-width",
    "body_height": "--body-height",
    "upper_length": "--upper-length",
    "lower_length": "--lower-length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5, help="Number of trials.")
    parser.add_argument(
        "--use-default", action="store_true",
        help="Do not pass any geometry arguments; use generate_geometry.py defaults."
    )
    parser.add_argument(
        "--ref-json", type=Path, default=None,
        help="Path to robot_description.json for reference geometry values."
    )
    parser.add_argument(
        "--jitter", type=float, default=0.05,
        help="Max relative jitter around reference values (e.g. 0.05 = ±5%)."
    )
    for p in PARAM_ARGS:
        name = p.lstrip("-").replace("-", "_")
        parser.add_argument(
            p, nargs=2, type=float, default=None,
            help=f"Min and max for {p} (default: 0.10 0.50)."
        )
    parser.add_argument("--seed-base", type=int, default=7)
    parser.add_argument("--seed-step", type=int, default=17)
    parser.add_argument("--leg-placement", default="random")
    parser.add_argument("--robot-name-prefix", default="tune")
    return parser.parse_args()


def load_reference_values(ref_json: Path) -> Dict[str, float]:
    """从 robot_description.json 提取几何基准值。"""
    with open(ref_json, "r") as f:
        data = json.load(f)
    ref = {}
    for json_key, arg_name in JSON_KEY_MAP.items():
        if json_key in data:
            ref[arg_name] = float(data[json_key])
    if len(ref) < len(JSON_KEY_MAP):
        missing = [k for k in JSON_KEY_MAP if JSON_KEY_MAP[k] not in ref]
        print(f"[WARN] Reference JSON missing keys: {missing}")
    return ref


def build_param_grid(args: argparse.Namespace) -> List[Optional[List[float]]]:
    """
    返回参数向量列表，每个元素要么是一个 List[float]，要么是 None（表示使用默认值）。
    """
    if args.use_default:
        return [None] * args.trials

    if args.ref_json:
        if not args.ref_json.exists():
            print(f"[ERROR] Reference JSON not found: {args.ref_json}")
            sys.exit(1)
        ref_vals = load_reference_values(args.ref_json)
        if not ref_vals:
            print("[ERROR] No reference values extracted.")
            sys.exit(1)
        # 在基准值附近做随机抖动
        trials = []
        for _ in range(args.trials):
            vec = []
            for p in PARAM_ARGS:
                base = ref_vals.get(p)
                if base is None:
                    # fallback to default if missing
                    base = 0.2
                jitter = base * args.jitter
                val = np.random.uniform(base - jitter, base + jitter)
                vec.append(round(max(0.01, val), 4))
            trials.append(vec)
        return trials

    # 否则用用户指定或默认范围随机
    ranges = {}
    for p in PARAM_ARGS:
        name = p.lstrip("-").replace("-", "_")
        min_max = getattr(args, name, None)
        if min_max is None:
            ranges[p] = (0.10, 0.50)
        else:
            ranges[p] = (min(min_max), max(min_max))

    trials = []
    for _ in range(args.trials):
        vec = []
        for p in PARAM_ARGS:
            lo, hi = ranges[p]
            val = np.random.uniform(lo, hi) if lo != hi else lo
            vec.append(round(val, 4))
        trials.append(vec)
    return trials


def try_variant(seed: int, name: str, param_vector: Optional[List[float]], leg_placement: str) -> bool:
    """如果 param_vector 为 None，则不传任何几何参数。"""
    cmd = [
        GEN_PYTHON,
        str(GEN_SCRIPT),
        "--robot-name", name,
        "--seed", str(seed),
        "--leg-placement", leg_placement,
    ]
    if param_vector is not None:
        for p, val in zip(PARAM_ARGS, param_vector):
            cmd.extend([p, str(val)])

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 打印最后几行 stderr 以便诊断
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-3:]:
            print(f"  STDERR: {line}")
        return False
    else:
        # 可选：打印 SSM 信息（如果脚本输出到 stdout）
        stdout = result.stdout.strip()
        if stdout:
            for line in stdout.splitlines()[-3:]:
                print(f"  STDOUT: {line}")
        return True


def main() -> int:
    args = parse_args()

    if not GEN_SCRIPT.exists():
        print(f"[ERROR] {GEN_SCRIPT} not found.")
        return 1

    trials = build_param_grid(args)
    print(f"Running {len(trials)} trials...")
    if args.use_default:
        print("Mode: using default geometry (no custom parameters)")
    elif args.ref_json:
        print(f"Mode: jitter around reference from {args.ref_json}")
    else:
        print("Mode: random in given ranges")

    successes = []
    for i, vec in enumerate(trials):
        seed = args.seed_base + i * args.seed_step
        name = f"{args.robot_name_prefix}_{i:03d}_s{seed}"
        if vec is None:
            param_str = "default"
        else:
            param_str = ", ".join(f"{p} {v}" for p, v in zip(PARAM_ARGS, vec))
        print(f"[{i+1}/{len(trials)}] seed={seed}  params={{{param_str}}}")
        ok = try_variant(seed, name, vec, args.leg_placement)
        if ok:
            print("  => PASS")
            successes.append((seed, vec))
        else:
            print("  => FAIL")
        time.sleep(0.01)

    print("\n=== Summary ===")
    print(f"Total: {len(trials)}")
    print(f"Pass:  {len(successes)}")
    if successes:
        for seed, vec in successes:
            if vec is None:
                print(f"  seed={seed}  default params PASSED")
            else:
                print(f"  seed={seed}\t" + "\t".join(f"{v:.4f}" for v in vec))
    else:
        print("\nNo success. Suggestions:")
        print("  - Try with --use-default to see if defaults work.")
        print("  - Provide a known good robot_description.json via --ref-json to narrow the search.")
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())