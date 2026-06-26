#!/usr/bin/env python3
"""分析 batch_test 的输出结果，生成统计报告。"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    ok = [r for r in data if r.get("status") == "ok"]
    failed = [r for r in data if r.get("status") != "ok"]

    # ── By leg count ──
    by_legs = defaultdict(list)
    for r in ok:
        n = r.get("num_legs", 0)
        if n > 0:
            by_legs[n].append(r)

    # ── By strategy ──
    strategies = defaultdict(list)
    for r in ok:
        s = r.get("strategy", "baseline")
        strategies[s].append(r)

    # ── EKF stats ──
    ekf_robots = strategies.get("ekf_online", [])
    baseline_robots = strategies.get("baseline", [])

    print("=" * 70)
    print("  500 机器人批量测试 — 分析报告")
    print("=" * 70)

    print(f"\n总机器人: {len(data)}")
    print(f"  仿真通过: {len(ok)} ({len(ok)/max(len(data),1)*100:.1f}%)")
    print(f"  失败/跳过: {len(failed)}")
    if failed:
        fail_types = defaultdict(int)
        for r in failed:
            fail_types[r.get("status", "unknown")] += 1
        for status, count in sorted(fail_types.items()):
            print(f"    {status}: {count}")

    # ── Strategy distribution ──
    print(f"\n策略分布:")
    for s in ["baseline", "ekf_online"]:
        count = len(strategies.get(s, []))
        print(f"  {s}: {count} ({count/max(len(ok),1)*100:.1f}%)")

    # ── Drift by strategy ──
    print(f"\n漂移比统计:")
    for s in ["baseline", "ekf_online"]:
        items = strategies.get(s, [])
        drifts = [abs(r.get("drift_ratio", float("nan"))) for r in items]
        if drifts:
            valid = [d for d in drifts if not np.isnan(d)]
            if valid:
                print(f"  {s} (n={len(valid)}): "
                      f"mean={np.mean(valid):.3f}  "
                      f"median={np.median(valid):.3f}  "
                      f"min={np.min(valid):.3f}  "
                      f"max={np.max(valid):.3f}")

    # ── By leg count ──
    print(f"\n按腿数分组:")
    header = f"{'腿数':<6} {'样本':<6} {'mean|SSM|':<12} {'mean drift':<12} {'median drift':<12} {'EKF触发':<10}"
    print(header)
    print("-" * len(header))
    for legs in sorted(by_legs.keys()):
        items = by_legs[legs]
        n = len(items)
        drifts = [abs(r.get("drift_ratio", 0)) for r in items]
        ssm_vals = [abs(r.get("ssm", 0)) for r in items]
        ekf_n = sum(1 for r in items if r.get("strategy") == "ekf_online")
        mean_d = np.mean(drifts)
        med_d = np.median(drifts)
        mean_s = np.mean(ssm_vals)
        print(f"{legs:<6} {n:<6} {mean_s:<12.4f} {mean_d:<12.3f} {med_d:<12.3f} {ekf_n}/{n}")

    # ── Improvement per robot (EKF) ──
    print(f"\nEKF 改善效果 (drift 变化):")
    ekf_drifts = [r.get("drift_ratio", float("nan")) for r in ekf_robots]
    baseline_drifts = [r.get("drift_ratio", float("nan")) for r in baseline_robots]
    if ekf_drifts:
        valid_ekf = [d for d in ekf_drifts if not np.isnan(d)]
        valid_base = [d for d in baseline_drifts if not np.isnan(d)]
        print(f"  EKF 机器人 mean drift: {np.mean(valid_ekf):.3f}" if valid_ekf else "  N/A")
        print(f"  Baseline 机器人 mean drift: {np.mean(valid_base):.3f}" if valid_base else "  N/A")

    return {
        "total": len(data),
        "ok": len(ok),
        "failed": len(failed),
        "ekf_count": len(ekf_robots),
        "baseline_count": len(baseline_robots),
        "by_legs": {l: len(v) for l, v in by_legs.items()},
    }


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is None:
        # Find latest summary
        results = sorted(Path("batch_results").glob("*/summary.json"))
        if results:
            path = results[-1]
            print(f"Using latest: {path}")
        else:
            print("No summary.json found")
            sys.exit(1)
    analyze(path)
