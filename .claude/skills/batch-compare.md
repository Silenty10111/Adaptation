---
name: batch-compare
description: Run batch robot tests with automatic comparison against baseline, generate PDF reports with charts showing SSM vs legs, forward distance vs legs, and drift analysis.
triggers:
  - "批量对比" "对比测试" "batch compare" "run comparison" "多机器人对比"
  - "/batch-compare"
---

# Batch Compare — 批量对比测试与报告

批量生成机器人变体，与基线对比，生成 PDF 对比报告。

## 使用方式

```
/batch-compare                        # 默认 30 个随机机器人 + 标准六足基线
/batch-compare --num 50               # 生成 50 个机器人
/batch-compare --seeds 7,42,137       # 指定种子
/batch-compare --standard-only        # 只测试标准六足 + 缺腿变体
/batch-compare --report               # 只生成已有结果的对比报告（不再仿真）
```

## 实现

调用 `scripts/batch_test.py`，该脚本：
1. 用 `generate_geometry.py` 生成随机机器人几何体
2. 用 `generate_urdf.py` 生成 URDF
3. 运行 SSM 检验（< 0.05m 跳过）
4. 用 `optimize_and_simulate` (probe→correct→full-sim) 进行步态仿真
5. 生成每机器人的 trajectory.png
6. 汇总所有结果为 summary.json + HTML/PDF 报告

关键参数:
- `NUM_ROBOTS = 30` — 生成数量
- `GPU_BATCH_SIZE = 10` — GPU 批量大小
- `INCLUDE_STANDARD_HEXAPOD = True` — 包含标准六足基准
- `SKIP_UNSTABLE = True` — 跳过 SSM 不达标的机器人
- `USE_ONLINE_EKF = False` — 在线偏航修正

## 输出

- `batch_results/<timestamp>/` — 完整结果（每机器人子目录 + png/）
- `batch_results/<timestamp>/summary.json` — 汇总
- `png/` — 汇总对比图 + PDF

## 对比指标

| 指标 | 说明 |
|------|------|
| SSM | 静态稳定裕度 (m) |
| Fwd Dist | 沿前进轴位移 (m) |
| Lat Dist | 侧向位移 (m) |
| Drift Ratio | 侧移/前进比 |
| Steps | 仿真步数 |
