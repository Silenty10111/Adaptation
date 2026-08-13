---
name: analyze-drift
description: Analyze drift root causes in test results — compute SSM vs drift correlation, leg symmetry impact, and generate diagnostic charts.
triggers:
  - "分析漂移" "drift analysis" "为什么漂移" "侧移分析" "analyze results"
  - "/analyze-drift"
---

# Analyze Drift — 漂移根因分析

从测试结果中分析侧向漂移的根因，生成诊断图表和建议。

## 使用方式

```
/analyze-drift <results_dir>                    # 分析指定结果目录
/analyze-drift amputation_results/20260729_212135  # 分析缺腿测试结果
/analyze-drift --latest                          # 分析最新结果
/analyze-drift --correlation                     # 计算 SSM/腿数/对称性 vs 漂移的相关性
```

## 分析维度

### 1. SSM vs Drift
- 绘制静态稳定裕度与漂移比的散点图
- 高 SSM 低漂移 = 理想；低 SSM 低漂移 = 意外（需研究）
- 低 SSM 高漂移 = 预期行为

### 2. 腿对称性 vs Drift
- 计算腿分布的不对称指数
- 对角线失腿 vs 同侧失腿的漂移对比
- Yaw 力矩名义值与实测漂移的关系

### 3. Forward Axis 质量
- 方向得分差值 (|positive - negative|) 与漂移相关性
- 差值越小表示方向越模糊，漂移风险越大

### 4. Gait Plan 诊断
- 步幅分配对称性
- 支撑多边形中心 vs 质心投影偏移
- Yaw 补偿量（net_yaw_nominal、λ）

## 实现

分析脚本路径: `scripts/analyze_batch.py`（如不存在则创建）
使用 `adaptation/stability.py`、`adaptation/gait.py` 中的分析函数。
