# Adaptation

任意多边形躯干多足机器人的资产生成、静态稳定性验证与自适应步态规划。

## 核心文件

| 文件 | 环境 | 作用 |
|---|---|---|
| `generate_geometry.py` | Adaptation | 随机生成多边形躯干、腿挂载点、STL 网格、质量惯性参数 → `robot_description.json` |
| `generate_urdf.py` | Adaptation | 读取 JSON，SSM 门控后输出标准 URDF |
| `stability.py` | 任意 | CGPM/SSM 静态稳定性独立模块（4 个子函数） |
| `adaptive_gait.py` | Adaptation | **自适应步态规划核心**：PCA 体轴估计、安全走廊、头尾方向、腿分组、偏航补偿 |
| `plan_gait.py` | Adaptation | 命令行规划入口，输出 gait plan JSON |
| `ssm_visualizer.py` | Adaptation | SSM 可视化，输出 `png/` 分析图 |
| `batch_test.py` | unitree-rl | 批量生成 + SSM 过滤 + GPU 并行仿真 + HTML 报告 |
| `test_gait.py` | unitree-rl | 自适应步态验证（前进箭头/支撑多边形/分组标记可视化） |
| `gait_pipeline.py` | unitree-rl | 多策略步态编排（baseline/DynSym/Topo），可配置门控 |
| `iterative_improve.py` | unitree-rl | 自动评估→分析→调参迭代循环 |
| `ablation_study.py` | unitree-rl | 5 种步态方法消融对比实验 |
| `utils.py` | 任意 | 共享工具（数学/运动学/环境切换/指标计算） |
| `dynamic_symmetry_gait.py` | 任意 | 动态对称步态：相位/占空比/步幅优化 |
| `topology_invariant_mapper.py` | 任意 | 拓扑不变映射：N 腿 → 4 节点虚拟支撑多边形，零样本步态 |
| `centroidal_wbc.py` | 任意 | 质心动力学 + QP 地面反力分配 + 偏航力矩平衡 |
| `online_state_estimator.py` | 任意 | EKF/RMA 在线状态估计（CoM 偏移/摩擦/腿健康） |
| `adaptation/morphology.py` | 任意 | 安全缺腿变换、腿编号原子重映射与拓扑完整性检查 |
| `adaptation/validation.py` | 任意 | 直线匀速轨迹指标与统一通过标准 |
| `adaptation/autotune.py` | 任意 | 仿真候选生成、评分、行走轴闭环重估 |
| `scripts/validate_locomotion.py` | unitree-rl | 标准、缺腿、任意构型的有序自动验证入口 |

## 环境

- Linux + NVIDIA GPU + Conda
- **两套 Python 环境**：

| 环境 | Python | 用途 |
|---|---|---|
| `Adaptation` | 3.10+ | 几何生成、步态规划（numpy, shapely, trimesh, pybullet） |
| `unitree-rl` | 3.8 | Isaac Gym 仿真（上述所有 + isaacgym） |

## 快速开始

```bash
# 1. 安装依赖
conda create -n Adaptation python=3.10 -y && conda activate Adaptation
pip install -r requirements.txt

conda create -n unitree-rl python=3.8 -y
# Isaac Gym 需单独安装：cd /path/to/isaacgym/python && pip install -e .

# 2. 设置环境变量（可选，代码内有 fallback）
export ISAAC_PYTHON=/data/conda/envs/unitree-rl/bin/python
export ISAAC_LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib
export ADAPTATION_PYTHON=/data/conda/envs/Adaptation/bin/python

# 3. 生成机器人（默认：0.90×0.32 m 细长主体，随机 8–14 条腿）
python generate_geometry.py --seed 42

# 显式覆盖仍可生成标准腿数或其他尺寸
python generate_geometry.py --num-legs 6 --seed 42
python generate_urdf.py

# 4. 步态规划
python plan_gait.py --output robot_assets/gait_plan.json

# 5. 仿真验证
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib \
  /data/conda/envs/unitree-rl/bin/python test_gait.py --headless --steps 2400

# 6. 批量测试
python scripts/batch_test.py --num-robots 500 --execution-mode parallel
```

批量测试固定保存可复现随机种子、机器人资产、1200 步根机身轨迹、统一直线匀速指标，以及每台机器人的：

- `trajectory.png` / `trajectory.json`：实际根机身路径与轨迹数据；
- `gait_heatmap.png`：按实际足端几何分成左、右两张子图的摆动关节目标热力图；
- `gait_visualization.json`：相位、占空比和绘图栅格原始数据。

热力图是控制器目标，不是实测 DOF 状态，图标题和 JSON 中会明确标注。支撑相和摆动相仍保留在 JSON 原始数据中，但不再单独绘图。汇总 HTML 可直接放大查看路径图和左右腿热力图。可使用 `--resume-dir PATH` 复用已生成资产，避免长批次重新生成几何。

旧批次没有保存最终探测计划或实测关节轨迹，可根据已保存构型和行走轴重建带来源标记的计划图：

```bash
python scripts/generate_gait_visuals.py batch_results/20260818_131804
```

### 在 Isaac Gym Viewer 中查看指定机器人

打开 `scripts/test_gait.py`，只需修改文件开头的 `ROBOT_MODEL_PATH`：

```python
# 标准六足
ROBOT_MODEL_PATH = REPO_ROOT / "robot_assets/standard_hexapod"

# 500 台批测中的指定机器人
ROBOT_MODEL_PATH = REPO_ROOT / "batch_results/20260818_131804/robot_00_seed7"
```

该地址可以是机器人目录、URDF 文件或 `robot_description.json`。直接运行下面的命令会打开 Viewer；关闭窗口即可结束仿真：

```bash
python scripts/test_gait.py
```

也可不修改源码，临时通过命令行覆盖：

```bash
python scripts/test_gait.py \
  --model batch_results/20260818_131804/robot_00_seed7
```

Viewer 中橙色箭头表示规划前进方向，绿色线框表示支撑多边形，青色十字表示机身投影，足端标记颜色表示当前腿组和支撑/摆动状态。

## 统一直线行走验证

`scripts/validate_locomotion.py` 会在 Isaac Gym 中自动探测候选方向和步态，失败后根据实际轨迹重新估计行走轴，再做长时确认。固定验证顺序为：标准六足 → 标准六足缺腿 → 任意生成机器人。

```bash
# 完整顺序验证
python scripts/validate_locomotion.py --stage all

# 分阶段验证
python scripts/validate_locomotion.py --stage standard \
  --output validation_results/standard.json
python scripts/validate_locomotion.py --stage amputated \
  --output validation_results/amputated.json
python scripts/validate_locomotion.py --stage arbitrary --arbitrary-count 5 \
  --output validation_results/arbitrary.json
```

脚本会自动切换到 `unitree-rl` 环境。每个机器人只有在 1200 步长时仿真中同时满足前向速度、侧向速度、轨迹漂移、行进方向、周期速度波动、机身高度及高度波动阈值后才标记为 `PASS`。可用 `--robots missing_legs_23` 单独复验指定机器人。

## 核心算法概要

### 静态稳定性（CGPM / SSM）

重心投影法：将复合质心投影到水平面，判断是否落在足端凸包内部。

- `com_xy` = Σ(m_k · c_k_xy) / Σ m_k
- `support_polygon` = 足端 XY 的 CCW 凸包
- `SSM` = min_i( cross(V_{i+1} - V_i, P - V_i) / |V_{i+1} - V_i| )
- SSM > 0 → 稳定，SSM ≤ 0 → 不稳定

两道门控：`generate_geometry.py` 预检（生成前）+ `generate_urdf.py` 备用检查（导出前）。SSM 失败默认中止；只有显式 `--allow-unstable` 才输出诊断资产。

详见 `stability.py`。

### 自适应步态规划

**阶段一 — 前进方向确定**：
1. 对支撑足端加权协方差矩阵做 PCA → 初始虚拟轴 ±û
2. 对 ±û 分别计算驱动力评分：s = Σ max(方向投影, 0) · 摆幅 · 相位增益 · 关节范围增益
3. 选高分侧为 `final_forward_axis`

**阶段二 — 腿分组与步态控制**：
1. 足端按前向轴投影排序，交替分配 group_a / group_b
2. 两组相位差 π，形成交替三角步态（泛化至任意腿数）
3. 质心→支撑中心平移代偿，拓扑屏蔽规则处理锁死/缺失腿

详见 `adaptive_gait.py`。

### 高级策略

| 策略 | 文件 | 原理 |
|---|---|---|
| Dynamic Symmetry | `dynamic_symmetry_gait.py` | 优化相位/占空比/步幅使偏航力矩积分为零 |
| Topology Invariant | `topology_invariant_mapper.py` | RBF 软分配将 N 腿映射到 4 节点虚拟多边形 |
| Centroidal WBC | `centroidal_wbc.py` | QP 分配 GRF + 质心动力学偏航平衡 |
| Online Estimation | `online_state_estimator.py` | EKF CoM偏移 + RLS 摩擦 + 腿健康监测 |

使用 `gait_pipeline.py` 自动选择策略，`iterative_improve.py` 自动迭代调优。

## CLI 参考

```bash
# 步态规划
python plan_gait.py [--output PATH] [--state state.json]

# 状态覆盖示例 (state.json)
{
  "locked_leg_ids": [1, 4],
  "missing_leg_ids": [7],
  "phases": {"0": "stance", "2": "swing"},
  "swing_vectors": {"2": [0.16, 0.02]},
  "com_xy": [0.02, -0.01]
}

# SSM 参数扫描
python test_generate_ssm.py --trials 20 [--use-default] [--ref-json PATH --jitter 0.05]

# 步态周期 PDF 报告
python test_leg_cycle.py --gait-frequency 0.85 --output png/report.pdf

# Isaac Gym 仿真（需 unitree-rl 环境）
ISAAC_PY=/data/conda/envs/unitree-rl/bin/python
ISAAC_LD=/data/conda/envs/unitree-rl/lib

LD_LIBRARY_PATH=$ISAAC_LD $ISAAC_PY test_gym.py --headless --steps 120
LD_LIBRARY_PATH=$ISAAC_LD $ISAAC_PY test_gait.py --headless --steps 2400 --hold-steps 300
LD_LIBRARY_PATH=$ISAAC_LD $ISAAC_PY import_isaac.py --headless --steps 2400 \
    --body-height 0.50 --gait-frequency 0.85 --swing-ratio-amplitude 0.26
```

## 仿真参数参考

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--body-height` | 0.50 m | 初始机身高度 |
| `--gait-frequency` | 0.85 Hz | 步态频率 |
| `--swing-ratio-amplitude` | 0.26 | 摆动关节摆幅比例 |
| `--stance-lift-ratio` | 0.05 | 支撑相抬腿 |
| `--swing-lift-ratio` | 0.78 | 摆动相抬腿 |
| `--stance-drop-ratio` | 0.90 | 支撑相落腿 |
| `--swing-drop-ratio` | 0.38 | 摆动相落腿 |
| `--hold-steps` | 300 | 步态前静止稳定步数 |

## 调试备忘

- **`collapse_fixed_joints = True`**：必须设置，否则髋关节球体会与机体碰撞导致腿不动。
- **PD 增益**：机体 ~10 kg 时 stiffness ≥ 200 N·m/rad，否则机身会缓慢下沉。
- **Smoothstep 窗口**：用 `±0.30` 而非 `±0.05`（占周期 ~25%），消除支撑↔摆动切换冲击。
- **摆腿方向符号**：用侧向轴点积（非前进轴），确保左右腿挥动方向一致向前。
- **SSM 阈值**：`batch_test.py` 默认 0.03 m，低于此值跳过仿真以避免 GPU 内存越界。
- **环境切换**：脚本会自动 re-exec 到 unitree-rl 环境，设置 `BATCH_REEXEC=1` 可跳过。

## 输出结构

```
robot_assets/          # 生成资产（.json, .urdf, meshes/）
batch_results/         # 批量测试结果（按时间戳分目录）
iterations/            # 迭代改进快照
ablation_results/      # 消融实验报告
png/                   # 可视化图片/PDF
```

## 测试

```bash
# 单元测试（Adaptation 环境）
python -m pytest tests/ -v
```
# 最新闭环步态验证流程（2026-08）

任意构型不再只使用“左右两组相差 π”的固定逻辑。当前批测入口会在
`binary`、`hildebrand`、`uniform_wave`、`balanced_wave` 和几何波候选间
比较，并在 JSON 的 `selected_by_case` 中记录优胜策略。轨迹通过和严格接触
安全通过是两个独立字段，禁止把前者写成完整安全成功。

推荐的任意构型验证命令（结果和所有图片均位于 `batch_results`）：

```bash
PYTHONPATH=. python scripts/phase_strategy_compare.py \
  --models-root batch_results/generation_coplanar_8 \
  --steps 600 --calibrate-course --calibration-probe-steps 600 \
  --swing-sign-mode kinematic_jacobian --stride-direction -1 \
  --touchdown-settle-steps 0 --disable-emergency-contact-recovery \
  --strategies binary hildebrand uniform_wave balanced_wave \
  --output batch_results/phase_selection/result.json \
  --artifacts-dir batch_results/phase_selection/artifacts
```

对接触失配较高的优胜策略，再增加一次 `--latch-stance-search` 候选；该滞回
补偿只应由新确认回合择优，不能全局强制开启（当前样本中 seed 19/23 会退化）。

探测与确认使用两个全新的 PhysX 上下文，避免接触求解器历史状态污染复位后
的结果。`final_forward_axis` 表示实测稳定运动方向；
`actuation_forward_axis` 保留生成摆动符号和腿序的原形态轴，因此航向标定不会
暗中重建另一套步态。

当前汇总报告可重新生成：

```bash
python scripts/build_final_iteration_report.py
```

输出包括 `batch_results/final_iteration_summary/summary.json`、嵌图 HTML 和固定
A4 横向版式 PDF。热力图按左腿、右腿分成两个面板，显示的是规划摆动关节目标，
不是 Isaac Gym 实测 DOF 状态。
