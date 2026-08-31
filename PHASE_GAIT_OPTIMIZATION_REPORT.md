# 步态相位与占空比执行链路优化报告

日期：2026-08-24

## 1. 结论

本次修改已将每腿 `cpg.phase_offsets`、全局/逐腿 `duty_factor` 接入主要规划、关节目标、接触调度、批量仿真、可视化、在线估计和 MPC 调度入口。旧 gait plan 没有逐腿字段时仍回退到 A/B 两组 `0/π`，默认策略保持 `binary`。

没有通过硬性缩小全局最大步幅来实现本次优化。现有每腿步幅缩放、touchdown ramp、偏航补偿和 group-C 被动承载语义均保留；核心变化是相位、占空比和周期轨迹执行方式统一。

Isaac Gym 对比没有证明连续行波在多种构型上稳定优于二组基线，所以没有把实验策略设为默认。

## 2. 基线与环境

- 修改前：`PYTHONPATH=. pytest -q tests`，`63 passed`。
- 修改前：`python -m py_compile adaptation/*.py scripts/*.py`，通过。
- 最终：`PYTHONPATH=. pytest -q tests`，`81 passed`。
- 最终：`python -m py_compile adaptation/*.py scripts/*.py`，通过。
- Isaac Gym：Python 3.8 环境可用；显式加入 `/data/conda/envs/unitree-rl/lib` 后导入成功。
- 物理设备：RTX 3090，PhysX CUDA 设备；GPU pipeline 在当前 Isaac 配置中为 disabled。
- 仿真频率：60 Hz；正式比较每条记录 600 步，即 10 秒。

工作区开始时已有 `scripts/test_gait.py` 修改和未跟踪的 `tests/test_visual_gait_cycle.py`，本次保留并在其基础上接入统一执行链路，没有回退用户已有修改。

## 3. 实现内容

### 3.1 统一相位模块

新增 `adaptation/phase.py`，提供：

- `[0, 2π)` 角度归一化；
- 圆周距离、圆周平均、最短弧插值和在线计划切换混合；
- 全局与逐腿占空比解析，默认 `0.60`，裁剪范围 `[0.35, 0.85]`；
- 裁剪诊断，不静默改变用户输入；
- 显式 `q < D` 支撑判据；
- C² 五次 smoothstep 支撑/摆动轨迹和连续抬腿钟形曲线；
- `binary`、`hildebrand`、`geometry_wave`、`adaptive_wave` 四种策略；
- 基于有向前进轴和实际足端纵向位置的几何行波；
- 非零偏航参考的 `omega_z - omega_z_ref` 误差函数。

### 3.2 规划与故障处理

- `adaptation/gait.py`：生成并保留逐腿相位/占空比；用户显式相位优先；缺失腿退出主动集合；锁止腿进入 group C；耦合矩阵使用明确腿序；失效/被动腿相关行列实际为零并记录 `coupling_matrix_zeroed_edges`。
- `adaptation/topology.py`：虚拟相位反映射改为圆周平均；支持配置的四种策略和逐腿占空比；缺失/锁止状态向后兼容。
- `adaptation/symmetry.py`：优化器读取、优化并写回逐腿相位和占空比；接触积分使用 `q < D`；跨 `0/2π` 正则化使用圆周距离。
- `adaptation/pipeline.py`：增加统一相位配置，并把状态传给 baseline/topology 路径。
- `adaptation/estimator.py`：腿健康监测的支撑集合与执行器使用相同相位/占空比判据。

### 3.3 执行入口

以下入口已统一使用共享相位工具，不再各自通过 `sin(phase)` 正负决定接触：

- `adaptation/sim.py`
- `scripts/test_gait.py`
- `scripts/batch_test.py` 的单机、GPU 并行、探测、完整仿真和 EKF 路径
- `scripts/direction_compare.py`
- `scripts/import_isaac.py`
- `scripts/srb_mpc_demo.py`
- `scripts/test_standard_gait.py`
- `scripts/test_leg_cycle.py`
- `scripts/final_compare.py`
- `scripts/direction_c_v2.py`
- `scripts/direction_c_v3.py`

`scripts/generate_urdf.py` 增加 Python 3.8 XML 缩进兼容，仅改变 XML 排版实现，不改变 URDF 语义。

## 4. 确定性测试

新增 `tests/test_phase.py`，覆盖：

1. `D=0.60` 的采样支撑比例；
2. `D=0.50` 的采样支撑比例；
3. 显式逐腿相位优先于 A/B 组；
4. 旧计划回退到 `0/π`；
5. 标准六足一波退化为等价三脚架；
6. 4、6、7、10 腿的有限、确定相位；
7. 腿 ID/字典迭代乱序不改变几何结果；
8. 缺腿后重新归一化并保持纵向公式；
9. 锁止腿不主动摆动，耦合行列为零；
10. `0/2π` 附近圆周平均、插值和计划切换；
11. 抬腿/落地与周期边界目标连续；
12. 旧 gait plan 和逐腿占空比裁剪仍可执行；
13. 非零偏航参考参与误差计算；
14. 规划器保留显式相位并记录占空比裁剪。

全量测试从 63 增加到 81，未删除、跳过或放宽原测试。

## 5. Isaac Gym 对比

脚本：`scripts/phase_strategy_compare.py`

最终原始数据：`batch_results/phase_strategy_comparison_final.json`

比较策略：

- `binary`
- `hildebrand`
- `geometry_wave`, `wave_count=1.0`
- `geometry_wave`, `wave_count=1.25`
- `geometry_wave`, `wave_count=1.5`

比较构型：标准六足、随机 4/6/7/10 腿、锁止一腿、物理删除一腿、物理删除两腿、质心偏置，共 9 类、45 条记录。随机构型当前只使用一个已有种子，因此结果是链路和初步基线证据，不是统计显著性结论。

| 策略 | 通过/构型数 |
|---|---:|
| binary | 4/9 |
| hildebrand | 3/9 |
| geometry_wave 1.0 | 3/9 |
| geometry_wave 1.25 | 1/9 |
| geometry_wave 1.5 | 4/9 |

各构型通过策略：

| 构型 | 通过策略 |
|---|---|
| 标准六足 | 全部五种 |
| 随机 4/6/7/10 腿 | 无 |
| 标准六足锁止 leg 1 | binary、hildebrand、geometry 1.0、geometry 1.5 |
| 标准六足物理删除 leg 1 | binary |
| 标准六足物理删除原 leg 1 和 4 | binary、geometry 1.5 |
| 标准六足质心偏置 | hildebrand、geometry 1.0、geometry 1.5 |

因此保留 `binary` 默认；`geometry_wave` 和 `adaptive_wave` 仅作为实验选项。

### 5.1 标准、单缺腿、双缺腿详细数据

`v` 为前向速度 m/s，`drift` 为 `|侧向位移|/(|前向位移|+eps)`，`yaw` 为零参考下的偏航率误差绝对值 rad/s。

| 构型 | 策略 | v | drift | yaw | 最大同时摆腿 | 通过 |
|---|---|---:|---:|---:|---:|---|
| 标准六足 | binary | 0.2042 | 0.0693 | 0.00008 | 3 | 是 |
| 标准六足 | hildebrand | 0.2038 | 0.0009 | 0.00309 | 3 | 是 |
| 标准六足 | geometry 1.0 | 0.2038 | 0.0009 | 0.00309 | 3 | 是 |
| 标准六足 | geometry 1.25 | 0.2083 | 0.0519 | 0.00189 | 3 | 是 |
| 标准六足 | geometry 1.5 | 0.1634 | 0.0166 | 0.00008 | 3 | 是 |
| 单缺腿 | binary | 0.0894 | 0.0683 | 0.01023 | 3 | 是 |
| 单缺腿 | hildebrand | 0.0673 | 0.6892 | 0.00378 | 3 | 否 |
| 单缺腿 | geometry 1.0 | 0.0673 | 0.6892 | 0.00378 | 3 | 否 |
| 单缺腿 | geometry 1.25 | 0.0786 | 0.0593 | 0.03511 | 3 | 否 |
| 单缺腿 | geometry 1.5 | 0.1216 | 0.1893 | 0.01287 | 3 | 否 |
| 双缺腿 | binary | 0.1228 | 0.1353 | 0.01346 | 2 | 是 |
| 双缺腿 | hildebrand | 0.0194 | 1.7819 | 0.00074 | 2 | 否 |
| 双缺腿 | geometry 1.0 | 0.0194 | 1.7819 | 0.00074 | 2 | 否 |
| 双缺腿 | geometry 1.25 | 0.0516 | 0.0982 | 0.00483 | 2 | 否 |
| 双缺腿 | geometry 1.5 | 0.1228 | 0.1353 | 0.01346 | 2 | 是 |

SSM 的最小值和 5% 分位已记录在 JSON 中，其基准是“指令接触集合 + 名义足位”，不是力传感器重建接触。碰撞数、滑移率和构型在线切换恢复时间在当前仿真路径中没有对应遥测，JSON 保持 `null` 并逐项写明原因，没有用估计值代替。

## 6. 实际运行命令

```bash
PYTHONPATH=. pytest -q tests
python -m py_compile adaptation/*.py scripts/*.py
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib:$LD_LIBRARY_PATH \
  /data/conda/envs/unitree-rl/bin/python -c 'from isaacgym import gymapi'
PYTHONPATH=. python scripts/phase_strategy_compare.py \
  --steps 600 \
  --output batch_results/phase_strategy_comparison_final.json
git diff --check
```

结果：81 个测试通过；编译与 diff whitespace 检查通过；45 条 Isaac Gym 正式记录生成成功。

## 7. 修改文件

- 新增：`adaptation/phase.py`
- 修改：`adaptation/gait.py`、`adaptation/sim.py`、`adaptation/symmetry.py`、`adaptation/topology.py`、`adaptation/pipeline.py`、`adaptation/estimator.py`
- 修改：`scripts/test_gait.py`、`scripts/batch_test.py`、`scripts/direction_compare.py`、`scripts/import_isaac.py`、`scripts/srb_mpc_demo.py`
- 修改：`scripts/test_standard_gait.py`、`scripts/test_leg_cycle.py`、`scripts/final_compare.py`、`scripts/direction_c_v2.py`、`scripts/direction_c_v3.py`
- 修改：`scripts/generate_urdf.py`
- 新增：`scripts/phase_strategy_compare.py`
- 新增：`tests/test_phase.py`
- 新增：本报告

## 8. 仍待验证

- 对随机 4/6/7/10 腿增加多个种子和更长运行时间；当前单种子均未通过，问题不能归因于相位策略一项。
- 若要研究碰撞和滑移，应在 Isaac 执行器中增加逐足接触对、切向速度和接触力采样后再比较。
- 若要验证恢复时间，应实现一次仿真 episode 内的真实形态/计划切换，并使用已有圆周相位混合函数平滑幸存腿。
- `adaptive_wave` 当前是保守接口和规则实现，尚无证据支持设为默认或宣称最优。
