# 任意构型步态失败诊断与稳定性约束步态选择报告

日期：2026-08-25  
项目：`/home/robot/code/yjh/Adaptation`

## 1. 结论

本轮已完成统一失败诊断、全周期可行性检查、稳定性约束候选选择、安全回退、最短圆周弧相位切换、生成器必要修复及 Isaac Gym 多构型对比。

最重要的实验结论是：当前 `adaptive_wave` 不能替代默认 `binary`。在 10 个场景、每场景 600 步的 40 个 CPU Isaac Gym episode 中，严格成功率（轨迹验收通过且无诊断失败事件）为：`binary` 40%，`hildebrand` 20%，原 `geometry_wave_1.0` 20%，新 `adaptive_wave` 20%。因此默认策略保持 `binary`。

新选择器的价值主要是“拒绝不安全计划”，而不是已经证明“走得更快”。它在标准六足与质心偏置六足上选择了可行几何候选；在多个随机 4/6/7 腿构型中没有找到满足硬约束的主动步态，按设计退到全腿支撑停止。这避免了返回未经检查的随机相位，同时明确暴露出随机构型的失败位于初始姿态、关节范围和支撑几何等更前层级，不能归因于相位策略单一因素。

## 2. 开始前检查与基线

- 未发现 `AGENTS.md`；已阅读 `README.md`、已有阶段报告及用户附件。
- 工作区开始时已有大量未提交修改，全部保留，未执行 reset、checkout 或结果删除。
- 已确认存在 `adaptation/phase.py`、`tests/test_phase.py`、`scripts/phase_strategy_compare.py`。
- 版本具有统一逐腿相位/占空比执行链路，符合附件描述的最新版本特征。
- 修改前命令与结果：

```text
PYTHONPATH=. pytest -q tests
81 passed

python -m py_compile adaptation/*.py scripts/*.py
passed

git diff --check
passed
```

## 3. 已实现内容

### 3.1 统一 episode 诊断

新增 `adaptation/diagnostics.py`，核心结构为 `EpisodeStepTelemetry`、`EpisodeDiagnosticAccumulator` 和 `EpisodeDiagnostics`。输出包括任务要求的位移、漂移、姿态 RMSE、高度、指令/实际接触、接触不匹配、足端滑移、碰撞、关节位置/速度/力矩比、实际接触 SSM、同时摆腿数、首次失败时刻及事件序列。

失败优先级固定为：数值失败、初始姿态无效、接触不足、支撑多边形失败、机身碰撞、自碰撞、关节限位、力矩饱和、roll 翻倒、pitch 翻倒、反向、前进不足、漂移过大、偏航误差过大、unknown。时间先后优先；同一时刻才使用该优先级。

为避免把正常换脚误报为失败，接触不足和支撑域失败要求持续 0.10 s；确认后仍记录异常首次出现的时间。第一轮正式实验曾发现标准六足在换脚时出现一个孤立的两足接触采样，修正后重新运行了全部正式实验。关节限位阈值也从接近限位改成达到/超过 100%，避免把 99.5% 附近的合法目标直接判作越界。

`adaptation/sim.py` 的统一仿真上下文直接读取 Isaac Gym 刚体接触、刚体状态、DOF 状态及 DOF force sensor，实际测量接触腿、接触足速度、碰撞对、足端位置、支撑 SSM 和关节比值。`phase_strategy_compare.py` 与 `validate_locomotion.py` 使用该实测链路。

`batch_test.py` 的旧隔离/并行子进程只回传轨迹，无法回传完整物理遥测，因此写入统一的“轨迹可测子集”；接触、碰撞、滑移和力矩均保持 `null` 并在 `measurement_unavailable_reasons` 中说明原因。`test_gait.py` 和 `import_isaac.py` 的独立查看/导入入口也输出同一结构的轨迹子集。没有使用估计值冒充 Isaac 实测值。

### 3.2 全周期可行性与候选选择

新增无 Isaac 依赖的 `adaptation/gait_selector.py`。默认每周期采样 360 点，并与执行器共用 `leg_phase_state`，严格执行：

```text
q_i = wrap_2pi(base_phase + phase_offset_i) / (2*pi)
stance = q_i < duty_factor_i
```

每个候选返回 `ssm_min`、`ssm_p05`、`ssm_mean`、最小支撑腿数、最大摆腿数、相位跳变代价、名义相邻摆腿碰撞风险及拒绝原因。硬拒绝非有限参数、缺失腿访问、锁止腿主动摆动、非动态模式下的退化支撑、SSM 不达标、相位跳变超限和明确的名义间距冲突。

候选包含旧 binary、Hildebrand、基于实际足端几何的 geometry wave，以及 wave count、全局相位原点、左右相位差和占空比的有限组合。搜索范围是项目实验空间，不解释为生物学普适规律。只有硬约束通过的候选才进入评分；推进能力、偏航力矩等不可精确预测量明确命名为 proxy。

回退顺序已实现为：可执行旧计划、占空比 0.8 且步幅缩放 0.5 的保守 binary、经检查的全腿支撑停止。步幅缩小只发生在第二级保守回退，并非对所有机器人硬性缩小最大步幅。

### 3.3 在线相位切换

`blend_and_validate_phase_switch` 使用圆周最短弧；0 与 2π 附近不会绕远路。缺失腿和锁止腿从插值集合剔除，锁止腿保持被动承载语义。`adaptive_wave` 支持 `transition_cycles`、`transition_cycle_index` 或显式 `transition_alpha`，每个混合中间计划重新执行完整周期检查；不安全时优先保持可行旧计划。

### 3.4 形态生成器的必要修复

代码检查确认并修复了三个明确问题：

1. `create_irregular_trunk_polygon` 原先把 `body_length/body_width` 当随机椭圆尺度，径向噪声可让最终机身外包络超过输入。现在接受的轮廓会按 X/Y 分别归一并居中，使输入等于最终刚性躯干网格 AABB。
2. 原随机安装点算法只向前推开相邻点，未验证最后一点到第一点的环形间距。现在直接构造闭环 gap，10 腿也检查周长接缝。
3. 几何和 URDF 两级 SSM 检查原先失败后仍默认输出。现在默认在写出模型前中止；只有显式 `--allow-unstable` 才生成标记清楚的诊断资产。

新增向后兼容 `morphology_type`：`irregular_rigid` 和 `serial_rigid` 已实现；`serial_flexible` 保留显式接口但抛出未实现错误，不伪装成已验证柔性躯干。

`generate_standardurdf.py` 使用长方体 extents，输入长度/宽度本来就等于刚性躯干外包络，未发现需要修改的尺寸错误。`search_morphology.py` 调用标准六足生成器，未发现独立的周向安装点逻辑，因此没有扩大修改范围。没有统一缩窄机身、增加腿长或减薄腿杆。

50 个随机轮廓的纯几何审计结果位于 `batch_results/morphology_generator_audit.json`：长度最大绝对误差 `1.11e-16 m`，宽度最大绝对误差 `5.55e-17 m`，10 腿闭环最小间距约束违规 `0/50`。这些是几何验证，不是动力学结果。

## 4. Isaac Gym 实验

### 4.1 配置

- 物理引擎：Isaac Gym PhysX，CPU pipeline。
- 正式 episode：600 步，诊断每 10 步采样一次。
- 策略：binary、hildebrand、原 geometry_wave_1.0、稳定性约束 adaptive_wave。
- 场景：标准六足；随机 4/6/7/10 腿；额外随机 4 腿种子；单腿锁止；单腿缺失（同时覆盖左右腿数不等）；双腿缺失；质心偏置。
- 随机种子：753757（4腿）、661096（6腿）、816009（7腿）、29291（10腿）、6545（4腿），共 5 个不同种子。
- 严格成功定义：原项目轨迹验收通过，且统一诊断没有安全失败事件。

短链路结果：`batch_results/gait_diagnostic_short_check.json`。  
正式原始逐 episode 结果：`batch_results/gait_failure_selection_comparison.json`。

### 4.2 策略汇总（实际 Isaac 数据）

| 策略 | 严格成功 | 仅轨迹通过 | 平均前向速度 m/s | 平均漂移比 | 平均实际 SSM min | 平均实际 SSM p05 | 平均滑移比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| binary | 4/10 | 5/10 | 0.0544 | 0.1794 | -0.0304 | 0.0512 | 0.8234 |
| hildebrand | 2/10 | 2/10 | 0.0383 | 0.1773 | 0.0817 | 0.1344 | 0.8111 |
| geometry_wave_1.0 | 2/10 | 2/10 | 0.0372 | 0.2605 | 0.0949 | 0.1354 | 0.8023 |
| adaptive_wave | 2/10 | 2/10 | 0.0444 | 0.1575 | 0.0252 | 0.1522 | 0.7027 |

adaptive_wave 的平均漂移比和滑移比低于 binary，但严格成功率明显更低，且其 SSM minimum 均值也未一致占优，不能据此修改默认策略。所有策略测得的腿—机身和腿—腿碰撞计数均为 0；平均峰值力矩比分别约为 0.0070、0.0059、0.0058、0.0054，未发现力矩饱和。滑移比有 8/40 个 episode 因没有有效接触足速度样本而为 `null`，汇总均值只使用实测非空样本。

平均 yaw/roll 数值受已经翻倒的随机构型强烈影响，不宜解释为成功步态的稳态姿态：四策略 yaw tracking RMSE 均值约 0.39～0.43 rad，roll RMSE 均值约 0.78～0.94 rad，pitch RMSE 均值约 0.020～0.025 rad。逐场景原始值应以 JSON 为准。

### 4.3 逐场景结论

- 标准六足：四策略均严格通过；adaptive 选择 `wave=1, origin=π, LR=π, D=0.5`，速度 0.234 m/s；binary 为 0.168 m/s。这里只是一个构型的一次结果，不是普适最优规律。
- 两个随机 4 腿：四策略均为 `invalid_initial_pose`；adaptive 无主动候选通过硬约束，安全停止。
- 随机 6、7 腿：主要根因为 `joint_limit`；相位变化没有移除初始关节/几何层失败。adaptive 安全停止。
- 随机 10 腿：主要根因为持续 `support_polygon_failure`；adaptive 选择回 binary，仍失败。
- 单腿锁止：binary 严格通过；其余三者轨迹标准未通过，归为 unknown（已有遥测不足以确定更具体根因）。
- 单腿缺失/左右腿数不等：binary 严格通过；Hildebrand 与原 geometry wave 为过大侧漂；adaptive 发生反向运动。
- 双腿缺失：binary 的轨迹验收通过，但发生持续支撑域失败，因此严格失败；另外三者失败。
- 质心偏置：四策略均严格通过；adaptive 速度最高，但不足以抵消它在其他场景的退化。

全 40 个 episode 的根因计数为：无失败 10、初始姿态无效 8、关节限位 8、支撑域失败 5、侧漂过大 4、unknown 3、反向 2。该分布支持“随机构型失败并非相位策略单因”的判断。

### 4.4 已测量、代理和未测量项

已测量：刚体姿态/高度、位移、实际接触腿、接触足速度、刚体接触对、DOF 位置/速度、force sensor 力矩、实际接触足位置构成的 SSM。

静态/运动学代理：候选周期内的名义足端 SSM、推进能力 proxy、左右支撑不平衡、偏航力矩 proxy、名义相邻摆腿碰撞风险和相位变化代价。它们只用于候选预筛和排序，不宣称精确预测速度、滑移或碰撞冲量。

未测量：构型/计划在线切换后的恢复时间。本实验从 episode 开始即使用目标构型，没有中途切换，因此 JSON 中为 `null` 并带原因。批量旧入口不能回传的物理量也为 `null`，不会用代理填充。

## 5. 测试与复核

新增测试覆盖任务列出的 15 类要求，并增加接触瞬态去抖、持续失败起始时刻、生成器外包络、10 腿闭环间距、URDF SSM 门控和可配置相位过渡。最终命令与结果：

```text
PYTHONPATH=. pytest -q tests
110 passed

python -m py_compile adaptation/*.py scripts/*.py
passed

git diff --check
passed
```

实际仿真命令：

```text
python scripts/phase_strategy_compare.py \
  --steps 120 --diagnostic-stride 10 --random-seed-count 5 \
  --cases standard_6 \
  --strategies binary hildebrand geometry_wave_1.0 adaptive_wave \
  --output batch_results/gait_diagnostic_short_check.json --cpu

python scripts/phase_strategy_compare.py \
  --steps 600 --diagnostic-stride 10 --random-seed-count 5 \
  --strategies binary hildebrand geometry_wave_1.0 adaptive_wave \
  --output batch_results/gait_failure_selection_comparison.json --cpu
```

最终 diff 特别复核了：相位偏置符号、`q<D` 判据、显式 leg ID 到数组/刚体索引映射、缺失/锁止腿排除、0/2π 最短弧、全周期索引、接触的 ground body `-1` 语义及空测量的 `null` 输出。

## 6. 修改文件

本任务直接新增或修改：

- `adaptation/diagnostics.py`
- `adaptation/gait_selector.py`
- `adaptation/gait.py`
- `adaptation/sim.py`
- `scripts/batch_test.py`
- `scripts/generate_geometry.py`
- `scripts/generate_urdf.py`
- `scripts/import_isaac.py`
- `scripts/phase_strategy_compare.py`
- `scripts/test_gait.py`
- `scripts/validate_locomotion.py`
- `tests/test_diagnostics.py`
- `tests/test_gait_selector.py`
- `tests/test_generate_geometry.py`
- `GAIT_FAILURE_DIAGNOSIS_AND_SELECTION_REPORT.md`

工作区中其余已有修改和结果均被保留。

## 7. 后续建议

下一步应优先针对随机构型的失败层级做分组实验：初始无碰撞姿态求解、关节可达域/默认足高检查、实际质量与 CoM 一致性，以及生成后短静态落地筛选。只有这些前置层通过后，比较相位策略才具有因果意义。建议随后扩展到每类 10 个种子和 20～30 s episode，并设计真正的 episode 中途缺腿/锁止事件，届时才能测量恢复时间。

在新策略未跨多个构型和种子稳定优于 binary 前，保持默认 `binary`。
