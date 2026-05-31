================================================================================
ADAPTIVE HEXAPOD GAIT CONTROL — 技术文档
================================================================================
文件: adaptive_gait.py + batch_test.py
更新: 2025-05

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第一章  前进方向的确定方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1.1  步骤概述
─────────────
前进方向由 compute_adaptive_plan(description, state, forced_axis) 自动计算，
流程分为以下四步：

  1. PCA 求主轴 → 得到 major_axis / minor_axis
  2. 构造候选方向集合（4 个轴方向 + X 轴先验）
  3. 对每个候选方向打分 _score_direction()
  4. 若 forced_axis 非 None，则直接使用指定轴（用于迭代优化器）

1.2  质量加权 PCA
─────────────────
将机器人所有足端的 (x, y) 位置做质量加权主成分分析（2D）：
- 权重：w_i = 1 (等权，因为 URDF 质量数据往往不准)
- 协方差矩阵 C = Σ (p_i - μ)(p_i - μ)^T
- 最大特征值方向 → major_axis（近似体轴方向）
- 垂直方向       → minor_axis（侧轴方向）

当足端数量 < 2 时，退化为 major_axis = [1, 0]（X 轴）。

候选方向集合：
  candidates = [+major, −major, +minor, −minor]

1.3  方向评分函数 _score_direction(direction)
─────────────────────────────────────────────
对每个候选轴计算综合得分（越高越好）：

  1. 摆动向量投影（主得分）
     ── 对每条活跃腿 i，计算默认摆动向量 v_i（足端指向体外方向）
     ── 对角步态下，摆动产生前进力：proj_i = dot(v_i, direction)
     ── score += proj_i × phase_gain × range_gain
       ▪ phase_gain ∈ [0.7, 1.0]：相位增益，对角步态中两组相位各取 1.0 / 0.7
       ▪ range_gain ∈ [0.5, 1.0]：关节范围增益，swing 关节角度范围越大越高

  2. 偏航力矩惩罚（新增）
     ── 对每条腿估算偏航杠杆臂 yaw_lever_i = 足端到前进轴的侧向距离（有符号）
     ── net_yaw = Σ proj_i × yaw_lever_i × phase_gain × range_gain
     ── score -= YAW_BALANCE_WEIGHT × |net_yaw|
        （当前 YAW_BALANCE_WEIGHT = 0.8；权重越大越避免偏航不平衡的方向）

  3. X 轴先验
     ── score += _X_PRIOR × dot(direction, [1, 0])
        （_X_PRIOR 约 0.05，轻微偏向 +X 方向以打破对称性）

  4. CoM 偏心惩罚（可选）
     ── 若 CoM 投影到 direction 上偏差过大，轻微降分，防止极偏心形态翻车

最终选择得分最高的候选方向作为 final_forward_axis。

1.4  强制轴参数（迭代优化器用）
─────────────────────────────────
compute_adaptive_plan(description, state, forced_axis=[1.0, 0.0])

当 forced_axis 非 None 时，跳过 PCA + 评分，直接使用该向量。
用于 optimize_and_simulate() 在探针失败后尝试备选轴：
  - −major_axis（反向）
  - ±minor_axis（侧向）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第二章  步态控制逻辑
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2.1  腿分组（对角三脚架步态）
───────────────────────────────
对角三脚架（alternating tripod gait）将活跃腿分为两组：

  group_a：第一组三脚架，相位 = phase_now
  group_b：第二组三脚架，相位 = phase_now + π（半周期偏移）
  group_c：被动腿（过细/过短/关节范围过小），保持站立姿势不摆动

分组规则（在 compute_adaptive_plan 中执行）：
  - 对于标准六足（6 条腿且对称排列）：
      group_a = [右前, 左中, 右后]
      group_b = [左前, 右中, 左后]
  - 对于非标准形态（腿数 ≠ 6 或不对称）：
      按足端的前进方向投影排序，交替分配到 group_a / group_b
  - 被动腿判定：per_leg_stride_amplitude < 0.15（低摆动增益）

2.2  相位计算
─────────────
  sim_time += dt          (dt = 1/60 s)
  phase_now = 2π × GAIT_FREQUENCY × sim_time

  GAIT_FREQUENCY = 0.85 Hz（默认）
  周期 T = 1 / 0.85 ≈ 1.18 s

每条活跃腿根据所在组计算自己的相位：
  leg_phase = phase_now         （group_a）
  leg_phase = phase_now + π     （group_b）

2.3  关节目标计算
─────────────────
对每条腿（lid），每帧根据 leg_phase 计算三个关节（lift / swing / drop）的目标比例：

  sw = sin(leg_phase)                   # 摆动信号 ∈ [−1, +1]
  alpha = smoothstep(−0.30, 0.30, sw)   # 地面接触判定（连续过渡）

  lift_ratio  = STANCE_LIFT_RATIO  + (SWING_LIFT_RATIO  − STANCE_LIFT_RATIO ) × alpha
  drop_ratio  = STANCE_DROP_RATIO  + (SWING_DROP_RATIO  − STANCE_DROP_RATIO ) × alpha
  swing_ratio = 0.5 + SWING_AMP × per_amp_i × dsign × sw

参数默认值：
  STANCE_LIFT_RATIO = 0.05   (支撑相：腿几乎放平)
  SWING_LIFT_RATIO  = 0.78   (摆动相：腿抬起 ~78% 行程)
  STANCE_DROP_RATIO = 0.90   (支撑相：足端下压 ~90% 行程)
  SWING_DROP_RATIO  = 0.38   (摆动相：足端缩回 ~38% 行程)
  SWING_AMP         = 0.26   (摆幅系数，乘以 per_amp_i)

侧向符号 dsign（决定摆腿方向）：
  foot_lateral = dot(foot_xy, lat_axis)   # 足端在侧轴上的投影
  dsign = −1   若 foot_lateral > 0（左侧腿，向左摆为后，向右摆为前）
  dsign = +1   若 foot_lateral ≤ 0（右侧腿）

关节目标角度转换：
  angle = lower + clamp(ratio, 0, 1) × (upper − lower)

2.4  落地缓冲（Touchdown Ramp）
──────────────────────────────
当 sw 从正（摆动）变为负（支撑）时触发 touchdown ramp：
  - 在 8 帧内用线性插值从"自由姿势"逐渐过渡到目标支撑姿势
  - 防止瞬间着地冲击导致机器人颤抖或翻倒

2.5  站立预热（HOLD_STEPS）
────────────────────────────
正式步态前先执行 HOLD_STEPS = 300 帧（5 s）的站立预热：
  - 先计算理想站立姿势 stand[]（lift/drop 接近 0°，swing 居中）
  - 设定 position control 目标 = stand[]
  - 每帧检查腿的实际位置，对下沉的腿施加"重力补偿式"增量调整：
      若 drop 不够压：stand[drop] 逐渐增大（最大 +0.012 rad/帧）
      若 lift 过高：   stand[lift] 逐渐减小（最大 −0.008 rad/帧）
  - 确保机器人稳定站立再开始步态，避免起始姿势不当导致翻倒

2.6  per_leg_stride_amplitudes（每腿独立摆幅）
──────────────────────────────────────────────
计划输出中包含 per_leg_stride_amplitudes 字典（lid → float）：
  - 基准计算：对角步态中每腿的 forward_projection × 归一化系数
  - 被动腿 (group_c) 的幅度被强制设为 0.05
  - 用于区分"强力腿"和"辅助腿"，实现非均匀摆幅步态


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第三章  偏航力矩补偿（Yaw Torque Cancellation）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.1  偏航产生原因
─────────────────
非对称形态的机器人足端不对称分布，导致各腿的摆动力产生净偏航力矩：
  net_yaw ≈ Σ amp_i × lever_i    (lever_i = 足端到前进轴的侧向距离)

若 net_yaw ≠ 0，机器人在直线步态下会持续偏转。

3.2  静态补偿（compute_adaptive_plan 内执行）
──────────────────────────────────────────────
在计划阶段对 per_leg_stride_amplitudes 做静态修正，使理论 net_yaw → 0：

  1. 计算 yaw_levers_by_leg[lid] = 足端到前进轴的侧向距离（有符号）
  2. 构造方程：Σ Δamp_i × lever_i = − net_yaw_nominal
     假设 Δamp_i = λ × lever_i（最小二乘形式）：
       λ = YAW_COMP_GAIN × net_yaw_nominal / Σ lever_i²
  3. 修正后幅度：amp_i' = clamp(amp_i − λ × lever_i, 1−MAX_ADJ, 1+MAX_ADJ)
     参数：YAW_COMP_GAIN = 0.50（保守，仅修正 50%，剩余由动态修正处理）
            YAW_COMP_MAX_ADJ = 0.30（单腿最多 ±30% 调整）

3.3  动态修正（_apply_amp_correction，基于实测偏航）
─────────────────────────────────────────────────────
在仿真探针后，根据实测偏航角速度（measured_yaw）再次修正各腿幅度：

  物理模型：偏航力矩 ∝ amp_i × |lever_i|
  修正规则（杠杆臂归一化设计）：
    s_i = 1 − strength × sign(measured_yaw) × lever_i / max_lever
    amp_i'' = clamp(amp_i' × s_i, 0.08, 0.85)
    s_i ∈ [0.25, 1.75]（夹紧）

  含义：
    - 偏转方向同侧的腿（lever_i 与偏航同号）：s_i < 1，幅度减小，减少偏航贡献
    - 偏转方向对侧的腿（lever_i 与偏航异号）：s_i > 1，幅度增大，提供对抗力矩
    - 对称腿（lever_i ≈ 0）：s_i ≈ 1，不受影响

  此设计保证量纲一致（不依赖 measured_yaw 的绝对值，只用其符号）。

3.4  迭代优化流程（optimize_and_simulate）
──────────────────────────────────────────
仿真流程（Case A / B / C 三种情况）：

  ┌─ 步骤 1：用默认计划运行探针（PROBE_STEPS = 360 步 ≈ 6 s）
  │          输出：yaw_rate, fwd_vel
  │
  ├─ Case A（直行且前进）
  │   条件：|yaw_rate| < YAW_BAD_THRESH  AND  fwd_vel > FWD_STUCK_THRESH
  │   处理：直接进行完整仿真（MAX_SIM_STEPS 步）
  │
  ├─ Case B（卡死 / 后退）
  │   条件：fwd_vel < FWD_STUCK_THRESH
  │   处理：
  │     a. 尝试 3 个备选轴 [−major, +minor, −minor]（forced_axis 参数）
  │     b. 每个轴跑一次探针，计算得分 = fwd_vel − 0.5 × |yaw_rate|
  │     c. 选最高分的轴，若也偏航则再做一次幅度修正
  │     d. 进行完整仿真
  │
  └─ Case C（偏转旋转）
      条件：|yaw_rate| ≥ YAW_BAD_THRESH
      处理（迭代修正，最多 MAX_YAW_ITERS = 3 次）：
        iter 0: strength = 0.55，修正后跑探针
        iter 1: strength = 0.73，若偏航仍超阈值
        iter 2: strength = 0.91，兜底修正
        → 用最后一次计划进行完整仿真

  常量：
    YAW_BAD_THRESH    = 0.18 rad/s
    FWD_STUCK_THRESH  = 0.004 m/s
    PROBE_STEPS       = 360 步
    MAX_YAW_ITERS     = 3

3.5  偏航补偿数据在 plan 中的位置
──────────────────────────────────
plan["yaw_balance"] 包含：
  {
    "net_yaw_nominal":   <修正前的净偏航力矩估算>,
    "net_yaw_corrected": <静态修正后的净偏航力矩>,
    "psi_by_leg":        {lid: lever_i},   # 各腿杠杆臂
    "yaw_levers":        {lid: lever_i},   # 同上，_apply_amp_correction 使用此项
  }

plan["_per_amp_override"] 存在时（动态修正后）覆盖 per_leg_stride_amplitudes。


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第四章  GPU 并行仿真（batch_test.py）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4.1  配置
──────────
  USE_GPU       = True    # 启用 PhysX GPU 加速
  GPU_BATCH_SIZE = 8      # 每批并行机器人数量

4.2  单机器人 GPU 仿真（run_gait_sim）
────────────────────────────────────────
  - sp.physx.use_gpu = True
  - create_sim(0, 0, SIM_PHYSX, sp)  # GPU device = 0
  - 对单机器人仿真已显著加速（物理求解在 GPU 上运行）

4.3  多机器人并行仿真（run_gait_sim_parallel_final）
──────────────────────────────────────────────────────
在单个 Isaac Gym 仿真实例中同时运行 N 个机器人：
  1. 创建一个 GPU SIM
  2. 每个机器人创建独立 env，间距 SPACING = 12 m（防止干扰）
  3. 所有机器人共享同一个 gym.simulate(sim) 调用（真正的 GPU 并行）
  4. 每帧分别为每个机器人计算关节目标（Python 侧串行，但物理并行）
  5. 返回 [(com_trail_i, forward_axis_i), ...]

  降级策略：若 create_sim 失败（无 GPU），自动串行执行 run_gait_sim。

4.4  批次主循环（main）
──────────────────────
  for i in range(0, N, GPU_BATCH_SIZE):
      batch = robots[i:i+GPU_BATCH_SIZE]
      results = run_gait_sim_parallel_final(batch_configs, use_gpu=USE_GPU)
      # 逐机器人记录结果 → summary_rows


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第五章  批次报告（generate_batch_report）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

每次批量测试结束后自动生成 report.md，内容包括：
  - 总体统计：OK/失败数量
  - 运动分类：
      ✓ 好（直行前进）：fwd > 0, |lat/fwd| < 1.8
      ↺ 圆弧（偏航过大）：|lat/fwd| > 1.8
      ✗ 卡死：|fwd| < 0.15 m
      ← 后退：fwd < −0.10 m
      ⚠ 失败：仿真或生成错误
  - 运动统计：fwd/lat 距离均值/最值，SSM 分布
  - 每台机器人的详细表格
  - 自动诊断与建议（中文）

报告路径：<batch_dir>/report.md


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
附录：关键常量速查表
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

文件: adaptive_gait.py
  YAW_BALANCE_WEIGHT  = 0.8      # 方向评分中偏航惩罚权重
  _X_PRIOR            = ~0.05    # X 轴方向先验强度
  YAW_COMP_GAIN       = 0.50     # 静态偏航补偿强度（0=不补，1=全补）
  YAW_COMP_MAX_ADJ    = 0.30     # 单腿幅度最大调整量（±30%）

文件: batch_test.py
  GAIT_FREQUENCY      = 0.85 Hz  # 步态频率
  SWING_AMP           = 0.26     # 摆幅基准系数
  STANCE_LIFT_RATIO   = 0.05     # 支撑相抬腿比例
  SWING_LIFT_RATIO    = 0.78     # 摆动相抬腿比例
  STANCE_DROP_RATIO   = 0.90     # 支撑相落足比例
  SWING_DROP_RATIO    = 0.38     # 摆动相收足比例
  BODY_HEIGHT         = 0.50 m   # 初始质心高度
  HOLD_STEPS          = 300      # 预热帧数（5 s @ 60 Hz）
  SIM_STEPS           = 1200     # 标准仿真帧数（20 s）
  MAX_SIM_STEPS       = 7200     # 最大仿真帧数（120 s）
  MIN_TRAVEL_BODY_LENGTHS = 2.0  # 提前终止前进阈值（体长倍数）
  PROBE_STEPS         = 360      # 探针仿真帧数（6 s）
  YAW_BAD_THRESH      = 0.18 r/s # 偏航速率告警阈值
  MAX_YAW_ITERS       = 3        # 最大偏航修正迭代次数
  USE_GPU             = True     # 启用 GPU 加速
  GPU_BATCH_SIZE      = 8        # 并行仿真批大小

================================================================================
