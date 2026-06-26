"""Adaptation — 自适应多足机器人步态规划框架。

子模块:
  gait        — 核心自适应步态规划 (compute_adaptive_plan)
  stability   — 静态稳定性 SSM (evaluate_ssm, compute_ssm)
  pipeline    — 多策略步态流水线 (PipelineConfig, build_gait_plan)
  sim         — Isaac Gym 仿真上下文 (_RobotSimCtx, gait constants)
  estimator   — 在线状态估计 EKF (OnlineStateEstimator)
  wbc         — 质心动力学 WBC (run_centroidal_wbc)
  symmetry    — 动态对称步态 (integrate_dynamic_symmetry)
  topology    — 拓扑不变零样本步态 (zero_shot_gait_plan)
  utils       — 共享工具 (compute_metrics, ratio_to_joint, ...)
  mpc         — MPC 桥接 (RobotPhysicsParser, AdaptiveMPCWeights)
"""
