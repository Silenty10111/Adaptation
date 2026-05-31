# Batch Test Report

**Run directory**: `/data/code/yjh/Adaptation/batch_results/20260530_221021`  
**Date**: 2026-05-30 22:55:28  
**Config**: 160 robots, seeds=[7, 42, 137, 256, 512]  
**GPU**: True, batch_size=8

---

## 1. Summary

| Category | Count | % |
|---|---:|---:|
| Total robots | 161 | 100% |
| Simulation OK | 129 | 80% |
| **Good (straight forward)** | 46 | 28% |
| Circular motion | 11 | 6% |
| Stuck / barely moving | 31 | 19% |
| Backward motion | 41 | 25% |
| Failed (gen/sim error) | 32 | 19% |

## 2. Motion Statistics (OK robots only)

| Metric | Min | Mean | Max |
|---|---:|---:|---:|
| Forward dist (m) | -5.469 | +0.114 | +3.218 |
| |Lateral| dist (m) | 0.000 | 0.758 | 3.822 |
| SSM (m) | -0.3156 | 0.1946 | 0.5800 |

**Leg count distribution**: 4-legged: 14, 5-legged: 17, 6-legged: 15, 7-legged: 19, 8-legged: 21, 9-legged: 21, 10-legged: 22

## 3. Robot Categories

**✓ Good (forward, low yaw)**:  
robot_ref_standard, robot_09_seed6743, robot_10_seed9931, robot_12_seed8716, robot_25_seed6236, robot_28_seed9916, robot_29_seed8068, robot_37_seed8436, robot_41_seed4624, robot_43_seed6780, robot_50_seed272, robot_56_seed2819, robot_60_seed6507, robot_67_seed4530, robot_71_seed4988, robot_72_seed4255, robot_73_seed3869, robot_74_seed7302, robot_75_seed4487, robot_77_seed5183, robot_78_seed357, robot_85_seed6743, robot_88_seed394, robot_89_seed4253, robot_90_seed2939, robot_91_seed4761, robot_92_seed5791, robot_94_seed9623, robot_95_seed3011, robot_100_seed6081, robot_103_seed7748, robot_105_seed5758, robot_106_seed6945, robot_111_seed6148, robot_116_seed3220, robot_118_seed6933, robot_122_seed8975, robot_131_seed3774, robot_136_seed8103, robot_137_seed5703, robot_139_seed3401, robot_146_seed7607, robot_147_seed9160, robot_153_seed4540, robot_157_seed2516, robot_159_seed4025

**↺ Circular motion** (|lat|/|fwd| > 1.8):  
robot_01_seed42, robot_14_seed4524, robot_22_seed7416, robot_23_seed2275, robot_27_seed9679, robot_31_seed5180, robot_101_seed8558, robot_104_seed511, robot_119_seed3331, robot_134_seed5426, robot_155_seed977

**✗ Stuck** (|fwd| < 0.15 m):  
robot_04_seed512, robot_11_seed193, robot_18_seed289, robot_19_seed852, robot_26_seed8434, robot_38_seed1030, robot_42_seed1043, robot_44_seed7850, robot_47_seed27, robot_52_seed7504, robot_53_seed5824, robot_59_seed6999, robot_62_seed9038, robot_70_seed9463, robot_76_seed4785, robot_79_seed8966, robot_83_seed6152, robot_98_seed3946, robot_109_seed2634, robot_117_seed157, robot_123_seed9907, robot_124_seed5812, robot_130_seed938, robot_141_seed2674, robot_143_seed9658, robot_144_seed2259, robot_148_seed6137, robot_149_seed6791, robot_150_seed5036, robot_156_seed8933, robot_158_seed2946

**← Backward** (fwd < −0.10 m):  
robot_00_seed7, robot_03_seed256, robot_05_seed4555, robot_07_seed516, robot_08_seed5160, robot_13_seed8856, robot_17_seed4386, robot_20_seed1687, robot_21_seed402, robot_32_seed3386, robot_33_seed9743, robot_36_seed4670, robot_39_seed9475, robot_40_seed7121, robot_45_seed3235, robot_46_seed9206, robot_49_seed8957, robot_51_seed9535, robot_54_seed4093, robot_55_seed8595, robot_58_seed3957, robot_61_seed851, robot_63_seed6645, robot_64_seed3372, robot_65_seed7319, robot_66_seed8563, robot_86_seed4287, robot_87_seed2107, robot_99_seed3956, robot_107_seed4226, robot_110_seed6156, robot_112_seed6515, robot_113_seed6519, robot_114_seed7969, robot_115_seed5696, robot_120_seed2931, robot_125_seed5344, robot_126_seed7668, robot_132_seed7733, robot_140_seed5729, robot_142_seed1679

**⚠ Failed**:  
robot_02_seed137, robot_06_seed4811, robot_15_seed748, robot_16_seed4597, robot_24_seed3612, robot_30_seed8625, robot_34_seed3739, robot_35_seed9949, robot_48_seed1670, robot_57_seed6912, robot_68_seed3589, robot_69_seed7560, robot_80_seed2106, robot_81_seed4649, robot_82_seed3380, robot_84_seed9882, robot_93_seed6153, robot_96_seed1780, robot_97_seed3877, robot_102_seed8403, robot_108_seed3322, robot_121_seed2835, robot_127_seed6518, robot_128_seed3374, robot_129_seed9733, robot_133_seed149, robot_135_seed1139, robot_138_seed8396, robot_145_seed993, robot_151_seed6207, robot_152_seed7104, robot_154_seed5479

## 4. Per-Robot Details

| Robot | Status | Legs | SSM | Fwd (m) | Lat (m) | Category |
|---|---|---:|---:|---:|---:|---|
| robot_ref_standard | ok | 6 | 0.580 | +2.560 | -0.646 | ✓ good |
| robot_02_seed137 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_06_seed4811 | unstable | — | -0.139 | — | — | ⚠ unstable |
| robot_15_seed748 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_16_seed4597 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_24_seed3612 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_30_seed8625 | unstable | — | 0.014 | — | — | ⚠ unstable |
| robot_34_seed3739 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_35_seed9949 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_48_seed1670 | unstable | — | -0.193 | — | — | ⚠ unstable |
| robot_57_seed6912 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_68_seed3589 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_69_seed7560 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_80_seed2106 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_81_seed4649 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_82_seed3380 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_84_seed9882 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_93_seed6153 | unstable | — | 0.005 | — | — | ⚠ unstable |
| robot_96_seed1780 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_97_seed3877 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_102_seed8403 | unstable | — | -0.120 | — | — | ⚠ unstable |
| robot_108_seed3322 | unstable | — | -0.076 | — | — | ⚠ unstable |
| robot_121_seed2835 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_127_seed6518 | unstable | — | -0.203 | — | — | ⚠ unstable |
| robot_128_seed3374 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_129_seed9733 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_133_seed149 | unstable | — | 0.018 | — | — | ⚠ unstable |
| robot_135_seed1139 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_138_seed8396 | unstable | — | 0.017 | — | — | ⚠ unstable |
| robot_145_seed993 | unstable | — | -0.084 | — | — | ⚠ unstable |
| robot_151_seed6207 | unstable | — | -0.104 | — | — | ⚠ unstable |
| robot_152_seed7104 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_154_seed5479 | unstable | — | -0.316 | — | — | ⚠ unstable |
| robot_00_seed7 | ok | 10 | 0.095 | -0.184 | +0.016 | ← backward |
| robot_01_seed42 | ok | 4 | 0.176 | +0.244 | +1.344 | ↺ circular |
| robot_03_seed256 | ok | 7 | 0.131 | -0.823 | +2.810 | ← backward |
| robot_04_seed512 | ok | 6 | 0.391 | +0.005 | +0.186 | ✗ stuck |
| robot_05_seed4555 | ok | 10 | 0.315 | -0.911 | -0.540 | ← backward |
| robot_07_seed516 | ok | 5 | 0.350 | -0.845 | +1.526 | ← backward |
| robot_08_seed5160 | ok | 9 | 0.246 | -0.516 | -3.023 | ← backward |
| robot_09_seed6743 | ok | 10 | 0.066 | +0.580 | -0.347 | ✓ good |
| robot_10_seed9931 | ok | 8 | 0.060 | +2.037 | -1.761 | ✓ good |
| robot_11_seed193 | ok | 8 | 0.297 | -0.004 | +2.287 | ✗ stuck |
| robot_12_seed8716 | ok | 10 | 0.402 | +1.721 | +2.036 | ✓ good |
| robot_13_seed8856 | ok | 8 | 0.327 | -2.900 | +0.118 | ← backward |
| robot_14_seed4524 | ok | 7 | 0.284 | +0.341 | -0.899 | ↺ circular |
| robot_17_seed4386 | ok | 6 | 0.280 | -1.019 | +0.485 | ← backward |
| robot_18_seed289 | ok | 8 | 0.224 | -0.037 | -0.215 | ✗ stuck |
| robot_19_seed852 | ok | 5 | 0.112 | +0.018 | +0.004 | ✗ stuck |
| robot_20_seed1687 | ok | 10 | 0.300 | -0.551 | -0.729 | ← backward |
| robot_21_seed402 | ok | 9 | 0.328 | -0.217 | -0.289 | ← backward |
| robot_22_seed7416 | ok | 6 | 0.282 | +0.164 | +0.894 | ↺ circular |
| robot_23_seed2275 | ok | 9 | 0.255 | +0.232 | -0.581 | ↺ circular |
| robot_25_seed6236 | ok | 7 | 0.451 | +1.035 | -1.310 | ✓ good |
| robot_26_seed8434 | ok | 8 | 0.194 | +0.078 | +1.036 | ✗ stuck |
| robot_27_seed9679 | ok | 9 | 0.187 | +0.549 | -2.031 | ↺ circular |
| robot_28_seed9916 | ok | 9 | 0.146 | +0.619 | +0.749 | ✓ good |
| robot_29_seed8068 | ok | 7 | 0.274 | +2.298 | +0.120 | ✓ good |
| robot_31_seed5180 | ok | 7 | 0.406 | +0.209 | +0.925 | ↺ circular |
| robot_32_seed3386 | ok | 4 | 0.255 | -1.324 | +1.705 | ← backward |
| robot_33_seed9743 | ok | 4 | 0.185 | -1.074 | -0.213 | ← backward |
| robot_36_seed4670 | ok | 5 | 0.319 | -1.493 | +0.081 | ← backward |
| robot_37_seed8436 | ok | 9 | 0.275 | +0.590 | +0.512 | ✓ good |
| robot_38_seed1030 | ok | 6 | 0.128 | -0.071 | -0.089 | ✗ stuck |
| robot_39_seed9475 | ok | 7 | 0.196 | -0.562 | -0.519 | ← backward |
| robot_40_seed7121 | ok | 9 | 0.301 | -0.759 | +1.384 | ← backward |
| robot_41_seed4624 | ok | 4 | 0.293 | +0.651 | -0.968 | ✓ good |
| robot_42_seed1043 | ok | 8 | 0.102 | +0.032 | +0.015 | ✗ stuck |
| robot_43_seed6780 | ok | 8 | 0.172 | +1.688 | -0.042 | ✓ good |
| robot_44_seed7850 | ok | 10 | 0.360 | -0.091 | -0.269 | ✗ stuck |
| robot_45_seed3235 | ok | 8 | 0.384 | -5.469 | +2.630 | ← backward |
| robot_46_seed9206 | ok | 8 | 0.411 | -0.758 | +1.438 | ← backward |
| robot_47_seed27 | ok | 4 | 0.095 | +0.108 | +0.008 | ✗ stuck |
| robot_49_seed8957 | ok | 9 | 0.103 | -2.204 | -1.486 | ← backward |
| robot_50_seed272 | ok | 4 | 0.278 | +1.569 | -0.985 | ✓ good |
| robot_51_seed9535 | ok | 6 | 0.193 | -0.656 | +1.113 | ← backward |
| robot_52_seed7504 | ok | 8 | 0.089 | +0.014 | +0.005 | ✗ stuck |
| robot_53_seed5824 | ok | 4 | 0.116 | +0.123 | +0.134 | ✗ stuck |
| robot_54_seed4093 | ok | 10 | 0.404 | -0.624 | -2.537 | ← backward |
| robot_55_seed8595 | ok | 9 | 0.226 | -2.676 | +1.426 | ← backward |
| robot_56_seed2819 | ok | 5 | 0.276 | +2.248 | +0.400 | ✓ good |
| robot_58_seed3957 | ok | 7 | 0.162 | -0.900 | -0.444 | ← backward |
| robot_59_seed6999 | ok | 7 | 0.227 | +0.135 | -0.114 | ✗ stuck |
| robot_60_seed6507 | ok | 9 | 0.113 | +0.405 | +0.132 | ✓ good |
| robot_61_seed851 | ok | 6 | 0.329 | -0.418 | -0.660 | ← backward |
| robot_62_seed9038 | ok | 8 | 0.127 | +0.085 | -0.047 | ✗ stuck |
| robot_63_seed6645 | ok | 4 | 0.198 | -0.102 | -0.017 | ← backward |
| robot_64_seed3372 | ok | 7 | 0.132 | -1.782 | -1.058 | ← backward |
| robot_65_seed7319 | ok | 9 | 0.194 | -0.313 | +0.528 | ← backward |
| robot_66_seed8563 | ok | 5 | 0.064 | -0.333 | +0.145 | ← backward |
| robot_67_seed4530 | ok | 7 | 0.370 | +2.416 | +0.540 | ✓ good |
| robot_70_seed9463 | ok | 5 | 0.046 | +0.144 | +0.093 | ✗ stuck |
| robot_71_seed4988 | ok | 6 | 0.213 | +2.320 | +0.048 | ✓ good |
| robot_72_seed4255 | ok | 6 | 0.335 | +1.037 | -1.083 | ✓ good |
| robot_73_seed3869 | ok | 8 | 0.184 | +0.280 | -0.326 | ✓ good |
| robot_74_seed7302 | ok | 9 | 0.091 | +0.641 | -0.117 | ✓ good |
| robot_75_seed4487 | ok | 9 | 0.218 | +2.141 | -0.866 | ✓ good |
| robot_76_seed4785 | ok | 9 | 0.047 | +0.034 | -0.122 | ✗ stuck |
| robot_77_seed5183 | ok | 8 | 0.070 | +0.246 | +0.028 | ✓ good |
| robot_78_seed357 | ok | 8 | 0.267 | +3.218 | -0.804 | ✓ good |
| robot_79_seed8966 | ok | 8 | 0.044 | +0.009 | -0.001 | ✗ stuck |
| robot_83_seed6152 | ok | 5 | 0.035 | +0.142 | -0.072 | ✗ stuck |
| robot_85_seed6743 | ok | 10 | 0.066 | +0.665 | -0.084 | ✓ good |
| robot_86_seed4287 | ok | 4 | 0.231 | -3.239 | +0.730 | ← backward |
| robot_87_seed2107 | ok | 8 | 0.260 | -2.171 | +0.368 | ← backward |
| robot_88_seed394 | ok | 5 | 0.188 | +1.095 | +0.487 | ✓ good |
| robot_89_seed4253 | ok | 10 | 0.251 | +1.283 | -0.594 | ✓ good |
| robot_90_seed2939 | ok | 4 | 0.103 | +0.480 | -0.086 | ✓ good |
| robot_91_seed4761 | ok | 4 | 0.132 | +0.175 | -0.053 | ✓ good |
| robot_92_seed5791 | ok | 5 | 0.333 | +0.529 | +0.129 | ✓ good |
| robot_94_seed9623 | ok | 8 | 0.302 | +0.410 | +0.718 | ✓ good |
| robot_95_seed3011 | ok | 10 | 0.455 | +1.663 | +1.650 | ✓ good |
| robot_98_seed3946 | ok | 9 | 0.187 | +0.113 | +2.416 | ✗ stuck |
| robot_99_seed3956 | ok | 6 | 0.106 | -0.131 | -1.424 | ← backward |
| robot_100_seed6081 | ok | 7 | 0.193 | +1.115 | +1.188 | ✓ good |
| robot_101_seed8558 | ok | 6 | 0.190 | +0.574 | -1.167 | ↺ circular |
| robot_103_seed7748 | ok | 5 | 0.315 | +0.933 | +1.046 | ✓ good |
| robot_104_seed511 | ok | 9 | 0.379 | +0.694 | +2.739 | ↺ circular |
| robot_105_seed5758 | ok | 10 | 0.051 | +0.792 | -0.052 | ✓ good |
| robot_106_seed6945 | ok | 8 | 0.162 | +1.007 | -0.936 | ✓ good |
| robot_107_seed4226 | ok | 8 | 0.265 | -1.298 | -0.141 | ← backward |
| robot_109_seed2634 | ok | 4 | 0.039 | +0.002 | +0.008 | ✗ stuck |
| robot_110_seed6156 | ok | 6 | 0.334 | -1.162 | +2.269 | ← backward |
| robot_111_seed6148 | ok | 7 | 0.167 | +0.839 | +0.999 | ✓ good |
| robot_112_seed6515 | ok | 10 | 0.244 | -0.312 | +0.887 | ← backward |
| robot_113_seed6519 | ok | 5 | 0.175 | -0.642 | -0.500 | ← backward |
| robot_114_seed7969 | ok | 7 | 0.204 | -0.733 | +1.002 | ← backward |
| robot_115_seed5696 | ok | 9 | 0.374 | -0.496 | +0.797 | ← backward |
| robot_116_seed3220 | ok | 9 | 0.136 | +0.739 | +0.086 | ✓ good |
| robot_117_seed157 | ok | 4 | 0.051 | +0.000 | -0.000 | ✗ stuck |
| robot_118_seed6933 | ok | 7 | 0.112 | +0.813 | +0.190 | ✓ good |
| robot_119_seed3331 | ok | 7 | 0.199 | +0.854 | +2.644 | ↺ circular |
| robot_120_seed2931 | ok | 10 | 0.454 | -1.909 | +0.952 | ← backward |
| robot_122_seed8975 | ok | 5 | 0.126 | +1.566 | -0.491 | ✓ good |
| robot_123_seed9907 | ok | 10 | 0.049 | -0.007 | -0.019 | ✗ stuck |
| robot_124_seed5812 | ok | 4 | 0.122 | +0.047 | -0.006 | ✗ stuck |
| robot_125_seed5344 | ok | 10 | 0.298 | -0.170 | -1.043 | ← backward |
| robot_126_seed7668 | ok | 6 | 0.404 | -0.304 | +0.055 | ← backward |
| robot_130_seed938 | ok | 5 | 0.094 | +0.016 | +0.064 | ✗ stuck |
| robot_131_seed3774 | ok | 6 | 0.116 | +2.032 | +2.392 | ✓ good |
| robot_132_seed7733 | ok | 7 | 0.390 | -0.242 | +0.277 | ← backward |
| robot_134_seed5426 | ok | 9 | 0.365 | +0.759 | +2.565 | ↺ circular |
| robot_136_seed8103 | ok | 9 | 0.174 | +0.977 | -0.030 | ✓ good |
| robot_137_seed5703 | ok | 10 | 0.126 | +0.826 | +0.960 | ✓ good |
| robot_139_seed3401 | ok | 10 | 0.452 | +0.484 | +0.412 | ✓ good |
| robot_140_seed5729 | ok | 10 | 0.327 | -0.160 | +1.448 | ← backward |
| robot_141_seed2674 | ok | 6 | 0.173 | +0.044 | -0.059 | ✗ stuck |
| robot_142_seed1679 | ok | 10 | 0.288 | -2.180 | +1.990 | ← backward |
| robot_143_seed9658 | ok | 5 | 0.120 | -0.040 | +0.019 | ✗ stuck |
| robot_144_seed2259 | ok | 8 | 0.126 | -0.080 | -0.042 | ✗ stuck |
| robot_146_seed7607 | ok | 5 | 0.350 | +1.106 | +0.834 | ✓ good |
| robot_147_seed9160 | ok | 7 | 0.168 | +0.414 | +0.431 | ✓ good |
| robot_148_seed6137 | ok | 7 | 0.085 | +0.032 | -0.121 | ✗ stuck |
| robot_149_seed6791 | ok | 5 | 0.149 | -0.055 | -0.078 | ✗ stuck |
| robot_150_seed5036 | ok | 5 | 0.069 | +0.011 | +0.002 | ✗ stuck |
| robot_153_seed4540 | ok | 10 | 0.278 | +0.164 | +0.249 | ✓ good |
| robot_155_seed977 | ok | 9 | 0.464 | +1.352 | +3.822 | ↺ circular |
| robot_156_seed8933 | ok | 8 | 0.052 | +0.136 | -0.241 | ✗ stuck |
| robot_157_seed2516 | ok | 10 | 0.218 | +0.785 | -0.636 | ✓ good |
| robot_158_seed2946 | ok | 10 | 0.198 | +0.074 | +0.275 | ✗ stuck |
| robot_159_seed4025 | ok | 7 | 0.392 | +1.100 | -0.725 | ✓ good |

## 5. Diagnosis & Recommendations

**[卡死/无法移动]** 31/129 个机器人无明显前进，
可能原因：
- 形态极端不对称导致方向选择失败
- SSM 过低导致站立不稳
- 步态频率/幅度不匹配该形态关节范围
建议：降低 `SWING_AMP`，增加 `PROBE_STEPS`

**[后退]** 41 个机器人向后运动，
说明前进方向判断出错（swing vector 方向与实际相反）。
建议：检查 `_build_default_swing_vector` 中的 outward 方向约定

**平均前进距离**: +0.114 m，**平均偏移**: 0.758 m  
（偏移/前进比越小越直；< 0.3 视为良好）

---
*Auto-generated by batch_test.py*