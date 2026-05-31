# Batch Test Report

**Run directory**: `/data/code/yjh/Adaptation/batch_results/20260529_231121`  
**Date**: 2026-05-29 23:52:03  
**Config**: 160 robots, seeds=[7, 42, 137, 256, 512]  
**GPU**: True, batch_size=8

---

## 1. Summary

| Category | Count | % |
|---|---:|---:|
| Total robots | 161 | 100% |
| Simulation OK | 114 | 70% |
| **Good (straight forward)** | 39 | 24% |
| Circular motion | 14 | 8% |
| Stuck / barely moving | 33 | 20% |
| Backward motion | 28 | 17% |
| Failed (gen/sim error) | 47 | 29% |

## 2. Motion Statistics (OK robots only)

| Metric | Min | Mean | Max |
|---|---:|---:|---:|
| Forward dist (m) | -2.229 | +0.248 | +5.499 |
| |Lateral| dist (m) | 0.003 | 0.808 | 3.297 |
| SSM (m) | -0.1702 | 0.1800 | 0.5800 |

**Leg count distribution**: 4-legged: 9, 5-legged: 18, 6-legged: 17, 7-legged: 19, 8-legged: 19, 9-legged: 17, 10-legged: 15

## 3. Robot Categories

**✓ Good (forward, low yaw)**:  
robot_ref_standard, robot_06_seed3424, robot_12_seed951, robot_13_seed5514, robot_22_seed2071, robot_30_seed5370, robot_32_seed8535, robot_35_seed7635, robot_37_seed380, robot_38_seed8349, robot_39_seed116, robot_60_seed7369, robot_68_seed4366, robot_70_seed5663, robot_71_seed4113, robot_73_seed5064, robot_83_seed3916, robot_85_seed3917, robot_93_seed8020, robot_94_seed4253, robot_100_seed7080, robot_105_seed12, robot_107_seed6472, robot_110_seed2654, robot_119_seed6638, robot_126_seed1624, robot_127_seed5769, robot_130_seed3106, robot_133_seed1527, robot_135_seed3424, robot_137_seed1260, robot_139_seed7577, robot_141_seed9423, robot_142_seed7451, robot_143_seed3762, robot_147_seed3514, robot_152_seed9404, robot_153_seed168, robot_157_seed9665

**↺ Circular motion** (|lat|/|fwd| > 1.8):  
robot_01_seed42, robot_17_seed9939, robot_25_seed2613, robot_27_seed5215, robot_42_seed2783, robot_51_seed4406, robot_58_seed9570, robot_62_seed3571, robot_64_seed614, robot_79_seed1204, robot_109_seed7269, robot_132_seed2902, robot_145_seed3331, robot_146_seed1978

**✗ Stuck** (|fwd| < 0.15 m):  
robot_04_seed512, robot_08_seed874, robot_09_seed9106, robot_10_seed8385, robot_16_seed7320, robot_20_seed2938, robot_28_seed2649, robot_29_seed3420, robot_31_seed6025, robot_36_seed937, robot_45_seed4350, robot_46_seed4949, robot_52_seed937, robot_53_seed1576, robot_57_seed8746, robot_65_seed2909, robot_66_seed3437, robot_69_seed1580, robot_75_seed7432, robot_77_seed3724, robot_90_seed2321, robot_92_seed1972, robot_95_seed6136, robot_98_seed7597, robot_101_seed9118, robot_112_seed6265, robot_116_seed8045, robot_121_seed8368, robot_123_seed2197, robot_129_seed6222, robot_134_seed7152, robot_144_seed4528, robot_151_seed3866

**← Backward** (fwd < −0.10 m):  
robot_00_seed7, robot_03_seed256, robot_07_seed1960, robot_14_seed3956, robot_15_seed4685, robot_21_seed9798, robot_26_seed5727, robot_48_seed4767, robot_54_seed9310, robot_55_seed5141, robot_56_seed5748, robot_61_seed9411, robot_67_seed5188, robot_80_seed9920, robot_84_seed8459, robot_87_seed8875, robot_88_seed6370, robot_91_seed1767, robot_97_seed9746, robot_99_seed7022, robot_108_seed8876, robot_117_seed4156, robot_118_seed4794, robot_122_seed1418, robot_125_seed5217, robot_136_seed9037, robot_138_seed9632, robot_150_seed5229

**⚠ Failed**:  
robot_02_seed137, robot_05_seed9488, robot_11_seed4681, robot_18_seed5181, robot_19_seed2504, robot_23_seed7780, robot_24_seed8393, robot_33_seed6116, robot_34_seed9654, robot_40_seed2582, robot_41_seed7221, robot_43_seed7488, robot_44_seed7513, robot_47_seed6238, robot_49_seed4597, robot_50_seed7253, robot_59_seed7044, robot_63_seed8486, robot_72_seed4392, robot_74_seed6901, robot_76_seed7463, robot_78_seed7407, robot_81_seed9571, robot_82_seed4913, robot_86_seed8371, robot_89_seed9254, robot_96_seed6926, robot_102_seed9700, robot_103_seed2989, robot_104_seed6454, robot_106_seed5740, robot_111_seed4502, robot_113_seed1436, robot_114_seed6281, robot_115_seed8403, robot_120_seed7419, robot_124_seed7209, robot_128_seed9465, robot_131_seed2554, robot_140_seed9505, robot_148_seed8988, robot_149_seed1135, robot_154_seed6495, robot_155_seed1730, robot_156_seed1575, robot_158_seed7276, robot_159_seed1362

## 4. Per-Robot Details

| Robot | Status | Legs | SSM | Fwd (m) | Lat (m) | Category |
|---|---|---:|---:|---:|---:|---|
| robot_ref_standard | ok | 6 | 0.580 | +2.560 | -0.646 | ✓ good |
| robot_02_seed137 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_05_seed9488 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_11_seed4681 | unstable | — | -0.133 | — | — | ⚠ unstable |
| robot_18_seed5181 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_19_seed2504 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_23_seed7780 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_24_seed8393 | unstable | — | -0.070 | — | — | ⚠ unstable |
| robot_33_seed6116 | unstable | — | 0.028 | — | — | ⚠ unstable |
| robot_34_seed9654 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_40_seed2582 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_41_seed7221 | unstable | — | -0.015 | — | — | ⚠ unstable |
| robot_43_seed7488 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_44_seed7513 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_47_seed6238 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_49_seed4597 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_50_seed7253 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_59_seed7044 | unstable | — | 0.027 | — | — | ⚠ unstable |
| robot_63_seed8486 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_72_seed4392 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_74_seed6901 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_76_seed7463 | unstable | — | 0.014 | — | — | ⚠ unstable |
| robot_78_seed7407 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_81_seed9571 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_82_seed4913 | unstable | — | -0.091 | — | — | ⚠ unstable |
| robot_86_seed8371 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_89_seed9254 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_96_seed6926 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_102_seed9700 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_103_seed2989 | unstable | — | 0.016 | — | — | ⚠ unstable |
| robot_104_seed6454 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_106_seed5740 | unstable | — | -0.088 | — | — | ⚠ unstable |
| robot_111_seed4502 | unstable | — | -0.170 | — | — | ⚠ unstable |
| robot_113_seed1436 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_114_seed6281 | unstable | — | 0.004 | — | — | ⚠ unstable |
| robot_115_seed8403 | unstable | — | -0.120 | — | — | ⚠ unstable |
| robot_120_seed7419 | unstable | — | 0.028 | — | — | ⚠ unstable |
| robot_124_seed7209 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_128_seed9465 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_131_seed2554 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_140_seed9505 | unstable | — | 0.004 | — | — | ⚠ unstable |
| robot_148_seed8988 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_149_seed1135 | unstable | — | 0.013 | — | — | ⚠ unstable |
| robot_154_seed6495 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_155_seed1730 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_156_seed1575 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_158_seed7276 | unstable | — | 0.024 | — | — | ⚠ unstable |
| robot_159_seed1362 | gen_failed | — | — | — | — | ⚠ gen_failed |
| robot_00_seed7 | ok | 10 | 0.095 | -0.184 | +0.016 | ← backward |
| robot_01_seed42 | ok | 4 | 0.176 | +0.248 | +1.338 | ↺ circular |
| robot_03_seed256 | ok | 7 | 0.131 | -0.772 | +2.811 | ← backward |
| robot_04_seed512 | ok | 6 | 0.391 | +0.011 | +0.193 | ✗ stuck |
| robot_06_seed3424 | ok | 5 | 0.298 | +1.157 | +0.577 | ✓ good |
| robot_07_seed1960 | ok | 6 | 0.305 | -1.713 | +0.733 | ← backward |
| robot_08_seed874 | ok | 8 | 0.052 | -0.073 | +0.077 | ✗ stuck |
| robot_09_seed9106 | ok | 8 | 0.444 | +0.127 | +0.015 | ✗ stuck |
| robot_10_seed8385 | ok | 7 | 0.181 | -0.005 | -0.051 | ✗ stuck |
| robot_12_seed951 | ok | 9 | 0.123 | +0.808 | +0.599 | ✓ good |
| robot_13_seed5514 | ok | 4 | 0.094 | +0.441 | -0.041 | ✓ good |
| robot_14_seed3956 | ok | 6 | 0.106 | -0.850 | -0.871 | ← backward |
| robot_15_seed4685 | ok | 10 | 0.212 | -0.304 | -0.142 | ← backward |
| robot_16_seed7320 | ok | 9 | 0.056 | +0.104 | +0.129 | ✗ stuck |
| robot_17_seed9939 | ok | 8 | 0.273 | +0.206 | +0.381 | ↺ circular |
| robot_20_seed2938 | ok | 6 | 0.129 | +0.126 | -0.308 | ✗ stuck |
| robot_21_seed9798 | ok | 5 | 0.247 | -0.246 | +0.867 | ← backward |
| robot_22_seed2071 | ok | 6 | 0.286 | +0.933 | +0.836 | ✓ good |
| robot_25_seed2613 | ok | 5 | 0.371 | +0.223 | +0.499 | ↺ circular |
| robot_26_seed5727 | ok | 10 | 0.336 | -1.911 | -2.352 | ← backward |
| robot_27_seed5215 | ok | 9 | 0.447 | +0.664 | -1.565 | ↺ circular |
| robot_28_seed2649 | ok | 4 | 0.076 | +0.051 | +0.282 | ✗ stuck |
| robot_29_seed3420 | ok | 6 | 0.161 | +0.031 | +1.082 | ✗ stuck |
| robot_30_seed5370 | ok | 8 | 0.234 | +5.499 | -1.990 | ✓ good |
| robot_31_seed6025 | ok | 6 | 0.058 | +0.039 | -0.118 | ✗ stuck |
| robot_32_seed8535 | ok | 4 | 0.109 | +0.484 | -0.589 | ✓ good |
| robot_35_seed7635 | ok | 8 | 0.310 | +1.677 | -2.312 | ✓ good |
| robot_36_seed937 | ok | 8 | 0.076 | -0.026 | +0.142 | ✗ stuck |
| robot_37_seed380 | ok | 7 | 0.119 | +0.268 | -0.014 | ✓ good |
| robot_38_seed8349 | ok | 7 | 0.072 | +0.600 | -0.281 | ✓ good |
| robot_39_seed116 | ok | 5 | 0.396 | +0.208 | +0.013 | ✓ good |
| robot_42_seed2783 | ok | 6 | 0.292 | +0.956 | +2.612 | ↺ circular |
| robot_45_seed4350 | ok | 7 | 0.049 | -0.094 | -0.038 | ✗ stuck |
| robot_46_seed4949 | ok | 9 | 0.392 | +0.054 | -1.141 | ✗ stuck |
| robot_48_seed4767 | ok | 8 | 0.299 | -1.757 | +0.337 | ← backward |
| robot_51_seed4406 | ok | 7 | 0.337 | +0.720 | -2.443 | ↺ circular |
| robot_52_seed937 | ok | 8 | 0.076 | +0.034 | +0.016 | ✗ stuck |
| robot_53_seed1576 | ok | 9 | 0.128 | +0.076 | +0.038 | ✗ stuck |
| robot_54_seed9310 | ok | 8 | 0.278 | -2.229 | +0.667 | ← backward |
| robot_55_seed5141 | ok | 5 | 0.268 | -1.332 | -0.132 | ← backward |
| robot_56_seed5748 | ok | 10 | 0.439 | -1.261 | +1.418 | ← backward |
| robot_57_seed8746 | ok | 6 | 0.035 | -0.033 | -0.170 | ✗ stuck |
| robot_58_seed9570 | ok | 7 | 0.331 | +0.517 | +2.453 | ↺ circular |
| robot_60_seed7369 | ok | 4 | 0.233 | +1.185 | -0.486 | ✓ good |
| robot_61_seed9411 | ok | 7 | 0.365 | -1.528 | -0.215 | ← backward |
| robot_62_seed3571 | ok | 4 | 0.231 | +0.242 | -0.576 | ↺ circular |
| robot_64_seed614 | ok | 9 | 0.461 | +0.598 | -2.967 | ↺ circular |
| robot_65_seed2909 | ok | 9 | 0.268 | -0.051 | -0.026 | ✗ stuck |
| robot_66_seed3437 | ok | 7 | 0.061 | +0.071 | +0.011 | ✗ stuck |
| robot_67_seed5188 | ok | 8 | 0.148 | -0.199 | +0.028 | ← backward |
| robot_68_seed4366 | ok | 9 | 0.126 | +0.619 | +0.180 | ✓ good |
| robot_69_seed1580 | ok | 5 | 0.088 | -0.039 | -0.051 | ✗ stuck |
| robot_70_seed5663 | ok | 5 | 0.318 | +1.525 | -1.144 | ✓ good |
| robot_71_seed4113 | ok | 8 | 0.182 | +1.832 | -0.006 | ✓ good |
| robot_73_seed5064 | ok | 7 | 0.124 | +2.353 | -0.935 | ✓ good |
| robot_75_seed7432 | ok | 5 | 0.138 | -0.024 | +0.054 | ✗ stuck |
| robot_77_seed3724 | ok | 9 | 0.136 | +0.045 | -0.096 | ✗ stuck |
| robot_79_seed1204 | ok | 8 | 0.447 | +0.234 | -2.528 | ↺ circular |
| robot_80_seed9920 | ok | 6 | 0.049 | -0.157 | -0.302 | ← backward |
| robot_83_seed3916 | ok | 8 | 0.337 | +0.863 | -1.397 | ✓ good |
| robot_84_seed8459 | ok | 7 | 0.272 | -1.094 | -1.486 | ← backward |
| robot_85_seed3917 | ok | 8 | 0.293 | +1.026 | -1.196 | ✓ good |
| robot_87_seed8875 | ok | 10 | 0.103 | -0.185 | +3.297 | ← backward |
| robot_88_seed6370 | ok | 9 | 0.179 | -0.483 | -0.446 | ← backward |
| robot_90_seed2321 | ok | 8 | 0.122 | +0.072 | +0.038 | ✗ stuck |
| robot_91_seed1767 | ok | 8 | 0.369 | -1.932 | -2.839 | ← backward |
| robot_92_seed1972 | ok | 10 | 0.059 | +0.117 | -2.812 | ✗ stuck |
| robot_93_seed8020 | ok | 5 | 0.097 | +0.177 | -0.210 | ✓ good |
| robot_94_seed4253 | ok | 10 | 0.251 | +1.618 | +1.500 | ✓ good |
| robot_95_seed6136 | ok | 6 | 0.052 | +0.021 | +0.011 | ✗ stuck |
| robot_97_seed9746 | ok | 5 | 0.184 | -0.327 | +0.110 | ← backward |
| robot_98_seed7597 | ok | 4 | 0.163 | -0.065 | -0.946 | ✗ stuck |
| robot_99_seed7022 | ok | 6 | 0.325 | -1.792 | -0.564 | ← backward |
| robot_100_seed7080 | ok | 5 | 0.214 | +1.378 | +0.850 | ✓ good |
| robot_101_seed9118 | ok | 7 | 0.043 | -0.002 | -0.024 | ✗ stuck |
| robot_105_seed12 | ok | 8 | 0.128 | +0.749 | -0.703 | ✓ good |
| robot_107_seed6472 | ok | 4 | 0.102 | +0.517 | -0.628 | ✓ good |
| robot_108_seed8876 | ok | 7 | 0.296 | -1.128 | -1.077 | ← backward |
| robot_109_seed7269 | ok | 10 | 0.235 | +0.285 | -2.248 | ↺ circular |
| robot_110_seed2654 | ok | 5 | 0.112 | +0.206 | -0.045 | ✓ good |
| robot_112_seed6265 | ok | 9 | 0.263 | +0.077 | -1.052 | ✗ stuck |
| robot_116_seed8045 | ok | 6 | 0.137 | +0.146 | +0.324 | ✗ stuck |
| robot_117_seed4156 | ok | 7 | 0.039 | -0.342 | -0.192 | ← backward |
| robot_118_seed4794 | ok | 10 | 0.388 | -0.615 | -0.684 | ← backward |
| robot_119_seed6638 | ok | 10 | 0.133 | +0.752 | -0.259 | ✓ good |
| robot_121_seed8368 | ok | 7 | 0.249 | +0.048 | +2.094 | ✗ stuck |
| robot_122_seed1418 | ok | 7 | 0.290 | -1.360 | +0.761 | ← backward |
| robot_123_seed2197 | ok | 7 | 0.079 | +0.031 | +0.060 | ✗ stuck |
| robot_125_seed5217 | ok | 8 | 0.044 | -0.475 | +1.620 | ← backward |
| robot_126_seed1624 | ok | 5 | 0.375 | +2.217 | +0.547 | ✓ good |
| robot_127_seed5769 | ok | 9 | 0.298 | +1.394 | +0.328 | ✓ good |
| robot_129_seed6222 | ok | 9 | 0.279 | +0.043 | -2.542 | ✗ stuck |
| robot_130_seed3106 | ok | 10 | 0.032 | +0.303 | +0.130 | ✓ good |
| robot_132_seed2902 | ok | 6 | 0.306 | +1.470 | +2.679 | ↺ circular |
| robot_133_seed1527 | ok | 5 | 0.266 | +2.003 | -0.507 | ✓ good |
| robot_134_seed7152 | ok | 10 | 0.041 | -0.080 | -0.079 | ✗ stuck |
| robot_135_seed3424 | ok | 5 | 0.298 | +1.304 | +0.637 | ✓ good |
| robot_136_seed9037 | ok | 10 | 0.354 | -2.085 | -1.412 | ← backward |
| robot_137_seed1260 | ok | 8 | 0.049 | +1.227 | +0.078 | ✓ good |
| robot_138_seed9632 | ok | 10 | 0.308 | -0.180 | -2.170 | ← backward |
| robot_139_seed7577 | ok | 10 | 0.048 | +0.534 | +0.110 | ✓ good |
| robot_141_seed9423 | ok | 4 | 0.162 | +1.749 | +0.028 | ✓ good |
| robot_142_seed7451 | ok | 9 | 0.370 | +1.518 | +0.196 | ✓ good |
| robot_143_seed3762 | ok | 5 | 0.374 | +0.904 | +0.534 | ✓ good |
| robot_144_seed4528 | ok | 6 | 0.074 | +0.008 | +0.028 | ✗ stuck |
| robot_145_seed3331 | ok | 7 | 0.199 | +0.633 | +2.643 | ↺ circular |
| robot_146_seed1978 | ok | 9 | 0.089 | +0.210 | +0.430 | ↺ circular |
| robot_147_seed3514 | ok | 7 | 0.256 | +1.450 | +1.130 | ✓ good |
| robot_150_seed5229 | ok | 5 | 0.101 | -0.342 | +0.175 | ← backward |
| robot_151_seed3866 | ok | 5 | 0.150 | +0.035 | +0.013 | ✗ stuck |
| robot_152_seed9404 | ok | 9 | 0.093 | +0.357 | -0.003 | ✓ good |
| robot_153_seed168 | ok | 6 | 0.225 | +0.362 | +0.296 | ✓ good |
| robot_157_seed9665 | ok | 9 | 0.388 | +2.181 | -3.005 | ✓ good |

## 5. Diagnosis & Recommendations

**[卡死/无法移动]** 33/114 个机器人无明显前进，
可能原因：
- 形态极端不对称导致方向选择失败
- SSM 过低导致站立不稳
- 步态频率/幅度不匹配该形态关节范围
建议：降低 `SWING_AMP`，增加 `PROBE_STEPS`

**[后退]** 28 个机器人向后运动，
说明前进方向判断出错（swing vector 方向与实际相反）。
建议：检查 `_build_default_swing_vector` 中的 outward 方向约定

**平均前进距离**: +0.248 m，**平均偏移**: 0.808 m  
（偏移/前进比越小越直；< 0.3 视为良好）

---
*Auto-generated by batch_test.py*