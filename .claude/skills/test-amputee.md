---
name: test-amputee
description: Run hexapod amputation tests — remove 1 or 2 legs from the standard hexapod, run SSM + gait simulation, and compare against baseline.
triggers:
  - "测试缺腿" "缺腿测试" "amputee test" "missing leg" "断腿测试" "少一条腿" "少两条腿"
  - "/test-amputee"
---

# Test Amputee — 六足机器人缺腿测试

对标准六足机器人 (`robot_assets/standard_hexapod/`) 进行缺腿变体测试。

## 使用方式

```
/test-amputee                         # 运行全部测试（基线 + 6 缺1腿 + 11 缺2腿）
/test-amputee --missing 1             # 只测缺 1 条腿
/test-amputee --missing 2             # 只测缺 2 条腿
/test-amputee --legs 0,1              # 只测移除腿 0 和腿 1 的变体
/test-amputee --quick                 # 快速测试（减少仿真步数）
```

## 实现

运行脚本 `scripts/test_amputated_hexapod.py`，该脚本：
1. 加载标准六足 `robot_description.json`
2. 用 `amputate_legs()` 移除指定腿并重新编号
3. 用 `generate_urdf.py` 生成变体 URDF
4. 运行 SSM 静态稳定性检验
5. 用 `_RobotSimCtx` + probe→correct→full-sim 流程进行步态仿真
6. 生成 trajectory.png + summary.json

仿真参数（可在脚本中修改）:
- `SIM_STEPS = 1200` — 最短仿真步数
- `MAX_SIM_STEPS = 6000` — 最大步数
- `MIN_TRAVEL_BODY_LENGTHS = 10.0` — 达到 10 倍体长提前停止
- `USE_GPU = True` — GPU 物理加速
- `SSM_THRESHOLD = 0.05` — 静态稳定阈值

## 输出

- `amputation_results/<timestamp>/<variant>/trajectory.png` — 每变体的轨迹图
- `amputation_results/<timestamp>/summary.json` — 汇总数据

## 腿编号

标准六足的腿按逆时针排列：
```
Leg 0: 右前 (right-front)     Leg 5: 左前 (left-front)
Leg 1: 右中 (right-middle)    Leg 4: 左中 (left-middle)  
Leg 2: 右后 (right-rear)      Leg 3: 左后 (left-rear)
```
