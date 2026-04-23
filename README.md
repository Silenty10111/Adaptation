# Adaptation

本仓库用于生成任意多边形躯干的多足机器人，并在 PyBullet、Isaac Gym、Isaac Sim 中完成资产导入、静态验证与自适应步态规划。

## 1. 当前实现范围

本次工作新增了“阶段一 + 阶段二”的自适应规划能力，目标是在随机多边形形态生成之后，自动重新定义机器人前进方向，并给出分组步态控制结果。

新增文件与作用如下：

- `adaptive_gait.py`：实现虚拟身体轴线估计、支撑安全走廊提取、头尾方向判定、质心平移代偿、腿分组与拓扑屏蔽。
- `plan_gait.py`：命令行入口，直接读取 `robot_description.json` 并输出规划结果。
- `import_isaac.py`：接入规划结果，在 Isaac Sim 中打印完整步态规划摘要，并用分组结果生成宏观摆腿示例。

## 2. 环境要求

- Linux（建议 Ubuntu）
- NVIDIA GPU 与可用驱动（Isaac Gym / Isaac Sim 场景）
- Conda
- Git

Isaac Gym 在本项目中按 Python 3.8 环境验证。

## 3. 安装

克隆仓库：

```bash
git clone <your-adaptation-repo-url> Adaptation
cd Adaptation
```

创建并激活 Isaac Gym 环境：

```bash
conda create -n unitree-rl python=3.8 -y
conda activate unitree-rl
```

安装依赖：

```bash
pip install -r requirements.txt
```

安装 Isaac Gym：

```bash
cd /data/code/yjh/isaacgym/python
pip install -e .
```

建议环境变量：

```bash
export PYTHONPATH=/data/code/yjh/isaacgym/python:${PYTHONPATH}
export LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib:${LD_LIBRARY_PATH}
```

## 4. 资产生成

在仓库根目录执行：

```bash
cd /home/robot/code/yjh/Adaptation
python generate_geometry.py
python generate_urdf.py
```

生成结果默认写入：

- `robot_assets/generated_robot.urdf`
- `robot_assets/robot_description.json`

## 5. 阶段一：虚拟身体轴线与头尾方向确定

### 5.1 基于足端点云的初始对称轴提取

`adaptive_gait.py` 会先从当前有效腿中提取支撑腿与近支撑腿：

- 支撑腿优先使用相位为 `stance` 的腿。
- 若没有显式相位，则退化为使用最低足端附近的触地点。
- 近支撑腿可以通过状态覆盖文件中的 `upcoming_stance_leg_ids` 提供。

随后在水平面上完成如下处理：

- 计算支撑中心 CoS。
- 对支撑点和近支撑点构造加权协方差矩阵。
- 对协方差矩阵做特征值分解。
- 取最大特征值对应特征向量，作为初始虚拟前向轴。

### 5.2 基于中轴线的绝对安全走廊划定

当前实现使用支撑腿足端的凸包作为动态支撑多边形，并使用“横截面拓扑细化近似”提取中轴走廊：

- 对支撑足端计算最小凸包。
- 沿初始前向轴均匀采样。
- 在每个采样位置构造法向横截线，与支撑多边形求交。
- 取每条有效横截段的中点，形成安全走廊中轴线。

这一步等价于以拓扑细化方式近似 Medial Axis，避免在当前依赖集里额外引入复杂 Voronoi 图后处理。

### 5.3 结合驱动力与动态摆动的头尾方向最终确立

头尾方向不再仅依赖静态几何轴线，而是将每条腿抽象为三阶段动作：

- 抬腿 `lift`
- 摆动 `swing`
- 落腿 `drop`

实现中只保留对前向推进有贡献的 `swing` 水平向量，并对两个候选方向 `+axis` 与 `-axis` 分别评分：

- 评分项包含摆动向量在候选前向轴上的正投影。
- 摆幅越大、摆动关节范围越大、当前相位越接近摆动态，则权重越高。
- 最终取得分更高的一侧作为最终头尾方向。

如果用户没有提供显式摆动向量，系统会根据躯干尺度、腿在机体周围的角序以及横向侧别生成默认摆动向量，用于完成初始自适应规划。

### 5.4 质心与支撑中心的对称性平移代偿

系统同时计算投影质心 CoM，并求解以下二次目标的闭式近似：

$$
\min_t \; \| (CoM + t) - CoS \|_2^2 + \lambda \|W t\|_2^2
$$

其中：

- $t$ 为躯干平移代偿量。
- $W$ 为关节力矩惩罚权重的对角近似。
- $\lambda$ 为代偿正则项。

求解结果作为 `translational_compensation_xy` 输出，用于把偏心质心拉回支撑中心附近，降低法向接触力分布不均。

## 6. 阶段二：腿间协调规律与稳定步态生成

### 6.1 基于分组控制的拓扑重组

系统会在最终前向轴确定后，将有效腿按前向投影和横向位置排序，并采用交替方式划分为两组：

- `group_a`
- `group_b`

这一步对应六足三角步态的泛化版本，目的是减少直接控制变量数量，使任意腿数下都能获得稳定的交替节律框架。

### 6.2 拓扑屏蔽与耦合权重清零

对于以下情况的腿，规划器会自动执行拓扑屏蔽：

- 明确在状态文件中声明为 `locked_leg_ids`
- 明确声明为 `missing_leg_ids`
- 在描述文件中无法找到足端、髋部或关节信息的无效腿

屏蔽后会在输出的 `topology.inhibition_rules` 中写出：

- `leg_id`
- `reason`
- `in_degree = 0.0`
- `out_degree = 0.0`

这表示控制矩阵中该节点的入度与出度耦合权重被置零，不参与分组步态传播。

## 7. 命令行使用

### 7.1 直接规划当前机器人

```bash
python plan_gait.py
```

### 7.2 输出规划 JSON

```bash
python plan_gait.py --output robot_assets/gait_plan.json
```

### 7.3 使用状态覆盖文件

```bash
python plan_gait.py --state robot_assets/gait_state.json --output robot_assets/gait_plan.json
```

状态覆盖文件支持如下字段：

```json
{
	"locked_leg_ids": [1, 4],
	"missing_leg_ids": [7],
	"upcoming_stance_leg_ids": [2, 5],
	"phases": {
		"0": "stance",
		"2": "swing",
		"3": "drop"
	},
	"swing_vectors": {
		"2": [0.16, 0.02],
		"3": [0.12, -0.03]
	},
	"torque_weights": {
		"0": 1.0,
		"2": 1.4,
		"3": 1.2
	},
	"com_xy": [0.02, -0.01]
}
```

## 8. Isaac Gym 可视化仿真

Isaac Gym 静态加载验证：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py
```

无界面运行：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py --headless --steps 120
```

运行 Isaac Gym 版宏步态示例（默认打开 Viewer）：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python import_isaac.py
```

无界面运行：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python import_isaac.py --headless --steps 1200
```

`import_isaac.py` 现在会：

- 自动计算自适应步态规划结果。
- 打印完整规划摘要 JSON。
- 选择一个有效分组作为当前宏动作组。
- 对组内各腿下发 `lift -> swing -> drop` 的示例控制序列。
- 直接在 Isaac Gym 中加载 URDF 并执行位置控制仿真。

说明：

- 如果 `unitree-rl` 环境未安装 `shapely`，脚本会自动降级为“简化分组规划”（仍可仿真）。
- 若要使用完整阶段一/阶段二几何规划，请在用于运行 `import_isaac.py` 的环境中安装 `shapely`。

## 9. 输出字段说明

`plan_gait.py` 的输出结果包含以下核心字段：

- `support_center_xy`：支撑中心 CoS。
- `projected_com_xy`：投影质心 CoM。
- `initial_virtual_forward_axis`：PCA 初始轴。
- `final_forward_axis`：结合驱动力修正后的最终前向轴。
- `drive_resultant_xy`：有效推进合力。
- `support_polygon_xy`：支撑凸包。
- `safety_corridor_xy`：安全走廊中轴采样点。
- `translational_compensation_xy`：平移代偿量。
- `planned_swings`：各腿的摆动向量与前向有效分量。
- `topology.groups`：分组控制结果。
- `topology.inhibition_rules`：拓扑屏蔽结果。

## 10. 已完成工作记录

本次已完成的工作如下：

- 为任意多边形多足机器人实现基于足端点云的虚拟前向轴估计。
- 为支撑凸包实现中轴安全走廊提取。
- 为头尾方向实现结合摆动向量与驱动力贡献的判定逻辑。
- 为 CoM 到 CoS 的对称性回拉实现二次优化近似求解。
- 为多足系统实现基于分组控制的拓扑重组。
- 为锁死腿、缺失腿、无效腿实现拓扑屏蔽规则。
- 为现有 Isaac Sim 宏控制示例接入新的分组规划结果。
- 将 README 全量改为中文，并补充算法说明、输入输出说明和使用示例。

## 11. 机器环境备注

如果 `conda run` 触发 `libtinfo` 警告，优先直接使用目标解释器：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py
```
