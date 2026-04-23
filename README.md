# Adaptation

本仓库用于生成任意多边形躯干的多足机器人，并在 PyBullet、Isaac Gym、Isaac Sim 中完成资产导入、静态验证与自适应步态规划。

## 1. 当前实现范围

本次工作新增了“阶段一 + 阶段二”的自适应规划能力，目标是在随机多边形形态生成之后，自动重新定义机器人前进方向，并给出分组步态控制结果。

本仓库所有文件作用如下：

| 文件 | 运行环境 | 作用 |
|---|---|---|
| `generate_geometry.py` | 普通 Python 环境（Adaptation） | **几何与物理参数生成器**：根据参数随机生成多边形躯干轮廓，计算腿挂载点、各链杆 STL 网格及质量惯性张量，输出 `robot_description.json`。 |
| `generate_urdf.py` | 普通 Python 环境（Adaptation） | **URDF 构建器**：读取 `robot_description.json`，先做 CGPM/SSM 静态稳定性门控，通过后才将几何与关节数据写成标准 URDF 文件（`generated_robot.urdf`）。 |
| `stability.py` | 任意环境 | **静态稳定性独立模块**：封装 CGPM/SSM 四个子函数，供 `generate_urdf.py` 调用，也可单独导入使用。 |
| `adaptive_gait.py` | 普通 Python 环境（Adaptation） | **自适应步态规划核心**：实现虚拟身体轴线估计、支撑安全走廊提取、头尾方向判定、质心平移代偿、腿分组与拓扑屏蔽。 |
| `plan_gait.py` | 普通 Python 环境（Adaptation） | **命令行规划入口**：直接读取 `robot_description.json` 并输出规划结果 JSON。 |
| `import_isaac.py` | **unitree-rl 环境**（含 Isaac Gym） | **Isaac Gym 仿真入口**：加载 URDF，执行分组交替周期步态控制，绘制前进方向地面箭头，输出稳定性与力矩裕度诊断。 |
| `test_gym.py` | **unitree-rl 环境**（含 Isaac Gym） | **Isaac Gym 静态加载验证**：批量加载多个变体 URDF，验证模型可正确导入并保持站立姿态。 |

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

### 双环境分工说明

本项目需要**两套独立的 Python 环境**，分工如下：

| 环境名 | Python 版本 | 用途 | 包含 |
|---|---|---|---|
| `Adaptation`（或系统 Python） | 3.10+ | 几何生成、步态规划 | numpy / trimesh / shapely / pybullet |
| `unitree-rl` | **3.8**（Isaac Gym 要求） | Isaac Gym 仿真 | 上述所有包 + isaacgym |

> **为什么需要两套环境？**  
> Isaac Gym 官方仅支持 Python 3.8，而 trimesh / shapely 的最新版在 3.10+ 下性能更好。  
> `generate_geometry.py` 和 `plan_gait.py` 不依赖 Isaac Gym，可在任意普通环境运行。  
> `import_isaac.py` 和 `test_gym.py` **必须在 `unitree-rl` 环境中运行**，否则无法导入 `isaacgym`。

**普通环境（Adaptation）安装额外依赖：**

```bash
conda create -n Adaptation python=3.10 -y
conda activate Adaptation
pip install -r requirements.txt
```

**切换到 unitree-rl 环境运行仿真：**

```bash
# 不激活 conda，直接使用完整路径指定解释器
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python import_isaac.py
```

## 4. 资产生成

### `generate_geometry.py` — 几何与物理参数生成器

该脚本是**机器人形态生成的第一步**，不依赖 Isaac Gym，在普通环境即可运行。主要功能：

- 使用 `shapely` 随机生成带凹口的多边形躯干轮廓（可控体长、体宽、凸凹程度）。
- 沿躯干多边形边缘计算腿挂载点，支持 `uniform`（均匀）和 `random`（聚类）两种分布模式。
- 使用 `trimesh` 生成各链杆的 STL 网格文件（hip / upper_link / lower_link / foot 等）。
- 按设定密度自动计算每个链杆的质量、质心坐标和惯性张量。
- 将全部几何与物理参数写入 `robot_assets/robot_description.json`，供后续流程使用。

### `generate_urdf.py` — URDF 构建器（备用 SSM 安全限）

该脚本是**机器人形态生成的第二步**，读取 `robot_description.json` 并输出 URDF 文件。主要功能：

- **备用 SSM 安全限**：当 `robot_description.json` 是由外部工具生成或手工编辑时，调用 `stability.validate_static_stability_before_export()` 做二次核检，SSM < 0 则中止导出。正常通过 `generate_geometry.py` 生成的文件已经通过了第一道阈值检验。
- 将链杆的惯性参数、STL 网格路径、关节类型（revolute / fixed）、关节限位、阻尼摩擦等信息按 URDF 标准格式写出。
- **Isaac Gym 导入的是 URDF 文件**（`generated_robot.urdf`），STL 网格文件是 URDF 内部引用的资产，不直接传入 Isaac Gym。
- 将 `urdf_path` 字段回写到 `robot_description.json`，保持数据一致。

在仓库根目录执行：

```bash
cd /home/robot/code/yjh/Adaptation
python generate_geometry.py
python generate_urdf.py
```

生成结果默认写入：

- `robot_assets/generated_robot.urdf`
- `robot_assets/robot_description.json`

## 4a. 静态稳定性检验（CGPM / SSM）

### 原理：重心投影法（CGPM）

重心投影法（Centre of Gravity Projection Method，CGPM）是多足机器人静态稳定性分析中最常用的判据之一（McGhee & Frank 1968）。核心思路是：**将机器人总质心投影到水平面，判断该投影点是否落在足端支撑多边形内部**。

### 静态稳定裕度（SSM）公式推导

**第一步：计算总质心 XY 投影**

设机器人共有 $M$ 个刚体链节，第 $k$ 个链节质量为 $m_k$，其质心在世界坐标系下的坐标为 $\mathbf{c}_k = (c_{k,x},\ c_{k,y},\ c_{k,z})$，则复合质心投影为：

$$
\mathbf{P}_{xy} = \frac{\sum_{k=1}^{M} m_k \cdot \mathbf{c}_k^{xy}}{\sum_{k=1}^{M} m_k}
$$

其中 $\mathbf{c}_k^{xy} = (c_{k,x},\ c_{k,y})$。

**第二步：计算支撑多边形**

取所有触地足端位置的 XY 坐标，对该点集计算凸包，得到 CCW（逆时针）顶点序列 $\mathbf{V}_0, \mathbf{V}_1, \ldots, \mathbf{V}_{n-1}$，即支撑多边形 $S$。

**第三步：多边形有向边距离**

对每条有向边 $e_i = \mathbf{V}_i \to \mathbf{V}_{i+1}$（下标模 $n$），定义从 $\mathbf{P}_{xy}$ 到该边的有符号距离：

$$
d_i = \frac{(\mathbf{V}_{i+1} - \mathbf{V}_i) \times (\mathbf{P}_{xy} - \mathbf{V}_i)}{|\mathbf{V}_{i+1} - \mathbf{V}_i|}
$$

其中 $\times$ 为二维标量叉积：$\mathbf{u} \times \mathbf{v} = u_x v_y - u_y v_x$。

对于 CCW 多边形，当 $\mathbf{P}_{xy}$ 位于边的左侧（即内侧）时 $d_i > 0$。

**第四步：SSM 定义**

$$
\boxed{\text{SSM} = \min_{i=0}^{n-1} d_i}
$$

| SSM 取值 | 物理含义 |
|---|---|
| SSM > 0 | 质心投影严格在支撑域内，静态稳定 |
| SSM = 0 | 质心投影正好在支撑域边界，临界状态 |
| SSM < 0 | 质心投影在支撑域外，静态不稳定 |

### 生成门控机制（共两道检验）

**第一道：`generate_geometry.py` 内嵌预检（主门控）**

在 `assemble_robot()` 中，生成流程分三阶段：

1. **阶段一：几何计算**——消耗随机数，计算所有腿的挂载点、色界点、**足端世界坐标**，不写任何文件。
2. **阶段二：SSM 预检**——调用 `validate_foot_layout_ssm()`，以足端 XY 凸包为支撑多边形、躯干质心估算为 $[0,0]$，计算 SSM。**SSM < 0 则立即中止，不会写出任何 STL、JSON 或 URDF 文件。**
3. **阶段三：导出网格**——检验通过后才批量写出 STL 网格、计算质量慢性参数、生成 `robot_description.json`。

**第二道：`generate_urdf.py` 备用安全限**

对手动编辑或外部工具产生的 `robot_description.json`，调用 `stability.validate_static_stability_before_export()`，SSM < 0 则中止导出 URDF。

`stability.py` 提供的函数均独立封装，可单独调用：

```python
from stability import evaluate_ssm
result = evaluate_ssm(description, threshold=0.0)
print(result["ssm"], result["passed"])
```

## 4b. 前进方向计算原理

前进方向计算分两个阶段，完整实现位于 `adaptive_gait.py`。

### 阶段一：PCA 初始轴

设共有 $L$ 条有效支撑腿，第 $i$ 条腿的足端 XY 坐标为 $\mathbf{f}_i$，权重为 $w_i$（支撑腿 1.0，近支撑腿 0.35）。

**支撑中心：**
$$
\mathbf{C}_{\text{os}} = \frac{\sum_i w_i \mathbf{f}_i}{\sum_i w_i}
$$

**加权协方差矩阵：**
$$
\Sigma = \frac{1}{\sum_i w_i} \sum_i w_i (\mathbf{f}_i - \mathbf{C}_{\text{os}})(\mathbf{f}_i - \mathbf{C}_{\text{os}})^\top
$$

**PCA 初始轴：**  对 $\Sigma$ 做特征值分解，取最大特征值对应特征向量 $\hat{\mathbf{u}}$ 作为初始候选前向轴。

### 阶段二：驱动力评分选方向

对两个候选方向 $\hat{k} \in \{+\hat{\mathbf{u}},\ -\hat{\mathbf{u}}\}$ 分别评分：

$$
s_k = \sum_{\text{腿} i} \max\!\left(\hat{k} \cdot \hat{\mathbf{v}}_i,\ 0\right) \cdot \|\mathbf{v}_i\| \cdot w_{\text{phase},i} \cdot w_{\text{range},i}
$$

其中：
- $\mathbf{v}_i$：第 $i$ 条腿的摆动向量（用户提供或自动生成）
- $\hat{\mathbf{v}}_i = \mathbf{v}_i / \|\mathbf{v}_i\|$
- $w_{\text{phase},i}$：相位增益，摆动/抬腿/落腿相取 1.15，支撑相取 0.75
- $w_{\text{range},i}$：关节范围增益 $= 1 + \min(\Delta\theta_i,\ 1.2)$，$\Delta\theta_i$ 为摆动关节角度范围（rad）

**最终前向轴：**
$$
\hat{\mathbf{d}} = \arg\max_{\hat{k} \in \{\pm\hat{\mathbf{u}}\}} s_k
$$

### 前进方向可视化

在 Isaac Gym 有 Viewer 模式下，`import_isaac.py` 每帧调用 `draw_forward_direction_line()`，在地面上绘制一个**橙色箭头**（三条 debug 线段：轴杆 + 两个箭头翼），箭尾跟踪机器人 base_link 的实时 XY 坐标，箭头朝向 `final_forward_axis`。

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
- 使用分组交替相位生成连续周期控制序列（不再只发送一次静态目标）。
- 直接在 Isaac Gym 中加载 URDF 并执行平地直线推进仿真。
- 输出静稳定性与力矩裕度诊断结果，用于区分“静稳不足”与“驱动不足”。

平地直线推进推荐参数：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python import_isaac.py \
	--headless --steps 2400 \
	--gait-frequency 0.85 \
	--swing-ratio-amplitude 0.26 \
	--stance-lift-ratio 0.54 --swing-lift-ratio 0.78 \
	--stance-drop-ratio 0.90 --swing-drop-ratio 0.38
```

新增参数说明：

- `--body-height`：初始机身高度，过低时容易起步碰撞。
- `--gait-frequency`：步态频率（Hz）。
- `--swing-ratio-amplitude`：摆动关节围绕中位点的摆幅比例。
- `--stance-lift-ratio` / `--swing-lift-ratio`：支撑相/摆动相抬腿关节目标比例。
- `--stance-drop-ratio` / `--swing-drop-ratio`：支撑相/摆动相落腿关节目标比例。

诊断指标解释：

- `static_margin_xy < 0`：质心投影在支撑域外，静稳定性存在问题。
- `drop_torque_margin_ratio < 1`：估算腿部关节力矩不足。
- `trunk_mass_ratio > 0.75`：躯干质量占比偏高，动态步态下更易过载。

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
- 在 Isaac Gym 控制脚本中新增分组交替的连续周期步态控制，用于平地直线推进。
- 在 Isaac Gym 控制脚本中新增静稳定性/质量分配/关节力矩裕度诊断输出。
- 新建 `stability.py`：独立封装 CGPM/SSM 四个子函数（CoM投影、支撑凸包、SSM计算、评估入口），含公式推导注释。
- `generate_urdf.py` 新增生成前 SSM 门控：SSM < 0 时中止导出，返回错误提示。
- `import_isaac.py` 新增 `draw_forward_direction_line()`：每帧在地面绘制橙色箭头标注机器人前进方向。
- `import_isaac.py` 移除重复的 `estimate_static_margin`，统一使用 `stability.py` 模块计算。
- README 新增 4a（CGPM/SSM 公式推导）与 4b（前进方向计算原理与公式）两节。

## 11. 机器环境备注

如果 `conda run` 触发 `libtinfo` 警告，优先直接使用目标解释器：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py
```
