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
| `test_generate_ssm.py` | 普通 Python 环境（Adaptation） | **几何参数扫描器**：批量调用 `generate_geometry.py`，用于测试不同参数/种子下的 SSM 通过率，并输出通过组合统计。 |
| `ssm_visualizer.py` | 普通 Python 环境（Adaptation） | **SSM 可视化工具**：从描述文件计算支撑多边形与 SSM，并把分析图保存到 `png/` 目录。 |
| `import_isaac.py` | **unitree-rl 环境**（含 Isaac Gym） | **Isaac Gym 仿真入口**：加载 URDF，执行分组交替周期步态控制，绘制前进方向地面箭头，输出稳定性与力矩裕度诊断。 |
| `test_gym.py` | **unitree-rl 环境**（含 Isaac Gym） | **Isaac Gym 静态加载验证**：批量加载多个变体 URDF，验证模型可正确导入并保持站立姿态。 |
| `test_gait.py` | **unitree-rl 环境**（含 Isaac Gym） | **Isaac Gym 自适应步态验证**：结合 `plan_gait.py` 计算前进方向，分组交替步态，可视化前进箭头 / 支撑多边形 / 分组标记 / 质心投影，输出直线运动轨迹及姿态角。 |
| `test_leg_cycle.py` | 普通 Python 环境（Adaptation） | **步态周期 PDF 报告生成器**：模拟一个完整步态周期，绘制所有腿的 lift/swing/drop 关节目标曲线 + 相位甘特图 + 统计摘要，输出 PDF。 |

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

### 7.4 SSM 参数扫描（`test_generate_ssm.py`）

`test_generate_ssm.py` 用于批量试验几何参数组合，快速评估 `generate_geometry.py` 的 SSM 通过率。

常见用法：

```bash
# 使用脚本内默认范围随机试验
python test_generate_ssm.py --trials 20

# 只测试 generate_geometry.py 默认几何（不传几何参数）
python test_generate_ssm.py --trials 10 --use-default

# 以已有 robot_description.json 为基准，在其附近抖动搜索
python test_generate_ssm.py \
	--trials 30 \
	--ref-json robot_assets/robot_description.json \
	--jitter 0.05
```

说明：

- `--use-default`：每次仅改变种子，不传 `--body-length/--body-width/...` 参数。
- `--ref-json`：从参考 JSON 读取几何基准值，再按 `--jitter` 比例做随机扰动。
- `--seed-base` 与 `--seed-step`：控制批量试验时的种子序列。
- 结束后会输出 PASS/FAIL 汇总，并列出通过样本参数。

### 7.5 `png/` 图片生成功能位置

当前 `png/` 目录图片的生成逻辑在可视化脚本中，关键位置如下：

- `ssm_visualizer.py`：
	- `OUTPUT_DIR = REPO_ROOT / "png"` 定义输出目录。
	- `SSMVisualizer.plot(...)` 内部使用 `fig.savefig(save_path, ...)` 负责最终落盘。
	- `main()` 中调用 `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` 创建目录。
	- `main()` 里拼接 `filename = f"{robot_name}_{ssm_status}_ssm{...}.png"` 并写入 `save_path = OUTPUT_DIR / filename`。

- `adaptive_gait.py`：
	- 当前文件中存在与 `ssm_visualizer.py` 同步的同名可视化实现（同样包含 `OUTPUT_DIR`、`plot()` 和 `fig.savefig(...)` 路径）。
	- 若你后续只保留一份可视化实现，建议优先保留 `ssm_visualizer.py` 作为单一入口。

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
	--body-height 0.50 \
	--gait-frequency 0.85 \
	--swing-ratio-amplitude 0.26 \
	--stance-lift-ratio 0.05 --swing-lift-ratio 0.78 \
	--stance-drop-ratio 0.90 --swing-drop-ratio 0.38
```

新增参数说明：

- `--body-height`：初始机身高度（默认 0.50 m），需匹配腿长使足端能够触地。
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

### 8a. `test_gait.py` — 自适应步态验证

`test_gait.py` 将 `plan_gait.py` 的规划结果接入 Isaac Gym，执行**分组交替步态**并可视化。

**运行方式：**

```bash
# 有界面 — 实时观察前进方向、分组标记、支撑多边形
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib \
  /data/conda/envs/unitree-rl/bin/python test_gait.py

# 无界面 — 输出运动距离与姿态角
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib \
  /data/conda/envs/unitree-rl/bin/python test_gait.py \
  --headless --steps 2400 --hold-steps 300
```

**可视化标注（Viewer 模式）：**

| 标注 | 颜色 | 含义 |
|---|---|---|
| 前进方向箭头 | **橙色** (1.5 m 长) | 由 `final_forward_axis` 确定的头尾方向 |
| 支撑多边形边框 | **绿色** | 足端凸包（实时跟踪 body_xy 偏移） |
| 质心十字 | **青色** | `projected_com_xy` 投影质心位置 |
| 足端标记 | **蓝色** = group_a, **红色** = group_b | 腿分组标识 |
| 足端明暗 | **亮色** = 支撑相, **灰色** = 摆动相 | 当前步态相位 |
| 终端输出 | 每 200 帧打印 `heading = xxx°` | 前进方向角 |

**无界面输出示例：**

```
[ 400/2400] body_xy = [0.0712, -0.0031]  roll = -1.2°  pitch = 0.8°
...
[Motion summary after 2400 steps]
  forward dist  = +0.4221 m
  lateral drift = -0.0193 m
```

**新增 / 变更参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hold-steps` | **300** | 步态启动前静止稳定步数（自适应 sag 补偿） |
| `--gpu-pipeline` | off | 启用 GPU 渲染管线 |
| `--body-height` | 0.50 m | 初始机身高度 |
| `--gait-frequency` | 0.85 Hz | 组间交替频率 |
| `--swing-ratio-amplitude` | 0.26 | 摆动关节摆幅 |
| `--stance-lift-ratio` / `--swing-lift-ratio` | 0.05 / 0.78 | 支撑/摆动相抬腿比例 |
| `--stance-drop-ratio` / `--swing-drop-ratio` | 0.90 / 0.38 | 支撑/摆动相落腿比例 |

**稳定性改进（相比初版）：**

- **按关节类型分级 PD 参数**：drop 关节承力最大 (stiffness=80 / damping=4 / effort=200)，lift 次之 (50 / 2.5 / 120)，swing 最小 (60 / 3 / 150)，均来自 `test_gym.py` 已验证的静态站立配置。
- **Smoothstep 过渡**：用 Hermite 平滑插值取代 `max(sin,0)`，消除支撑↔摆动切换处的力矩突变（一阶导连续）。
- **静止稳定阶段**：步态前执行 `--hold-steps` 步自适应 sag 补偿站立，确保初始姿态稳定。
- **姿态角监控**：headless 模式每 400 步输出 body roll / pitch。

### 8b. 前进方向确认原理（含扭矩贡献评分）

前进方向不能仅由躯干几何或腿的分布决定，因为“朝向”不等于“能产生有效推力的方向”。
本系统采用**两阶段判定**：

**阶段一：PCA 几何初始轴（`initial_virtual_forward_axis`）**

对支撑腿足端 XY 坐标构建加权协方差矩阵，PCA 提取最大特征值对应的特征向量
作为初始候选轴 $\hat{\mathbf{u}}$。这一步提供的是**腿分布的几何对称轴**。

**阶段二：扭矩贡献评分选方向（`final_forward_axis`）**

对 $\pm\hat{\mathbf{u}}$ 两个候选方向分别计算推进力评分 $s_k$：

$$s_k = \sum_{\text{腿 } i} \max\left(\hat{k} \cdot \hat{\mathbf{v}}_i,\ 0\right) \cdot |\mathbf{v}_i| \cdot w_{\text{phase},i} \cdot w_{\text{range},i}$$

各项含义：

| 因子 | 符号 | 物理含义 |
|---|---|---|
| 方向投影 | $\max(\hat{k} \cdot \hat{\mathbf{v}}_i, 0)$ | 仅计入向前挥动的腿，向后的摆动不贡献推进 |
| 摆幅 | $\|\mathbf{v}_i\|$ | 摆动向量模长，对应腿的步幅能力 |
| 相位增益 | $w_{\text{phase},i}$ | 摆动/抬腿/落腿相 ×1.15（正在产生推力），支撑相 ×0.75 |
| 关节扭矩增益 | $w_{\text{range},i} = 1 + \min(\Delta\theta_i, 1.2)$ | $\Delta\theta_i$ 为第 $i$ 条腿的 swing 关节总活动范围（rad）。**范围越大，代表该腿能通过关节力矩产生的水平摆动幅度越大，推进能力越强** |

> **为什么考虑关节扭矩范围？**
> 
> 足端位置分布（阶段一 PCA）只反映了“腿放在哪里”，不反映“腿能往哪个方向用力”。
> 两条足端位置对称的腿，若 swing 关节范围分别为 $\pm 0.55$ rad 和 $\pm 0.12$ rad，
> 它们的有效摆动幅度相差近 5 倍。仅靠几何位置无法区分这一差异，因此必须引入
> **关节活动范围作为扭矩贡献的代理指标**，使评分向“更适合推动前进的腿”倾斜。
> 
> 最终选择得分高的一侧作为 `final_forward_axis`：
> $$\hat{\mathbf{d}} = \arg\max_{k \in \{\pm\hat{\mathbf{u}}\}} s_k$$

### 8c. 分组交替步态（匀速直线运动）

分组策略将有效腿按最终前向轴的投影位置排序后交替分配：

1. 将所有有效腿的足端位置投影到 `final_forward_axis` 上。
2. 按投影坐标从小到大排序（沿轴向从前到后）。
3. 交替分配：第 0, 2, 4, … 条 → `group_a`，第 1, 3, 5, … 条 → `group_b`。

**控制方程：**

设 `group_a` 相位为 $\phi(t) = 2\pi f t$，`group_b` 相位为 $\phi(t) + \pi$。
两组的正弦波相位差 $\pi$，形成交替周期：

$$
\begin{cases}
\text{lift}_i(t) = r_{\text{lift}}^{\text{stance}} + \Delta r_{\text{lift}} \cdot \max(\sin\phi_i(t), 0) \\
\text{drop}_i(t) = r_{\text{drop}}^{\text{stance}} + \Delta r_{\text{drop}} \cdot \max(\sin\phi_i(t), 0) \\
\text{swing}_i(t) = 0.5 + A \cdot d_i \cdot \sin\phi_i(t)
\end{cases}
$$

其中 $d_i = \operatorname{sgn}(\mathbf{f}_i \cdot \hat{\mathbf{d}})$ 为摆动方向符号
（前腿向前挥、后腿向后挥），$A$ 为摆幅参数。

**该分组方式保证**：
- 任意时刻，一组腿处于支撑相（推地），另一组处于摆动相（迈腿）。
- 前腿向前挥动、后腿向后挥动，合力指向 $\hat{\mathbf{d}}$ 方向。
- 无论机器人有 4 / 6 / 8 / 10 条腿，均无需调整控制拓扑。

### 8d. `test_leg_cycle.py` — 步态周期 PDF 报告

**运行环境**：Adaptation（普通 Python，无需 Isaac Gym），依赖 matplotlib。

**功能**：模拟一个完整步态周期（$T = 1 / f$），输出四页 PDF 报告。

```bash
# 使用默认参数生成
python test_leg_cycle.py

# 自定义参数
python test_leg_cycle.py \
  --gait-frequency 0.85 \
  --stance-lift-ratio 0.05 --swing-lift-ratio 0.78 \
  --stance-drop-ratio 0.90 --swing-drop-ratio 0.38 \
  --swing-ratio-amplitude 0.26 \
  --samples 120 \
  --output png/my_gait_report.pdf
```

**PDF 内容布局：**

| 子图 | 内容 |
|---|---|
| 左上 | **Lift 关节目标曲线**：每条腿一条曲线，灰色半透明竖带标记摆动相 |
| 右上 | **Swing 关节目标曲线**：同上布局 |
| 左下 | **Drop 关节目标曲线**：同上布局 |
| 右下 | **相位甘特图**：每条腿一行，绿色 = 支撑相，灰色 = 摆动相，清晰展示 π 相位差交替模式 |
| 底部 | **统计摘要表**：支撑/摆动占比、各关节每腿均值与摆幅范围、参数表 |

**参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gait-frequency` | 0.85 | 步态频率 (Hz) |
| `--stance-lift-ratio` | 0.05 | 支撑相抬腿比例 |
| `--swing-lift-ratio` | 0.78 | 摆动相抬腿比例 |
| `--stance-drop-ratio` | 0.90 | 支撑相落腿比例 |
| `--swing-drop-ratio` | 0.38 | 摆动相落腿比例 |
| `--swing-ratio-amplitude` | 0.26 | 摆动关节摆幅 |
| `--samples` | 120 | 每周期采样点数 |
| `--output` | `png/leg_cycle_report.pdf` | 输出路径 |

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
- `adaptive_gait.py` 新增 `compute_adaptive_plan()`：实现完整阶段一 (PCA 初始轴 + 扭矩评分选方向) 与阶段二 (分组拓扑 + 安全走廊 + 平移代偿)，替代之前缺失的实现。
- 新建 `test_gait.py`：在 Isaac Gym 中验证自适应步态，可视化前进方向箭头 / 支撑多边形 / 质心投影，实现分组交替匀速直线运动，并输出轨迹统计。
- README 新增 8a (`test_gait.py` 使用说明)、8b (前进方向确认原理含关节扭矩增益推导)、8c (分组交替步态控制方程)。
- `test_gait.py` 稳定性增强：按关节类型分级 PD 参数 (沿用 test_gym 验证值)、smoothstep 平滑过渡消除力矩突变、`--hold-steps` 步态前静止稳定阶段、headless 模式姿态角 (roll/pitch) 监控。
- `test_gait.py` 可视化增强：前进箭头加长至 1.5 m、足端按 group_a (蓝) / group_b (红) 颜色标记、支撑/摆动相明暗区分、终端每 200 帧打印前进方向角。
- 新建 `test_leg_cycle.py`：在 Adaptation 环境运行，matplotlib 生成 PDF 步态周期报告 (4 子图 + 甘特图 + 统计摘要)。
- README 新增 8d (`test_leg_cycle.py` 使用说明及 PDF 内容布局)。

## 11. 机器环境备注

如果 `conda run` 触发 `libtinfo` 警告，优先直接使用目标解释器：

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py
```

## 12. 仿真调试备注（Isaac Gym）

在将生成的 URDF 导入 Isaac Gym 并尝试应用关节位置控制（Position Control）时，必须注意以下几点：

1. **缺省驱动模式（`default_dof_drive_mode`）**：
   哪怕随后使用 `set_actor_dof_properties` 将 `driveMode` 填充为 `gymapi.DOF_MODE_POS`，在 `AssetOptions` 初始化时也必须主动配置：
   ```python
   asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
   ```
   若遗漏，下层 PhysX 引擎可能不会创建基于位置的力矩控制器，导致机器人出现 **“完全像没有输出扭矩一样瘫倒”** 的异常。

2. **过载刚度引起数值爆炸翻车（Flipping Over）**：
   将 `test_gym.py` 和 `import_isaac.py` 中的刚度（Stiffness）设为高达 `1800~6500` 时，即使对于静态维持依然具有严重的潜在数值不稳定性。在默认 `dt=1/60s` 的设定下，由于机器人质量较轻，过高的刚合阻尼会产生极大角加速度并导致向外发散或相互穿模反弹，表现为 **部分机器人剧烈抖动或直接空翻**。正确的比例约在 `stiffness=40~80`、`damping=2~5`（对应力矩缩放后）。

## 13. 步态控制调试备注（test_gait.py & test_standard_gait.py）

### 13.1 摆腿方向符号错误（Dir Sign Bug）

`test_gait.py` 和 `test_standard_gait.py` 早期版本中，计算摆腿方向符号（`dir_sign` / `ds`）时错误地使用了脚端位置与**前进轴**的点积（纵向，X分量），正确做法是与**侧向轴**（垂直于前进方向）做点积：

```python
# 错误写法（已废弃）：
dir_sign = 1.0 if dot(foot_xy, forward_axis) >= 0.0 else -1.0

# 正确写法：
lateral_axis = np.array([-forward_axis[1], forward_axis[0]])  # 前进方向左侧90°
lateral_pos = dot(foot_xy, lateral_axis)
dir_sign = -1.0 if lateral_pos > 0.0 else 1.0
```

**物理原因**：摆腿关节（swing）绕 Z 轴旋转，对于侧向挂腿的机器人：
- 身体左侧（+Y）的腿：关节**负向**旋转才会使脚端向前（+X）运动 → `dir_sign = -1`
- 身体右侧（-Y）的腿：关节**正向**旋转才会使脚端向前（+X）运动 → `dir_sign = +1`

原来使用前进轴点积会导致靠前的腿（foot_x > 0）和靠后的腿（foot_x < 0）获得相反的符号，使前半部分的腿产生的摆动力方向完全相反，机器人原地乱晃甚至无法前进。

### 13.2 smoothstep 过渡窗口过窄（急停抖动问题）

早期使用 `smoothstep(-0.05, 0.05, sin(phase))` 的过渡区间非常窄，sin 值从 -0.05 变化到 0.05 大约只占步态周期的 **1%**（在 60 Hz 下约 1 个仿真步），导致站立→摆腿切换几乎是瞬间完成的，身体会产生剧烈冲击并倾倒。

修复后使用 `smoothstep(-0.30, 0.30, sin(phase))`，过渡区间约占步态周期的 **25%**（约 13 个仿真步），切换平滑，动态稳定性显著改善。

### 13.3 关键 Bug：髋关节球体与机体碰撞体重叠（`collapse_fixed_joints` 必须为 True）

在由 `test_standard_gait.py` 生成的标准六足 URDF 中，每条腿的髋关节球（`leg_N_hip`，半径 0.035m）通过固定关节挂在机体侧面，但因为 `attach_clearance` 仅约 2.5cm，球面会向内突入机体碰撞网格 **0.9–2.2cm**。

当 `collapse_fixed_joints = False` 时，Isaac Gym 将这些髋球作为独立物理体处理，它们与机体之间产生持续碰撞。后续每当关节电机尝试转动腿部，腿部立刻顶到机体，关节力矩无法驱动腿运动，反力全部传递给机体 → **腿不动、机体移动**。

**正确设置**（两个脚本均已修复）：

```python
asset_options.collapse_fixed_joints = True   # 将所有固定关节子链合并进父体
```

通过合并：
- `base_link` + 所有 `leg_N_hip` 球 → 一个复合体（不再有髋球-机体内部碰撞）
- `leg_N_upper` + `leg_N_knee` → 每腿一个复合上臂体
- `leg_N_lower` + `leg_N_foot` → 每腿一个复合下臂体

活动自由度（revolute joints）完全不受影响，DOF 名称/索引不变。

### 13.4 PD 增益需与机体重量匹配

机体 9.7 kg，3 条支撑腿分担约 32 N/腿。提升位关节到 lift 轴的力矩臂约 0.3m → 每关节需承受 ~10 N·m。

旧增益（stiffness=50 N·m/rad）导致 ~0.2 rad（11°）的角度下沉，视觉上是"机器人缓慢坐到地上"。正确最小增益约 200 N·m/rad，对应 ~3° 内的下沉量。
