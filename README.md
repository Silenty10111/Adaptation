# Adaptation

Utilities for generating multi-legged robot assets and testing them in PyBullet / Isaac Gym.

## 1. Prerequisites (new machine)

- Linux (Ubuntu recommended)
- NVIDIA GPU + working driver (for Isaac Gym GPU workflows)
- Conda (Miniconda/Anaconda)
- Git

## 2. Clone repositories

```bash
git clone <your-adaptation-repo-url> Adaptation
cd Adaptation
```

Isaac Gym should also be available locally (example path below):

```bash
git clone <your-isaacgym-repo-url> /data/code/yjh/isaacgym
```

## 3. Create and activate environment for Isaac Gym

Isaac Gym bindings in this project are verified with Python 3.8.

```bash
conda create -n unitree-rl python=3.8 -y
conda activate unitree-rl
```

## 4. Install project dependencies

```bash
pip install -r requirements.txt
```

## 5. Install Isaac Gym Python package

From your Isaac Gym folder:

```bash
cd /data/code/yjh/isaacgym/python
pip install -e .
```

If your Isaac Gym path is different, replace it in all commands.

## 6. Environment variables

For this machine layout, use:

```bash
export PYTHONPATH=/data/code/yjh/isaacgym/python:${PYTHONPATH}
export LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib:${LD_LIBRARY_PATH}
```

You can put these lines into your shell rc file.

## 7. Generate robot assets

Back in this repository root:

```bash
cd /home/robot/code/yjh/Adaptation
python generate_geometry.py
python generate_urdf.py
```

This produces `robot_assets/generated_robot.urdf`.

## 8. Test in Isaac Gym

Viewer mode (default):

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py
```

Headless mode:

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py --headless --steps 120
```

Expected successful output includes:

- `[OK] Asset loaded. DOFs=...`
- `[OK] Isaac Gym simulation stepped successfully.`

## 9. Optional: avoid conda run warning

If `conda run` prints a `libtinfo` warning, prefer direct interpreter invocation:

```bash
LD_LIBRARY_PATH=/data/conda/envs/unitree-rl/lib /data/conda/envs/unitree-rl/bin/python test_gym.py
```
