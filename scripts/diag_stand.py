"""Diagnose standing adaptation.stability and foot contact heights.

Usage:
  python scripts/diag_stand.py \\
      --description batch_results/20260712_162341/robot_00_seed7/robot_description.json \\
      --urdf batch_results/20260712_162341/robot_00_seed7/robot.urdf

  # Use defaults (standard hexapod)
  python scripts/diag_stand.py
"""
import sys, math, argparse, os, numpy as np
from pathlib import Path

# --- auto-switch to unitree-rl environment for Isaac Gym ---
TARGET_PYTHON = "/data/conda/envs/unitree-rl/bin/python"
TARGET_LD_PATH = "/data/conda/envs/unitree-rl/lib"

if os.environ.get("DIAG_STAND_REEXEC") != "1" and sys.executable != TARGET_PYTHON and Path(TARGET_PYTHON).exists():
    env = dict(os.environ)
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{TARGET_LD_PATH}:{ld_path}" if ld_path else TARGET_LD_PATH
    project_root = str(Path(__file__).resolve().parent.parent)
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{py_path}" if py_path else project_root
    env["DIAG_STAND_REEXEC"] = "1"
    print("[INFO] Auto-switching to unitree-rl environment.")
    os.execvpe(TARGET_PYTHON, [TARGET_PYTHON, *sys.argv], env)

# --- parse user arguments before importing test_gait ---
parser = argparse.ArgumentParser(
    description="Standing stability diagnosis in Isaac Gym",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""Examples:
  python diag_stand.py
  python diag_stand.py --description robot_00/robot_description.json --urdf robot_00/robot.urdf
  python diag_stand.py --desc-dir batch_results/20260712_162341/robot_00_seed7""",
)
parser.add_argument("--description", type=str, default=None,
                    help="Path to robot_description.json")
parser.add_argument("--urdf", type=str, default=None,
                    help="Path to robot.urdf")
parser.add_argument("--desc-dir", type=str, default=None,
                    help="Directory containing robot_description.json and robot.urdf (shortcut)")
args = parser.parse_args()

# Resolve --desc-dir shortcut
if args.desc_dir is not None:
    if args.description is None:
        args.description = f"{args.desc_dir}/robot_description.json"
    if args.urdf is None:
        args.urdf = f"{args.desc_dir}/robot.urdf"

# Ensure isaacgym is findable
for _p in ['/data/code/yjh/isaacgym/python', '/home/robot/IsaacGym/python']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Protect test_gait's argparse from seeing user args
_saved_argv = sys.argv
sys.argv = ['test_gait.py', '--headless']
import test_gait as tg
from isaacgym import gymapi
sys.argv = _saved_argv

desc_path, urdf_path = tg.resolve_asset_paths(
    Path(args.description) if args.description else Path(""),
    Path(args.urdf) if args.urdf else Path(""),
)
description = tg.load_description(desc_path)
gait_plan = tg.compute_plan(description, {})

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.up_axis = gymapi.UP_AXIS_Z; sp.gravity = gymapi.Vec3(0,0,-9.81)
sp.dt = 1/60.0; sp.substeps = 2; sp.physx.use_gpu = False
sim = gym.create_sim(0,-1,gymapi.SIM_PHYSX,sp)
pln = gymapi.PlaneParams(); pln.normal = gymapi.Vec3(0,0,1)
pln.static_friction=1.8; pln.dynamic_friction=1.6
gym.add_ground(sim, pln)

ao = gymapi.AssetOptions()
ao.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
ao.fix_base_link = False; ao.collapse_fixed_joints = True
asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, ao)
env = gym.create_env(sim, gymapi.Vec3(-3,-3,0), gymapi.Vec3(3,3,2), 1)
pose = gymapi.Transform(); pose.p = gymapi.Vec3(0,0,0.42)
actor = gym.create_actor(env, asset, pose, 'bot', 0, 1)

dp = gym.get_actor_dof_properties(env, actor)
dp['driveMode'].fill(gymapi.DOF_MODE_POS)
dp['stiffness'].fill(200); dp['damping'].fill(20)
gym.set_actor_dof_properties(env, actor, dp)
triplets = tg.resolve_joint_triplets(gym, env, actor, description)

lower = np.asarray(dp['lower'], dtype=np.float32)
upper = np.asarray(dp['upper'], dtype=np.float32)
st = 0.5*(lower+upper)
for lid, j in triplets.items():
    st[j['lift_idx']] = tg.ratio_to_joint(j['lift_lower'], j['lift_upper'], 0.457)
    st[j['drop_idx']] = tg.ratio_to_joint(j['drop_lower'], j['drop_upper'], 0.083)
    st[j['swing_idx']] = tg.ratio_to_joint(j['swing_lower'], j['swing_upper'], 0.5)

ds = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
ds['pos'][:] = st; ds['vel'][:] = 0
gym.set_actor_dof_states(env, actor, ds, gymapi.STATE_ALL)

print('=== STANDING STABILITY DIAGNOSIS ===')
# Simulate 300 steps with stand targets
for step in range(300):
    gym.set_actor_dof_position_targets(env, actor, st)
    gym.simulate(sim); gym.fetch_results(sim, True)
    if (step+1) % 50 == 0:
        bs = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
        p0 = bs['pose']['p'][0]; r0 = bs['pose']['r'][0]
        roll,pitch,_ = tg.quat_to_euler(float(r0['w']),float(r0['x']),float(r0['y']),float(r0['z']))
        print(f'  step {step+1:3d}: trunk_z={float(p0["z"]):.4f} xy=({float(p0["x"]):.3f},{float(p0["y"]):.3f}) roll={math.degrees(roll):.1f}°')

bs = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
bn = gym.get_actor_rigid_body_names(env, actor)
print('\nAll body z-positions:')
for i,nm in enumerate(bn):
    z = float(bs['pose']['p'][i]['z'])
    x = float(bs['pose']['p'][i]['x'])
    y = float(bs['pose']['p'][i]['y'])
    print(f'  [{i:2d}] {nm:30s} z={z:.4f} xy=({x:.3f},{y:.3f})')

gym.destroy_sim(sim)
