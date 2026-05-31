import pybullet as p
import pybullet_data
import time
import math
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("robot_assets/standard_hexapod/generated_robot.urdf", [0, 0, 0.46], useFixedBase=True)

# build naive targets
stance_lift_ratio = 0.46
stance_drop_ratio = 0.08
swing_ratio = 0.5

joint_targets = {}
foot_links = []
movable_joints = []
for joint_index in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, joint_index)
    name = info[1].decode("utf-8")
    lower = float(info[8]) if math.isfinite(info[8]) else -0.5
    upper = float(info[9]) if math.isfinite(info[9]) else 0.5
    
    if info[2] != p.JOINT_FIXED:
        movable_joints.append(joint_index)
        
    if 'drop' in name:
        foot_links.append(joint_index)
        
    if upper <= lower:
        target = 0.0
    elif "_lift" in name:
        target = lower + np.clip(stance_lift_ratio, 0.0, 1.0) * (upper - lower)
    elif "_drop" in name:
        target = lower + np.clip(stance_drop_ratio, 0.0, 1.0) * (upper - lower)
    elif "_swing" in name:
        target = lower + np.clip(swing_ratio, 0.0, 1.0) * (upper - lower)
    else:
        target = 0.5 * (lower + upper)
    p.resetJointState(robot_id, joint_index, target)
    joint_targets[joint_index] = float(target)

end_eff_targets = []
for link in foot_links:
    pos = p.getLinkState(robot_id, link)[0]
    # target Z = 0
    end_eff_targets.append((pos[0], pos[1], 0.0))

print("IK on:", foot_links)
ik_angles = p.calculateInverseKinematics2(robot_id, foot_links, end_eff_targets, maxNumIterations=100)

for i, m in enumerate(movable_joints):
    print(f"Joint {m}: naive={joint_targets[m]:.3f}, ik={ik_angles[i]:.3f}")

