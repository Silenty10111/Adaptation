import pybullet as p
import pybullet_data
import time
import math
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("robot_assets/standard_hexapod/generated_robot.urdf", [0, 0, 0.46], useFixedBase=True)

# Build limits array and rest poses
lower_limits = []
upper_limits = []
joint_ranges = []
rest_poses = []

foot_links = []
movable_joints = []
for joint_index in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, joint_index)
    name = info[1].decode("utf-8")
    
    if info[2] != p.JOINT_FIXED:
        movable_joints.append(joint_index)
        lower = float(info[8]) if math.isfinite(info[8]) else -1.0
        upper = float(info[9]) if math.isfinite(info[9]) else 1.0
        
        lower_limits.append(lower)
        upper_limits.append(upper)
        joint_ranges.append(abs(upper - lower))
        
        # Determine rest pose based on the previous naive math
        if "_lift" in name:
            target = lower + 0.46 * (upper - lower)
        elif "_drop" in name:
            target = lower + 0.08 * (upper - lower)
        elif "_swing" in name:
            target = lower + 0.5 * (upper - lower)
        else:
            target = 0.5 * (lower + upper)
        rest_poses.append(target)
        p.resetJointState(robot_id, joint_index, target)
        
    if 'drop' in name:
        foot_links.append(joint_index)

end_eff_targets = []
for link in foot_links:
    pos = p.getLinkState(robot_id, link)[0]
    end_eff_targets.append((pos[0], pos[1], 0.0))

print("IK on foot_links:", foot_links)
ik_angles = p.calculateInverseKinematics2(
    robot_id, foot_links, end_eff_targets,
    lowerLimits=lower_limits,
    upperLimits=upper_limits,
    jointRanges=joint_ranges,
    restPoses=rest_poses,
    maxNumIterations=1000,
    residualThreshold=1e-5
)

for i, m in enumerate(movable_joints):
    print(f"Joint {m}: naive={rest_poses[i]:.3f}, ik={ik_angles[i]:.3f}")

