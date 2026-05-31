import pybullet as p
import pybullet_data
import time
import math
import numpy as np

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("robot_assets/standard_hexapod/generated_robot.urdf", [0, 0, 0.46], useFixedBase=False)

foot_links = []
movable_joints = []
lower_limits = []
upper_limits = []
joint_ranges = []
rest_poses = []
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode("utf-8")
    if info[2] != p.JOINT_FIXED:
        movable_joints.append(i)
        lower_limits.append(float(info[8]) if math.isfinite(info[8]) else -1.0)
        upper_limits.append(float(info[9]) if math.isfinite(info[9]) else 1.0)
        joint_ranges.append(abs(upper_limits[-1] - lower_limits[-1]))
        if "_lift" in name: t = lower_limits[-1] + 0.46 * joint_ranges[-1]
        elif "_drop" in name: t = lower_limits[-1] + 0.08 * joint_ranges[-1]
        elif "_swing" in name: t = lower_limits[-1] + 0.5 * joint_ranges[-1]
        else: t = 0.5 * (lower_limits[-1] + upper_limits[-1])
        rest_poses.append(t)
        p.resetJointState(robot_id, i, t)
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, t, force=180)
    if 'drop' in name:
        foot_links.append(i)

end_eff_targets = []
for link in foot_links:
    pos = p.getLinkState(robot_id, link)[0]
    end_eff_targets.append([pos[0], pos[1], 0.0]) # target ground Z=0
for step in range(1200):
    p.stepSimulation()
    # update X and Y targets to keep them where they are
    # but adjust Z based on base position error
    base_pos = p.getBasePositionAndOrientation(robot_id)[0]
    error_z = 0.46 - base_pos[2]
    # We want base to go up by error_z, so feet must be driven DOWN by error_z
    for idx, link in enumerate(foot_links):
        cur_pos = p.getLinkState(robot_id, link)[0]
        end_eff_targets[idx][0] = cur_pos[0] # stay under current X
        end_eff_targets[idx][1] = cur_pos[1] # stay under current Y
        end_eff_targets[idx][2] = 0.0 - error_z * 0.5 # push proportional down

    ik_angles = p.calculateInverseKinematics2(
        robot_id, foot_links, end_eff_targets,
        lowerLimits=lower_limits, upperLimits=upper_limits,
        jointRanges=joint_ranges, restPoses=rest_poses
    )
    for i, idx in enumerate(movable_joints):
        p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=ik_angles[i], force=180, positionGain=0.1)

print("Final base pos:", p.getBasePositionAndOrientation(robot_id)[0])
print("Max final foot Z:", max([p.getLinkState(robot_id, l)[0][2] for l in foot_links]))
print("Min final foot Z:", min([p.getLinkState(robot_id, l)[0][2] for l in foot_links]))
