import pybullet as p
import pybullet_data
import time
import getpass
import math

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("robot_assets/standard_hexapod/generated_robot.urdf", [0, 0, 0.46], useFixedBase=False)

foot_links = []
movable_joints = []
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    if info[2] != p.JOINT_FIXED:
        movable_joints.append(i)
    if 'foot' in info[1].decode('utf-8') and 'mount' not in info[1].decode('utf-8'):
        foot_links.append(i)

if not foot_links:
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if 'drop' in info[1].decode('utf-8'):
            foot_links.append(i)

# initial angles
for i in movable_joints:
    p.resetJointState(robot_id, i, 0)
    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 0, force=180)

# step physics
for step in range(1200):
    p.stepSimulation()
    # Apply IK
    end_eff_targets = []
    for link in foot_links:
        state = p.getLinkState(robot_id, link)
        pos = state[0]
        end_eff_targets.append((pos[0], pos[1], 0.04)) # Target Z=0.04 or 0
    
    ik_angles = p.calculateInverseKinematics2(robot_id, foot_links, end_eff_targets, maxNumIterations=10)
    for i, idx in enumerate(movable_joints):
        p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, targetPosition=ik_angles[i], force=180, positionGain=0.1, velocityGain=0.5)

pos = p.getBasePositionAndOrientation(robot_id)[0]
print("Final base pos:", pos)

