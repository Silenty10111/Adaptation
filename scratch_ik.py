import pybullet as p
import pybullet_data
import time

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("robot_assets/standard_hexapod/generated_robot.urdf", [0, 0, 0.46], useFixedBase=True)

# Find foot candidate links and movable joints
foot_links = []
movable_joints = []
for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    name = info[1].decode('utf-8')
    if info[2] != p.JOINT_FIXED:
        movable_joints.append(i)
    if 'drop' in name:
        foot_links.append(i)

print("Movable:", movable_joints)
print("Foot links:", foot_links)

# Get current foot positions
end_eff_targets = []
for link in foot_links:
    pos = p.getLinkState(robot_id, link)[0]
    end_eff_targets.append((pos[0], pos[1], 0.0)) # Flat on the ground

ik_angles = p.calculateInverseKinematics2(robot_id, foot_links, end_eff_targets)
print("IK Angles mapping:")
joint_targets = {}
for i, idx in enumerate(movable_joints):
    # ik_angles contains exactly the movable joints in order
    joint_targets[idx] = ik_angles[i]
print(joint_targets)
