with open('test_standard_static.py', 'r') as f:
    code = f.read()

import re

# We simply replace the whole step simulation block
old_loop = """
    roll_samples: List[float] = []
    pitch_samples: List[float] = []
    base_z_samples: List[float] = []
    contact_samples: List[int] = []

    print(f"[HOLD] Stabilizing for {args.hold_steps} steps ...")
    for step in range(max(args.steps, 1)):
        apply_joint_targets(robot_id, joint_targets, args.stiffness, args.damping, args.effort)
        p.stepSimulation()"""

new_loop = """
    import numpy as np

    end_eff_targets = []
    if args.use_ik:
        for link in joint_data['foot_links']:
            pos = p.getLinkState(robot_id, link)[0]
            end_eff_targets.append([pos[0], pos[1], 0.0])

    roll_samples: List[float] = []
    pitch_samples: List[float] = []
    base_z_samples: List[float] = []
    contact_samples: List[int] = []

    print(f"[HOLD] Stabilizing for {args.hold_steps} steps ...")
    for step in range(max(args.steps, 1)):
        if args.use_ik and step > 10:
            base_pos = p.getBasePositionAndOrientation(robot_id)[0]
            z_err = float(args.body_height) - base_pos[2]
            for idx, link in enumerate(joint_data['foot_links']):
                cur_pos = p.getLinkState(robot_id, link)[0]
                end_eff_targets[idx][0] = cur_pos[0]
                end_eff_targets[idx][1] = cur_pos[1]
                end_eff_targets[idx][2] = np.clip(-z_err * 0.5, -0.05, 0.05)
                
            ik_angles = p.calculateInverseKinematics2(
                robot_id, joint_data['foot_links'], end_eff_targets,
                lowerLimits=joint_data['lower'],
                upperLimits=joint_data['upper'],
                jointRanges=joint_data['ranges'],
                restPoses=joint_data['rest'],
                maxNumIterations=20,
                residualThreshold=1e-4
            )
            for i, m in enumerate(joint_data['movable_joints']):
                joint_targets[m] = float(ik_angles[i])
       
        apply_joint_targets(robot_id, joint_targets, args.stiffness, args.damping, args.effort)
        p.stepSimulation()"""

if old_loop in code:
    code = code.replace(old_loop, new_loop)
    with open('test_standard_static.py', 'w') as f:
        f.write(code)
    print("Patched!")
else:
    print("Old loop not found.")

