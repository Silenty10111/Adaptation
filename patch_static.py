import re

with open('test_standard_static.py', 'r') as f:
    code = f.read()

# Replace the signature change
code = code.replace(
    'def build_joint_target_map(',
    'def build_joint_target_map('
)
# I already replaced the signature before in Python! I should just replace the main loop now

code = code.replace(
'''    joint_targets = build_joint_target_map(
        robot_id,
        stance_lift_ratio=args.stance_lift_ratio,
        stance_drop_ratio=args.stance_drop_ratio,
        swing_ratio=args.swing_ratio,
        use_ik=args.use_ik,
    )
    
    # Pre-snap joints mathematically to avoid violent initial drop
    for j_i, tgt in joint_targets.items():
        p.resetJointState(robot_id, j_i, tgt)

    apply_joint_targets(robot_id, joint_targets, args.stiffness, args.damping, args.effort)

    roll_samples: List[float] = []
    pitch_samples: List[float] = []
    base_z_samples: List[float] = []
    contact_samples: List[int] = []

    print(f"[HOLD] Stabilizing for {args.hold_steps} steps ...")
    for step in range(max(args.steps, 1)):
        apply_joint_targets(robot_id, joint_targets, args.stiffness, args.damping, args.effort)
        p.stepSimulation()''',
'''    joint_targets, joint_data = build_joint_target_map(
        robot_id,
        stance_lift_ratio=args.stance_lift_ratio,
        stance_drop_ratio=args.stance_drop_ratio,
        swing_ratio=args.swing_ratio,
    )
    
    # Pre-snap joints mathematically to avoid violent initial drop
    for j_i, tgt in joint_targets.items():
        p.resetJointState(robot_id, j_i, tgt)

    # Active IK target storage
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
        if args.use_ik and step >= 10:
            base_pos = p.getBasePositionAndOrientation(robot_id)[0]
            # Closed-loop stabilization: push feet up/down to actively correct base error
            z_err = float(args.body_height) - base_pos[2]
            for idx, link in enumerate(joint_data['foot_links']):
                cur_pos = p.getLinkState(robot_id, link)[0]
                end_eff_targets[idx][0] = cur_pos[0]
                end_eff_targets[idx][1] = cur_pos[1]
                end_eff_targets[idx][2] = np.clip(-z_err * 0.8, -0.05, 0.05)
                
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
        p.stepSimulation()'''
)

with open('test_standard_static.py', 'w') as f:
    f.write(code)

