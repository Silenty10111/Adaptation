with open('test_standard_static.py', 'r') as f:
    code = f.read()

code = code.replace(
'''    joint_targets = build_joint_target_map(
        robot_id,
        stance_lift_ratio=args.stance_lift_ratio,
        stance_drop_ratio=args.stance_drop_ratio,
        swing_ratio=args.swing_ratio,
        use_ik=args.use_ik,
    )''',
'''    joint_targets, joint_data = build_joint_target_map(
        robot_id,
        stance_lift_ratio=args.stance_lift_ratio,
        stance_drop_ratio=args.stance_drop_ratio,
        swing_ratio=args.swing_ratio,
    )'''
)

with open('test_standard_static.py', 'w') as f:
    f.write(code)

