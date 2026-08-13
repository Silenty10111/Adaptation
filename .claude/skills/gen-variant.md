---
name: gen-variant
description: Generate custom hexapod leg-amputation variants — modify robot_description.json, regenerate URDF, and optionally run a quick validation test.
triggers:
  - "生成变体" "generate variant" "创建变体" "移除腿" "custom robot" "gen-variant"
  - "/gen-variant"
---

# Gen Variant — 生成自定义缺腿变体

交互式生成任意缺腿组合的六足机器人变体，含 URDF 生成和快速验证。

## 使用方式

```
/gen-variant --remove 0                    # 移除腿 0，生成 5 腿变体
/gen-variant --remove 0,3                  # 移除腿 0 和腿 3，生成 4 腿变体
/gen-variant --remove 0,1,2                # 移除 3 条腿，生成 3 腿变体
/gen-variant --remove 1 --name my_test     # 自定义变体名称
/gen-variant --remove 0 --quick-test       # 生成后运行快速仿真验证
/gen-variant --list                        # 列出标准六足的所有腿位置
```

## 腿编号参考

```
        ┌─────────── 标准六足 trunk (0.55m × 0.36m) ───────────┐
        │                                                       │
  Leg 5 (左前)  ●───────────────┬───────────────●  Leg 0 (右前) │
        │       │               │               │       │       │
  Leg 4 (左中)  ●───────────────┼───────────────●  Leg 1 (右中) │
        │       │               │               │       │       │
  Leg 3 (左后)  ●───────────────┴───────────────●  Leg 2 (右后) │
        │                                                       │
        └───────────────────────────────────────────────────────┘
                         ←── +X (前进方向) ──→
```

## 实现步骤

1. 加载 `robot_assets/standard_hexapod/robot_description.json`
2. 用 `amputate_legs()` 移除指定腿，重新编号（leg_id 保持 0..N-1 连续）
3. 复制对应的 STL 网格文件
4. 调用 `scripts/generate_urdf.py` 生成 URDF
5. (可选) 用 `_RobotSimCtx` 运行快速仿真验证

## 关键函数

`amputate_legs(description, remove_leg_ids) -> dict`:
- 输入: 原始 robot_description + 要移除的 leg_id 列表
- 输出: 新的 robot_description（leg_id 重新编号、link/joint 名称更新）
- mesh_path 保持原始文件名（STL 文件不需要重命名）

`copy_meshes_for_variant(description, dst_dir)`:
- 从 standard_hexapod/meshes 复制所需的 STL 文件

## 注意事项

- 标准六足有 6 条腿，每条腿 6 个 links + 6 个 joints
- 移除 N 条腿后，num_legs = 6-N，links = 1+6*(6-N)，joints = 6*(6-N)
- 变体 URDF 必须在 Isaac Gym 中可加载（load_asset 成功）
- SSM 随腿数减少而降低，4 腿时可能 < 0.05m 阈值
