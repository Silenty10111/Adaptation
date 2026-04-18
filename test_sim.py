import pybullet as p
import pybullet_data
import time

# 1. 启动 GUI 并设置路径
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 用于加载内置的平面

# 2. 设置重力和加载地面
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf") # 【修复】加载坚硬的地面！

# 3. 加载你的机器人 (稍微放高一点，防止一上来就卡进地里)
startPos = [0, 0, 0.5] 
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF("robot_assets/generated_robot.urdf", startPos, startOrientation)

# 4. 获取关节数量，并施加“站立”的控制力矩
num_joints = p.getNumJoints(robot_id)
for j in range(num_joints):
    # 将所有关节锁定在初始位置 (0 弧度)，提供极大的力量让它保持刚硬
    p.setJointMotorControl2(bodyIndex=robot_id,
                            jointIndex=j,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=0.0,
                            force=50.0) # 提供 50N·m 的保持力

print("开始仿真！机器人应该稳稳站在地上了。")

# 5. 持续仿真
while True:
    p.stepSimulation()
    time.sleep(1./240.)