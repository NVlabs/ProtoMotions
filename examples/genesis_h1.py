import torch
import platform
import genesis as gs
from protomotions.simulator.genesis.config import GenesisSimulatorConfig, GenesisSimParams
from protomotions.simulator.genesis.simulator import GenesisSimulator
from protomotions.simulator.base_simulator.config import (
    RobotConfig,
    RobotAssetConfig,
    InitState,
    ControlConfig,
    ControlType,
)
from protomotions.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig

# Create robot asset configuration
robot_asset_config = RobotAssetConfig(
    robot_type="h1",
    asset_file_name="urdf/h1.urdf",
    self_collisions=False,
    collapse_fixed_joints=False,
)

# Create robot configuration
robot_config = RobotConfig(
    body_names=['pelvis', 'head', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'left_foot_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'right_foot_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_arm_end_effector', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_arm_end_effector'],
    dof_names=['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'],
    dof_body_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    joint_axis=['z', 'x', 'y', 'y', 'y', 'z', 'x', 'y', 'y', 'y', 'z', 'y', 'x', 'z', 'y', 'y', 'x', 'z', 'y'],
    dof_obs_size=114,  # 19 joints * 6 (pos, vel, etc.)
    number_of_actions=19,
    self_obs_max_coords_size=373,
    left_foot_name="left_foot_link",
    right_foot_name="right_foot_link",
    head_body_name="head",
    key_bodies=[ "left_foot_link", "right_foot_link", "left_arm_end_effector",  "right_arm_end_effector" ],
    non_termination_contact_bodies=[ "left_foot_link", "left_ankle_link", "right_foot_link", "right_ankle_link" ],
    asset=robot_asset_config,
    init_state=InitState(
        pos=[0.0, 0.0, 1.0],
        default_joint_angles={
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.4,
            "left_knee_joint": 0.8,
            "left_ankle_joint": -0.4,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.4,
            "right_knee_joint": 0.8,
            "right_ankle_joint": -0.4,
            "torso_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "head": 0.0,
            "left_arm_end_effector_joint": 0.0,
            "right_arm_end_effector_joint": 0.0,
            "left_foot_link": 0.0,
            "right_foot_link": 0.0,
        },
    ),
    control=ControlConfig(
        control_type=ControlType.PROPORTIONAL,
        action_scale=1.0,
        clamp_actions=100.0,
        stiffness={
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 300,
            'ankle': 40,
            'torso': 300,
            'shoulder': 100,
            'elbow': 100,
        },
        damping={
            'hip_yaw': 5,
            'hip_roll': 5,
            'hip_pitch': 5,
            'knee': 6,
            'ankle': 2,
            'torso': 6,
            'shoulder': 2,
            'elbow': 2,
        },
    )
)

# Create simulator configuration
simulator_config = GenesisSimulatorConfig(
    sim=GenesisSimParams(
        fps=200,
        decimation=4,
        substeps=1,
    ),
    headless=False,  # Set to True for headless mode
    robot=robot_config,
    num_envs=4,  # Number of parallel environments
    experiment_name="h1_genesis_example",
    w_last=False,  # Genesis uses wxyz quaternions
)

if platform.system()=='Darwin':
    device = torch.device("mps")
else:
    device = torch.device("cuda")

# Create a flat terrain using the default config
terrain_config = TerrainConfig()
terrain = FlatTerrain(config=terrain_config, num_envs=simulator_config.num_envs, device=device)

# Create and initialize the simulator
simulator = GenesisSimulator(config=simulator_config, terrain=terrain, scene_lib=None, visualization_markers=None, device=device)
simulator.on_environment_ready()

# Get robot default state
default_state = simulator.get_default_state()
# Set the robot to a new random position above the ground
root_pos = 2 * torch.rand(simulator_config.num_envs, 3, device=device) + 10
# We don't want to spawn at [0,0] since the terrain corner is at [0,0]
# For more robust spawning, use terrain.sample_valid_locations(num_envs)
root_pos[:, 2] = 1.0
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(default_state, env_ids=torch.arange(simulator_config.num_envs, device=device))

# Run the simulation loop
if platform.system() == "Darwin":
    def run_sim(simulator, sim_config, device):
        while True:
            actions = torch.randn(sim_config.num_envs, sim_config.robot.number_of_actions, device=device)
            simulator.step(actions)

    gs.tools.run_in_another_thread(fn=run_sim, args=(simulator, simulator_config, device))
    if True:
        simulator._scene.viewer.start()
else:
    try:
        while True:
            actions = torch.randn(simulator_config.num_envs, simulator_config.robot.number_of_actions, device=device)
            simulator.step(actions)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        simulator.close()
