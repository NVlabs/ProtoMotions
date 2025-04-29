"""
    IsaacGym has to be imported before any torch modules.
"""
import isaacgym

import torch
from protomotions.simulator.isaacgym.config import IsaacGymSimulatorConfig, IsaacGymSimParams
from protomotions.simulator.isaacgym.simulator import IsaacGymSimulator
from protomotions.simulator.base_simulator.config import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
)
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig

# Create robot asset configuration
robot_asset_config = RobotAssetConfig(
    robot_type="smplx",
    asset_file_name="mjcf/smplx_humanoid.xml",
    self_collisions=False,
    collapse_fixed_joints=False,
)

# Create robot configuration
robot_config = RobotConfig(
    body_names=['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3'],
    dof_names=['L_Hip_x', 'L_Hip_y', 'L_Hip_z', 'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 'Torso_x', 'Torso_y', 'Torso_z', 'Spine_x', 'Spine_y', 'Spine_z', 'Chest_x', 'Chest_y', 'Chest_z', 'Neck_x', 'Neck_y', 'Neck_z', 'Head_x', 'Head_y', 'Head_z', 'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 'L_Index1_x', 'L_Index1_y', 'L_Index1_z', 'L_Index2_x', 'L_Index2_y', 'L_Index2_z', 'L_Index3_x', 'L_Index3_y', 'L_Index3_z', 'L_Middle1_x', 'L_Middle1_y', 'L_Middle1_z', 'L_Middle2_x', 'L_Middle2_y', 'L_Middle2_z', 'L_Middle3_x', 'L_Middle3_y', 'L_Middle3_z', 'L_Pinky1_x', 'L_Pinky1_y', 'L_Pinky1_z', 'L_Pinky2_x', 'L_Pinky2_y', 'L_Pinky2_z', 'L_Pinky3_x', 'L_Pinky3_y', 'L_Pinky3_z', 'L_Ring1_x', 'L_Ring1_y', 'L_Ring1_z', 'L_Ring2_x', 'L_Ring2_y', 'L_Ring2_z', 'L_Ring3_x', 'L_Ring3_y', 'L_Ring3_z', 'L_Thumb1_x', 'L_Thumb1_y', 'L_Thumb1_z', 'L_Thumb2_x', 'L_Thumb2_y', 'L_Thumb2_z', 'L_Thumb3_x', 'L_Thumb3_y', 'L_Thumb3_z', 'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', 'R_Index1_x', 'R_Index1_y', 'R_Index1_z', 'R_Index2_x', 'R_Index2_y', 'R_Index2_z', 'R_Index3_x', 'R_Index3_y', 'R_Index3_z', 'R_Middle1_x', 'R_Middle1_y', 'R_Middle1_z', 'R_Middle2_x', 'R_Middle2_y', 'R_Middle2_z', 'R_Middle3_x', 'R_Middle3_y', 'R_Middle3_z', 'R_Pinky1_x', 'R_Pinky1_y', 'R_Pinky1_z', 'R_Pinky2_x', 'R_Pinky2_y', 'R_Pinky2_z', 'R_Pinky3_x', 'R_Pinky3_y', 'R_Pinky3_z', 'R_Ring1_x', 'R_Ring1_y', 'R_Ring1_z', 'R_Ring2_x', 'R_Ring2_y', 'R_Ring2_z', 'R_Ring3_x', 'R_Ring3_y', 'R_Ring3_z', 'R_Thumb1_x', 'R_Thumb1_y', 'R_Thumb1_z', 'R_Thumb2_x', 'R_Thumb2_y', 'R_Thumb2_z', 'R_Thumb3_x', 'R_Thumb3_y', 'R_Thumb3_z'],
    dof_body_ids=[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 ],
    joint_axis=['xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz'],
    dof_obs_size=306,
    number_of_actions=153,
    self_obs_max_coords_size=778,
    left_foot_name="L_Ankle",
    right_foot_name="R_Ankle",
    head_body_name="Head",
    key_bodies=[ "L_Ankle", "R_Ankle", "L_Wrist", "R_Wrist" ],
    non_termination_contact_bodies=[ "L_Ankle", "R_Ankle", "L_Toe", "R_Toe" ],
    asset=robot_asset_config,
    control=ControlConfig(
        control_type=ControlType.BUILT_IN_PD,
    )
)

# Create simulator configuration
simulator_config = IsaacGymSimulatorConfig(
    sim=IsaacGymSimParams(
        fps=60,
        decimation=2,
        substeps=2,
    ),
    headless=False,  # Set to True for headless mode
    robot=robot_config,
    num_envs=4,  # Number of parallel environments
    experiment_name="complex_terrain_isaacgym_example",
    w_last=True,  # IsaacGym uses xyzw quaternions
)

device = torch.device("cuda:0")

# Create a flat terrain using the default config
terrain_config = TerrainConfig(
    num_terrains=7,
    num_levels=7,
    terrain_proportions=[ 0.2, 0.1, 0.1, 0.1, 0.05, 0., 0., 0.45 ],
    minimal_humanoid_spacing=0,  # We defined the terrain size, so no need for additional humanoid spacing
)
terrain = Terrain(config=terrain_config, num_envs=simulator_config.num_envs, device=device)

# Create and initialize the simulator
simulator = IsaacGymSimulator(config=simulator_config, terrain=terrain, scene_lib=None, visualization_markers=None, device=device)
simulator.on_environment_ready()

# Get robot default state
default_state = simulator.get_default_state()
# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_config.num_envs, 3, device=device)
xy_pos = terrain.sample_valid_locations(simulator_config.num_envs)
height = terrain.get_ground_heights(xy_pos).view(-1)
root_pos[:, :2] = xy_pos
root_pos[:, 2] = height + 1.1 # Height determines the height of the terrain, add offset to properly spawn above ground without collisions
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(default_state, env_ids=torch.arange(simulator_config.num_envs, device=device))

# Run the simulation loop
try:
    while True:
        actions = torch.randn(simulator_config.num_envs, simulator_config.robot.number_of_actions, device=device)
        simulator.step(actions)
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()
