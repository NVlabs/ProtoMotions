"""
    IsaacLab and app launcher must be setup before all other imports.
"""

from isaaclab.app import AppLauncher

headless = False
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

import numpy as np
import torch

from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig, IsaacLabSimParams
from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
from protomotions.simulator.base_simulator.config import (
    RobotConfig,
    RobotAssetConfig,
    InitState,
    ControlConfig,
    ControlType,
)
from protomotions.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig
from protomotions.utils.scene_lib import (
    Scene,
    SceneObject,
    ObjectOptions,
    SceneLib,
)

# Create robot asset configuration
robot_asset_config = RobotAssetConfig(
    robot_type="h1",
    usd_asset_file_name="usd/h1.usd",
    self_collisions=False,
    collapse_fixed_joints=False,
)

# Create robot configuration
robot_config = RobotConfig(
    body_names=['pelvis', 'head', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link', 'left_foot_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link', 'right_foot_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_arm_end_effector', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_arm_end_effector'],
    dof_names=['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint'],
    dof_body_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
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
simulator_config = IsaacLabSimulatorConfig(
    sim=IsaacLabSimParams(
        fps=200,
        decimation=4,
    ),
    headless=headless,  # Set to True for headless mode
    robot=robot_config,
    num_envs=4,  # Number of parallel environments
    experiment_name="scene_isaaclab_example",
    w_last=False,  # IsaacLab uses wxyz quaternions
)

device = torch.device("cuda")

# Create a flat terrain using the default config
terrain_config = TerrainConfig()
terrain = FlatTerrain(config=terrain_config, num_envs=simulator_config.num_envs, device=device)


"""
    We create a single scene with two objects.
    The scene defines an elephant on-top of a table.
    Since there are 4 humanoids and a single scene, the scene will be replicated across all humanoids.
"""
# Create object options for elephant
elephant_options = ObjectOptions(
    density=1000,
    fix_base_link=False,
    angular_damping=0.01,
    linear_damping=0.01,
    max_angular_velocity=100.0,
    default_dof_drive_mode="DOF_MODE_NONE",
    override_com=True,
    override_inertia=True,
    vhacd_enabled=True,
    vhacd_params={
        "max_convex_hulls": 32,
        "max_num_vertices_per_ch": 72,
        "resolution": 300000,
    },
)

# Create object options for table
table_options = ObjectOptions(
    density=1000,
    fix_base_link=True,
    angular_damping=0.01,
    linear_damping=0.01,
    max_angular_velocity=100.0,
    default_dof_drive_mode="DOF_MODE_NONE",
    vhacd_enabled=True,
    override_com=True,
    override_inertia=True,
    vhacd_params={
        "max_convex_hulls": 10,
        "max_num_vertices_per_ch": 64,
        "resolution": 300000,
    },
)

# Create elephant object with motion
elephant = SceneObject(
    object_path="examples/data/elephant.usda",
    options=elephant_options,
    translation=(0.0, 0.0, 0.94),
    rotation=(0.0, 0.0, 0.0, 1.0),
)


# Create table object with motion
table = SceneObject(
    object_path="examples/data/table.urdf",
    options=table_options,
    translation=(0.0, 0.0, 0.87),
    rotation=(0.0, 0.0, 0.0, 1.0),
)

# Create scene with both objects
scene = Scene(id=1, objects=[elephant, table])

# Create SceneLib instance
scene_lib = SceneLib(num_envs=simulator_config.num_envs, device=device)

# Create scenes
scene_lib.create_scenes([scene], terrain)

# Create and initialize the simulator
simulator = IsaacLabSimulator(config=simulator_config, terrain=terrain, scene_lib=scene_lib, visualization_markers=None, device=device, simulation_app=simulation_app)
simulator.on_environment_ready()

# Get the scene positions. This indicates the center of the scene.
scene_positions = torch.stack(simulator.get_scene_positions())

# Get robot default state
default_state = simulator.get_default_state()
# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_config.num_envs, 3, device=device)
# Add small offset so we don't spawn ontop of the scene
root_pos[:, :2] = scene_positions[:, :2] + 0.5
root_pos[:, 2] = 1.0
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(default_state, env_ids=torch.arange(simulator_config.num_envs, device=device))

# Run the simulation loop
try:
    while True:
        """
            Camera controls in IsaacLab and IsaacGym:
            1. L - start/stop recording. Once stopped it will save the video.
            2. ; - cancel recording and delete the video.
            3. O - toggle camera target. This will cycle through the available camera targets, such as humanoids and objects in the scene.
            4. Q - close the simulator.
        """
        actions = torch.randn(simulator_config.num_envs, simulator_config.robot.number_of_actions, device=device)
        simulator.step(actions)
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()
