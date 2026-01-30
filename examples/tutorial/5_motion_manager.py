# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Tutorial 5: Motion Manager and Motion Library

This tutorial focuses on understanding motion management in the protomotions framework.
We demonstrate:
1. How to load and inspect motion data (MotionLib)
2. How to configure motion manager parameters and what they mean
3. How motion sampling works (init_start_prob, motion selection)
4. How motion tracking works (motion IDs, times, progress)
5. How to set specific motions and control motion playback
6. Understanding different motion manager configurations

This tutorial uses the teapot pour motion data, which includes:
- A 52-body SMPL-X humanoid motion
- A scene with a teapot (dynamic) and table (static)

IsaacLab and IsaacGym must be imported before torch is imported.

As many modules may import torch internally, it is best practice to simply detect the selected simulator
at the top and import it right away.
"""

# Parse arguments first (argparse is safe, doesn't import torch)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulator",
    type=str,
    required=True,
    help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
)
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from protomotions.simulator.base_simulator.config import SimulatorConfig  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.base_env.config import EnvConfig  # noqa: E402
from protomotions.envs.obs import max_coords_obs_factory  # noqa: E402
from protomotions.envs.control.kinematic_replay_control import KinematicReplayControlConfig  # noqa: E402
from protomotions.components.motion_lib import MotionLibConfig  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import factory functions
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.smplx import SMPLXRobotConfig  # noqa: E402
from protomotions.robot_configs.base import ControlConfig, ControlType, ControlInfo

robot_cfg = SMPLXRobotConfig(
    control= ControlConfig(
        override_control_info={
            # For precise kinematic replay we set the stiffness and damping to 1
            # Otherwise physics-in-the-loop will introduce small errors
            ".*": ControlInfo(
                stiffness=1, damping=1, effort_limit=500, velocity_limit=100
            ),
        },
    ),
)

print("\n=== Robot Configuration ===")
print("Robot type is hard-coded to: smplx (52 bodies with fingers)")
print(f"Robot config class: {type(robot_cfg).__name__}")
print(f"Number of actions: {robot_cfg.number_of_actions}")
print(f"Number of DOFs: {robot_cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {robot_cfg.kinematic_info.num_bodies}")
print(f"Contact bodies: {robot_cfg.contact_bodies}")


# We don't need to create a custom environment - we'll use Mimic directly
# This builds on Tutorial 4's approach of using the framework's existing environments


# Extra simulator parameters allow you to pass in additional parameters to the simulator constructor.
# For example, if you use IsaacLab, you need to pass in the simulation app.
extra_simulator_params = {}
if args.simulator == "isaaclab":
    app_launcher_flags = {"headless": False, "device": str(device)}
    app_launcher = AppLauncher(app_launcher_flags)
    simulation_app = app_launcher.app
    extra_simulator_params["simulation_app"] = simulation_app

# Create simulator configuration
simulator_cfg: SimulatorConfig = simulator_config(
    args.simulator,
    robot_cfg,
    headless=False,
    num_envs=4,
    experiment_name="kinematic_scene_playback_tutorial",
)

print("\n=== Simulator Configuration ===")
print(f"Simulator type: {args.simulator}")
print(f"Simulator class: {get_class(simulator_cfg._target_).__name__}")
print(f"Number of environments: {simulator_cfg.num_envs}")
print(f"Device: {device}")
print(f"Headless: {simulator_cfg.headless}")

# Motion Library Setup - This is the core of motion management
# The motion file contains pre-recorded motion capture data
# This teapot pour motion was converted from SkeletonMotion format
import numpy as np  # noqa: E402

motion_file = "examples/data/grab_teapot_pour/s1_teapot_pour_1.motion"

print("\n=== Motion Library Deep Dive ===")
print(f"Motion file: {motion_file}")
print("This contains reference motion data for pouring from a teapot")
print("  - 1130 frames at 120 FPS (~9.4 seconds)")
print("  - 52-body SMPLX humanoid with full hand articulation")

# The motion file is saved with torch.save (for MotionLib compatibility)
# MotionLib will load it directly using torch.load
print("\nMotion file will be loaded by MotionLib using torch.load")
print("(Verifying file format...)")
motion_data = torch.load(motion_file, weights_only=False)
print(f"  ✓ Loaded: {motion_data['rigid_body_pos'].shape[0]} frames, {motion_data['rigid_body_pos'].shape[1]} bodies")
print(f"  ✓ FPS: {motion_data['fps']}")

# Scene Setup - Load objects from individual numpy files
from protomotions.components.scene_lib import (  # noqa: E402
    SceneLibConfig,
    SceneLib,
    Scene,
    MeshSceneObject,
    BoxSceneObject,
    ObjectOptions,
)

data_dir = "examples/data/grab_teapot_pour"

# Load teapot (dynamic object with motion)
print("\n=== Loading Scene Objects ===")
teapot_file = f"{data_dir}/s1_teapot_pour_1_teapot.npy"
print(f"\nTeapot file: {teapot_file}")
teapot_data = np.load(teapot_file, allow_pickle=True).item()

teapot_translation = torch.from_numpy(teapot_data['translation']).float()
teapot_rotation = torch.from_numpy(teapot_data['rotation']).float()
print(f"  Type: {teapot_data['type']}")
print(f"  Frames: {teapot_translation.shape[0]}, FPS: {teapot_data['fps']}")
print(f"  Mesh path: {teapot_data['object_path']}")

# Use URDF for isaacgym, USDA for isaaclab
teapot_mesh_path = f"{data_dir}/teapot.usda"
if args.simulator == "isaaclab":
    pass
elif args.simulator == "newton":
    teapot_mesh_path = teapot_mesh_path.replace('.usda', '.ply')
else:
    teapot_mesh_path = teapot_mesh_path.replace('.usda', '.urdf')

teapot = MeshSceneObject(
    object_path=teapot_mesh_path,
    options=ObjectOptions(
        fix_base_link=False,  # Dynamic object
        density=1000,
        angular_damping=0.01,
        linear_damping=0.01,
        max_angular_velocity=100.0,
    ),
    translation=teapot_translation,
    rotation=teapot_rotation,
    fps=teapot_data['fps'],
)

# Load table (static object)
table_file = f"{data_dir}/s1_teapot_pour_1_table.npy"
print(f"\nTable file: {table_file}")
table_data = np.load(table_file, allow_pickle=True).item()

table_translation = torch.from_numpy(table_data['translation']).float()
table_rotation = torch.from_numpy(table_data['rotation']).float()
print(f"  Type: {table_data['type']}")
print(f"  Dimensions: {table_data['width']:.3f} x {table_data['depth']:.3f} x {table_data['height']:.3f}")
print(f"  Static: {table_data['is_static']}")

table = BoxSceneObject(
    width=table_data['width'],
    depth=table_data['depth'],
    height=table_data['height'],
    options=ObjectOptions(
        fix_base_link=True,  # Static object
        density=1000,
        angular_damping=0.01,
        linear_damping=0.01,
        max_angular_velocity=100.0,
    ),
    translation=table_translation,
    rotation=table_rotation,
    fps=table_data['fps'],
)

# Create the scene with both objects
scene = Scene(objects=[teapot, table], humanoid_motion_id=0)
print(f"\nScene created with {len(scene.objects)} objects")

# Motion Manager Configuration - This is where we control motion behavior
print("\n=== Motion Manager Configuration Deep Dive ===")

# Let's explore different motion manager configurations and their meanings

init_start_prob = 1.0  # 100% chance to start from motion beginning

print("\nRandom Configuration (Challenging for training):")
print(f"  - init_start_prob: {init_start_prob}")
print(f"    → {init_start_prob:.0%} chance to start from time=0")
print(f"  → Remaining {1.0 - init_start_prob:.0%} chance for random time sampling")

print("\n=== Motion Manager Parameter Meanings ===")
print("Key Motion Manager Parameters:")
print("1. init_start_prob: Probability of starting motion from time=0")
print("   - Higher values = more predictable starts")
print("   - Lower values = more diverse training scenarios")
print("   - 1.0 = always start from beginning (good for demos)")
print("   - 0.2 = balanced approach for training")

print("\n2. Random sampling (remaining probability):")
print("   - Samples random time points in the motion")
print("   - Creates diverse training scenarios")
print("   - Helps policy generalize to different motion phases")

# For this tutorial, we'll use the demo configuration
# Building on Tutorial 4, we use BaseEnv with motion library and motion manager
env_config = EnvConfig(
    max_episode_length=1000,
    # Uses reference motion resets automatically (motion_lib is set)
    # Modular control components - KinematicReplayControl enables kinematic playback
    control_components={
        "kinematic_replay": KinematicReplayControlConfig(),
    },
    # Modular observation components
    observation_components={
        "max_coords_obs": max_coords_obs_factory(),
    },
    # No terminations needed for kinematic playback
    termination_components={},
    # No rewards needed for kinematic playback
    reward_components={},
    # Motion manager configuration
    motion_manager=MimicMotionManagerConfig(
        init_start_prob=init_start_prob  # Use demo config for clear visualization
    ),
)

print("\n=== Environment Configuration Summary ===")
print("Environment type: BaseEnv with modular components")
print("Control: KinematicReplayControl (kinematic motion playback)")
print("Motion sampling: Always start from beginning (demo mode)")
print(f"Max episode length: {env_config.max_episode_length}")

# Create terrain with simple flat configuration
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator  # noqa: E402

terrain_config = (
    TerrainConfig()
)  # Simple flat terrain for this motion demo (default is flat)

# Convert friction settings for the specific simulator
# Newton requires CombineMode.MAX, IsaacGym requires CombineMode.AVERAGE
# This utility handles the conversion automatically
terrain_config, simulator_cfg = convert_friction_for_simulator(terrain_config, simulator_cfg)

terrain = Terrain(config=terrain_config, num_envs=simulator_cfg.num_envs, device=device)

# Create scene library from the manually constructed scene
scene_lib_config = SceneLibConfig(scene_file=None)
scene_lib = SceneLib(
    config=scene_lib_config,
    num_envs=simulator_cfg.num_envs,
    scenes=[scene],  # Use the scene we created from numpy data
    device=device,
    terrain=terrain,
)

motion_lib_config = MotionLibConfig(motion_file=motion_file)
from protomotions.components.motion_lib import MotionLib  # noqa: E402

motion_lib = MotionLib(config=motion_lib_config, device=device)

from protomotions.utils.hydra_replacement import get_class  # noqa: E402

SimulatorClass = get_class(simulator_cfg._target_)
simulator = SimulatorClass(
    config=simulator_cfg,
    robot_config=robot_cfg,
    terrain=terrain,
    scene_lib=scene_lib,
    device=device,
    **extra_simulator_params,
)

# Create the environment using BaseEnv (same as Tutorial 4)
# BaseEnv will automatically create motion library and motion manager from config
# KinematicReplayControl handles the kinematic playback
env = BaseEnv(
    config=env_config,
    robot_config=robot_cfg,
    device=device,
    simulator=simulator,
    motion_lib=motion_lib,
    terrain=terrain,
    scene_lib=scene_lib,
)

print("\n=== Environment Initialization ===")
print("Environment created successfully")
print(f"Motion library loaded: {env.motion_lib is not None}")

# Now we can analyze the motion library that the environment created
if env.motion_lib is not None:
    print("\n=== Motion Library Analysis (Created by Environment) ===")
    print("Motion library details:")
    print(f"  - Number of motions: {env.motion_lib.num_motions()}")
    print(f"  - Motion file path: {env.motion_lib.motion_file}")
    print(f"  - Device: {env.motion_lib.device}")

    # Analyze each motion in detail
    for motion_id in range(env.motion_lib.num_motions()):
        motion_length = env.motion_lib.get_motion_length(motion_id)
        motion_dt = env.motion_lib.motion_dt[motion_id]
        num_frames = int(motion_length.item() / motion_dt.item())

        print(f"\n  Motion {motion_id}:")
        print(f"    - Duration: {motion_length.item():.2f} seconds")
        print(f"    - Time step (dt): {motion_dt.item():.4f} seconds")
        print(f"    - Number of frames: {num_frames}")
        print(f"    - FPS: {1.0/motion_dt.item():.1f}")

        # Sample motion at different time points to show data structure
        sample_times = torch.tensor(
            [0.0, motion_length.item() * 0.5, motion_length.item() * 0.9], device=device
        )
        sample_states = env.motion_lib.get_motion_state(
            torch.tensor([motion_id] * len(sample_times), device=device), sample_times
        )

        print(f"    - Root position at start: {sample_states.root_pos[0]}")
        print(f"    - Root position at middle: {sample_states.root_pos[1]}")
        print(f"    - Root position at end: {sample_states.root_pos[2]}")
        print(f"    - DOF positions shape: {sample_states.dof_pos.shape}")
        print(f"    - Root rotation shape: {sample_states.root_rot.shape}")
        print(f"    - Body positions shape: {sample_states.rigid_body_pos.shape}")

print(f"Scene library created: {env.scene_lib is not None}")
print(f"Motion manager created: {env.motion_manager is not None}")
print(f"Motion manager type: {type(env.motion_manager).__name__}")

# Reset the environment to initialize motion manager
print("\n=== Resetting Environment ===")
env.reset()

print("Environment reset - motion manager initialized")
print(f"Motion IDs: {env.motion_manager.motion_ids}")
print(f"Motion times: {env.motion_manager.motion_times}")

# Run simulation with random actions
print("\n=== Starting Simulation ===")
print("Camera controls:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    step_count = 0
    while env.is_simulation_running():
        # Generate random actions
        actions = torch.randn(env.num_envs, robot_cfg.number_of_actions, device=device)

        # Step the environment
        obs, rewards, dones, terminated, infos = env.step(actions)

        step_count += 1

        # Print information every 100 steps
        if step_count % 100 == 0:
            motion_times = env.motion_manager.motion_times
            motion_ids = env.motion_manager.motion_ids

            print(f"Step {step_count}:")
            print(f"  Motion IDs: {motion_ids}")
            print(f"  Motion times: {motion_times}")
            print(f"  Rewards: {rewards}")
            print(f"  Dones: {dones.sum().item()} environments")

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    env.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated:")
print("1. How to load robot motion from .motion file (torch format)")
print("2. How to load object data from separate .npy files (numpy format)")
print("3. How to create Scene objects programmatically")
print("4. How motion manager parameters control motion sampling")
print("5. How to track motion progress during simulation")
print("6. How to use KinematicReplayControl for motion visualization")
print("\nData files used:")
print("  - s1_teapot_pour_1.motion  : Robot motion (torch format for MotionLib)")
print("  - s1_teapot_pour_1_teapot.npy  : Teapot motion (numpy format)")
print("  - s1_teapot_pour_1_table.npy   : Table data (numpy format)")
print("\nRobot: SMPLX humanoid (52 bodies with hand articulation)")
print("Scene: Dynamic teapot + static table")
print("\nBuilds on Tutorial 4: Uses BaseEnv with modular control/observation components")
print("Next: Tutorial 6 will show how to train policies to mimic motions!")
