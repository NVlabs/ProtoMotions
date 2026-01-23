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
Tutorial 6: Mimic Environment

This tutorial demonstrates how to use the Mimic environment for imitation learning.
Building on Tutorial 5 (motion management), we now show:
1. How to use Mimic environment (extends BaseEnv with imitation learning features)
2. How to configure mimic observations for policy training
3. How sync_motion mode works for kinematic playback
4. How to access reference motion data and compute tracking errors
5. Understanding mimic rewards and observations for training

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
from protomotions.envs.base_env.config import EnvConfig, RewardComponentConfig  # noqa: E402
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig  # noqa: E402
from protomotions.envs.obs.observation_component import ObservationComponentConfig  # noqa: E402
from protomotions.envs.control.mimic_control import MimicControlConfig  # noqa: E402
from protomotions.envs.obs import (  # noqa: E402
    max_coords_obs_factory,
    previous_actions_factory,
    mimic_target_poses_max_coords_factory,
)
from protomotions.envs.rewards import mean_squared_error_exp, rotation_error_exp, norm  # noqa: E402
from protomotions.components.motion_lib import MotionLibConfig  # noqa: E402
from protomotions.components.scene_lib import SceneLibConfig  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import factory functions
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402

robot_cfg = robot_config("smpl")

print("\n=== Robot Configuration ===")
print("Robot type: smpl")
print(f"Robot config class: {type(robot_cfg).__name__}")
print(f"Number of actions: {robot_cfg.number_of_actions}")
print(f"Number of DOFs: {robot_cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {robot_cfg.kinematic_info.num_bodies}")
print(f"Contact bodies: {robot_cfg.contact_bodies}")

# Extra simulator parameters
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
    experiment_name="mimic_environment_tutorial",
)

print("\n=== Simulator Configuration ===")
print(f"Simulator type: {args.simulator}")
print(f"Simulator class: {get_class(simulator_cfg._target_).__name__}")
print(f"Number of environments: {simulator_cfg.num_envs}")
print(f"Device: {device}")
print(f"Headless: {simulator_cfg.headless}")

# Motion file for imitation learning
motion_file = "examples/data/smpl_humanoid_sit_armchair.motion"

print("\n=== Mimic Environment Configuration ===")
print(f"Motion file: {motion_file}")
print("This contains reference motion data for imitation learning")

# Configure modular components - this is key for imitation learning
# The new modular system uses control_components, observation_components,
# reward_components, and termination_components

print("\n=== Modular Component Configuration ===")
print("Using new modular component system:")
print("  - control_components: MimicControl manages reference motion tracking")
print("  - observation_components: Stateless functions for computing observations")
print("  - reward_components: Reward functions for imitation learning")
print("  - termination_components: Termination conditions")

# Control component - MimicControl manages reference motion and tracking
control_components = {
    "mimic": MimicControlConfig(
        bootstrap_on_episode_end=True,  # Continue at end of motion instead of terminating
    )
}
print("\nControl Components:")
print("  - 'mimic': MimicControl for reference motion management")
print("    → Provides ref_state context for observations and rewards")

# Observation components - using factory functions
observation_components = {
    # Current robot state observation
    "max_coords_obs": max_coords_obs_factory(),
    # Previous actions for temporal awareness
    "previous_actions": previous_actions_factory(),
    # Mimic target poses - reference motion for policy to track
    "mimic_target_poses": mimic_target_poses_max_coords_factory(
        with_velocities=True,
        num_future_steps=1,
    ),
}
print("\nObservation Components:")
print("  - 'max_coords_obs': Current robot state (positions, rotations, velocities)")
print("  - 'previous_actions': Action history for temporal awareness")
print("  - 'mimic_target_poses': Reference poses from motion library")

# Reward configuration - tracking rewards for imitation learning
reward_components = {
    "action_smoothness": RewardComponentConfig(
        function=norm,
        variables={"x": "current_actions - previous_actions"},
        weight=-0.02,
    ),
    "position_tracking": RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_pos",
            "ref_x": "ref_state_rigid_body_pos",
            "coefficient": -100.0,
        },
        weight=0.5,
    ),
    "rotation_tracking": RewardComponentConfig(
        function=rotation_error_exp,
        variables={
            "q": "current_state_rigid_body_rot",
            "ref_q": "ref_state_rigid_body_rot",
            "coefficient": -5.0,
        },
        weight=0.3,
    ),
    "velocity_tracking": RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_vel",
            "ref_x": "ref_state_rigid_body_vel",
            "coefficient": -0.5,
        },
        weight=0.1,
    ),
    "angular_velocity_tracking": RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_ang_vel",
            "ref_x": "ref_state_rigid_body_ang_vel",
            "coefficient": -0.1,
        },
        weight=0.1,
    ),
}
print("\nReward Components:")
print("  - 'action_smoothness': Penalize jerky actions")
print("  - 'position_tracking': Match reference body positions")
print("  - 'rotation_tracking': Match reference body rotations")
print("  - 'velocity_tracking': Match reference velocities")
print("  - 'angular_velocity_tracking': Match reference angular velocities")


# Motion sampling configuration - for kinematic playback, use demo mode
init_start_prob = (
    0.5  # 50% chance to start from beginning, 50% to start randomly as in DeepMimic RSI
)

print("\n=== Motion Sampling for Kinematic Playback ===")
print(f"init_start_prob: {init_start_prob}")
print(f"Random sampling: {1.0 - init_start_prob:.0%}")


from protomotions.components.scene_lib import (  # noqa: E402
    ObjectOptions,
    MeshSceneObject,
    Scene,
    SceneLib,
)

# Define object physics properties
chair_options = ObjectOptions(
    density=1000,
    fix_base_link=True,
    angular_damping=0.01,
    linear_damping=0.01,
    max_angular_velocity=100.0,
    vhacd_enabled=True,
    vhacd_params={
        "max_convex_hulls": 10,
        "max_num_vertices_per_ch": 64,
        "resolution": 300000,
    },
)

object_path = None
if args.simulator == "isaaclab":
    object_path = "examples/data/armchair.usda"
elif args.simulator == "newton":
    object_path = "examples/data/armchair.obj"
else:
    object_path = "examples/data/armchair.urdf"
# Create scene with chair
chair = MeshSceneObject(
    object_path=object_path,
    options=chair_options,
    translation=(0.0, 0.9, 0.0),
    rotation=(0.0, 0.0, 0.0, 1.0),
)

scene = Scene(objects=[chair], humanoid_motion_id=0)

# Create environment configuration with modular components
env_config = EnvConfig(
    max_episode_length=300,  # Shorter episodes for training
    # Modular components replace old-style config classes
    control_components=control_components,
    observation_components=observation_components,
    reward_components=reward_components,
    # Motion manager configuration
    motion_manager=MimicMotionManagerConfig(init_start_prob=init_start_prob),
)

print("\n=== Environment Configuration Summary ===")
print("Environment type: BaseEnv with modular MimicControl")
print(f"Episode length: {env_config.max_episode_length} (shorter for training)")
print(f"Control components: {list(control_components.keys())}")
print(f"Observation components: {list(observation_components.keys())}")
print(f"Reward components: {list(reward_components.keys())}")

# Create terrain with simple flat configuration
from protomotions.components.terrains.terrain import Terrain  # noqa: E402

terrain_config = TerrainConfig()  # Simple flat terrain for this demo (default is flat)
terrain = Terrain(config=terrain_config, num_envs=simulator_cfg.num_envs, device=device)

scene_lib_config = SceneLibConfig(scene_file=None)
scene_lib = SceneLib(
    config=scene_lib_config,
    num_envs=simulator_cfg.num_envs,
    scenes=[scene],
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

# Create the environment with modular components
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
print("BaseEnv with MimicControl created successfully")
print(f"Motion library loaded: {env.motion_lib is not None}")

# Analyze the motion library
if env.motion_lib is not None:
    print("\n=== Motion Library Analysis ===")
    print(f"Number of motions: {env.motion_lib.num_motions()}")
    print(f"Motion file: {env.motion_lib.motion_file}")

    for motion_id in range(env.motion_lib.num_motions()):
        motion_length = env.motion_lib.get_motion_length(motion_id)
        print(f"  Motion {motion_id}: {motion_length.item():.2f}s duration")

print(f"Motion manager type: {type(env.motion_manager).__name__}")
print(f"Control components: {list(env.control_manager.components.keys())}")

# Reset environment - this also returns observations!
print("\n=== Resetting Environment ===")
env.reset()

print("Environment reset - ready for imitation learning")
print(f"Motion IDs: {env.motion_manager.motion_ids}")
print(f"Motion times: {env.motion_manager.motion_times}")

# Use the observations from reset
obs = env.get_obs()
print("\n=== Observation Structure After Reset ===")
print(f"Observation keys: {list(obs.keys())}")

# Print all observation shapes
for key, value in obs.items():
    print(f"{key} observation shape: {value.shape}")

# Print detailed breakdown of what each observation contains
print("\n=== Observation Content Breakdown ===")
for key, value in obs.items():
    print(
        f"'{key}': shape {value.shape}, range [{value.min().item():.3f}, {value.max().item():.3f}]"
    )

    if key == "max_coords_obs":
        print("  → Current robot state: positions, rotations, velocities")
    elif key == "historical_max_coords_obs":
        print("  → Historical robot states for temporal awareness")
    elif key == "terrain":
        print("  → Terrain height map around robot")
    elif key == "mimic_phase":
        print("  → Motion phase (sin/cos of progress)")
    elif key == "mimic_time_left":
        print("  → Time remaining in current motion")
    elif key == "mimic_target_poses":
        print("  → Future reference poses for imitation learning")
    else:
        print(f"  → {key} observations")

# Explain the modular control approach
print("\n=== Modular Control System ===")
print("The new modular system uses control components instead of sync_motion flag:")
print("")
print("POLICY TRAINING MODE (MimicControl):")
print("  - Actions control the robot via physics simulation")
print("  - MimicControl provides ref_state context for rewards/observations")
print("  - Robot learns to follow motion data through reward signals")
print("  - Good for imitation learning training")
print("")
print("KINEMATIC PLAYBACK MODE (KinematicReplayControl):")
print("  - Would use KinematicReplayControl instead of MimicControl")
print("  - Actions are ignored, robot follows motion exactly")
print("  - Good for visualization and debugging")
print("  - See Tutorial 5 for KinematicReplayControl example")

# Run simulation
print("\n=== Starting Modular Mimic Environment Simulation ===")
print("This demonstrates modular imitation learning:")
print("  - MimicControl provides reference motion context")
print("  - Observation components compute policy inputs")
print("  - Reward components drive learning towards motion tracking")
print("Camera controls:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    step_count = 0
    done_indices = None
    while True:
        # reset if Done
        _ = env.reset(done_indices)

        # Generate random actions (in real training, these would come from a policy)
        actions = torch.randn(env.num_envs, robot_cfg.number_of_actions, device=device)

        # Step the environment
        obs, rewards, dones, terminated, infos = env.step(actions)
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

        step_count += 1

        # Print detailed information every 100 steps
        if step_count % 100 == 0:
            motion_times = env.motion_manager.motion_times
            motion_ids = env.motion_manager.motion_ids

            print(f"\nStep {step_count} - Mimic Environment Status:")
            print("  Motion Tracking:")
            print(f"    - Motion IDs: {motion_ids}")
            print(f"    - Motion times: {motion_times}")

            print("  Imitation Learning:")
            print(f"    - Rewards: {rewards}")
            print(f"    - Average reward: {rewards.mean().item():.3f}")
            print(f"    - Dones: {dones.sum().item()} environments")
            print(
                f"    - Actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]"
            )

            print("  Observations after env.step():")
            obs = env.get_obs()
            print(f"    - Observation keys: {list(obs.keys())}")
            for key, value in obs.items():
                print(
                    f"    - '{key}': shape {value.shape}, range [{value.min().item():.3f}, {value.max().item():.3f}]"
                )

            # Show environment resets due to motion completion or failure
            if dones.sum() > 0:
                print(
                    f"    - {dones.sum().item()} environments reset (motion complete or tracking failed)"
                )

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    env.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated the modular component system for imitation learning:")
print("")
print("1. Control Components (MimicControl):")
print("   - Manages reference motion tracking")
print("   - Provides ref_state context to observations and rewards")
print("   - bootstrap_on_episode_end controls motion looping")
print("")
print("2. Observation Components:")
print("   - max_coords_obs: Current robot state")
print("   - previous_actions: Action history for temporal awareness")
print("   - mimic_target_poses: Reference poses for policy to track")
print("")
print("3. Reward Components:")
print("   - Position, rotation, velocity tracking rewards")
print("   - Action smoothness penalty")
print("   - All configured via eval strings referencing context")
print("")
print("Key Modular Concepts:")
print("- Control components provide task-specific context (ref_state)")
print("- Observation functions are stateless, configured via variables dict")
print("- Rewards use eval strings to access current_state, ref_state, etc.")
print("- Everything is configurable without subclassing environments")
print("")
print("Builds on Tutorial 5: Adds reward-based imitation learning")
print("Next: Advanced topics like AMP, ASE, and multi-task learning!")
