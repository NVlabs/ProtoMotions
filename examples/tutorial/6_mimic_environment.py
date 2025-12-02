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
from protomotions.envs.mimic.env import Mimic  # noqa: E402
from protomotions.envs.mimic.config import MimicEnvConfig  # noqa: E402
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig  # noqa: E402
from protomotions.envs.obs.config import (  # noqa: E402
    MimicObsConfig,
    HumanoidObsConfig,
    MaxCoordsSelfObsConfig,
)
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

# Configure mimic observations - this is key for imitation learning
from protomotions.envs.obs.config import (  # noqa: E402
    MimicPhaseObsConfig,
    MimicTimeLeftObsConfig,
    MimicTargetPoseConfig,
)

mimic_obs_config = MimicObsConfig(
    enabled=True,  # Enable mimic observations
    mimic_phase_obs=MimicPhaseObsConfig(
        enabled=True  # Enable phase observations (sin/cos of motion progress)
    ),
    mimic_time_left_obs=MimicTimeLeftObsConfig(
        enabled=True  # Enable time left observations
    ),
    mimic_target_pose=MimicTargetPoseConfig(
        enabled=True,  # Enable target pose observations
        future_steps=1,  # Look 1 step ahead
        with_time=True,  # Include time information
        with_contacts=False,  # Don't include contact information
        with_velocities=False,  # Don't include velocity information
    ),
)

print("\n=== Mimic Observations Configuration ===")
print(f"Mimic observations enabled: {mimic_obs_config.enabled}")
print(f"Phase observations: {mimic_obs_config.mimic_phase_obs.enabled}")
print("  → Provides sin/cos of motion progress (cyclical)")
print(f"Time left observations: {mimic_obs_config.mimic_time_left_obs.enabled}")
print("  → Provides remaining time in current motion")
print(f"Target pose observations: {mimic_obs_config.mimic_target_pose.enabled}")
print("  → Provides future reference poses for policy")
print(f"  → Future steps: {mimic_obs_config.mimic_target_pose.future_steps}")
print(f"  → With time: {mimic_obs_config.mimic_target_pose.with_time}")
print(f"  → With contacts: {mimic_obs_config.mimic_target_pose.with_contacts}")
print(f"  → With velocities: {mimic_obs_config.mimic_target_pose.with_velocities}")

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

# Create Mimic environment configuration
env_config = MimicEnvConfig(
    max_episode_length=300,  # Shorter episodes for training
    # Uses reference motion resets automatically (motion_lib is set)
    sync_motion=False,  # IMPORTANT: True for kinematic playback, False for policy training
    humanoid_obs=HumanoidObsConfig(
        max_coords_obs=MaxCoordsSelfObsConfig(
            enabled=True,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            num_historical_steps=2,  # Include some history for better policies
        )
    ),
    mimic_obs=mimic_obs_config,  # Add mimic-specific observations
    motion_manager=MimicMotionManagerConfig(init_start_prob=init_start_prob),
)

print("\n=== Mimic Environment Summary ===")
print("Environment type: Mimic (extends BaseEnv for imitation learning)")
print(
    f"Sync motion: {env_config.sync_motion} (False = policy training, True = kinematic)"
)
print(f"Episode length: {env_config.max_episode_length} (shorter for training)")
print(
    f"Historical steps: {env_config.humanoid_obs.max_coords_obs.num_historical_steps}"
)
print(f"Mimic observations: {env_config.mimic_obs.enabled}")

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

# Create the Mimic environment
env = Mimic(
    config=env_config,
    robot_config=robot_cfg,
    device=device,
    simulator=simulator,
    motion_lib=motion_lib,
    terrain=terrain,
    scene_lib=scene_lib,
    **extra_simulator_params,
)

print("\n=== Environment Initialization ===")
print("Mimic environment created successfully")
print(f"Motion library loaded: {env.motion_lib is not None}")

# Analyze the motion library that the environment created
if env.motion_lib is not None:
    print("\n=== Motion Library Analysis ===")
    print(f"Number of motions: {env.motion_lib.num_motions}")
    print(f"Motion file: {env.motion_lib.motion_file}")

    for motion_id in range(env.motion_lib.num_motions()):
        motion_length = env.motion_lib.get_motion_length(motion_id)
        print(f"  Motion {motion_id}: {motion_length.item():.2f}s duration")

print(f"Motion manager type: {type(env.motion_manager).__name__}")
print(f"Mimic observations callback: {hasattr(env, 'mimic_obs_cb')}")

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

# Demonstrate the difference between sync_motion modes
print("\n=== Sync Motion Modes Demonstration ===")
print(f"Current mode: sync_motion={env.config.sync_motion}")

if env.config.sync_motion:
    print("KINEMATIC MODE (sync_motion=True):")
    print("  - Actions are ignored (zeroed internally)")
    print("  - Robot follows motion data exactly")
    print("  - Good for visualization and debugging")
    print("  - Motion time advances automatically")
else:
    print("POLICY TRAINING MODE (sync_motion=False):")
    print("  - Actions control the robot")
    print("  - Robot learns to follow motion data")
    print("  - Good for imitation learning training")
    print("  - Rewards based on tracking reference motion")

# Run simulation
print("\n=== Starting Mimic Environment Simulation ===")
print("This demonstrates imitation learning concepts:")
print("  - Reference motion tracking")
print("  - Mimic observations for policy training")
print("  - Motion-based rewards and termination")
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
print("This tutorial demonstrated:")
print("1. How to use Mimic environment (extends BaseEnv for imitation learning)")
print("2. How to configure mimic observations for policy training:")
print("   - Reference body tracking")
print("   - Local vs global observations")
print("   - Historical observations for temporal awareness")
print("3. How sync_motion modes work:")
print("   - sync_motion=True: Kinematic playback (demo/visualization)")
print("   - sync_motion=False: Policy training (imitation learning)")
print("4. How motion sampling creates diverse training scenarios")
print("5. How mimic observations provide reference data to policies")
print("6. How rewards and termination work for imitation learning")
print("\nKey Mimic Environment Concepts:")
print("- Reference Motion Tracking: Compare robot pose to motion data")
print("- Mimic Observations: Provide reference poses/velocities to policy")
print("- Imitation Learning: Train policies to reproduce motion data")
print("- Motion Diversity: Random sampling creates robust training")
print("\nBuilds on Tutorial 5: Adds imitation learning to motion management")
print("Next: Advanced topics like AMP, ASE, and multi-task learning!")
