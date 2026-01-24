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
Tutorial 4: Basic Environment

This tutorial shows how to use the existing BaseEnv framework from protomotions.envs.base_env.env
to extend the scene creation from tutorial 3. We demonstrate:
1. How to properly configure and use the BaseEnv class
2. Getting robot state from the environment
3. Accessing basic observations from the framework
4. Using the environment in a simulation loop

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
    "--robot", type=str, required=True, help="Robot to use (e.g., 'g1', 'smpl')"
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
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import factory functions
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402

robot_cfg = robot_config(args.robot)


# We don't need to create a custom class - BaseEnv handles everything!
# BaseEnv will automatically create terrain and scene based on the config we provide


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
    experiment_name="basic_environment_tutorial",
)

# Create environment configuration
# The BaseEnv requires proper configuration to work correctly
# We configure terrain and scene through the config, not by overriding methods
from protomotions.components.scene_lib import (  # noqa: E402
    ObjectOptions,
    MeshSceneObject,
    Scene,
    SceneLibConfig,
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


env_config = EnvConfig(
    max_episode_length=1000,
    # Modular observation components (new pattern)
    observation_components={
        "max_coords_obs": max_coords_obs_factory(),
    },
)

# Create terrain with complex configuration
terrain_config = TerrainConfig()
from protomotions.components.terrains.terrain import Terrain  # noqa: E402

terrain = Terrain(config=terrain_config, num_envs=simulator_cfg.num_envs, device=device)

# Create SceneLib with inline scene
scene_lib_config = SceneLibConfig(scene_file=None)
scene_lib = SceneLib(
    config=scene_lib_config,
    num_envs=simulator_cfg.num_envs,
    scenes=[scene],  # Pass scene directly
    device=device,
    terrain=terrain,
)

# Create empty motion_lib (no motions for this example)
from protomotions.components.motion_lib import MotionLib  # noqa: E402

motion_lib = MotionLib.empty(device=device)

# Create simulator shell
SimulatorClass = get_class(simulator_cfg._target_)
simulator = SimulatorClass(
    config=simulator_cfg,
    robot_config=robot_cfg,
    terrain=terrain,
    scene_lib=scene_lib,
    device=device,
    **extra_simulator_params,  # Used to pass in simulation_app for IsaacLab
)

# Create the environment using BaseEnv directly
# Env automatically initializes simulator at end of __init__
env = BaseEnv(
    config=env_config,
    robot_config=robot_cfg,
    device=device,
    terrain=terrain,
    scene_lib=scene_lib,
    motion_lib=motion_lib,  # Always provided (empty if no motions)
    simulator=simulator,
)

print(f"Environment initialized with {env.num_envs} environments")
print(f"Robot has {robot_cfg.number_of_actions} actions")

# Reset the environment to get initial observations
print("\n=== Demonstrating BaseEnv Usage ===")

# The BaseEnv provides a reset method for resetting specific environments
env_ids = torch.arange(env.num_envs, device=device)
env.reset(env_ids)

# Get observations using the framework's get_obs method
print("\n=== Getting Observations from BaseEnv ===")
obs = env.get_obs()
print(f"Observation keys: {list(obs.keys())}")
print(f"Humanoid observation shape: {obs['max_coords_obs'].shape}")
print(f"Terrain observation shape: {obs['terrain'].shape}")

# The BaseEnv provides direct access to robot state through the simulator
print("\n=== Accessing Robot State ===")
current_state = env.simulator.get_robot_state()

print(f"Root positions shape: {current_state.root_pos.shape}")
print(f"Root rotations shape: {current_state.root_rot.shape}")
print(f"Joint positions shape: {current_state.dof_pos.shape}")
print(f"Joint velocities shape: {current_state.dof_vel.shape}")
print(f"Body positions shape: {current_state.rigid_body_pos.shape}")

print("\n=== Starting Simulation Loop ===")
print("This demonstrates the basic step loop with BaseEnv")
print("Camera controls:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    step_count = 0
    while env.is_simulation_running():
        # Generate random actions (normally these would come from a policy)
        actions = torch.randn(env.num_envs, robot_cfg.number_of_actions, device=device)

        # BaseEnv.step() returns the full RL interface: (obs, rewards, done, extras)
        # This automatically handles:
        # - Stepping the simulator
        # - Computing observations
        # - Computing rewards (if implemented)
        # - Checking termination conditions
        # - Resetting terminated environments
        obs, rewards, dones, terminated, extras = env.step(actions)

        step_count += 1

        # Print some information every 100 steps
        if step_count % 100 == 0:
            obs = env.get_obs()
            print(f"Step {step_count}:")
            print(f"  Observation keys: {list(obs.keys())}")
            print(f"  Humanoid obs shape: {obs['max_coords_obs'].shape}")
            print(f"  Terrain obs shape: {obs['terrain'].shape}")
            print(f"  Rewards shape: {rewards.shape}")
            print(f"  Done/reset buffer shape: {dones.shape}")
            print(f"  Extras keys: {list(extras.keys()) if extras else 'None'}")
            print(f"  Average reward: {rewards.mean().item():.3f}")
            print(f"  Environments reset this step: {dones.sum().item()}")
            print(
                f"  Humanoid obs range: [{obs['max_coords_obs'].min().item():.3f}, {obs['max_coords_obs'].max().item():.3f}]"
            )
            print(
                f"  Terrain obs range: [{obs['terrain'].min().item():.3f}, {obs['terrain'].max().item():.3f}]"
            )

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    env.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated:")
print(
    "1. How to configure BaseEnv with EnvConfig and modular observation_components"
)
print("2. How BaseEnv automatically creates terrain and scene based on configuration")
print("3. How to access robot state through env.simulator")
print("4. How to get structured observations through env.get_obs()")
print("5. How env.step() returns the full RL interface: (obs, rewards, done, extras)")
print("6. How BaseEnv handles the complete RL loop including resets automatically")
print(
    "7. How BaseEnv provides the foundation for more complex environments like Mimic, AMP, etc."
)
print(
    "\nKey takeaway: BaseEnv provides a complete RL environment - just configure it and call step()!"
)
