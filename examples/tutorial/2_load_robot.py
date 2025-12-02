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
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import factory functions
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402

robot_cfg = robot_config(args.robot)

print("\n=== Robot Configuration ===")
print(f"Robot type: {args.robot}")
print(f"Robot config class: {type(robot_cfg).__name__}")
print(f"Number of actions: {robot_cfg.number_of_actions}")
print(f"Number of DOFs: {robot_cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {robot_cfg.kinematic_info.num_bodies}")
print(f"Body names: {robot_cfg.kinematic_info.body_names}")
print(f"Contact bodies: {robot_cfg.contact_bodies}")
print(
    f"Robot asset path: {robot_cfg.asset.asset_root}/{robot_cfg.asset.asset_file_name}"
)
print(f"Control type: {robot_cfg.control}")

# Extra simulator parameters allow you to pass in additional parameters to the simulator constructor.
# For example, if you use IsaacLab, you need to pass in the simulation app.
extra_simulator_params = {}
if args.simulator == "isaaclab":
    app_launcher_flags = {"headless": False, "device": str(device)}
    app_launcher = AppLauncher(app_launcher_flags)
    simulation_app = app_launcher.app
    extra_simulator_params["simulation_app"] = simulation_app

simulator_cfg: SimulatorConfig = simulator_config(
    args.simulator,
    robot_cfg,
    headless=False,
    num_envs=4,
    experiment_name="smpl_humanoid_isaaclab_example",
)
SimulatorClass = get_class(simulator_cfg._target_)

print("\n=== Simulator Configuration ===")
print(f"Simulator type: {args.simulator}")
print(f"Simulator class: {SimulatorClass.__name__}")
print(f"Number of environments: {simulator_cfg.num_envs}")
print(f"Device: {device}")
print(f"Headless: {simulator_cfg.headless}")

from protomotions.components.terrains.config import ComplexTerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402

# We always require the surface plane to be defined.
# In this case, we define an irregular terrain.
# We provide convenient defaults in the config, which can be overridden.
# Here we override the default terrain properties to include stepping stones and poles.
terrain_config = ComplexTerrainConfig(
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]
    # default: terrain_proportions=[ 0.2, 0.1, 0.1, 0.1, 0.05, 0., 0., 0.45 ],
    terrain_proportions=[0.2, 0.1, 0.1, 0.1, 0.05, 0.2, 0.3, 0.1],
)
# The terrain config provides a pointer to the specific terrain class.
TerrainClass = get_class(terrain_config._target_)
terrain: Terrain = TerrainClass(
    config=terrain_config, num_envs=simulator_cfg.num_envs, device=device
)

# Create empty scene_lib (no scenes for this tutorial)
from protomotions.components.scene_lib import SceneLib  # noqa: E402

scene_lib = SceneLib.empty(num_envs=simulator_cfg.num_envs, device=device)

from protomotions.simulator.base_simulator.simulator import Simulator  # noqa: E402

# Create the simulator shell. This is the main class that handles the simulation loop.
# In the later tutorials, we will use the environment class to wrap the simulator and provide a more user-friendly interface.
simulator: Simulator = SimulatorClass(
    config=simulator_cfg,
    robot_config=robot_cfg,
    scene_lib=scene_lib,  # Always provide (empty if no scenes)
    terrain=terrain,
    device=device,
    **extra_simulator_params,  # Used to pass in simulation_app for IsaacLab
)

# Initialize the simulator (two-phase: shell created above, now finalize)
simulator._initialize_with_markers({})  # Empty markers for this tutorial

print("\n=== Simulator Initialization ===")
print("Simulator initialized successfully")
print(f"Simulation timestep (dt): {simulator.dt}")
print(f"Robot loaded: {args.robot}")

# Get robot default state.
default_state = simulator.get_default_robot_reset_state()

print("\n=== Robot State Information ===")
print(f"Default state type: {type(default_state).__name__}")
print(f"Root positions shape: {default_state.root_pos.shape}")
print(f"Root rotations shape: {default_state.root_rot.shape}")
print(f"DOF positions shape: {default_state.dof_pos.shape}")
print(f"DOF velocities shape: {default_state.dof_vel.shape}")
print(f"Default root position: {default_state.root_pos[0]}")
print(f"Default joint positions (first 5): {default_state.dof_pos[0][:5]}")

# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_cfg.num_envs, 3, device=torch.device("cpu"))
xy_pos = terrain.sample_valid_locations(simulator_cfg.num_envs)
height = terrain.get_ground_heights(xy_pos).view(-1)
root_pos[:, :2] = xy_pos
root_pos[:, 2] = (
    height + 1.1
)  # Height determines the height of the terrain, add offset to properly spawn above ground without collisions
default_state.root_pos[:] = root_pos

print("\n=== Robot Positioning ===")
print(f"Sampled XY positions shape: {xy_pos.shape}")
print(f"Ground heights shape: {height.shape}")
print(f"Final root positions shape: {root_pos.shape}")
print(f"First environment position: {root_pos[0]}")
print(f"Robot type '{args.robot}' positioned on complex terrain")

# Reset the robots
simulator.reset_envs(
    default_state, env_ids=torch.arange(simulator_cfg.num_envs, device=device)
)
print(f"All {args.robot} robots reset to new positions")

# Run the simulation loop
print("\n=== Starting Simulation Loop ===")
print(f"This demonstrates {args.robot} robot behavior on complex terrain")
print("Different robot types will show different movement characteristics")
print("Camera controls:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    step_count = 0
    while simulator.is_simulation_running():
        # Generate random actions for all environments
        # Different robot types will have different action spaces
        actions = torch.randn(
            simulator_cfg.num_envs, robot_cfg.number_of_actions, device=device
        )

        # Step the simulator forward by one timestep
        # Robot-specific dynamics will be evident in the movement
        simulator.step(actions)

        step_count += 1

        # Print information every 100 steps to show robot-specific behavior
        if step_count % 100 == 0:
            # Get current robot state to show robot type differences
            current_state = simulator.get_root_state()
            dof_state = simulator.get_dof_state()
            avg_height = current_state.root_pos[:, 2].mean().item()

            print(f"Step {step_count}:")
            print(f"  Robot type: {args.robot}")
            print(
                f"  Actions shape: {actions.shape} ({robot_cfg.number_of_actions} actions)"
            )
            print(
                f"  Actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]"
            )
            print(f"  Average robot height: {avg_height:.3f}")
            print(f"  Root positions shape: {current_state.root_pos.shape}")
            print(
                f"  DOF positions shape: {dof_state.dof_pos.shape} ({robot_cfg.kinematic_info.num_dofs} DOFs)"
            )
            print(
                f"  Joint position range: [{dof_state.dof_pos.min().item():.3f}, {dof_state.dof_pos.max().item():.3f}]"
            )
            print(
                f"  Joint velocity range: [{dof_state.dof_vel.min().item():.3f}, {dof_state.dof_vel.max().item():.3f}]"
            )

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated:")
print(f"1. How to load different robot types (you used: {args.robot})")
print("2. How robot configurations differ between robot types")
print("3. How different robots have different action spaces and DOF counts")
print("4. How to access robot-specific state information")
print("5. How different robots behave differently with the same actions")
print("6. The importance of robot selection for different tasks")
print("\nTry running with different robots: --robot smpl, --robot g1")
print("Next: Tutorial 3 shows how to add objects and scenes!")
