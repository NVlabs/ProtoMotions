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
from protomotions.robot_configs.smpl import SmplRobotConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import the simulator factory function
from protomotions.simulator.factory import simulator_config  # noqa: E402

# Create a simple SMPL humanoid robot configuration.
robot_cfg = SmplRobotConfig()

print("\n=== Robot Configuration ===")
print("Robot type: SMPL Humanoid")
print(f"Number of actions: {robot_cfg.number_of_actions}")
print(f"Number of DOFs: {robot_cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {robot_cfg.kinematic_info.num_bodies}")
print(f"Contact bodies: {robot_cfg.contact_bodies}")

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

print("\n=== Complex Terrain Configuration ===")
print("Terrain type: Complex (multi-type terrain)")
print(f"Terrain proportions: {terrain_config.terrain_proportions}")
print(f"  - Smooth slope: {terrain_config.terrain_proportions[0]:.1%}")
print(f"  - Rough slope: {terrain_config.terrain_proportions[1]:.1%}")
print(f"  - Stairs up: {terrain_config.terrain_proportions[2]:.1%}")
print(f"  - Stairs down: {terrain_config.terrain_proportions[3]:.1%}")
print(f"  - Discrete: {terrain_config.terrain_proportions[4]:.1%}")
print(f"  - Stepping stones: {terrain_config.terrain_proportions[5]:.1%}")
print(f"  - Poles: {terrain_config.terrain_proportions[6]:.1%}")
print(f"  - Flat: {terrain_config.terrain_proportions[7]:.1%}")

# The terrain config provides a pointer to the specific terrain class.
TerrainClass = get_class(terrain_config._target_)
terrain: Terrain = TerrainClass(
    config=terrain_config, num_envs=simulator_cfg.num_envs, device=device
)

print("\n=== Terrain Properties ===")
print(f"Terrain class: {TerrainClass.__name__}")
print(f"Number of height points: {terrain.num_height_points}")
print(
    f"Terrain dimensions: {terrain_config.map_length * terrain.num_maps}x{terrain_config.map_width * terrain.num_maps}"
)
print(f"Number of terrain types: {terrain.env_cols * terrain.env_rows}")
print(
    f"Height samples shape: {terrain.height_samples.shape if hasattr(terrain, 'height_samples') else 'N/A'}"
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

# Get robot default state.
default_state = simulator.get_default_robot_reset_state()

print("\n=== Robot State Information ===")
print(f"Default state type: {type(default_state).__name__}")
print(f"Root positions shape: {default_state.root_pos.shape}")
print(f"Root rotations shape: {default_state.root_rot.shape}")
print(f"DOF positions shape: {default_state.dof_pos.shape}")
print(f"DOF velocities shape: {default_state.dof_vel.shape}")

# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_cfg.num_envs, 3, device=torch.device("cpu"))
xy_pos = terrain.sample_valid_locations(simulator_cfg.num_envs)
height = terrain.get_ground_heights(xy_pos).view(-1)
root_pos[:, :2] = xy_pos
root_pos[:, 2] = (
    height + 1.1
)  # Height determines the height of the terrain, add offset to properly spawn above ground without collisions
default_state.root_pos[:] = root_pos

print("\n=== Robot Positioning on Complex Terrain ===")
print(f"Sampled XY positions shape: {xy_pos.shape}")
print(f"Ground heights shape: {height.shape}")
print(f"Ground height range: [{height.min().item():.3f}, {height.max().item():.3f}]")
print(f"Final root positions shape: {root_pos.shape}")
print(f"First environment position: {root_pos[0]}")
print("Height variation demonstrates complex terrain features")

# Reset the robots
simulator.reset_envs(
    default_state, env_ids=torch.arange(simulator_cfg.num_envs, device=device)
)
print("Robots reset to positions on different terrain types")

# Run the simulation loop
print("\n=== Starting Simulation Loop ===")
print("This demonstrates simulator usage with complex terrain")
print("Robots will interact with slopes, stairs, stepping stones, and poles")
print("Camera controls:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    step_count = 0
    while simulator.is_simulation_running():
        # Generate random actions for all environments
        # On complex terrain, robots will experience different challenges based on their location
        actions = torch.randn(
            simulator_cfg.num_envs, robot_cfg.number_of_actions, device=device
        )

        # Step the simulator forward by one timestep
        # Complex terrain will create more varied robot behaviors
        simulator.step(actions)

        step_count += 1

        # Print information every 100 steps to show terrain interaction
        if step_count % 100 == 0:
            # Get current robot state to show terrain interaction
            current_state = simulator.get_root_state()
            avg_height = current_state.root_pos[:, 2].mean().item()
            height_std = current_state.root_pos[:, 2].std().item()

            # Get terrain heights at robot positions for comparison
            terrain_heights = terrain.get_ground_heights(current_state.root_pos[:, :2])
            avg_terrain_height = terrain_heights.mean().item()

            print(f"Step {step_count}:")
            print(f"  Actions shape: {actions.shape}")
            print(
                f"  Actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]"
            )
            print(f"  Average robot height: {avg_height:.3f}")
            print(f"  Robot height std dev: {height_std:.3f} (shows terrain variety)")
            print(f"  Average terrain height: {avg_terrain_height:.3f}")
            print(f"  Height above terrain: {(avg_height - avg_terrain_height):.3f}")
            print(f"  Root positions shape: {current_state.root_pos.shape}")

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated:")
print("1. How to create complex terrain with multiple obstacle types")
print("2. How terrain proportions control the mix of terrain features")
print("3. How robots are positioned on varied terrain heights")
print("4. How complex terrain creates more challenging robot behaviors")
print("5. How to monitor terrain interaction through height measurements")
print("6. The difference between flat terrain (Tutorial 0) and complex terrain")
print("\nNext: Tutorial 2 shows how to load different robot types!")
