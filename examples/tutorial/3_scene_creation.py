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
from protomotions.components.scene_lib import SceneLibConfig  # noqa: E402
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
    num_envs=1,
    experiment_name="smpl_humanoid_isaaclab_example",
)
SimulatorClass = get_class(simulator_cfg._target_)

print("\n=== Simulator Configuration ===")
print(f"Simulator type: {args.simulator}")
print(f"Simulator class: {SimulatorClass.__name__}")
print(f"Number of environments: {simulator_cfg.num_envs}")
print(f"Device: {device}")
print(f"Headless: {simulator_cfg.headless}")

from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator  # noqa: E402

# We always require the surface plane to be defined.
# In this case, we define an irregular terrain.
# We provide convenient defaults in the config, which can be overridden.
# Here we override the default terrain properties to include stepping stones and poles.
terrain_config = TerrainConfig()
# We also provide a pre-defined helper for complex terrains in from protomotions.components.terrains.config import ComplexTerrainConfig

# Convert friction settings for the specific simulator
# Newton requires CombineMode.MAX, IsaacGym requires CombineMode.AVERAGE
# This utility handles the conversion automatically
terrain_config, simulator_cfg = convert_friction_for_simulator(terrain_config, simulator_cfg)

# The terrain config provides a pointer to the specific terrain class.
terrain = Terrain(config=terrain_config, num_envs=simulator_cfg.num_envs, device=device)

from protomotions.components.scene_lib import (  # noqa: E402
    ObjectOptions,
    MeshSceneObject,
    Scene,
    SceneLib,
    BoxSceneObject,
)

print("\n=== Scene and Object Creation ===")
print("Creating a scene with an elephant placed on a table for robot interaction")

# Create object options for elephant
elephant_options = ObjectOptions(
    density=1000,
    fix_base_link=False,
    angular_damping=0.01,
    linear_damping=0.01,
    max_angular_velocity=100.0,
    vhacd_enabled=True,
    vhacd_params={
        "max_convex_hulls": 32,
        "max_num_vertices_per_ch": 72,
        "resolution": 300000,
    },
)

print("Elephant object options configured:")
print(f"  - Density: {elephant_options.density}")
print(f"  - Fixed base: {elephant_options.fix_base_link}")
print(f"  - VHACD collision: {elephant_options.vhacd_enabled}")
print(f"  - Max convex hulls: {elephant_options.vhacd_params['max_convex_hulls']}")

# Create object options for table
table_options = ObjectOptions(
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
    object_path = "examples/data/elephant.usda"
elif args.simulator == "newton":
    object_path = "examples/data/elephant.stl"
else:
    object_path = "examples/data/elephant.urdf"

# Create elephant object
elephant = MeshSceneObject(
    object_path=object_path,
    options=elephant_options,
    translation=(0.0, 0.0, 1.5),
    rotation=(0.0, 0.0, 0.0, 1.0),
)

print("Elephant object created:")
print(f"  - Asset path: {elephant.object_path}")
print(f"  - Position: {elephant.translation}")
print(f"  - Rotation: {elephant.rotation}")
print(f"  - Object type: {type(elephant).__name__}")

# Create table object
table = BoxSceneObject(
    width=1.0,
    depth=1.0,
    height=0.1,
    options=table_options,
    translation=(0.0, 0.0, 0.87),
    rotation=(0.0, 0.0, 0.0, 1.0),
)

# Define joint objects as a single scene. This will be replicated across environments.
scene = Scene(objects=[elephant, table], humanoid_motion_id=0)

# Configure scene lib. In this case, we provide a single scene and it will be replicated.
scene_lib_config = SceneLibConfig(
    scene_file=None,  # No file, using inline scene
    replicate_method="random",
    subset_method="random",
    pointcloud_samples_per_object=None,
)

print("Scene created:")
print(f"  - Number of objects: {len(scene.objects)}")
print(f"  - Humanoid motion ID: {scene.humanoid_motion_id}")
print(f"  - Scene type: {type(scene).__name__}")

# Create SceneLib instance with inline scene
scene_lib = SceneLib(
    config=scene_lib_config,
    num_envs=simulator_cfg.num_envs,
    scenes=[scene],
    device=device,
    terrain=terrain,
)

print("SceneLib created:")
print(f"  - Number of environments: {scene_lib.num_envs}")
print(f"  - Device: {scene_lib.device}")

print("Scenes instantiated:")
print(f"  - Scenes created for all {simulator_cfg.num_envs} environments")
print("  - Each environment has its own elephant and table instance")
print("  - Objects positioned on complex terrain")

from protomotions.simulator.base_simulator.simulator import Simulator  # noqa: E402

# Create the simulator shell. This is the main class that handles the simulation loop.
# In the later tutorials, we will use the environment class to wrap the simulator and provide a more user-friendly interface.
simulator: Simulator = SimulatorClass(
    config=simulator_cfg,
    robot_config=robot_cfg,
    scene_lib=scene_lib,
    terrain=terrain,
    device=device,
    **extra_simulator_params,  # Used to pass in simulation_app for IsaacLab
)

# Initialize the simulator (two-phase: shell created above, now finalize)
# Note: Normally Env does this, but here we're using simulator directly
simulator._initialize_with_markers({})  # Empty markers for this tutorial

print("\n=== Simulator Initialization ===")
print("Simulator initialized successfully")
print(f"Simulation timestep (dt): {simulator.dt}")
print("Simulator now includes scenes with objects")

# Get robot and object default states
default_state = simulator.get_default_robot_reset_state()
default_object_state = scene_lib.get_default_object_state(device)
default_state.root_pos[:, :2] += 10.5
default_state.root_pos[:, 2] = robot_cfg.default_root_height + 0.1
default_object_state.root_pos[:, :, :2] += 10.0

print("\n=== Robot State Information ===")
print(f"Default robot state: {default_state}")
print(f"Default object state: {default_object_state}")


# Reset the robots and objects
simulator.reset_envs(
    default_state,
    default_object_state,
    env_ids=torch.arange(simulator_cfg.num_envs, device=device),
)
print(f"All {args.robot} robots reset near their scene objects")

# Get object information
object_state = simulator.get_object_root_state()
print("\n=== Object State Information ===")
print(f"Object root positions shape: {object_state.root_pos.shape}")
print(f"Object root rotations shape: {object_state.root_rot.shape}")
print(f"First environment - Elephant position: {object_state.root_pos[0, 0]}")
print(f"First environment - Elephant rotation: {object_state.root_rot[0, 0]}")

# Run the simulation loop
print("\n=== Starting Simulation Loop ===")
print("This demonstrates robot-object interaction with scenes")
print("Robots are positioned near objects and can potentially interact with them")
print("Camera controls:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    step_count = 0
    while simulator.is_simulation_running():
        # Generate random actions for all environments
        # Robots will move randomly and may interact with the armchairs
        actions = torch.randn(
            simulator_cfg.num_envs, robot_cfg.number_of_actions, device=device
        )

        # Step the simulator forward by one timestep
        # Both robots and objects are simulated
        simulator.step(actions)

        step_count += 1

        # Print information every 100 steps to show robot-object interaction
        if step_count % 100 == 0:
            # Get current robot and object states
            current_robot_state = simulator.get_root_state()
            current_object_state = simulator.get_object_root_state()

            robot_avg_height = current_robot_state.root_pos[:, 2].mean().item()
            elephant_avg_height = current_object_state.root_pos[:, 0, 2].mean().item()

            # Calculate distance between robots and elephants
            robot_pos = current_robot_state.root_pos
            elephant_pos = current_object_state.root_pos[
                :, 0, :
            ]  # First (and only) object
            distances = torch.norm(robot_pos - elephant_pos, dim=1)
            avg_distance = distances.mean().item()

            print(f"Step {step_count}:")
            print(f"  Robot type: {args.robot}")
            print(f"  Actions shape: {actions.shape}")
            print(
                f"  Actions range: [{actions.min().item():.3f}, {actions.max().item():.3f}]"
            )
            print(f"  Average robot height: {robot_avg_height:.3f}")
            print(f"  Average elephant height: {elephant_avg_height:.3f}")
            print(f"  Average robot-elephant distance: {avg_distance:.3f}")
            print(f"  Robot positions shape: {current_robot_state.root_pos.shape}")
            print(f"  Object positions shape: {current_object_state.root_pos.shape}")
            print(f"  Closest robot-elephant distance: {distances.min().item():.3f}")

except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated:")
print("1. How to create objects with physics properties (ObjectOptions)")
print("2. How to create mesh-based scene objects (MeshSceneObject)")
print("3. How to compose scenes with multiple objects (Scene)")
print("4. How to manage scenes across environments (SceneLib)")
print("5. How to position robots relative to scene objects")
print("6. How to access object state information during simulation")
print("7. How to monitor robot-object interactions")
print("\nNext: Tutorial 4 shows how to wrap this in a BaseEnv for RL!")
