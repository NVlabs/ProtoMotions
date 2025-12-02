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
from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

import torch  # noqa: E402
import time  # noqa: E402
import easydict  # noqa: E402

from rich.progress import track  # noqa: E402
from protomotions.simulator.isaaclab.config import (  # noqa: E402
    IsaacLabSimulatorConfig,
    IsaacLabSimParams,
)
from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator  # noqa: E402
from protomotions.robot_configs.base import (  # noqa: E402
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
)

from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.utils.config_utils import load_control_info_resolver  # noqa: E402
from protomotions.components.pose_lib import extract_kinematic_info  # noqa: E402
import os  # noqa: E402

# Create robot asset configuration
robot_asset_config = RobotAssetConfig(
    asset_file_name="mjcf/h1.xml",
    usd_asset_file_name="usd/h1.usd",
    self_collisions=False,
    usd_bodies_root_prim_path="/World/envs/env_.*/Robot/",
)

kinematic_info = extract_kinematic_info(
    os.path.join(robot_asset_config.asset_root, robot_asset_config.asset_file_name)
)
# Create robot configuration
override_control_info = {
    ".*_hip_.*": {
        "stiffness": 200,
        "damping": 5,
        "effort_limit": 200,
        "velocity_limit": 23,
        "armature": 0.1,
        "friction": 0.0,
    },
    ".*_knee_joint": {
        "stiffness": 300,
        "damping": 6,
        "effort_limit": 300,
        "velocity_limit": 14,
        "armature": 0.1,
        "friction": 0.0,
    },
    ".*_ankle_joint": {
        "stiffness": 40,
        "damping": 2,
        "effort_limit": 40,
        "velocity_limit": 9,
        "armature": 0.1,
        "friction": 0.0,
    },
    "torso_joint": {
        "stiffness": 300,
        "damping": 6,
        "effort_limit": 200,
        "velocity_limit": 23,
        "armature": 0.1,
        "friction": 0.0,
    },
    ".*_shoulder_(pitch|roll)_joint": {
        "stiffness": 100,
        "damping": 2,
        "effort_limit": 40,
        "velocity_limit": 9,
        "armature": 0.1,
        "friction": 0.0,
    },
    ".*_(shoulder_yaw|elbow)_joint": {
        "stiffness": 100,
        "damping": 2,
        "effort_limit": 18,
        "velocity_limit": 20,
        "armature": 0.1,
        "friction": 0.0,
    },
}

robot_config = RobotConfig(
    kinematic_info=kinematic_info,
    number_of_actions=kinematic_info.num_dofs,
    common_naming_to_robot_body_names={
        "all_left_foot_bodies": ["left_foot_link"],
        "all_right_foot_bodies": ["right_foot_link"],
        "all_left_hand_bodies": ["left_arm_end_effector"],
        "all_right_hand_bodies": ["right_arm_end_effector"],
        "head_body_name": ["head"],
        "torso_body_name": ["torso_link"],
    },
    non_termination_contact_bodies=[
        "left_foot_link",
        "left_ankle_link",
        "right_foot_link",
        "right_ankle_link",
    ],
    asset=robot_asset_config,
    control=ControlConfig(
        control_type=ControlType.PROPORTIONAL,
        control_info=load_control_info_resolver(
            robot_asset_config.asset_root,
            robot_asset_config.asset_file_name,
            easydict.EasyDict(override_control_info),
        ),
    ),
)

# Create simulator configuration
simulator_config = IsaacLabSimulatorConfig(
    sim=IsaacLabSimParams(
        fps=200,
        decimation=4,
    ),
    headless=headless,  # Set to True for headless mode
    robot=robot_config,
    num_envs=4096,  # Number of parallel environments
    experiment_name="h1_isaaclab_example",
    w_last=False,  # IsaacLab uses wxyz quaternions
)

device = torch.device("cuda")

# Create a flat terrain using the default config
terrain_config = TerrainConfig()
terrain = Terrain(
    config=terrain_config, num_envs=simulator_config.num_envs, device=device
)

# Create empty scene_lib
from protomotions.components.scene_lib import SceneLib  # noqa: E402

scene_lib = SceneLib.empty(num_envs=simulator_config.num_envs, device=device)

# Create and initialize the simulator
simulator = IsaacLabSimulator(
    config=simulator_config,
    robot_config=robot_config,
    terrain=terrain,
    scene_lib=scene_lib,
    device=device,
    simulation_app=simulation_app,
)
simulator._initialize_with_markers({})

# Get robot default state
default_state = simulator.get_default_robot_reset_state()
# Set the robot to a new random position above the ground
root_pos = torch.zeros(simulator_config.num_envs, 3, device=device)
root_pos[:, :2] = terrain.sample_valid_locations(simulator_config.num_envs).view(-1, 2)
root_pos[:, 2] = 1.0
default_state.root_pos[:] = root_pos

# Reset the robots
simulator.reset_envs(
    default_state, env_ids=torch.arange(simulator_config.num_envs, device=device)
)

# Run the simulation loop
try:
    for _ in track(range(100), description="Performing warmup steps"):
        actions = torch.randn(
            simulator_config.num_envs, robot_config.number_of_actions, device=device
        )
        simulator.step(actions)
    print("Warmup complete")
    # Run benchmark
    start_time = time.perf_counter()
    num_steps = 1000

    for _ in track(range(num_steps), description="Running benchmark"):
        actions = torch.randn(
            simulator_config.num_envs, robot_config.number_of_actions, device=device
        )
        simulator.step(actions)

    total_time = time.perf_counter() - start_time
    avg_time_per_step = total_time / num_steps

    print("\nBenchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per step: {avg_time_per_step*1000:.2f} ms")
    simulator.close()
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    simulator.close()
