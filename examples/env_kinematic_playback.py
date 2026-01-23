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
Environment Kinematic Playback Script

This script allows you to visualize reference motions in kinematic playback mode without training.
It uses the KinematicReplayControl component to directly set robot state to reference motion poses,
bypassing physics simulation entirely.

Usage:
    python examples/env_kinematic_playback.py \
        --experiment-path=examples/experiments/mimic/mlp.py \
        --motion-file=xxx.pt \
        --robot-name=g1 \
        --simulator=isaacgym \
        --num-envs=14 \
        --scenes-file=xxx.pt
"""


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Visualize environment in kinematic playback mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        help="Name of the robot (e.g., 'h1', 'g1', 'smpl')",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to motion file for playback",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="File path to experiment configuration (e.g., 'examples/experiments/mimic/mlp.py')",
    )

    # Optional arguments
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="kinematic_playback",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Use CPU only for simulation (experimental, GPU is default)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
import importlib.util  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)


def main():
    # Re-use the parser and args from module level
    global parser, args

    device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

    # Dynamically import the module from file path
    experiment_path = Path(args.experiment_path)
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    args = parser.parse_args()

    print("\n=== Environment Kinematic Playback Configuration ===")
    print(f"Experiment path: {args.experiment_path}")
    print(f"Robot: {args.robot_name}")
    print(f"Simulator: {args.simulator}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Motion file: {args.motion_file}")
    print(f"Scenes file: {args.scenes_file}")
    print(f"Device: {device}")
    print(f"Headless: {args.headless}")

    # Extra simulator parameters
    extra_simulator_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulation_app = app_launcher.app
        extra_simulator_params["simulation_app"] = simulation_app

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Get config functions from experiment module
    from protomotions.utils.config_builder import build_standard_configs
    from protomotions.simulator.base_simulator.config import SimulatorConfig
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.robot_configs.base import RobotConfig

    # Build configs from experiment (without agent for kinematic playback)
    print("\n=== Building Configuration from Experiment ===")

    # Get required config functions
    terrain_config_fn = getattr(experiment_module, "terrain_config")
    scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
    motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
    env_config_fn = getattr(experiment_module, "env_config")

    # Get optional config functions
    configure_robot_and_simulator_fn = getattr(
        experiment_module, "configure_robot_and_simulator", None
    )

    configs = build_standard_configs(
        args=args,
        terrain_config_fn=terrain_config_fn,
        scene_lib_config_fn=scene_lib_config_fn,
        motion_lib_config_fn=motion_lib_config_fn,
        env_config_fn=env_config_fn,
        configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
        agent_config_fn=None,  # No agent needed for kinematic playback
    )

    robot_config: RobotConfig = configs["robot"]
    simulator_config: SimulatorConfig = configs["simulator"]
    terrain_config = configs["terrain"]
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config: EnvConfig = configs["env"]

    print(f"Robot config class: {type(robot_config).__name__}")
    print(f"Simulator config class: {type(simulator_config).__name__}")
    print(f"Environment config class: {type(env_config).__name__}")

    if args.motion_file is not None:
        print(f"Motion library configured from: {args.motion_file}")

    if args.scenes_file is not None:
        print(f"Scene library configured from: {args.scenes_file}")

    # Enable kinematic playback mode using KinematicReplayControl
    from protomotions.envs.control.kinematic_replay_control import (
        KinematicReplayControlConfig,
    )
    
    print("Enabling kinematic playback via KinematicReplayControl component")
    env_config.show_terrain_markers = False
    
    # Add kinematic replay control component (replaces any existing control components)
    env_config.control_components = {
        "kinematic_replay": KinematicReplayControlConfig(),
    }
    
    # Disable terminations - kinematic replay should run indefinitely
    env_config.termination_components = {}
    
    # Disable observations - not needed for kinematic playback
    env_config.observation_components = {}
    
    # Disable rewards - not needed for kinematic playback
    env_config.reward_components = {}

    print("\n=== Creating Environment ===")

    # Convert friction settings for simulator compatibility
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # Create components using configs from build_standard_configs
    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        save_dir=save_dir_for_weights,
        **extra_simulator_params,
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # Create environment - use BaseEnv directly for kinematic playback
    from protomotions.envs.base_env.env import BaseEnv

    env: BaseEnv = BaseEnv(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )

    print("Environment created successfully")
    print(f"Environment class: {type(env).__name__}")
    print(f"Motion library loaded: {env.motion_lib is not None}")
    print(f"  - Number of motions: {env.motion_lib.num_motions()}")
    print(f"  - Motion file: {env.motion_lib.motion_file}")
    print(f"Scene library loaded: {env.scene_lib is not None}")
    print(f"  - Number of scenes: {env.scene_lib.num_scenes()}")
    if hasattr(env.scene_lib, "scenes_file"):
        print(f"  - Scenes file: {env.scene_lib.scenes_file}")
    print(f"Motion manager created: {env.motion_manager is not None}")
    if env.motion_manager is not None:
        print(f"  - Motion manager type: {type(env.motion_manager).__name__}")

    # Reset the environment
    print("\n=== Resetting Environment ===")
    env.reset()
    print("Environment reset complete")

    if env.motion_manager is not None:
        print(f"Motion IDs assigned: {env.motion_manager.motion_ids}")
        print(f"Motion times initialized: {env.motion_manager.motion_times}")

    # Run simulation loop
    print("\n=== Starting Kinematic Playback ===")
    print("This will play back the reference motion kinematically")
    print("The humanoid will follow the motion capture data exactly")
    print("\nCamera controls:")
    print("  L - start/stop recording")
    print("  ; - cancel recording")
    print("  O - toggle camera target")
    print("  Q - close simulator")

    try:
        step_count = 0
        while env.is_simulation_running():
            # In kinematic playback mode, actions are ignored
            # The environment will automatically follow the reference motion
            actions = torch.zeros(
                env.num_envs, robot_config.number_of_actions, device=device
            )

            # Step the environment
            obs, rewards, dones, terminated, infos = env.step(actions)

            step_count += 1

            # Print information every 100 steps
            if step_count % 100 == 0 and env.motion_manager is not None:
                motion_times = env.motion_manager.motion_times
                motion_ids = env.motion_manager.motion_ids

                print(f"\nStep {step_count}:")
                print(
                    f"  Motion IDs: {motion_ids[:4].tolist()}..."
                    if env.num_envs > 4
                    else f"  Motion IDs: {motion_ids.tolist()}"
                )
                print(
                    f"  Motion times: {motion_times[:4].tolist()}..."
                    if env.num_envs > 4
                    else f"  Motion times: {motion_times.tolist()}"
                )
                print(f"  Rewards: {rewards.mean().item():.4f} (mean)")
                print(f"  Dones: {dones.sum().item()} environments reset")

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    finally:
        env.close()

    print("\n=== Playback Complete ===")
    print(f"Total steps: {step_count}")
    print("Environment closed successfully")


if __name__ == "__main__":
    main()
