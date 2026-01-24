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


def build_standard_configs(
    args,
    terrain_config_fn,
    scene_lib_config_fn,
    motion_lib_config_fn,
    env_config_fn,
    configure_robot_and_simulator_fn=None,
    agent_config_fn=None,
):
    """Build standard robot, simulator, terrain, scene_lib, motion_lib, env, and optionally agent configs.

    This is a helper function to reduce boilerplate in experiment files.
    All configs are built with training defaults - eval overrides applied separately via apply_inference_overrides().

    Parameter order matches execution order: robot → sim → terrain → scene_lib → motion_lib → env → agent

    Args:
        args: Command line arguments containing robot_name, simulator, etc.
        terrain_config_fn: REQUIRED function that takes (args) and returns TerrainConfig (or None for no terrain)
        scene_lib_config_fn: REQUIRED function that takes (args) and returns SceneLibConfig (scene_file can be None for empty)
        motion_lib_config_fn: REQUIRED function that takes (args) and returns MotionLibConfig (motion_file can be None for empty)
        env_config_fn: REQUIRED function that takes (robot_config, args) and returns env config
        configure_robot_and_simulator_fn: Optional function that takes (robot_config, simulator_config, args)
        agent_config_fn: Optional function that takes (robot_config, env_config, args) and returns agent config

    Returns:
        Dict with keys: robot, simulator, terrain, scene_lib, motion_lib, env, agent (optional)
    """
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.factory import simulator_config as simulator_config_func

    # Build robot config from factory
    robot_cfg = robot_config(args.robot_name)

    # Build simulator config from factory
    simulator_cfg = simulator_config_func(
        args.simulator, robot_cfg, args.headless, args.num_envs, args.experiment_name
    )

    # Configure robot and simulator for this experiment (if function provided)
    if configure_robot_and_simulator_fn is not None:
        configure_robot_and_simulator_fn(robot_cfg, simulator_cfg, args)

    # Build component configs (independent of robot_config)
    # These functions must always be provided
    terrain_cfg = terrain_config_fn(args)  # Can return None for no terrain (exception)
    scene_lib_cfg = scene_lib_config_fn(
        args
    )  # Must return SceneLibConfig (scene_file can be None)
    motion_lib_cfg = motion_lib_config_fn(
        args
    )  # Must return MotionLibConfig (motion_file can be None)

    # Build env config (depends on robot_config)
    env_cfg = env_config_fn(robot_cfg, args)

    # Build agent config if function provided (depends on robot_config and env_config)
    agent_cfg = (
        agent_config_fn(robot_cfg, env_cfg, args)
        if agent_config_fn is not None
        else None
    )

    return {
        "robot": robot_cfg,
        "simulator": simulator_cfg,
        "terrain": terrain_cfg,
        "scene_lib": scene_lib_cfg,
        "motion_lib": motion_lib_cfg,
        "env": env_cfg,
        "agent": agent_cfg,
    }
