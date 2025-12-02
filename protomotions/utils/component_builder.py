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
"""Utility functions for building environment components.

This module provides helper functions to create terrain, scene_lib, motion_lib,
and simulator objects from their configs, reducing boilerplate in entry scripts.
"""

from typing import Optional, Dict
import torch
from protomotions.utils.hydra_replacement import get_class


def build_terrain_from_config(terrain_config, num_envs: int, device: torch.device):
    """Build Terrain from config.

    Args:
        terrain_config: TerrainConfig or None (exception: can be None for no terrain)
        num_envs: Number of environments
        device: PyTorch device

    Returns:
        Terrain instance or None if config is None
    """
    if terrain_config is None:
        return None

    from protomotions.components.terrains.terrain import Terrain

    return Terrain(config=terrain_config, num_envs=num_envs, device=device)


def build_scene_lib_from_config(
    scene_lib_config,
    num_envs: int,
    device: torch.device,
    terrain,
    scene_weights: Optional[list] = None,
):
    """Build SceneLib from config.

    Always returns a SceneLib instance. If config.scene_file is None,
    creates an empty SceneLib (Null Object pattern).

    Args:
        scene_lib_config: SceneLibConfig (required, scene_file can be None for empty)
        num_envs: Number of environments
        device: PyTorch device
        terrain: Terrain instance (required by SceneLib)
        scene_weights: Optional scene weights for curriculum learning

    Returns:
        SceneLib instance (empty if scene_file is None)
    """
    from protomotions.components.scene_lib import SceneLib

    # Create SceneLib (config required, handles None scene_file - creates empty)
    return SceneLib(
        config=scene_lib_config,
        num_envs=num_envs,
        scenes=None,  # Will be loaded from scene_file if specified, otherwise empty
        device=device,
        terrain=terrain,
        scene_weights=scene_weights,
    )


def build_motion_lib_from_config(motion_lib_config, device: torch.device):
    """Build MotionLib from config.

    Always returns a MotionLib instance. If config.motion_file is None,
    creates an empty MotionLib (Null Object pattern).

    Note: Contact smoothing is NOT applied here - it's the Env's responsibility
    to modify the motion_lib during initialization based on its config.

    Args:
        motion_lib_config: MotionLibConfig (required, motion_file can be None for empty)
        device: PyTorch device

    Returns:
        MotionLib instance (empty if motion_file is None)
    """
    from protomotions.components.motion_lib import MotionLib

    return MotionLib(config=motion_lib_config, device=device)


def build_simulator_from_config(
    simulator_config,
    robot_config,
    terrain,
    scene_lib,
    device: torch.device,
    **simulator_extra_params,
):
    """Build Simulator from config.

    Creates simulator shell (deferred initialization - will be finalized by Env).

    Args:
        simulator_config: SimulatorConfig
        robot_config: RobotConfig
        terrain: Terrain instance
        scene_lib: SceneLib instance or None
        device: PyTorch device
        **simulator_extra_params: Simulator-specific params (e.g., simulation_app for IsaacLab)

    Returns:
        Simulator instance (shell, not yet initialized)
    """
    SimulatorClass = get_class(simulator_config._target_)
    return SimulatorClass(
        config=simulator_config,
        robot_config=robot_config,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
        **simulator_extra_params,
    )


def build_all_components(
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    simulator_config,
    robot_config,
    device: torch.device,
    save_dir: Optional[str] = None,
    **simulator_extra_params,
) -> Dict:
    """Build all environment components from configs.

    Convenience function that builds terrain, scene_lib, motion_lib, and simulator.

    Args:
        terrain_config: TerrainConfig (or None for no terrain - exception)
        scene_lib_config: SceneLibConfig (always provided, scene_file can be None for empty)
        motion_lib_config: MotionLibConfig (always provided, motion_file can be None for empty)
        simulator_config: SimulatorConfig
        robot_config: RobotConfig
        device: PyTorch device
        save_dir: Optional save directory for loading motion weights as scene weights
        **simulator_extra_params: Simulator-specific params (e.g., simulation_app for IsaacLab)

    Returns:
        Dict with keys: terrain, scene_lib, motion_lib, simulator
    """
    # Create terrain (can be None)
    terrain = build_terrain_from_config(
        terrain_config, simulator_config.num_envs, device
    )

    # Load motion weights from checkpoint to use as scene weights for prioritized sampling
    scene_weights = None
    if save_dir and motion_lib_config.motion_file:
        from protomotions.envs.base_env.env import BaseEnv

        scene_weights = BaseEnv.apply_motion_weights_to_scene_weights(
            save_dir=save_dir, motion_file=motion_lib_config.motion_file, device=device
        )

    # Create scene_lib (always created, empty if scene_file is None)
    scene_lib = build_scene_lib_from_config(
        scene_lib_config, simulator_config.num_envs, device, terrain, scene_weights
    )

    # Create motion_lib (always created, empty if motion_file is None)
    motion_lib = build_motion_lib_from_config(motion_lib_config, device)

    # Create simulator shell
    simulator = build_simulator_from_config(
        simulator_config,
        robot_config,
        terrain,
        scene_lib,
        device,
        **simulator_extra_params,
    )

    return {
        "terrain": terrain,
        "scene_lib": scene_lib,
        "motion_lib": motion_lib,
        "simulator": simulator,
    }
