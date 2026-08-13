# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for building environment components.

This module provides helper functions to create terrain, scene_lib, motion_lib,
and simulator objects from their configs, reducing boilerplate in entry scripts.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict
import torch
from protomotions.components.motion_lib import MotionFileSwitchMode, resolve_shard_file
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
        scenes=scene_lib_config.inline_scenes,
        device=device,
        terrain=terrain,
        scene_weights=scene_weights,
    )


def build_motion_lib_from_config(
    motion_lib_config, device: torch.device, shard_cycle: int = 0
):
    """Build MotionLib from config.

    Always returns a MotionLib instance. If config.motion_file is None,
    creates an empty MotionLib (Null Object pattern).

    Note: Contact smoothing is NOT applied here - it's the Env's responsibility
    to modify the motion_lib during initialization based on its config.

    Args:
        motion_lib_config: MotionLibConfig (required, motion_file can be None for empty)
        device: PyTorch device
        shard_cycle: Distributed shard-selection cycle.

    Returns:
        MotionLib instance (empty if motion_file is None)
    """
    from protomotions.components.motion_lib import MotionLib

    return MotionLib(
        config=motion_lib_config, device=device, shard_cycle=shard_cycle
    )


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
    motion_shard_cycle: int = 0,
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
        motion_shard_cycle: Distributed shard-selection cycle.
        **simulator_extra_params: Simulator-specific params (e.g., simulation_app for IsaacLab)

    Returns:
        Dict with keys: terrain, scene_lib, motion_lib, simulator
    """
    # Create terrain (can be None)
    terrain = build_terrain_from_config(
        terrain_config, simulator_config.num_envs, device
    )

    # Select motions before any paired scene geometry or weight lookup.
    motion_lib = build_motion_lib_from_config(
        motion_lib_config, device, shard_cycle=motion_shard_cycle
    )

    scene_config = scene_lib_config
    scene_file = scene_lib_config.scene_file
    scene_is_sharded = scene_file is not None and "slurmrank" in Path(scene_file).name
    if scene_is_sharded:
        if motion_lib.motion_file_shard_index is None:
            raise ValueError("A sharded scene file requires a sharded motion file")
        if motion_lib_config.motion_file_switch_mode is MotionFileSwitchMode.LIVE:
            raise ValueError(
                "Live motion switching cannot replace simulator scene geometry; "
                "use restart mode"
            )
        scene_config = deepcopy(scene_lib_config)
        scene_config.scene_file = resolve_shard_file(
            scene_file, motion_lib.motion_file_shard_index
        )

    # Load motion weights from checkpoint to use as scene weights for prioritized sampling.
    scene_weights = None
    if save_dir and motion_lib.motion_file:
        from protomotions.envs.base_env.env import BaseEnv

        scene_weights = BaseEnv.apply_motion_weights_to_scene_weights(
            save_dir=save_dir, motion_file=motion_lib.motion_file, device=device
        )

    # Create scene_lib (always created, empty if scene_file is None)
    scene_lib = build_scene_lib_from_config(
        scene_config, simulator_config.num_envs, device, terrain, scene_weights
    )

    if motion_lib_config.motion_file_switch_mode is not MotionFileSwitchMode.FIXED:
        paired_motion_ids = scene_lib.get_humanoid_motion_ids()
        if paired_motion_ids is not None:
            if motion_lib_config.motion_file_switch_mode is MotionFileSwitchMode.LIVE:
                raise ValueError(
                    "Live motion switching is incompatible with motion-paired scenes; "
                    "use restart mode"
                )
            if not scene_is_sharded:
                raise ValueError(
                    "Restart motion switching requires a matching sharded scene file "
                    "for motion-paired scenes"
                )

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
