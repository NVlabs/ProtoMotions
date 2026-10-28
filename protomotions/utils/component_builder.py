# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for building environment components.

This module provides helper functions to create terrain, scene_lib, motion_lib,
and simulator objects from their configs, reducing boilerplate in entry scripts,
and :func:`build_env_from_configs`, the environment of a resolved-config bundle
as every entry point builds it.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict
import torch
from protomotions.components.motion_lib import MotionFileSwitchMode, resolve_shard_file
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
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
    motion_weights: Optional[list] = None,
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
        motion_weights=motion_weights,
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

    return MotionLib(config=motion_lib_config, device=device, shard_cycle=shard_cycle)


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


def apply_motion_weights_to_scene_weights(
    save_dir: Optional[str], motion_file: Optional[str], device: torch.device
) -> Optional[list]:
    """Apply motion weights from checkpoint as scene weights for curriculum learning.

    Loads motion weights from a previous training checkpoint and uses them as
    scene replication weights, allowing over-sampling of scenes corresponding to
    failed motions in curriculum learning. Called from :func:`build_all_components`
    only; it reads a checkpoint the env wrote (``env_<motion file>.ckpt``).

    The returned per-motion weights reach each scene by its humanoid_motion_id
    in SceneLib (scene_weights_by_clip_id), so no 1:1 scene-to-motion ordering
    is assumed.

    Args:
        save_dir: Directory where checkpoints are saved (or None)
        motion_file: Motion file path to identify checkpoint (or None)
        device: PyTorch device

    Returns:
        List of scene weights from motion training or None if not available

    Missing checkpoints or checkpoints without a motion manager have no
    curriculum to restore. Corrupt files and malformed manager state must
    raise rather than silently changing a resumed run to uniform sampling.
    """
    if not save_dir or not motion_file:
        return None

    evaluated_motions = Path(motion_file).name
    checkpoint_path = Path(save_dir) / f"env_{evaluated_motions}.ckpt"
    if not checkpoint_path.exists():
        return None

    print(f"Loading motion weights from checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if "motion_manager" not in checkpoint_data:
        print("No motion_manager found in checkpoint, using uniform scene weights.")
        return None

    motion_weights = checkpoint_data["motion_manager"]["motion_weights"]
    print(f"Loaded {len(motion_weights)} motion weights for curriculum sampling")
    return motion_weights.cpu().tolist()


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
    motion_weights = None
    if save_dir and motion_lib.motion_file:
        motion_weights = apply_motion_weights_to_scene_weights(
            save_dir=save_dir, motion_file=motion_lib.motion_file, device=device
        )

    # Create scene_lib (always created, empty if scene_file is None). Curriculum
    # weights reach each scene by clip id inside SceneLib.
    scene_lib = build_scene_lib_from_config(
        scene_config,
        simulator_config.num_envs,
        device,
        terrain,
        motion_weights=motion_weights,
    )

    # When the motion file is sharded, a motion-paired scene library needs a
    # matching sharded scene file; that holds in every switch mode, not only when
    # switching, so it sits after the switch-mode branch. A paired library on a
    # single unsharded motion file (ordinary scene-interaction training) builds.
    # The LIVE refusal for paired scenes is unconditional.
    if scene_lib.pairing.is_paired:
        if motion_lib_config.motion_file_switch_mode is MotionFileSwitchMode.LIVE:
            raise ValueError(
                "Live motion switching is incompatible with motion-paired scenes; "
                "use restart mode"
            )
        if motion_lib.motion_file_shard_index is not None and not scene_is_sharded:
            raise ValueError(
                "A sharded motion file with motion-paired scenes requires a "
                "matching sharded scene file"
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


def build_env_from_configs(
    configs,
    device: torch.device,
    motion_shard_cycle: int = 0,
    **simulator_extra_params,
):
    """Build the environment of a resolved-config bundle.

    The one path from ``configs`` to a constructed env, shared by training,
    inference, the examples and the tests: the friction conversion the
    simulator's combine mode needs, written back into ``configs.terrain`` and
    ``configs.simulator`` so a bundle saved afterwards describes what the env
    was built from; :func:`build_all_components`; the env class
    ``configs.env._target_`` names. The curriculum weights of a resumed run
    come from ``configs.env.save_dir``.

    Args:
        configs: ExperimentConfigs (``terrain``, ``scene_lib``, ``motion_lib``,
            ``simulator``, ``robot``, ``env``); ``terrain`` and ``simulator``
            are replaced by their converted forms.
        device: PyTorch device of the components and the env.
        motion_shard_cycle: Distributed shard-selection cycle (a resumed run's).
        **simulator_extra_params: Simulator-specific params (``simulation_app``
            for IsaacLab).

    Returns:
        The constructed env (the simulator initialises inside its constructor).
    """
    configs.terrain, configs.simulator = convert_friction_for_simulator(
        configs.terrain, configs.simulator
    )
    components = build_all_components(
        terrain_config=configs.terrain,
        scene_lib_config=configs.scene_lib,
        motion_lib_config=configs.motion_lib,
        simulator_config=configs.simulator,
        robot_config=configs.robot,
        device=device,
        save_dir=configs.env.save_dir,
        motion_shard_cycle=motion_shard_cycle,
        **simulator_extra_params,
    )
    return get_class(configs.env._target_)(
        config=configs.env,
        robot_config=configs.robot,
        device=device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )
