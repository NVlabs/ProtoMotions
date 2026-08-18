# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import replace

import torch

if TYPE_CHECKING:
    from protomotions.components.terrains.config import (
        TerrainSimConfig,
        TerrainConfig,
        CombineMode,
    )
    from protomotions.simulator.base_simulator.config import (
        FrictionDomainRandomizationConfig,
        SimulatorConfig,
    )


def get_friction_bucket_count(friction_dr: Dict[str, Any]) -> int:
    """Return the number of rows in the first configured friction table."""
    for key in ("static_friction", "dynamic_friction", "restitution"):
        table = friction_dr.get(key)
        if table is not None:
            return table.shape[0]
    return 0


def get_friction_table(friction_dr: Dict[str, Any]) -> Optional[torch.Tensor]:
    """Return the table used for simulators with one friction coefficient."""
    static_friction = friction_dr.get("static_friction")
    if static_friction is not None:
        return static_friction
    return friction_dr.get("dynamic_friction")


def build_motion_data(
    recorded_motion: Dict[str, List[torch.Tensor]],
    fps: int,
    num_dof: int = 0,
) -> Dict[str, Any]:
    """
    Build a .motion file compatible data dictionary from recorded motion.

    The .motion format uses RobotState field names directly (rigid_body_pos,
    rigid_body_rot, etc.) and is loaded by MotionLib via RobotState.from_dict().

    Args:
        recorded_motion: Dictionary containing lists of tensors for each field.
            Expected keys: gts, grs, gvs, gavs, dps, dvs, contacts
            (using MotionLib naming convention)
        fps: Frames per second of the recorded motion.
        num_dof: Number of DOFs for the robot (used to create placeholder tensors
            if dof data wasn't recorded).

    Returns:
        Dictionary compatible with .motion format (RobotState field names + fps).
    """
    from protomotions.simulator.base_simulator.simulator_state import StateConversion

    # Mapping from recorded field names to RobotState field names
    field_mapping = {
        "gts": "rigid_body_pos",
        "grs": "rigid_body_rot",
        "gvs": "rigid_body_vel",
        "gavs": "rigid_body_ang_vel",
        "dps": "dof_pos",
        "dvs": "dof_vel",
        "contacts": "rigid_body_contacts",
    }

    motion_data: Dict[str, Any] = {
        "fps": fps,
        "state_conversion": StateConversion.COMMON,
    }

    # Concatenate recorded frames for each field
    for field_name, frame_list in recorded_motion.items():
        if len(frame_list) > 0 and field_name in field_mapping:
            # Each frame is already single-env: [num_bodies, 3] or [num_dofs]
            stacked = torch.stack(frame_list, dim=0)
            motion_data[field_mapping[field_name]] = stacked

    num_frames = motion_data["rigid_body_pos"].shape[0]
    num_bodies = motion_data["rigid_body_pos"].shape[1]
    device = motion_data["rigid_body_pos"].device
    dtype = motion_data["rigid_body_pos"].dtype

    # Ensure all required fields exist (create zero tensors for missing fields)
    if "rigid_body_vel" not in motion_data:
        motion_data["rigid_body_vel"] = torch.zeros(
            (num_frames, num_bodies, 3), device=device, dtype=dtype
        )
    if "rigid_body_ang_vel" not in motion_data:
        motion_data["rigid_body_ang_vel"] = torch.zeros(
            (num_frames, num_bodies, 3), device=device, dtype=dtype
        )
    if "dof_pos" not in motion_data:
        motion_data["dof_pos"] = torch.zeros(
            (num_frames, num_dof), device=device, dtype=dtype
        )
    if "dof_vel" not in motion_data:
        motion_data["dof_vel"] = torch.zeros(
            (num_frames, num_dof), device=device, dtype=dtype
        )
    if "rigid_body_contacts" not in motion_data:
        motion_data["rigid_body_contacts"] = torch.zeros(
            (num_frames, num_bodies), device=device, dtype=torch.bool
        )
    else:
        # Ensure contacts are boolean type
        motion_data["rigid_body_contacts"] = motion_data["rigid_body_contacts"].bool()

    return motion_data


def convert_friction_for_combine_mode(
    terrain_sim_config: "TerrainSimConfig",
    friction_dr_config: Optional["FrictionDomainRandomizationConfig"],
    target_mode: "CombineMode",
    tolerance: float = 1e-6,
    default_robot_friction: float = 1.0,
    default_robot_restitution: float = 0.0,
) -> Tuple["TerrainSimConfig", Optional["FrictionDomainRandomizationConfig"]]:
    """Convert friction configs between combine modes preserving effective friction.

    PhysX uses AVERAGE: effective = (robot + terrain) / 2
    MuJoCo uses MAX: effective = max(robot, terrain)

    For MAX mode without DR, assumes the simulator applies
    ``default_robot_friction`` to character shapes.
    """

    source_mode = terrain_sim_config.combine_mode
    if source_mode == target_mode:
        return terrain_sim_config, friction_dr_config

    ground_static = terrain_sim_config.static_friction
    ground_dynamic = terrain_sim_config.dynamic_friction
    ground_restitution = terrain_sim_config.restitution

    # Compute effective values - either from DR ranges or from default robot friction
    if friction_dr_config is not None:
        expected_static = _compute_effective_friction_range(
            friction_dr_config.static_friction_range, ground_static, source_mode
        )
        expected_dynamic = _compute_effective_friction_range(
            friction_dr_config.dynamic_friction_range, ground_dynamic, source_mode
        )
        expected_restitution = _compute_effective_friction_range(
            friction_dr_config.restitution_range, ground_restitution, source_mode
        )
    else:
        # No DR: compute effective using default robot values (PhysX defaults)
        robot_friction = (default_robot_friction, default_robot_friction)
        robot_restitution = (default_robot_restitution, default_robot_restitution)
        expected_static = _compute_effective_friction_range(
            robot_friction, ground_static, source_mode
        )
        expected_dynamic = _compute_effective_friction_range(
            robot_friction, ground_dynamic, source_mode
        )
        expected_restitution = _compute_effective_friction_range(
            robot_restitution, ground_restitution, source_mode
        )

    adjusted_terrain, adjusted_friction = _convert_material_to_combine_mode(
        terrain_sim_config,
        friction_dr_config,
        expected_static,
        expected_dynamic,
        expected_restitution,
        target_mode,
    )

    # Verify conversion preserves effective values (only when DR is present)
    if adjusted_friction is not None and friction_dr_config is not None:
        actual_static = _compute_effective_friction_range(
            adjusted_friction.static_friction_range,
            adjusted_terrain.static_friction,
            target_mode,
        )
        actual_dynamic = _compute_effective_friction_range(
            adjusted_friction.dynamic_friction_range,
            adjusted_terrain.dynamic_friction,
            target_mode,
        )
        actual_restitution = _compute_effective_friction_range(
            adjusted_friction.restitution_range,
            adjusted_terrain.restitution,
            target_mode,
        )

        for name, expected, actual in [
            ("static friction", expected_static, actual_static),
            ("dynamic friction", expected_dynamic, actual_dynamic),
            ("restitution", expected_restitution, actual_restitution),
        ]:
            if expected is None:
                if actual is not None:
                    raise ValueError(
                        f"Conversion failed: {name} was not configured but produced a range."
                    )
                continue
            if actual is None:
                raise ValueError(
                    f"Conversion failed: {name} configured range was dropped."
                )
            if not _friction_ranges_match(expected, actual, tolerance):
                raise ValueError(
                    f"Conversion failed: {name} effective range mismatch. "
                    f"Expected {expected}, got {actual}"
                )

    print(
        f"[INFO] Material conversion {source_mode.value} -> {target_mode.value}: "
        f"ground friction {ground_static:.2f} -> {adjusted_terrain.static_friction:.2f} , ground restitution {ground_restitution:.2f} -> {adjusted_terrain.restitution:.2f}"
    )

    return adjusted_terrain, adjusted_friction


def _friction_ranges_match(
    expected: Tuple[float, float], actual: Tuple[float, float], tol: float
) -> bool:
    """Check if two friction ranges match within tolerance."""
    return abs(expected[0] - actual[0]) < tol and abs(expected[1] - actual[1]) < tol


def _compute_effective_friction_range(
    robot_range: Optional[Tuple[float, float]],
    ground: float,
    mode: "CombineMode",
) -> Optional[Tuple[float, float]]:
    """Compute effective friction range for robot+ground under a combine mode."""
    from protomotions.components.terrains.config import CombineMode

    if robot_range is None:
        return None

    r_min, r_max = robot_range

    if mode == CombineMode.AVERAGE:
        return ((r_min + ground) / 2, (r_max + ground) / 2)
    elif mode == CombineMode.MIN:
        return (min(r_min, ground), min(r_max, ground))
    elif mode == CombineMode.MAX:
        return (max(r_min, ground), max(r_max, ground))
    elif mode == CombineMode.MULTIPLY:
        return (r_min * ground, r_max * ground)
    else:
        raise ValueError(f"Unknown combine mode: {mode}")


def _convert_effective_range_to_target(
    effective_range: Optional[Tuple[float, float]],
    ground: float,
    target_mode: "CombineMode",
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Map one effective range to a simulator-supported combine mode.

    For an omitted range, leave the terrain value unchanged and keep it
    omitted. Configured ranges are represented exactly in the target mode.
    """
    from protomotions.components.terrains.config import CombineMode

    if effective_range is None:
        return ground, None

    effective_min, effective_max = effective_range
    if target_mode == CombineMode.MAX:
        return effective_min, (effective_min, effective_max)

    if target_mode == CombineMode.AVERAGE:
        # (robot + ground) / 2 = effective, and robot friction must be >= 0.
        target_ground = min(ground, 2 * effective_min)
        return target_ground, (
            2 * effective_min - target_ground,
            2 * effective_max - target_ground,
        )

    raise ValueError(f"Unsupported target mode: {target_mode.value}")


def _convert_material_to_combine_mode(
    terrain_sim_config: "TerrainSimConfig",
    friction_dr_config: Optional["FrictionDomainRandomizationConfig"],
    effective_static_range: Optional[Tuple[float, float]],
    effective_dynamic_range: Optional[Tuple[float, float]],
    effective_restitution_range: Optional[Tuple[float, float]],
    target_mode: "CombineMode",
) -> Tuple["TerrainSimConfig", Optional["FrictionDomainRandomizationConfig"]]:
    """Convert effective friction/restitution ranges to work with target combine mode."""
    from protomotions.components.terrains.config import CombineMode

    if target_mode == CombineMode.MAX:
        terrain_static, static_range = _convert_effective_range_to_target(
            effective_static_range,
            terrain_sim_config.static_friction,
            target_mode,
        )
        terrain_dynamic, dynamic_range = _convert_effective_range_to_target(
            effective_dynamic_range,
            terrain_sim_config.dynamic_friction,
            target_mode,
        )
        terrain_restitution, restitution_range = _convert_effective_range_to_target(
            effective_restitution_range,
            terrain_sim_config.restitution,
            target_mode,
        )

        adjusted_terrain = replace(
            terrain_sim_config,
            static_friction=terrain_static,
            dynamic_friction=terrain_dynamic,
            restitution=terrain_restitution,
            combine_mode=CombineMode.MAX,
        )

        if friction_dr_config is not None:
            adjusted_friction = replace(
                friction_dr_config,
                static_friction_range=static_range,
                dynamic_friction_range=dynamic_range,
                restitution_range=restitution_range,
            )
        else:
            adjusted_friction = None

        return adjusted_terrain, adjusted_friction

    elif target_mode == CombineMode.AVERAGE:
        terrain_static, static_range = _convert_effective_range_to_target(
            effective_static_range,
            terrain_sim_config.static_friction,
            target_mode,
        )
        terrain_dynamic, dynamic_range = _convert_effective_range_to_target(
            effective_dynamic_range,
            terrain_sim_config.dynamic_friction,
            target_mode,
        )
        terrain_restitution, restitution_range = _convert_effective_range_to_target(
            effective_restitution_range,
            terrain_sim_config.restitution,
            target_mode,
        )
        adjusted_terrain = replace(
            terrain_sim_config,
            static_friction=terrain_static,
            dynamic_friction=terrain_dynamic,
            restitution=terrain_restitution,
            combine_mode=CombineMode.AVERAGE,
        )
        adjusted_friction = None
        if friction_dr_config is not None:
            adjusted_friction = replace(
                friction_dr_config,
                static_friction_range=static_range,
                dynamic_friction_range=dynamic_range,
                restitution_range=restitution_range,
            )
        return adjusted_terrain, adjusted_friction

    raise ValueError(f"Unsupported target mode: {target_mode.value}")


def get_simulator_friction_combine_mode(simulator_name: str) -> Optional["CombineMode"]:
    """Return the fixed friction combine mode for a simulator, or None if configurable."""
    from protomotions.components.terrains.config import CombineMode

    if simulator_name == "newton":
        return CombineMode.MAX
    if simulator_name == "isaacgym":
        return CombineMode.AVERAGE
    return None


def convert_friction_for_simulator(
    terrain_config: "TerrainConfig",
    simulator_config: "SimulatorConfig",
) -> Tuple["TerrainConfig", "SimulatorConfig"]:
    """Convert friction configs if simulator requires a specific combine mode."""
    from dataclasses import replace

    simulator_name = simulator_config._target_.split(".")[-3]
    target_mode = get_simulator_friction_combine_mode(simulator_name)

    if target_mode is None:
        return terrain_config, simulator_config
    if terrain_config is None or terrain_config.sim_config is None:
        return terrain_config, simulator_config
    if terrain_config.sim_config.combine_mode == target_mode:
        return terrain_config, simulator_config

    friction_dr_config = None
    if (
        simulator_config.domain_randomization is not None
        and simulator_config.domain_randomization.friction is not None
    ):
        friction_dr_config = simulator_config.domain_randomization.friction

    adjusted_sim_config, adjusted_friction = convert_friction_for_combine_mode(
        terrain_config.sim_config,
        friction_dr_config,
        target_mode,
        default_robot_friction=getattr(simulator_config, "default_robot_friction", 1.0),
    )

    adjusted_terrain = replace(terrain_config, sim_config=adjusted_sim_config)
    adjusted_simulator = simulator_config
    if adjusted_friction is not None:
        adjusted_simulator = replace(
            simulator_config,
            domain_randomization=replace(
                simulator_config.domain_randomization, friction=adjusted_friction
            ),
        )

    return adjusted_terrain, adjusted_simulator
