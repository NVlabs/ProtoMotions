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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import replace

import torch
import numpy as np

if TYPE_CHECKING:
    from protomotions.components.terrains.config import TerrainSimConfig, TerrainConfig, CombineMode
    from protomotions.simulator.base_simulator.config import (
        FrictionDomainRandomizationConfig,
        SimulatorConfig,
    )


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
            # Concatenate along dim 0 (time dimension)
            # Each frame has shape [num_envs, ...], we take only env 0
            stacked = torch.stack(
                [frame[0:1] for frame in frame_list], dim=0
            ).squeeze(1)
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


def build_pd_action_offset_scale(
    hinge_axes_map, dof_limits_lower, dof_limits_upper, action_scale, device
):
    sorted_body_ids = list(hinge_axes_map.keys())
    sorted_body_ids.sort()

    lim_low = dof_limits_lower.cpu().numpy()
    lim_high = dof_limits_upper.cpu().numpy()

    dof_offset = 0

    for body_id in sorted_body_ids:
        dof_size = len(hinge_axes_map[body_id])

        if dof_size == 3:
            curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
            curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
            curr_low = np.max(np.abs(curr_low))
            curr_high = np.max(np.abs(curr_high))
            curr_scale = max([curr_low, curr_high])
            curr_scale = 2 * action_scale * curr_scale
            curr_scale = min([curr_scale, np.pi])

            lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
            lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

        elif dof_size == 1:
            curr_low = lim_low[dof_offset]
            curr_high = lim_high[dof_offset]
            curr_mid = 0.5 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = action_scale * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[dof_offset] = curr_low
            lim_high[dof_offset] = curr_high

        else:
            raise ValueError(f"Invalid dof size: {dof_size}")

        dof_offset += dof_size

    pd_action_offset = 0.5 * (lim_high + lim_low)
    pd_action_scale = 0.5 * (lim_high - lim_low)
    pd_action_offset = torch.tensor(pd_action_offset, device=device)
    pd_action_scale = torch.tensor(pd_action_scale, device=device)

    return pd_action_offset, pd_action_scale


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

    For MAX mode without DR, assumes simulator sets robot friction to epsilon.
    See _set_robot_friction_to_minimum() in newton/simulator.py.
    """
    from protomotions.components.terrains.config import CombineMode

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
    robot_range: Tuple[float, float],
    ground: float,
    mode: "CombineMode",
) -> Tuple[float, float]:
    """Compute effective friction range for robot+ground under a combine mode."""
    from protomotions.components.terrains.config import CombineMode

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
        # Set terrain to min of effective range so robot friction controls effective value
        # max(robot, terrain_min) = robot (since robot >= terrain_min)
        terrain_static = effective_static_range[0]
        terrain_dynamic = effective_dynamic_range[0]
        terrain_restitution = effective_restitution_range[0]

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
                static_friction_range=effective_static_range,
                dynamic_friction_range=effective_dynamic_range,
                restitution_range=effective_restitution_range,
            )
        else:
            adjusted_friction = None

        return adjusted_terrain, adjusted_friction

    elif target_mode == CombineMode.AVERAGE:
        # effective = (robot + ground) / 2  =>  robot = 2 * effective - ground
        ground_s = terrain_sim_config.static_friction
        ground_d = terrain_sim_config.dynamic_friction
        ground_r = terrain_sim_config.restitution
        adjusted_terrain = replace(terrain_sim_config, combine_mode=CombineMode.AVERAGE)
        adjusted_friction = None
        if friction_dr_config is not None and effective_static_range is not None:
            adjusted_friction = replace(
                friction_dr_config,
                static_friction_range=(
                    2 * effective_static_range[0] - ground_s,
                    2 * effective_static_range[1] - ground_s,
                ),
                dynamic_friction_range=(
                    2 * effective_dynamic_range[0] - ground_d,
                    2 * effective_dynamic_range[1] - ground_d,
                ),
                restitution_range=(
                    2 * effective_restitution_range[0] - ground_r,
                    2 * effective_restitution_range[1] - ground_r,
                ),
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
        terrain_config.sim_config, friction_dr_config, target_mode
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
