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
from protomotions.simulator.base_simulator.simulator_state import RobotState


def passes_exclude_motion_filter(
    motion: RobotState,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.2,
    duration_height_seconds: float = 1.0,
):
    """
    Filter function to exclude motions that don't meet quality criteria.

    Args:
        motion: RobotState motion object to check
        min_height_threshold: Minimum height threshold for any body part
        max_velocity_threshold: Maximum velocity threshold for any body part
        max_dof_vel_threshold: Maximum DOF velocity threshold
        duration_height_filter: Height threshold for duration filter
        duration_height_seconds: Duration in seconds for height filter

    Returns:
        bool: True if motion passes all filters, False otherwise
    """
    # Check if any global_translation has z smaller than min_height_threshold
    if motion.rigid_body_pos[..., 2].min() < min_height_threshold:
        print(
            f"Skipping because it has {motion.rigid_body_pos[..., 2].min()} smaller than {min_height_threshold}"
        )
        return False

    # Check if any global_velocity is too large (using finite difference)
    # not calling rigid_body_vel directly because it's smoothed by gaussian filter
    global_velocity_fin_diff = (
        motion.rigid_body_pos[1:] - motion.rigid_body_pos[:-1]
    ) * motion.fps
    if global_velocity_fin_diff.abs().max() > max_velocity_threshold:
        print(
            f"Skipping because it has {global_velocity_fin_diff.abs().max()} larger than {max_velocity_threshold}"
        )
        return False

    # Check if any dof_vels is too large
    if motion.dof_vel.abs().max() > max_dof_vel_threshold:
        print(
            f"Skipping because it has {motion.dof_vel.abs().max()} larger than {max_dof_vel_threshold}"
        )
        return False

    if duration_height_filter is not None:
        floor_estimate = motion.rigid_body_pos[..., 2].min()
        lowest_point_per_frame = motion.rigid_body_pos[..., 2].min(dim=-1).values
        lowest_distance_from_floor = lowest_point_per_frame - floor_estimate
        too_high_per_frame = lowest_distance_from_floor > duration_height_filter
        # check if too high for n consecutive frames
        consecutive_frames = int(duration_height_seconds * motion.fps)
        num_consecutive = 0
        for i in range(len(too_high_per_frame)):
            if too_high_per_frame[i]:
                num_consecutive += 1
            else:
                num_consecutive = 0

            if num_consecutive >= consecutive_frames:
                print(
                    f"Skipping because it has {num_consecutive} consecutive frames with height larger than {duration_height_filter}"
                )
                return False

    return True
