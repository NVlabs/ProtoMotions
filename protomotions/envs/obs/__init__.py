# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""Observation utilities and compute kernels for environments.

Contains pure tensor compute kernels for:
- Humanoid observations (max-coords, reduced-coords, historical)
- Mimic target poses
- Masked mimic observations
- Steering observations
- Path following observations
- State history buffers
- Observation noise utilities

Use MdpComponent in experiment configs to bind kernels to context paths.
"""

# Humanoid observation compute kernels
from protomotions.envs.obs.humanoid import (
    dof_to_local,
    dof_to_obs,
    obs_to_dof,
    root_projected_gravity,
    compute_local_ang_vel,
    compute_humanoid_reduced_coords_observations,
    compute_humanoid_max_coords_observations,
)

# Humanoid observation compute kernels (historical)
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_time_offsets,
    compute_historical_reduced_coords_from_state,
    compute_historical_max_coords_from_state,
    compute_historical_actions_from_state,
    compute_historical_poses_with_time,
    compute_historical_poses_with_time_reduced_coords,
    compute_historical_max_coords_from_motion_lib,
)

# Target pose building compute kernels
from protomotions.envs.obs.target_poses import (
    build_max_coords_target_poses_future_rel,
    build_max_coords_target_poses,
    build_reduced_coords_target_poses,
    build_deploy_target_poses,
    build_sparse_target_poses,
    build_target_root_rot,
    build_target_xy_offset,
    build_target_height,
    build_target_root_vel,
    build_target_root_ang_vel,
)

# Masked mimic observation compute kernels
from protomotions.envs.obs.masked_mimic import (
    compute_target_poses_only,
    compute_target_masks_only,
    compute_target_time_offsets,
)

# Steering observation compute kernel
from protomotions.envs.obs.steering import compute_steering_obs

# Path observation compute kernel
from protomotions.envs.obs.path import compute_path_obs

# Observation noise utilities
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    apply_observation_noise,
)


def to_float(x):
    """Convert a tensor to float dtype. Picklable alternative to lambda."""
    return x.float()


__all__ = [
    # Humanoid observation compute kernels
    "dof_to_local",
    "dof_to_obs",
    "obs_to_dof",
    "root_projected_gravity",
    "compute_local_ang_vel",
    "compute_humanoid_reduced_coords_observations",
    "compute_humanoid_max_coords_observations",
    # Humanoid historical observation compute kernels
    "compute_historical_time_offsets",
    "compute_historical_reduced_coords_from_state",
    "compute_historical_max_coords_from_state",
    "compute_historical_actions_from_state",
    "compute_historical_poses_with_time",
    "compute_historical_poses_with_time_reduced_coords",
    "compute_historical_max_coords_from_motion_lib",
    # Target pose building compute kernels
    "build_max_coords_target_poses_future_rel",
    "build_max_coords_target_poses",
    "build_reduced_coords_target_poses",
    "build_deploy_target_poses",
    "build_sparse_target_poses",
    "build_target_root_rot",
    "build_target_xy_offset",
    "build_target_height",
    "build_target_root_vel",
    "build_target_root_ang_vel",
    # Masked mimic observation compute kernels
    "compute_target_poses_only",
    "compute_target_masks_only",
    "compute_target_time_offsets",
    # Steering observation compute kernel
    "compute_steering_obs",
    # Path observation compute kernel
    "compute_path_obs",
    # Observation noise utilities
    "NoisyObservations",
    "apply_observation_noise",
    # Utilities
    "to_float",
]
