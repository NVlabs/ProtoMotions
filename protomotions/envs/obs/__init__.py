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
"""Observation utilities and functions for environments.

Contains:
- Observation factories for humanoid, mimic, steering, path tasks
- State history buffers
- Humanoid observation computation utilities
- Target pose building functions
- Observation noise utilities
"""

from protomotions.envs.obs.general import (
    passthrough,
    passthrough_float,
    passthrough_factory,
    passthrough_float_factory,
)

from protomotions.envs.obs.humanoid import (
    dof_to_local,
    dof_to_obs,
    obs_to_dof,
    root_projected_gravity,
    compute_local_ang_vel,
    compute_humanoid_reduced_coords_observations,
    compute_humanoid_max_coords_observations,
)
from protomotions.envs.obs.target_poses import (
    build_max_coords_target_poses_future_rel,
    build_max_coords_target_poses,
    build_reduced_coords_target_poses,
    build_sparse_target_poses,
    build_target_root_rot,
    build_target_xy_offset,
    build_target_height,
    build_target_root_vel,
    build_target_root_ang_vel,
)
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    apply_observation_noise,
)
from protomotions.envs.obs.humanoid_obs_functions import (
    max_coords_obs_factory,
    reduced_coords_obs_factory,
    historical_reduced_coords_obs_factory,
    historical_max_coords_obs_factory,
    historical_actions_factory,
    historical_poses_with_time_factory,
    historical_poses_with_time_reduced_coords_factory,
    previous_actions_factory,
    historical_max_coords_ref_obs_factory,
)
from protomotions.envs.obs.mimic_obs_functions import (
    mimic_target_poses_max_coords_factory,
    mimic_target_poses_max_coords_future_rel_factory,
    mimic_target_poses_simple_factory,
    mimic_target_poses_reduced_coords_factory,
    target_dof_pos_factory,
    target_dof_vel_factory,
    target_root_rot_factory,
    target_xy_offset_factory,
    target_height_factory,
    target_root_vel_factory,
    target_root_ang_vel_factory,
)
from protomotions.envs.obs.masked_mimic_obs_functions import (
    masked_mimic_target_poses_factory,
    target_masks_factory,
    target_time_offsets_factory,
)
from protomotions.envs.obs.prior_obs_functions import prior_historical_obs_factory
from protomotions.envs.obs.steering_obs_functions import steering_obs_factory
from protomotions.envs.obs.path_obs_functions import path_obs_factory

__all__ = [
    # General utilities
    "passthrough",
    "passthrough_float",
    # General factories
    "passthrough_factory",
    "passthrough_float_factory",
    # Humanoid observation factories
    "max_coords_obs_factory",
    "reduced_coords_obs_factory",
    "previous_actions_factory",
    "historical_reduced_coords_obs_factory",
    "historical_max_coords_obs_factory",
    "historical_actions_factory",
    "historical_poses_with_time_factory",
    "historical_poses_with_time_reduced_coords_factory",
    # Humanoid observation utilities
    "dof_to_local",
    "dof_to_obs",
    "obs_to_dof",
    "root_projected_gravity",
    "compute_local_ang_vel",
    "compute_humanoid_reduced_coords_observations",
    "compute_humanoid_max_coords_observations",
    # Mimic observation factories
    "mimic_target_poses_max_coords_factory",
    "mimic_target_poses_max_coords_future_rel_factory",
    "mimic_target_poses_simple_factory",
    "mimic_target_poses_reduced_coords_factory",
    # Individual target component factories
    "target_dof_pos_factory",
    "target_dof_vel_factory",
    "target_root_rot_factory",
    "target_xy_offset_factory",
    "target_height_factory",
    "target_root_vel_factory",
    "target_root_ang_vel_factory",
    # Target pose building functions
    "build_max_coords_target_poses_future_rel",
    "build_max_coords_target_poses",
    "build_reduced_coords_target_poses",
    "build_sparse_target_poses",
    "build_target_root_rot",
    "build_target_xy_offset",
    "build_target_height",
    "build_target_root_vel",
    "build_target_root_ang_vel",
    # Observation noise
    "NoisyObservations",
    "apply_observation_noise",
    # Masked Mimic
    "masked_mimic_target_poses_factory",
    "target_masks_factory",
    "target_time_offsets_factory",
    # Prior
    "prior_historical_obs_factory",
    # AMP Reference Observations
    "historical_max_coords_ref_obs_factory",
    # Other
    "steering_obs_factory",
    "path_obs_factory",
]
