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
"""Humanoid observation component.

This module handles the computation of standard humanoid observations, including:
- Max coordinate observations (root-relative body positions/rotations)
- Reduced coordinate observations (joint angles/velocities)
- Action history
"""

import torch
from torch import Tensor, nn

from protomotions.envs.utils.humanoid import (
    compute_humanoid_reduced_coords_observations,
    compute_humanoid_max_coords_observations,
    root_projected_gravity,
)
from protomotions.envs.obs.config import HumanoidObsConfig
from protomotions.simulator.base_simulator.simulator_state import RobotState


class HistoryBuffer(nn.Module):
    """Circular buffer for storing temporal history of observations or actions.

    Stores the past N frames of data where index 0 is the most recent frame.
    Used for temporal observations like action history or historical poses.

    Args:
        num_steps: Number of historical timesteps to store.
        num_envs: Number of parallel environments.
        shape: Shape of each data element (default: scalar).
        dtype: Data type for storage.
        device: Device for tensor storage.
    """

    data: Tensor

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        shape: tuple = (),
        dtype=torch.float,
        device="cpu",
    ):
        super().__init__()
        data = torch.zeros(num_steps, num_envs, *shape, dtype=dtype, device=device)
        self.register_buffer("data", data, persistent=False)
        self.to(device)

    def rotate(self):
        """Shift history forward by one timestep (oldest frame is discarded)."""
        self.data = self.data.roll(
            shifts=1, dims=0
        )  # equivalent to self.data[i + 1] = self.data[i]

    @torch.no_grad()
    def update(self, fresh_data: Tensor):
        """Rotate buffer and update current frame with new data.

        Args:
            fresh_data: New data to insert at current frame [num_envs, *shape]
        """
        self.rotate()
        self.set_curr(fresh_data)

    @torch.no_grad()
    def set_all(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set all timesteps for specified environments.

        Args:
            fresh_data: Data for all timesteps [num_steps, num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[:, env_ids] = fresh_data

    @torch.no_grad()
    def set_hist(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set historical data (excluding current frame).

        Args:
            fresh_data: Historical data [num_steps-1, num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[1:, env_ids] = fresh_data

    @torch.no_grad()
    def set_curr(self, fresh_data: Tensor, env_ids=slice(None)):
        """Set current frame data.

        Args:
            fresh_data: Current frame data [num_envs, *shape]
            env_ids: Environment indices to update
        """
        self.data[0, env_ids] = fresh_data

    def get_hist(self, env_ids=slice(None)):
        """Get historical data (excluding current frame).

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Historical data [num_steps-1, num_envs, *shape]
        """
        return self.data[1:, env_ids]

    def get_current(self, env_ids=slice(None)):
        """Get current frame data.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Current frame data [num_envs, *shape]
        """
        return self.data[0, env_ids]

    def get_all(self, env_ids=slice(None)):
        """Get all timesteps.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            All historical data [num_steps, num_envs, *shape]
        """
        return self.data[:, env_ids]

    def get_all_flattened(self, env_ids=slice(None)):
        """Get all timesteps flattened into a single feature vector per environment.

        Args:
            env_ids: Environment indices to retrieve

        Returns:
            Flattened history [num_envs, num_steps * features]
        """
        data = self.get_all(env_ids)
        num_envs = data.shape[1]
        return data.permute(1, 0, 2).reshape(num_envs, -1)

    def get_index(self, idx: int, env_ids=slice(None)):
        """Get data at specific timestep index.

        Args:
            idx: Timestep index (0 = current, 1 = previous, etc.)
            env_ids: Environment indices to retrieve

        Returns:
            Data at specified timestep [num_envs, *shape]
        """
        return self.data[idx, env_ids]

    @property
    def device(self) -> torch.device:
        """Get device from registered buffers."""
        return self.data.device


class HumanoidObs:
    """Handles computation of humanoid state observations.

    Manages history buffers and computes current observations for the agent.

    Args:
        config: Configuration for humanoid observations.
        env: Parent environment instance.
    """

    def __init__(self, config: HumanoidObsConfig, env):
        self.config = config
        self.env = env

        self.humanoid_max_coords_obs = None
        self.humanoid_max_coords_obs_hist_buf = None

        self.humanoid_reduced_coords_obs = None
        self.humanoid_reduced_coords_obs_hist_buf = None
        
        self.previous_actions_hist_buf = None
        if self.config.action_history.enabled:
            self.previous_actions_hist_buf = HistoryBuffer(
                self.config.action_history.num_historical_steps,
                self.env.num_envs,
                shape=(self.env.robot_config.number_of_actions,),
                device=self.env.device,
            )
            
        self._initialized = False

    def post_physics_step(self):
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.env.device))

        if self.config.max_coords_obs.enabled:
            self.humanoid_max_coords_obs_hist_buf.rotate()

        if self.config.reduced_coords_obs.enabled:
            self.humanoid_reduced_coords_obs_hist_buf.rotate()

        if self.config.action_history.enabled:
            self.previous_actions_hist_buf.rotate()

    def reset_hist_buf(self, env_ids, default_mask, motion_ids, motion_times):
        """Reset history buffers for given environments.

        Args:
            env_ids: All environment IDs being reset
            default_mask: Boolean mask indicating which envs used default reset
            motion_ids: Motion IDs for ref-reset envs (can be None if all default)
            motion_times: Motion times for ref-reset envs (can be None if all default)
        """
        default_env_ids = env_ids[default_mask]
        ref_mask = ~default_mask
        ref_env_ids = env_ids[ref_mask]

        if len(default_env_ids) > 0:
            self.reset_hist_default(default_env_ids)

        if len(ref_env_ids) > 0:
            # motion_ids and motion_times are already filtered to only ref envs in reset()
            self.reset_hist_ref(ref_env_ids, motion_ids, motion_times)

    def reset_hist_default(self, env_ids):
        if not self._initialized:
            self.compute_observations(env_ids)

        if (
            self.config.max_coords_obs.enabled
            and self.config.max_coords_obs.num_historical_steps > 1
        ):
            self.humanoid_max_coords_obs_hist_buf.set_hist(
                self.humanoid_max_coords_obs_hist_buf.get_current(env_ids),
                env_ids=env_ids,
            )

        if (
            self.config.reduced_coords_obs.enabled
            and self.config.reduced_coords_obs.num_historical_steps > 1
        ):
            self.humanoid_reduced_coords_obs_hist_buf.set_hist(
                self.humanoid_reduced_coords_obs_hist_buf.get_current(env_ids),
                env_ids=env_ids,
            )

    def reset_hist_ref(self, env_ids, motion_ids, motion_times):
        if not self._initialized:
            self.compute_observations(env_ids)

        if (
            self.config.max_coords_obs.enabled
            and self.config.max_coords_obs.num_historical_steps > 1
        ):
            obs_ref = self._get_ref_obs_historical(
                motion_ids,
                motion_times,
                self.config.max_coords_obs.num_historical_steps - 1,
                max_coords=True,
                exclude_current=True,
            )
            self.humanoid_max_coords_obs_hist_buf.set_hist(
                obs_ref.view(
                    len(env_ids),
                    self.config.max_coords_obs.num_historical_steps - 1,
                    -1,
                ).permute(1, 0, 2),
                env_ids,
            )

        if (
            self.config.reduced_coords_obs.enabled
            and self.config.reduced_coords_obs.num_historical_steps > 1
        ):
            obs_ref = self._get_ref_obs_historical(
                motion_ids,
                motion_times,
                self.config.reduced_coords_obs.num_historical_steps - 1,
                max_coords=False,
                exclude_current=True,
            )
            self.humanoid_reduced_coords_obs_hist_buf.set_hist(
                obs_ref.view(
                    len(env_ids),
                    self.config.reduced_coords_obs.num_historical_steps - 1,
                    -1,
                ).permute(1, 0, 2),
                env_ids,
            )

    def compute_observations(self, env_ids):
        self._initialized = True

        current_state: RobotState = self.env.simulator.get_robot_state(env_ids)
        body_contacts = current_state.rigid_body_contacts[
            :, self.env.contact_body_ids
        ].bool()  # [num_envs, num_contact_bodies]

        ground_heights = self.env.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).clone()

        if self.config.max_coords_obs.enabled:
            obs = compute_humanoid_max_coords_observations(
                body_pos=current_state.rigid_body_pos,
                body_rot=current_state.rigid_body_rot,
                body_vel=current_state.rigid_body_vel,
                body_ang_vel=current_state.rigid_body_ang_vel,
                ground_height=ground_heights,
                body_contacts=body_contacts,
                local_obs=self.config.max_coords_obs.local_obs,
                root_height_obs=self.config.max_coords_obs.root_height_obs,
                observe_contacts=self.config.max_coords_obs.observe_contacts,
                w_last=True,
            )
            
            if self.humanoid_max_coords_obs is None:
                self.humanoid_max_coords_obs = torch.zeros(
                    self.env.num_envs,
                    obs.shape[-1],
                    dtype=torch.float,
                    device=self.env.device,
                )
                self.humanoid_max_coords_obs_hist_buf = HistoryBuffer(
                    self.config.max_coords_obs.num_historical_steps,
                    self.env.num_envs,
                    shape=(obs.shape[-1],),
                    device=self.env.device,
                )

            self.humanoid_max_coords_obs[env_ids] = obs
            self.humanoid_max_coords_obs_hist_buf.set_curr(obs, env_ids)

        if self.config.reduced_coords_obs.enabled:
            obs = compute_humanoid_reduced_coords_observations(
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                root_ang_vel=current_state.rigid_body_ang_vel[:, 0, :],
                root_projected_gravity=root_projected_gravity(
                    current_state.root_rot, w_last=True
                ),
                hinge_axes_map=self.env.robot_config.kinematic_info.hinge_axes_map,
                w_last=True,
            )
            
            if self.humanoid_reduced_coords_obs is None:
                self.humanoid_reduced_coords_obs = torch.zeros(
                    self.env.num_envs,
                    obs.shape[-1],
                    dtype=torch.float,
                    device=self.env.device,
                )
                self.humanoid_reduced_coords_obs_hist_buf = HistoryBuffer(
                    self.config.reduced_coords_obs.num_historical_steps,
                    self.env.num_envs,
                    shape=(obs.shape[-1],),
                    device=self.env.device,
                )

            self.humanoid_reduced_coords_obs[env_ids] = obs
            self.humanoid_reduced_coords_obs_hist_buf.set_curr(obs, env_ids)

        if self.config.action_history.enabled:
            self.previous_actions_hist_buf.set_curr(
                self.env.simulator.get_previous_actions(env_ids), env_ids
            )

    def build_self_obs_demo(self, motion_ids: Tensor, motion_times0: Tensor):
        obs_demo = {}

        if self.config.max_coords_obs.enabled:
            obs_demo["historical_max_coords_obs"] = self._get_ref_obs_historical(
                motion_ids,
                motion_times0,
                self.config.max_coords_obs.num_historical_steps,
                max_coords=True,
                exclude_current=False,
            )

        if self.config.reduced_coords_obs.enabled:
            obs_demo["historical_reduced_coords_obs"] = self._get_ref_obs_historical(
                motion_ids,
                motion_times0,
                self.config.reduced_coords_obs.num_historical_steps,
                max_coords=False,
                exclude_current=False,
            )

        return obs_demo

    def _get_ref_obs_historical(
        self,
        motion_ids: Tensor,
        motion_times0: Tensor,
        num_historical_steps: int,
        max_coords: bool,
        exclude_current: bool = False,
    ) -> Tensor:
        """Sample historical reference observations from motion library.

        Args:
            motion_ids: Motion IDs to sample from [num_envs]
            motion_times0: Base motion times [num_envs]
            num_historical_steps: Number of historical timesteps to sample
            max_coords: Whether to compute max_coords or reduced_coords observations
            exclude_current: If True, sample from t=-dt to t=-(num_steps*dt) (for reset).
                           If False, sample from t=0 to t=-(num_steps-1)*dt (for demo).

        Returns:
            Historical observations [num_envs * num_historical_steps, obs_size]
        """
        dt = self.env.dt

        motion_ids_expanded = torch.tile(
            motion_ids.unsqueeze(-1), [1, num_historical_steps]
        )
        motion_times_expanded = motion_times0.unsqueeze(-1)

        if exclude_current:
            # Sample from t=-dt to t=-(num_steps*dt) for reset
            time_steps = -dt * (
                torch.arange(0, num_historical_steps, device=self.env.device) + 1
            )
        else:
            # Sample from t=0 to t=-(num_steps-1)*dt for demo
            time_steps = -dt * torch.arange(
                0, num_historical_steps, device=self.env.device
            )

        motion_times_expanded = motion_times_expanded + time_steps

        motion_ids_expanded = motion_ids_expanded.view(-1)
        lengths = self.env.motion_lib.motion_lengths[motion_ids_expanded]
        motion_times_expanded = (
            motion_times_expanded.view(-1).clamp(min=0).clamp(max=lengths)
        )

        ref_state = self.env.motion_lib.get_motion_state(
            motion_ids_expanded, motion_times_expanded
        )
        return self._compute_ref_obs(ref_state, max_coords=max_coords)

    def _compute_ref_obs(self, ref_state: RobotState, max_coords: bool) -> Tensor:
        """Compute reference observations from a RobotState.

        This is used both for building demo observations and historical observations.

        Args:
            ref_state: Reference robot state
            max_coords: Whether to compute max_coords or reduced_coords observations

        Returns:
            Observations [batch_size, obs_size]
        """
        if max_coords:
            # Get contact binary flags from reference motion
            if ref_state.rigid_body_contacts is not None:
                body_contacts = ref_state.rigid_body_contacts[
                    :, self.env.contact_body_ids
                ]
            else:
                # Fallback to zeros if contact data not available
                body_contacts = torch.zeros(
                    ref_state.rigid_body_pos.shape[0],
                    len(self.env.contact_body_ids),
                    dtype=torch.bool,
                    device=self.env.device,
                )

            obs_ref = compute_humanoid_max_coords_observations(
                body_pos=ref_state.rigid_body_pos,
                body_rot=ref_state.rigid_body_rot,
                body_vel=ref_state.rigid_body_vel,
                body_ang_vel=ref_state.rigid_body_ang_vel,
                ground_height=torch.zeros(
                    ref_state.rigid_body_pos.shape[0], 1, device=self.env.device
                ),
                body_contacts=body_contacts,
                local_obs=self.config.max_coords_obs.local_obs,
                root_height_obs=self.config.max_coords_obs.root_height_obs,
                observe_contacts=self.config.max_coords_obs.observe_contacts,
                w_last=True,
            )
        else:
            obs_ref = compute_humanoid_reduced_coords_observations(
                dof_pos=ref_state.dof_pos,
                dof_vel=ref_state.dof_vel,
                root_ang_vel=ref_state.rigid_body_ang_vel[:, 0, :],
                root_projected_gravity=root_projected_gravity(
                    ref_state.root_rot, w_last=True
                ),
                hinge_axes_map=self.env.robot_config.kinematic_info.hinge_axes_map,
                w_last=True,
            )
        return obs_ref

    def get_obs(self):
        obs = {}
        
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.env.device))
        
        if self.config.action_history.enabled:
            obs["historical_previous_actions"] = (
                self.previous_actions_hist_buf.get_all_flattened().clone()
            )

        if self.config.max_coords_obs.enabled:
            obs["max_coords_obs"] = self.humanoid_max_coords_obs.clone()
            obs["historical_max_coords_obs"] = (
                self.humanoid_max_coords_obs_hist_buf.get_all_flattened().clone()
            )

        if self.config.reduced_coords_obs.enabled:
            obs["reduced_coords_obs"] = self.humanoid_reduced_coords_obs.clone()
            obs["historical_reduced_coords_obs"] = (
                self.humanoid_reduced_coords_obs_hist_buf.get_all_flattened().clone()
            )

        return obs
