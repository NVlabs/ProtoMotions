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
"""Mimic observation component.

Computes observations related to motion tracking, including:
- Phase information (sin/cos of motion phase)
- Time remaining in motion
- Target poses (future/reference poses)
"""

import torch

from protomotions.envs.utils.target_poses import (
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_max_coords_target_poses_simple,
)
from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.envs.obs.config import MimicObsConfig, FuturePoseType
from protomotions.simulator.base_simulator.simulator_state import RobotState

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from protomotions.envs.mimic.env import Mimic
else:
    Mimic = object


class MimicObs:
    """Handles computation of motion tracking observations.

    Args:
        config: Configuration for mimic observations.
        env: Parent Mimic environment.
    """

    env: Mimic

    def __init__(self, config: MimicObsConfig, env: Mimic):
        self.config = config
        self.env = env

        if self.config.mimic_phase_obs.enabled:
            self.mimic_phase = torch.zeros(
                self.env.num_envs, 2, dtype=torch.float, device=self.env.device
            )
        else:
            self.mimic_phase = None

        if self.config.mimic_time_left_obs.enabled:
            self.mimic_time_left = torch.zeros(
                self.env.num_envs, 1, dtype=torch.float, device=self.env.device
            )
        else:
            self.mimic_time_left = None

        self.mimic_target_poses = None
        if self.config.mimic_target_pose.enabled:
            self.num_future_steps = self.config.mimic_target_pose.num_future_steps
            if isinstance(self.config.mimic_target_pose.future_steps, int):
                self.future_steps = torch.arange(
                    1,
                    self.num_future_steps + 1,
                    device=self.env.device,
                    dtype=torch.long,
                )
            else:
                self.future_steps = torch.tensor(
                    self.config.mimic_target_pose.future_steps,
                    device=self.env.device,
                    dtype=torch.long,
                )

            self.mimic_target_poses = None
            
        self._initialized = False

    def compute_observations(self, env_ids: torch.Tensor):
        """Compute mimic-specific observations for given environments.

        Args:
            env_ids: Environment indices to update
        """
        self._initialized = True
        
        if self.config.mimic_phase_obs.enabled:
            self.mimic_phase[env_ids] = self.get_phase_obs(
                self.env.motion_manager.motion_ids[env_ids],
                self.env.motion_manager.motion_times[env_ids],
            )

        if self.config.mimic_time_left_obs.enabled:
            self.mimic_time_left[env_ids] = (
                self.env.motion_lib.get_motion_length(
                    self.env.motion_manager.motion_ids[env_ids]
                )
                - self.env.motion_manager.motion_times[env_ids]
            ).unsqueeze(-1)

        if self.config.mimic_target_pose.enabled:
            target_poses = self._build_target_poses(
                env_ids=env_ids,
            )
            if self.mimic_target_poses is None:
                self.mimic_target_poses = torch.zeros(
                    self.env.num_envs,
                    target_poses.shape[-1],
                    dtype=torch.float,
                    device=self.env.device,
                )
            self.mimic_target_poses[env_ids] = target_poses

    def get_phase_obs(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> torch.Tensor:
        """Compute phase observations as (sin, cos) of normalized motion time.

        Args:
            motion_ids: Motion indices [batch]
            motion_times: Current times in motion [batch]

        Returns:
            Phase observations [batch, 2] with (sin(phase), cos(phase))
        """
        phase = motion_times / self.env.motion_lib.get_motion_length(motion_ids)
        sin_phase = phase.sin().unsqueeze(-1)
        cos_phase = phase.cos().unsqueeze(-1)

        phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
        return phase_obs

    def _get_future_ref_states(
        self, env_ids: torch.Tensor, future_steps: torch.Tensor
    ) -> RobotState:
        """Sample future reference motion states at specified timestep offsets.

        Args:
            env_ids: Environment indices [batch]
            future_steps: Timestep offsets to sample [num_future_steps]

        Returns:
            RobotState containing future reference states
        """
        time_offsets = future_steps * self.env.dt
        num_future_steps = len(future_steps)

        raw_future_times = self.env.motion_manager.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        motion_ids = (
            self.env.motion_manager.motion_ids[env_ids]
            .unsqueeze(-1)
            .tile([1, num_future_steps])
        )
        flat_ids = motion_ids.view(-1)

        lengths = self.env.motion_lib.get_motion_length(flat_ids)
        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.env.motion_lib.get_motion_state(flat_ids, flat_times)

        return ref_state

    def _build_target_poses(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Build target pose observations by sampling future reference states.

        Args:
            env_ids: Environment indices to build poses for

        Returns:
            Target pose observations [batch, features]
        """
        num_envs = env_ids.shape[0]

        num_future_steps = len(self.future_steps)
        ref_state = self._get_future_ref_states(env_ids, self.future_steps)
        ref_state_gt = ref_state.rigid_body_pos.reshape(
            num_envs, num_future_steps, -1, 3
        )

        # Flatten future steps into batch dimension for terrain correction
        ref_state_gt_flat = ref_state_gt.reshape(num_envs * num_future_steps, -1, 3)
        env_ids_repeated = env_ids.unsqueeze(1).expand(-1, num_future_steps).reshape(-1)
        # Apply consistent terrain height correction across all bodies (same as spawning logic)
        ref_state_gt_flat += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                ref_state_gt_flat, env_ids_repeated
            )
        )
        ref_state_gt = (
            ref_state_gt_flat  # Shape: (num_envs * num_future_steps, num_bodies, 3)
        )

        ref_state_rigid_body_contacts = ref_state.rigid_body_contacts[
            :, self.env.contact_body_ids
        ]

        current_state = self.env.simulator.get_bodies_state(env_ids)
        
        target_pose_type = self.config.mimic_target_pose.type
        with_velocities = self.config.mimic_target_pose.with_velocities

        return self.build_target_poses(
            target_pose_type,
            with_velocities,
            current_state.rigid_body_pos,
            current_state.rigid_body_rot,
            current_state.rigid_body_vel,
            current_state.rigid_body_ang_vel,
            ref_state_gt,
            ref_state.rigid_body_rot,
            ref_state.rigid_body_vel,
            ref_state.rigid_body_ang_vel,
            ref_state_rigid_body_contacts,
            env_ids,
            num_future_steps,
        )

    def build_target_poses(
        self,
        target_pose_type: FuturePoseType,
        with_velocities: bool,
        current_state_gt: torch.Tensor,
        current_state_gr: torch.Tensor,
        current_state_rigid_body_vel: torch.Tensor,
        current_state_rigid_body_ang_vel: torch.Tensor,
        ref_state_gt: torch.Tensor,
        ref_state_gr: torch.Tensor,
        ref_state_rigid_body_vel: torch.Tensor,
        ref_state_rigid_body_ang_vel: torch.Tensor,
        ref_state_rigid_body_contacts: torch.Tensor,
        env_ids: torch.Tensor,
        num_future_steps: int,
    ) -> torch.Tensor:
        """Build target pose observations using specified encoding type.

        Dispatches to appropriate builder function based on target_pose_type configuration.

        Args:
            target_pose_type: Encoding type (MAX_COORDS, MAX_COORDS_FUTURE_REL, MAX_COORDS_SIMPLE)
            with_velocities: Whether to include velocity information
            current_state_gt: Current body positions [batch, bodies, 3]
            current_state_gr: Current body rotations [batch, bodies, 4]
            current_state_rigid_body_vel: Current body velocities [batch, bodies, 3]
            current_state_rigid_body_ang_vel: Current body angular velocities [batch, bodies, 3]
            ref_state_gt: Reference body positions [batch*num_future_steps, bodies, 3]
            ref_state_gr: Reference body rotations [batch*num_future_steps, bodies, 4]
            ref_state_rigid_body_vel: Reference body velocities [batch*num_future_steps, bodies, 3]
            ref_state_rigid_body_ang_vel: Reference angular velocities [batch*num_future_steps, bodies, 3]
            ref_state_rigid_body_contacts: Reference contact states
            env_ids: Environment indices
            num_future_steps: Number of future timesteps encoded

        Returns:
            Target pose observations [batch, features]
        """
        num_envs = env_ids.shape[0]
        if target_pose_type == FuturePoseType.MAX_COORDS:
            target_pose_obs = build_max_coords_target_poses(
                cur_gt=current_state_gt,
                cur_gr=current_state_gr,
                flat_target_pos=ref_state_gt,
                flat_target_rot=ref_state_gr,
                cur_vel=current_state_rigid_body_vel,
                cur_ang_vel=current_state_rigid_body_ang_vel,
                flat_target_vel=ref_state_rigid_body_vel,
                flat_target_ang_vel=ref_state_rigid_body_ang_vel,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                with_velocities=with_velocities,
                w_last=True,
            )
        elif target_pose_type == FuturePoseType.MAX_COORDS_FUTURE_REL:
            target_pose_obs = build_max_coords_target_poses_future_rel(
                cur_gt=current_state_gt,
                cur_gr=current_state_gr,
                cur_vel=current_state_rigid_body_vel,
                cur_ang_vel=current_state_rigid_body_ang_vel,
                flat_target_pos=ref_state_gt,
                flat_target_rot=ref_state_gr,
                flat_target_vel=ref_state_rigid_body_vel,
                flat_target_ang_vel=ref_state_rigid_body_ang_vel,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                with_velocities=with_velocities,
                w_last=True,
            )
        elif target_pose_type == FuturePoseType.MAX_COORDS_SIMPLE:
            target_pose_obs = build_max_coords_target_poses_simple(
                cur_gt=current_state_gt,
                cur_gr=current_state_gr,
                flat_target_pos=ref_state_gt,
                flat_target_rot=ref_state_gr,
                flat_target_vel=ref_state_rigid_body_vel,
                flat_target_ang_vel=ref_state_rigid_body_ang_vel,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                with_velocities=with_velocities,
                w_last=True,
            )
        else:
            raise ValueError(f"Unknown target pose type '{target_pose_type}'")

        if self.config.mimic_target_pose.with_time:
            target_pose_obs = self.add_time_to_target_poses(
                env_ids=env_ids,
                target_pose_obs=target_pose_obs,
                future_steps=self.future_steps,
            )

        if self.config.mimic_target_pose.with_contacts:
            target_pose_obs = self.add_contacts_to_target_poses(
                env_ids=env_ids,
                flat_target_contacts=ref_state_rigid_body_contacts,
                target_pose_obs=target_pose_obs,
                num_future_steps=num_future_steps,
            )
        return target_pose_obs

    def add_time_to_target_poses(
        self,
        env_ids: torch.Tensor,
        target_pose_obs: torch.Tensor,
        future_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Append time-to-target information to pose observations.

        Args:
            env_ids: Environment indices [batch]
            target_pose_obs: Base pose observations [batch, features]
            future_steps: Timestep offsets [num_future_steps]

        Returns:
            Pose observations with time appended [batch, features + num_future_steps]
        """
        num_envs = env_ids.shape[0]
        num_future_steps = len(future_steps)
        target_pose_obs = target_pose_obs.view(num_envs, num_future_steps, -1)

        time_offsets = future_steps * self.env.dt

        raw_future_times = self.env.motion_manager.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        motion_ids = (
            self.env.motion_manager.motion_ids[env_ids]
            .unsqueeze(-1)
            .tile([1, num_future_steps])
        )
        flat_ids = motion_ids.view(-1)

        lengths = self.env.motion_lib.get_motion_length(flat_ids)

        times = torch.minimum(raw_future_times.view(-1), lengths).view(
            num_envs, num_future_steps, 1
        ) - self.env.motion_manager.motion_times[env_ids].view(num_envs, 1, 1)

        obs = torch.cat([target_pose_obs, times], dim=-1).view(num_envs, -1)

        return obs

    def add_contacts_to_target_poses(
        self,
        env_ids: torch.Tensor,
        flat_target_contacts: torch.Tensor,
        target_pose_obs: torch.Tensor,
        num_future_steps: int,
    ) -> torch.Tensor:
        """Append expected contact states to pose observations.

        Args:
            env_ids: Environment indices [batch]
            flat_target_contacts: Flattened contact flags [batch*num_future_steps, contact_bodies]
            target_pose_obs: Base pose observations [batch, features]
            num_future_steps: Number of future timesteps

        Returns:
            Pose observations with contacts appended [batch, features + contacts]
        """
        num_envs = env_ids.shape[0]
        target_pose_obs = target_pose_obs.view(num_envs, num_future_steps, -1)
        expected_contacts = flat_target_contacts.view(
            num_envs, num_future_steps, -1
        ).float()

        return torch.cat([target_pose_obs, expected_contacts], dim=-1).view(
            num_envs, -1
        )

    def get_obs(self) -> Dict[str, torch.Tensor]:
        """Get mimic observation dictionary.

        Returns:
            Dictionary with enabled mimic observations (phase, time_left, target_poses)
        """
        obs = {}
        
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.env.device))
            
        if self.config.mimic_phase_obs.enabled:
            obs["mimic_phase"] = self.mimic_phase.clone()
        if self.config.mimic_time_left_obs.enabled:
            obs["mimic_time_left"] = self.mimic_time_left.clone()
        if self.config.mimic_target_pose.enabled:
            obs["mimic_target_poses"] = self.mimic_target_poses.clone()
        return obs
