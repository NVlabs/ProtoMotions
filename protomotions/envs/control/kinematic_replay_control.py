# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kinematic replay control - plays reference motions without physics."""

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.simulator.base_simulator.simulator_state import ResetState

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class KinematicReplayControlConfig(ControlComponentConfig):
    _target_: str = "protomotions.envs.control.kinematic_replay_control.KinematicReplayControl"


class KinematicReplayControl(ControlComponent):
    """Plays reference motions kinematically (bypasses physics)."""
    
    config: KinematicReplayControlConfig
    
    def __init__(self, config: KinematicReplayControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
    
    def reset(self, env_ids: Tensor):
        pass
    
    def step(self):
        all_env_ids = torch.arange(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )

        # Honour user-requested reset (e.g. R key). This control component zeros
        # progress_buf/reset_buf below to avoid double-resetting on done clips,
        # which would otherwise swallow the env.user_reset() → max-length path.
        # Intercept the flag here instead: re-sample motions (respecting fixed
        # IDs) and restart every env from t=0.
        if self.env.consume_reset_request():
            self.env.motion_manager.sample_motions(all_env_ids)
            self.env.motion_manager.motion_times[:] = 0.0

        # Advance motion time
        sync_motion_dt = self.env.simulator.decimation * 1.0 / self.env.simulator.config.sim.fps
        self.env.motion_manager.motion_times += sync_motion_dt

        # Handle done clips
        done_clip = self.env.motion_manager.get_done_tracks()
        if any(done_clip):
            done_env_ids = torch.where(done_clip)[0]
            self.env.motion_manager.sample_motions(done_env_ids)
        
        # Get reference state
        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids,
            self.env.motion_manager.motion_times,
        )
        
        # Zero velocities for kinematic replay
        ref_state.dof_vel *= 0
        ref_state.rigid_body_vel *= 0
        ref_state.rigid_body_ang_vel *= 0
        ref_reset_state = ResetState.from_robot_state(ref_state)
        
        env_ids = all_env_ids

        # Get object state
        ref_object_state = self.env.scene_lib.get_scene_pose(
            env_ids,
            self.env.motion_manager.motion_times,
            self.env.config.ref_object_respawn_offset,
        )
        ref_object_state.root_vel = torch.zeros_like(ref_object_state.root_pos)
        ref_object_state.root_ang_vel = torch.zeros_like(ref_object_state.root_pos)
        
        # Apply terrain offset
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_reset_state.root_pos[:, None, :], env_ids
        ).squeeze(1)
        ref_reset_state.root_pos += offset
        
        if self.env.scene_lib.num_scenes() > 0:
            ref_object_state.root_pos += offset.unsqueeze(1)
        
        # Set robot state directly
        self.env.simulator.reset_envs(ref_reset_state, ref_object_state, env_ids)
        
        # Prevent double reset
        self.env.progress_buf[env_ids] = 0
        self.env.reset_buf[env_ids] = 0
        self.env.terminate_buf[env_ids] = 0
    
    def populate_context(self, ctx: EnvContext) -> None:
        """Kinematic replay doesn't add any context variables."""
        pass
