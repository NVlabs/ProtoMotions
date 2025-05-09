import torch

from protomotions.envs.base_env.components.base_component import BaseComponent

from protomotions.envs.mimic.mimic_utils import (
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
)


class MimicObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)

        if self.config.mimic_phase_obs.enabled:
            self.mimic_phase = torch.zeros(
                self.env.num_envs, 2, dtype=torch.float, device=self.env.device
            )
        else:
            self.mimic_phase = None

        if self.config.mimic_target_pose.enabled:
            self.mimic_target_poses = torch.zeros(
                self.env.num_envs,
                self.config.mimic_target_pose.num_future_steps
                * self.config.mimic_target_pose.num_obs_per_target_pose,
                dtype=torch.float,
                device=self.env.device,
            )
        else:
            self.mimic_target_poses = None

    def compute_observations(self, env_ids):
        if self.config.mimic_phase_obs.enabled:
            self.mimic_phase[env_ids] = self.get_phase_obs(
                self.env.motion_manager.motion_ids[env_ids],
                self.env.motion_manager.motion_times[env_ids],
            )

        if self.config.mimic_target_pose.enabled:
            self.mimic_target_poses[env_ids] = self.build_target_poses(
                self.config.mimic_target_pose.num_future_steps,
                self.config.mimic_target_pose.type,
                self.config.mimic_target_pose.with_time,
                env_ids,
            )

    def get_phase_obs(self, motion_ids, motion_times):
        phase = motion_times / self.env.motion_lib.get_motion_length(motion_ids)
        sin_phase = phase.sin().unsqueeze(-1)
        cos_phase = phase.cos().unsqueeze(-1)

        phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
        return phase_obs

    def _get_future_ref_states(self, env_ids, num_future_steps):
        time_offsets = (
            torch.arange(
                1, num_future_steps + 1, device=self.env.device, dtype=torch.long
            )
            * self.env.dt
        )

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

    def build_target_poses(
        self,
        num_future_steps,
        target_pose_type,
        with_time,
        env_ids,
    ):
        num_envs = env_ids.shape[0]
        ref_state = self._get_future_ref_states(env_ids, num_future_steps)

        flat_target_pos = ref_state.rigid_body_pos
        flat_target_rot = ref_state.rigid_body_rot

        current_state = self.env.simulator.get_bodies_state(env_ids)

        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        current_state.rigid_body_pos[:, :, -1:] -= (
            self.env.terrain.get_ground_heights(current_state.rigid_body_pos[:, 0]).view(num_envs, 1, 1).clone()
        )
        current_state.rigid_body_pos[..., :2] -= self.env.respawn_offset_relative_to_data.clone()[env_ids][
            ..., :2
        ].view(num_envs, 1, 2)

        if target_pose_type == "max-coords":
            target_pose_obs = build_max_coords_target_poses(
                cur_gt=current_state.rigid_body_pos,
                cur_gr=current_state.rigid_body_rot,
                flat_target_pos=flat_target_pos,
                flat_target_rot=flat_target_rot,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                w_last=True,
            )
        elif target_pose_type == "max-coords-future-rel":
            target_pose_obs = build_max_coords_target_poses_future_rel(
                cur_gt=current_state.rigid_body_pos,
                cur_gr=current_state.rigid_body_rot,
                flat_target_pos=flat_target_pos,
                flat_target_rot=flat_target_rot,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                w_last=True,
            )
        else:
            raise ValueError(f"Unknown target pose type '{target_pose_type}'")

        if with_time:
            target_pose_obs = self.add_time_to_target_poses(
                env_ids=env_ids,
                target_pose_obs=target_pose_obs,
                num_future_steps=num_future_steps,
            )

        return target_pose_obs

    def add_time_to_target_poses(self, env_ids, target_pose_obs, num_future_steps):
        num_envs = env_ids.shape[0]
        target_pose_obs = target_pose_obs.view(num_envs, num_future_steps, -1)

        time_offsets = (
            torch.arange(
                1, num_future_steps + 1, device=self.env.device, dtype=torch.long
            )
            * self.env.dt
        )

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

    def add_bodies_in_contact_to_target_poses(
        self, env_ids, num_future_steps, target_pose_obs, expected_contacts
    ):
        num_envs = env_ids.shape[0]
        target_pose_obs = target_pose_obs.view(num_envs, num_future_steps, -1)
        expected_contacts = expected_contacts.view(
            num_envs, num_future_steps, -1
        ).float()

        return torch.cat([target_pose_obs, expected_contacts], dim=-1).view(
            num_envs, -1
        )

    def get_obs(self):
        obs = {}
        if self.config.mimic_phase_obs.enabled:
            obs["mimic_phase"] = self.mimic_phase.clone()
        if self.config.mimic_target_pose.enabled:
            obs["mimic_target_poses"] = self.mimic_target_poses.clone()
        return obs
