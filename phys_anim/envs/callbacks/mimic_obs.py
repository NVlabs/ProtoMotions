import torch

from phys_anim.envs.callbacks.base_callback import BaseCallback

from phys_anim.envs.mimic.mimic_utils import (
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
)


class MimicObs(BaseCallback):
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
                self.env.motion_ids[env_ids], self.env.motion_times[env_ids]
            )

        if self.config.mimic_target_pose.enabled:
            self.mimic_target_poses[env_ids] = self.build_target_poses(
                self.config.mimic_target_pose.num_future_steps,
                self.config.mimic_target_pose.type,
                self.config.mimic_target_pose.with_time,
                env_ids,
            )

    def get_phase_obs(self, motion_ids, motion_times):
        phase = (
            motion_times - self.env.motion_lib.state.motion_timings[motion_ids, 0]
        ) / self.env.motion_lib.get_sub_motion_length(motion_ids)
        sin_phase = phase.sin().unsqueeze(-1)
        cos_phase = phase.cos().unsqueeze(-1)

        phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
        return phase_obs

    def build_target_poses(
        self, num_future_steps, target_pose_type, with_time, env_ids
    ):
        num_envs = env_ids.shape[0]
        time_offsets = (
            torch.arange(
                1, num_future_steps + 1, device=self.env.device, dtype=torch.long
            )
            * self.env.dt
        )

        raw_future_times = self.env.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        motion_ids = (
            self.env.motion_ids[env_ids].unsqueeze(-1).tile([1, num_future_steps])
        )
        flat_ids = motion_ids.view(-1)

        lengths = self.env.motion_lib.get_motion_length(flat_ids)
        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.env.motion_lib.get_motion_state(flat_ids, flat_times)
        flat_target_pos = ref_state.rb_pos
        flat_target_rot = ref_state.rb_rot

        current_state = self.env.get_bodies_state()
        cur_gt, cur_gr = (
            current_state.body_pos[env_ids],
            current_state.body_rot[env_ids],
        )

        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.env.terrain_obs_cb.ground_heights[env_ids].view(
            num_envs, 1, 1
        )
        cur_gt[..., :2] -= self.env.respawn_offset_relative_to_data.clone()[env_ids][
            ..., :2
        ].view(num_envs, 1, 2)

        if target_pose_type == "max-coords":
            target_pose_obs = build_max_coords_target_poses(
                cur_gt=cur_gt,
                cur_gr=cur_gr,
                flat_target_pos=flat_target_pos,
                flat_target_rot=flat_target_rot,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                w_last=self.env.w_last,
            )
        elif target_pose_type == "max-coords-future-rel":
            target_pose_obs = build_max_coords_target_poses_future_rel(
                cur_gt=cur_gt,
                cur_gr=cur_gr,
                flat_target_pos=flat_target_pos,
                flat_target_rot=flat_target_rot,
                num_envs=num_envs,
                num_future_steps=num_future_steps,
                w_last=self.env.w_last,
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

        raw_future_times = self.env.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        motion_ids = (
            self.env.motion_ids[env_ids].unsqueeze(-1).tile([1, num_future_steps])
        )
        flat_ids = motion_ids.view(-1)

        lengths = self.env.motion_lib.get_motion_length(flat_ids)

        times = torch.minimum(raw_future_times.view(-1), lengths).view(
            num_envs, num_future_steps, 1
        ) - self.env.motion_times[env_ids].view(num_envs, 1, 1)

        obs = torch.cat([target_pose_obs, times], dim=-1).view(num_envs, -1)

        return obs
