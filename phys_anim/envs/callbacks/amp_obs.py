import torch
from torch import Tensor

from phys_anim.envs.humanoid.humanoid_utils import build_disc_observations
from phys_anim.envs.env_utils.general import HistoryBuffer
from phys_anim.envs.callbacks.base_callback import BaseCallback


class AmpObs(BaseCallback):
    def __init__(self, config, env):
        super().__init__(config, env)
        print(config)
        if not self.config.disable_discriminator:
            self.discriminator_obs_historical_steps = (
                self.config.discriminator_obs_historical_steps
            )
            assert self.discriminator_obs_historical_steps >= 2

            self.discriminator_obs_size_per_step = (
                self.config.discriminator_obs_size_per_step
            )

            self.disc_hist_buf = HistoryBuffer(
                self.discriminator_obs_historical_steps,
                self.env.num_envs,
                shape=(self.discriminator_obs_size_per_step,),
                device=self.env.device,
            )

    def post_physics_step(self):
        if not self.config.disable_discriminator:
            self.disc_hist_buf.rotate()
            self.compute_disc_observations()

    def reset_envs(self, env_ids):
        if self.config.disable_discriminator or len(env_ids) == 0:
            return

        self.reset_disc_hist_buf(env_ids)

    def reset_disc_hist_buf(self, env_ids):
        self.compute_disc_observations(env_ids)

        if len(self.env.reset_default_env_ids) > 0:
            self.reset_disc_hist_default(self.env.reset_default_env_ids)

        if len(self.env.reset_ref_env_ids) > 0:
            self.reset_disc_hist_ref(
                self.env.reset_ref_env_ids,
                self.env.reset_ref_motion_ids,
                self.env.reset_ref_motion_times,
            )

    def reset_disc_hist_default(self, env_ids):
        self.disc_hist_buf.set_hist(
            self.disc_hist_buf.get_current(env_ids), env_ids=env_ids
        )

    def reset_disc_hist_ref(self, env_ids, motion_ids, motion_times):
        dt = self.env.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.discriminator_obs_historical_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(
                0, self.discriminator_obs_historical_steps - 1, device=self.env.device
            )
            + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1).clamp(min=0)

        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_observations(
            ref_state.root_pos,
            ref_state.root_rot,
            ref_state.root_vel,
            ref_state.root_ang_vel,
            ref_state.dof_pos,
            ref_state.dof_vel,
            ref_state.key_body_pos,
            torch.zeros(len(motion_ids), 1, device=self.env.device),
            self.env.config.humanoid_obs.local_root_obs,
            self.env.config.humanoid_obs.root_height_obs,
            self.env.dof_obs_size,
            self.env.get_dof_offsets(),
            False,
            self.env.w_last,
        )
        self.disc_hist_buf.set_hist(
            disc_obs_demo.view(
                len(env_ids), self.discriminator_obs_historical_steps - 1, -1
            ).permute(1, 0, 2),
            env_ids,
        )
        
    def make_disc_obs(self):
        return self.disc_hist_buf.get_all_flattened()

    def build_disc_obs_demo(self, motion_ids: Tensor, motion_times0: Tensor):
        dt = self.env.dt

        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.discriminator_obs_historical_steps]
        )
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(
            0, self.discriminator_obs_historical_steps, device=self.env.device
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        # motion ids above are "sub_motions" so first we map to motion file itself and then extract the length.
        lengths = self.env.motion_lib.state.motion_lengths[
            self.env.motion_lib.state.sub_motion_to_motion[motion_ids]
        ]

        assert torch.all(motion_times >= 0)
        assert torch.all(motion_times <= lengths)

        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        disc_obs_demo = build_disc_observations(
            ref_state.root_pos,
            ref_state.root_rot,
            ref_state.root_vel,
            ref_state.root_ang_vel,
            ref_state.dof_pos,
            ref_state.dof_vel,
            ref_state.key_body_pos,
            torch.zeros(len(motion_ids), 1, device=self.env.device),
            self.env.config.humanoid_obs.local_root_obs,
            self.env.config.humanoid_obs.root_height_obs,
            self.env.dof_obs_size,
            self.env.get_dof_offsets(),
            False,
            self.env.w_last,
        )
        return disc_obs_demo

    def compute_disc_observations(self, env_ids=None):
        current_state = self.env.get_bodies_state()

        dof_pos, dof_vel = self.env.get_dof_state()
        key_body_pos = current_state.body_pos[:, self.env.key_body_ids, :]

        if env_ids is None:
            disc_obs = build_disc_observations(
                current_state.body_pos[:, 0, :],
                current_state.body_rot[:, 0, :],
                current_state.body_vel[:, 0, :],
                current_state.body_ang_vel[:, 0, :],
                dof_pos,
                dof_vel,
                key_body_pos,
                self.env.terrain_obs_cb.ground_heights,
                self.env.config.humanoid_obs.local_root_obs,
                self.env.config.humanoid_obs.root_height_obs,
                self.env.dof_obs_size,
                self.env.get_dof_offsets(),
                False,
                self.env.w_last,
            )
            self.disc_hist_buf.set_curr(disc_obs)
        else:
            disc_obs = build_disc_observations(
                current_state.body_pos[env_ids, 0, :],
                current_state.body_rot[env_ids, 0, :],
                current_state.body_vel[env_ids, 0, :],
                current_state.body_ang_vel[env_ids, 0, :],
                dof_pos[env_ids],
                dof_vel[env_ids],
                key_body_pos[env_ids],
                self.env.terrain_obs_cb.ground_heights[env_ids],
                self.env.config.humanoid_obs.local_root_obs,
                self.env.config.humanoid_obs.root_height_obs,
                self.env.dof_obs_size,
                self.env.get_dof_offsets(),
                False,
                self.env.w_last,
            )
            self.disc_hist_buf.set_curr(disc_obs, env_ids)
