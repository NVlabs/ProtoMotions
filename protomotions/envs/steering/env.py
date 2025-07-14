import numpy as np
import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from protomotions.envs.base_env.env import BaseEnv
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState

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
        ]
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


class Steering(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_speed_min = self.config.steering_params.tar_speed_min
        self._tar_speed_max = self.config.steering_params.tar_speed_max

        self._heading_change_steps_min = (
            self.config.steering_params.heading_change_steps_min
        )
        self._heading_change_steps_max = (
            self.config.steering_params.heading_change_steps_max
        )
        self._random_heading_probability = (
            self.config.steering_params.random_heading_probability
        )
        self._standard_heading_change = (
            self.config.steering_params.standard_heading_change
        )
        self._standard_speed_change = self.config.steering_params.standard_speed_change
        self._stop_probability = self.config.steering_params.stop_probability

        self.steering_obs = torch.zeros(
            (self.config.num_envs, self.config.steering_params.obs_size),
            device=device,
            dtype=torch.float,
        )

        self._heading_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._tar_dir_theta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._tar_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0

        self._tar_speed = torch.ones(
            [self.num_envs], device=self.device, dtype=torch.float
        )

        self.mimic_obs_cb = MimicObs(self.config, self)

    def create_visualization_markers(self):
        if self.config.headless:
            return {}

        visualization_markers = super().create_visualization_markers()
        
        steering_markers = []
        steering_markers.append(MarkerConfig(size="regular"))
        steering_markers_cfg = VisualizationMarker(
            type="arrow",
            color=(0.0, 1.0, 1.0),
            markers=steering_markers
        )
        visualization_markers["steering_markers"] = steering_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()

        marker_root_pos = self.simulator.get_root_state().root_pos
        marker_root_pos[..., 0:2] += self._tar_dir

        heading_axis = torch.zeros_like(marker_root_pos)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            self._tar_dir_theta, heading_axis, True
        )
        markers_state["steering_markers"] = MarkerState(
            translation=marker_root_pos.view(self.num_envs, -1, 3),
            orientation=marker_rot.view(self.num_envs, -1, 4),
        )

        return markers_state

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            self.reset_heading_task(env_ids)
        return super().reset(env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        self.check_update_task()

    def check_update_task(self):
        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_heading_task(rest_env_ids)

    def reset_heading_task(self, env_ids):
        n = len(env_ids)
        if np.random.binomial(1, self._random_heading_probability):
            dir_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(
                n, device=self.device
            ) + self._tar_speed_min
        else:
            dir_delta_theta = (
                2 * self._standard_heading_change * torch.rand(n, device=self.device)
                - self._standard_heading_change
            )
            # map tar_dir_theta back to [0, 2pi], add delta, project back into [0, 2pi] and then shift.
            dir_theta = (dir_delta_theta + self._tar_dir_theta[env_ids] + np.pi) % (
                2 * np.pi
            ) - np.pi

            speed_delta = (
                2 * self._standard_speed_change * torch.rand(n, device=self.device)
                - self._standard_speed_change
            )
            tar_speed = torch.clamp(
                speed_delta + self._tar_speed[env_ids],
                min=self._tar_speed_min,
                max=self._tar_speed_max,
            )

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        change_steps = torch.randint(
            low=self._heading_change_steps_min,
            high=self._heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        stop_probs = torch.ones(n, device=self.device) * self._stop_probability
        should_stop = torch.bernoulli(stop_probs)

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            root_states = self.simulator.get_root_state()
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
        else:
            root_states = self.simulator.get_root_state(env_ids)
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]

        obs = compute_heading_observations(root_states.root_rot, tar_dir, tar_speed)
        self.steering_obs[env_ids] = obs

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"steering": self.steering_obs})
        obs["motion_text_embeddings_mask"] = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        num_historical_conditioned_steps = getattr(self.config.historical_obs, 'num_historical_conditioned_steps', 15) if hasattr(self.config, 'historical_obs') else 15
        obs_size = self.self_obs_cb.config.obs_size
        obs["historical_pose_obs"] = torch.zeros(self.num_envs, num_historical_conditioned_steps * (obs_size + 1), dtype=torch.float, device=self.device)
        obs["masked_mimic_target_poses"] = torch.zeros(self.num_envs, 11 * 184, dtype=torch.float, device=self.device)
        obs["motion_text_embeddings"] = torch.zeros(self.num_envs, 512, dtype=torch.float, device=self.device)
        obs["masked_mimic_target_poses_masks"] = torch.zeros(self.num_envs, 11, dtype=torch.bool, device=self.device)
        obs["masked_mimic_target_bodies_masks"] = torch.zeros(self.num_envs, 10 * 14, dtype=torch.bool, device=self.device)
        # Add mimic_obs_cb keys (e.g. mimic_target_poses)
        obs.update(self.mimic_obs_cb.get_obs())
        return obs

    def compute_reward(self):
        root_pos = self.simulator.get_root_state().root_pos
        self.rew_buf[:] = compute_heading_reward(
            root_pos, self._prev_root_pos, self._tar_dir, self._tar_speed, self.dt
        )
        self._prev_root_pos[:] = root_pos


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_heading_observations(
    root_rot: Tensor, tar_dir: Tensor, tar_speed: Tensor
) -> Tensor:
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, True)

    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)
    local_tar_dir = local_tar_dir[..., 0:2]

    tar_speed = tar_speed.unsqueeze(-1)

    obs = torch.cat([local_tar_dir, tar_speed], dim=-1)
    return obs


@torch.jit.script
def compute_heading_reward(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    dt: float,
) -> Tensor:
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale
        * (
            tar_vel_err * tar_vel_err
            + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    speed_mask = tar_dir_speed < -0.5
    dir_reward[speed_mask] = 0

    return dir_reward
