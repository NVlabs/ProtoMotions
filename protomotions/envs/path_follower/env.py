from isaac_utils import torch_utils, rotations

import torch
from torch import Tensor
from protomotions.envs.path_follower.path_generator import PathGenerator
from protomotions.envs.base_env.env import BaseEnv
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState

# --- Begin copied MimicObs ---
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
# --- End copied MimicObs ---

class PathFollowing(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        head_body_name = self.config.robot.head_body_name
        self.head_body_id = self.simulator.build_body_ids_tensor([head_body_name]).item()

        self._num_traj_samples = self.config.path_follower_params.num_traj_samples
        self._traj_sample_timestep = (
            self.config.path_follower_params.traj_sample_timestep
        )

        self.path_obs = torch.zeros(
            (
                self.config.num_envs,
                self.config.path_follower_params.path_obs_size,
            ),
            device=device,
            dtype=torch.float,
        )

        self._fail_dist = 4.0
        self._fail_height_dist = 0.5

        self.mimic_obs_cb = MimicObs(self.config, self)

        self.build_path_generator()

    def create_visualization_markers(self):
        if self.config.headless:
            return {}

        visualization_markers = super().create_visualization_markers()

        path_markers = []
        for i in range(self.config.path_follower_params.num_traj_samples):
            path_markers.append(MarkerConfig(size="regular"))
        path_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(1.0, 0.0, 0.0),
            markers=path_markers
        )
        visualization_markers["path_markers"] = path_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()

        traj_samples = self.fetch_path_samples().clone()
        if not self.config.path_follower_params.height_conditioned:
            traj_samples[..., 2] = 0.8  # CT hack

        ground_below_marker = self.terrain.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        traj_samples = traj_samples.view(self.num_envs, -1, 3)
        markers_state["path_markers"] = MarkerState(
            translation=traj_samples,
            orientation=torch.zeros(
                self.num_envs, traj_samples.shape[1], 4, device=self.device
            ),
        )

        return markers_state

    ###############################################################
    # Handle resets
    ###############################################################
    def reset(self, env_ids=None):
        obs = super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        bodies_positions = self.simulator.get_bodies_state(env_ids).rigid_body_pos
        head_position = bodies_positions[:, self.head_body_id]

        if len(env_ids) > 0:
            flat_reset_head_position = head_position.view(-1, 3)
            ground_below_reset_head = self.terrain.get_ground_heights(
                head_position[..., :2]
            )
            flat_reset_head_position[..., 2] -= ground_below_reset_head.view(-1)
            self.path_generator.reset(env_ids, flat_reset_head_position)

        return obs

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        bodies_positions = self.simulator.get_bodies_state(env_ids).rigid_body_pos
        root_states = self.simulator.get_root_state(env_ids)
        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position = bodies_positions[:, self.head_body_id, :]

        traj_samples = self.fetch_path_samples(env_ids)

        flat_head_position = head_position.view(-1, 3)
        flat_head_position[..., 2] -= ground_below_head.view(-1)

        obs = compute_path_observations(
            root_states.root_rot,
            flat_head_position,
            traj_samples,
            self.config.path_follower_params.height_conditioned,
        )

        self.path_obs[env_ids] = obs

    def get_obs(self):
        obs = super().get_obs()
        obs.update({"path": self.path_obs})
        # Add masked mimic keys
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
        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos
        head_position = bodies_positions[:, self.head_body_id, :]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position[..., 2] -= ground_below_head.view(-1)

        self.rew_buf[:] = compute_path_reward(
            head_position, tar_pos, self.config.path_follower_params.height_conditioned
        )

    def compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos
        bodies_contact_buf = self.self_obs_cb.body_contacts.clone()

        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.non_termination_contact_body_ids,
            bodies_positions,
            tar_pos,
            self.config.max_episode_length,
            self._fail_dist,
            self._fail_height_dist,
            self.config.enable_height_termination,
            self.config.path_follower_params.enable_path_termination,
            self.config.path_follower_params.height_conditioned,
            self.termination_heights
            + self.terrain.get_ground_heights(
                bodies_positions[:, self.head_body_id, :2]
            ),
            self.head_body_id,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def build_path_generator(self):
        episode_dur = self.config.max_episode_length * self.dt
        self.path_generator = PathGenerator(
            self.config.path_follower_params.path_generator,
            self.device,
            self.num_envs,
            episode_dur,
            self.config.path_follower_params.height_conditioned,
        )

    def fetch_path_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(
            self._num_traj_samples, device=self.device, dtype=torch.float
        )
        timesteps = timesteps * self._traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

        return traj_samples


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_path_observations(
    root_rot: Tensor,
    head_states: Tensor,
    traj_samples: Tensor,
    height_conditioned: bool,
) -> Tensor:
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, True)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_states.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    local_traj_pos = rotations.quat_rotate(
        heading_rot_exp, traj_samples_delta_flat, True
    )
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs


@torch.jit.script
def compute_path_reward(head_pos, tar_pos, height_conditioned):
    # type: (Tensor, Tensor, bool) -> Tensor
    pos_err_scale = 2.0
    height_err_scale = 10.0

    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        reward = (pos_reward + height_reward) * 0.5
    else:
        reward = pos_reward

    return reward


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    non_termination_contact_body_ids,
    rigid_body_pos,
    tar_pos,
    max_episode_length,
    fail_dist,
    fail_height_dist,
    enable_early_termination,
    enable_path_termination,
    enable_height_termination,
    termination_heights,
    head_body_id,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, bool, bool, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, non_termination_contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, non_termination_contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
    else:
        has_fallen = progress_buf < -1

    if enable_path_termination:
        head_pos = rigid_body_pos[..., head_body_id, :]
        tar_delta = tar_pos - head_pos
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_overall_fail = tar_dist_sq > fail_dist * fail_dist

        if enable_height_termination:
            tar_height = tar_pos[..., 2]
            height_delta = tar_height - head_pos[..., 2]
            tar_head_dist_sq = height_delta * height_delta
            tar_height_fail = tar_head_dist_sq > fail_height_dist * fail_height_dist
            tar_height_fail *= progress_buf > 20

            tar_fail = torch.logical_or(tar_overall_fail, tar_height_fail)
        else:
            tar_fail = tar_overall_fail
    else:
        tar_fail = progress_buf < -1

    has_failed = torch.logical_or(has_fallen, tar_fail)

    terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated
