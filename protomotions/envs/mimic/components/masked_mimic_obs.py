import torch
import numpy as np

from protomotions.envs.base_env.components.base_component import BaseComponent
from protomotions.envs.mimic.mimic_utils import build_sparse_target_poses
from protomotions.envs.base_env.env_utils.general import StepTracker


class MaskedMimicObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.num_conditionable_bodies = (
            len(self.env.config.robot.trackable_bodies_subset) + 1
        )  # + 1 for velocity/heading
        self.conditionable_body_ids = self.env.simulator.build_body_ids_tensor(
            self.env.config.robot.trackable_bodies_subset
        )

        num_future_steps = self.config.masked_mimic_target_pose.num_future_steps
        num_obs_per_sparse_target_pose = (
            self.config.masked_mimic_target_pose.num_obs_per_sparse_target_pose
        )

        # Target poses for masked mimic learning
        self.masked_mimic_target_poses = torch.zeros(
            self.env.num_envs,
            (num_future_steps + 1) * num_obs_per_sparse_target_pose,  # +1 for far away pose
            dtype=torch.float,
            device=self.env.device,
        )
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print(self.masked_mimic_target_poses)
        print(self.env.num_envs)
        print(num_future_steps)
        print(num_obs_per_sparse_target_pose)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        # Masks indicating which bodies are visible in each future pose
        self.masked_mimic_target_bodies_masks = torch.zeros(
            self.env.num_envs,
            self.num_conditionable_bodies * 2 * num_future_steps,
            dtype=torch.bool,
            device=self.env.device,
        )

        # Masks indicating which future poses are visible
        self.masked_mimic_target_poses_masks = torch.zeros(
            self.env.num_envs,
            num_future_steps + 1,  # +1 for far away pose
            dtype=torch.bool,
            device=self.env.device,
        )

        # Step tracker for time-gap masking
        self.time_gap_mask_steps = StepTracker(
            self.env.num_envs,
            min_steps=self.config.masked_mimic_masking.joint_masking.time_gap_mask_min_steps,
            max_steps=self.config.masked_mimic_masking.joint_masking.time_gap_mask_max_steps,
            device=self.env.device,
        )

        # Masks and time for the target pose (far future pose)
        self.target_pose_joints_mask = torch.zeros(
            self.env.num_envs,
            self.num_conditionable_bodies * 2,
            dtype=torch.bool,
            device=self.env.device,
        )
        self.target_pose_time = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.env.device
        )
        self.target_pose_visible_mask = torch.zeros(
            self.env.num_envs, 1, dtype=torch.bool, device=self.env.device
        )

        # Probabilities for masking and conditioning
        masking_config = self.config.masked_mimic_masking.joint_masking
        self.long_term_gap_prob = (
            torch.ones(self.env.num_envs, dtype=torch.float, device=self.env.device)
            * masking_config.with_conditioning_max_gap_probability
        )
        self.visible_target_pose_prob = (
            torch.ones(self.env.num_envs, dtype=torch.float, device=self.env.device)
            * self.config.masked_mimic_masking.target_pose_visible_prob
        )

        # Motion text embeddings
        self.motion_text_embeddings = torch.zeros(
            self.env.num_envs,
            self.config.motion_text_embeddings.embedding_dim,
            dtype=torch.float,
            device=self.env.device,
        )
        self.motion_text_embeddings_mask = torch.zeros(
            self.env.num_envs, 1, dtype=torch.bool, device=self.env.device
        )
        self.visible_text_embeddings_prob = (
            torch.ones(self.env.num_envs, dtype=torch.float, device=self.env.device)
            * self.config.masked_mimic_masking.motion_text_embeddings_visible_prob
        )

        # Historical pose observations
        self.historical_pose_obs = torch.zeros(
            self.env.num_envs,
            self.config.historical_obs.num_historical_conditioned_steps
            * (self.env.self_obs_cb.config.obs_size + 1),  # +1 for time
            device=self.env.device,
            dtype=torch.float,
        )

        assert (
            self.config.historical_obs.num_historical_conditioned_steps
            <= self.env.self_obs_cb.config.num_historical_steps
        ), (
            f"Requesting {self.config.historical_obs.num_historical_conditioned_steps} "
            f"historical steps, but self_obs is configured for only "
            f"{self.env.self_obs_cb.config.num_historical_steps} steps"
        )

        self.joint_masking_config = self.config.masked_mimic_masking.joint_masking
        self.target_pose_config = self.config.masked_mimic_target_pose
        self.historical_obs_config = self.config.historical_obs
        self.motion_text_config = self.config.motion_text_embeddings

    def sample_body_masks(self, num_envs, env_ids=None, reset_track=False):
        """Samples body masks for masking joint information."""
        if not reset_track:
            if env_ids is None:
                env_ids = torch.arange(num_envs, device=self.env.device)
            remaining_time = (
                self.time_gap_mask_steps.cur_max_steps[env_ids]
                - self.time_gap_mask_steps.steps[env_ids]
            )
            active_time_gap = remaining_time > 0
            no_time_gap_or_repeated = torch.ones(
                num_envs, dtype=torch.bool, device=self.env.device
            )
            no_time_gap_or_repeated[active_time_gap] = False

            restart_timegap = (remaining_time <= 0) & (
                torch.rand(num_envs, device=self.env.device)
                < self.joint_masking_config.masked_mimic_time_gap_probability
            )
            self.time_gap_mask_steps.reset_steps(env_ids[restart_timegap])
            longer_time_gap_envs = restart_timegap & (
                self.motion_text_embeddings_mask[env_ids].view(-1)
                | self.target_pose_visible_mask[env_ids].view(-1)
            )
            if longer_time_gap_envs.any():
                self.time_gap_mask_steps.cur_max_steps[longer_time_gap_envs] *= (
                    self.joint_masking_config.with_conditioning_time_gap_mask_max_steps
                    // self.joint_masking_config.time_gap_mask_max_steps
                )

            repeat_mask_envs = (remaining_time < 0) & (
                torch.rand(num_envs, device=self.env.device)
                < self.joint_masking_config.masked_mimic_repeat_mask_probability
            )
            single_step_mask_size = self.num_conditionable_bodies * 2
            new_mask = torch.zeros(
                num_envs,
                self.num_conditionable_bodies,
                2,  # translation and rotation
                dtype=torch.bool,
                device=self.env.device,
            )
            new_mask[repeat_mask_envs] = self.masked_mimic_target_bodies_masks[
                env_ids[repeat_mask_envs], -single_step_mask_size:
            ].view(-1, self.num_conditionable_bodies, 2)
            no_time_gap_or_repeated[repeat_mask_envs] = False
        else:
            # For reset_track=True, we don't apply time-gap logic, so simply initialize defaults.
            new_mask = torch.zeros(
                num_envs,
                self.num_conditionable_bodies,
                2,  # translation and rotation
                dtype=torch.bool,
                device=self.env.device,
            )
            no_time_gap_or_repeated = torch.ones(num_envs, dtype=torch.bool, device=self.env.device)

        num_bodies_mask = (
            torch.rand(num_envs, device=self.env.device)
            >= self.joint_masking_config.force_small_num_conditioned_bodies_prob
        )
        max_num_bodies = torch.where(num_bodies_mask, self.num_conditionable_bodies, 3)
        num_active_bodies = torch.round(
            torch.rand(num_envs, device=self.env.device) * (max_num_bodies - 1) + 1
        ).long()

        max_bodies_mask = (
            torch.rand(num_envs, device=self.env.device)
            <= self.joint_masking_config.force_max_conditioned_bodies_prob
        )
        num_active_bodies[max_bodies_mask] = self.num_conditionable_bodies

        active_body_ids = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            device=self.env.device,
            dtype=torch.bool,
        )
        constraint_states = torch.randint(
            0, 3, (num_envs, self.num_conditionable_bodies), device=self.env.device
        )

        if self.joint_masking_config.masked_mimic_fixed_conditioning is None:
            for idx in range(num_envs):
                active_body_ids[
                    idx,
                    np.random.choice(
                        self.num_conditionable_bodies,
                        size=num_active_bodies[idx].item(),
                        replace=False,
                    ),
                ] = True
        else:
            fixed_conditioning = self.joint_masking_config.masked_mimic_fixed_conditioning
            body_names = [entry.body_name for entry in fixed_conditioning]
            conditioned_body_ids = self.env.simulator.build_body_ids_tensor(body_names).tolist()
            fixed_body_indices = [
                self.conditionable_body_ids.tolist().index(body_id)
                for body_id in conditioned_body_ids
            ]

            for body_index in fixed_body_indices:
                body_name = self.env.config.robot.trackable_bodies_subset[body_index]
                conditioned_body_names = [
                    entry.body_name for entry in fixed_conditioning
                ]
                constraint_index = conditioned_body_names.index(body_name)

                active_body_ids[:, body_index] = True
                constraint_states[:, body_index] = fixed_conditioning[
                    constraint_index
                ].constraint_state

        translation_mask = (constraint_states <= 1) & active_body_ids
        rotation_mask = (constraint_states >= 1) & active_body_ids

        new_mask[no_time_gap_or_repeated, :, 0] = translation_mask[
            no_time_gap_or_repeated
        ]
        new_mask[no_time_gap_or_repeated, :, 1] = rotation_mask[
            no_time_gap_or_repeated
        ]

        return new_mask.view(num_envs, -1)

    def reset_track(self, env_ids):
        """Resets masks and target poses when a new motion track starts."""
        new_motion_ids, new_times = self.env.motion_manager.get_respawn_info(env_ids)

        # Sample for each env whether it should be conditioned using text
        visible_text_embeddings = (
            torch.bernoulli(self.visible_text_embeddings_prob[: len(env_ids)]) > 0
        )
        self.motion_text_embeddings[env_ids] = (
            self.env.motion_lib.sample_text_embeddings(new_motion_ids)
        )
        has_text = self.env.motion_lib.state.has_text_embeddings[new_motion_ids]
        self.motion_text_embeddings_mask[env_ids] = visible_text_embeddings.view(
            -1, 1
        ) & has_text.view(-1, 1)

        # Sample for each env whether it should be conditioned using a "far away" target pose
        visible_target_pose = (
            torch.bernoulli(self.visible_target_pose_prob[: len(env_ids)]) > 0
        )
        self.target_pose_visible_mask[env_ids] = visible_target_pose.view(-1, 1)

        # Sample "far away" target pose time
        max_time = self.env.motion_lib.state.motion_lengths[new_motion_ids]
        target_pose_time = (
            torch.rand(len(env_ids), device=self.env.device) * (max_time - new_times)
            + new_times
        )
        sample_max_time = torch.rand(len(env_ids), device=self.env.device) < 0.1
        target_pose_time[sample_max_time] = max_time[sample_max_time]
        self.target_pose_time[env_ids] = target_pose_time

        # Sample "far away" target pose joints
        visible_target_pose_joints = self.sample_body_masks(
            len(env_ids), None, reset_track=True
        )
        self.target_pose_joints_mask[env_ids] = visible_target_pose_joints

        # Sample new body masks for the "near future" target poses
        new_body_masks = self.sample_body_masks(len(env_ids), env_ids, reset_track=True)
        single_step_mask_size = self.num_conditionable_bodies * 2
        new_body_masks = (
            new_body_masks.view(len(env_ids), 1, single_step_mask_size)
            .expand(-1, self.target_pose_config.num_future_steps, -1)
            .reshape(len(env_ids), -1)
        )
        self.time_gap_mask_steps.reset_steps(env_ids)
        if self.joint_masking_config.masked_mimic_time_gap_probability == 1:
            # Special case where we want to force non-visible joints. Turn masks on for all joints right away.
            new_body_masks[:, :] = 0

        # Determine whether we have a long-term conditioning signal (text / far away target pose).
        # When such a signal exists, we can (with probability) disable all near-term joints.
        has_long_term_conditioning = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.env.device
        )
        has_long_term_conditioning = torch.logical_or(
            has_long_term_conditioning,
            self.motion_text_embeddings_mask[env_ids].view(-1),
        )
        has_long_term_conditioning = torch.logical_or(
            has_long_term_conditioning,
            self.target_pose_visible_mask[env_ids].view(-1),
        )

        if has_long_term_conditioning.any():
            long_term_gap = (
                torch.bernoulli(self.long_term_gap_prob[: len(env_ids)])[
                    has_long_term_conditioning
                ]
                > 0
            )
            if long_term_gap.any():
                long_term_gap_env_ids = env_ids[has_long_term_conditioning][
                    long_term_gap
                ]
                # For all long-term gap envs, set the time-gap to go beyond the maximal episode length
                # (no near-term joints). And turn off the near-term joints (mask out).
                self.time_gap_mask_steps.cur_max_steps[long_term_gap_env_ids] = (
                    self.env.config.max_episode_length * 2
                )
                new_body_masks[has_long_term_conditioning][long_term_gap, :] = 0

        self.masked_mimic_target_bodies_masks[env_ids] = new_body_masks

    def build_sparse_target_poses_masked_with_time(self, env_ids, num_future_steps):
        """
        Builds sparse target poses with masks and time information.
        """
        num_envs = env_ids.shape[0]
        time_offsets = torch.arange(
            1, num_future_steps + 1, device=self.env.device, dtype=torch.long
        ) * self.env.dt

        near_future_times = (
            self.env.motion_manager.motion_times[env_ids].unsqueeze(-1)
            + time_offsets.unsqueeze(0)
        )
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time[env_ids].view(-1, 1)], dim=1
        )

        motion_ids = self.env.motion_manager.motion_ids[env_ids].unsqueeze(-1).tile(
            [1, num_future_steps + 1]
        )

        obs = self.build_sparse_target_poses(env_ids, all_future_times).view(
            num_envs, num_future_steps + 1, self.num_conditionable_bodies, 2, 12
        )

        near_mask = self.masked_mimic_target_bodies_masks[env_ids].view(
            num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )
        far_mask = self.target_pose_joints_mask[env_ids].view(num_envs, 1, -1, 2, 1)
        mask = torch.cat([near_mask, far_mask], dim=1)

        masked_obs = obs * mask
        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            num_envs, num_future_steps + 1, -1
        )

        flat_motion_ids = motion_ids.view(-1)
        motion_lengths = self.env.motion_lib.get_motion_length(flat_motion_ids)
        times = (
            torch.minimum(all_future_times.view(-1), motion_lengths).view(
                num_envs, num_future_steps + 1, 1
            )
            - self.env.motion_manager.motion_times[env_ids].view(num_envs, 1, 1)
        )
        ones_vec = torch.ones(num_envs, num_future_steps + 1, 1, device=self.env.device)
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(num_envs, -1)

    def build_sparse_target_poses(self, env_ids, future_times):
        """
        Builds sparse target poses relative to the current pose.
        """
        num_envs = env_ids.shape[0]
        num_future_steps = future_times.shape[1]

        motion_ids = self.env.motion_manager.motion_ids[env_ids].unsqueeze(-1).tile(
            [1, num_future_steps]
        )
        flat_motion_ids = motion_ids.view(-1)
        motion_lengths = self.env.motion_lib.get_motion_length(flat_motion_ids)
        flat_future_times = torch.minimum(future_times.view(-1), motion_lengths)

        ref_state = self.env.motion_lib.get_motion_state(
            flat_motion_ids, flat_future_times
        )
        flat_target_pos = ref_state.rigid_body_pos
        flat_target_rot = ref_state.rigid_body_rot
        flat_target_vel = ref_state.rigid_body_vel

        current_body_state = self.env.simulator.get_bodies_state(env_ids)

        # Adjust current global translations to be relative to the data origin
        current_body_state.rigid_body_pos[:, :, -1:] -= (
            self.env.terrain.get_ground_heights(current_body_state.rigid_body_pos[:, 0]).view(num_envs, 1, 1)
        )
        current_body_state.rigid_body_pos[..., :2] -= (
            self.env.respawn_offset_relative_to_data[env_ids]
            .clone()[..., :2]
            .view(num_envs, 1, 2)
        )

        return build_sparse_target_poses(
            cur_gt=current_body_state.rigid_body_pos,
            cur_gr=current_body_state.rigid_body_rot,
            flat_target_pos=flat_target_pos,
            flat_target_rot=flat_target_rot,
            flat_target_vel=flat_target_vel,
            masked_mimic_conditionable_bodies_ids=self.conditionable_body_ids,
            num_future_steps=num_future_steps,
            num_envs=num_envs,
            w_last=True,
        )

    def compute_observations(self, env_ids):
        """Computes and updates masked mimic observations."""
        future_poses = self.build_sparse_target_poses_masked_with_time(
            env_ids, self.target_pose_config.num_future_steps
        )
        self.masked_mimic_target_poses[env_ids] = future_poses

        num_envs = env_ids.shape[0]
        reshaped_masked_bodies_masks = self.masked_mimic_target_bodies_masks[
            env_ids
        ].view(num_envs, self.target_pose_config.num_future_steps, -1)

        time_offsets = torch.arange(
            1,
            self.target_pose_config.num_future_steps + 1,
            device=self.env.device,
            dtype=torch.long,
        ) * self.env.dt

        near_future_times = (
            self.env.motion_manager.motion_times[env_ids].unsqueeze(-1)
            + time_offsets.unsqueeze(0)
        )
        motion_ids = self.env.motion_manager.motion_ids[env_ids].unsqueeze(-1).tile(
            [1, self.target_pose_config.num_future_steps]
        )
        motion_lengths = self.env.motion_lib.get_motion_length(motion_ids.view(-1)).view(
            num_envs, self.target_pose_config.num_future_steps
        )
        in_bound_times = near_future_times <= motion_lengths

        for i in range(self.target_pose_config.num_future_steps):
            any_visible_joint = reshaped_masked_bodies_masks[:, i].any(dim=-1)
            self.masked_mimic_target_poses_masks[env_ids, i] = (
                any_visible_joint & in_bound_times[:, i]
            )

        self.masked_mimic_target_poses_masks[env_ids, -1:] = self.target_pose_visible_mask[
            env_ids
        ]

        sub_sampling_factor = (
            self.env.self_obs_cb.config.num_historical_steps
            // self.historical_obs_config.num_historical_conditioned_steps
        )
        historical_poses = (
            self.env.self_obs_cb.humanoid_obs_hist_buf.get_all_flattened()[env_ids]
        )
        historical_poses = historical_poses.view(
            num_envs, self.env.self_obs_cb.config.num_historical_steps, -1
        )[:, ::sub_sampling_factor]

        times = (
            torch.arange(
                self.env.self_obs_cb.config.num_historical_steps, device=self.env.device
            )
            * self.env.dt
        )
        time_offsets = (
            times[::sub_sampling_factor]
            .view(1, self.historical_obs_config.num_historical_conditioned_steps, 1)
            .expand(num_envs, -1, -1)
        )
        historical_poses_with_time = torch.cat(
            [historical_poses, time_offsets], dim=-1
        ).view(num_envs, -1)
        self.historical_pose_obs[env_ids] = historical_poses_with_time

    def post_physics_step(self):
        """Updates masks and target pose parameters after physics step."""
        self.time_gap_mask_steps.advance()
        had_long_term_conditioning = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        had_long_term_conditioning = (
            had_long_term_conditioning | self.motion_text_embeddings_mask.view(-1)
        )
        had_long_term_conditioning = (
            had_long_term_conditioning | self.target_pose_visible_mask.view(-1)
        )

        passed_target_pose_time = (
            self.target_pose_time < self.env.motion_manager.motion_times
        )

        new_target_pose_visibility = (
            torch.bernoulli(
                self.visible_target_pose_prob[
                    : self.target_pose_visible_mask[passed_target_pose_time].shape[0]
                ]
            )
            > 0
        )
        self.target_pose_visible_mask[
            passed_target_pose_time
        ] = new_target_pose_visibility.view(-1, 1)

        motion_lengths = self.env.motion_lib.state.motion_lengths[
            self.env.motion_manager.motion_ids[passed_target_pose_time]
        ]
        target_pose_time = (
            torch.rand(
                self.target_pose_visible_mask[passed_target_pose_time].shape[0],
                device=self.env.device,
            )
            * (
                motion_lengths
                - self.env.motion_manager.motion_times[passed_target_pose_time]
            )
            + self.env.motion_manager.motion_times[passed_target_pose_time]
        )
        sample_max_time = (
            torch.rand(
                self.target_pose_visible_mask[passed_target_pose_time].shape[0],
                device=self.env.device,
            )
            > 0.5
        )
        target_pose_time[sample_max_time] = motion_lengths[sample_max_time]
        self.target_pose_time[passed_target_pose_time] = target_pose_time

        has_no_long_term_conditioning = torch.logical_not(
            self.motion_text_embeddings_mask.view(-1)
            | self.target_pose_visible_mask.view(-1)
        )
        lost_long_term_conditioning = (
            had_long_term_conditioning & has_no_long_term_conditioning
        )

        if lost_long_term_conditioning.any():
            # If no longer has long-term conditioning, reset the time-gap masking to bring back the near-term
            # joint constraints.
            self.time_gap_mask_steps.cur_max_steps[lost_long_term_conditioning] = -1
        all_env_ids = torch.arange(
            self.env.num_envs, dtype=torch.long, device=self.env.device
        )
        new_body_masks = self.sample_body_masks(
            self.env.num_envs, all_env_ids, reset_track=False
        )

        single_step_mask_size = self.num_conditionable_bodies * 2
        self.masked_mimic_target_bodies_masks[
            :, :-single_step_mask_size
        ] = self.masked_mimic_target_bodies_masks[:, single_step_mask_size:].clone()
        self.masked_mimic_target_bodies_masks[:, -single_step_mask_size:] = (
            new_body_masks
        )

    def get_obs(self):
        """Returns the masked mimic observations."""
        return {
            "masked_mimic_target_poses": self.masked_mimic_target_poses.clone(),
            "masked_mimic_target_bodies_masks": self.masked_mimic_target_bodies_masks.clone(),
            "masked_mimic_target_poses_masks": self.masked_mimic_target_poses_masks.clone(),
            "motion_text_embeddings": self.motion_text_embeddings.clone(),
            "motion_text_embeddings_mask": self.motion_text_embeddings_mask.clone(),
            "historical_pose_obs": self.historical_pose_obs.clone(),
        }
