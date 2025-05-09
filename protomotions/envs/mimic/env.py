from typing import Dict, Optional

import torch
from torch import Tensor
from isaac_utils import rotations, torch_utils
from protomotions.envs.mimic.mimic_utils import (
    dof_to_local,
    exp_tracking_reward,
)
from protomotions.envs.base_env.env_utils.humanoid_utils import quat_diff_norm
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.mimic.components.mimic_obs import MimicObs
from protomotions.envs.mimic.components.mimic_motion_manager import MimicMotionManager
from protomotions.envs.mimic.components.masked_mimic_obs import MaskedMimicObs


class Mimic(BaseEnv):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config, device, *args, **kwargs)
        # Tracks the internal mimic metrics.
        self.mimic_info_dict = {}

        self.mimic_obs_cb = MimicObs(self.config, self)
        if self.config.masked_mimic.enabled:
            self.masked_mimic_obs_cb = MaskedMimicObs(self.config.masked_mimic, self)

        self.failed_due_bad_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # For dynamic sampling, we record whether the motion was respawned on a flat terrain.
        # We do not record failures on irregular terrain for prioritized sampling as there are no guarantees it should have succeeded.
        self.respawned_on_flat = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        
    def create_motion_manager(self):
        self.motion_manager = MimicMotionManager(self.config.motion_manager, self)
        
    def create_visualization_markers(self):
        if self.config.headless:
            return {}
        
        visualization_markers = super().create_visualization_markers()
        
        body_markers = []
        if self.config.masked_mimic.enabled:
            body_names = self.config.robot.trackable_bodies_subset
        else:
            body_names = self.config.robot.body_names

        for body_name in body_names:
            if (
                self.config.robot.mimic_small_marker_bodies is not None
                and body_name in self.config.robot.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
                
        body_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(1.0, 0.0, 0.0),
            markers=body_markers
        )
        visualization_markers["body_markers"] = body_markers_cfg

        if self.config.masked_mimic.enabled:
            future_body_markers = []
            for body_name in self.config.robot.trackable_bodies_subset:
                if (
                    self.config.robot.mimic_small_marker_bodies is not None
                    and body_name in self.config.robot.mimic_small_marker_bodies
                ):
                    future_body_markers.append(MarkerConfig(size="small"))
                else:
                    future_body_markers.append(MarkerConfig(size="regular"))
            future_body_markers_cfg = VisualizationMarker(
                type="sphere",
                color=(1.0, 1.0, 0.0),
                markers=future_body_markers
            )
            visualization_markers["future_body_markers"] = future_body_markers_cfg
        
        return visualization_markers
        
    def get_markers_state(self):
        if self.config.headless:
            return {}

        markers_state = super().get_markers_state()
        
        # Update mimic markers
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids, self.motion_manager.motion_times
        )

        target_pos = ref_state.rigid_body_pos
        target_pos += self.respawn_offset_relative_to_data.clone().view(
            self.num_envs, 1, 3
        )

        target_pos[..., -1:] += self.terrain.get_ground_heights(
            target_pos[:, 0]
        ).view(self.num_envs, 1, 1)

        if self.config.masked_mimic.enabled:
            num_conditionable_bodies = len(
                self.masked_mimic_obs_cb.conditionable_body_ids
            )
            target_pos = target_pos[
                :, self.masked_mimic_obs_cb.conditionable_body_ids, :
            ]

            inactive_markers = torch.ones(
                self.num_envs,
                num_conditionable_bodies,
                dtype=torch.bool,
                device=self.device,
            )

            mask_time_len = (
                self.config.masked_mimic.masked_mimic_target_pose.num_future_steps
            )

            translation_view = (
                self.masked_mimic_obs_cb.masked_mimic_target_bodies_masks.view(
                    self.num_envs, mask_time_len, num_conditionable_bodies + 1, 2
                )[:, 0, :-1, 0]
            )  # ignore the last entry, that is for speed/heading
            active_translations = translation_view == 1

            inactive_markers[active_translations] = False

            target_pos[inactive_markers] += 100

        target_pos = target_pos.view(self.num_envs, -1, 3)
        markers_state["body_markers"] = MarkerState(
            translation=target_pos,
            orientation=torch.zeros(self.num_envs, target_pos.shape[1], 4, device=self.device),
        )

        # Inbetweening markers
        if self.config.masked_mimic.enabled:
            ref_state = self.motion_lib.get_motion_state(
                self.motion_manager.motion_ids,
                self.masked_mimic_obs_cb.target_pose_time,
            )
            target_pos = ref_state.rigid_body_pos
            target_pos += self.respawn_offset_relative_to_data.clone().view(
                self.num_envs, 1, 3
            )
            target_pos[..., -1:] += self.terrain.get_ground_heights(
                target_pos[:, 0]
            ).view(self.num_envs, 1, 1)

            target_pos = target_pos[
                :, self.masked_mimic_obs_cb.conditionable_body_ids, :
            ]

            translation_view = self.masked_mimic_obs_cb.target_pose_joints_mask.view(
                self.num_envs, num_conditionable_bodies + 1, 2
            )[
                :, :-1, 0
            ]  # ignore the last entry, that is for speed/heading
            active_translations = translation_view == 1

            inactive_markers[active_translations] = False

            target_pos[inactive_markers] += 100

            target_pos[
                torch.logical_not(
                    self.masked_mimic_obs_cb.target_pose_visible_mask.view(-1)
                )
            ] += 100
            target_pos = target_pos.view(self.num_envs, -1, 3)

            markers_state["future_body_markers"] = MarkerState(
                translation=target_pos,
                orientation=torch.zeros(self.num_envs, target_pos.shape[1], 4, device=self.device),
            )
        
        return markers_state

    def get_obs(self):
        obs = super().get_obs()
        mimic_obs = self.mimic_obs_cb.get_obs()
        obs.update(mimic_obs)
        if self.config.masked_mimic.enabled:
            masked_mimic_obs = self.masked_mimic_obs_cb.get_obs()
            obs.update(masked_mimic_obs)
        return obs

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rigid_body_pos: torch.tensor = None,
        requires_scene: torch.tensor = None,
    ):
        """
        Get the offset of the respawn position relative to the current position.
        Also updates the respawned_on_flat flag.
        """
        respawn_position = super().get_envs_respawn_position(
            env_ids, offset=offset, rigid_body_pos=rigid_body_pos, requires_scene=requires_scene
        )

        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )
        target_cur_gt = ref_state.rigid_body_pos
        target_cur_root_pos = target_cur_gt[:, 0, :]

        self.respawn_offset_relative_to_data[env_ids, :2] = (
            respawn_position[:, :2] - target_cur_root_pos[:, :2]
        )

        # Check if spawned on flat, for prioritized sampling
        new_root_pos = respawn_position[..., :2].clone().reshape(env_ids.shape[0], 1, 2)
        new_root_pos = (new_root_pos / self.terrain.horizontal_scale).long()
        px = new_root_pos[:, :, 0].view(-1)
        py = new_root_pos[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.terrain.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain.height_samples.shape[1] - 2)

        self.respawned_on_flat[env_ids] = self.terrain.flat_field_raw[px, py] == 0
        # if scene interaction motion -- also consider as "flat" for dynamic sampling measurements
        if requires_scene is not None and torch.any(requires_scene):
            self.respawned_on_flat[env_ids[requires_scene]] = True

        return respawn_position
    
    def get_motion_requires_scene(self, motion_ids):
        requires_scene = (
            torch.zeros_like(motion_ids, dtype=torch.bool, device=self.device)
        )
        # TODO
        # if (
        #     self.motion_lib.motion_to_scene_ids.shape[0] > 0
        #     and self.num_objects_per_scene > 0
        # ):
        #     motions_lacking_a_scene = self.motion_lib.scenes_per_motion[motion_ids] == 0
        #     assert not torch.any(motions_lacking_a_scene), "Motions lacking a scene are not supported."
        #
        #     motions_with_scenes = self.motion_lib.scenes_per_motion[motion_ids] > 0
        #     requires_scene[motions_with_scenes] = True

        return requires_scene

    def compute_reset(self):
        super().compute_reset()

        if self.config.mimic_early_termination:
            reward_too_bad = torch.zeros_like(self.reset_buf, dtype=torch.bool)
            for entry in self.config.mimic_early_termination:
                key = entry.mimic_early_termination_key
                thresh = entry.mimic_early_termination_thresh
                thresh_on_flat = entry.mimic_early_termination_thresh_on_flat
                value = self.mimic_info_dict[key]

                if entry.less_than:
                    entry_too_bad = value < thresh
                    entry_on_flat_too_bad = value < thresh_on_flat
                else:
                    entry_too_bad = value > thresh
                    entry_on_flat_too_bad = value > thresh_on_flat

                no_scene_interaction = ~self.agent_in_scene
                tight_tracking_threshold = no_scene_interaction & self.respawned_on_flat
                entry_too_bad[tight_tracking_threshold] = entry_on_flat_too_bad[tight_tracking_threshold]
                reward_too_bad |= entry_too_bad

            has_reset_grace = self.motion_manager.get_has_reset_grace()
            reward_too_bad &= ~has_reset_grace

            self.reset_buf[reward_too_bad] = 1
            self.terminate_buf[reward_too_bad] = 1
            self.log_dict["reward_too_bad"] = reward_too_bad.float().mean()

        done_clip = self.motion_manager.get_done_tracks()
        self.reset_buf[done_clip] = 1

    def process_kb(self, gt: Tensor, gr: Tensor):
        kb = gt[:, self.key_body_ids]

        if self.config.mimic_reward_config.relative_kb_pos:
            rt = gt[:, 0]
            rr = gr[:, 0]
            kb = kb - rt.unsqueeze(1)

            heading_rot = torch_utils.calc_heading_quat_inv(rr, True)
            rr_expand = heading_rot.unsqueeze(1).expand(rr.shape[0], kb.shape[1], 4)
            kb = rotations.quat_rotate(
                rr_expand.reshape(-1, 4), kb.view(-1, 3), True
            ).view(kb.shape)

        return kb

    def rotate_pos_to_local(self, pos: Tensor, heading: Optional[Tensor] = None):
        if heading is None:
            raise NotImplementedError("Heading is required for local rotation")
            # root_rot = self.rigid_body_rot[:, 0]
            root_rot = self.get_bodies_state().body_rot[:, 0]
            heading = torch_utils.calc_heading_quat_inv(root_rot, True)

        pos_num_dims = len(pos.shape)
        expanded_heading = heading.view(
            [heading.shape[0]] + [1] * (pos_num_dims - 2) + [heading.shape[1]]
        ).expand(pos.shape[:-1] + (4,))

        rotated = rotations.quat_rotate(
            expanded_heading.reshape(-1, 4), pos.reshape(-1, 3), True
        ).view(pos.shape)
        return rotated

    def compute_reward(self):
        """
        Abbreviations:

        gt = global translation
        gr = global rotation
        rt = root translation
        rr = root rotation
        kb = key bodies
        dv = dof (degrees of freedom velocity)
        """
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids, self.motion_manager.motion_times
        )
        ref_gt = ref_state.rigid_body_pos
        ref_gr = ref_state.rigid_body_rot
        ref_gv = ref_state.rigid_body_vel
        ref_gav = ref_state.rigid_body_ang_vel
        ref_dv = ref_state.dof_vel

        ref_lr = dof_to_local(ref_state.dof_pos, self.simulator.robot_config.dof_offsets, self.simulator.robot_config.joint_axis, True)
        ref_kb = self.process_kb(ref_gt, ref_gr)

        current_state = self.simulator.get_bodies_state()
        gt, gr, gv, gav = (
            current_state.rigid_body_pos,
            current_state.rigid_body_rot,
            current_state.rigid_body_vel,
            current_state.rigid_body_ang_vel,
        )
        # first remove height based on current position
        relative_to_data_gt = gt.clone()
        relative_to_data_gt[:, :, -1:] -= self.terrain.get_ground_heights(gt[:, 0]).view(
            self.num_envs, 1, 1
        )
        # then remove offset to get back to the ground-truth data position
        relative_to_data_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[
            ..., :2
        ].view(self.num_envs, 1, 2)

        kb = self.process_kb(relative_to_data_gt, gr)

        rt = relative_to_data_gt[:, 0]
        ref_rt = ref_gt[:, 0]

        if self.config.mimic_reward_config.rt_ignore_height:
            rt = rt[..., :2]
            ref_rt = ref_rt[..., :2]

        rr = gr[:, 0]
        ref_rr = ref_gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, True)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, True)

        rv = gv[:, 0]
        ref_rv = ref_gv[:, 0]

        rav = gav[:, 0]
        ref_rav = ref_gav[:, 0]

        dof_state = self.simulator.get_dof_state()
        lr = dof_to_local(dof_state.dof_pos, self.simulator.robot_config.dof_offsets, self.simulator.robot_config.joint_axis, True)

        if self.config.mimic_reward_config.add_rr_to_lr:
            rr = gr[:, 0]
            ref_rr = ref_gr[:, 0]

            lr = torch.cat([rr.unsqueeze(1), lr], dim=1)
            ref_lr = torch.cat([ref_rr.unsqueeze(1), ref_lr], dim=1)

        rew_dict = exp_tracking_reward(
            gt=relative_to_data_gt,
            rt=rt,
            kb=kb,
            gr=gr,
            lr=lr,
            rv=rv,
            rav=rav,
            gv=gv,
            gav=gav,
            dv=dof_state.dof_vel,
            ref_gt=ref_gt,
            ref_rt=ref_rt,
            ref_kb=ref_kb,
            ref_gr=ref_gr,
            ref_lr=ref_lr,
            ref_rv=ref_rv,
            ref_rav=ref_rav,
            ref_gv=ref_gv,
            ref_gav=ref_gav,
            ref_dv=ref_dv,
            config=self.config.mimic_reward_config
        )
        dof_forces = self.simulator.get_dof_forces()
        power = torch.abs(torch.multiply(dof_forces, dof_state.dof_vel)).sum(dim=-1)
        pow_rew = -power

        has_reset_grace = self.motion_manager.get_has_reset_grace()
        pow_rew[has_reset_grace] = 0

        rew_dict["pow_rew"] = pow_rew

        local_ref_gt = self.rotate_pos_to_local(ref_gt, ref_inv_heading)
        local_gt = self.rotate_pos_to_local(relative_to_data_gt, inv_heading)
        cartesian_err = (
            ((local_ref_gt - local_ref_gt[:, 0:1]) - (local_gt - local_gt[:, 0:1]))
            .pow(2)
            .sum(-1)
            .sqrt()
            .mean(-1)
        )

        gt_per_joint_err = (ref_gt - relative_to_data_gt).pow(2).sum(-1).sqrt()
        gt_err = gt_per_joint_err.mean(-1)
        max_joint_err = gt_per_joint_err.max(-1)[0]

        rh_err = (ref_gt - relative_to_data_gt)[:, 0, -1].abs()

        gr_diff = quat_diff_norm(gr, ref_gr, True)
        gr_err = gr_diff.mean(-1)
        gr_err_degrees = gr_err * 180 / torch.pi

        max_gr_err = gr_diff.max(-1)[0]
        max_gr_err_degrees = max_gr_err * 180 / torch.pi

        lr_diff = quat_diff_norm(lr, ref_lr, True)
        lr_err = lr_diff.mean(-1)
        lr_err_degrees = lr_err * 180 / torch.pi
        max_lr_err = lr_diff.max(-1)[0]
        max_lr_err_degrees = max_lr_err * 180 / torch.pi

        scaled_rewards: Dict[str, Tensor] = {
            k: v * getattr(self.config.mimic_reward_config.component_weights, f"{k}_w")
            for k, v in rew_dict.items()
        }

        tracking_rew = sum(scaled_rewards.values())

        self.rew_buf = tracking_rew + self.config.mimic_reward_config.positive_constant

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        for rew_name, rew in scaled_rewards.items():
            self.log_dict[f"scaled/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"scaled/{rew_name}_std"] = rew.std()

        other_log_terms = {
            "tracking_rew": tracking_rew,
            "total_rew": self.rew_buf,
            "cartesian_err": cartesian_err,
            "gt_err": gt_err,
            "gr_err": gr_err,
            "gr_err_degrees": gr_err_degrees,
            "lr_err_degrees": lr_err_degrees,
            "max_joint_err": max_joint_err,
            "max_lr_err_degrees": max_lr_err_degrees,
            "max_gr_err_degrees": max_gr_err_degrees,
            "root_height_error": rh_err,
        }

        for rew_name, rew in other_log_terms.items():
            self.log_dict[f"mimic_other/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"mimic_other/{rew_name}_std"] = rew.std()

        self.mimic_info_dict.update(rew_dict)
        self.mimic_info_dict.update(other_log_terms)

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()

        self.mimic_obs_cb.compute_observations(env_ids)
        if self.config.masked_mimic.enabled:
            self.masked_mimic_obs_cb.compute_observations(env_ids)

    def pre_physics_step(self, actions):
        if self.config.mimic_residual_control:
            actions = self.residual_actions_to_actual(actions)

        return super().pre_physics_step(actions)

    def post_physics_step(self):
        self.motion_manager.post_physics_step()
        super().post_physics_step()
        self.motion_manager.handle_reset_track()
        
        if self.config.masked_mimic.enabled:
            self.masked_mimic_obs_cb.post_physics_step()

    def user_reset(self):
        super().user_reset()
        self.motion_manager.motion_times[:] = 1e6

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            if self.config.masked_mimic.enabled:
                self.masked_mimic_obs_cb.reset_track(env_ids)
        return super().reset(env_ids)

    def residual_actions_to_actual(
        self,
        residual_actions: Tensor,
        target_ids: Optional[Tensor] = None,
        target_times: Optional[Tensor] = None,
    ):
        if target_ids is None:
            target_ids = self.motion_manager.motion_ids

        if target_times is None:
            target_times = self.motion_manager.motion_times + self.dt

        ref_state = self.motion_lib.get_motion_state(target_ids, target_times)

        target_local_rot = dof_to_local(
            ref_state.dof_pos, self.simulator.robot_config.dof_offsets, self.simulator.robot_config.joint_axis, True
        )
        residual_actions_as_quats = dof_to_local(
            residual_actions, self.simulator.robot_config.dof_offsets, self.simulator.robot_config.joint_axis, True
        )

        actions_as_quats = rotations.quat_mul(
            residual_actions_as_quats, target_local_rot, True
        )
        actions = torch_utils.quat_to_exp_map(actions_as_quats, True).view(
            self.num_envs, -1
        )

        return actions

    def get_state_dict(self):
        state_dict = super().get_state_dict()
        state_dict["motion_manager"] = self.motion_manager.get_state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.motion_manager.load_state_dict(state_dict["motion_manager"])
