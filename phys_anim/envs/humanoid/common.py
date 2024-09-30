# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from torch import Tensor

from isaac_utils import torch_utils
from phys_anim.envs.base_interface.utils import (
    compute_humanoid_observations,
    compute_humanoid_observations_max,
    compute_humanoid_reset,
    compute_humanoid_reward,
    get_height_maps_jit,
    get_heights,
)
from phys_anim.envs.env_utils.terrains.terrain import Terrain
from phys_anim.utils.motion_lib import MotionLib
from phys_anim.utils.scene_lib import SceneLib

if TYPE_CHECKING:
    from phys_anim.envs.humanoid.isaacgym import Humanoid
else:
    Humanoid = object


class BaseHumanoid(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.num_envs = self.config.num_envs
        self.init_done = False
        self.scene_lib = None

        if self.config.sync_motion:
            control_freq_inv = self.config.simulator.sim.control_freq_inv
            self.config.simulator.sim.control_freq_inv = 1
            self.sync_motion_dt = control_freq_inv / config.simulator.sim.fps
            print("HACK SLOW DOWN")
            self.config.robot.control.control_type = "T"

        self.state_init = self.StateInit[config.state_init]
        self.hybrid_init_prob = config.hybrid_init_prob
        self.reset_default_env_ids = []
        self.reset_ref_env_ids = []

        # Scene storage
        self.total_num_objects = 0
        self.spawned_object_names = []
        self.scene_position = []
        self.object_id_to_scene_id = []
        self.object_dims = []
        self.object_root_states_offsets = []
        self.object_target_position = []

        # General configurations
        self.isaac_pd = self.config.robot.control.control_type == "isaac_pd"
        self.local_root_obs = self.config.local_root_obs
        self.root_height_obs = self.config.root_height_obs
        self.enable_height_termination = self.config.enable_height_termination
        self.max_episode_length = self.config.max_episode_length
        self.control_freq_inv = self.config.simulator.sim.control_freq_inv

        self.setup_character_props()

        # Buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.get_obs_size()), device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.extras = {}
        self.log_dict = {}

        self.terrain = None
        self.force_respawn_on_flat = False

        self.create_terrain_and_scene_lib()

        self.reset_happened = False
        self.reset_ref_env_ids = []
        self.reset_states = None
        self.reset_ref_object_ids = []
        self.object_reset_states = None

        super().__init__(config, device)

        # After objects have been populated, finalize structure
        if self.config.scene_lib is not None:
            self.scene_position = torch.stack(self.scene_position)
            self.object_id_to_scene_id = torch.tensor(
                self.object_id_to_scene_id, device=self.device
            )
            self.object_dims = torch.stack(self.object_dims).reshape(
                self.total_num_objects, -1
            )
            self.object_root_states_offsets = torch.stack(
                self.object_root_states_offsets
            )
            self.object_target_position = torch.stack(self.object_target_position)
            self.env_id_to_object_ids = (
                torch.zeros(
                    self.num_envs,
                    self.scene_lib.config.max_objects_per_scene,
                    dtype=torch.long,
                    device=self.device,
                )
                - 1
            )  # -1 indicates no object

        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_times = torch.zeros(self.num_envs, device=self.device)

        # A tensor to store the scene IDs for each environment.
        # -1 indicates no scene.
        self.scene_ids = (
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device) - 1
        )

        self.respawn_offset_relative_to_data = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

    ###############################################################
    # Getters
    ###############################################################
    def get_obs_size(self):
        return self.num_obs

    def get_action_size(self):
        return self.num_act

    def get_body_id(self, body_name):
        raise NotImplementedError

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rb_pos: torch.tensor = None,
        scene_ids: torch.tensor = None,
    ):
        """
        Samples a new starting position. This takes into account scene and terrain requirements.
        force_respawn_on_flat ensures the character is spawned on flat terrain.
        When provided a valid scene_id, the character will be spawned relative to the scene.

        For non-scene and non-flat env_ids, we sample a random valid coordinate and then shift the vertical offset
        relative to the terrain.
        """
        xy_position = torch.zeros((len(env_ids), 2), device=self.device)
        xy_position[:, :2] += offset

        if self.terrain is not None:
            if self.force_respawn_on_flat:
                xy_position = self.terrain.sample_flat_locations(len(env_ids))
            else:
                xy_position = self.terrain.sample_valid_locations(len(env_ids))

        if scene_ids is not None:
            if -2 in scene_ids:
                raise NotImplementedError(
                    "Attempting to use a motion that requires a scene that is not spawned."
                )

            scene_interaction_envs_mask = (
                scene_ids != -1
            )  # Scene id -1 corresponds to no scene
            if torch.any(scene_interaction_envs_mask):
                xy_position[scene_interaction_envs_mask] = self.scene_position[
                    scene_ids[scene_interaction_envs_mask], :2
                ]

                if isinstance(offset, torch.Tensor):
                    xy_position[scene_interaction_envs_mask, :2] += offset[
                        scene_interaction_envs_mask
                    ]
                else:
                    xy_position[scene_interaction_envs_mask, :2] += offset

        if rb_pos is not None:
            """
            This code block adjusts the character's spawn position to account for terrain and scene elements:

            1. Ground height adjustment:
               - Calculates the ground height below each joint of the character.
               - Shifts the character vertically to maintain its relative height above the terrain.
               - Example: If a jump motion starts 0.5m above ground, the character will spawn 0.5m above the terrain, even on hills.

            2. Scene collision prevention:
               - Checks for potential intersections between the character and scene objects (e.g., furniture).
               - If collisions are detected, slightly elevates the character to avoid intersections.
               - This ensures proper interaction with scene elements, even if the original motion capture data isn't perfectly aligned.

            3. Minimal adjustment principle:
               - Applies the smallest vertical shift necessary to resolve collisions.
               - Preserves the original motion's intent while ensuring stable spawning in various environments.

            This approach allows for flexible character spawning across different terrains and scenes while maintaining motion fidelity and preventing unwanted collisions.
            """
            normalized_rb_pos = rb_pos.clone()
            normalized_rb_pos[:, :, :2] -= rb_pos[:, :1, :2]  # remove root position
            normalized_rb_pos[:, :, :2] += xy_position.unsqueeze(
                1
            )  # add respawn offset
            flat_normalized_dof_pos = normalized_rb_pos.view(-1, 3)
            z_all_joints = self.get_ground_heights(flat_normalized_dof_pos)
            z_all_joints = z_all_joints.view(normalized_rb_pos.shape[:-1])

            z_diff = z_all_joints - normalized_rb_pos[:, :, 2]
            z_indices = torch.max(z_diff, dim=1).indices.view(-1, 1)

            # We want to add the offset based on the ground terrain-height below the joint.
            # Unlike the diff. The reason is that while jumping, we want to ensure the character retains
            # the relative height above the terrain.
            z_offset = z_all_joints.gather(1, z_indices).view(-1, 1)

            z_all_joints_with_scene = self.get_heights_with_scene(
                flat_normalized_dof_pos
            )
            z_all_joints_with_scene = z_all_joints_with_scene.view(
                normalized_rb_pos.shape[:-1]
            )
            # Check if after added offset, if any joint is BELOW the scene (+ respawn offset). If yes, shift up.
            # Otherwise, don't change the height.
            z_diff_with_scene = (
                z_all_joints_with_scene
                + self.config.ref_respawn_offset
                - (normalized_rb_pos[:, :, 2] + z_offset)
            )
            z_with_scene_offset = torch.max(z_diff_with_scene, dim=1).values.view(-1, 1)
            z_with_scene_offset = torch.clamp(z_with_scene_offset, min=0)

            z_offset = z_offset + z_with_scene_offset
        else:
            z_root = self.get_ground_heights(xy_position)
            z_offset = z_root.view(-1, 1) + self.config.ref_respawn_offset

        respawn_position = torch.cat([xy_position, z_offset], dim=-1)

        return respawn_position

    def sample_scene_ids(self, motion_ids, get_first_matching_scene=False):
        """
        For each motion_id that is associated with a scene element it will sample a scene element to condition on.

        For example, a motion sitting on a chair corresponds to multiple possible chairs spawned in the scene. This
        will uniformly sample one of those chairs. If "get_first_matching_scene" is true, it will pick the
        first chair in the list.
        """
        scene_ids = (
            torch.zeros_like(motion_ids, dtype=torch.long, device=self.device) - 1
        )  # index -1 corresponds to no scene element
        valid_mask = torch.ones_like(motion_ids, dtype=torch.bool, device=self.device)
        if (
            self.motion_lib.motion_to_scene_ids.shape[0] > 0
            and self.total_num_objects > 0
        ):
            # Prepare valid_scenes and valid_count tensors
            valid_scenes = self.motion_lib.motion_to_scene_ids[motion_ids]
            valid_count = self.motion_lib.scenes_per_motion[motion_ids]

            # Use scene_lib.sample_scenes to get sampled scenes and valid mask
            sampled_scenes, valid_mask = self.scene_lib.sample_scenes(
                valid_scenes,
                valid_count,
                get_first_matching_scene=get_first_matching_scene,
            )

            # Update scene_ids based on the sampling results
            scene_ids[valid_mask] = sampled_scenes[valid_mask]

            # Handle special cases
            scene_ids[valid_count == -1] = -1  # No scene needed
            scene_ids[valid_count == 0] = -2  # No scene spawned

        return scene_ids, valid_mask

    ###############################################################
    # Environment step logic
    ###############################################################
    def on_environment_ready(self):
        self.motion_lib: MotionLib = self.instantiate_motion_lib()
        if self.config.sync_motion:
            self.sync_motion_times = torch.zeros_like(
                self.motion_lib.state.motion_timings[
                    torch.zeros(self.num_envs, dtype=torch.long, device=self.device), 0
                ]
            )
            self.sync_motion_just_reset = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )

    def pre_physics_step(self, actions):
        if self.config.sync_motion:
            actions *= 0

        self.actions = actions.to(self.device)
        clamp_actions = self.config.robot.control.clamp_actions
        if clamp_actions is not None:
            self.actions = torch.clamp(self.actions, -clamp_actions, clamp_actions)
            self.log_dict["action_clamp_frac"] = (
                self.actions.abs() == clamp_actions
            ).sum() / self.actions.numel()

    def post_physics_step(self):
        self.progress_buf += 1

        if self.world_running():
            self.compute_observations()
            self.compute_reward(self.actions)
            if not self.disable_reset:
                self.compute_reset()

            if self.config.sync_motion:
                self.sync_motion()

            if self.config.output_motion:
                self.output_motion()

        self.log_dict["terminate_frac"] = self.terminate_buf.float().mean()

        self.extras["terminate"] = self.terminate_buf
        self.extras["to_log"] = self.log_dict

    def compute_torques(self, action):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            action (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = action * self.config.robot.control.action_scale
        control_type = self.config.robot.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def update_inference_parameters(self):
        pass

    def world_running(self):
        # Override in IsaacSim.
        return True

    def compute_observations(self, env_ids=None):
        obs = self.compute_humanoid_obs(env_ids)

        if self.terrain is not None:
            height_obs = self.get_height_maps(env_ids)
            if env_ids is None:
                self.terrain_obs[:] = height_obs
            else:
                self.terrain_obs[env_ids] = height_obs

        if env_ids is None:
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

    def compute_humanoid_obs(self, env_ids=None):
        # Retrieve body transforms & velocities
        current_state = self.get_bodies_state()
        body_pos, body_rot, body_vel, body_ang_vel = (
            current_state.body_pos,
            current_state.body_rot,
            current_state.body_vel,
            current_state.body_ang_vel,
        )

        ground_heights = self.get_heights_with_scene(
            self.get_humanoid_root_states()[..., :2]
        )

        if self.config.use_max_coords_obs:
            if env_ids is not None:
                body_pos = body_pos[env_ids]
                body_rot = body_rot[env_ids]
                body_vel = body_vel[env_ids]
                body_ang_vel = body_ang_vel[env_ids]
                ground_heights = ground_heights[env_ids]

            obs = compute_humanoid_observations_max(
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                ground_heights,
                self.local_root_obs,
                self.root_height_obs,
                self.w_last,
            )

        else:
            dof_pos, dof_vel = self.get_dof_state()
            if env_ids is None:
                root_pos = body_pos[:, 0, :]
                root_rot = body_rot[:, 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                dof_pos = dof_pos
                dof_vel = dof_vel
                key_body_pos = body_pos[:, self.key_body_ids, :]
            else:
                root_pos = body_pos[env_ids][:, 0, :]
                root_rot = body_rot[env_ids][:, 0, :]
                root_vel = body_vel[env_ids][:, 0, :]
                root_ang_vel = body_ang_vel[env_ids][:, 0, :]
                dof_pos = dof_pos[env_ids]
                dof_vel = dof_vel[env_ids]
                key_body_pos = body_pos[env_ids][:, self.key_body_ids, :]
                ground_heights = ground_heights[env_ids]

            obs = compute_humanoid_observations(
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                dof_pos,
                dof_vel,
                key_body_pos,
                ground_heights,
                self.local_root_obs,
                self.dof_obs_size,
                self.get_dof_offsets(),
                self.w_last,
            )
        return obs

    def compute_reset(self):
        bodies_positions = self.get_body_positions()
        bodies_contact_buf = self.get_bodies_contact_buf()

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.contact_body_ids,
            bodies_positions,
            self.max_episode_length,
            self.enable_height_termination,
            self.termination_heights
            + self.get_ground_heights(self.get_humanoid_root_states()[..., :2]),
        )

    def compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_default(self, env_ids):
        super().reset_default(env_ids)
        self.reset_default_env_ids = env_ids

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        if len(env_ids) > 0:
            self.reset_happened = True

        self.reset_envs(env_ids)
        return self.obs_buf

    def reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return

        self.reset_default_env_ids = []
        self.reset_ref_env_ids = []
        self.reset_ref_object_ids = []

    def reset_env_tensors(self, env_ids, object_ids=None):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.terminate_buf[env_ids] = 0

    def reset_actors(self, env_ids):
        if self.state_init == self.StateInit.Default:
            self.reset_default(env_ids)
        elif (
            self.state_init == self.StateInit.Start
            or self.state_init == self.StateInit.Random
        ):
            self.reset_ref_state_init(env_ids)
        elif self.state_init == self.StateInit.Hybrid:
            self.reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(
                str(self.state_init)
            )

    def reset_ref_state_init(
        self,
        env_ids,
        motion_ids: Optional[Tensor] = None,
        motion_times: Optional[Tensor] = None,
        scene_ids: Optional[Tensor] = None,
        append_to_lists=False,
    ):
        num_envs = env_ids.shape[0]

        if self.scene_lib is None:
            if motion_ids is None:
                motion_ids = self.motion_lib.sample_motions(num_envs)
        elif motion_ids is None or scene_ids is None:
            if scene_ids is None and motion_ids is not None:
                scene_ids = self.scene_lib.sample_scenes(motion_ids)
            elif scene_ids is None and motion_ids is None:
                self.scene_lib.mark_scene_not_in_use(self.scene_ids[env_ids])
                available_scenes = self.scene_lib.get_available_scenes_mask()
                motion_ids, scene_ids = self.motion_lib.sample_motions_scene_aware(
                    self.sync_motion_just_reset.sum(),
                    available_scenes,
                    self.scene_lib.single_robot_in_scene,
                    with_replacement=True,
                )
                for env_id in env_ids:
                    object_mask = self.object_id_to_scene_id == scene_ids[env_id]
                    self.env_id_to_object_ids[env_id, :] = -1
                    if object_mask.any():
                        object_ids = torch.where(object_mask)[0]
                        self.env_id_to_object_ids[env_id, : len(object_ids)] = (
                            object_ids
                        )
            else:
                raise ValueError(
                    "reset_ref_state_init: scene_ids and motion_ids must be provided together."
                )

        if motion_times is None:
            if (
                self.state_init == self.StateInit.Random
                or self.state_init == self.StateInit.Hybrid
            ):
                max_steps = self.get_required_history_length()

                motion_times = self.sample_time_without_negatives(
                    motion_ids, earliest_time=self.dt * max_steps
                )
            elif self.state_init == self.StateInit.Start:
                motion_times = torch.zeros(num_envs, device=self.device)
            else:
                assert False, "Unsupported state initialization strategy: {:s}".format(
                    str(self.state_init)
                )

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        root_offset = ref_state.root_pos[:, :2].clone()

        ref_state.root_pos[:, :2] = 0
        ref_state.root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids, rb_pos=ref_state.rb_pos, offset=root_offset, scene_ids=scene_ids
        )

        ref_state.dof_pos, ref_state.dof_vel = self.convert_dof(
            ref_state.dof_pos, ref_state.dof_vel
        )

        ref_state.rb_pos[:, :, :3] -= ref_state.rb_pos[:, 0, :3].unsqueeze(1).clone()
        ref_state.rb_pos[:, :, :3] += ref_state.root_pos.unsqueeze(1)

        self.set_env_state(
            env_ids=env_ids,
            root_pos=ref_state.root_pos,
            root_rot=ref_state.root_rot,
            dof_pos=ref_state.dof_pos,
            root_vel=ref_state.root_vel,
            root_ang_vel=ref_state.root_ang_vel,
            dof_vel=ref_state.dof_vel,
            rb_pos=ref_state.rb_pos,
            rb_rot=ref_state.rb_rot,
            rb_vel=ref_state.rb_vel,
            rb_ang_vel=ref_state.rb_ang_vel,
        )

        if append_to_lists and len(self.reset_ref_env_ids) > 0:
            self.reset_ref_env_ids = torch.cat([env_ids, self.reset_ref_env_ids], dim=0)
            self.reset_ref_motion_ids = torch.cat(
                [motion_ids, self.reset_ref_motion_ids], dim=0
            )
            self.reset_ref_motion_times = torch.cat(
                [motion_times, self.reset_ref_motion_times], dim=0
            )
        else:
            self.reset_ref_env_ids = env_ids
            self.reset_ref_motion_ids = motion_ids
            self.reset_ref_motion_times = motion_times

        # Reset objects associated with the scene
        if scene_ids is not None and self.scene_lib is not None:
            has_scene = self.scene_ids > -1
            active_scenes = self.scene_ids[has_scene]
            active_object_mask = torch.isin(self.object_id_to_scene_id, active_scenes)
            active_object_ids = torch.arange(
                len(self.object_id_to_scene_id), device=self.device
            )[active_object_mask]

            # Filter out static objects
            non_static_mask = ~torch.tensor(
                [obj["is_static"] for obj in self.scene_lib.object_spawn_list],
                device=self.device,
            )[active_object_ids]
            non_static_object_ids = active_object_ids[non_static_mask]

            if len(non_static_object_ids) > 0:
                # Create a mapping from scene_id to sync_motion_time
                scene_to_time = {
                    scene_id.item(): time.item()
                    for scene_id, time in zip(
                        active_scenes, self.sync_motion_times[has_scene]
                    )
                }

                # Get the corresponding times for the non-static objects
                non_static_scene_times = torch.tensor(
                    [
                        scene_to_time[self.object_id_to_scene_id[obj_id].item()]
                        for obj_id in non_static_object_ids
                    ],
                    device=self.device,
                )

                non_static_object_states = self.scene_lib.get_object_pose(
                    non_static_object_ids, non_static_scene_times
                )

                # Update object states in the simulation
                self.set_object_state(
                    object_ids=non_static_object_ids,
                    positions=non_static_object_states.translations
                    + self.config.object_ref_respawn_offset,
                    rotations=non_static_object_states.rotations,
                )
                if append_to_lists and len(self.reset_ref_object_ids) > 0:
                    self.reset_ref_object_ids = torch.cat(
                        [non_static_object_ids, self.reset_ref_object_ids], dim=0
                    )
                else:
                    self.reset_ref_object_ids = non_static_object_ids

    def reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = torch_utils.to_torch(
            np.array([self.hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self.reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self.reset_default(default_reset_ids)

    ###############################################################
    # Terrain Helpers
    ###############################################################
    def create_terrain_and_scene_lib(self):
        if self.config.force_flat_terrain:
            """
            Dummy terrain will use a minimal flat terrain (loads much faster), while ensuring the agent
            is provided proper height-map observations.
            """
            self.config.terrain._target_ = (
                "phys_anim.envs.env_utils.terrains.flat_terrain.FlatTerrain"
            )

        if self.config.scene_lib is not None:
            self.scene_lib = SceneLib(self.config.scene_lib, device=self.device)
        self.terrain: Terrain = instantiate(
            self.config.terrain,
            scene_lib=self.scene_lib,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.only_terrain_height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
            * self.terrain.vertical_scale
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
            * self.terrain.vertical_scale
        )
        self.height_points = self.init_height_points()

        self.terrain_obs = torch.zeros(
            self.num_envs,
            self.num_height_points,
            device=self.device,
            dtype=torch.float,
        )

        if self.scene_lib is not None:
            max_objects_per_scene = self.scene_lib.config.max_objects_per_scene
        else:
            max_objects_per_scene = 1

    def get_ground_heights(self, root_states):
        """
        This provides the height of the ground beneath the character.
        Not to confuse with the height-map projection that a sensor would see.
        Use this function for alignment between mocap and new terrains.
        """
        if self.terrain is None:
            has_terrain = False
            height_samples = root_states
            horizontal_scale = 0
        else:
            has_terrain = True
            height_samples = self.only_terrain_height_samples
            horizontal_scale = self.terrain.horizontal_scale

        return get_heights(
            root_states=root_states,
            height_samples=height_samples,
            has_terrain=has_terrain,
            horizontal_scale=horizontal_scale,
        )

    def get_heights_with_scene(self, root_states):
        """
        This provides the height-map projection that a sensor would see.
        This takes into account objects in the scene, such as chairs, tables, etc...
        Use this function to provide a heightmap representation for the character.
        """
        if self.terrain is None:
            has_terrain = False
            height_samples = root_states
            horizontal_scale = 0
        else:
            has_terrain = True
            height_samples = self.height_samples
            horizontal_scale = self.terrain.horizontal_scale

        return get_heights(
            root_states=root_states,
            height_samples=height_samples,
            has_terrain=has_terrain,
            horizontal_scale=horizontal_scale,
        )

    def init_height_points(self):
        """
        Pre-defines the grid for the height-map observation.
        """
        y = torch.tensor(
            np.linspace(
                -self.config.terrain.config.sample_width,
                self.config.terrain.config.sample_width,
                self.config.terrain.config.num_samples_per_axis,
            ),
            device=self.device,
            requires_grad=False,
        )
        x = torch.tensor(
            np.linspace(
                -self.config.terrain.config.sample_width,
                self.config.terrain.config.sample_width,
                self.config.terrain.config.num_samples_per_axis,
            ),
            device=self.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_height_maps(self, env_ids=None, return_all_dims=False):
        """
        Generates a 2D heightmap grid observation rotated w.r.t. the character's heading.
        Each sample is the billinear interpolation between adjacent points.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()
        num_envs = len(env_ids)

        if self.terrain is None:
            return torch.zeros(
                num_envs,
                self.num_height_points,
                1,
                device=self.device,
                requires_grad=False,
            ).view(num_envs, -1)

        root_states = (
            self.get_humanoid_root_states()[env_ids].clone().view(num_envs, -1)
        )

        base_pos = root_states[:, :3]

        return get_height_maps_jit(
            root_states=root_states,
            base_pos=base_pos,
            env_ids=env_ids,
            num_envs=num_envs,
            height_points=self.height_points,
            height_samples=self.height_samples,
            num_height_points=self.num_height_points,
            terrain_horizontal_scale=self.terrain.horizontal_scale,
            w_last=self.w_last,
            return_all_dims=return_all_dims,
        )

    def get_required_history_length(self):
        return 0

    ###############################################################
    # Helpers
    ###############################################################
    def action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def build_termination_heights(self):
        head_term_height = self.config.head_termination_height
        termination_height = self.config.termination_height

        termination_heights = np.array([termination_height] * self.num_bodies)

        if "smpl" in self.config.robot.asset.asset_file_name:
            head_id = self.get_body_id("Head")
        else:
            head_id = self.get_body_id("head")

        termination_heights[head_id] = max(
            head_term_height, termination_heights[head_id]
        )

        asset_file = self.config.robot.asset.asset_file_name
        if "amp_humanoid_sword_shield" in asset_file:
            left_arm_id = self.get_body_id("left_lower_arm")

            shield_term_height = self.config.shield_termination_height
            termination_heights[left_arm_id] = max(
                shield_term_height, termination_heights[left_arm_id]
            )

        self.termination_heights = torch_utils.to_torch(
            termination_heights, device=self.device
        )

    def randomize_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.set_char_color(rand_col, env_ids)

    def set_char_color(self, rand_col, env_ids):
        raise NotImplementedError

    def convert_dof(self, dof_pos, dof_vel):
        return dof_pos, dof_vel

    def sample_time_without_negatives(self, motion_ids: Tensor, earliest_time: float):
        """
        Samples time in the range [earliest_time, motion_lengths[motion_ids]].
        This is done so that calls to the motion lib get-state functions
        don't have any negative times in them, even after subtracting dts
        when calculating histories.
        """
        times = self.motion_lib.sample_time(motion_ids, truncate_time=earliest_time)
        return times + earliest_time

    def sync_motion(self):
        self.sync_motion_times[self.sync_motion_just_reset] = 0
        self.motion_times[:] = self.sync_motion_times
        if self.sync_motion_just_reset.any():
            if self.scene_lib is not None:
                self.scene_lib.mark_scene_not_in_use(
                    self.scene_ids[self.sync_motion_just_reset]
                )

            if (
                self.config.motion_index_offset is not None
                or self.config.fixed_motion_id is not None
            ):
                num_motions = self.motion_lib.num_motions()
                motion_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                if self.config.motion_index_offset is not None:
                    motion_ids += self.config.motion_index_offset
                motion_ids = torch.fmod(motion_ids, num_motions)
                if self.config.fixed_motion_id is not None:
                    motion_ids *= 0
                    motion_ids += self.config.fixed_motion_id
                motion_ids = motion_ids[self.sync_motion_just_reset]
            else:
                if self.scene_lib is not None:
                    available_scenes = self.scene_lib.get_available_scenes_mask()
                    motion_ids, scene_ids = self.motion_lib.sample_motions_scene_aware(
                        self.sync_motion_just_reset.sum(),
                        available_scenes,
                        self.scene_lib.single_robot_in_scene,
                        with_replacement=True,
                    )
                    self.scene_ids[self.sync_motion_just_reset] = scene_ids
                    self.scene_lib.mark_scene_in_use(scene_ids)

                    reset_env_ids = torch.where(self.sync_motion_just_reset)[0]
                    for env_id, scene_id in zip(reset_env_ids, scene_ids):
                        self.env_id_to_object_ids[env_id, :] = -1
                        object_mask = self.object_id_to_scene_id == scene_id
                        if object_mask.any():
                            object_ids = torch.where(object_mask)[0]
                            self.env_id_to_object_ids[env_id, : len(object_ids)] = (
                                object_ids
                            )
                else:
                    motion_ids = self.motion_lib.sample_motions(
                        self.sync_motion_just_reset.sum()
                    )

                self.motion_ids[self.sync_motion_just_reset] = motion_ids
            self.sync_motion_just_reset[:] = False

        ref_state = self.motion_lib.get_motion_state(
            self.motion_ids,
            self.motion_times
            + self.motion_lib.state.motion_timings[self.motion_ids, 0],
        )

        ref_state.root_vel *= 0
        ref_state.root_ang_vel *= 0
        ref_state.dof_vel *= 0
        ref_state.rb_vel *= 0
        ref_state.rb_ang_vel *= 0

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Transfer to the proper coordinates
        root_offset = ref_state.root_pos[:, :2].clone()

        has_scene = self.scene_ids >= 0
        utilized_object_ids = None

        if torch.any(has_scene):
            ref_state.root_pos[has_scene, :2] = 0

            scene_position = self.get_envs_respawn_position(
                env_ids[has_scene],
                rb_pos=ref_state.rb_pos[has_scene],
                offset=root_offset[has_scene],
                scene_ids=self.scene_ids[has_scene],
            )
            ref_state.root_pos[has_scene, :3] += scene_position

            ref_state.dof_pos, ref_state.dof_vel = self.convert_dof(
                ref_state.dof_pos, ref_state.dof_vel
            )

            ref_state.rb_pos[has_scene, :, :3] -= (
                ref_state.rb_pos[has_scene, 0, :3].unsqueeze(1).clone()
            )
            ref_state.rb_pos[has_scene, :, :3] += ref_state.root_pos.unsqueeze(1)

            # Get active objects states
            active_scenes = self.scene_ids[has_scene]
            active_object_mask = torch.isin(self.object_id_to_scene_id, active_scenes)
            active_object_ids = torch.arange(
                len(self.object_id_to_scene_id), device=self.device
            )[active_object_mask]

            # Filter out static objects
            non_static_mask = ~torch.tensor(
                [obj["is_static"] for obj in self.scene_lib.object_spawn_list],
                device=self.device,
            )[active_object_ids]
            non_static_active_object_ids = active_object_ids[non_static_mask]

            if len(non_static_active_object_ids) > 0:
                utilized_object_ids = non_static_active_object_ids

                # Create a mapping from scene_id to sync_motion_time
                active_scene_to_time = {
                    scene_id.item(): time.item()
                    for scene_id, time in zip(
                        active_scenes, self.sync_motion_times[has_scene]
                    )
                }

                # Get the corresponding times for the non-static objects
                non_static_active_scene_times = torch.tensor(
                    [
                        active_scene_to_time[self.object_id_to_scene_id[obj_id].item()]
                        for obj_id in non_static_active_object_ids
                    ],
                    device=self.device,
                )

                object_states = self.scene_lib.get_object_pose(
                    non_static_active_object_ids, non_static_active_scene_times
                )

                # Update object states in the simulation
                self.set_object_state(
                    object_ids=non_static_active_object_ids,
                    positions=object_states.translations,
                    rotations=object_states.rotations,
                )

        # TODO: for non-scene interactions, sample an initial random offset and then add terrain offset.
        self.respawn_offset_relative_to_data[~has_scene, :] = 0

        self.set_env_state(
            env_ids=env_ids,
            root_pos=ref_state.root_pos,
            root_rot=ref_state.root_rot,
            dof_pos=ref_state.dof_pos,
            root_vel=ref_state.root_vel,
            root_ang_vel=ref_state.root_ang_vel,
            dof_vel=ref_state.dof_vel,
            rb_pos=ref_state.rb_pos,
            rb_rot=ref_state.rb_rot,
            rb_vel=ref_state.rb_vel,
            rb_ang_vel=ref_state.rb_ang_vel,
        )

        self.reset_env_tensors(env_ids, utilized_object_ids)

        motion_dur = (
            self.motion_lib.state.motion_timings[self.motion_ids, 1]
            - self.motion_lib.state.motion_timings[self.motion_ids, 0]
        )
        to_fmod = self.sync_motion_times + self.sync_motion_dt

        self.sync_motion_times = torch.fmod(to_fmod, motion_dur)
        # Check for motions that wrapped around
        wrapped_motions = to_fmod >= motion_dur
        # Set sync_motion_just_reset to True for wrapped motions
        self.sync_motion_just_reset[wrapped_motions] = True

    def create_legged_robot_tensors(self):
        if self.config.robot.init_state is None:
            # Only create tensors for simulated robotic humanoids.
            return

        self.p_gains = torch.zeros(
            self.num_act, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_act, dtype=torch.float, device=self.device, requires_grad=False
        )

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.config.robot.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def process_dof_props(self, props):
        if self.config.robot.control.control_type == "isaac_pd":
            # Only create tensors for simulated robotic humanoids.
            return

        self.torque_limits = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(len(props)):
            self.torque_limits[i] = props["effort"][i].item()

    def object_id_to_object_bounding_box(self, object_id):
        if object_id is None:
            object_id = torch.arange(self.object_dims.shape[0], device=self.device)
        object_dims = self.object_dims[object_id]

        object_root_states = self.object_root_states[object_id]

        object_pos = object_root_states[:, 0:3]
        object_rot = object_root_states[..., 3:7]

        min_x = object_dims[:, 0]
        max_x = object_dims[:, 1]
        min_y = object_dims[:, 2]
        max_y = object_dims[:, 3]
        min_z = object_dims[:, 4]
        max_z = object_dims[:, 5]

        object_bounding_box = torch.stack(
            [
                torch.stack([min_x, min_y, min_z], dim=-1),
                torch.stack([min_x, max_y, min_z], dim=-1),
                torch.stack([max_x, max_y, min_z], dim=-1),
                torch.stack([max_x, min_y, min_z], dim=-1),
                torch.stack([min_x, min_y, max_z], dim=-1),
                torch.stack([min_x, max_y, max_z], dim=-1),
                torch.stack([max_x, max_y, max_z], dim=-1),
                torch.stack([max_x, min_y, max_z], dim=-1),
            ],
            dim=1,
        )

        expanded_object_rot = (
            object_rot.unsqueeze(1).expand(object_rot.shape[0], 8, 4).reshape(-1, 4)
        )
        # Shift to origin
        centered_bounding_box = (object_bounding_box).view(-1, 3)

        # Rotate
        rotated_centered_box = torch_utils.quat_rotate(
            expanded_object_rot, centered_bounding_box, self.w_last
        ).view(object_id.shape[0], 8, 3)

        # Shift back
        rotated_bounding_box = rotated_centered_box + object_pos.unsqueeze(1)

        return rotated_bounding_box
