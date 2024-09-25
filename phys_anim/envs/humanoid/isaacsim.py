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

import os

import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
from hydra.utils import instantiate
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, Vt

from phys_anim.data.assets.skeleton_configs import isaacsim_asset_file_to_stats
from phys_anim.envs.humanoid.common import BaseHumanoid
from phys_anim.envs.base_interface.utils import build_pd_action_offset_scale
from phys_anim.envs.base_interface.isaacsim.robots.humanoid import HumanoidRobot
from phys_anim.envs.base_interface.isaacsim.sim_base_task import SimBaseTask
from phys_anim.envs.base_interface.isaacsim.utils.humanoid_utils import *
from phys_anim.envs.base_interface.isaacsim.utils.humanoid_utils import (
    convert_dof_to_isaac_sim_format,
)
from phys_anim.envs.base_interface.isaacsim.utils.perspective_viewer import (
    PerspectiveViewer,
)
from phys_anim.envs.base_interface.isaacsim.utils.usd_utils import add_terrain_to_stage
from phys_anim.utils.motion_lib import MotionLib

MAX_VALUE = 1000.0


class Humanoid(BaseHumanoid, SimBaseTask):
    def __init__(self, config, device: torch.device) -> None:
        self.w_last = False  # quaternion definition in isaacsim is w_first.
        self.config = config
        self.cameras_config = self.config.cameras
        self.device = device
        self.robot_type = self.config.robot_type
        self._name = "Humanoid"

        self.control_freq_inv = self.config.simulator.sim.control_freq_inv
        self.dt: float = self.control_freq_inv * self.config.simulator.sim.dt
        self.perspective_view = None
        self.cameras = None

        super().__init__(config, device)

        self.build_termination_heights()

        # Allows the agent to disable resets temporarily.
        self.disable_reset = False
        self.num_humanoid_cams = min(self.num_envs, 5)

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self, scene) -> None:
        if self.terrain is not None:
            self.add_terrain()

        self.get_humanoid()
        super().set_up_scene(scene)
        root_prim = self.bodies_names[0]
        self.joints_humanoids_view = ArticulationView(
            prim_paths_expr=f"/World/envs/.*/{self.robot_type}/bodies/{root_prim}",
            name="humanoid_articulation_view",
            reset_xform_properties=False,
        )
        self.bodies_humanoids_view = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/{self.robot_type}/bodies/.*",
            name="humanoid_rigid_prim_view",
            reset_xform_properties=False,
            track_contact_forces=True,
            prepare_contact_sensors=True,
        )

        scene.add(self.joints_humanoids_view)
        scene.add(self.bodies_humanoids_view)

    def _debug_joints(self):
        print(self.joints_humanoids_view.dof_names)
        print(self.joints_humanoids_view.body_names)

    def get_humanoid(self):
        main_dir_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
        self.asset_path = os.path.join(
            main_dir_path,
            self.config.robot.asset.asset_root,
            self.config.robot.asset.asset_file_name,
        )
        humanoid = HumanoidRobot(
            prim_path=self.default_zero_env_path + f"/{self.robot_type}",
            name="Humanoid",
            translation=self.humanoid_positions,
            asset_path=self.asset_path,
        )
        self.sim_config.apply_articulation_settings(
            "Humanoid",
            get_prim_at_path(humanoid.prim_path),
            self.sim_config.parse_actor_config("Humanoid"),
        )

    def add_terrain(self):
        stage = get_current_stage()
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([0.0, 0.0, 0.0])
        add_terrain_to_stage(
            stage=stage, vertices=vertices, triangles=triangles, position=position
        )

    ###############################################################
    # Getters
    ###############################################################
    def get_body_id(self, body_name):
        return self.bodies_names.index(body_name)

    def get_observations(self):
        return self.obs_buf

    def get_dof_offsets(self):
        return self.dof_offsets_sim

    def get_bodies_state(self) -> tuple:
        bodies_positions, bodies_rotations = self.bodies_humanoids_view.get_world_poses(
            clone=True
        )
        velocities = self.bodies_humanoids_view.get_velocities(clone=True)
        bodies_velocities = velocities[..., :3]
        bodies_ang_velocities = velocities[..., 3:]
        bodies_positions = bodies_positions.view(self.num_envs, self.num_bodies, 3)
        bodies_rotations = bodies_rotations.view(self.num_envs, self.num_bodies, 4)
        bodies_velocities = bodies_velocities.view(self.num_envs, self.num_bodies, 3)
        bodies_ang_velocities = bodies_ang_velocities.view(
            self.num_envs, self.num_bodies, 3
        )
        return (
            bodies_positions,
            bodies_rotations,
            bodies_velocities,
            bodies_ang_velocities,
        )

    def get_dof_state(self) -> tuple:
        dof_pos = self.joints_humanoids_view.get_joint_positions(clone=True)
        dof_vel = self.joints_humanoids_view.get_joint_velocities(clone=True)
        return dof_pos, dof_vel

    def get_body_positions(self):
        return self.bodies_humanoids_view.get_world_poses(clone=True)[0].view(
            self.num_envs, self.num_bodies, 3
        )

    def get_bodies_contact_buf(self):
        return (
            self.bodies_humanoids_view.get_net_contact_forces(clone=True)
            .view(self.num_envs, self.num_bodies, 3)
            .clone()
        )

    def get_humanoid_root_states(self):
        root_pos, root_rot = self.joints_humanoids_view.get_world_poses()
        return torch.cat((root_pos, root_rot), dim=-1)

    def get_humanoid_root_velocities(self):
        return self.bodies_humanoids_view.get_velocities()[..., :3].view(
            self.num_envs, self.num_bodies, 3
        )

    def get_num_actors_per_env(self):
        root_pos, root_rot = self.joints_humanoids_view.get_world_poses()
        return root_pos.shape[0] // self.num_envs

    ###############################################################
    # Environment step logic
    ###############################################################
    def post_reset(self):
        """CALLED ONCE AFTER SCENE LOAD"""
        dof_limits = self.joints_humanoids_view.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self.device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self.device)

        # Build pd_actions scales
        self._pd_action_offset, self._pd_action_scale = build_pd_action_offset_scale(
            self.dof_offsets_sim,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.device,
        )
        self.initial_root_pos, self.initial_root_rot = (
            self.joints_humanoids_view.get_world_poses()
        )
        self.initial_root_vel = torch.zeros(
            (len(self.initial_root_pos), 6), device=self.device
        )
        self.initial_dof_pos = torch.zeros_like(
            self.joints_humanoids_view.get_joint_positions(),
            device=self.device,
            dtype=torch.float32,
        )
        self.initial_dof_vel = torch.zeros_like(
            self.joints_humanoids_view.get_joint_velocities(),
            device=self.device,
            dtype=torch.float32,
        )

        self.reset(env_ids=None)
        self.init_done = True

    def world_running(self):
        return self._env._world.is_playing()

    def apply_pd_control(self):
        pd_tar = self.action_to_pd_targets(self.actions)
        self.joints_humanoids_view.set_joint_position_targets(pd_tar)

    def apply_motor_forces(self):
        pass

    ###############################################################
    # Handle Resets
    ###############################################################
    def set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rb_pos,
        rb_rot,
        rb_vel,
        rb_ang_vel,
    ):
        root_vel = torch.cat([root_vel, root_ang_vel], -1)

        self.joints_humanoids_view.set_world_poses(root_pos, root_rot, indices=env_ids)
        self.joints_humanoids_view.set_velocities(root_vel, indices=env_ids)
        self.joints_humanoids_view.set_joint_positions(dof_pos, indices=env_ids)
        self.joints_humanoids_view.set_joint_velocities(dof_vel, indices=env_ids)

        # TODO: set rigid body states
        # TODO: set reset states

    def reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self.reset_actors(env_ids)
            self.reset_env_tensors(env_ids)
            self.set_char_color(np.array([1, 0, 0]), env_ids)
            # step once to update physx with the newly set joint velocities
            SimulationContext.step(self._env._world, render=True)

            self.compute_observations(env_ids)

    def reset_default(self, env_ids):
        respawn_position = self.get_envs_respawn_position(env_ids)
        initial_root_pos = self.initial_root_pos[env_ids].clone()
        initial_root_pos[..., :2] = 0
        initial_root_pos[..., :3] += respawn_position

        self.joints_humanoids_view.set_world_poses(
            initial_root_pos[env_ids], self.initial_root_rot[env_ids], indices=env_ids
        )
        self.joints_humanoids_view.set_velocities(
            self.initial_root_vel[env_ids], indices=env_ids
        )
        self.joints_humanoids_view.set_joint_positions(
            self.initial_dof_pos[env_ids], indices=env_ids
        )
        self.joints_humanoids_view.set_joint_velocities(
            self.initial_dof_vel[env_ids], indices=env_ids
        )

    ###############################################################
    # Helpers
    ###############################################################
    def setup_character_props(self):
        (
            self.bodies_names,
            self.num_dof,
            self.num_bodies,
            self.humanoid_positions,
            self.dof_body_ids_gym,
            self.dof_offsets_gym,
            self.dof_offsets_sim,
            self.dof_offset_indices_isaac_gym_to_sim,
            self.dof_obs_size,
            self.num_obs,
            self.num_act,
            self.contact_body_ids,
            self.key_body_ids,
        ) = isaacsim_asset_file_to_stats(self.robot_type, self.device)

    def set_char_color(self, col, env_ids):
        if self.init_done and self._env._render:
            stage = get_current_stage()
            all_prim_paths = self.bodies_humanoids_view._prim_paths
            color_array = Vt.Vec3fArray.FromNumpy(col)
            for i in range(len(all_prim_paths)):
                if i // self.num_bodies in env_ids:
                    UsdGeom.Gprim.Get(
                        stage, all_prim_paths[i]
                    ).CreateDisplayColorAttr().Set(value=color_array)

    def render(self):
        if not self._env.headless and self.init_done:
            from omni.isaac.sensor import Camera

            if self.perspective_view is None:
                self.perspective_view = PerspectiveViewer()
                self.init_camera()

                self.cameras = []
                for i in range(self.num_humanoid_cams):
                    self.cameras.append(
                        Camera(
                            prim_path="/World/envs/env_"
                            + str(i)
                            + "/camera_human_"
                            + str(i),
                            position=np.array([0.0, 0.0, 25.0]),
                            frequency=15,
                            resolution=(256, 256),
                            orientation=rot_utils.euler_angles_to_quats(
                                np.array([0, 90, 0]), degrees=True
                            ),
                        )
                    )
                    self.cameras[i].initialize()
                for camera_config in self.config.cameras:
                    self.cameras.append(
                        Camera(
                            prim_path="/World/envs/env_"
                            + str(i)
                            + "/camera_"
                            + camera_config.name,
                            position=np.array(camera_config.position),
                            frequency=camera_config.frequency,
                            resolution=tuple(camera_config.resolution),
                            orientation=rot_utils.euler_angles_to_quats(
                                np.array(camera_config.orientation), degrees=True
                            ),
                        )
                    )
                    self.cameras[-1].initialize()
            else:
                self.update_camera()

    def init_camera(self):
        self.cam_prev_char_pos = (
            self.get_humanoid_root_states()[: self.num_humanoid_cams, :3].cpu().numpy()
        )
        pos = self.cam_prev_char_pos[0, :] + np.array([0, -5, 1])
        self.perspective_view.set_camera_view(
            pos, self.cam_prev_char_pos[0, :] + np.array([0, 0, 0.2])
        )

    def update_camera(self):
        char_root_pos = (
            self.get_humanoid_root_states()[: self.num_humanoid_cams, :3].cpu().numpy()
        )
        cam_pos = np.array(self.perspective_view.get_camera_state())
        cam_delta = cam_pos - self.cam_prev_char_pos[0]

        ego_char_root_pos = char_root_pos[0, :]
        new_cam_target = np.array(
            [ego_char_root_pos[0], ego_char_root_pos[1], ego_char_root_pos[2] + 0.2]
        )
        new_cam_pos = np.array(
            [
                ego_char_root_pos[0] + cam_delta[0],
                ego_char_root_pos[1] + cam_delta[1],
                ego_char_root_pos[2] + cam_delta[2],
            ]
        )
        self.perspective_view.set_camera_view(new_cam_pos, new_cam_target)

        root_pos = self.get_humanoid_root_states()[: self.num_humanoid_cams, :]
        import math

        for cam_idx in range(self.num_humanoid_cams):
            camera_pos = self.cameras[cam_idx].get_world_pose()[0]
            cam_delta = camera_pos - torch.tensor(
                self.cam_prev_char_pos[cam_idx, :], device=self.device
            )
            X = root_pos[cam_idx, 0] - camera_pos[0]
            Y = root_pos[cam_idx, 1] - camera_pos[1]
            Z = root_pos[cam_idx, 2] - camera_pos[2]
            pitch = math.atan2(math.sqrt(X**2 + Y**2), Z) * 180 / math.pi - 90
            yaw = math.atan2(Y, X) * 180 / math.pi
            looking_to_root = torch.tensor(
                rot_utils.euler_angles_to_quats(
                    np.array([0, pitch, yaw]), degrees=True
                ),
                device=root_pos.device,
            )
            self.cameras[cam_idx].set_world_pose(
                root_pos[cam_idx, :3] + cam_delta, looking_to_root
            )

        self.cam_prev_char_pos[:] = char_root_pos

    def convert_dof(self, dof_pos, dof_vel):
        dof_pos, dof_vel = convert_dof_to_isaac_sim_format(
            dof_pos, dof_vel, self.dof_offset_indices_isaac_gym_to_sim
        )
        return dof_pos, dof_vel

    def output_motion(self):
        # TODO: add code to record states
        raise NotImplementedError

    def instantiate_motion_lib(self):
        motion_lib: MotionLib = instantiate(
            self.config.motion_lib,
            dof_body_ids=self.dof_body_ids_gym,
            dof_offsets=self.dof_offsets_gym,
            key_body_ids=self.key_body_ids.cpu().numpy(),
            device=self.device,
            w_last=self.w_last,
            object_names=self.spawned_object_names,
        )
        return motion_lib
