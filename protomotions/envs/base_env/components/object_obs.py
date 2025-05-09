import torch
from isaac_utils import torch_utils

from protomotions.envs.base_env.components.base_component import BaseComponent
from protomotions.envs.base_env.env_utils.humanoid_utils import (
    compute_relative_to_object_pointcloud_contact_bodies_jit,
    get_relative_object_pointclouds_jit,
    get_object_bounding_box_obs_jit,
)


"""
    Provides the object-based observations.
    This includes the bounding boxes, point cloud obs, masks, and human-object relative obs (body-part to object vectors).
"""


class ObjectObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.initial_object_pointclouds = []

        self.object_pointclouds = torch.zeros(
            self.env.num_envs,
            self.env.scene_lib.num_objects_per_scene,
            self.config.num_pointcloud_samples,
            3,
            device=self.env.device,
        )

        self.relative_object_pointclouds = torch.zeros(
            self.env.num_envs,
            self.env.scene_lib.num_objects_per_scene,
            self.config.num_pointcloud_samples,
            3,
            device=self.env.device,
        )

        self.object_pointclouds_obs = (
            torch.zeros(
                self.env.num_envs,
                self.env.scene_lib.num_objects_per_scene
                * self.config.num_features_per_object,
                device=self.env.device,
            )
            + 1000.0
        )

        self.object_bounding_box_obs = torch.zeros(
            self.env.num_envs,
            self.env.scene_lib.num_objects_per_scene,
            (8 * 3 + 6),
            device=self.env.device,
        )

        self.relative_contact_bodies = torch.zeros(
            self.env.num_envs,
            self.env.scene_lib.num_objects_per_scene,
            len(self.env.config.robot.contact_bodies),
            3,
            device=self.env.device,
        )

        self.object_mask = torch.zeros(
            self.env.num_envs,
            self.env.scene_lib.num_objects_per_scene,
            device=self.env.device,
            dtype=torch.bool,
        )

    def objects_spawned(self):
        if len(self.initial_object_pointclouds) > 0:
            self.initial_object_pointclouds = torch.stack(
                self.initial_object_pointclouds
            ).reshape(self.env.total_num_objects, -1, 3)

    def add_initial_object_pointcloud(self, object_pointcloud):
        self.initial_object_pointclouds.append(object_pointcloud)

    def compute_observations(self, env_ids):
        if self.env.config.scene_lib is None:
            return

        num_envs = len(env_ids)
        active_objects = self.env.env_id_to_object_ids[env_ids] >= 0

        active_scenes = self.env.scene_ids[env_ids] >= 0
        if active_scenes.any() and self.env.config.point_cloud_obs.enabled:
            object_pointclouds, _ = self.get_active_object_pointclouds(env_ids)
            self.object_pointclouds[env_ids] = object_pointclouds

            relative_object_pointclouds, _ = self.get_relative_object_pointclouds(
                env_ids, object_pointclouds
            )
            self.relative_object_pointclouds[env_ids] = relative_object_pointclouds

            object_bounding_box = torch.zeros(
                num_envs,
                self.env.num_objects_per_scene,
                (8 * 3 + 6),
                dtype=torch.float,
                device=self.env.device,
            )

            object_bounding_box[active_scenes] = self.get_object_bounding_box_obs(
                env_ids[active_scenes]
            )

            self.object_bounding_box_obs[env_ids] = object_bounding_box

            if self.env.contact_body_ids.numel() > 0:
                current_body_states = self.env.get_bodies_state()
                root_pos = current_body_states.body_pos[env_ids, 0, :3].unsqueeze(1)

                # Pre-compute the inverse root rotation once
                root_rot_inv = self.env.self_obs_cb.root_rot_inv[env_ids].clone()

                contact_bodies = current_body_states.body_pos[env_ids][
                    :, self.env.contact_body_ids, :3
                ]
                contact_bodies = contact_bodies.view(num_envs, -1, 3)
                shifted_contact_bodies = contact_bodies - root_pos

                # Transform to human-relative frame efficiently
                root_rot_inv_expanded = (
                    root_rot_inv.unsqueeze(1)
                    .expand(-1, self.env.contact_body_ids.shape[0], -1)
                    .reshape(-1, 4)
                )

                relative_contact_bodies = torch_utils.quat_rotate(
                    root_rot_inv_expanded,
                    shifted_contact_bodies.reshape(-1, 3),
                    True,
                ).reshape(num_envs, self.env.contact_body_ids.shape[0], 3)

                target_points_for_contact = relative_object_pointclouds

                contact_bodies_obs = (
                    compute_relative_to_object_pointcloud_contact_bodies_jit(
                        target_points_for_contact,
                        relative_contact_bodies,
                        True,
                    )
                )

                self.relative_contact_bodies[env_ids] = contact_bodies_obs

                object_pointclouds_obs = torch.cat(
                    [
                        relative_object_pointclouds.view(
                            num_envs, self.env.num_objects_per_scene, -1
                        ),
                        object_bounding_box,
                        contact_bodies_obs.view(
                            num_envs, self.env.num_objects_per_scene, -1
                        ),
                    ],
                    dim=-1,
                )
            else:
                object_pointclouds_obs = torch.cat(
                    [
                        relative_object_pointclouds.view(
                            num_envs, self.env.num_objects_per_scene, -1
                        ),
                        object_bounding_box,
                    ],
                    dim=-1,
                )

            self.object_pointclouds_obs[env_ids] = object_pointclouds_obs.view(
                num_envs, -1
            )
        self.object_mask[env_ids] = active_objects

    def get_relative_object_pointclouds(self, env_ids, object_pointclouds):
        num_envs = len(env_ids)

        root_states = self.env.get_humanoid_root_states(env_ids)

        object_ids = self.env.env_id_to_object_ids[env_ids].flatten()

        pointclouds = get_relative_object_pointclouds_jit(
            root_pos=root_states.root_pos,
            root_rot=root_states.root_rot,
            pointclouds=object_pointclouds,
            w_last=True,
        )

        active_objects = object_ids >= 0

        return pointclouds.view(
            num_envs, self.env.num_objects_per_scene, -1, 3
        ), active_objects.view(num_envs, -1)

    def get_active_object_pointclouds(self, env_ids):
        """
        Retrieves the point clouds of active objects in the environment.

        Returns:
            pointclouds (torch.Tensor): A tensor containing the point clouds of active objects.
            has_active_objects (torch.Tensor): A boolean tensor indicating which environments have active objects.
        """
        num_envs = len(env_ids)

        active_scenes = self.env.scene_ids[env_ids]
        has_active_objects = active_scenes >= 0

        if not torch.any(has_active_objects):
            # Initialize the pointcloud tensor with a large value (e.g., 1000) to represent "no point"
            pointclouds = (
                torch.zeros(
                    env_ids.shape[0],
                    self.env.num_objects_per_scene,
                    self.config.num_pointcloud_samples,
                    3,
                    dtype=torch.float,
                    device=self.env.device,
                )
                + 1000.0
            )

            return pointclouds, has_active_objects

        object_ids = self.env.env_id_to_object_ids[env_ids].flatten()
        pointclouds = self.object_id_to_object_pointcloud(object_ids)
        pointclouds[object_ids == -1] = 1000.0

        # Reshape to [num_envs, total_objects * total_points_per_object * 3]
        pointclouds = pointclouds.reshape(
            num_envs, self.env.num_objects_per_scene, -1, 3
        )

        return pointclouds, has_active_objects

    def object_id_to_object_pointcloud(self, object_id):
        if object_id is None:
            object_id = torch.arange(
                self.env.object_dims.shape[0], device=self.env.device
            )
        object_pointcloud = self.initial_object_pointclouds[object_id]

        # For now we only support static objects
        return object_pointcloud

        # # Get object root states
        # object_root_states = self.env.get_object_root_states()[object_id]
        # object_positions = object_root_states[:, 0:3]
        # object_rotations = object_root_states[:, 3:7]

        # # Rotate pointcloud
        # rotated_pointcloud = torch_utils.quat_rotate(
        #     object_rotations.unsqueeze(1)
        #     .expand(-1, object_pointcloud.shape[1], -1)
        #     .reshape(-1, 4),
        #     object_pointcloud.view(-1, 3),
        #     True,
        # ).view(object_id.shape[0], -1, 3)

        # # Shift rotated pointcloud to global position
        # global_pointcloud = rotated_pointcloud + object_positions.unsqueeze(1)

        # return global_pointcloud

    def get_object_bounding_box_obs(self, scene_env_ids):
        scene_ids = self.env.scene_ids[scene_env_ids]
        object_ids = self.env.scene_lib.scene_to_object_ids[scene_ids]

        num_scene_envs = scene_env_ids.shape[0]
        max_objects = self.env.num_objects_per_scene

        root_states = self.env.get_humanoid_root_states(scene_env_ids)
        root_states.root_pos[:, -1] -= self.env.terrain.get_ground_heights(root_states.root_pos).view(
            -1
        )

        object_root_states = self.env.get_object_root_states()

        # Flatten object_ids
        flat_object_ids = object_ids.view(-1)

        # Expand root_pos and root_quat
        expanded_root_pos = (
            root_states.root_pos.unsqueeze(1).expand(num_scene_envs, max_objects, 3).reshape(-1, 3)
        )
        expanded_root_quat = (
            root_states.root_rot.unsqueeze(1).expand(num_scene_envs, max_objects, 4).reshape(-1, 4)
        )

        # Use the JIT function to compute bounding box observations
        bounding_box_obs = get_object_bounding_box_obs_jit(
            object_ids=flat_object_ids,
            root_pos=expanded_root_pos,
            root_quat=expanded_root_quat,
            object_root_states=object_root_states,
            object_bounding_box=self.env.object_id_to_object_bounding_box(
                flat_object_ids
            ),
            w_last=True,
        )

        # Reshape the output to match the expected format
        bounding_box_obs = bounding_box_obs.view(num_scene_envs, max_objects, -1)

        return bounding_box_obs
