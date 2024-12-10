import torch
import numpy as np

from isaac_utils import torch_utils

from phys_anim.envs.callbacks.base_callback import BaseCallback
from phys_anim.envs.humanoid.humanoid_utils import (
    get_height_maps_jit,
    get_heights,
)


class TerrainObs(BaseCallback):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.num_height_points = None

        self.only_terrain_height_samples = (
            torch.tensor(self.env.terrain.heightsamples)
            .view(self.env.terrain.tot_rows, self.env.terrain.tot_cols)
            .to(self.env.device)
            * self.env.terrain.vertical_scale
        )
        self.height_samples = (
            torch.tensor(self.env.terrain.heightsamples)
            .view(self.env.terrain.tot_rows, self.env.terrain.tot_cols)
            .to(self.env.device)
            * self.env.terrain.vertical_scale
        )
        self.height_points = self.init_height_points()

        self.terrain_obs = torch.zeros(
            self.env.num_envs,
            self.num_height_points,
            device=self.env.device,
            dtype=torch.float,
        )

        self.ground_heights = torch.zeros(
            self.env.num_envs,
            1,
            device=self.env.device,
            dtype=torch.float,
        )

        self.ground_heights_with_scene = torch.zeros(
            self.env.num_envs,
            1,
            device=self.env.device,
            dtype=torch.float,
        )

    def init_height_points(self):
        """
        Pre-defines the grid for the height-map observation.
        """
        y = torch.tensor(
            np.linspace(
                -self.config.sample_width,
                self.config.sample_width,
                self.config.num_samples_per_axis,
            ),
            device=self.env.device,
            requires_grad=False,
        )
        x = torch.tensor(
            np.linspace(
                -self.config.sample_width,
                self.config.sample_width,
                self.config.num_samples_per_axis,
            ),
            device=self.env.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.env.num_envs,
            self.num_height_points,
            3,
            device=self.env.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_ground_heights(self, root_states):
        """
        This provides the height of the ground beneath the character.
        Not to confuse with the height-map projection that a sensor would see.
        Use this function for alignment between mocap and new terrains.
        """
        height_samples = self.only_terrain_height_samples
        horizontal_scale = self.env.terrain.horizontal_scale

        return get_heights(
            root_states=root_states,
            height_samples=height_samples,
            horizontal_scale=horizontal_scale,
        )

    def get_heights_with_scene(self, root_states):
        """
        This provides the height-map projection that a sensor would see.
        This takes into account objects in the scene, such as chairs, tables, etc...
        Use this function to provide a heightmap representation for the character.
        """
        height_samples = self.height_samples
        horizontal_scale = self.env.terrain.horizontal_scale

        return get_heights(
            root_states=root_states,
            height_samples=height_samples,
            horizontal_scale=horizontal_scale,
        )

    def get_height_maps(self, root_states, env_ids, return_all_dims=False):
        """
        Generates a 2D heightmap grid observation rotated w.r.t. the character's heading.
        Each sample is the billinear interpolation between adjacent points.
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )
            root_states = (
                self.env.get_humanoid_root_states().clone().view(len(env_ids), -1)
            )

        base_pos = root_states[:, :3]
        height_points = self.height_points[env_ids]

        return get_height_maps_jit(
            root_states=root_states,
            base_pos=base_pos,
            height_points=height_points,
            height_samples=self.height_samples,
            num_height_points=self.num_height_points,
            terrain_horizontal_scale=self.env.terrain.horizontal_scale,
            w_last=self.env.w_last,
            return_all_dims=return_all_dims,
        )

    def compute_height_under_character(self, env_ids):
        root_states = (
            self.env.get_humanoid_root_states()[env_ids].clone().view(len(env_ids), -1)
        )
        self.ground_heights[env_ids] = self.get_ground_heights(root_states)
        self.ground_heights_with_scene[env_ids] = self.get_heights_with_scene(
            root_states
        )

    def compute_observations(self, env_ids):
        root_states = (
            self.env.get_humanoid_root_states()[env_ids].clone().view(len(env_ids), -1)
        )
        self.terrain_obs[env_ids] = self.get_height_maps(root_states, env_ids)
