import torch

from phys_anim.envs.humanoid.humanoid_utils import (
    compute_humanoid_observations,
    compute_humanoid_observations_max,
)
from phys_anim.envs.callbacks.base_callback import BaseCallback
from isaac_utils import torch_utils


class HumanoidObs(BaseCallback):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.humanoid_obs = torch.zeros(
            self.env.num_envs,
            self.config.obs_size,
            dtype=torch.float,
            device=self.env.device,
        )
        body_names = self.env.config.robot.isaacgym_body_names
        num_bodies = len(body_names)
        self.root_rot_inv = torch.zeros(
            self.env.num_envs,
            4,
            dtype=torch.float,
            device=self.env.device,
        )
        self.body_contacts = torch.zeros(
            self.env.num_envs,
            num_bodies,
            3,
            dtype=torch.bool,
            device=self.env.device,
        )
        self.bodies_in_contact = torch.zeros(
            self.env.num_envs,
            num_bodies,
            dtype=torch.bool,
            device=self.env.device,
        )

    def compute_observations(self, env_ids):
        current_state = self.env.get_bodies_state()
        body_pos, body_rot, body_vel, body_ang_vel = (
            current_state.body_pos,
            current_state.body_rot,
            current_state.body_vel,
            current_state.body_ang_vel,
        )
        body_contacts = self.env.get_bodies_contact_buf()
        filtered_body_contacts = body_contacts[:, self.env.contact_body_ids]

        if len(self.env.contact_body_ids) > 0:
            # Normalize body contacts before passing to obs function
            filtered_body_contacts = filtered_body_contacts.clone()
            norm = filtered_body_contacts.norm(dim=-1, keepdim=True)
            filtered_body_contacts /= norm + 1e-6
            filtered_body_contacts *= torch.log(norm + 1)  # log scale forces

        ground_heights = self.env.terrain_obs_cb.ground_heights_with_scene

        if self.config.use_max_coords_obs:
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            root_rot = body_rot[:, 0, :]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]
            ground_heights = ground_heights[env_ids]
            filtered_body_contacts = filtered_body_contacts[env_ids]

            obs = compute_humanoid_observations_max(
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                ground_heights,
                filtered_body_contacts,
                self.config.observe_contacts,
                self.config.local_root_obs,
                self.config.root_height_obs,
                self.env.w_last,
            )

        else:
            dof_pos, dof_vel = self.env.get_dof_state()

            root_pos = body_pos[env_ids][:, 0, :]
            root_rot = body_rot[env_ids][:, 0, :]
            root_vel = body_vel[env_ids][:, 0, :]
            root_ang_vel = body_ang_vel[env_ids][:, 0, :]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            key_body_pos = body_pos[env_ids][:, self.env.key_body_ids, :]
            ground_heights = ground_heights[env_ids]
            filtered_body_contacts = filtered_body_contacts[env_ids]

            obs = compute_humanoid_observations(
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                dof_pos,
                dof_vel,
                key_body_pos,
                ground_heights,
                self.config.local_root_obs,
                self.env.dof_obs_size,
                self.env.get_dof_offsets(),
                filtered_body_contacts,
                self.config.observe_contacts,
                self.env.w_last,
            )
        self.humanoid_obs[env_ids] = obs

        root_rot_inv = torch_utils.calc_heading_quat_inv(root_rot, self.env.w_last)

        self.root_rot_inv[env_ids] = root_rot_inv

        self.body_contacts[:] = body_contacts
        self.bodies_in_contact[:] = torch.any(torch.abs(body_contacts) > 0.1, dim=-1)

    def get_obs(self):
        return {"humanoid_obs": self.humanoid_obs}
