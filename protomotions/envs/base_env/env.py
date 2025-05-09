from enum import Enum
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate, get_class
from torch import Tensor
from isaac_utils import torch_utils

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import MarkerConfig, VisualizationMarker, MarkerState, SimulatorConfig
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.envs.base_env.env_utils.humanoid_utils import (
    compute_humanoid_reset,
)
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig
from protomotions.envs.base_env.components.humanoid_obs import HumanoidObs
from protomotions.envs.base_env.components.terrain_obs import TerrainObs
from protomotions.envs.base_env.components.motion_manager import MotionManager

from protomotions.utils.motion_lib import MotionLib
from protomotions.utils.scene_lib import SceneLib

class BaseEnv:
    class StateInit(Enum):
        Default = 0
        Data = 1
        Hybrid = 2

    def __init__(self, config, device: torch.device, *args, **kwargs):
        self.config = config
        self.device = device
        self.num_envs = self.config.num_envs
        self.create_terrain_and_scene_lib()
        self.visualization_markers = self.create_visualization_markers()
        
        if self.config.sync_motion:
            decimation = self.config.simulator.config.sim.decimation
            self.config.simulator.config.sim.decimation = 1
            self.sync_motion_dt = decimation / self.config.simulator.config.sim.fps
            print("HACK SLOW DOWN")
            self.config.robot.control.control_type = "torque"

        SimulatorConfigClass = get_class(self.config.simulator._config_target_)
        simulator_config: SimulatorConfig = SimulatorConfigClass.from_dict(self.config.simulator.config)
        SimulatorClass = get_class(self.config.simulator._target_)

        self.simulator: Simulator = SimulatorClass(
            config=simulator_config,
            scene_lib=self.scene_lib,
            terrain=self.terrain,
            visualization_markers=self.visualization_markers,
            device=self.device,
            **kwargs
        )
        self.simulator.on_environment_ready()
        self.default_state = self.simulator.get_default_state()

        self.dt = self.simulator.dt

        self.key_body_ids = self.simulator.build_body_ids_tensor(
            self.config.robot.key_bodies
        )
        self.non_termination_contact_body_ids = self.simulator.build_body_ids_tensor(
            self.config.robot.non_termination_contact_bodies
        )
        self.contact_body_ids = self.simulator.build_body_ids_tensor(
            self.config.robot.contact_bodies
        )
        self.build_termination_heights()

        self.state_init = self.StateInit[config.state_init]
        self.hybrid_init_prob = config.hybrid_init_prob

        if self.config.motion_lib is not None and self.config.motion_lib.motion_file is not None:
            self.motion_lib: MotionLib = self.instantiate_motion_lib()
            self.create_motion_manager()
        else:
            assert self.state_init == self.StateInit.Default, "Motion lib must be set and a motion file must be provided if state_init is not Default"
            self.motion_lib = None
            self.motion_manager = None

        if self.config.sync_motion:
            assert self.motion_lib is not None, "Motion lib must be set if sync_motion is True"
            self.sync_motion_times = torch.zeros_like(self.motion_manager.motion_times)
            self.sync_motion_just_reset = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        # Allows the agent to disable resets temporarily.
        self.disable_reset = False
        self.init_done = False

        # Buffers
        self.self_obs_cb = HumanoidObs(self.config.humanoid_obs, self)
        self.terrain_obs_cb = TerrainObs(self.config.terrain.config, self)

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

        self.force_respawn_on_flat = False

        # After objects have been populated, finalize structure
        if self.scene_lib is not None:
            self.scene_position = torch.stack(self.simulator.get_scene_positions())
            self.object_dims = torch.stack(self.simulator.get_object_dims()).reshape(
                self.num_envs, self.scene_lib.num_objects_per_scene, -1
            )

        # Record whether agent is interacting in its scene, or "out in the world".
        self.agent_in_scene = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        self.respawn_offset_relative_to_data = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

    ###############################################################
    # Getters
    ###############################################################
    def get_obs(self):
        obs = self.self_obs_cb.get_obs()
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        return obs

    def get_action_size(self):
        return self.simulator.num_act

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rigid_body_pos: torch.tensor = None,
        requires_scene: torch.tensor = None,
    ):
        """
        Samples a new starting position for the environment.
        This method considers both scene and terrain requirements.

        If `force_respawn_on_flat` is True, the character is spawned on flat terrain.
        When a valid scene is required (indicated by `requires_scene`), the character is spawned relative to the scene's position.

        For environments without a scene or when not forcing flat terrain, a random valid coordinate is sampled,
        and the vertical offset is adjusted relative to the terrain.
        """
        xy_position = torch.zeros((len(env_ids), 2), device=self.device)
        xy_position[:, :2] += offset

        if self.terrain is not None:
            if self.force_respawn_on_flat:
                xy_position = self.terrain.sample_flat_locations(len(env_ids))
            else:
                xy_position = self.terrain.sample_valid_locations(len(env_ids))

        if requires_scene is not None and torch.any(requires_scene):
            xy_position[requires_scene] = self.scene_position[
                env_ids[requires_scene], :2
            ]

            if isinstance(offset, torch.Tensor):
                xy_position[requires_scene, :2] += offset[requires_scene]
            else:
                xy_position[requires_scene, :2] += offset

        if rigid_body_pos is not None:
            """
            This code block adjusts the character's spawn position to account for terrain and scene elements:

            1. Ground height adjustment:
               - Calculates the ground height below each joint of the character.
               - Shifts the character vertically to maintain its relative height above the terrain.
               - Example: If a jump motion starts 0.5m above ground, the character will spawn 0.5m above the terrain, even on hills.

            2. Minimal adjustment principle:
               - Applies the smallest vertical shift necessary to resolve collisions.
               - Preserves the original motion's intent while ensuring stable spawning in various environments.

            This approach allows for flexible character spawning across different terrains and scenes while maintaining motion fidelity and preventing unwanted collisions.
            """
            normalized_rigid_body_pos = rigid_body_pos.clone()
            normalized_rigid_body_pos[:, :, :2] -= rigid_body_pos[
                :, :1, :2
            ]  # remove root position
            normalized_rigid_body_pos[:, :, :2] += xy_position.unsqueeze(
                1
            )  # add respawn offset
            flat_normalized_rigid_body_pos = normalized_rigid_body_pos.view(-1, 3)
            z_all_joints = self.terrain.get_ground_heights(
                flat_normalized_rigid_body_pos
            )
            z_all_joints = z_all_joints.view(normalized_rigid_body_pos.shape[:-1])

            z_diff = z_all_joints - normalized_rigid_body_pos[:, :, 2]
            z_indices = torch.max(z_diff, dim=1).indices.view(-1, 1)

            # We want to add the offset based on the ground terrain-height below the joint.
            # Unlike the diff. The reason is that while jumping, we want to ensure the character retains
            # the relative height above the terrain.
            z_offset = z_all_joints.gather(1, z_indices).view(-1, 1) + self.config.ref_respawn_offset
        else:
            z_root = self.terrain.get_ground_heights(xy_position)
            z_offset = z_root.view(-1, 1) + self.config.ref_respawn_offset

        respawn_position = torch.cat([xy_position, z_offset], dim=-1)

        return respawn_position

    def get_motion_requires_scene(self, motion_ids):
        """
        Returns a boolean tensor indicating whether each motion requires a scene.
        By default, all motions don't require a scene.
        """
        requires_scene = torch.zeros_like(
            motion_ids, dtype=torch.bool, device=self.device
        )
        return requires_scene

    def get_markers_state(self):
        if self.config.headless:
            return {}
        
        markers_state = {}
        
        # Update terrain markers
        height_maps = self.terrain.get_height_maps(
            self.simulator.get_root_state(), None, return_all_dims=True
        ).view(self.num_envs, -1, 3)
        markers_state["terrain_markers"] = MarkerState(
            translation=height_maps,
            orientation=torch.zeros(self.num_envs, height_maps.shape[1], 4, device=self.device),
        )
        
        return markers_state

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, actions):
        actions = self.pre_physics_step(actions)
        clamp_actions = self.config.robot.control.clamp_actions
        if clamp_actions is not None:
            actions = torch.clamp(actions, -clamp_actions, clamp_actions)
            self.log_dict["action_clamp_frac"] = (
                actions.abs() == clamp_actions
            ).sum() / actions.numel()

        markers_state = self.get_markers_state()
        self.simulator.step(actions, markers_state)

        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        # self.render()

        return self.get_obs(), self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self, actions):
        if self.config.sync_motion:
            actions *= 0
        return actions

    def on_epoch_end(self, current_epoch: int):
        pass

    def post_physics_step(self):
        self.progress_buf += 1
        self.self_obs_cb.post_physics_step()

        self.compute_observations()
        self.compute_reward()
        if not self.disable_reset:
            self.compute_reset()

        if self.config.sync_motion:
            self.sync_motion()

        self.log_dict["terminate_frac"] = self.terminate_buf.float().mean()

        self.extras["terminate"] = self.terminate_buf
        self.extras["to_log"] = self.log_dict

    def user_reset(self):
        self.progress_buf[:] = 1e6

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self.self_obs_cb.compute_observations(env_ids)
        self.terrain_obs_cb.compute_observations(env_ids)

    def compute_reset(self):
        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos
        bodies_contact_buf = self.self_obs_cb.body_contacts.clone()

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.non_termination_contact_body_ids,
            bodies_positions,
            self.config.max_episode_length,
            self.config.enable_height_termination,
            self.termination_heights
            + self.terrain.get_ground_heights(bodies_positions[:, 0]),
        )

    def compute_reward(self):
        self.rew_buf[:] = torch.ones(
            self.num_envs, dtype=torch.float, device=self.device
        )

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset_default(self, env_ids):
        # Adjust root position
        default_state = self.default_state

        root_pos = default_state.root_pos[env_ids].clone()
        root_rot = default_state.root_rot[env_ids].clone()
        dof_pos = default_state.dof_pos[env_ids].clone()
        root_vel = default_state.root_vel[env_ids].clone()
        root_ang_vel = default_state.root_ang_vel[env_ids].clone()
        dof_vel = default_state.dof_vel[env_ids].clone()
        rigid_body_pos = default_state.rigid_body_pos[env_ids].clone()
        rigid_body_rot = default_state.rigid_body_rot[env_ids].clone()
        rigid_body_vel = default_state.rigid_body_vel[env_ids].clone()
        rigid_body_ang_vel = default_state.rigid_body_ang_vel[env_ids].clone()

        root_pos[:, :2] = 0
        root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids,
            rigid_body_pos=rigid_body_pos,
            offset=0,
        )

        # Transfer entire body to the proper coordinates
        rigid_body_pos[:, :, :3] -= (
            rigid_body_pos[:, 0, :3].unsqueeze(1).clone()
        )
        rigid_body_pos[:, :, :3] += root_pos.unsqueeze(1)
        
        new_states = RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
        )

        return new_states

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            if isinstance(env_ids, list):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            env_ids = env_ids.to(self.device)

            if self.motion_manager is not None:
                self.motion_manager.reset_envs(env_ids)

            if self.state_init == self.StateInit.Default:
                new_states = self.reset_default(env_ids)
                reset_default_env_ids = env_ids
                reset_ref_env_ids = []
                reset_ref_motion_ids = []
                reset_ref_motion_times = []
            elif self.state_init == self.StateInit.Data:
                new_states, motion_ids, motion_times = self.reset_ref_state_init(
                    env_ids
                )
                reset_default_env_ids = []
                reset_ref_env_ids = env_ids
                reset_ref_motion_ids = motion_ids
                reset_ref_motion_times = motion_times
            elif self.state_init == self.StateInit.Hybrid:
                new_states, default_env_ids, ref_env_ids, motion_ids, motion_times = (
                    self.reset_hybrid_state_init(env_ids)
                )
                reset_default_env_ids = default_env_ids
                reset_ref_env_ids = ref_env_ids
                reset_ref_motion_ids = motion_ids
                reset_ref_motion_times = motion_times
            else:
                assert False, "Unsupported state initialization strategy: {:s}".format(
                    str(self.state_init)
                )

            self.simulator.reset_envs(new_states, env_ids)

            self.self_obs_cb.reset_envs(
                env_ids,
                reset_default_env_ids,
                reset_ref_env_ids,
                reset_ref_motion_ids,
                reset_ref_motion_times,
            )

            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.terminate_buf[env_ids] = 0

        return self.get_obs()

    def reset_ref_state_init(
        self,
        env_ids,
        motion_ids: Optional[Tensor] = None,
        motion_times: Optional[Tensor] = None,
    ):
        assert not (
            motion_ids is None and motion_times is not None
        ), "Motion times are set, but no corresponding motion ids provided."
        if motion_ids is not None:
            motion_times = self.motion_lib.sample_time(motion_ids)
            self.motion_manager.motion_ids[env_ids] = motion_ids
            self.motion_manager.motion_times[env_ids] = motion_times

        motion_ids, motion_times = self.motion_manager.get_respawn_info(env_ids)

        requires_scene = None
        if self.scene_lib is not None:
            # Sample scenes for each motion
            requires_scene = self.get_motion_requires_scene(motion_ids)

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)

        root_offset = ref_state.root_pos[:, :2].clone()

        # Ignore root offset from the data
        ref_state.root_pos[:, :2] = 0
        # Sample offset and fix spawn height
        ref_state.root_pos[:, :3] += self.get_envs_respawn_position(
            env_ids,
            rigid_body_pos=ref_state.rigid_body_pos,
            offset=root_offset,
            requires_scene=requires_scene,
        )

        # Transfer entire body to the proper coordinates
        ref_state.rigid_body_pos[:, :, :3] -= (
            ref_state.rigid_body_pos[:, 0, :3].unsqueeze(1).clone()
        )
        ref_state.rigid_body_pos[:, :, :3] += ref_state.root_pos.unsqueeze(1)

        return ref_state, motion_ids, motion_times

    def reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = torch_utils.to_torch(
            np.array([self.hybrid_init_prob] * num_envs), device=self.device
        )
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        motion_ids = motion_times = []

        default_state = self.default_state
        # Create placeholder for all states
        new_states = RobotState(
            root_pos=default_state.root_pos.clone(),
            root_rot=default_state.root_rot.clone(),
            dof_pos=default_state.dof_pos.clone(),
            root_vel=default_state.root_vel.clone(),
            root_ang_vel=default_state.root_ang_vel.clone(),
            dof_vel=default_state.dof_vel.clone(),
            rigid_body_pos=default_state.rigid_body_pos.clone(),
            rigid_body_rot=default_state.rigid_body_rot.clone(),
            rigid_body_vel=default_state.rigid_body_vel.clone(),
            rigid_body_ang_vel=default_state.rigid_body_ang_vel.clone(),
        )

        # Each reset function will fill in the appropriate states for the reset ids
        if len(ref_reset_ids) > 0:
            ref_states, motion_ids, motion_times = self.reset_ref_state_init(
                ref_reset_ids
            )
            for key in new_states.keys():
                new_states[key][ref_reset_ids] = ref_states[key]

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            default_states = self.reset_default(default_reset_ids)
            for key in new_states.keys():
                new_states[key][default_reset_ids] = default_states[key]

        # Only return the states for the envs that are being reset
        for key in new_states.keys():
            new_states[key] = new_states[key][env_ids]

        return new_states, default_reset_ids, ref_reset_ids, motion_ids, motion_times

    ###############################################################
    # Terrain Helpers
    ###############################################################
    def create_terrain_and_scene_lib(self):
        terrain_config = TerrainConfig.from_dict(self.config.terrain.config)
        TerrainClass = get_class(self.config.terrain._target_)
        self.terrain: Terrain = TerrainClass(
            config=terrain_config,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.scene_lib: SceneLib = None

    def create_motion_manager(self):
        self.motion_manager = MotionManager(self.config.motion_manager, self)

    ###############################################################
    # Helpers
    ###############################################################
    def create_visualization_markers(self):
        if self.config.headless:
            return {}
        
        visualization_markers = {}
        
        terrain_markers = []
        for _ in range(self.terrain.num_height_points):
            terrain_markers.append(MarkerConfig(size="small"))
        terrain_markers_cfg = VisualizationMarker(
            type="sphere",
            color=(0.008, 0.345, 0.224),
            markers=terrain_markers
        )
        visualization_markers["terrain_markers"] = terrain_markers_cfg
        
        return visualization_markers

    def build_termination_heights(self):
        head_term_height = self.config.head_termination_height
        termination_height = self.config.termination_height

        termination_heights = np.array([termination_height] * self.simulator.robot_config.num_bodies)

        head_id = self.config.robot.body_names.index(self.config.robot.head_body_name)

        termination_heights[head_id] = max(
            head_term_height, termination_heights[head_id]
        )

        self.termination_heights = torch_utils.to_torch(
            termination_heights, device=self.device
        )

    def sync_motion(self):
        if self.sync_motion_just_reset.any():
            if (
                self.config.motion_manager.motion_index_offset is not None
                or self.config.motion_manager.fixed_motion_id is not None
            ):
                num_motions = self.motion_lib.num_motions()
                motion_ids = torch.arange(
                    self.num_envs, dtype=torch.long, device=self.device
                )
                if self.config.motion_manager.motion_index_offset is not None:
                    motion_ids += self.config.motion_manager.motion_index_offset
                motion_ids = torch.fmod(motion_ids, num_motions)
                if self.config.motion_manager.fixed_motion_id is not None:
                    motion_ids *= 0
                    motion_ids += self.config.motion_manager.fixed_motion_id
                motion_ids = motion_ids[self.sync_motion_just_reset]
            else:
                motion_ids = self.motion_lib.sample_motions(
                    self.sync_motion_just_reset.sum()
                )
                if self.scene_lib is not None:
                    requires_scene = self.get_motion_requires_scene(motion_ids)
                    self.agent_in_scene[self.sync_motion_just_reset] = requires_scene

            self.motion_manager.motion_ids[self.sync_motion_just_reset] = motion_ids

            self.sync_motion_times[self.sync_motion_just_reset] = 0
            self.sync_motion_just_reset[:] = False

        self.motion_manager.motion_times[:] = self.sync_motion_times

        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids,
            self.motion_manager.motion_times,
        )

        ref_state.root_vel *= 0
        ref_state.root_ang_vel *= 0
        ref_state.dof_vel *= 0
        ref_state.rigid_body_vel *= 0
        ref_state.rigid_body_ang_vel *= 0

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Transfer to the proper coordinates
        root_offset = ref_state.root_pos[:, :2].clone()

        has_scene = self.agent_in_scene

        if torch.any(has_scene):
            ref_state.root_pos[has_scene, :2] = 0

            scene_position = self.get_envs_respawn_position(
                env_ids[has_scene],
                rigid_body_pos=ref_state.rigid_body_pos[has_scene],
                offset=root_offset[has_scene],
                requires_scene=has_scene[has_scene],
            )
            ref_state.root_pos[has_scene, :3] += scene_position

            ref_state.rigid_body_pos[has_scene, :, :3] -= (
                ref_state.rigid_body_pos[has_scene, 0, :3].unsqueeze(1).clone()
            )
            ref_state.rigid_body_pos[has_scene, :, :3] += ref_state.root_pos.unsqueeze(
                1
            )

        # TODO: for non-scene interactions, sample an initial random offset and then add terrain offset.
        self.respawn_offset_relative_to_data[~has_scene, :] = 0

        self.simulator.reset_envs(ref_state, env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.terminate_buf[env_ids] = 0

        self.sync_motion_times = self.sync_motion_times + self.sync_motion_dt
        # Check for motions that wrapped around
        wrapped_motions = (
            self.sync_motion_times
            >= self.motion_lib.state.motion_lengths[self.motion_manager.motion_ids]
        )
        # Set sync_motion_just_reset to True for wrapped motions
        self.sync_motion_just_reset[wrapped_motions] = True

    def instantiate_motion_lib(self):
        # CT hack: we do not use hydra.instantiate here because we need to pass the robot_config
        # the robot_config can not be parsed by OmegaConf/Hydra, which causes a failure.
        MotionLibClass = get_class(self.config.motion_lib._target_)
        motion_lib_params = {}
        for key, value in self.config.motion_lib.items():
            if key != "_target_":
                motion_lib_params[key] = value

        motion_lib: MotionLib = MotionLibClass(
            robot_config=self.simulator.robot_config,
            key_body_ids=self.key_body_ids,
            device=self.device,
            skeleton_tree=None,
            **motion_lib_params
        )
        return motion_lib

    def get_state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
