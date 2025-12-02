# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.markers import VisualizationMarkers as IsaacLabVisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg as IsaacLabVisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from protomotions.components.terrains.terrain import Terrain
from protomotions.components.scene_lib import (
    SceneLib,
    MeshSceneObject,
    BoxSceneObject,
    SphereSceneObject,
    CylinderSceneObject,
)
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from protomotions.simulator.isaaclab.utils.scene import SceneCfg
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimulatorConfig,
    ProtoMotionsIsaacLabMarkers,
)
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimBodyOrdering,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    StateConversion,
    ObjectState,
    ResetState,
)


class IsaacLabSimulator(Simulator):
    config: IsaacLabSimulatorConfig

    # =====================================================
    # Group 1: Initialization & Configuration
    # =====================================================
    def __init__(
        self,
        config: IsaacLabSimulatorConfig,
        robot_config,
        terrain: Terrain,
        device: torch.device,
        simulation_app: Any,
        scene_lib: SceneLib,
        custom_key_handlers: Optional[Dict[str, callable]] = None,
    ) -> None:
        """
        Initialize the IsaacLabSimulator shell.

        Parameters:
            config (SimulatorConfig): The configuration dictionary.
            robot_config (RobotConfig): The robot configuration.
            terrain (Terrain): Terrain data for simulation.
            device (torch.device): Device to use for computation.
            simulation_app (Any): The simulation application instance.
            scene_lib (SceneLib): Scene library (always provided, can be empty).
        """
        super().__init__(
            config=config,
            robot_config=robot_config,
            scene_lib=scene_lib,
            terrain=terrain,
            device=device,
        )

        # Store custom key handlers
        self._custom_key_handlers = custom_key_handlers or {}

        sim_cfg = sim_utils.SimulationCfg(
            device=str(device),
            dt=1.0 / self.config.sim.fps,
            render_interval=self.config.sim.decimation,
            physx=PhysxCfg(
                solver_type=self.config.sim.physx.solver_type,
                max_position_iteration_count=self.config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.config.sim.physx.num_velocity_iterations,
                bounce_threshold_velocity=self.config.sim.physx.bounce_threshold_velocity,
                gpu_max_rigid_contact_count=self.config.sim.physx.gpu_max_rigid_contact_count,
                gpu_found_lost_pairs_capacity=self.config.sim.physx.gpu_found_lost_pairs_capacity,
                gpu_found_lost_aggregate_pairs_capacity=self.config.sim.physx.gpu_found_lost_aggregate_pairs_capacity,
            ),
        )
        self._simulation_app = simulation_app
        self._sim = SimulationContext(sim_cfg)
        self._sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

        scene_cfg = self._get_scene_cfg()

        self._scene = InteractiveScene(scene_cfg)
        if not self.headless:
            self._setup_keyboard()
        print("[INFO]: Setup complete...")

    def _create_simulation(self) -> None:
        """Create the IsaacLab simulation environment.

        Called by base class _initialize_with_markers() after visualization markers
        are set. Completes scene setup and resets simulation.
        """
        self._robot = self._scene["robot"]
        # Build a mapping from body name to contact sensor (if it exists)
        self._contact_sensor_map = {}
        for body_name in self._body_names:
            if f"contact_sensor_{body_name}" in self._scene.keys():
                self._contact_sensor_map[body_name] = self._scene[
                    f"contact_sensor_{body_name}"
                ]

        self._object = []
        self._object_contact_sensor = []
        if self.scene_lib.num_scenes() > 0:
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                self._object.append(self._scene[f"object_{obj_idx}"])
                if f"object_{obj_idx}_contact_sensor" in self._scene.keys():
                    self._object_contact_sensor.append(
                        self._scene[f"object_{obj_idx}_contact_sensor"]
                    )
                else:
                    self._object_contact_sensor.append(None)
        if self._visualization_markers:
            self._build_markers(self._visualization_markers)
        self._sim.reset()

    def _get_scene_cfg(self) -> SceneCfg:
        """
        Construct and return the scene configuration from the current config, scene library, and terrain.

        Returns:
            SceneCfg: The constructed scene configuration.
        """
        scene_cfgs = None
        if self.scene_lib.num_scenes() > 0:
            scene_cfgs, self._initial_scene_pos = self._preprocess_object_playground()

        scene_cfg = SceneCfg(
            config=self.config,
            robot_config=self.robot_config,
            num_envs=self.config.num_envs,
            env_spacing=2.0,
            scene_cfgs=scene_cfgs,
            terrain=self.terrain,
            replicate_physics=scene_cfgs
            is None,  # When there are objects, disable physics replication
        )
        return scene_cfg

    def _preprocess_object_playground(self) -> Tuple[List[Any], torch.Tensor]:
        """
        Process and build the object playground from the scene library.

        Returns:
            Tuple[List[Any], torch.Tensor]: A tuple containing the object configurations and the initial object positions.
        """
        print("=========== Building object playground")

        # Spawn objects at origin (actual positions set via reset_envs later)
        initial_obj_pos = torch.zeros(
            (self.num_envs, self.scene_lib.num_objects_per_scene, 7),
            device=self.device,
            dtype=torch.float,
        )
        # Set identity quaternions (wxyz format for IsaacLab)
        initial_obj_pos[..., 3] = 1.0  # w=1 for identity quaternion

        # Build object configurations for IsaacLab
        objects_cfgs = []
        for _ in range(self.scene_lib.num_objects_per_scene):
            objects_cfgs.append([])

        for env_id, scene in enumerate(self.scene_lib.scenes):
            for obj_idx, obj in enumerate(scene.objects):
                # Common properties based on object options
                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=obj.options.fix_base_link,
                )
                collision_props = sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                )

                # Handle different object types
                if isinstance(obj, MeshSceneObject):
                    main_dir_path = (
                        f"{os.path.dirname(os.path.abspath(__file__))}/../../../"
                    )
                    asset_path = Path(
                        os.path.join(main_dir_path, obj.object_path)
                    ).resolve()

                    spawn_cfg = sim_utils.UsdFileCfg(
                        usd_path=str(asset_path),
                        rigid_props=rigid_props,
                        collision_props=collision_props,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 0.7, 0.3), metallic=0.2
                        ),
                    )
                elif isinstance(obj, BoxSceneObject):
                    spawn_cfg = sim_utils.CuboidCfg(
                        size=(obj.width, obj.depth, obj.height),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.8, 0.3, 0.3), metallic=0.2
                        ),
                        rigid_props=rigid_props,
                        mass_props=sim_utils.MassPropertiesCfg(mass=-1, density=100),
                        collision_props=collision_props,
                    )
                elif isinstance(obj, SphereSceneObject):
                    spawn_cfg = sim_utils.SphereCfg(
                        radius=obj.radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.3, 0.3, 0.8), metallic=0.2
                        ),
                        rigid_props=rigid_props,
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=-1, density=obj.options.density
                        ),
                        collision_props=collision_props,
                    )
                elif isinstance(obj, CylinderSceneObject):
                    spawn_cfg = sim_utils.CylinderCfg(
                        radius=obj.radius,
                        height=obj.height,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.3, 0.8, 0.3), metallic=0.2
                        ),
                        rigid_props=rigid_props,
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=-1, density=obj.options.density
                        ),
                        collision_props=collision_props,
                    )
                else:
                    raise ValueError(f"Unsupported object type: {type(obj)}")

                objects_cfgs[obj_idx].append(spawn_cfg)

        return objects_cfgs, initial_obj_pos

    def _setup_keyboard(self) -> None:
        """
        Set up keyboard callbacks for control using the Se2Keyboard interface.
        """
        from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard

        try:
            # Try Isaac Sim 5.0.0+ API - requires a cfg parameter with sim_device
            from dataclasses import dataclass

            @dataclass
            class Se2KeyboardCfg:
                v_x_sensitivity: float = 0.8
                v_y_sensitivity: float = 0.4
                omega_z_sensitivity: float = 1.0
                sim_device: str = "cuda:0"  # Add required sim_device attribute

            cfg = Se2KeyboardCfg()
            self.keyboard_interface = Se2Keyboard(cfg=cfg)
        except (TypeError, ImportError):
            try:
                # Try older API without cfg parameter
                self.keyboard_interface = Se2Keyboard()
            except TypeError:
                # Fallback for older versions with individual parameters
                self.keyboard_interface = Se2Keyboard(
                    v_x_sensitivity=0.8, v_y_sensitivity=0.4, omega_z_sensitivity=1.0
                )

        self.keyboard_interface.add_callback("R", self._requested_reset)
        self.keyboard_interface.add_callback("L", self._toggle_video_record)
        self.keyboard_interface.add_callback(";", self._cancel_video_record)
        self.keyboard_interface.add_callback("Q", self.close)
        self.keyboard_interface.add_callback("O", self._toggle_camera_target)
        self.keyboard_interface.add_callback("J", self._push_robot)
        self.keyboard_interface.add_callback("M", self._toggle_markers)

        # Register custom key handlers for keys 1-0
        self._register_custom_key_handlers()

    def _register_custom_key_handlers(self) -> None:
        """Register custom keyboard event handlers for keys 1-0"""
        # Define available keys for custom handlers (1-0)
        available_keys = {
            "1": "NUMPAD_1",
            "2": "NUMPAD_2",
            "3": "NUMPAD_3",
            "4": "NUMPAD_4",
            "5": "NUMPAD_5",
            "6": "NUMPAD_6",
            "7": "NUMPAD_7",
            "8": "NUMPAD_8",
            "9": "NUMPAD_9",
            "0": "NUMPAD_0",
        }

        # Register custom key handlers
        for key_name, handler in self._custom_key_handlers.items():
            if key_name in available_keys.keys():
                try:
                    self.keyboard_interface.add_callback(
                        available_keys[key_name], handler
                    )
                    print(f"Registered custom key handler for '{key_name}'")
                    # input()
                except Exception as e:
                    print(
                        f"Warning: Failed to register custom key handler for '{key_name}': {e}"
                    )
            else:
                print(f"Warning: Key '{key_name}' not available for custom handlers")
                print(f"Available keys: {list(available_keys.keys())}")

    # =====================================================
    # Group 2: Environment Setup & Configuration
    # =====================================================
    def _finalize_setup(self) -> None:
        """
        Configure initial environment settings when the simulation is ready.
        This includes setting up joint limits and initializing state tensors.
        """
        super()._finalize_setup()

        # Update initial object positions
        if self.scene_lib.num_scenes() > 0:
            objects_start_pos = torch.zeros(
                (self.num_envs, 13), device=self.device, dtype=torch.float
            )
            for obj_idx, object in enumerate(self._object):
                objects_start_pos[:, :7] = self._initial_scene_pos[:, obj_idx, :]
                object.write_root_state_to_sim(objects_start_pos)

        self._apply_domain_randomization_if_needed()

    def _apply_domain_randomization_if_needed(self) -> None:
        all_env_ids = torch.arange(self.config.num_envs, dtype=torch.int)
        if (
            self._domain_randomization is not None
            and "friction" in self._domain_randomization
        ):
            # Adapted from https://github.com/isaac-sim/IsaacLab/blob/be083bf1f70466e1d41bf9ffdc405bb89394e92c/source/isaaclab/isaaclab/envs/mdp/events.py#L203
            num_shapes_per_body = []
            for link_path in self._robot.root_physx_view.link_paths[0]:
                link_physx_view = self._robot._physics_sim_view.create_rigid_body_view(
                    link_path
                )
                num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(num_shapes_per_body)
            expected_shapes = self._robot.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )

            materials = self._robot.root_physx_view.get_material_properties()
            body_names = [
                self.robot_config.kinematic_info.body_names[
                    self._domain_randomization["friction"]["body_indices"][idx]
                ]
                for idx in range(
                    len(self._domain_randomization["friction"]["body_indices"])
                )
            ]
            isaaclab_body_ids, _ = self._robot.find_bodies(
                body_names, preserve_order=True
            )
            for idx in range(
                len(self._domain_randomization["friction"]["body_indices"])
            ):
                # bodies may span multiple "shapes" in the physx view, so we need to assign the materials to the correct shapes
                start_idx = sum(num_shapes_per_body[: isaaclab_body_ids[idx]])
                end_idx = start_idx + num_shapes_per_body[isaaclab_body_ids[idx]]

                num_buckets = self._domain_randomization["friction"][
                    "static_friction"
                ].shape[0]
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,))
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[:, start_idx:end_idx, 0] = self._domain_randomization[
                    "friction"
                ]["static_friction"][bucket_ids, idx].unsqueeze(-1)
                materials[:, start_idx:end_idx, 1] = self._domain_randomization[
                    "friction"
                ]["dynamic_friction"][bucket_ids, idx].unsqueeze(-1)
                materials[:, start_idx:end_idx, 2] = self._domain_randomization[
                    "friction"
                ]["restitution"][bucket_ids, idx].unsqueeze(-1)
            self._robot.root_physx_view.set_material_properties(
                materials, indices=all_env_ids
            )

        if (
            self._domain_randomization is not None
            and "center_of_mass" in self._domain_randomization
        ):
            # get the current com of the bodies (num_assets, num_bodies)
            coms = self._robot.root_physx_view.get_coms().clone()

            # Randomize the com in range
            coms[
                :, self._domain_randomization["center_of_mass"]["body_indices"], :3
            ] += self._domain_randomization["center_of_mass"]["com"].to(coms.device)

            # Set the new comsfa
            self._robot.root_physx_view.set_coms(coms, all_env_ids)

    # =====================================================
    # Group 3: Simulation Steps & State Management
    # =====================================================
    def _physics_step(self) -> None:
        """
        Advance the simulation by stepping for a number of iterations equal to the decimation factor.
        """
        for idx in range(self.decimation):
            self._apply_control()
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            if (idx + 1) % self.decimation == 0 and not self.headless:
                self._sim.render()
            self._scene.update(dt=self._sim.get_physics_dt())

    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Applies PD position targets using IsaacLab's internal PD controller."""
        self._robot.set_joint_position_target(pd_targets, joint_ids=None)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Applies torques to the robot DOFs."""
        self._robot.set_joint_effort_target(torques, joint_ids=None)

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: ObjectState = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """
        Apply the provided state to the simulation by writing root and joint states.

        Parameters:
            new_states (ResetState): The new simulation state.
            new_object_states (ObjectState): The new object state.
            env_ids (torch.Tensor): Specific environment IDs to update.
        """
        init_root_state = torch.cat(
            [
                new_states.root_pos,
                new_states.root_rot,
                new_states.root_vel,
                new_states.root_ang_vel,
            ],
            dim=-1,
        )
        self._robot.write_root_state_to_sim(init_root_state, env_ids)
        self._robot.set_joint_position_target(
            new_states.dof_pos, joint_ids=None, env_ids=env_ids
        )
        self._robot.write_joint_state_to_sim(
            new_states.dof_pos, new_states.dof_vel, None, env_ids
        )
        if new_object_states is not None:
            init_object_root_state = torch.cat(
                [
                    new_object_states.root_pos,
                    new_object_states.root_rot,
                    new_object_states.root_vel,
                    new_object_states.root_ang_vel,
                ],
                dim=-1,
            ).reshape(len(env_ids), self.scene_lib.num_objects_per_scene, 13)
            for object_idx in range(self.scene_lib.num_objects_per_scene):
                self._object[object_idx].write_root_state_to_sim(
                    init_object_root_state[:, object_idx], env_ids
                )

    # =====================================================
    # Group 4: State Getters
    # =====================================================
    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Obtain the ordering of body and degree-of-freedom names.

        Returns:
            SimBodyOrdering: An object containing the body names and DOF names.
        """
        return SimBodyOrdering(
            body_names=self._robot.data.body_names,
            dof_names=self._robot.data.joint_names,
        )

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions, rotations, velocities) of all simulation bodies.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The state of the bodies.
        """
        isaacsim_bodies_positions = self._robot.data.body_pos_w.clone()
        isaacsim_bodies_rotations = self._robot.data.body_quat_w.clone()
        isaacsim_bodies_velocities = self._robot.data.body_lin_vel_w.clone()
        isaacsim_bodies_ang_velocities = self._robot.data.body_ang_vel_w.clone()

        isaacsim_bodies_positions = isaacsim_bodies_positions.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_rotations = isaacsim_bodies_rotations.view(
            self.num_envs, self._num_bodies, 4
        )
        isaacsim_bodies_velocities = isaacsim_bodies_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities.view(
            self.num_envs, self._num_bodies, 3
        )
        if env_ids is not None:
            isaacsim_bodies_positions = isaacsim_bodies_positions[env_ids]
            isaacsim_bodies_rotations = isaacsim_bodies_rotations[env_ids]
            isaacsim_bodies_velocities = isaacsim_bodies_velocities[env_ids]
            isaacsim_bodies_ang_velocities = isaacsim_bodies_ang_velocities[env_ids]
        return RobotState(
            rigid_body_pos=isaacsim_bodies_positions,
            rigid_body_rot=isaacsim_bodies_rotations,
            rigid_body_vel=isaacsim_bodies_velocities,
            rigid_body_ang_vel=isaacsim_bodies_ang_velocities,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve applied torque forces for the robot's degrees of freedom.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict query to specific environments if provided.

        Returns:
            torch.Tensor: The DOF forces.
        """
        isaacsim_dof_forces = self._robot.data.applied_torque.clone()
        if env_ids is not None:
            isaacsim_dof_forces = isaacsim_dof_forces[env_ids]
        return RobotState(
            dof_forces=isaacsim_dof_forces, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the state (positions and velocities) of the robot's DOFs.

        Parameters:
            env_ids (Optional[torch.Tensor]): Restrict state retrieval to specific environments if provided.

        Returns:
            RobotState: The DOF state.
        """
        isaacsim_dof_pos = self._robot.data.joint_pos.clone()
        isaacsim_dof_vel = self._robot.data.joint_vel.clone()
        if env_ids is not None:
            isaacsim_dof_pos = isaacsim_dof_pos[env_ids]
            isaacsim_dof_vel = isaacsim_dof_vel[env_ids]
        return RobotState(
            dof_pos=isaacsim_dof_pos,
            dof_vel=isaacsim_dof_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from IsaacLab's internal API for verification purposes only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in IsaacLab's DOF ordering.
        """
        # Extract limits from the robot data
        dof_limits = self._robot.data.joint_pos_limits.clone()
        # IsaacLab stores limits as [num_envs, num_dofs, 2], we take from first env
        return dof_limits[0, :, 0].to(self.device), dof_limits[0, :, 1].to(self.device)

    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the contact force buffer for simulation bodies in sim body order.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RobotState: Robot state containing contact forces in simulator body order.
        """
        # Get simulator body ordering
        sim_body_names = self._robot.data.body_names
        num_bodies = len(sim_body_names)

        # Pre-allocate tensor for contact forces (initialized to zeros)
        rigid_body_contact_forces = torch.zeros(
            self.num_envs, num_bodies, 3, device=self.device
        )

        # Fill in contact forces for bodies that have sensors
        for body_idx, body_name in enumerate(sim_body_names):
            if body_name in self._contact_sensor_map:
                contact_sensor = self._contact_sensor_map[body_name]
                # net_forces_w has shape [num_envs, 1, 3], extract the single body dimension
                rigid_body_contact_forces[:, body_idx, :] = (
                    contact_sensor.data.net_forces_w.clone()[:, 0, :]
                )

        if env_ids is not None:
            rigid_body_contact_forces = rigid_body_contact_forces[env_ids]
        return RobotState(
            rigid_body_contact_forces=rigid_body_contact_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(
        self,
        env_ids: Optional[torch.Tensor] = None,
    ) -> ObjectState:
        """
        Retrieve the contact buffer for simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            torch.Tensor: The object contact buffer.
        """
        if self.scene_lib.num_scenes() > 0:
            object_forces = []
            for obj_idx in range(self.scene_lib.num_objects_per_scene):
                if self._object_contact_sensor[obj_idx] is not None:
                    object_forces.append(
                        self._object_contact_sensor[obj_idx].data.force_matrix_w.clone()
                    )
                else:
                    object_forces.append(
                        torch.zeros(
                            self.num_envs,
                            1,
                            1,
                            3,
                            device=self.device,
                            dtype=torch.float,
                        )
                    )
            if env_ids is not None:
                object_forces = object_forces[env_ids]
            return torch.cat(object_forces, dim=1)
        else:
            return_tensor = torch.zeros(
                self.num_envs, 1, 1, 3, device=self.device, dtype=torch.float
            )
            if env_ids is not None:
                return_tensor = return_tensor[env_ids]
            return return_tensor

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """
        Retrieve the root state (position, rotation, velocity) of the robot.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            RootOnlyState: The robot's root state.
        """
        isaacsim_root_pos = self._robot.data.root_pos_w.clone()
        isaacsim_root_rot = self._robot.data.root_quat_w.clone()
        isaacsim_root_vel = self._robot.data.root_lin_vel_w.clone()
        isaacsim_root_ang_vel = self._robot.data.root_ang_vel_w.clone()
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return RootOnlyState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the combined root state for all simulation objects.

        Parameters:
            env_ids (Optional[torch.Tensor]): Specific environments to query.

        Returns:
            ObjectState: The objects' root state.
        """
        isaacsim_root_pos = []
        isaacsim_root_rot = []
        isaacsim_root_vel = []
        isaacsim_root_ang_vel = []
        for obj_idx in range(self.scene_lib.num_objects_per_scene):
            isaacsim_root_pos.append(self._object[obj_idx].data.root_pos_w.clone())
            isaacsim_root_rot.append(self._object[obj_idx].data.root_quat_w.clone())
            isaacsim_root_vel.append(self._object[obj_idx].data.root_lin_vel_w.clone())
            isaacsim_root_ang_vel.append(
                self._object[obj_idx].data.root_ang_vel_w.clone()
            )
        isaacsim_root_pos = torch.stack(isaacsim_root_pos, dim=1)
        isaacsim_root_rot = torch.stack(isaacsim_root_rot, dim=1)
        isaacsim_root_vel = torch.stack(isaacsim_root_vel, dim=1)
        isaacsim_root_ang_vel = torch.stack(isaacsim_root_ang_vel, dim=1)
        if env_ids is not None:
            isaacsim_root_pos = isaacsim_root_pos[env_ids]
            isaacsim_root_rot = isaacsim_root_rot[env_ids]
            isaacsim_root_vel = isaacsim_root_vel[env_ids]
            isaacsim_root_ang_vel = isaacsim_root_ang_vel[env_ids]
        return ObjectState(
            root_pos=isaacsim_root_pos,
            root_rot=isaacsim_root_rot,
            root_vel=isaacsim_root_vel,
            root_ang_vel=isaacsim_root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def get_num_actors_per_env(self) -> int:
        """
        Compute and return the number of actor instances per environment.

        Returns:
            int: Number of actors per environment.
        """
        root_pos = self._robot.data.root_pos_w
        return root_pos.shape[0] // self.num_envs

    # =====================================================
    # Group 5: Control & Computation Methods
    # =====================================================

    def _push_robot(self):
        vel_w = self._robot.data.root_vel_w
        self._robot.write_root_velocity_to_sim(
            vel_w + torch.ones_like(vel_w),
            env_ids=torch.arange(self.num_envs, device=self.device),
        )

    # =====================================================
    # Group 6: Rendering & Visualization
    # =====================================================
    def render(self) -> None:
        """
        Render the simulation view. Initializes or updates the camera if the simulator is not in headless mode.
        """
        if not self.headless:
            if not hasattr(self, "_perspective_view"):
                from protomotions.simulator.isaaclab.utils.perspective_viewer import (
                    PerspectiveViewer,
                )

                self._perspective_view = PerspectiveViewer()
                self._init_camera()
            else:
                self._update_camera()
        super().render()

    def _init_camera(self) -> None:
        """
        Initialize the camera view based on the current simulation root state.
        """
        self._cam_prev_char_pos = (
            self._get_simulator_root_state(0).root_pos.cpu().numpy()
        )
        pos = self._cam_prev_char_pos + np.array([0, -5, 1])
        self._perspective_view.set_camera_view(
            pos, self._cam_prev_char_pos + np.array([0, 0, 0.2])
        )

    def _update_camera(self) -> None:
        """
        Update the camera view based on the target's position and current camera movement.
        """
        if self._camera_target["element"] == 0:
            char_root_pos = (
                self._get_simulator_root_state(self._camera_target["env"])
                .root_pos.cpu()
                .numpy()
            )
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            char_root_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .cpu()
                .numpy()
            )
            height_offset = 0

        cam_pos = np.array(self._perspective_view.get_camera_state())
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = np.array(
            [char_root_pos[0], char_root_pos[1], char_root_pos[2] + height_offset]
        )
        new_cam_pos = np.array(
            [
                char_root_pos[0] + cam_delta[0],
                char_root_pos[1] + cam_delta[1],
                char_root_pos[2] + cam_delta[2],
            ]
        )
        self._perspective_view.set_camera_view(new_cam_pos, new_cam_target)
        self._cam_prev_char_pos[:] = char_root_pos

    def _write_viewport_to_file(self, file_name: str) -> None:
        """
        Capture the current viewport and save it to the specified file.

        Parameters:
            file_name (str): The filename for the saved image.
        """
        from omni.kit.viewport.utility import (
            get_active_viewport,
            capture_viewport_to_file,
        )

        vp_api = get_active_viewport()
        capture_viewport_to_file(vp_api, file_name)

    def close(self) -> None:
        """
        Close the simulation application and perform cleanup.
        """
        super().close()
        self._simulation_app.close()

    def _build_markers(
        self, visualization_markers: Dict[str, VisualizationMarkerConfig]
    ) -> None:
        """Build and configure visualization markers.

        Args:
            visualization_markers (Dict[str, VisualizationMarkerConfig]): Dictionary mapping marker names to their configurations
        """
        self._visualization_markers: Dict[str, ProtoMotionsIsaacLabMarkers] = {}
        if visualization_markers is None:
            return

        for marker_name, markers_cfg in visualization_markers.items():
            if markers_cfg.type == "sphere":
                marker_obj_cfg = IsaacLabVisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "marker": sim_utils.SphereCfg(
                            radius=1,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                )
                            ),
                        ),
                    },
                )
            elif markers_cfg.type == "arrow":
                marker_obj_cfg = IsaacLabVisualizationMarkersCfg(
                    prim_path=f"/Visuals/{marker_name}",
                    markers={
                        "marker": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=(1.0, 1.0, 1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(
                                    markers_cfg.color[0],
                                    markers_cfg.color[1],
                                    markers_cfg.color[2],
                                ),
                                opacity=0.5,
                            ),
                        ),
                    },
                )
            else:
                raise ValueError(f"Marker type {markers_cfg.type} not supported")

            marker_scale = []
            for i, marker in enumerate(markers_cfg.markers):
                if markers_cfg.type == "sphere":
                    if marker.size == "tiny":
                        scale = 0.007
                    elif marker.size == "small":
                        scale = 0.01
                    else:
                        scale = 0.05
                    marker_scale.append([scale, scale, scale])
                elif markers_cfg.type == "arrow":
                    if marker.size == "small":
                        scale = 0.1
                    else:
                        scale = 0.5
                    marker_scale.append([scale, 0.2 * scale, 0.2 * scale])

            if len(marker_scale) == 0:
                continue

            self._visualization_markers[marker_name] = ProtoMotionsIsaacLabMarkers(
                marker=IsaacLabVisualizationMarkers(marker_obj_cfg),
                scale=torch.tensor(marker_scale, device=self.device).repeat(
                    self.num_envs, 1
                ),
            )

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Update the visualization markers with new state information.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary mapping marker names to their state (translation and orientation)
        """
        if markers_state is None:
            return

        for marker_name, markers_state_item in markers_state.items():
            if markers_state_item.translation.numel() == 0:
                continue
            assert (
                marker_name in self._visualization_markers
            ), f"Marker {marker_name} passed to update_markers but not defined at instantiation"
            marker_dict = self._visualization_markers[marker_name]
            marker_dict.marker.visualize(
                translations=markers_state_item.translation.view(-1, 3),
                orientations=markers_state_item.orientation.view(-1, 4),
                scales=marker_dict.scale,
            )
