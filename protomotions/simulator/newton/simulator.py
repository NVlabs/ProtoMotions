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
import os
import torch
import numpy as np
from rich.progress import Progress
import sys
from typing import Dict, Optional, Tuple

from protomotions.components.scene_lib import SceneLib, MeshSceneObject
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimBodyOrdering,
)
from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    RootOnlyState,
    StateConversion,
    ObjectState,
    ResetState,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.simulator.newton.config import NewtonSimulatorConfig
import openmesh
import warp as wp
import newton
from newton.selection import ArticulationView
from newton import Contacts
from newton.sensors import ContactSensor, populate_contacts
import copy


wp.config.enable_backward = False
wp.config.quiet = True


def convert_to_indexed_mesh(vertices, triangles):
    # Flatten the triangles tensor to get a single list of vertex indices
    vertex_indices = triangles.flatten()

    # Get the unique vertices and their corresponding indices
    unique_points, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)

    # Map the original vertex indices to the new unique point indices
    new_indices = inverse_indices[vertex_indices]

    # Reshape the new indices to form the indexed triangles
    indexed_triangles = new_indices.reshape(-1, 3)

    return unique_points, indexed_triangles


class NewtonSimulator(Simulator):
    """Newton physics engine wrapper for our simulation framework."""

    config: NewtonSimulatorConfig

    def __init__(
        self,
        config: NewtonSimulatorConfig,
        robot_config,
        terrain: Terrain,
        device: torch.device,
        scene_lib: SceneLib,
        custom_key_handlers: Optional[Dict[str, callable]] = None,
    ) -> None:
        super().__init__(
            config=config,
            robot_config=robot_config,
            scene_lib=scene_lib,
            terrain=terrain,
            device=device,
        )

        self._custom_key_handlers = custom_key_handlers or {}
        self._any_key_pressed = False  # used to avoid repeating the same key press

        # Configure timing
        self.sim_time = 0.0
        self.sim_dt = 1.0 / self.config.sim.fps
        self.decimation = self.config.sim.decimation
        self.frame_dt = self.sim_dt * self.decimation

        self._contact_sensors = {}
        self._contact_forces = {}  # Store contact forces per body
        self.contacts = Contacts(0, 0)  # Initialize contacts storage
        self._camera_initialized = False

    def _create_simulation(self) -> None:
        """Create the Newton simulation environment.

        Called by base class _initialize_with_markers() after visualization markers
        are set. Creates environments, robot, and simulation.
        """
        self._create_envs()
        self._setup_robot()
        self._setup_sim()
        if self.robot_config.contact_bodies is not None:
            self._setup_contact_sensors()

        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            torch_tensor = torch.zeros(
                self.num_envs,
                self.robot_config.number_of_actions,
                device=self.device,
                dtype=torch.float32,
            )
            a_wp = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            self.robot_view.set_attribute("joint_target_pos", self.control, a_wp)

            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    def _create_envs(self) -> None:
        """Creates environments and loads robot assets."""
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name
        asset_path = os.path.join(asset_root, asset_file)

        print(f"Loading robot from: {asset_path}")

        # Initialize model builder
        self.use_mujoco = False
        self.robot = newton.ModelBuilder(up_axis=newton.Axis.Z)

        # Set default joint configuration
        self.robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig()

        self.robot.add_mjcf(
            asset_path,
            ignore_names=["floor", "ground"],
            collapse_fixed_joints=False,  # We manually filter out and report only the joints/bodies we are interested in.
            floating=not self.robot_config.asset.fix_base_link,
            enable_self_collisions=self.robot_config.asset.self_collisions,
        )
        # Set the articulation key to "robot", allowing easy wrapping with ArticulationView
        self.robot.articulation_key = ["robot"]

        self.robot.approximate_meshes("convex_hull")

        self._object_assets = {}
        self._static_object_poses = []
        if self.scene_lib.num_scenes() > 0:
            self._load_object_assets()

        # Replicate robot across.
        # We set env-spacing to 0 so all are aligned in global space.
        builder = newton.ModelBuilder()
        builder.current_env_group = -1
        self._add_terrain(builder)
        for env_id in range(self.num_envs):
            builder.current_env_group = env_id
            builder.add_builder(self.robot, world=env_id)

            if self.scene_lib.num_scenes() > 0:
                scene = self.scene_lib.scenes[env_id]

                # Get scene offset for dummy spawn position
                scene_offset_x, scene_offset_y = self.scene_lib.scene_offsets[env_id]

                env_static_object_poses = []
                for obj_idx, obj in enumerate(scene.objects):
                    object_asset = self._object_assets[obj.first_instance_id]

                    # Spawn at scene offset (x,y) with z=0 to avoid collision, actual pose via reset
                    object_pose = torch.tensor(
                        [scene_offset_x, scene_offset_y, 0.0, 0.0, 0.0, 0.0, 1.0],
                        device=self.device,
                        dtype=torch.float,
                    )
                    if obj.options.fix_base_link:
                        env_static_object_poses.append(object_pose)

                    if isinstance(object_asset, newton.Mesh):
                        if not obj.options.fix_base_link:
                            builder.add_articulation(key=f"object_{env_id}_{obj_idx}")
                            body_id = builder.add_body(xform=object_pose)
                            builder.add_joint_free(body_id)
                            xform = None
                        else:
                            body_id = -1
                            xform = object_pose
                        builder.add_shape_mesh(
                            body=body_id, xform=xform, mesh=object_asset
                        )
                    else:
                        if not obj.options.fix_base_link:
                            builder.add_articulation(key=f"object_{env_id}_{obj_idx}")
                            body_id = builder.add_body(xform=object_pose)
                            builder.add_joint_free(body_id)
                            xform = None
                        else:
                            body_id = -1
                            xform = object_pose
                        builder.add_shape_box(
                            body=body_id,
                            xform=xform,
                            hx=obj.width / 2,
                            hy=obj.depth / 2,
                            hz=obj.height / 2,
                        )
                if len(env_static_object_poses) > 0:
                    env_static_object_poses = torch.stack(
                        env_static_object_poses, dim=0
                    )  # [num_static_objects, 7]
                    self._static_object_poses.append(env_static_object_poses)

        if len(self._static_object_poses) > 0:
            self._static_object_poses = torch.stack(
                self._static_object_poses, dim=0
            )  # [num_envs, num_static_objects, 7]

        # Finalize model and create solver
        # TODO: add parameters
        builder.approximate_meshes(method="coacd")
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

    def _load_object_assets(self) -> None:
        if self.scene_lib.num_scenes() > 0:
            self._object_names = []

            # Count assets
            asset_count = sum(
                1 for scene in self.scene_lib.scenes for obj in scene.objects
            )

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Loading object assets...",
                    total=asset_count,
                )

                # Iterate through all scenes and their objects
                for scene in self.scene_lib.scenes:
                    for obj in scene.objects:
                        # Skip if we've already processed this object type
                        if not obj.is_first_instance:
                            progress.update(task, advance=1)
                            continue

                        first_object_id = obj.first_instance_id
                        if isinstance(obj, MeshSceneObject):
                            object_name = os.path.splitext(
                                os.path.basename(obj.object_path)
                            )[0]
                            asset_path = obj.object_path

                            m = openmesh.read_trimesh(asset_path)
                            mesh_points = np.array(m.points())
                            mesh_indices = np.array(
                                m.face_vertex_indices(), dtype=np.int32
                            ).flatten()
                            # TODO: use vhacd or other form of convex decomposition
                            asset = newton.Mesh(mesh_points, mesh_indices)
                        else:
                            # Primitive object (Box, Sphere, Cylinder)
                            object_name = (
                                obj.object_identifier
                            )  # Use object_identifier for primitives
                            asset = obj

                        # Create asset options
                        # TODO

                        # Add force sensor - TODO
                        self._object_names.append(object_name)
                        self._object_assets[first_object_id] = asset

                        progress.update(task, advance=1)

            print(
                f"=========== Total number of unique objects is {len(self._object_assets)}"
            )

    def _setup_robot(self) -> None:
        common_dof_names = copy.deepcopy(self._dof_names)
        newton_dof_names = {}  # Create mapping from newton dof_names to common dof_names. Newton combines multiple dof-joints into a single joint key.

        while len(common_dof_names) > 0:
            common_dof_name = common_dof_names[0]
            if common_dof_name in self.robot.joint_key:
                # We have a precise match!
                newton_dof_names[common_dof_name] = common_dof_name
                common_dof_names.pop(0)
            else:
                # We don't have a precise match, so we need to check for a partial match.
                multi_dof_name = None
                for newton_dof_name in self.robot.joint_key:
                    if common_dof_name in newton_dof_name:
                        multi_dof_name = newton_dof_name
                        break
                assert (
                    multi_dof_name is not None
                ), f"No joint key match found for {common_dof_name} in {self.robot.joint_key}"

                newton_dof_names[multi_dof_name] = []
                while (
                    len(common_dof_names) > 0 and common_dof_names[0] in multi_dof_name
                ):
                    newton_dof_names[multi_dof_name].append(common_dof_names[0])
                    common_dof_names.pop(0)

        self._newton_dof_names = newton_dof_names

        self.robot_view = ArticulationView(
            self.model,
            pattern="robot",
            include_joints=self._newton_dof_names.keys(),
            include_links=self._body_names,
        )

        if self.scene_lib.num_scenes() > 0 and any(
            not obj.options.fix_base_link for obj in self.scene_lib.scenes[0].objects
        ):
            self.object_view = ArticulationView(
                self.model,
                pattern="object_*",
            )
        else:
            self.object_view = None

        # Apply per-DOF control parameters from config
        joint_stiffness = []
        joint_damping = []
        joint_armature = []
        joint_friction = []
        # joint_dof_mode = []
        joint_effort_limit = []
        joint_velocity_limit = []

        for dof_name in self.robot_view.joint_names:
            common_dof_names = self._newton_dof_names[dof_name]
            if not isinstance(common_dof_names, list):
                common_dof_names = [common_dof_names]
            for common_dof_name in common_dof_names:
                ke = (
                    self.robot_config.control.control_info[common_dof_name].stiffness
                    if self.control_type == ControlType.BUILT_IN_PD
                    else 0.0
                )
                joint_stiffness.append(ke)

                kd = (
                    self.robot_config.control.control_info[common_dof_name].damping
                    if self.control_type == ControlType.BUILT_IN_PD
                    else 0.0
                )
                joint_damping.append(kd)

                # Set armature (adds inertia to joint for stability)
                armature = self.robot_config.control.control_info[
                    common_dof_name
                ].armature
                joint_armature.append(armature)

                # Set friction loss
                friction = self.robot_config.control.control_info[
                    common_dof_name
                ].friction
                if joint_friction is not None and friction is not None:
                    joint_friction.append(friction)
                else:
                    joint_friction = None

                effort_limit = self.robot_config.control.control_info[
                    common_dof_name
                ].effort_limit
                if joint_effort_limit is not None and effort_limit is not None:
                    joint_effort_limit.append(effort_limit)
                else:
                    joint_effort_limit = None

                velocity_limit = self.robot_config.control.control_info[
                    common_dof_name
                ].velocity_limit
                if joint_velocity_limit is not None and velocity_limit is not None:
                    joint_velocity_limit.append(velocity_limit)
                else:
                    joint_velocity_limit = None

                # if self.robot_config.control.control_type == ControlType.BUILT_IN_PD:
                #     joint_dof_mode.append(newton.JointMode.TARGET_POSITION)
                # else:
                #     joint_dof_mode.append(newton.JointMode.NONE)

        joint_stiffness = wp.from_torch(
            torch.tensor(joint_stiffness, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )
        joint_damping = wp.from_torch(
            torch.tensor(joint_damping, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )
        joint_armature = wp.from_torch(
            torch.tensor(joint_armature, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
        )
        # joint_dof_mode = wp.from_torch(
        #     torch.tensor(joint_dof_mode, device=self.device, dtype=torch.int32)
        #     .unsqueeze(0)
        #     .expand(self.num_envs, -1)
        # )
        if joint_friction is not None:
            joint_friction = wp.from_torch(
                torch.tensor(joint_friction, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )
        if joint_effort_limit is not None:
            joint_effort_limit = wp.from_torch(
                torch.tensor(
                    joint_effort_limit, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )
        if joint_velocity_limit is not None:
            joint_velocity_limit = wp.from_torch(
                torch.tensor(
                    joint_velocity_limit, device=self.device, dtype=torch.float32
                )
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )

        self.robot_view.set_attribute("joint_target_ke", self.model, joint_stiffness)
        self.robot_view.set_attribute("joint_target_kd", self.model, joint_damping)
        self.robot_view.set_attribute("joint_armature", self.model, joint_armature)
        # self.robot_view.set_attribute("joint_dof_mode", self.model, joint_dof_mode)
        if joint_friction is not None:
            self.robot_view.set_attribute("joint_friction", self.model, joint_friction)
        if joint_effort_limit is not None:
            self.robot_view.set_attribute(
                "joint_effort_limit", self.model, joint_effort_limit
            )
        if joint_velocity_limit is not None:
            self.robot_view.set_attribute(
                "joint_velocity_limit", self.model, joint_velocity_limit
            )

        self.default_body_transforms = (
            wp.to_torch(self.robot_view.get_link_transforms(self.model))
            .clone()
            .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        )
        self.default_body_velocities = (
            wp.to_torch(self.robot_view.get_link_velocities(self.model))
            .clone()
            .view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        )
        self.default_root_transforms = wp.to_torch(
            self.robot_view.get_root_transforms(self.model)
        ).clone()
        self.default_root_velocities = wp.to_torch(
            self.robot_view.get_root_velocities(self.model)
        ).clone()
        self.default_dof_positions = wp.to_torch(
            self.robot_view.get_dof_positions(self.model)
        ).clone()
        self.default_dof_velocities = wp.to_torch(
            self.robot_view.get_dof_velocities(self.model)
        ).clone()

    def _setup_sim(self) -> None:
        """Creates simulation."""
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            # integrator="implicitfast",
            # njmax=250,
            # nconmax=250,
            # iterations=10,
            # ls_iterations=20,
            # ls_parallel=True,
            # cone="pyramidal",
            # impratio=1
            integrator="implicitfast",
            njmax=450,
            nconmax=300,
            iterations=100,
            ls_iterations=50,
            ls_parallel=True,
            cone="pyramidal",
            # contact_stiffness_time_const=0.02,
            impratio=10.0,
            default_actuator_gear=None,
            disable_contacts=False,
        )

        geom_margin = wp.to_torch(self.solver.mjw_model.geom_margin)
        geom_margin[:] = 0.01  # margin for earlier detection
        self.solver.mjw_model.geom_margin = wp.from_torch(geom_margin, dtype=wp.float32)

        geom_gap = wp.to_torch(self.solver.mjw_model.geom_gap)
        geom_gap[:] = 0.01  # Forces start when touching
        self.solver.mjw_model.geom_gap = wp.from_torch(geom_gap, dtype=wp.float32)

        # Create viewer if not headless
        self.viewer = None
        if not self.headless:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)
            self.viewer.vsync = True

        # Initialize simulation state
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_0
        )

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """Returns the ordering of bodies and DOFs in the simulation."""
        joint_names = self.robot_view.joint_names
        dof_names = []

        for joint_name in joint_names:
            if type(self._newton_dof_names[joint_name]) is list:
                dof_names.extend(self._newton_dof_names[joint_name])
            else:
                dof_names.append(self._newton_dof_names[joint_name])
        return SimBodyOrdering(
            body_names=self.robot_view.body_names,
            dof_names=dof_names,
        )

    def _setup_markers(
        self, visualization_markers: Dict[str, VisualizationMarkerConfig]
    ) -> None:
        """Setup visualization markers."""
        return

    def _add_terrain(self, builder: newton.ModelBuilder) -> None:
        """Adds terrain."""
        if self.terrain is None:
            return

        print("Adding terrain")
        if (
            sum(self.terrain.config.terrain_proportions[:-1]) == 0
            and self.terrain.config.terrain_proportions[-1] == 1.0
        ):
            # When using a default terrain, we spawn the built-in plane.
            # This is faster and more memory efficient than spawning a trimesh terrain.
            # cfg=newton.ModelBuilder.ShapeConfig(mu=xxx, restitution=xxx)
            builder.add_ground_plane()
        else:
            points, indices = convert_to_indexed_mesh(
                self.terrain.vertices, self.terrain.triangles
            )
            ground_mesh = newton.Mesh(vertices=points, indices=indices)
            builder.add_shape_mesh(body=-1, mesh=ground_mesh, key="ground_plane")
        print("Terrain added")

    def _setup_contact_sensors(self) -> None:
        """Setup contact sensors for each contact body."""
        if (
            self.robot_config.contact_bodies is None
            or len(self.robot_config.contact_bodies) == 0
        ):
            return

        print(
            f"[INFO] Setting up contact sensors for bodies: {self.robot_config.contact_bodies}"
        )

        # Create a contact sensor for each specified contact body
        for body_name in self.robot_config.contact_bodies:
            # Create sensor that detects contacts between this body and anything
            # The sensor will aggregate contacts across all environments
            sensor = ContactSensor(
                self.model, sensing_obj_bodies=body_name, verbose=False
            )
            self._contact_sensors[body_name] = sensor

            # Initialize storage for contact forces (will be populated during simulation)
            self._contact_forces[body_name] = torch.zeros(
                self.num_envs, 3, device=self.device, dtype=torch.float32
            )

        print(
            f"[INFO] Contact sensors setup complete for {len(self._contact_sensors)} bodies"
        )

    # ===== Group 3: Simulation Steps & State Management =====
    def _simulate(self) -> None:
        """Internal simulation step for physics engine."""
        state_0_dict = self.state_0.__dict__
        state_1_dict = self.state_1.__dict__
        state_temp_dict = self.state_temp.__dict__

        self.state_0.clear_forces()

        # Apply forces from viewer (picking, wind, etc.)
        if self.viewer:
            self.viewer.apply_forces(self.state_0)

        # Step physics
        self.solver.step(
            self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
        )

        # Swap state buffers
        for key, value in state_0_dict.items():
            if isinstance(value, wp.array):
                if key not in state_temp_dict:
                    state_temp_dict[key] = wp.empty_like(value)
                state_temp_dict[key].assign(value)
                state_0_dict[key].assign(state_1_dict[key])
                state_1_dict[key].assign(state_temp_dict[key])

    def _update_contact_sensors(self) -> None:
        """Update contact sensors after physics step. Must be called outside CUDA graph."""
        if len(self._contact_sensors) > 0:
            populate_contacts(self.contacts, self.solver)
            for body_name, sensor in self._contact_sensors.items():
                sensor.eval(self.contacts)
                # Store the net contact force for this body (across all environments)
                # sensor.net_force has shape [num_worlds, num_bodies, 3] where num_bodies=1
                if hasattr(sensor, "net_force") and sensor.net_force is not None:
                    net_force = wp.to_torch(sensor.net_force).clone()
                    # Squeeze the body dimension if present (shape [N, 1, 3] -> [N, 3])
                    if net_force.dim() == 3 and net_force.shape[1] == 1:
                        net_force = net_force.squeeze(1)
                    self._contact_forces[body_name] = net_force

    def _physics_step(self) -> None:
        """Performs a physics simulation step."""
        # For BUILT_IN_PD, set targets once before loop (efficiency)
        # For PROPORTIONAL/TORQUE, apply inside loop (needs fresh DOF state each substep)
        if self.control_type == ControlType.BUILT_IN_PD:
            self._apply_control()
        for i in range(self.decimation):
            if self.control_type != ControlType.BUILT_IN_PD:
                self._apply_control()
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._simulate()

        # Update contact sensors after all substeps (outside CUDA graph)
        self._update_contact_sensors()

        self.sim_time += self.frame_dt

    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: ObjectState = None,
        env_ids: torch.Tensor = None,
    ) -> None:
        """Sets the state of specified environments using vectorized operations."""
        # assert new_object_states is None, "Newton does not yet support setting object states."

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_mask[env_ids] = True

        # Newton expects the state setter to be provided with the states for all envs.
        # The mask is used to determine which envs to apply the update to.
        robot_state = self._get_simulator_bodies_state()
        robot_dof_state = self._get_simulator_dof_state()
        robot_state.merge_fields_from(robot_dof_state)

        robot_state.root_pos[env_ids] = new_states.root_pos
        robot_state.root_rot[env_ids] = new_states.root_rot
        robot_state.root_vel[env_ids] = new_states.root_vel
        robot_state.root_ang_vel[env_ids] = new_states.root_ang_vel
        robot_state.dof_pos[env_ids] = new_states.dof_pos
        robot_state.dof_vel[env_ids] = new_states.dof_vel

        root_state = torch.cat([robot_state.root_pos, robot_state.root_rot], dim=1)
        root_vel_state = torch.cat(
            [robot_state.root_vel, robot_state.root_ang_vel], dim=1
        )

        # Set state_0 using ArticulationView
        self.robot_view.set_root_transforms(self.state_0, root_state, mask=env_mask)
        self.robot_view.set_root_velocities(self.state_0, root_vel_state, mask=env_mask)
        self.robot_view.set_dof_positions(
            self.state_0, robot_state.dof_pos, mask=env_mask
        )
        self.robot_view.set_dof_velocities(
            self.state_0, robot_state.dof_vel, mask=env_mask
        )

        # also update state_1 to match state_0 (following Newton examples like example_robot_policy.py)
        # This ensures both states are synchronized after reset, which is essential for the state-swapping
        # mechanism used in the simulation loop
        self.robot_view.set_root_transforms(self.state_1, root_state, mask=env_mask)
        self.robot_view.set_root_velocities(self.state_1, root_vel_state, mask=env_mask)
        self.robot_view.set_dof_positions(
            self.state_1, robot_state.dof_pos, mask=env_mask
        )
        self.robot_view.set_dof_velocities(
            self.state_1, robot_state.dof_vel, mask=env_mask
        )

        # Clear forces after reset (good practice to avoid residual forces affecting next step)
        self.state_0.clear_forces()
        self.state_1.clear_forces()

        # Recompute forward kinematics to refresh derived body states
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )
        newton.eval_fk(
            self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1
        )

    # ===== Group 4: State Getters =====
    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns contact forces for robot bodies."""
        # Initialize with zeros for all bodies
        rigid_body_contact_forces = torch.zeros(
            self.num_envs, len(self._body_names), 3, device=self.device
        )

        # Populate contact forces from sensors
        if len(self._contact_sensors) > 0:
            for body_name, contact_force in self._contact_forces.items():
                # Find the index of this body in the body_names list
                if body_name in self._body_names:
                    body_idx = self._body_names.index(body_name)
                    rigid_body_contact_forces[:, body_idx, :] = contact_force

        if env_ids is not None:
            rigid_body_contact_forces = rigid_body_contact_forces[env_ids]

        return RobotState(
            rigid_body_contact_forces=rigid_body_contact_forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_contact_binary(
        self, env_ids: Optional[torch.Tensor] = None, force_threshold: float = 1.0
    ) -> torch.Tensor:
        """
        Returns binary contact labels for robot bodies.

        A body is considered in contact if its contact force magnitude exceeds the threshold.

        Args:
            env_ids: Optional tensor of environment IDs to query
            force_threshold: Minimum contact force magnitude to consider as contact (default: 1.0 N)

        Returns:
            Binary tensor of shape [num_envs, num_bodies] where 1 indicates contact
        """
        # Get contact forces
        contact_state = self._get_simulator_bodies_contact_buf(env_ids=env_ids)
        contact_forces = (
            contact_state.rigid_body_contact_forces
        )  # [num_envs, num_bodies, 3]

        # Compute force magnitudes
        force_magnitudes = torch.norm(contact_forces, dim=-1)  # [num_envs, num_bodies]

        # Apply threshold to get binary labels
        contact_binary = (force_magnitudes > force_threshold).float()

        return contact_binary

    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns the state of robot bodies."""
        # Extract positions and quaternions: [x, y, z, qx, qy, qz, qw] per body
        body_transforms = wp.to_torch(
            self.robot_view.get_link_transforms(self.state_0)
        ).view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        body_pos = body_transforms[:, :, :3]
        body_rot = body_transforms[:, :, 3:]

        body_vel_transforms = wp.to_torch(
            self.robot_view.get_link_velocities(self.state_0)
        ).view(self.num_envs, self.robot_config.kinematic_info.num_bodies, -1)
        body_vel = body_vel_transforms[:, :, :3]
        body_ang_vel = body_vel_transforms[:, :, 3:]

        if env_ids is not None:
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]

        return RobotState(
            rigid_body_pos=body_pos,
            rigid_body_rot=body_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """Returns the root state of the robot. TODO: Implement."""
        root_transforms = wp.to_torch(self.robot_view.get_root_transforms(self.state_0))
        root_velocities = wp.to_torch(self.robot_view.get_root_velocities(self.state_0))

        if env_ids is not None:
            root_transforms = root_transforms[env_ids]
            root_velocities = root_velocities[env_ids]

        return RootOnlyState(
            root_pos=root_transforms[:, :3],
            root_rot=root_transforms[:, 3:],
            root_vel=root_velocities[:, :3],
            root_ang_vel=root_velocities[:, 3:],
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Returns the root state of simulation objects. Not implemented."""
        newton_root_pos = []
        newton_root_rot = []
        newton_root_vel = []
        newton_root_ang_vel = []

        if self.robot_view is not None:
            num_dynamic_objects = sum(
                not obj.options.fix_base_link
                for obj in self.scene_lib.scenes[0].objects
            )
            view_transforms = wp.to_torch(
                self.object_view.get_root_transforms(self.state_0)
            ).view(self.num_envs, num_dynamic_objects, -1)
            view_velocities = wp.to_torch(
                self.object_view.get_root_velocities(self.state_0)
            ).view(self.num_envs, num_dynamic_objects, -1)
        else:
            view_transforms = view_velocities = None

        articulated_object_index = 0
        static_object_index = 0
        for obj_idx in range(self.scene_lib.num_objects_per_scene):
            if self.scene_lib.scenes[0].objects[obj_idx].options.fix_base_link:
                newton_root_pos.append(
                    self._static_object_poses[:, static_object_index, :3]
                )
                newton_root_rot.append(
                    self._static_object_poses[:, static_object_index, 3:]
                )
                newton_root_vel.append(torch.zeros_like(newton_root_pos[-1]))
                newton_root_ang_vel.append(torch.zeros_like(newton_root_pos[-1]))
                static_object_index += 1
            else:
                newton_root_pos.append(view_transforms[:, articulated_object_index, :3])
                newton_root_rot.append(view_transforms[:, articulated_object_index, 3:])
                newton_root_vel.append(view_velocities[:, articulated_object_index, :3])
                newton_root_ang_vel.append(
                    view_velocities[:, articulated_object_index, 3:]
                )
                articulated_object_index += 1

        newton_root_pos = torch.stack(newton_root_pos, dim=1)
        newton_root_rot = torch.stack(newton_root_rot, dim=1)
        newton_root_vel = torch.stack(newton_root_vel, dim=1)
        newton_root_ang_vel = torch.stack(newton_root_ang_vel, dim=1)
        if env_ids is not None:
            newton_root_pos = newton_root_pos[env_ids]
            newton_root_rot = newton_root_rot[env_ids]
            newton_root_vel = newton_root_vel[env_ids]
            newton_root_ang_vel = newton_root_ang_vel[env_ids]
        return ObjectState(
            root_pos=newton_root_pos,
            root_rot=newton_root_rot,
            root_vel=newton_root_vel,
            root_ang_vel=newton_root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """Returns contact forces for simulation objects. Not implemented."""
        # TODO: add contact sensors
        return ObjectState(state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_dof_forces(self, env_ids=None):
        """Returns the DOF forces. TODO: Implement."""
        dof_forces = wp.to_torch(self.robot_view.get_dof_forces(self.control))
        if env_ids is not None:
            dof_forces = dof_forces[env_ids]
        return RobotState(
            dof_forces=dof_forces, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """Returns the state of robot DOFs."""
        # Extract positions (skip floating base: 6 DOF)
        dof_pos = wp.to_torch(self.robot_view.get_dof_positions(self.state_0)).view(
            self.num_envs, -1
        )
        dof_vel = wp.to_torch(self.robot_view.get_dof_velocities(self.state_0)).view(
            self.num_envs, -1
        )

        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]

        return RobotState(
            dof_pos=dof_pos, dof_vel=dof_vel, state_conversion=StateConversion.SIMULATOR
        )

    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from Newton's internal API for verification purposes only.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in Newton's DOF ordering.
        """
        # Extract limits from Newton's internal representation
        dof_limits_lower = wp.to_torch(
            self.robot_view.get_attribute("joint_limit_lower", self.model)
        )[0]
        dof_limits_upper = wp.to_torch(
            self.robot_view.get_attribute("joint_limit_upper", self.model)
        )[0]
        return dof_limits_lower, dof_limits_upper

    # ===== Group 5: Control & Computation Methods =====
    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """Applies PD position targets using Newton's internal PD controller."""
        a_wp = wp.from_torch(pd_targets, dtype=wp.float32, requires_grad=False)
        self.robot_view.set_attribute("joint_target_pos", self.control, a_wp)

    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """Applies torques to the robot DOFs."""
        a_wp = wp.from_torch(torques, dtype=wp.float32, requires_grad=False)
        self.robot_view.set_dof_forces(self.control, a_wp)

    # ===== Group 6: Rendering & Visualization =====
    def _init_camera(self) -> None:
        """Initializes camera."""
        char_root_pos = (
            self._get_simulator_root_state([self._camera_target["env"]])
            .root_pos.flatten()
            .cpu()
            .numpy()
        )

        cam_pos = char_root_pos + np.array([0, -5.0, 1])

        camera_target = char_root_pos + np.array([0, 0, 0.2])
        vector_to_target = camera_target - cam_pos
        normalized_vector_to_target = vector_to_target / np.linalg.norm(
            vector_to_target
        )
        pitch = np.rad2deg(np.arcsin(normalized_vector_to_target[2]))
        yaw = np.rad2deg(
            np.arctan2(normalized_vector_to_target[1], normalized_vector_to_target[0])
        )

        self.viewer.set_camera(wp.vec3(cam_pos.tolist()), pitch, yaw)
        self._cam_prev_char_pos = char_root_pos

    def _init_keyboard(self) -> None:
        """Initializes keyboard controls. TODO: Implement."""
        pass

    def _update_camera(self) -> None:
        """Updates camera position."""
        if self._camera_target["element"] == 0:
            char_root_pos = (
                self._get_simulator_root_state([self._camera_target["env"]])
                .root_pos.flatten()
                .cpu()
                .numpy()
            )
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            char_root_pos = (
                self._get_simulator_object_root_state(self._camera_target["env"])
                .root_pos[in_scene_object_id]
                .flatten()
                .cpu()
                .numpy()
            )
            height_offset = 0

        cam_pos = np.array(self.viewer.camera.pos)
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = char_root_pos + np.array([0, 0, height_offset])
        new_cam_pos = char_root_pos + cam_delta

        vector_to_target = new_cam_target - new_cam_pos
        normalized_vector_to_target = vector_to_target / np.linalg.norm(
            vector_to_target
        )
        pitch = np.rad2deg(np.arcsin(normalized_vector_to_target[2]))
        yaw = np.rad2deg(
            np.arctan2(normalized_vector_to_target[1], normalized_vector_to_target[0])
        )

        self.viewer.set_camera(wp.vec3(new_cam_pos.tolist()), pitch, yaw)
        self._cam_prev_char_pos = char_root_pos

    def close(self) -> None:
        """Closes the simulator and cleans up resources."""
        pass

    def _write_viewport_to_file(self, file_name: str) -> None:
        """Writes viewport to file. TODO: Implement."""
        pass

    def render(self) -> None:
        """Renders the current simulation state."""
        if not self.headless:
            if not self._camera_initialized:
                self._init_camera()
                self._camera_initialized = True
            else:
                self._update_camera()

            any_key_pressed = False
            if self.viewer.is_key_down("q"):
                sys.exit()
            elif self.viewer.is_key_down("j"):
                if not self._any_key_pressed:
                    self._push_robot()
                any_key_pressed = True
            elif self.viewer.is_key_down("l"):
                if not self._any_key_pressed:
                    self._toggle_video_record()
                any_key_pressed = True
            elif self.viewer.is_key_down(";"):
                if not self._any_key_pressed:
                    self._cancel_video_record()
                any_key_pressed = True
            elif self.viewer.is_key_down("o"):
                if not self._any_key_pressed:
                    self._toggle_camera_target()
                any_key_pressed = True
            elif self.viewer.is_key_down("m"):
                if not self._any_key_pressed:
                    self._toggle_markers()
                any_key_pressed = True
            elif self.viewer.is_key_down("r"):
                if not self._any_key_pressed:
                    self._requested_reset()
                any_key_pressed = True
            for key, handler in self._custom_key_handlers.items():
                if self.viewer.is_key_down(key):
                    if not self._any_key_pressed:
                        handler()
                    any_key_pressed = True

            self._any_key_pressed = any_key_pressed

            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()

        super().render()

    def _write_viewport_to_file(self, file_name: str) -> None:
        import matplotlib.pyplot as plt

        viewport = self.viewer.get_frame().numpy()  # [H, W, 3] as uint8
        plt.imsave(file_name, viewport)

    def _push_robot(self):
        """Pushes the robot."""
        robot_state = self._get_simulator_root_state()
        root_vel_state = torch.cat(
            [
                robot_state.root_vel + torch.ones_like(robot_state.root_vel),
                robot_state.root_ang_vel,
            ],
            dim=1,
        )
        self.robot_view.set_root_velocities(self.state_0, root_vel_state)
        newton.eval_fk(
            self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0
        )

    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """Updates visualization markers. TODO: Implement."""
        pass
