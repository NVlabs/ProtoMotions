import os
import torch
from typing import Dict, Optional
from easydict import EasyDict

from isaac_utils import rotations
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    ControlType,
    VisualizationMarker,
    SimBodyOrdering,
    SimulatorConfig
)
from protomotions.simulator.base_simulator.robot_state import RobotState
from protomotions.utils.scene_lib import SceneLib
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.envs.base_env.env_utils.terrains.flat_terrain import FlatTerrain

import genesis as gs
import numpy as np
import matplotlib.pyplot as plt


class GenesisSimulator(Simulator):
    """
    GenesisSimulator wraps the Genesis physics engine for use in our simulation framework.

    This simulator initializes the Genesis simulation instance, creates environments, and implements all the
    required simulation methods by interfacing with the genesis API.
    """
    # ===== Group 1: Initialization & Configuration =====
    def __init__(
        self,
        config: SimulatorConfig,
        terrain: Terrain,
        device: torch.device,
        scene_lib: Optional[SceneLib] = None,
        visualization_markers: Optional[Dict[str, VisualizationMarker]] = None,
    ) -> None:
        super().__init__(config=config, scene_lib=scene_lib, terrain=terrain, visualization_markers=visualization_markers, device=device)
        
        assert scene_lib is None, "Genesis does not support spawning objects in the scene"
        
        # Initialize the Genesis engine
        gs.init(backend=gs.cpu) if device.type == "cpu" else gs.init(backend=gs.gpu)
        self._create_sim(visualization_markers)

    def _create_sim(self, visualization_markers: Dict) -> None:
        """Creates the Genesis simulation environment with specified configuration."""
        # Create a Genesis Scene with the configuration
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1. / self.config.sim.fps,
                substeps=self.config.sim.substeps
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=self.robot_config.asset.self_collisions,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.config.sim.fps * self.config.sim.decimation,
                camera_pos=(2.5, 0.0, 4.0),
                camera_lookat=(0.0, 0.0, 2.0),
                camera_fov=40,
            ),
            show_viewer=not self.headless,
            vis_options=gs.options.VisOptions(
                show_world_frame = True, # visualize the coordinate frame of `world` at its origin
                world_frame_size = 1.0, # length of the world frame in meter
                show_link_frame  = False, # do not visualize coordinate frames of entity links
                show_cameras     = False, # do not visualize mesh and frustum of the cameras added
                plane_reflection = True, # turn on plane reflection
                ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
            ),
            renderer=gs.renderers.Rasterizer(), # using rasterizer for camera rendering
        )
        
        if not self.headless:
            self._perspective_view = self._scene.add_camera(
                # Use the same parameters as the viewer
                res=self._scene.viewer.res,  
                pos=(2.5, 0.0, 4.0),
                lookat=(0.0, 0.0, 2.0),
                fov=40,
                GUI=False
            )

        self._add_terrain()
        self._setup_markers(visualization_markers)

        self._create_envs()
        
        if not self.headless:
            self._init_camera()
            self._init_keyboard()

    # ===== Group 2: Environment Setup & Configuration =====
    def _create_envs(self) -> None:
        """Creates the simulation environments and loads robot assets."""
        asset_root = self.robot_config.asset.asset_root
        asset_file = self.robot_config.asset.asset_file_name
        
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_file_ext = os.path.splitext(asset_file)[1]
        
        # Load robot asset based on file type (MJCF or URDF)
        if asset_file_ext == ".xml":
            self._robot = self._scene.add_entity(
                gs.morphs.MJCF(
                    file=asset_path,
                ),
                visualize_contact=False,
            )
        else:
            self._robot = self._scene.add_entity(
                gs.morphs.URDF(
                    file=asset_path,
                    merge_fixed_links=True,
                    links_to_keep=self.robot_config.body_names,
                ),
                visualize_contact=False,
            )
        
        # Setup DOF indices and limits
        self._genesis_dof_indices = []
        dof_limits_lower = []
        dof_limits_upper = []
        for joint in self._robot.joints:
            if joint.name in self.robot_config.dof_names:
                if type(joint.dof_idx_local) is list:
                    self._genesis_dof_indices.extend(joint.dof_idx_local)
                else:
                    self._genesis_dof_indices.append(joint.dof_idx_local)

                dof_limits_lower.extend(joint.dofs_limit[:, 0])
                dof_limits_upper.extend(joint.dofs_limit[:, 1])

        self._genesis_dof_indices = torch.tensor(self._genesis_dof_indices, device=self.device)
        self._dof_limits_lower_sim = torch.tensor(dof_limits_lower, device=self.device, dtype=gs.tc_float)
        self._dof_limits_upper_sim = torch.tensor(dof_limits_upper, device=self.device, dtype=gs.tc_float)
        
        self._scene.build(n_envs=self.num_envs)
        
    def on_environment_ready(self) -> None:
        """
        Configure internal tensors after the simulation environment is initialized.
        This includes conversion tensors for bodies, DOFs, and contact sensors.
        """
        self._genesis_default_state = RobotState(
            root_pos=self._robot.get_pos(),
            root_rot=self._robot.get_quat(),
            root_vel=self._robot.get_vel() * 0,
            root_ang_vel=self._robot.get_ang() * 0,
            dof_pos=self._robot.get_dofs_position(self._genesis_dof_indices),
            dof_vel=self._robot.get_dofs_velocity(self._genesis_dof_indices),
            rigid_body_pos=self._robot.get_links_pos(),
            rigid_body_rot=self._robot.get_links_quat(),
            rigid_body_vel=self._robot.get_links_vel(),
            rigid_body_ang_vel=self._robot.get_links_ang(),
        )
        super().on_environment_ready()

    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """Returns the ordering of bodies and DOFs in the simulation."""
        dof_names = []
        for joint in self._robot.joints:
            if joint.name in self.robot_config.dof_names:
                common_dof_idx = self.robot_config.dof_names.index(joint.name)
                if type(joint.dof_idx_local) is list:
                    for dof_idx_local in joint.dof_idx_local:
                        dof_offset = dof_idx_local - joint.dof_idx_local[0]
                        dof_names.append(self.robot_config.dof_names[common_dof_idx + dof_offset])
                else:
                    dof_names.append(joint.name)
        
        return SimBodyOrdering(
            body_names=[link.name for link in self._robot.links],
            dof_names=dof_names,
            contact_sensor_body_names=[link.name for link in self._robot.links],
        )
    
    def _load_object_assets(self) -> None:
        """Loads object assets for the simulation environment."""
        pass
        
    def _build_object_playground(self, env_id: int, env_ptr) -> None:
        """Builds the object playground for a specific environment."""
        pass
    
    def _setup_markers(self, visualization_markers: Dict[str, VisualizationMarker]) -> None:
        """Build and configure visualization markers.
        Genesis visualization works much better when parallelized.
        However, each marker call supports a single scale value.
        Therefore, we group marker indices by scale value and store the indices in a dictionary.
        This enables parallel rendering of markers with different scales.

        Args:
            visualization_markers (Dict[str, VisualizationMarker]): Dictionary mapping marker names to their configurations
        """
        self._visualization_markers = {}
        if visualization_markers is None:
            return

        for marker_name, markers_cfg in visualization_markers.items():
            if markers_cfg.type not in ["sphere", "arrow"]:
                raise ValueError(f"Marker type {markers_cfg.type} not supported")
            scale_dict = {}
            for i, marker in enumerate(markers_cfg.markers):
                # Append the marker index to the corresponding scale_value key.
                scale_dict.setdefault(marker.size, []).append(i)
            
            for marker_size, indices in scale_dict.items():
                scale_dict[marker_size] = torch.tensor(indices, device=self.device, dtype=torch.long)
            
            self._visualization_markers[marker_name] = EasyDict({
                "marker_type": markers_cfg.type,
                "marker_color": markers_cfg.color,
                "scale": scale_dict,
            })
    
    def _add_terrain(self) -> None:
        """Adds terrain to the simulation environment."""
        if isinstance(self.terrain, FlatTerrain):
            # When using a flat terrain, we spawn the built-in plane.
            # This is faster and more memory efficient than spawning a trimesh terrain.
            # The Genesis plane spans the entire environment.
            self._genesis_terrain = self._scene.add_entity(
                gs.morphs.Plane()
            )
        else:
            self._genesis_terrain = self._scene.add_entity(
                morph=gs.morphs.Terrain(
                    horizontal_scale=self.terrain.horizontal_scale,
                    vertical_scale=self.terrain.vertical_scale,
                    height_field=self.terrain.height_field_raw,
                ),
            )

    # ===== Group 3: Simulation Steps & State Management =====
    def _physics_step(self) -> None:
        """Performs a physics simulation step."""
        for i in range(self.decimation):
            if self.control_type == ControlType.BUILT_IN_PD:
                self._apply_pd_control()
            else:
                self._apply_motor_forces()
            self._scene.step()

    def _set_simulator_env_state(self, new_states: RobotState, env_ids: Optional[torch.Tensor] = None) -> None:
        """Sets the state of specified environments."""
        self._robot.set_pos(
            new_states.root_pos, zero_velocity=False, envs_idx=env_ids
        )
        self._robot.set_quat(
            new_states.root_rot, zero_velocity=False, envs_idx=env_ids
        )
        self._robot.set_dofs_velocity(
            new_states.root_vel, dofs_idx_local=[0, 1, 2],  envs_idx=env_ids
        )
        self._robot.set_dofs_velocity(
            new_states.root_ang_vel, dofs_idx_local=[3, 4, 5],  envs_idx=env_ids
        )
        self._robot.set_dofs_position(
            position=new_states.dof_pos,
            dofs_idx_local=self._genesis_dof_indices,
            envs_idx=env_ids,
        )
        self._robot.set_dofs_velocity(
            velocity=new_states.dof_vel,
            dofs_idx_local=self._genesis_dof_indices,
            envs_idx=env_ids,
        )

    # ===== Group 4: State Getters =====
    def _get_simulator_bodies_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns contact forces for robot bodies."""
        contact_forces = self._robot.get_links_net_contact_force()
        if env_ids is not None:
            contact_forces = contact_forces[env_ids]
        return contact_forces
    
    def _get_simulator_bodies_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """Returns the state of robot bodies."""
        body_pos = self._robot.get_links_pos()
        body_rot = self._robot.get_links_quat()
        body_vel = self._robot.get_links_vel()
        body_ang_vel = self._robot.get_links_ang()
        if env_ids is not None:
            body_pos = body_pos[env_ids]
            body_rot = body_rot[env_ids]
            body_vel = body_vel[env_ids]
            body_ang_vel = body_ang_vel[env_ids]
        return RobotState(
            rigid_body_pos=body_pos,
            rigid_body_rot=body_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel
        )

    def _get_simulator_default_state(self) -> RobotState:
        """Returns the default state of the simulator."""
        return self._genesis_default_state

    def _get_simulator_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """Returns the root state of the robot."""
        root_pos = self._robot.get_pos()
        root_rot = self._robot.get_quat()
        root_vel = self._robot.get_vel()
        root_ang_vel = self._robot.get_ang()
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
        return RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel
        )

    def _get_simulator_object_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """Returns the root state of simulation objects."""
        pass
    
    def _get_simulator_object_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns contact forces for simulation objects."""
        pass

    def _get_simulator_dof_forces(self, env_ids=None):
        """Returns the DOF forces."""
        dof_forces = self._robot.get_dofs_force(self._genesis_dof_indices)
        if env_ids is not None:
            dof_forces = dof_forces[env_ids]
        return dof_forces

    def _get_simulator_dof_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """Returns the state of robot DOFs."""
        dof_pos = self._robot.get_dofs_position(self._genesis_dof_indices)
        dof_vel = self._robot.get_dofs_velocity(self._genesis_dof_indices)
        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
        return RobotState(
            dof_pos=dof_pos,
            dof_vel=dof_vel
        )

    # ===== Group 5: Control & Computation Methods =====
    def _apply_motor_forces(self) -> None:
        """Applies computed motor forces to the robot."""
        common_torques = self._compute_torques(self._common_actions)
        genesis_torques = common_torques[:, self.data_conversion.dof_convert_to_sim]
        self._robot.control_dofs_force(genesis_torques, self._genesis_dof_indices)

    def _apply_pd_control(self) -> None:
        """Applies PD control to the robot."""
        common_pd_tar = self._action_to_pd_targets(self._common_actions)
        genesis_pd_tar = common_pd_tar[:, self.data_conversion.dof_convert_to_sim]
        self._robot.control_dofs_position(genesis_pd_tar, self._genesis_dof_indices)

    # ===== Group 6: Rendering & Visualization =====
    def _init_camera(self) -> None:
        """Initializes the camera position and orientation."""
        self._cam_prev_char_pos = self._get_simulator_root_state(self._camera_target["env"]).root_pos.cpu().numpy()

        cam_pos = np.array([
            self._cam_prev_char_pos[0],
            self._cam_prev_char_pos[1] - 3.0,
            self._cam_prev_char_pos[2] + 0.4,
        ])
        cam_target = np.array([
            self._cam_prev_char_pos[0],
            self._cam_prev_char_pos[1],
            self._cam_prev_char_pos[2] + 0.2,
        ])
        self._scene.viewer.set_camera_pose(pos=cam_pos, lookat=cam_target)
        
    def _init_keyboard(self) -> None:
        pass
        # TODO: implement
    
    def _update_camera(self) -> None:
        """Updates the camera position based on the target."""
        if self._camera_target["element"] == 0:
            current_char_pos = self._get_simulator_root_state(self._camera_target["env"]).root_pos.cpu().numpy()
            height_offset = 0.2
        else:
            in_scene_object_id = self._camera_target["element"] - 1
            current_char_pos = self._get_simulator_object_root_state(self._camera_target["env"]).root_pos[in_scene_object_id].cpu().numpy()
            height_offset = 0

        current_cam_transform = self._scene.viewer.camera_pos
        
        cam_offset = current_cam_transform - self._cam_prev_char_pos

        new_cam_target = np.array([
            current_char_pos[0], current_char_pos[1], current_char_pos[2] + height_offset
        ])

        new_cam_pos = np.array([
            current_char_pos[0] + cam_offset[0],
            current_char_pos[1] + cam_offset[1],
            current_char_pos[2] + cam_offset[2],
        ])

        self._scene.viewer.set_camera_pose(pos=new_cam_pos, lookat=new_cam_target)
        self._perspective_view.set_pose(pos=new_cam_pos, lookat=new_cam_target)

        self._cam_prev_char_pos[:] = current_char_pos
    
    def close(self) -> None:
        """Closes the simulator and cleans up resources."""
        pass
    
    def _write_viewport_to_file(self, file_name: str) -> None:
        """Writes the current viewport to a file.

        Args:
            file_name (str): Path where the image should be saved
        """
        rgb = self._perspective_view.render()
        plt.imsave(file_name, rgb)

    def render(self) -> None:
        """Renders the current simulation state."""
        if not self.headless:
            self._update_camera()
        super().render()

    def _update_simulator_markers(self, markers_state: Optional[Dict[str, MarkerState]] = None) -> None:
        """Updates the state of visualization markers."""
        self._scene.clear_debug_objects()
        if markers_state is None:
            return
        
        for marker_name, markers_state_item in markers_state.items():
            marker_dict = self._visualization_markers[marker_name]
            
            marker_pos_all = markers_state_item.translation.view(self.num_envs, -1, 3)
            marker_quat = markers_state_item.orientation.view(self.num_envs, -1, 4)
            if marker_dict.marker_type == "sphere":
                # Iterate over each scale group and render all markers with that scale at once.
                for marker_size, indices in marker_dict.scale.items():
                    if marker_size == "tiny":
                        scale_val = 0.007
                    elif marker_size == "small":
                        scale_val = 0.01
                    else:
                        scale_val = 0.05
                    
                    group_marker_pos = marker_pos_all[:, indices]  # indices list for this scale group
                    self._scene.draw_debug_spheres(
                        poss=group_marker_pos.view(-1, 3).cpu().numpy(),
                        radius=scale_val,
                        color=marker_dict.marker_color
                    )
            elif marker_dict.marker_type == "arrow":
                # Define a default arrow direction (e.g., pointing along x-axis)
                base_dir = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=marker_quat.dtype)
                # Iterate over each scale group and render all arrow markers with that scale at once.
                for marker_size, indices in marker_dict.scale.items():
                    if marker_size == "small":
                        scale_val = 0.01
                    else:
                        scale_val = 0.05
                    
                    group_marker_pos = marker_pos_all[:, indices].view(-1, 3).cpu().numpy()  # positions for this group
                    group_marker_quat = marker_quat[:, indices]      # quaternions for this group
                    # Flatten the quaternion array for vectorized rotation.
                    flat_quat = group_marker_quat.view(-1, 4)
                    # Repeat base_dir to match the number of quaternions.
                    repeated_base = base_dir.unsqueeze(0).repeat(flat_quat.shape[0], 1)
                    # Compute the arrow vectors using the quaternion rotation.
                    arrow_vectors = rotations.quat_rotate(flat_quat, repeated_base, w_last=self.data_conversion.sim_w_last).view(-1, 3).cpu().numpy()
                    for arrow_idx in range(len(group_marker_pos)):
                        # Genesis does not support vectorized arrow rendering.
                        self._scene.draw_debug_arrow(
                            pos=group_marker_pos[arrow_idx],
                            vec=arrow_vectors[arrow_idx],
                            radius=scale_val,
                            color=marker_dict.marker_color
                        )
            else:
                raise ValueError(f"Marker type {marker_dict.marker_type} not supported")
