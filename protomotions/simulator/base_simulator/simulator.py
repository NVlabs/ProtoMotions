from abc import ABC, abstractmethod
import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import torch
from isaac_utils import torch_utils, rotations
from protomotions.utils.scene_lib import SceneLib
from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
from protomotions.envs.base_env.env_utils.humanoid_utils import build_pd_action_offset_scale
from protomotions.simulator.base_simulator.robot_state import RobotState, DataConversion
from protomotions.simulator.base_simulator.config import MarkerState, VisualizationMarker, ControlType, SimulatorConfig, SimBodyOrdering


class Simulator(ABC):
    # -------------------------
    # ⚙️ Group 1: Initialization & Configuration
    # -------------------------
    def __init__(
        self,
        config: SimulatorConfig,
        terrain: Terrain,
        device: torch.device,
        scene_lib: Optional[SceneLib] = None,
        visualization_markers: Optional[Dict[str, VisualizationMarker]] = None,
    ) -> None:
        """
        Initialize the Simulator with configuration, scene library, terrain,
        visualization markers, and device.
        """
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        
        self.config = config
        self.robot_config = config.robot
        self.device = device
        self.scene_lib = scene_lib
        if self.scene_lib is not None:
            self._num_objects_per_scene: int = self.scene_lib.num_objects_per_scene
        else:
            self._num_objects_per_scene = 0
        self.terrain = terrain
        self.headless: bool = self.config.headless
        self.num_envs: int = self.config.num_envs

        self.control_type: ControlType = self.robot_config.control.control_type
        self.decimation: int = self.config.sim.decimation
        self.simulator_fps: int = self.config.sim.fps

        self.dt: float = self.decimation * 1.0 / self.simulator_fps

        # Scene storage
        self._scene_position: List[torch.Tensor] = []
        self._object_dims: List[torch.Tensor] = []

        self._num_bodies: int = self.robot_config.num_bodies
        self._num_dof: int = len(self.robot_config.dof_names)
        self._dof_limits_lower_sim: torch.Tensor = None
        self._dof_limits_upper_sim: torch.Tensor = None
        self._dof_limits_lower_common: torch.Tensor = None
        self._dof_limits_upper_common: torch.Tensor = None

        self.user_requested_reset: bool = False

        self._camera_target: Dict[str, int] = {"env": 0, "element": 0}
        
        self._user_is_recording, self._user_recording_state_change = False, False
        self._user_recording_video_queue_size = 100000
        self._delete_user_viewer_recordings = False
        os.makedirs("output/renderings", exist_ok=True)
        self._user_recording_video_path = os.path.join(
            "output/renderings", f"{self.config.experiment_name}-%s"
        )

    # -------------------------
    # 🌄 Group 2: Environment Setup & Configuration
    # -------------------------
    def _update_inference_parameters(self) -> None:
        """
        Update parameters required for inference. This is an optional override.
        """
        pass

    def get_scene_positions(self) -> List[torch.Tensor]:
        """
        Return the stored scene positions.

        Returns:
            List[torch.Tensor]: Scene positions.
        """
        return self._scene_position

    def get_object_dims(self) -> List[torch.Tensor]:
        """
        Return the stored object dimensions.

        Returns:
            List[torch.Tensor]: Object dimensions.
        """
        return self._object_dims

    def build_body_ids_tensor(self, body_names: List[str]) -> torch.Tensor:
        """
        Build a tensor of body IDs based on the provided body names.

        Args:
            body_names (List[str]): List of body names.

        Returns:
            torch.Tensor: Tensor containing indices corresponding to the body names.
        """
        body_ids: List[int] = []
        for body_name in body_names:
            body_id = self.robot_config.body_names.index(body_name)
            assert body_id != -1, f"Body part {body_name} not found in {self.robot_config.body_names}"
            body_ids.append(body_id)
        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
            
    def on_environment_ready(self) -> None:
        """
        Configure internal tensors after the simulation environment is initialized.
        This includes conversion tensors for bodies, DOFs, and contact sensors.
        """
        self._process_dof_props()

        body_ordering = self._get_sim_body_ordering()

        body_convert_to_common = torch.tensor(
            [
                body_ordering.body_names.index(body_name)
                for body_name in self.robot_config.body_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        
        body_convert_to_sim = torch.tensor(
            [
                self.robot_config.body_names.index(body_name)
                for body_name in body_ordering.body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        contact_sensor_convert_to_common = torch.tensor(
            [
                body_ordering.contact_sensor_body_names.index(body_name)
                for body_name in self.robot_config.body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        dof_convert_to_sim = torch.tensor(
            [
                self.robot_config.dof_names.index(dof_name)
                for dof_name in body_ordering.dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        dof_convert_to_common = torch.tensor(
            [
                body_ordering.dof_names.index(dof_name)
                for dof_name in self.robot_config.dof_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        
        self.data_conversion = DataConversion(
            body_convert_to_common=body_convert_to_common,
            body_convert_to_sim=body_convert_to_sim,
            contact_sensor_convert_to_common=contact_sensor_convert_to_common,
            dof_convert_to_sim=dof_convert_to_sim,
            dof_convert_to_common=dof_convert_to_common,
            sim_w_last=self.config.w_last,
        )
        
        self._create_legged_robot_tensors()

        self._dof_limits_lower_common = self._dof_limits_lower_sim[self.data_conversion.dof_convert_to_common]
        self._dof_limits_upper_common = self._dof_limits_upper_sim[self.data_conversion.dof_convert_to_common]
        self._common_pd_action_offset, self._common_pd_action_scale = (
            build_pd_action_offset_scale(
                self.robot_config.dof_offsets,
                self._dof_limits_lower_common,
                self._dof_limits_upper_common,
                self.device,
            )
        )

    # -------------------------
    # ⏱️ Group 3: Simulation Steps & State Management
    # -------------------------
    def _requested_reset(self) -> None:
        """
        Set the flag indicating that a user-requested reset has been made.
        """
        print("User requested reset")
        self.user_requested_reset = True

    def step(self, common_actions: torch.Tensor, markers_state: Optional[Dict[str, MarkerState]] = None) -> None:
        """
        Perform a simulation step by:
          1. Converting common actions to simulator-specific actions.
          2. Stepping the physics simulation.
          3. Updating visualization markers.
          4. Rendering the environment.

        Args:
            common_actions (torch.Tensor): Action tensor in common format.
            markers_state (Dict[str, MarkerState]): Dictionary of marker states.
        """
        self.user_requested_reset = False
        common_actions = torch.clamp(common_actions * self.robot_config.control.action_scale, -self.robot_config.control.clamp_actions, self.robot_config.control.clamp_actions)
        self._common_actions = common_actions.to(self.device)
        self._physics_step()
        self._update_markers(markers_state)
        self.render()

    def _update_simulator_tensors_after_reset(self, env_ids: Optional[torch.Tensor]) -> None:
        """
        Update the state of the simulator for the specified environments.
        Default implementation does nothing.
        
        Args:
            new_states (Dict[str, torch.Tensor]): New state data structure.
            env_ids (Optional[torch.Tensor]): Tensor of environment ids to update.
        """
        return

    def reset_envs(self, new_states: Dict[str, torch.Tensor], env_ids: Optional[torch.Tensor]) -> None:
        """
        Reset the specified environments with the given new states.

        Args:
            new_states (Dict[str, torch.Tensor]): New state data structure.
            env_ids (Optional[torch.Tensor]): Tensor of environment ids to reset.
        """
        self.set_env_state(new_states, env_ids)
        self._update_simulator_tensors_after_reset(env_ids)

    def set_env_state(self, new_states: Dict[str, torch.Tensor], env_ids: Optional[torch.Tensor]) -> None:
        """
        Set the state of the simulator for the specified environments.
        
        Updates positions, rotations, velocities, and DOF states according to new_states.

        Args:
            new_states (RobotState): New state data structure.
            env_ids (Optional[torch.Tensor]): Tensor of environment ids to set.
        """
        new_states = new_states.convert_to_sim(self.data_conversion)
        self._set_simulator_env_state(new_states, env_ids)

    @abstractmethod
    def _set_simulator_env_state(
        self, new_states: RobotState, env_ids: Optional[torch.Tensor]
    ) -> None:
        """
        Set the simulator-specific environment state.
        
        Must be implemented by concrete simulator subclasses.

        Args:
            new_states (RobotState): New state data structure as a dataclass.
            env_ids (Optional[torch.Tensor]): Tensor of environment IDs to update.
        """
        raise NotImplementedError

    @abstractmethod
    def _physics_step(self) -> None:
        """
        Advance the physics simulation by one step.
        
        Must be implemented in a simulator-specific manner.
        """
        raise NotImplementedError

    # -------------------------
    # 📊 Group 4: State Getters
    # -------------------------
    def get_default_state(self) -> RobotState:
        """
        Retrieve the default state of the simulator.
        
        Returns:
            RobotState: The default simulator state.
        """
        simulator_default_state: RobotState = self._get_simulator_default_state()
        simulator_default_state = simulator_default_state.convert_to_common(self.data_conversion)
        minimal_height = simulator_default_state.rigid_body_pos[:, :, 2].min(dim=1)[0]
        simulator_default_state.rigid_body_pos[:, :, 2] -= minimal_height.unsqueeze(1)
        simulator_default_state.root_pos[:, 2] -= minimal_height
        
        # Make sure we start with zero velocity
        simulator_default_state.root_vel[:, 2] *= 0
        simulator_default_state.root_ang_vel[:, 2] *= 0
        simulator_default_state.dof_vel *= 0
        simulator_default_state.rigid_body_vel *= 0
        simulator_default_state.rigid_body_ang_vel *= 0
        
        return simulator_default_state
    
    @abstractmethod
    def _get_simulator_default_state(self) -> RobotState:
        """
        Retrieve the default state of the simulator.
        
        Returns:
            RobotState: The default simulator state.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Retrieve the ordering of bodies and DOFs as defined by the simulator.
        
        Returns:
            SimBodyOrdering: A dictionary with keys 'body_names', 'dof_names',
                                  and 'contact_sensor_body_names'.
        """
        raise NotImplementedError

    def get_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the root state of the simulator as an RobotState.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: The environment state corresponding to the robot root.
        """
        simulator_root_state: RobotState = self._get_simulator_root_state(env_ids)
        simulator_root_state = simulator_root_state.convert_to_common(self.data_conversion)
        return simulator_root_state

    @abstractmethod
    def _get_simulator_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the raw simulator root state as an RobotState.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: The raw environment state for the robot root.
        """
        raise NotImplementedError

    def get_bodies_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the simulator's bodies state as an RobotState.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: An RobotState instance with rigid body state fields set.
        """
        bodies_state: RobotState = self._get_simulator_bodies_state(env_ids)
        bodies_state = bodies_state.convert_to_common(self.data_conversion)
        return bodies_state

    @abstractmethod
    def _get_simulator_bodies_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the raw simulator bodies state.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: The raw bodies state (with rigid body fields set).
        """
        raise NotImplementedError

    def get_dof_forces(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve the DOF forces from the simulator.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.
        
        Returns:
            torch.Tensor: Tensor of DOF forces in the simulator's common ordering.
        """
        simulator_dof_forces = self._get_simulator_dof_forces(env_ids)
        return simulator_dof_forces[:, self.data_conversion.dof_convert_to_common]
    
    @abstractmethod
    def _get_simulator_dof_forces(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve the raw simulator DOF forces.
        """
        raise NotImplementedError

    def get_dof_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the simulator's DOF state as an RobotState.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: An RobotState instance with dof_pos and dof_vel set.
        """
        simulator_dof_state: RobotState = self._get_simulator_dof_state(env_ids)
        simulator_dof_state = simulator_dof_state.convert_to_common(self.data_conversion)
        return simulator_dof_state

    @abstractmethod
    def _get_simulator_dof_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the raw simulator DOF state.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: The raw DOF state containing dof_pos and dof_vel.
        """
        raise NotImplementedError

    def get_bodies_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve the bodies' contact buffer.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.
        
        Returns:
            torch.Tensor: Tensor containing contact forces for bodies in the common ordering.
        """
        simulator_bodies_contact_forces = self._get_simulator_bodies_contact_buf(env_ids)
        return simulator_bodies_contact_forces[:, self.data_conversion.contact_sensor_convert_to_common]

    @abstractmethod
    def _get_simulator_bodies_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve the raw simulator buffer of bodies' contact forces.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.
        
        Returns:
            torch.Tensor: Raw bodies contact buffer.
        """
        raise NotImplementedError

    def get_object_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the root state of objects in the simulator as an RobotState.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: The environment state corresponding to objects.
        """
        simulator_object_root_state: RobotState = self._get_simulator_object_root_state(env_ids)
        simulator_object_root_state = simulator_object_root_state.convert_to_common(self.data_conversion)
        return simulator_object_root_state

    @abstractmethod
    def _get_simulator_object_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the raw simulator object root state as an RobotState.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.
        
        Returns:
            RobotState: The raw environment state for object roots.
        """
        raise NotImplementedError

    def get_object_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve object contact forces from the simulator.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.
        
        Returns:
            torch.Tensor: Tensor of object contact forces.
        """
        simulator_object_contact_forces = self._get_simulator_object_contact_buf(env_ids)
        return simulator_object_contact_forces

    @abstractmethod
    def _get_simulator_object_contact_buf(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Retrieve the raw object contact buffer.
        
        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.
        
        Returns:
            torch.Tensor: Raw object contact forces.
        """
        raise NotImplementedError

    # -------------------------
    # 🎮 Group 5: Control & Computation Methods
    # -------------------------
    def _action_to_pd_targets(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert a common action tensor into PD targets for simulation.

        Args:
            action (torch.Tensor): Input actions.
        
        Returns:
            torch.Tensor: PD targets computed as offset + scale * action.
        """
        if self.robot_config.control.map_actions_to_pd_range:
            pd_tar = self._common_pd_action_offset + self._common_pd_action_scale * action
        else:
            pd_tar = action
        return pd_tar

    def _create_legged_robot_tensors(self) -> None:
        """
        Create tensors necessary for simulating legged robots.
        
        This sets up the PD gains, default DOF positions, and related tensors only if an initial state is defined.
        """
        if self.robot_config.init_state is None:
            return

        p_gains: torch.Tensor = torch.zeros(self.robot_config.number_of_actions, dtype=torch.float, device=self.device, requires_grad=False)
        d_gains: torch.Tensor = torch.zeros(self.robot_config.number_of_actions, dtype=torch.float, device=self.device, requires_grad=False)

        default_dof_pos: torch.Tensor = torch.zeros(self._num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self._num_dof):
            name: str = self.robot_config.dof_names[i]
            angle: float = self.robot_config.init_state.default_joint_angles[name]
            default_dof_pos[i] = angle
            found: bool = False
            for dof_name in self.robot_config.control.stiffness.keys():
                if dof_name in name:
                    p_gains[i] = self.robot_config.control.stiffness[dof_name]
                    d_gains[i] = self.robot_config.control.damping[dof_name]
                    found = True
            if not found:
                p_gains[i] = 0.0
                d_gains[i] = 0.0
                if self.robot_config.control.control_type in [ControlType.PROPORTIONAL, ControlType.VELOCITY]:
                    raise ValueError(f"PD gain of joint {name} were not defined.")
        self._default_dof_pos = default_dof_pos.unsqueeze(0)
        self._common_p_gains = p_gains
        self._common_d_gains = d_gains

    def _process_dof_props(self) -> None:
        """
        Process DOF properties from the asset's properties.
        
        For non-built-in PD controllers, initialize torque limits based on provided property effort values.
        
        Args:
            props (Dict[str, torch.Tensor]): Properties of DOFs from the asset.
        """
        if self.robot_config.dof_effort_limits is not None:
            self._torque_limits_common = torch.tensor(self.robot_config.dof_effort_limits, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self._torque_limits_common = torch.ones(self._num_dof, dtype=torch.float, device=self.device, requires_grad=False) * 300.0

    def _compute_torques(self, action: torch.Tensor) -> torch.Tensor:
        """
        Compute torques from actions.

        Actions can be interpreted as position or velocity targets given to a PD controller,
        or directly as scaled torques. The returned torques have the same dimension as the number of DOFs.

        Args:
            action (torch.Tensor): Input actions.
        
        Returns:
            torch.Tensor: Computed torques clipped to the torque limits.
        """
        common_dof_state = self._get_simulator_dof_state().convert_to_common(self.data_conversion)

        if self.control_type == ControlType.PROPORTIONAL:
            # Map action to dof ranges.
            pd_tar = self._action_to_pd_targets(action)

            if self.robot_config.control.use_biased_controller:
                pd_tar += self._default_dof_pos

            torques: torch.Tensor = (
                    self._common_p_gains * (pd_tar - common_dof_state.dof_pos)
                    - self._common_d_gains * common_dof_state.dof_vel
            )
        elif self.control_type == ControlType.VELOCITY:
            raise NotImplementedError("Velocity control is not properly implemented yet.")
            torques = (
                    self._common_p_gains * (action - dof_state.dof_vel)
                    - self._common_d_gains * (dof_state.dof_vel - self.last_dof_vel) / self.dt
            )
        elif self.control_type == ControlType.TORQUE:
            torques = action
        else:
            raise NameError(f"Unknown controller type: {self.control_type}")
        return torch.clip(torques, -self._torque_limits_common, self._torque_limits_common)

    # -------------------------
    # 🎨 Group 6: Rendering & Visualization
    # -------------------------
    def _toggle_camera_target(self) -> None:
        """
        Toggle the camera target between different environments and objects.
        
        The target cycles through all objects in the scene, with 0 referring to the environment.
        """
        if self.scene_lib is not None:
            self._camera_target["element"] = (self._camera_target["element"] + 1) % (self._num_objects_per_scene + 1)
            print("Updated camera target to element", self._camera_target["element"])

        if self._camera_target["element"] == 0:
            self._camera_target["env"] = (self._camera_target["env"] + 1) % self.num_envs
            print("Updated camera target to env", self._camera_target["env"])

    def render(self):
        """
        Render the current simulation state and handle video recording if enabled.
        
        This method manages:
        1. Video recording state transitions and initialization
        2. Frame capture and saving during recording
        3. Video compilation when recording ends
        4. Cleanup of temporary image files
        """
        if not self.headless:
            # Handle recording state transitions
            if self._user_recording_state_change:
                if self._user_is_recording:
                    # Initialize new recording
                    self._user_recording_video_queue = deque(
                        maxlen=self._user_recording_video_queue_size
                    )
                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    self._curr_user_recording_name = (
                        self._user_recording_video_path % curr_date_time
                    )
                    self._user_recording_frame = 0
                    
                    self._recorded_motion = {"global_translation": [], "global_rotation": []}
                    
                    if not os.path.exists(self._curr_user_recording_name):
                        os.makedirs(self._curr_user_recording_name)
                    print(f"Started recording to folder {self._curr_user_recording_name}")
                    
                else:
                    # Finalize recording and create video
                    from moviepy.editor import ImageSequenceClip
                    
                    image_dir = self._curr_user_recording_name
                    images = sorted([
                        os.path.join(image_dir, f)
                        for f in os.listdir(image_dir)
                        if f.endswith('.png')
                    ])

                    clip = ImageSequenceClip(images, fps=30)
                    clip.write_videofile(
                        f"{self._curr_user_recording_name}.mp4",
                        codec='libx264',
                        audio=False,
                        threads=32,
                        preset='veryfast',
                        ffmpeg_params=[
                            '-profile:v', 'main',
                            '-level', '4.0',
                            '-pix_fmt', 'yuv420p', 
                            '-movflags', '+faststart',
                            '-crf', '23',
                            '-x264-params', 'keyint=60:min-keyint=30'
                        ]
                    )
                    self._delete_user_viewer_recordings = True
                    print(f"Video saved to {self._curr_user_recording_name}.mp4")
                    
                    # Save the recorded motion to a file
                    global_translation = torch.cat(self._recorded_motion["global_translation"], dim=0)
                    global_rotation = torch.cat(self._recorded_motion["global_rotation"], dim=0)
                    with open(f"{self._curr_user_recording_name}.pt", "wb") as f:
                        torch.save({"global_translation": global_translation, "global_rotation": global_rotation}, f)
                    self._recorded_motion = None
                    
                self._user_recording_state_change = False

            # Capture frame if recording
            if self._user_is_recording:
                file_name = (
                    self._curr_user_recording_name
                    + "/%04d.png" % self._user_recording_frame
                )
                self._write_viewport_to_file(file_name)
                self._user_recording_frame += 1
                
                bodies_state = self.get_bodies_state()
                self._recorded_motion["global_translation"].append(bodies_state.rigid_body_pos)
                self._recorded_motion["global_rotation"].append(bodies_state.rigid_body_rot)

            # Clean up temporary files if needed
            if self._delete_user_viewer_recordings:
                images = [
                    img 
                    for img in os.listdir(self._curr_user_recording_name)
                    if img.endswith(".png")
                ]
                for image in images:
                    os.remove(os.path.join(self._curr_user_recording_name, image))
                os.removedirs(self._curr_user_recording_name)
                self._delete_user_viewer_recordings = False
                self._recorded_motion = None

    @abstractmethod
    def _write_viewport_to_file(self, file_name: str) -> None:
        """
        Write the current viewport to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_camera(self) -> None:
        """
        Initialize the camera for visualization.
        
        Must be implemented in a simulator-specific manner.
        """
        raise NotImplementedError
    
    def _toggle_video_record(self):
        self._user_is_recording = not self._user_is_recording
        self._user_recording_state_change = True

    def _cancel_video_record(self):
        self._user_is_recording = False
        self._user_recording_state_change = False
        self._delete_user_viewer_recordings = True


    def _update_markers(self, markers_state: Optional[Dict[str, MarkerState]] = None) -> None:
        """
        Update visualization markers for the simulator.

        Converts marker orientations if necessary and delegates to the simulator-specific update.
        
        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """
        if not markers_state:
            return

        if not self.config.w_last:
            for key in markers_state.keys():
                markers_state[key].orientation = rotations.xyzw_to_wxyz(markers_state[key].orientation)
        self._update_simulator_markers(markers_state)

    @abstractmethod
    def _update_simulator_markers(self, markers_state: Optional[Dict[str, MarkerState]] = None) -> None:
        """
        Simulator-specific update of marker states.
        
        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Close the simulator and perform cleanup operations.
        
        Must be implemented in a simulator-specific manner.
        """
        raise NotImplementedError
