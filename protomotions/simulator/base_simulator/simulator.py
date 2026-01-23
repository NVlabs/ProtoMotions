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
"""Base simulator interface for physics engines.

This module defines the abstract base class for physics simulators. It provides a
unified interface across different physics engines (IsaacGym, IsaacLab, Genesis, Newton)
while handling simulator-specific details in subclasses.

Key Classes:
    - Simulator: Abstract base class for all physics simulators

Key Features:
    - Unified robot state representation
    - Multi-simulator support with consistent API
    - PD control and torque control
    - Terrain integration
    - Scene and object management
    - Visualization marker system
    - Domain randomization support
"""

from abc import ABC, abstractmethod
import os

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable

import torch
from protomotions.utils import rotations

from protomotions.components.scene_lib import SceneLib
from protomotions.components.terrains.terrain import Terrain
from protomotions.simulator.base_simulator.utils import (
    build_motion_data,
    build_pd_action_offset_scale,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    DataConversionMapping,
    RootOnlyState,
    ObjectState,
    ResetState,
)
from protomotions.simulator.base_simulator.config import (
    MarkerState,
    VisualizationMarkerConfig,
    SimulatorConfig,
    SimBodyOrdering,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    get_matching_indices,
)
from protomotions.robot_configs.base import ControlType, RobotConfig


class Simulator(ABC):
    """Base class for physics simulators.

    Provides a unified interface for different physics engines (IsaacGym, IsaacLab, Genesis, Newton).
    Handles robot spawning, environment setup, scene management, terrain integration,
    and state management. Subclasses implement simulator-specific details while
    maintaining a consistent API.

    Key responsibilities:
    - **Environment setup**: Spawns robots, objects, and terrain
    - **State management**:
        - Getters return RobotState with full rigid body data (FK computed i.e. max coord)
        - Setters accept ResetState with only root + DOF (simulators compute FK from reduced corrd)
    - **Control**: Applies PD control or direct torques
    - **Visualization**: Manages markers and rendering
    - **Data conversion**: Handles ordering differences between simulators

    Args:
        config: Simulator configuration (num_envs, physics params, etc.).
        robot_config: Robot morphology and control configuration.
        terrain: Optional terrain for complex ground surfaces.
        device: PyTorch device for computations.
        scene_lib: Optional scene library for object spawning.
        visualization_markers: Optional markers for visualization.

    Attributes:
        num_envs: Number of parallel environments.
        dt: Simulation timestep.
        robot_state: Current robot state in unified format.

    Example:
        >>> from protomotions.simulator.isaacgym.simulator import IsaacGymSimulator
        >>> sim = IsaacGymSimulator(config, robot_config, device=device)
        >>> sim.reset()
        >>> for _ in range(1000):
        >>>     actions = policy(sim.robot_state)
        >>>     sim.step(actions)
    """

    # -------------------------
    # ⚙️ Group 1: Initialization & Configuration
    # -------------------------
    def __init__(
        self,
        config: SimulatorConfig,
        robot_config: RobotConfig,
        terrain: Optional[Terrain],
        device: torch.device,
        scene_lib: SceneLib,
    ) -> None:
        """Initialize the Simulator shell without creating simulation.

        Creates a minimal simulator shell. The actual simulation is created later
        via _initialize_with_markers() after Env creates visualization markers.

        Args:
            config: Simulator configuration including num_envs and physics parameters.
            robot_config: Robot morphology, control parameters, and asset files.
            terrain: Terrain instance (can be None for some visualizers).
            device: PyTorch device for tensor operations.
            scene_lib: SceneLib instance (always provided, can be empty).
        """
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.config = config
        self.robot_config = robot_config
        self.device = device
        self.scene_lib = scene_lib  # Always provided (empty if no scenes)
        self.terrain = terrain  # Always provided
        self.headless: bool = self.config.headless
        self.num_envs: int = self.config.num_envs

        self.control_type: ControlType = self.robot_config.control.control_type
        self.decimation: int = self.config.sim.decimation
        self.dt: float = self.decimation * 1.0 / self.config.sim.fps

        self._num_bodies: int = self.robot_config.kinematic_info.num_bodies
        self._num_dof: int = self.robot_config.kinematic_info.num_dofs
        self._dof_names: List[str] = self.robot_config.kinematic_info.dof_names
        self._body_names: List[str] = self.robot_config.kinematic_info.body_names
        # Joint limits are now parsed from MJCF by pose_lib.py
        # Simulator-specific limits are only retrieved for verification via _get_simulator_dof_limits_for_verification()

        self._domain_randomization: Dict[str, Any] = (
            self._process_domain_randomization()
        )

        self.user_requested_reset: bool = False

        self._camera_target: Dict[str, int] = {"env": 0, "element": 0}
        self._show_markers: bool = True
        self._simulation_running: bool = True

        self._user_is_recording, self._user_recording_state_change = False, False
        self._user_recording_video_queue_size = 100000
        self._delete_user_viewer_recordings = False
        os.makedirs("output/renderings", exist_ok=True)
        self._user_recording_video_path = os.path.join(
            "output/renderings", f"{self.config.experiment_name}-%s"
        )
        self._common_actions = torch.zeros(
            self.num_envs,
            self.robot_config.number_of_actions,
            device=self.device,
            dtype=torch.float,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            self.robot_config.number_of_actions,
            device=self.device,
            dtype=torch.float,
        )

        # Two-phase initialization support
        self._initialized = False
        self._visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]] = (
            None
        )

    def _initialize_with_markers(
        self, visualization_markers: Optional[Dict[str, VisualizationMarkerConfig]]
    ) -> None:
        """Finalize simulator initialization with visualization markers.

        Called by Env after it creates task-specific markers. This triggers
        the actual simulation creation in subclasses.

        Args:
            visualization_markers: Visualization markers configuration created by Env
        """
        if self._initialized:
            raise RuntimeError("Simulator already initialized")

        self._visualization_markers = visualization_markers
        # Call simulator-specific initialization (subclass implements this)
        self._create_simulation()
        # Setup data conversion and finalize
        self._finalize_setup()
        self._initialized = True

    @abstractmethod
    def _create_simulation(self) -> None:
        """Create the actual simulation environment.

        Subclasses must implement this to create their simulation environments,
        load assets, and prepare for physics simulation. Can access
        self._visualization_markers set by _initialize_with_markers().
        """
        raise NotImplementedError

    # -------------------------
    # 🌄 Group 2: Environment Setup & Configuration
    # -------------------------
    def _finalize_setup(self) -> None:
        """
        Configure internal tensors after the simulation environment is initialized.
        This includes conversion tensors for bodies, DOFs, and contact sensors.
        """
        self._process_control_properties()

        body_ordering = self._get_sim_body_ordering()

        body_convert_to_common = torch.tensor(
            [
                body_ordering.body_names.index(body_name)
                for body_name in self._body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        body_convert_to_sim = torch.tensor(
            [
                self._body_names.index(body_name)
                for body_name in body_ordering.body_names
            ],
            dtype=torch.long,
            device=self.device,
        )

        dof_convert_to_sim = torch.tensor(
            [self._dof_names.index(dof_name) for dof_name in body_ordering.dof_names],
            dtype=torch.long,
            device=self.device,
        )
        dof_convert_to_common = torch.tensor(
            [body_ordering.dof_names.index(dof_name) for dof_name in self._dof_names],
            dtype=torch.long,
            device=self.device,
        )

        self.data_conversion = DataConversionMapping(
            body_convert_to_common=body_convert_to_common,
            body_convert_to_sim=body_convert_to_sim,
            dof_convert_to_sim=dof_convert_to_sim,
            dof_convert_to_common=dof_convert_to_common,
            sim_w_last=self.config.w_last,
        )

        # Use joint limits from KinematicInfo instead of simulator-specific ones
        # Verify that simulator-specific limits match the parsed ones
        self._verify_joint_limits()

        self._common_pd_action_offset, self._common_pd_action_scale = (
            build_pd_action_offset_scale(
                self.robot_config.kinematic_info.hinge_axes_map,
                self.robot_config.kinematic_info.dof_limits_lower.to(self.device),
                self.robot_config.kinematic_info.dof_limits_upper.to(self.device),
                self.robot_config.control.action_scale,
                self.device,
            )
        )
        
        # Initialize push randomization state
        self._init_push_randomization()

    def _init_push_randomization(self) -> None:
        """Initialize push randomization state buffers."""
        push_cfg = None
        if (
            self.config.domain_randomization is not None
            and self.config.domain_randomization.push is not None
            and self.config.domain_randomization.push.has_push()
        ):
            push_cfg = self.config.domain_randomization.push
        
        self._push_enabled = push_cfg is not None
        
        if self._push_enabled:
            self._simulation_time = torch.zeros(self.num_envs, device=self.device)
            self._push_next_time = torch.zeros(self.num_envs, device=self.device)
            self._push_interval_range = push_cfg.push_interval_range
            self._push_max_lin_vel = torch.tensor(
                push_cfg.max_linear_velocity, device=self.device, dtype=torch.float
            )
            self._push_max_ang_vel = torch.tensor(
                push_cfg.max_angular_velocity, device=self.device, dtype=torch.float
            )
            self._schedule_push(torch.arange(self.num_envs, device=self.device))

    def _schedule_push(self, env_ids: torch.Tensor) -> None:
        """Schedule next push time for specified environments."""
        if not self._push_enabled or len(env_ids) == 0:
            return
        
        interval_min, interval_max = self._push_interval_range
        random_intervals = (
            torch.rand(len(env_ids), device=self.device)
            * (interval_max - interval_min)
            + interval_min
        )
        self._push_next_time[env_ids] = self._simulation_time[env_ids] + random_intervals

    def _apply_push_if_due(self) -> None:
        """Check if any environments are due for a push and apply it."""
        if not self._push_enabled:
            return
        
        due_mask = self._simulation_time >= self._push_next_time
        if not due_mask.any():
            return
        
        due_env_ids = torch.where(due_mask)[0]
        num_due = len(due_env_ids)
        
        lin_vel = (
            (torch.rand(num_due, 3, device=self.device) * 2 - 1)
            * self._push_max_lin_vel
        )
        ang_vel = (
            (torch.rand(num_due, 3, device=self.device) * 2 - 1)
            * self._push_max_ang_vel
        )
        
        self._apply_root_velocity_impulse(lin_vel, ang_vel, due_env_ids)
        self._schedule_push(due_env_ids)

    @abstractmethod
    def _apply_root_velocity_impulse(
        self,
        linear_velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        """Apply velocity impulse to robot root.
        
        Adds the given velocities to the robot's current root velocities.
        
        Args:
            linear_velocity: Linear velocity impulse [num_envs, 3] in m/s.
            angular_velocity: Angular velocity impulse [num_envs, 3] in rad/s.
            env_ids: Environment indices to apply impulse to.
        """
        raise NotImplementedError

    def _push_robot(self) -> None:
        """Apply a random push to all robots (triggered by user button press)."""
        push_magnitude = 1.0
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        
        random_dir = torch.randn(self.num_envs, 3, device=self.device)
        random_dir = random_dir / (torch.norm(random_dir, dim=-1, keepdim=True) + 1e-8)
        lin_vel = random_dir * push_magnitude
        ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        self._apply_root_velocity_impulse(lin_vel, ang_vel, all_env_ids)
        print("Push applied to all robots")

    def _verify_joint_limits(self) -> None:
        """
        Verify that if we instead load the joint limits from the simulator's internal API,
        they match those parsed from MJCF with pose_lib.py.

        This is useful for verifying that the joint limits are correctly parsed from MJCF.
        It also serves as a sanity check that the simulator's internal API is correctly implemented.
        """
        try:
            # Get simulator's internal joint limits for verification
            sim_lower, sim_upper = self._get_simulator_dof_limits_for_verification()

            # Convert simulator limits to common ordering for comparison
            sim_lower_common = sim_lower[self.data_conversion.dof_convert_to_common]
            sim_upper_common = sim_upper[self.data_conversion.dof_convert_to_common]

            # Get MJCF-parsed limits directly from robot_config
            dof_limits_lower = self.robot_config.kinematic_info.dof_limits_lower.to(
                self.device
            )
            dof_limits_upper = self.robot_config.kinematic_info.dof_limits_upper.to(
                self.device
            )

            # Compare with MJCF-parsed limits
            lower_diff = torch.abs(sim_lower_common - dof_limits_lower)
            upper_diff = torch.abs(sim_upper_common - dof_limits_upper)

            tolerance = 1e-5

            # Check for mismatches and raise errors instead of printing warnings
            for i, dof_name in enumerate(self._dof_names):
                if lower_diff[i] > tolerance:
                    raise ValueError(
                        f"Joint limit mismatch for {dof_name} (lower): "
                        f"MJCF={dof_limits_lower[i]:.4f}, "
                        f"Simulator={sim_lower_common[i]:.4f}"
                    )
                if upper_diff[i] > tolerance:
                    raise ValueError(
                        f"Joint limit mismatch for {dof_name} (upper): "
                        f"MJCF={dof_limits_upper[i]:.4f}, "
                        f"Simulator={sim_upper_common[i]:.4f}"
                    )
        except NotImplementedError:
            # Simulator hasn't implemented verification yet - raise error
            raise NotImplementedError(
                f"{self.__class__.__name__} has not implemented _get_simulator_dof_limits_for_verification()"
            )
        except Exception as e:
            # Re-raise any other exceptions
            if not isinstance(e, (ValueError, NotImplementedError)):
                raise RuntimeError(f"Failed to verify joint limits: {e}") from e
            else:
                raise

    # -------------------------
    # ⏱️ Group 3: Simulation Steps & State Management
    # -------------------------
    def _requested_reset(self) -> None:
        """
        Set the flag indicating that a user-requested reset has been made.
        """
        print("User requested reset")
        self.user_requested_reset = True

    def get_previous_actions(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the previous actions.
        """
        if env_ids is not None:
            return self._previous_actions[env_ids]
        return self._previous_actions

    def get_current_actions(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the current actions.
        """
        if env_ids is not None:
            return self._common_actions[env_ids]
        return self._common_actions

    def step(
        self,
        common_actions: torch.Tensor,
        markers_callback: Optional[Callable[[], Dict[str, MarkerState]]] = None,
    ) -> None:
        """
        Perform a simulation step by:
          1. Converting common actions to simulator-specific actions.
          2. Stepping the physics simulation.
          3. Updating visualization markers (via callback to get fresh state).
          4. Rendering the environment.

        Args:
            common_actions (torch.Tensor): Action tensor in common format.
            markers_callback (Callable): Optional callback function that returns marker states.
                                        Called after physics step but before rendering.
        """
        # Store the previous actions
        self._previous_actions = self._common_actions.clone()
        self.user_requested_reset = False
        self._common_actions = common_actions.to(self.device)
        self._physics_step()
        
        # Update simulation time and apply push randomization
        if self._push_enabled:
            self._simulation_time += self.dt
            self._apply_push_if_due()

        # Get fresh markers state after physics step
        markers_state = markers_callback() if markers_callback is not None else None
        self._update_markers(markers_state)

        self.render()

    def reset_envs(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Reset the specified environments with the given new states.

        Args:
            new_states: Reset state containing root pose/vel and DOF pos/vel.
                       Simulators will compute FK internally - do NOT provide rigid_body_pos/rot/vel.
            new_object_states: Optional object states.
            env_ids: Tensor of environment ids to reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        new_states = new_states.convert_to_sim(self.data_conversion)
        
        self._previous_actions[env_ids] = 0.0
        if new_object_states is not None:
            new_object_states = new_object_states.convert_to_sim(self.data_conversion)
        self._set_simulator_env_state(new_states, new_object_states, env_ids)
        
        # Reset push randomization state for reset environments
        if self._push_enabled:
            self._simulation_time[env_ids] = 0.0
            self._schedule_push(env_ids)

    @abstractmethod
    def _set_simulator_env_state(
        self,
        new_states: ResetState,
        new_object_states: Optional[ObjectState] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Apply reset state to simulation environments.

        IMPORTANT: new_states is ResetState with only root + DOF state.
        Simulators must compute forward kinematics internally to update rigid body positions/rotations.
        Never pass or expect full RobotState with rigid_body_pos/rot/vel - those are outputs, not inputs.

        Args:
            new_states: Reset state containing root pose/vel and DOF pos/vel.
            new_object_states: Optional object states.
            env_ids: Tensor of environment IDs to update.
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
    def get_default_robot_reset_state(self) -> ResetState:
        """
        Get default reset state for the robot.

        Uses robot_config.default_dof_pos if specified, otherwise zeros.
        Root position uses robot_config.default_root_height for z-axis.
        All velocities are zero.

        Returns:
            ResetState: Default reset state in COMMON format.
        """
        from protomotions.simulator.base_simulator.simulator_state import (
            StateConversion,
        )

        num_envs = self.num_envs
        device = self.device

        # DOF positions: from robot_config (guaranteed to be set in post_init, defaults to zeros)
        dof_pos = (
            self.robot_config.default_dof_pos.unsqueeze(0)
            .repeat(num_envs, 1)
            .to(device)
        )

        # Root pose
        root_pos = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        root_pos[:, 2] = self.robot_config.default_root_height
        root_rot = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )  # xyzw

        # Zero velocities
        dof_vel = torch.zeros_like(dof_pos)
        root_vel = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        root_ang_vel = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)

        return ResetState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            state_conversion=StateConversion.COMMON,
        )

    @abstractmethod
    def _get_sim_body_ordering(self) -> SimBodyOrdering:
        """
        Retrieve the ordering of bodies and DOFs as defined by the simulator.

        Returns:
            SimBodyOrdering: A dictionary with keys 'body_names', 'dof_names',
                                  and 'contact_sensor_body_names'.
        """
        raise NotImplementedError

    def get_root_state(self, env_ids: Optional[torch.Tensor] = None) -> RootOnlyState:
        """
        Retrieve the root state of the simulator as an RootOnlyState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RootOnlyState: The environment state corresponding to the robot root.
        """
        simulator_root_state: RootOnlyState = self._get_simulator_root_state(env_ids)
        simulator_root_state = simulator_root_state.convert_to_common(
            self.data_conversion
        )
        return simulator_root_state

    @abstractmethod
    def _get_simulator_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RootOnlyState:
        """
        Retrieve the raw simulator root state as an RootOnlyState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RootOnlyState: The raw environment state for the robot root.
        """
        raise NotImplementedError

    def get_robot_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the simulator's bodies and DOF state as an RobotState.
        """
        bodies_state: RobotState = self.get_bodies_state(env_ids)
        dof_state: RobotState = self.get_dof_state(env_ids)
        contact_state: RobotState = self.get_binary_body_contacts(env_ids)
        dof_forces: torch.Tensor = self.get_dof_forces(env_ids)
        bodies_state.merge_fields_from(dof_state)
        bodies_state.merge_fields_from(contact_state)
        bodies_state.merge_fields_from(dof_forces)
        return bodies_state

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
    def _get_simulator_bodies_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the raw simulator bodies state.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw bodies state (with rigid body fields set).
        """
        raise NotImplementedError

    def get_dof_forces(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
        """
        Retrieve the DOF forces from the simulator.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            RobotState: RobotState containing DOF forces in the simulator's common ordering.
        """
        simulator_dof_forces = self._get_simulator_dof_forces(env_ids)
        simulator_dof_forces = simulator_dof_forces.convert_to_common(
            self.data_conversion
        )
        return simulator_dof_forces

    @abstractmethod
    def _get_simulator_dof_forces(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the raw simulator DOF forces.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw DOF forces.
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
        simulator_dof_state = simulator_dof_state.convert_to_common(
            self.data_conversion
        )
        return simulator_dof_state

    @abstractmethod
    def _get_simulator_dof_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the raw simulator DOF state.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw DOF state containing dof_pos and dof_vel.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_simulator_dof_limits_for_verification(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve DOF limits from the simulator's internal API for verification purposes only.

        This method should query the simulator's internal representation of joint limits
        and return them in the simulator's native DOF ordering. These limits are used
        solely for verification against the MJCF-parsed limits and should NOT be used
        for any control or computation purposes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (lower_limits, upper_limits)
                                              in the simulator's DOF ordering.
        """
        raise NotImplementedError

    def get_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> RobotState:
        """
        Retrieve the bodies' contact buffer.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            torch.Tensor: Tensor containing contact forces for bodies in the common ordering.
        """
        simulator_bodies_contact_forces: RobotState = (
            self._get_simulator_bodies_contact_buf(env_ids)
        )
        simulator_bodies_contact_forces = (
            simulator_bodies_contact_forces.convert_to_common(self.data_conversion)
        )
        return simulator_bodies_contact_forces

    def get_binary_body_contacts(
        self, env_ids: Optional[torch.Tensor] = None, threshold: float = 0.01
    ) -> RobotState:
        """
        Get binary contact flags for specified bodies.

        Converts contact forces to binary contact indicators based on force magnitude.
        This is the canonical method for computing contact states from simulator forces.

        Args:
            body_ids: Indices of bodies to get contacts for [num_bodies]
            threshold: Force magnitude threshold in Newtons (default: 0.01)
            env_ids: Optional environment indices to query

        Returns:
            Binary contact flags [num_envs, num_bodies] as float (0.0 or 1.0)
        """
        contact_state = self.get_bodies_contact_buf(env_ids)
        force_magnitudes = torch.norm(
            contact_state.rigid_body_contact_forces, dim=-1
        )  # [num_envs, num_bodies]
        binary_contacts = (force_magnitudes > threshold).float()
        contact_state.rigid_body_contacts = binary_contacts

        contact_state = contact_state.convert_to_common(self.data_conversion)
        return contact_state

    @abstractmethod
    def _get_simulator_bodies_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve the raw simulator buffer of bodies' contact forces.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            torch.Tensor: Raw bodies contact buffer.
        """
        raise NotImplementedError

    def get_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the root state of objects in the simulator as an RobotState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The environment state corresponding to objects.
        """
        simulator_object_root_state: ObjectState = (
            self._get_simulator_object_root_state(env_ids)
        )
        simulator_object_root_state = simulator_object_root_state.convert_to_common(
            self.data_conversion
        )
        return simulator_object_root_state

    @abstractmethod
    def _get_simulator_object_root_state(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the raw simulator object root state as an RobotState.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment IDs.

        Returns:
            RobotState: The raw environment state for object roots.
        """
        raise NotImplementedError

    def get_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve object contact forces from the simulator.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            ObjectState: Containing tensor of object contact forces.
        """
        simulator_object_contact_forces = self._get_simulator_object_contact_buf(
            env_ids
        )
        return simulator_object_contact_forces

    @abstractmethod
    def _get_simulator_object_contact_buf(
        self, env_ids: Optional[torch.Tensor] = None
    ) -> ObjectState:
        """
        Retrieve the raw object contact buffer.

        Args:
            env_ids (Optional[torch.Tensor]): Optional tensor of environment ids.

        Returns:
            ObjectState: Raw object contact forces.
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
        pd_tar = self._common_pd_action_offset + self._common_pd_action_scale * action
        return pd_tar

    def _action_to_torque_targets(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert a common action tensor into torque targets for simulation.

        Maps actions from [-1, 1] to [-torque_limit, torque_limit].

        Args:
            action (torch.Tensor): Input actions in range [-1, 1].

        Returns:
            torch.Tensor: Torque targets scaled to torque limits.
        """
        return action * self._torque_limits_common

    @abstractmethod
    def _apply_simulator_pd_targets(self, pd_targets: torch.Tensor) -> None:
        """
        Apply PD position targets using the simulator's internal PD controller.

        Called by _apply_control() when control_type is BUILT_IN_PD.
        pd_targets are already in simulator ordering.

        Args:
            pd_targets (torch.Tensor): PD position targets in simulator DOF ordering.
        """
        raise NotImplementedError

    @abstractmethod
    def _apply_simulator_torques(self, torques: torch.Tensor) -> None:
        """
        Apply torques/forces to DOFs using the simulator's API.

        Called by _apply_control() when control_type is PROPORTIONAL or TORQUE.
        torques are already in simulator ordering.

        Args:
            torques (torch.Tensor): Torques in simulator DOF ordering.
        """
        raise NotImplementedError

    def _apply_control(self) -> None:
        """
        Apply control based on control type.

        All three control modes (BUILT_IN_PD, PROPORTIONAL, TORQUE) are co-located here.
        Child simulators call this method from _physics_step() instead of branching
        on control_type themselves.
        """
        if self.control_type == ControlType.BUILT_IN_PD:
            pd_targets = self._action_to_pd_targets(self._common_actions)
            
            if self._domain_randomization is not None and "action_noise" in self._domain_randomization:
                pd_targets[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]
            
            sim_targets = pd_targets[:, self.data_conversion.dof_convert_to_sim]
            self._apply_simulator_pd_targets(sim_targets)
        elif self.control_type == ControlType.PROPORTIONAL:
            pd_tar = self._action_to_pd_targets(self._common_actions)
            
            if self._domain_randomization is not None and "action_noise" in self._domain_randomization:
                pd_tar[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]

            common_dof_state = self._get_simulator_dof_state().convert_to_common(
                self.data_conversion
            )
            torques = (
                self._common_p_gains * (pd_tar - common_dof_state.dof_pos)
                - self._common_d_gains * common_dof_state.dof_vel
            )
            torques = torch.clip(
                torques, -self._torque_limits_common, self._torque_limits_common
            )
            sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
            self._apply_simulator_torques(sim_torques)
        elif self.control_type == ControlType.TORQUE:
            torques = self._action_to_torque_targets(self._common_actions)
            if self._domain_randomization is not None and "action_noise" in self._domain_randomization:
                torques[
                    ..., self._domain_randomization["action_noise"]["dof_indices"]
                ] += self._domain_randomization["action_noise"]["action_noise"]

            torques = torch.clip(
                torques, -self._torque_limits_common, self._torque_limits_common
            )
            sim_torques = torques[:, self.data_conversion.dof_convert_to_sim]
            self._apply_simulator_torques(sim_torques)
        else:
            raise NameError(f"Unknown controller type: {self.control_type}")

    def _process_control_properties(self) -> None:
        """
        Process control properties from robot config.

        Creates tensors for:
        - PD gains (stiffness and damping)
        - Torque/effort limits
        """

        # Initialize tensors
        p_gains = torch.zeros(
            self.robot_config.number_of_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        d_gains = torch.zeros(
            self.robot_config.number_of_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_effort_limits = torch.ones(
            self.robot_config.number_of_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Populate from robot config
        for dof_name in self.robot_config.kinematic_info.dof_names:
            dof_idx = self.robot_config.kinematic_info.dof_names.index(dof_name)
            dof_info = self.robot_config.control.control_info[dof_name]

            # PD gains
            assert (
                dof_info.stiffness is not None and dof_info.damping is not None
            ), f"PD gains must be defined for DOF {dof_name}"
            p_gains[dof_idx] = dof_info.stiffness
            d_gains[dof_idx] = dof_info.damping

            # Effort limits
            if dof_info.effort_limit is not None:
                dof_effort_limits[dof_idx] = dof_info.effort_limit

        self._common_p_gains = p_gains
        self._common_d_gains = d_gains
        self._torque_limits_common = dof_effort_limits

    def _process_domain_randomization(self) -> None:
        """
        Process domain randomization from the config.
        """
        if self.config.domain_randomization is None:
            return

        domain_randomization_dict: Dict[str, Any] = {}
        if self.config.domain_randomization.action_noise is not None:
            domain_randomization_dict["action_noise"] = (
                self._process_action_noise_domain_randomization(
                    self.config.domain_randomization.action_noise
                )
            )
        if self.config.domain_randomization.friction is not None:
            domain_randomization_dict["friction"] = (
                self._process_friction_domain_randomization(
                    self.config.domain_randomization.friction
                )
            )
        if self.config.domain_randomization.center_of_mass is not None:
            domain_randomization_dict["center_of_mass"] = (
                self._process_center_of_mass_domain_randomization(
                    self.config.domain_randomization.center_of_mass
                )
            )

        return domain_randomization_dict

    def _process_action_noise_domain_randomization(
        self, domain_randomization: ActionNoiseDomainRandomizationConfig
    ) -> None:
        """
        Process action noise domain randomization.
        """
        dof_indices = get_matching_indices(
            self.robot_config.kinematic_info.dof_names,
            domain_randomization.dof_names,
            domain_randomization.dof_indices,
        )
        num_matching_dofs = len(dof_indices)
        action_noise = (
            torch.rand(self.num_envs, num_matching_dofs, device=self.device)
            * (
                domain_randomization.action_noise_range[1]
                - domain_randomization.action_noise_range[0]
            )
            + domain_randomization.action_noise_range[0]
        )

        noise_dict = {"dof_indices": dof_indices, "action_noise": action_noise}
        return noise_dict

    def _process_friction_domain_randomization(
        self, domain_randomization: FrictionDomainRandomizationConfig
    ) -> None:
        """
        Process friction domain randomization.
        """
        body_indices = get_matching_indices(
            self.robot_config.kinematic_info.body_names,
            domain_randomization.body_names,
            domain_randomization.body_indices,
        )
        num_matching_bodies = len(body_indices)

        static_friction = dynamic_friction = restitution = None

        num_samples = min(self.num_envs, domain_randomization.num_buckets)

        if domain_randomization.static_friction_range is not None:
            static_friction = (
                torch.rand(num_samples, num_matching_bodies)
                * (
                    domain_randomization.static_friction_range[1]
                    - domain_randomization.static_friction_range[0]
                )
                + domain_randomization.static_friction_range[0]
            )
            # # or linspace?
            # static_friction = torch.linspace(domain_randomization.static_friction_range[0], domain_randomization.static_friction_range[1], num_samples)
            # static_friction = static_friction.unsqueeze(1).repeat(1, num_matching_bodies)
        if domain_randomization.dynamic_friction_range is not None:
            dynamic_friction = (
                torch.rand(num_samples, num_matching_bodies)
                * (
                    domain_randomization.dynamic_friction_range[1]
                    - domain_randomization.dynamic_friction_range[0]
                )
                + domain_randomization.dynamic_friction_range[0]
            )
            # # or linspace?
            # dynamic_friction = torch.linspace(domain_randomization.dynamic_friction_range[0], domain_randomization.dynamic_friction_range[1], num_samples)
            # dynamic_friction = dynamic_friction.unsqueeze(1).repeat(1, num_matching_bodies)
        if domain_randomization.restitution_range is not None:
            restitution = (
                torch.rand(num_samples, num_matching_bodies)
                * (
                    domain_randomization.restitution_range[1]
                    - domain_randomization.restitution_range[0]
                )
                + domain_randomization.restitution_range[0]
            )
            # # or linspace?
            # restitution = torch.linspace(domain_randomization.restitution_range[0], domain_randomization.restitution_range[1], num_samples)
            # restitution = restitution.unsqueeze(1).repeat(1, num_matching_bodies)

        friction_dict = {
            "body_indices": body_indices,
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "restitution": restitution,
        }
        return friction_dict

    def _process_center_of_mass_domain_randomization(
        self, domain_randomization: CenterOfMassDomainRandomizationConfig
    ) -> None:
        """
        Process center of mass domain randomization.
        """
        body_indices = get_matching_indices(
            self.robot_config.kinematic_info.body_names,
            domain_randomization.body_names,
            domain_randomization.body_indices,
        )
        num_matching_bodies = len(body_indices)
        com_range = domain_randomization.com_range
        com_range_x = com_range["x"]
        com_range_y = com_range["y"]
        com_range_z = com_range["z"]
        com = torch.rand(self.num_envs, num_matching_bodies, 3)
        com[..., 0] = com[..., 0] * (com_range_x[1] - com_range_x[0]) + com_range_x[0]
        com[..., 1] = com[..., 1] * (com_range_y[1] - com_range_y[0]) + com_range_y[0]
        com[..., 2] = com[..., 2] * (com_range_z[1] - com_range_z[0]) + com_range_z[0]

        com_dict = {"body_indices": body_indices, "com": com}
        return com_dict

    # -------------------------
    # 🎨 Group 6: Rendering & Visualization
    # -------------------------
    def _toggle_camera_target(self) -> None:
        """
        Toggle the camera target between different environments and objects.

        The target cycles through all objects in the scene, with 0 referring to the environment.
        """
        if self.scene_lib.num_objects_per_scene > 0:
            self._camera_target["element"] = (self._camera_target["element"] + 1) % (
                self.scene_lib.num_objects_per_scene + 1
            )
            print("Updated camera target to element", self._camera_target["element"])

        if self._camera_target["element"] == 0:
            self._camera_target["env"] = (
                self._camera_target["env"] + 1
            ) % self.num_envs
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

                    self._recorded_motion = {
                        "gts": [],  # rigid_body_pos (global translations)
                        "grs": [],  # rigid_body_rot (global rotations)
                        "gvs": [],  # rigid_body_vel (global velocities)
                        "gavs": [],  # rigid_body_ang_vel (global angular velocities)
                        "dps": [],  # dof_pos
                        "dvs": [],  # dof_vel
                        "contacts": [],  # rigid_body_contacts
                    }

                    if not os.path.exists(self._curr_user_recording_name):
                        os.makedirs(self._curr_user_recording_name)
                    print(
                        f"Started recording to folder {self._curr_user_recording_name}"
                    )
                else:
                    # Finalize recording and create video
                    from moviepy import ImageSequenceClip

                    image_dir = self._curr_user_recording_name
                    images = sorted(
                        [
                            os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.endswith(".png")
                        ]
                    )

                    clip = ImageSequenceClip(images, fps=30)
                    clip.write_videofile(
                        f"{self._curr_user_recording_name}.mp4",
                        codec="libx264",
                        audio=False,
                        threads=32,
                        preset="veryfast",
                        ffmpeg_params=[
                            "-profile:v",
                            "main",
                            "-level",
                            "4.0",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            "-crf",
                            "23",
                            "-x264-params",
                            "keyint=60:min-keyint=30",
                        ],
                    )
                    self._delete_user_viewer_recordings = True
                    print(f"Video saved to {self._curr_user_recording_name}.mp4")

                    # Save the recorded motion as a .motion file
                    motion_data = build_motion_data(
                        self._recorded_motion,
                        fps=30,  # Video recording FPS
                        num_dof=self._num_dof,
                    )
                    motion_file_path = f"{self._curr_user_recording_name}.motion"
                    torch.save(motion_data, motion_file_path)
                    print(f"Motion saved to {motion_file_path}")
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

                robot_state = self.get_robot_state()
                self._recorded_motion["gts"].append(robot_state.rigid_body_pos)
                self._recorded_motion["grs"].append(robot_state.rigid_body_rot)
                if robot_state.rigid_body_vel is not None:
                    self._recorded_motion["gvs"].append(robot_state.rigid_body_vel)
                if robot_state.rigid_body_ang_vel is not None:
                    self._recorded_motion["gavs"].append(robot_state.rigid_body_ang_vel)
                if robot_state.dof_pos is not None:
                    self._recorded_motion["dps"].append(robot_state.dof_pos)
                if robot_state.dof_vel is not None:
                    self._recorded_motion["dvs"].append(robot_state.dof_vel)
                if robot_state.rigid_body_contacts is not None:
                    self._recorded_motion["contacts"].append(robot_state.rigid_body_contacts)

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

    def _toggle_markers(self):
        self._show_markers = not self._show_markers
        print(f"Markers are now {'visible' if self._show_markers else 'hidden'}")

    def _update_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Update visualization markers for the simulator.

        Converts marker orientations if necessary and delegates to the simulator-specific update.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """

        if not markers_state or len(markers_state) == 0:
            return

        if not self.config.w_last:
            for key in markers_state.keys():
                markers_state[key].orientation = rotations.xyzw_to_wxyz(
                    markers_state[key].orientation
                )
        if not self._show_markers:
            for key in markers_state.keys():
                # Throw it out of view
                markers_state[key].translation = (
                    torch.zeros_like(markers_state[key].translation) - 1000000
                )
        self._update_simulator_markers(markers_state)

    @abstractmethod
    def _update_simulator_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Simulator-specific update of marker states.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """
        raise NotImplementedError

    def is_simulation_running(self) -> bool:
        """
        Check if the simulation is running.
        """
        return self._simulation_running

    def close(self) -> None:
        """
        Close the simulator and perform cleanup operations.
        """
        self._simulation_running = False
