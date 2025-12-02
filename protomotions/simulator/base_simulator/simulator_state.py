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
"""
Simulator state representations.

Key concepts:
- RobotState: Full state including rigid body positions/rotations from FK.
              Used for reading simulator state and motion data.
- ResetState: Minimal state for resets (root + DOF only).
              Simulators compute FK internally - NEVER provide rigid_body_pos/rot to resetters.
- ObjectState: Object root states.
- RootOnlyState: Convenience class for root-only data.
"""

from dataclasses import dataclass, fields
from typing import Dict, Optional, Callable, TypeVar, Tuple
from abc import ABC, abstractmethod
from enum import Enum

import torch
from protomotions.utils import rotations


@dataclass
class DataConversionMapping:
    """
    Dataclass containing the conversion mappings between simulator and common state orderings.

    Attributes:
        body_convert_to_common (torch.Tensor): Indices to convert simulator rigid-body ordering to common.
        body_convert_to_sim (torch.Tensor): Indices to convert common rigid-body ordering to simulator.
        dof_convert_to_common (torch.Tensor): Indices to convert simulator DOF ordering to common.
        dof_convert_to_sim (torch.Tensor): Indices to convert common DOF ordering to simulator.
        sim_w_last (bool): Flag indicating if the simulator uses w_last quaternion ordering.
    """

    body_convert_to_common: torch.Tensor
    body_convert_to_sim: torch.Tensor
    dof_convert_to_common: torch.Tensor
    dof_convert_to_sim: torch.Tensor
    sim_w_last: bool


class StateConversion(Enum):
    """
    Enum to indicate whether the state is in simulator-specific ordering or common ordering.
    """

    SIMULATOR = "simulator"
    COMMON = "common"

    @classmethod
    def from_str(cls, value: str) -> "StateConversion":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )


T = TypeVar("T", bound="BaseBatchedState")


@dataclass
class BaseBatchedState(ABC):
    """
    Base abstract class for state representations.

    Attributes:
        each field should have expected shape: [batch_size, ...].
            batch_size can either mean num_envs or num_frames.
            when used as a motion, batch_size is num_frames.
            when used as a batched sim state during GPU simulation, batch_size is num_envs.

        fps: Frames per second.
            when used as a motion, fps is required.
            when used as a batched sim state during GPU simulation, fps should be None.
    """

    # indicates the current conversion of the state, so that the state is aware if it's in sim or common.
    state_conversion: Optional[StateConversion]  # Required field

    # when used as Motion with fps
    fps: Optional[float] = None

    @property
    @abstractmethod
    def motion_num_frames(self) -> Optional[torch.Tensor]:
        pass

    @property
    def motion_dt(self) -> Optional[torch.Tensor]:
        assert self.fps is not None, "accessing motion_dt before setting fps"
        return 1.0 / self.fps

    @property
    def motion_length(self) -> Optional[torch.Tensor]:
        assert self.fps is not None, "accessing motion_length before setting fps"
        return (self.motion_num_frames - 1) * self.motion_dt

    def clone(self) -> T:
        """Clone the BaseBatchedState object."""
        # Create a new instance of the same class
        new_state = type(self)(state_conversion=self.state_conversion)

        # Iterate through the fields defined in the dataclass
        for field_info in fields(self):
            key = field_info.name
            if key == "state_conversion":
                continue  # Already set in constructor
            value = getattr(self, key)
            # Copy the value if it's not None
            if value is not None:
                setattr(
                    new_state,
                    key,
                    value.clone() if isinstance(value, torch.Tensor) else value,
                )

        return new_state

    @abstractmethod
    def convert_to_common(self, conversion: DataConversionMapping) -> T:
        pass

    @abstractmethod
    def convert_to_sim(self, conversion: DataConversionMapping) -> T:
        pass

    # Common utility methods
    def __getitem__(self, key):
        """
        Support both field access and batch indexing:
        - state["field_name"] -> returns the tensor for that field
        - state[env_ids] -> returns a new state with indexed batch dimension
        - state[0] -> returns a new state for the first environment
        - state[:5] -> returns a new state for the first 5 environments
        """
        # If key is a string, return the field value (existing behavior)
        if isinstance(key, str):
            if not hasattr(self, key):
                raise KeyError(f"Invalid state field: {key}")
            return getattr(self, key)

        # Otherwise, treat as batch indexing (int, slice, tensor, list, etc.)
        # Create a new instance of the same class
        new_state = type(self)(state_conversion=self.state_conversion)

        # Index into all tensor fields along the batch dimension
        for field_info in fields(self):
            field_name = field_info.name
            if field_name == "state_conversion":
                continue  # Already set in constructor
            value = getattr(self, field_name)

            # Index the value if it's a tensor, otherwise copy as-is
            if value is not None:
                if isinstance(value, torch.Tensor):
                    setattr(new_state, field_name, value[key])
                else:
                    # For non-tensor fields like fps, copy directly
                    setattr(new_state, field_name, value)

        return new_state

    def __setitem__(self, key, value):
        """
        Support both field assignment and batch assignment:
        - state["field_name"] = tensor -> sets the field value
        - state[env_ids] = another_state -> assigns another_state's fields to indexed positions
        """
        # If key is a string, set the field value (existing behavior)
        if isinstance(key, str):
            if not hasattr(self, key):
                raise KeyError(f"Invalid state field: {key}")
            setattr(self, key, value)
            return

        # Otherwise, treat as batch assignment (int, slice, tensor, list, etc.)
        # Value should be a BaseBatchedState instance of the same type
        if not isinstance(value, BaseBatchedState):
            raise TypeError(
                f"When using indexed assignment (state[indices] = ...), the right-hand side "
                f"must be a BaseBatchedState instance, got {type(value)}"
            )

        # Ensure both states are the same type (e.g., both ObjectState or both RobotState)
        if type(value) is not type(self):
            raise TypeError(
                f"Type mismatch in batch assignment: cannot assign {type(value).__name__} "
                f"to {type(self).__name__}. Both states must be the same type."
            )

        # Assign all tensor fields from value to indexed positions in self
        for field_info in fields(self):
            field_name = field_info.name
            if field_name == "state_conversion":
                continue  # Skip metadata field

            self_field = getattr(self, field_name)
            value_field = getattr(value, field_name)

            # Only assign if both fields are tensors
            if self_field is not None and value_field is not None:
                if isinstance(self_field, torch.Tensor) and isinstance(
                    value_field, torch.Tensor
                ):
                    self_field[key] = value_field
                # For non-tensor fields like fps, we don't assign through indexing

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        return iter(f.name for f in fields(self))

    def __len__(self) -> int:
        return len(self.to_dict())

    def _convert_helper(self, conversion_mapping: torch.Tensor, field_name: str):
        if getattr(self, field_name) is not None:
            new_field = getattr(self, field_name)[:, conversion_mapping]
            setattr(self, field_name, new_field)

    def _convert_helper_rot(self, convert_rot: Callable, field_name: str):
        if getattr(self, field_name) is not None:
            new_field = convert_rot(getattr(self, field_name))
            setattr(self, field_name, new_field)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        data: Dict[str, torch.Tensor] = {}
        for field_info in fields(self):
            key = field_info.name
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, torch.Tensor],
        state_conversion: Optional[StateConversion] = None,
    ) -> T:
        # Create an instance with default values
        res = cls(state_conversion=state_conversion)
        # Get valid field names from the dataclass definition
        valid_fields = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in valid_fields:
                setattr(res, k, v)
            else:
                print(
                    f"Warning: Key '{k}' in data is not a valid field in {cls.__name__}."
                )
        return res


@dataclass
class RobotState(BaseBatchedState):
    """
    Dataclass representing the simulator environment state.

    All fields are optional so that you can populate only the available parts.

    Attributes:

        dof_pos (Optional[torch.Tensor]): Joint positions.
            Expected shape: [batch_size, num_dof].
        dof_vel (Optional[torch.Tensor]): Joint velocities.
            Expected shape: [batch_size, num_dof].
        dof_forces (Optional[torch.Tensor]): Joint forces.
            Expected shape: [batch_size, num_dof].
        rigid_body_pos (Optional[torch.Tensor]): Positions of rigid bodies.
            Expected shape: [batch_size, num_bodies, 3].
        rigid_body_rot (Optional[torch.Tensor]): Rotations of rigid bodies (quaternions).
            NOTE: we use xyzw quaternion ordering for the common state shared by all simulators
            Expected shape: [batch_size, num_bodies, 4].
        rigid_body_vel (Optional[torch.Tensor]): Linear velocities of rigid bodies.
            Expected shape: [batch_size, num_bodies, 3].
        rigid_body_ang_vel (Optional[torch.Tensor]): Angular velocities of rigid bodies.
            Expected shape: [batch_size, num_bodies, 3].

        rigid_body_contacts (Optional[torch.Tensor]): Contacts of rigid bodies.
            Expected shape: [batch_size, num_bodies], True if in contact, False otherwise.
        rigid_body_contact_forces (Optional[torch.Tensor]): Contact forces of rigid bodies.
            (usually only obtained from simulator, not from reference motion data)
            Expected shape: [batch_size, num_bodies, 3].

        # redundant fields for caching
        local_rigid_body_rot (Optional[torch.Tensor]): Local rotations of rigid bodies.
            Expected shape: [batch_size, num_bodies, 4].
    """

    dof_pos: Optional[torch.Tensor] = None
    dof_vel: Optional[torch.Tensor] = None
    dof_forces: Optional[torch.Tensor] = None
    rigid_body_pos: Optional[torch.Tensor] = None
    rigid_body_rot: Optional[torch.Tensor] = None
    rigid_body_vel: Optional[torch.Tensor] = None
    rigid_body_ang_vel: Optional[torch.Tensor] = None
    rigid_body_contacts: Optional[torch.Tensor] = None
    rigid_body_contact_forces: Optional[torch.Tensor] = None

    # redundant fields for caching
    local_rigid_body_rot: Optional[torch.Tensor] = None

    @property
    def root_pos(self) -> Optional[torch.Tensor]:
        if self.rigid_body_pos is not None:
            return self.rigid_body_pos[:, 0, :]
        return None

    @property
    def root_rot(self) -> Optional[torch.Tensor]:
        if self.rigid_body_rot is not None:
            return self.rigid_body_rot[:, 0, :]
        return None

    @property
    def root_vel(self) -> Optional[torch.Tensor]:
        if self.rigid_body_vel is not None:
            return self.rigid_body_vel[:, 0, :]
        return None

    @property
    def root_ang_vel(self) -> Optional[torch.Tensor]:
        if self.rigid_body_ang_vel is not None:
            return self.rigid_body_ang_vel[:, 0, :]
        return None

    @property
    def motion_num_frames(self) -> Optional[torch.Tensor]:
        assert self.fps is not None
        return self.rigid_body_pos.shape[0]

    @property
    def num_bodies(self) -> int:
        nb = self.rigid_body_pos.shape[1]
        if self.rigid_body_rot is not None:
            assert self.rigid_body_rot.shape[1] == nb
        if self.rigid_body_vel is not None:
            assert self.rigid_body_vel.shape[1] == nb
        if self.rigid_body_ang_vel is not None:
            assert self.rigid_body_ang_vel.shape[1] == nb
        return nb

    @property
    def num_dofs(self) -> int:
        num_dofs = self.dof_pos.shape[1]
        if self.dof_vel is not None:
            assert self.dof_vel.shape[1] == num_dofs
        return num_dofs

    def flatten_bodies(self, field_name: str) -> torch.Tensor:
        field = getattr(self, field_name)
        if field is None:
            return None
        if field.dim() == 2:
            return field
        return field.view(-1, self.num_bodies * field.shape[-1])

    def unflatten_bodies(self, field_name: str, flattened_field: torch.Tensor):
        field = getattr(self, field_name)
        if field is None:
            return None
        if field.dim() == 2:
            return flattened_field
        v = flattened_field.view(-1, self.num_bodies, field.shape[-1])
        setattr(self, field_name, v)

    def get_shape_mapping(self, flattened: bool = False) -> Dict[str, Tuple[int]]:
        num_dofs = self.num_dofs
        num_bodies = self.num_bodies
        if flattened:
            return {
                "dof_pos": (num_dofs,),
                "dof_vel": (num_dofs,),
                "dof_forces": (num_dofs,),
                "rigid_body_pos": (num_bodies * 3,),
                "rigid_body_rot": (num_bodies * 4,),
                "rigid_body_vel": (num_bodies * 3,),
                "rigid_body_ang_vel": (num_bodies * 3,),
            }
        else:
            return {
                "dof_pos": (num_dofs,),
                "dof_vel": (num_dofs,),
                "dof_forces": (num_dofs,),
                "rigid_body_pos": (num_bodies, 3),
                "rigid_body_rot": (num_bodies, 4),
                "rigid_body_vel": (num_bodies, 3),
                "rigid_body_ang_vel": (num_bodies, 3),
            }

    def merge_fields_from(
        self,
        other: "RobotState",
    ) -> None:
        """Merge data from another RobotState object."""
        if other.state_conversion != self.state_conversion:
            if self.state_conversion == StateConversion.SIMULATOR:
                other = other.convert_to_sim(self.data_conversion)
            else:
                other = other.convert_to_common(self.data_conversion)
        for field in fields(self):
            if field.name == "state_conversion":
                continue
            if getattr(self, field.name) is None:
                setattr(self, field.name, getattr(other, field.name))
            else:
                # already has this field,
                assert (
                    getattr(other, field.name) is None
                ), f"Field {field.name} already has a value, cannot merge from other"

    def _convert_helper_all(
        self, body_conv_map: torch.Tensor, dof_conv_map: torch.Tensor
    ):
        self._convert_helper(dof_conv_map, "dof_pos")
        self._convert_helper(dof_conv_map, "dof_vel")
        self._convert_helper(dof_conv_map, "dof_forces")
        self._convert_helper(body_conv_map, "rigid_body_pos")
        self._convert_helper(body_conv_map, "rigid_body_rot")
        self._convert_helper(body_conv_map, "rigid_body_vel")
        self._convert_helper(body_conv_map, "rigid_body_ang_vel")
        self._convert_helper(body_conv_map, "rigid_body_contacts")
        self._convert_helper(body_conv_map, "rigid_body_contact_forces")

    def convert_to_common(self, conversion: DataConversionMapping) -> "RobotState":
        """
        Convert the simulator state to common ordering using the provided data conversion mapping.

        Args:
            conversion (DataConversionMapping): Dataclass containing conversion tensors and sim_w_last flag.

        Returns:
            RobotState: A new RobotState with fields converted to common ordering.
        """

        if self.state_conversion == StateConversion.SIMULATOR:
            if not conversion.sim_w_last:  # if sim uses wxyz
                self._convert_helper_rot(rotations.wxyz_to_xyzw, "rigid_body_rot")

            self._convert_helper_all(
                conversion.body_convert_to_common, conversion.dof_convert_to_common
            )

        self.state_conversion = StateConversion.COMMON
        return self

    def convert_to_sim(self, conversion: DataConversionMapping) -> "RobotState":
        """
        Convert the common state to simulator ordering using the provided data conversion mapping.

        Args:
            conversion (DataConversionMapping): Dataclass containing conversion tensors and sim_w_last flag.

        Returns:
            RobotState: A new RobotState with fields converted to simulator ordering.
        """

        if self.state_conversion == StateConversion.COMMON:
            if not conversion.sim_w_last:  # if sim uses wxyz
                self._convert_helper_rot(rotations.xyzw_to_wxyz, "rigid_body_rot")

            self._convert_helper_all(
                conversion.body_convert_to_sim, conversion.dof_convert_to_sim
            )

        self.state_conversion = StateConversion.SIMULATOR
        return self

    def translate(self, translation: torch.Tensor):
        """
        Translate the robot state by the given translation vector.
        NOTE: this function is in-place, use clone() to make a copy first if needed.
        """
        need_to_update_body_vel = False

        if translation.dim() == 1:
            translation = translation.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        elif translation.dim() == 2:
            assert translation.shape[0] == self.rigid_body_pos.shape[0]  # (B, 3)
            translation = translation.unsqueeze(1)  # (B, 1, 3)
            if (
                self.rigid_body_vel is not None and self.fps is not None
            ):  # meaning first dim B is time
                need_to_update_body_vel = True
        else:
            raise ValueError(f"Invalid translation shape: {translation.shape}")

        self.rigid_body_pos = self.rigid_body_pos + translation

        if need_to_update_body_vel:
            vel_delta = torch.zeros_like(translation)  # (B, 1, 3)
            vel_delta[:-1] = (translation[1:] - translation[:-1]) / self.motion_dt
            self.rigid_body_vel = self.rigid_body_vel + vel_delta

    def fix_height(self, z_up: bool = True, height_offset: float = 0.0):
        """
        Fix the height of the robot state to be above the ground.
        NOTE: this function is in-place, use clone() to make a copy first if needed.
        """
        axis = 2 if z_up else 1
        body_heights = self.rigid_body_pos[..., axis]
        min_height = body_heights.min()

        min_height_vec = torch.zeros(3).to(self.rigid_body_pos.device)
        min_height_vec[axis] = -min_height + height_offset

        self.translate(min_height_vec)

    def fix_height_per_frame(self, z_up: bool = True, height_offset: float = 0.0):
        """
        Fix the height of the robot state per frame so that each frame is above the ground.
        Only translates frames that are below the ground, leaving frames already above ground untouched.
        NOTE: this function is in-place, use clone() to make a copy first if needed.

        Args:
            z_up (bool): Whether Z axis is up (True) or Y axis is up (False). Default is True.
            height_offset (float): Minimum height above ground. Default is 0.0.
        """
        axis = 2 if z_up else 1
        body_heights = self.rigid_body_pos[..., axis]  # [batch_size, num_bodies]
        min_heights_per_frame = body_heights.min(dim=1)[0]  # [batch_size]

        # Calculate how much each frame needs to be lifted (0 if already above ground)
        lift_amounts = torch.clamp(
            height_offset - min_heights_per_frame, min=-0.02
        )  # [batch_size]

        # Create translation vectors for each frame
        translation_vecs = torch.zeros(lift_amounts.shape[0], 3).to(
            self.rigid_body_pos.device
        )  # [batch_size, 3]
        translation_vecs[:, axis] = lift_amounts

        self.translate(translation_vecs)

    def __post_init__(self):
        if self.rigid_body_pos is not None:
            assert torch.all(
                torch.isfinite(self.rigid_body_pos)
            ), f"rigid_body_pos is not finite: {self.rigid_body_pos}"
        if self.rigid_body_rot is not None:
            assert torch.all(
                torch.isfinite(self.rigid_body_rot)
            ), f"rigid_body_rot is not finite: {self.rigid_body_rot}"
        if self.rigid_body_vel is not None:
            assert torch.all(
                torch.isfinite(self.rigid_body_vel)
            ), f"rigid_body_vel is not finite: {self.rigid_body_vel}"
        if self.rigid_body_ang_vel is not None:
            assert torch.all(
                torch.isfinite(self.rigid_body_ang_vel)
            ), f"rigid_body_ang_vel is not finite: {self.rigid_body_ang_vel}"
        if self.dof_pos is not None:
            assert torch.all(
                torch.isfinite(self.dof_pos)
            ), f"dof_pos is not finite: {self.dof_pos}"
        if self.dof_vel is not None:
            assert torch.all(
                torch.isfinite(self.dof_vel)
            ), f"dof_vel is not finite: {self.dof_vel}"


@dataclass
class RootOnlyState(BaseBatchedState):
    """
    A convenience class when user only wants to store Root State without storing the whole robot state through rigid_body_pos

    Creating this because:
        NOTE: notice that following IsaacGym and IsaacSim, root offset is always ignored. Thus:

        root_pos should always just be a view of rigid_body_pos[:, 0, :],
        since for consistency it's not good to store both preventing the
        potentiality to accidently change one without changing the other,

        But sometimes user might just want to store root info in RobotState
        without storing everything like rigid_body_pos,
        In that case use RootOnlyState() instead.
    """

    root_pos: Optional[torch.Tensor] = None
    root_rot: Optional[torch.Tensor] = None
    root_vel: Optional[torch.Tensor] = None
    root_ang_vel: Optional[torch.Tensor] = None

    @property
    def motion_num_frames(self) -> Optional[torch.Tensor]:
        assert self.fps is not None
        return self.root_pos.shape[0]

    def convert_to_common(self, conversion: DataConversionMapping) -> "RootOnlyState":
        if self.state_conversion == StateConversion.SIMULATOR:
            if not conversion.sim_w_last:  # if sim uses wxyz
                self._convert_helper_rot(rotations.wxyz_to_xyzw, "root_rot")
        self.state_conversion = StateConversion.COMMON
        return self

    def convert_to_sim(self, conversion: DataConversionMapping) -> "RootOnlyState":
        if self.state_conversion == StateConversion.COMMON:
            if not conversion.sim_w_last:  # if sim uses wxyz
                self._convert_helper_rot(rotations.xyzw_to_wxyz, "root_rot")
        self.state_conversion = StateConversion.SIMULATOR
        return self

    def __post_init__(self):
        if self.root_pos is not None:
            assert torch.all(
                torch.isfinite(self.root_pos)
            ), f"root_pos is not finite: {self.root_pos}"
        if self.root_rot is not None:
            assert torch.all(
                torch.isfinite(self.root_rot)
            ), f"root_rot is not finite: {self.root_rot}"
        if self.root_vel is not None:
            assert torch.all(
                torch.isfinite(self.root_vel)
            ), f"root_vel is not finite: {self.root_vel}"
        if self.root_ang_vel is not None:
            assert torch.all(
                torch.isfinite(self.root_ang_vel)
            ), f"root_ang_vel is not finite: {self.root_ang_vel}"


@dataclass
class ResetState(BaseBatchedState):
    """
    Minimal state required for environment resets.

    Contains only root state and DOF state - simulators compute forward kinematics
    internally after setting these values. This is the ONLY state type needed for
    any simulator state setter (_set_simulator_env_state).

    Full RobotState with rigid_body_pos/rot/vel/ang_vel is NEVER needed for resets
    since simulators recompute FK from root + dof positions.

    Attributes:
        root_pos: Root position [batch_size, 3]
        root_rot: Root rotation quaternion [batch_size, 4] (xyzw for common)
        root_vel: Root linear velocity [batch_size, 3]
        root_ang_vel: Root angular velocity [batch_size, 3]
        dof_pos: Joint positions [batch_size, num_dof]
        dof_vel: Joint velocities [batch_size, num_dof]
    """

    root_pos: Optional[torch.Tensor] = None
    root_rot: Optional[torch.Tensor] = None
    root_vel: Optional[torch.Tensor] = None
    root_ang_vel: Optional[torch.Tensor] = None
    dof_pos: Optional[torch.Tensor] = None
    dof_vel: Optional[torch.Tensor] = None

    @property
    def motion_num_frames(self) -> Optional[torch.Tensor]:
        if self.fps is not None:
            return self.root_pos.shape[0]
        return None

    def convert_to_common(self, conversion: DataConversionMapping) -> "ResetState":
        if self.state_conversion == StateConversion.SIMULATOR:
            if not conversion.sim_w_last:
                self._convert_helper_rot(rotations.wxyz_to_xyzw, "root_rot")
            self._convert_helper(conversion.dof_convert_to_common, "dof_pos")
            self._convert_helper(conversion.dof_convert_to_common, "dof_vel")
        self.state_conversion = StateConversion.COMMON
        return self

    def convert_to_sim(self, conversion: DataConversionMapping) -> "ResetState":
        if self.state_conversion == StateConversion.COMMON:
            if not conversion.sim_w_last:
                self._convert_helper_rot(rotations.xyzw_to_wxyz, "root_rot")
            self._convert_helper(conversion.dof_convert_to_sim, "dof_pos")
            self._convert_helper(conversion.dof_convert_to_sim, "dof_vel")
        self.state_conversion = StateConversion.SIMULATOR
        return self

    @classmethod
    def from_robot_state(cls, robot_state: RobotState) -> "ResetState":
        """Extract reset-relevant fields from a full RobotState."""
        return cls(
            root_pos=robot_state.root_pos,
            root_rot=robot_state.root_rot,
            root_vel=robot_state.root_vel,
            root_ang_vel=robot_state.root_ang_vel,
            dof_pos=robot_state.dof_pos,
            dof_vel=robot_state.dof_vel,
            state_conversion=robot_state.state_conversion,
            fps=robot_state.fps,
        )

    def __post_init__(self):
        if self.root_pos is not None:
            assert torch.all(
                torch.isfinite(self.root_pos)
            ), f"root_pos is not finite: {self.root_pos}"
        if self.root_rot is not None:
            assert torch.all(
                torch.isfinite(self.root_rot)
            ), f"root_rot is not finite: {self.root_rot}"
        if self.root_vel is not None:
            assert torch.all(
                torch.isfinite(self.root_vel)
            ), f"root_vel is not finite: {self.root_vel}"
        if self.root_ang_vel is not None:
            assert torch.all(
                torch.isfinite(self.root_ang_vel)
            ), f"root_ang_vel is not finite: {self.root_ang_vel}"
        if self.dof_pos is not None:
            assert torch.all(
                torch.isfinite(self.dof_pos)
            ), f"dof_pos is not finite: {self.dof_pos}"
        if self.dof_vel is not None:
            assert torch.all(
                torch.isfinite(self.dof_vel)
            ), f"dof_vel is not finite: {self.dof_vel}"


@dataclass
class ObjectState(BaseBatchedState):
    """
    Dataclass representing the simulator environment state.

    All fields are optional so that you can populate only the available parts.

    Attributes:
        root_pos (Optional[torch.Tensor]): Positions of object roots.
            Expected shape: [batch_size, num_objects, 3].
        root_rot (Optional[torch.Tensor]): Rotations of object roots (quaternions).
            Expected shape: [batch_size, num_objects, 4].
        root_vel (Optional[torch.Tensor]): Linear velocities of object roots.
            Expected shape: [batch_size, num_objects, 3].
        root_ang_vel (Optional[torch.Tensor]): Angular velocities of object roots.
            Expected shape: [batch_size, num_objects, 3].
    """

    root_pos: Optional[torch.Tensor] = None
    root_rot: Optional[torch.Tensor] = None
    root_vel: Optional[torch.Tensor] = None
    root_ang_vel: Optional[torch.Tensor] = None
    contact_forces: Optional[torch.Tensor] = None

    @property
    def motion_num_frames(self) -> Optional[torch.Tensor]:
        assert self.fps is not None
        return self.root_pos.shape[0]

    def convert_to_common(self, conversion: DataConversionMapping) -> "ObjectState":
        if self.state_conversion == StateConversion.SIMULATOR:
            if not conversion.sim_w_last:  # if sim uses wxyz
                self._convert_helper_rot(rotations.wxyz_to_xyzw, "root_rot")
        self.state_conversion = StateConversion.COMMON
        return self

    def convert_to_sim(self, conversion: DataConversionMapping) -> "ObjectState":
        if self.state_conversion == StateConversion.COMMON:
            if not conversion.sim_w_last:  # if sim uses wxyz
                self._convert_helper_rot(rotations.xyzw_to_wxyz, "root_rot")
        self.state_conversion = StateConversion.SIMULATOR
        return self

    def __post_init__(self):
        if self.root_pos is not None:
            assert torch.all(
                torch.isfinite(self.root_pos)
            ), f"root_pos is not finite: {self.root_pos}"
        if self.root_rot is not None:
            assert torch.all(
                torch.isfinite(self.root_rot)
            ), f"root_rot is not finite: {self.root_rot}"
        if self.root_vel is not None:
            assert torch.all(
                torch.isfinite(self.root_vel)
            ), f"root_vel is not finite: {self.root_vel}"
        if self.root_ang_vel is not None:
            assert torch.all(
                torch.isfinite(self.root_ang_vel)
            ), f"root_ang_vel is not finite: {self.root_ang_vel}"
