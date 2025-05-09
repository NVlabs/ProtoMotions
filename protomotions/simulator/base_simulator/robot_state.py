from dataclasses import dataclass
from typing import Dict, Optional

import torch
from isaac_utils import rotations

@dataclass
class DataConversion:
    """
    Dataclass containing the conversion mappings between simulator and common state orderings.
    
    Attributes:
        body_convert_to_common (torch.Tensor): Indices to convert simulator rigid-body ordering to common.
        body_convert_to_sim (torch.Tensor): Indices to convert common rigid-body ordering to simulator.
        contact_sensor_convert_to_common (torch.Tensor): Indices to convert simulator contact sensor ordering to common.
        dof_convert_to_common (torch.Tensor): Indices to convert simulator DOF ordering to common.
        dof_convert_to_sim (torch.Tensor): Indices to convert common DOF ordering to simulator.
        sim_w_last (bool): Flag indicating if the simulator uses w_last quaternion ordering.
    """
    body_convert_to_common: torch.Tensor
    body_convert_to_sim: torch.Tensor
    contact_sensor_convert_to_common: torch.Tensor
    dof_convert_to_common: torch.Tensor
    dof_convert_to_sim: torch.Tensor
    sim_w_last: bool

@dataclass
class RobotState:
    """
    Dataclass representing the simulator environment state.
    
    All fields are optional so that you can populate only the available parts.
    
    Attributes:
        root_pos (Optional[torch.Tensor]): Positions of the robot root.
            Expected shape: [num_envs, 3].
        root_rot (Optional[torch.Tensor]): Rotations (quaternions) of the robot root.
            Expected shape: [num_envs, 4].
        root_vel (Optional[torch.Tensor]): Linear velocities of the robot root.
            Expected shape: [num_envs, 3].
        root_ang_vel (Optional[torch.Tensor]): Angular velocities of the robot root.
            Expected shape: [num_envs, 3].
        dof_pos (Optional[torch.Tensor]): Joint positions.
            Expected shape: [num_envs, num_dof].
        dof_vel (Optional[torch.Tensor]): Joint velocities.
            Expected shape: [num_envs, num_dof].
        rigid_body_pos (Optional[torch.Tensor]): Positions of rigid bodies.
            Expected shape: [num_envs, num_bodies, 3].
        rigid_body_rot (Optional[torch.Tensor]): Rotations of rigid bodies (quaternions).
            Expected shape: [num_envs, num_bodies, 4].
        rigid_body_vel (Optional[torch.Tensor]): Linear velocities of rigid bodies.
            Expected shape: [num_envs, num_bodies, 3].
        rigid_body_ang_vel (Optional[torch.Tensor]): Angular velocities of rigid bodies.
            Expected shape: [num_envs, num_bodies, 3].
    """
    root_pos: Optional[torch.Tensor] = None
    root_rot: Optional[torch.Tensor] = None
    root_vel: Optional[torch.Tensor] = None
    root_ang_vel: Optional[torch.Tensor] = None
    dof_pos: Optional[torch.Tensor] = None
    dof_vel: Optional[torch.Tensor] = None
    rigid_body_pos: Optional[torch.Tensor] = None
    rigid_body_rot: Optional[torch.Tensor] = None
    rigid_body_vel: Optional[torch.Tensor] = None
    rigid_body_ang_vel: Optional[torch.Tensor] = None
    key_body_pos: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(data: Dict[str, torch.Tensor]) -> "RobotState":
        """
        Construct an EnvState object from a dictionary.

        Args:
            data (Dict[str, torch.Tensor]): Dictionary containing state information.
                Keys can include "root_pos", "root_rot", "root_vel", "root_ang_vel",
                "dof_pos", "dof_vel", "rigid_body_pos", "rigid_body_rot",
                "rigid_body_vel", and "rigid_body_ang_vel".
        
        Returns:
            EnvState: The constructed environment state.
        """
        return RobotState(
            root_pos=data.get("root_pos", None),
            root_rot=data.get("root_rot", None),
            root_vel=data.get("root_vel", None),
            root_ang_vel=data.get("root_ang_vel", None),
            dof_pos=data.get("dof_pos", None),
            dof_vel=data.get("dof_vel", None),
            rigid_body_pos=data.get("rigid_body_pos", None),
            rigid_body_rot=data.get("rigid_body_rot", None),
            rigid_body_vel=data.get("rigid_body_vel", None),
            rigid_body_ang_vel=data.get("rigid_body_ang_vel", None),
            key_body_pos=data.get("key_body_pos", None),
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert the environment state to a dictionary, including only the fields that are not None.

        Returns:
            Dict[str, torch.Tensor]: A dictionary representation of the environment state.
        """
        data: Dict[str, torch.Tensor] = {}
        if self.root_pos is not None:
            data["root_pos"] = self.root_pos
        if self.root_rot is not None:
            data["root_rot"] = self.root_rot
        if self.root_vel is not None:
            data["root_vel"] = self.root_vel
        if self.root_ang_vel is not None:
            data["root_ang_vel"] = self.root_ang_vel
        if self.dof_pos is not None:
            data["dof_pos"] = self.dof_pos
        if self.dof_vel is not None:
            data["dof_vel"] = self.dof_vel
        if self.rigid_body_pos is not None:
            data["rigid_body_pos"] = self.rigid_body_pos
        if self.rigid_body_rot is not None:
            data["rigid_body_rot"] = self.rigid_body_rot
        if self.rigid_body_vel is not None:
            data["rigid_body_vel"] = self.rigid_body_vel
        if self.rigid_body_ang_vel is not None:
            data["rigid_body_ang_vel"] = self.rigid_body_ang_vel
        if self.key_body_pos is not None:
            data["key_body_pos"] = self.key_body_pos
        return data 

    def convert_to_common(self, conversion: DataConversion) -> "RobotState":
        """
        Convert the simulator state to common ordering using the provided data conversion mapping.
        
        Args:
            conversion (DataConversion): Dataclass containing conversion tensors and sim_w_last flag.
            
        Returns:
            RobotState: A new RobotState with fields converted to common ordering.
        """
        # For rotations: if the simulator does not use w_last ordering, convert from wxyz to xyzw.
        new_root_rot = self.root_rot
        if new_root_rot is not None and not conversion.sim_w_last:
            new_root_rot = rotations.wxyz_to_xyzw(new_root_rot)

        new_dof_pos = self.dof_pos[:, conversion.dof_convert_to_common] if self.dof_pos is not None else None
        new_dof_vel = self.dof_vel[:, conversion.dof_convert_to_common] if self.dof_vel is not None else None

        new_rigid_body_pos = self.rigid_body_pos[:, conversion.body_convert_to_common] if self.rigid_body_pos is not None else None
        new_rigid_body_rot = None
        if self.rigid_body_rot is not None:
            rb_rot = self.rigid_body_rot
            if not conversion.sim_w_last:
                rb_rot = rotations.wxyz_to_xyzw(rb_rot)
            new_rigid_body_rot = rb_rot[:, conversion.body_convert_to_common]
            
        new_rigid_body_vel = self.rigid_body_vel[:, conversion.body_convert_to_common] if self.rigid_body_vel is not None else None
        new_rigid_body_ang_vel = self.rigid_body_ang_vel[:, conversion.body_convert_to_common] if self.rigid_body_ang_vel is not None else None

        return RobotState(
            root_pos=self.root_pos,
            root_rot=new_root_rot,
            root_vel=self.root_vel,
            root_ang_vel=self.root_ang_vel,
            dof_pos=new_dof_pos,
            dof_vel=new_dof_vel,
            rigid_body_pos=new_rigid_body_pos,
            rigid_body_rot=new_rigid_body_rot,
            rigid_body_vel=new_rigid_body_vel,
            rigid_body_ang_vel=new_rigid_body_ang_vel,
            key_body_pos=self.key_body_pos,
        )

    def convert_to_sim(self, conversion: DataConversion) -> "RobotState":
        """
        Convert the common state to simulator ordering using the provided data conversion mapping.
        
        Args:
            conversion (DataConversion): Dataclass containing conversion tensors and sim_w_last flag.
            
        Returns:
            RobotState: A new RobotState with fields converted to simulator ordering.
        """
        new_root_rot = self.root_rot
        if new_root_rot is not None and not conversion.sim_w_last:
            new_root_rot = rotations.xyzw_to_wxyz(new_root_rot)

        new_dof_pos = self.dof_pos[:, conversion.dof_convert_to_sim] if self.dof_pos is not None else None
        new_dof_vel = self.dof_vel[:, conversion.dof_convert_to_sim] if self.dof_vel is not None else None

        new_rigid_body_pos = self.rigid_body_pos[:, conversion.body_convert_to_sim] if self.rigid_body_pos is not None else None
        new_rigid_body_rot = None
        if self.rigid_body_rot is not None:
            rb_rot = self.rigid_body_rot[:, conversion.body_convert_to_sim]
            if not conversion.sim_w_last:
                rb_rot = rotations.xyzw_to_wxyz(rb_rot)
            new_rigid_body_rot = rb_rot
            
        new_rigid_body_vel = self.rigid_body_vel[:, conversion.body_convert_to_sim] if self.rigid_body_vel is not None else None
        new_rigid_body_ang_vel = self.rigid_body_ang_vel[:, conversion.body_convert_to_sim] if self.rigid_body_ang_vel is not None else None

        return RobotState(
            root_pos=self.root_pos,
            root_rot=new_root_rot,
            root_vel=self.root_vel,
            root_ang_vel=self.root_ang_vel,
            dof_pos=new_dof_pos,
            dof_vel=new_dof_vel,
            rigid_body_pos=new_rigid_body_pos,
            rigid_body_rot=new_rigid_body_rot,
            rigid_body_vel=new_rigid_body_vel,
            rigid_body_ang_vel=new_rigid_body_ang_vel,
            key_body_pos=self.key_body_pos,
        )