# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base robot configuration classes.

This module defines the configuration dataclasses for robot morphology, control
parameters, and simulator-specific settings. All robot configurations (SMPL, G1, etc.)
inherit from RobotConfig and specify their robot-specific parameters.

Key Classes:
    - RobotConfig: Main robot configuration
    - RobotAssetConfig: Robot asset and physics properties
    - RobotControlConfig: PD control parameters
    - ControlType: Enum for control modes

Key Features:
    - Multi-simulator support
    - Configurable PD gains and action scaling
    - Asset file management
    - Body name mappings for observations
    - Kinematic structure extraction
"""

import copy
from dataclasses import dataclass, field, fields, replace
from protomotions.components.pose_lib import ControlInfo, KinematicInfo
from protomotions.assets import DEFAULT_ASSET_ROOT, resolve_asset_root
from protomotions.simulator.base_simulator.config import RobotNoiseConfig
from protomotions.simulator.isaacgym.config import (
    IsaacGymSimParams,
    IsaacGymPhysXParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.mujoco.config import MujocoSimParams

from typing import List, Optional, Dict, Tuple, Union
from enum import Enum
import os
import re
import torch


# Physics steps per control step, per backend, for a humanoid. fps is this times
# control_hz, so these are the one place the backend sub-stepping lives.
_HUMANOID_STEPS = {"isaacgym": 2, "isaaclab": 4, "newton": 4, "mujoco": 20}

# The humanoid default physics for each backend, as importable templates. fps is
# left at the backend's class default here and filled from control_hz by
# humanoid(), so a robot that needs different physics changes one field off a
# template (dataclasses.replace(HUMANOID_ISAACLAB.physx, num_position_iterations=16))
# instead of restating a whole block, and humanoid() builds its defaults from the
# same templates — one source for the tuning.
HUMANOID_ISAACGYM = IsaacGymSimParams(
    physx=IsaacGymPhysXParams(
        num_position_iterations=8,
        num_velocity_iterations=4,
        max_depenetration_velocity=1,
    )
)
HUMANOID_ISAACLAB = IsaacLabSimParams(
    physx=IsaacLabPhysXParams(
        num_position_iterations=8,
        num_velocity_iterations=4,
        max_depenetration_velocity=1,
    )
)
HUMANOID_NEWTON = NewtonSimParams()
HUMANOID_MUJOCO = MujocoSimParams()


@dataclass
class SimulatorParams:
    isaacgym: IsaacGymSimParams = field(default_factory=IsaacGymSimParams)
    isaaclab: IsaacLabSimParams = field(default_factory=IsaacLabSimParams)
    newton: NewtonSimParams = field(default_factory=NewtonSimParams)
    mujoco: MujocoSimParams = field(default_factory=MujocoSimParams)

    @classmethod
    def humanoid(
        cls,
        control_hz: int,
        *,
        isaacgym: Optional[IsaacGymSimParams] = None,
        isaaclab: Optional[IsaacLabSimParams] = None,
        newton: Optional[NewtonSimParams] = None,
        mujoco: Optional[MujocoSimParams] = None,
    ) -> "SimulatorParams":
        """Per-backend physics for a humanoid at ``control_hz`` policy steps/sec.

        Each backend starts from its ``HUMANOID_*`` template, unless the robot
        passes a block for that backend, in which case the passed block is used
        instead. A passed block *replaces* the template wholesale — it is not
        merged field by field — so any physics it leaves unset fall back to the
        backend's class defaults, not the template's tuning. To change one field
        off the tuned humanoid block, pass ``dataclasses.replace(HUMANOID_ISAACLAB,
        fps=400)`` (or ``replace(HUMANOID_ISAACLAB.physx, ...)``), not a fresh
        block. A block whose ``fps`` is ``None`` gets ``control_hz * steps``; an
        explicit fps is kept as-is (and the exact-division rule in RobotConfig
        checks it names control_hz and the block), so a block left without an fps
        gets the right rate instead of a stale constant. Blocks are copied, so
        the shared templates are never mutated.
        """
        templates = {
            "isaacgym": HUMANOID_ISAACGYM,
            "isaaclab": HUMANOID_ISAACLAB,
            "newton": HUMANOID_NEWTON,
            "mujoco": HUMANOID_MUJOCO,
        }
        passed = {
            "isaacgym": isaacgym,
            "isaaclab": isaaclab,
            "newton": newton,
            "mujoco": mujoco,
        }
        blocks = {}
        for backend, steps in _HUMANOID_STEPS.items():
            block = (
                passed[backend] if passed[backend] is not None else templates[backend]
            )
            fps = control_hz * steps if block.fps is None else block.fps
            blocks[backend] = replace(copy.deepcopy(block), fps=fps)
        return cls(**blocks)


@dataclass
class InitState:
    """Configuration for robot initial state."""

    pos: Optional[List[float]]  # [x, y, z] in meters


class ControlType(Enum):
    """Enum defining the available control types for the robot.

    BUILT_IN_PD: Built-in PD controller (e.g. Isaac Gym's internal PD controller)
    VELOCITY: Velocity control using custom PD controller
    TORQUE: Direct torque control
    PROPORTIONAL: Proportional control using custom PD controller
    """

    BUILT_IN_PD = "built_in_pd"
    TORQUE = "torque"
    PROPORTIONAL = "proportional"

    @classmethod
    def from_str(cls, value: str) -> "ControlType":
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


@dataclass
class RobotAssetConfig:
    """Configuration for robot asset properties."""

    # Optional fields with defaults
    # Stored as a portable, checkout-relative path so that configs pickled into
    # resolved_configs*.pt stay valid when a run is moved between machines.
    # Resolved to a concrete directory at read time by resolve_asset_root().
    asset_root: str = DEFAULT_ASSET_ROOT
    self_collisions: bool = True

    # Optional fields
    asset_file_name: str = None
    replace_cylinder_with_capsule: Optional[bool] = None
    thickness: Optional[float] = None
    max_angular_velocity: Optional[float] = None
    max_linear_velocity: Optional[float] = None
    density: Optional[float] = None
    angular_damping: Optional[float] = None
    linear_damping: Optional[float] = None
    disable_gravity: Optional[bool] = None
    fix_base_link: Optional[bool] = None

    def __post_init__(self):
        """Validate that asset_file_name is set."""

        if not self.asset_file_name or not self.asset_file_name.endswith(".xml"):
            raise ValueError(
                f"RobotAssetConfig.asset_file_name ('{self.asset_file_name}') "
                f"must be a valid path to an .xml MJCF file to extract kinematic info. "
                "if you are using URDF, convert it to MJCF first"
            )


@dataclass
class ControlConfig:
    """Configuration for robot control parameters."""

    # Control info overrides for specific joints instead of the values from the MJCF asset
    override_control_info: Optional[Dict[str, ControlInfo]] = None

    # Can be "built_in_pd" or "proportional"/"velocity"/"torque" for Proportional, Velocity, Torque control
    control_type: ControlType = ControlType.BUILT_IN_PD

    # the positional limits used for rewards
    soft_pos_limit: float = 0.9

    # Regex patterns of joints that deliberately keep the MJCF's control values
    # (no override). extract_control_info checks every other joint is claimed by
    # an override pattern; this lists the exceptions.
    asset_default_joints: Tuple[str, ...] = ()

    # The following field is loaded post-init and populated from the MJCF asset
    # Note: Using Field(init=False) to exclude from __init__ signature
    control_info: Dict[str, ControlInfo] = field(init=False)

    def __post_init__(self):
        """Refuse override_control_info values that are not ControlInfo."""
        if self.override_control_info is not None:
            for value in self.override_control_info.values():
                if not isinstance(value, ControlInfo):
                    raise TypeError(
                        f"override_control_info values must be ControlInfo, not "
                        f"{type(value).__name__}"
                    )

    def initialize_control_info(self, asset: RobotAssetConfig):
        """Initialize control info from asset configuration."""
        if not hasattr(self, "control_info"):
            from protomotions.components.pose_lib import extract_control_info

            self.control_info = extract_control_info(
                mjcf_path=os.path.join(
                    resolve_asset_root(asset.asset_root), asset.asset_file_name
                ),
                override_control_info=self.override_control_info,
                asset_default_joints=self.asset_default_joints,
            )


@dataclass
class RobotConfig:
    """Configuration for robot morphology and control parameters.

    Defines all robot-specific parameters including asset files, body names,
    control settings, and physical properties. Each robot (SMPL, G1, etc.)
    should subclass this and provide robot-specific values.

    Key configuration areas:

    - **Asset**: Robot mesh/URDF files and physical properties
    - **Body Names**: Named body parts for tracking and observations
    - **Control**: PD gains, action scales, and control modes
    - **Kinematic Info**: Joint structure extracted from MJCF/URDF
    - **Simulator Params**: Simulator-specific overrides

    Example::

        config = RobotConfig(
            semantic_forward_axis_xy=(1.0, 0.0),
            asset=RobotAssetConfig(asset_file_name="robot.xml"),
            control=ControlConfig(control_type=ControlType.BUILT_IN_PD)
        )
    """

    # Required nested config (subclasses must override with their own default_factory)
    asset: RobotAssetConfig

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {}
    )

    default_root_height: float = 1
    semantic_forward_axis_xy: Optional[Tuple[float, float]] = None
    """Robot-local xy axis that points in the anatomical facing direction.

    This is intentionally 2D, not 3D. ProtoMotions assumes Z-up humanoid
    assets, so heading should only choose the horizontal anatomical forward
    direction. Keeping the field in xy avoids using a forward vector with a
    vertical component that would require re-orienting the robot to stand
    upright before heading logic can be applied.

    Every robot config declares this explicitly: ``__post_init__`` refuses
    ``None``, and a saved robot subtree without the field fails to load, since
    ``from_tree`` rebuilds every dataclass through its constructor. To identify
    the axis for a new robot, run ``python scripts/identify_robot_facing_axis.py
    --robots <robot-name> --view``.
    """
    default_dof_pos: Optional[Union[List[float], torch.Tensor, Dict[str, float]]] = (
        None  # Default joint positions for resets (if None, uses zeros).
        # Can be a Dict[str, float] mapping regex patterns to values (e.g.,
        # {".*_knee_joint": 0.669}). Patterns are resolved against dof_names
        # in __post_init__. Unmatched DOFs default to 0.0.
    )

    # Privileged contact reporting: measured robot rows and "contact against"
    # partner columns. These are simulator truth, not deployable sensor readings.
    # "all" selects every body; None or [] disables contact reporting.
    contact_bodies: Optional[Union[List[str], str]] = None
    # Exact names/globs: "terrain", "robot/*", "object/7", "projectile/*".
    contact_partners: Union[List[str], str] = field(default_factory=lambda: ["terrain"])
    trackable_bodies_subset: Union[List[str], str] = "all"

    non_termination_contact_bodies: Union[List[str], str] = "all"
    init_state: Optional[InitState] = None
    reset_noise: Optional[RobotNoiseConfig] = None
    mimic_small_marker_bodies: Optional[List[str]] = None

    # Anchor body for observations/rewards (None = root body at index 0). This is
    # the only anchor slot: anchor_body_index and resolved_anchor_body_name are
    # read-only properties derived from it, so assigning the name (directly or via
    # update_fields(anchor_body_name=...)) re-derives the index on the next read.
    anchor_body_name: Optional[str] = None

    # Physical IMU placement is independent of the observation anchor.
    imu_body_name: Optional[str] = None
    imu_mount_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Unit xyzw rotation from sensor axes to the carrying body's axes."""

    # Optional with default
    contact_pairs_multiplier: int = 16
    control: ControlConfig = field(default_factory=ControlConfig)

    # Policy control rate in Hz (steps per second). Subclasses declare it
    # (e.g. control_hz=50). Dataclass field ordering forces a default, so None
    # means "not declared" and __post_init__ raises. control_dt is derived.
    control_hz: int = field(default=None)

    # The following fields are loaded post-init and populated from the MJCF asset
    # Note: Using Field(init=False) to exclude from __init__ signature
    kinematic_info: KinematicInfo = field(init=False)
    number_of_actions: int = field(init=False)

    # Per-backend simulation parameters. None resolves to the humanoid default
    # built from control_hz (SimulatorParams.humanoid). A block a robot passes
    # for a backend replaces that backend's template wholesale (fields it omits
    # are class defaults, not the humanoid tuning), so to change one field off the
    # tuning use dataclasses.replace(HUMANOID_<BACKEND>, ...), not a fresh block.
    simulation_params: Optional[SimulatorParams] = None

    @property
    def control_dt(self) -> float:
        """Seconds per policy control step (1 / control_hz)."""
        return 1.0 / self.control_hz

    def _check_sim_fps_divisible(self) -> None:
        """Every backend's physics fps must be a positive whole multiple of control_hz."""
        for backend in ("isaacgym", "isaaclab", "newton", "mujoco"):
            fps = getattr(self.simulation_params, backend).fps
            if fps is None:
                raise ValueError(
                    f"{type(self).__name__}: simulation_params.{backend}.fps is not "
                    f"set; declare it or let SimulatorParams.humanoid derive it from "
                    f"control_hz"
                )
            if fps < 1:
                raise ValueError(
                    f"{type(self).__name__}: simulation_params.{backend}.fps={fps} "
                    f"must be a positive integer"
                )
            if fps % self.control_hz != 0:
                raise ValueError(
                    f"{type(self).__name__}: simulation_params.{backend}.fps={fps} "
                    f"is not a multiple of control_hz={self.control_hz}"
                )

    def __post_init__(self):
        """Compute derived fields after initialization."""

        if self.control_hz is None:
            raise ValueError(
                "RobotConfig.control_hz must be declared: policy steps per second"
            )
        # Remember whether the physics was derived (no block declared) so an
        # override of control_hz re-derives it rather than keeping a stale rate.
        self._simulation_params_derived = self.simulation_params is None
        if self.simulation_params is None:
            self.simulation_params = SimulatorParams.humanoid(self.control_hz)
        self._check_sim_fps_divisible()

        self._normalize_semantic_forward_axis_xy()

        from protomotions.components.pose_lib import extract_kinematic_info

        self.kinematic_info = extract_kinematic_info(
            os.path.join(
                resolve_asset_root(self.asset.asset_root), self.asset.asset_file_name
            )
        )

        self._anchor_cache = (
            self.anchor_body_name,
            self._anchor_index_for(self.anchor_body_name),
        )

        # Initialize control info in the control config
        self.control.initialize_control_info(self.asset)

        self.number_of_actions = self.kinematic_info.num_dofs

        # Initialize default_dof_pos: use provided values or zeros
        if self.default_dof_pos is None:
            # Default to zero joint positions
            self.default_dof_pos = torch.zeros(
                self.kinematic_info.num_dofs, dtype=torch.float32
            )
        elif isinstance(self.default_dof_pos, dict):
            # Resolve regex pattern dict to tensor
            dof_pos = torch.zeros(self.kinematic_info.num_dofs, dtype=torch.float32)
            for pattern, value in self.default_dof_pos.items():
                for i, name in enumerate(self.kinematic_info.dof_names):
                    if re.fullmatch(pattern, name):
                        dof_pos[i] = value
            self.default_dof_pos = dof_pos
        else:
            # Convert to tensor if needed and validate
            if not isinstance(self.default_dof_pos, torch.Tensor):
                self.default_dof_pos = torch.tensor(
                    self.default_dof_pos, dtype=torch.float32
                )
            assert (
                len(self.default_dof_pos) == self.kinematic_info.num_dofs
            ), f"default_dof_pos length {len(self.default_dof_pos)} != num_dofs {self.kinematic_info.num_dofs}"

        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self.contact_bodies = abstract_names_to_body_names(self.contact_bodies, self)
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )

        required_abstract_names = [
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
            "head_body_name",
            "torso_body_name",
        ]
        for name in required_abstract_names:
            assert (
                name in self.common_naming_to_robot_body_names.keys()
            ), f"RobotConfig.common_naming_to_robot_body_names must contain {name}"

    def _anchor_index_for(self, name: Optional[str]) -> int:
        """The body index for an anchor name (None means the root body at 0)."""
        if name is None:
            return 0
        body_names = self.kinematic_info.body_names
        if name not in body_names:
            raise ValueError(
                f"anchor_body_name {name!r} not found in body_names: {body_names}"
            )
        return body_names.index(name)

    @property
    def anchor_body_index(self) -> int:
        """Body index observations/rewards anchor on (0 = root), derived from the name.

        There is one anchor slot, anchor_body_name. Assigning it re-derives the
        index on the next read, so the two cannot drift apart; there is no stored
        index to set (update_fields and the override funnel refuse it). The cache
        is rebuilt when it is absent too, so a config unpickled from a
        pre-property bundle (which has no cache) re-derives from the name rather
        than raising AttributeError on the first read.
        """
        cache = getattr(self, "_anchor_cache", None)
        if cache is None or cache[0] != self.anchor_body_name:
            self._anchor_cache = (
                self.anchor_body_name,
                self._anchor_index_for(self.anchor_body_name),
            )
        return self._anchor_cache[1]

    @property
    def resolved_anchor_body_name(self) -> str:
        """The anchored body's name, resolved through anchor_body_index."""
        return self.kinematic_info.body_names[self.anchor_body_index]

    def body_indices(self, *names: str) -> Tuple[int, ...]:
        """Indices for body names, 'root', 'all' or semantic groups, in declaration order.

        Each name is 'root', 'all', a key of common_naming_to_robot_body_names, or
        a body name; the expansion is abstract_names_to_body_names', with an
        unknown name and a body selected twice refused (both of which that
        function silently allows).
        """
        body_names = self.kinematic_info.body_names
        indices: List[int] = []
        seen = set()
        for name in names:
            for body in abstract_names_to_body_names(name, self):
                if body not in body_names:
                    raise ValueError(
                        f"{name!r} is neither a body of {type(self).__name__} nor "
                        f"a semantic group; groups: "
                        f"{sorted(self.common_naming_to_robot_body_names)}"
                    )
                if body in seen:
                    raise ValueError(f"body {body!r} selected twice")
                seen.add(body)
                indices.append(body_names.index(body))
        return tuple(indices)

    def bodies_matching(self, pattern: str) -> Tuple[int, ...]:
        """Indices of bodies whose name fully matches ``pattern`` (regex), in body order."""
        body_names = self.kinematic_info.body_names
        indices = tuple(
            i for i, name in enumerate(body_names) if re.fullmatch(pattern, name)
        )
        if not indices:
            raise ValueError(
                f"pattern {pattern!r} matches no body of {type(self).__name__}"
            )
        return indices

    def _normalize_semantic_forward_axis_xy(self) -> None:
        """Validate and normalize the declared anatomical forward axis."""
        if self.semantic_forward_axis_xy is None:
            raise ValueError(
                "RobotConfig.semantic_forward_axis_xy must be declared explicitly "
                "as the robot-local xy axis that points in the anatomical "
                "forward direction, e.g. (1.0, 0.0) or (0.0, -1.0). "
                "The axis is 2D because ProtoMotions assumes Z-up robots and "
                "uses this field only for horizontal heading, not upright pose "
                "re-orientation. Run `python scripts/identify_robot_facing_axis.py "
                "--robots <robot-name> --view` to identify and visualize it."
            )
        axis = torch.as_tensor(self.semantic_forward_axis_xy, dtype=torch.float32)
        if axis.shape != (2,):
            raise ValueError(
                "RobotConfig.semantic_forward_axis_xy must be a non-zero 2D "
                "axis. The field is xy-only because ProtoMotions assumes Z-up "
                "robots and separates horizontal heading from standing upright."
            )
        norm = axis.norm()
        if norm <= 1e-6:
            raise ValueError(
                "RobotConfig.semantic_forward_axis_xy must be a non-zero 2D "
                "axis. The field is xy-only because ProtoMotions assumes Z-up "
                "robots and separates horizontal heading from standing upright."
            )
        axis = axis / norm
        self.semantic_forward_axis_xy = (float(axis[0]), float(axis[1]))

    def update_fields(self, **kwargs):
        """Update robot config fields and reprocess derived fields.

        This is useful when you need to modify fields after initialization
        without manually calling __post_init__.

        Args:
            **kwargs: Fields to update on the robot config
        """
        for key in kwargs:
            if not hasattr(self, key):
                raise ValueError(f"RobotConfig has no field '{key}'")
            if _is_derived_field(type(self), key):
                source = (
                    "anchor_body_name" if key == "anchor_body_index" else "the asset"
                )
                raise ValueError(
                    f"RobotConfig.{key} is derived; set the field it is derived "
                    f"from ({source})"
                )
        for key, value in kwargs.items():
            setattr(self, key, value)

        if "semantic_forward_axis_xy" in kwargs:
            self._normalize_semantic_forward_axis_xy()

        # anchor_body_index re-derives lazily on read via its property, so an
        # updated anchor_body_name needs no explicit recompute here.

        if "simulation_params" in kwargs:
            self._simulation_params_derived = kwargs["simulation_params"] is None
        if "control_hz" in kwargs or "simulation_params" in kwargs:
            # A robot whose physics was derived re-derives it at the new rate; an
            # explicit block is kept and the divisibility check names it.
            if self.simulation_params is None or (
                "control_hz" in kwargs and self._simulation_params_derived
            ):
                self.simulation_params = SimulatorParams.humanoid(self.control_hz)
            self._check_sim_fps_divisible()

        # Reprocess fields that depend on the updated values
        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self.contact_bodies = abstract_names_to_body_names(self.contact_bodies, self)
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )


def abstract_names_to_body_names(
    names: Union[List[str], str], robot_config: RobotConfig
):
    if names is None:
        return None

    if names == "all":
        return robot_config.kinematic_info.body_names

    if isinstance(names, list):
        parsed_names = []
        for name in names:
            parsed_names.extend(abstract_names_to_body_names(name, robot_config))
        return parsed_names

    if names == "root":
        return [robot_config.kinematic_info.body_names[0]]
    elif names in robot_config.common_naming_to_robot_body_names.keys():
        return robot_config.common_naming_to_robot_body_names[names]
    else:
        return [names]


def _is_derived_field(cls, key: str) -> bool:
    """True if ``key`` is a derived member: a property or an ``init=False`` field."""
    if isinstance(getattr(cls, key, None), property):
        return True
    return any(f.name == key and not f.init for f in fields(cls))
