from dataclasses import dataclass, field
from typing import Literal, Tuple, List, Dict, Optional, Any, Type, TypeVar
import torch
from enum import Enum
            

T = TypeVar('T')

@dataclass
class ConfigBuilder:
    """Mixin class providing dictionary conversion functionality."""
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create an instance from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values.
            
        Returns:
            Instance of the class with values from the dictionary.
        """
        field_types = {field.name: field.type for field in cls.__dataclass_fields__.values()}
        processed_dict = {}
        
        for key, value in config_dict.items():
            if value is None:
                processed_dict[key] = None
            elif key in field_types:
                field_type = field_types[key]
                # Handle ControlType enum
                if field_type == ControlType and isinstance(value, str):
                    processed_dict[key] = ControlType.from_str(value)
                # Handle nested dataclass fields
                elif hasattr(field_type, 'from_dict'):
                    processed_dict[key] = field_type.from_dict(value)
                # Handle Optional types
                elif hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                    inner_type = field_type.__args__[0]
                    if hasattr(inner_type, 'from_dict'):
                        processed_dict[key] = inner_type.from_dict(value) if value is not None else None
                    else:
                        processed_dict[key] = value
                else:
                    processed_dict[key] = value
            else:
                print(f"Note: '{key}' is not a field in {cls.__name__}, it will be ignored")
        
        return cls(**processed_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary, handling nested dataclasses.
        
        Returns:
            Dictionary representation of the config.
        """
        result = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                result[field] = None
            elif hasattr(value, 'to_dict'):  # Handle nested dataclasses
                result[field] = value.to_dict()
            elif isinstance(value, Enum):  # Handle enums
                result[field] = value.value
            elif isinstance(value, (list, tuple)):  # Handle lists/tuples of dataclasses
                if value and hasattr(value[0], 'to_dict'):
                    result[field] = [item.to_dict() for item in value]
                else:
                    result[field] = value
            else:
                result[field] = value
        return result

    def __getitem__(self, key: str) -> Any:
        """Make configs behave like dicts for compatibility with external libraries."""
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for compatibility with external libraries."""
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility with external libraries."""
        return self.to_dict().get(key, default)


class ControlType(Enum):
    """Enum defining the available control types for the robot.
    
    BUILT_IN_PD: Built-in PD controller (e.g. Isaac Gym's internal PD controller)
    VELOCITY: Velocity control using custom PD controller
    TORQUE: Direct torque control
    PROPORTIONAL: Proportional control using custom PD controller
    """
    BUILT_IN_PD = "built_in_pd"
    VELOCITY = "velocity"
    TORQUE = "torque"
    PROPORTIONAL = "proportional"

    @classmethod
    def from_str(cls, value: str) -> 'ControlType':
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls 
                if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(f"'{value}' is not a valid {cls.__name__}. "
                           f"Valid values are: {[e.value for e in cls]}")


@dataclass
class MarkerConfig(ConfigBuilder):
    """Configuration for a single marker instance."""
    size: Literal["tiny", "small", "regular"]


@dataclass
class VisualizationMarker(ConfigBuilder):
    """Configuration for a group of visualization markers."""
    type: Literal["sphere", "arrow"]
    color: Tuple[float, float, float]  # RGB values
    markers: List[MarkerConfig]


@dataclass
class MarkerState(ConfigBuilder):
    """Represents the state of a marker in 3D space."""
    translation: torch.Tensor  # Translation vector (position)
    orientation: torch.Tensor  # Orientation quaternion


@dataclass
class RobotAssetConfig(ConfigBuilder):
    """Configuration for robot asset properties."""
    # Required fields
    robot_type: str
    
    collapse_fixed_joints: bool
    
    # Optional fields with defaults
    asset_root: str = "protomotions/data/assets"
    self_collisions: bool = True
    default_dof_drive_mode: int = 1
    
    # Optional fields
    asset_file_name: str = None
    usd_asset_file_name: str = None
    replace_cylinder_with_capsule: Optional[bool] = None
    flip_visual_attachments: Optional[bool] = None
    armature: Optional[float] = None
    thickness: Optional[float] = None
    max_angular_velocity: Optional[float] = None
    max_linear_velocity: Optional[float] = None
    density: Optional[float] = None
    angular_damping: Optional[float] = None
    linear_damping: Optional[float] = None
    disable_gravity: Optional[bool] = None
    fix_base_link: Optional[bool] = None
    filter_ints: Optional[List[int]] = None
    def __post_init__(self):
        """Validate that either asset_file_name or usd_asset_file_name is set."""
        if not self.asset_file_name and not self.usd_asset_file_name:
            raise ValueError("Either asset_file_name or usd_asset_file_name must be specified, depending on the simulator you are using")


@dataclass
class InitState(ConfigBuilder):
    """Configuration for robot initial state."""
    pos: Optional[List[float]] = None  # [x, y, z] in meters
    default_joint_angles: Optional[Dict[str, float]] = None  # joint name -> angle in radians


@dataclass
class ControlConfig(ConfigBuilder):
    """Configuration for robot control parameters."""
    # Required field with default
    control_type: ControlType = ControlType.BUILT_IN_PD
    
    # Optional fields with defaults
    action_scale: float = 1.0
    clamp_actions: float = 1.0
    
    # target angles [rad] when action = 0
    use_biased_controller: bool = False
    # maps [-1, 1] to [dof_min, dof_max]
    map_actions_to_pd_range: bool = True
    
    # Optional fields
    stiffness: Optional[Dict[str, float]] = None  # joint type -> stiffness in N*m/rad
    damping: Optional[Dict[str, float]] = None  # joint type -> damping in N*m*s/rad


@dataclass
class RobotConfig(ConfigBuilder):
    """Configuration for robot parameters."""
    # Required fields (marked with ??? in base.yaml)
    body_names: List[str]  # DFS ordering
    dof_names: List[str]
    dof_body_ids: List[int]
    dof_obs_size: int
    number_of_actions: int
    self_obs_max_coords_size: int
    left_foot_name: str
    right_foot_name: str
    head_body_name: str
    
    # Required nested config
    asset: RobotAssetConfig
    
    # Optional fields with computed defaults
    num_bodies: Optional[int] = None  # Computed from len(body_names)
    num_key_bodies: Optional[int] = None  # Computed from len(key_bodies)
    contact_bodies: Optional[List[str]] = None  # Defaults to body_names
    trackable_bodies_subset: Optional[List[str]] = None  # Defaults to body_names
    self_obs_size: Optional[int] = None  # Defaults to self_obs_max_coords_size
    
    # Optional fields
    dof_effort_limits: Optional[List[float]] = None
    dof_vel_limits: Optional[List[float]] = None
    dof_armatures: Optional[List[float]] = None
    dof_joint_frictions: Optional[List[float]] = None

    key_bodies: Optional[List[str]] = None
    non_termination_contact_bodies: Optional[List[str]] = None
    init_state: Optional[InitState] = None
    mimic_small_marker_bodies: Optional[List[str]] = None
    
    # Optional with default
    contact_pairs_multiplier: int = 16
    control: ControlConfig = field(default_factory=ControlConfig)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if self.num_bodies is None:
            self.num_bodies = len(self.body_names)
        
        if self.contact_bodies is None:
            self.contact_bodies = self.body_names.copy()
            
        if self.trackable_bodies_subset is None:
            self.trackable_bodies_subset = self.body_names.copy()
            
        if self.self_obs_size is None:
            self.self_obs_size = self.self_obs_max_coords_size
            
        if self.num_key_bodies is None:
            self.num_key_bodies = len(self.key_bodies)


@dataclass
class PlaneConfig(ConfigBuilder):
    """Configuration for the simulation plane properties."""
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


@dataclass
class SimParams(ConfigBuilder):
    """Configuration for core simulation parameters."""
    fps: int
    decimation: int


@dataclass
class SimulatorConfig(ConfigBuilder):
    """Main configuration class for the simulator."""
    w_last: bool  # quaternion format (xyzw vs wxyz)
    headless: bool
    robot: RobotConfig
    num_envs: int
    sim: SimParams
    experiment_name: str
    plane: PlaneConfig = PlaneConfig()
    camera: Optional[Any] = None
    record_viewer: bool = False
    viewer_record_dir: str = "output/recordings/viewer"


@dataclass
class SimBodyOrdering(ConfigBuilder):
    """Configuration for the ordering of bodies in the simulation."""
    body_names: List[str]
    dof_names: List[str]
    contact_sensor_body_names: List[str]
