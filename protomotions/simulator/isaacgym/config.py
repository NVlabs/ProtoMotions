from dataclasses import dataclass
from protomotions.simulator.base_simulator.config import ConfigBuilder, SimParams, SimulatorConfig

@dataclass
class IsaacGymPhysXParams(ConfigBuilder):
    """PhysX physics engine parameters."""
    num_threads: int = 4
    solver_type: int = 1  # 0: pgs, 1: tgs
    num_position_iterations: int = 4
    num_velocity_iterations: int = 0
    contact_offset: float = 0.02
    rest_offset: float = 0.0
    bounce_threshold_velocity: float = 0.2
    max_depenetration_velocity: float = 10.0
    default_buffer_size_multiplier: float = 10.0


@dataclass
class IsaacGymFlexParams(ConfigBuilder):
    """Flex physics parameters."""
    num_inner_iterations: int = 10
    warm_start: float = 0.25


@dataclass
class IsaacGymSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""
    substeps: int
    physx: IsaacGymPhysXParams = IsaacGymPhysXParams()
    flex: IsaacGymFlexParams = IsaacGymFlexParams()


@dataclass
class IsaacGymSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacGym simulator."""
    sim: IsaacGymSimParams  # Override sim type
    def __post_init__(self):
        self.w_last = True  # IsaacGym uses xyzw quaternions
