from dataclasses import dataclass
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig
from protomotions.simulator.isaacgym.config import IsaacGymPhysXParams


@dataclass
class IsaacLabPhysXParams(IsaacGymPhysXParams):
    """PhysX physics engine parameters."""
    gpu_found_lost_pairs_capacity: int = 2**21
    gpu_max_rigid_contact_count: int = 2**23
    gpu_found_lost_aggregate_pairs_capacity: int = 2**25


@dataclass
class IsaacLabSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""
    physx: IsaacLabPhysXParams = IsaacLabPhysXParams()


@dataclass
class IsaacLabSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacLab simulator."""
    sim: IsaacLabSimParams  # Override sim type
    def __post_init__(self):
        self.w_last = False  # IsaacLab uses wxyz quaternions
