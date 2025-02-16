from dataclasses import dataclass
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class GenesisSimParams(SimParams):
    """Genesis-specific simulation parameters."""
    substeps: int


@dataclass
class GenesisSimulatorConfig(SimulatorConfig):
    """Configuration specific to Genesis simulator."""
    sim: GenesisSimParams  # Override sim type
    def __post_init__(self):
        self.w_last = False  # Genesis uses wxyz quaternions
