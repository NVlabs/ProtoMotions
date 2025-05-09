from dataclasses import dataclass, field
from typing import List, Optional
from protomotions.simulator.base_simulator.config import ConfigBuilder


@dataclass
class TerrainConfig(ConfigBuilder):
    """Configuration for terrain generation.
    
    This class contains all parameters needed to configure terrain generation,
    including dimensions, scales, and various terrain-specific parameters.
    """
    # Basic terrain parameters
    map_length: float = 20.0  # Length of each terrain tile [meters]
    map_width: float = 20.0  # Width of each terrain tile [meters]
    num_levels: int = 10  # Number of terrain levels for curriculum
    num_terrains: int = 1  # Number of terrains per level
    horizontal_scale: float = 0.1  # Horizontal distance between terrain points [meters]
    vertical_scale: float = 0.005  # Vertical distance between terrain points [meters]
    border_size: float = 40.0  # Size of the border around the terrain [meters]
    
    # Scene and spacing parameters
    spacing_between_scenes: float = 5.0  # Minimum spacing between scenes [meters]
    minimal_humanoid_spacing: float = 1.0  # Minimum spacing between humanoids [meters]
    
    # Terrain generation parameters
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]
    terrain_proportions: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    terrain_composition: str = "curriculum"  # Type of terrain composition (e.g., "curriculum", "random")
    slope_threshold: float = 0.9  # Maximum slope threshold for walkable terrain
    
    # Height sampling parameters
    sample_width: float = 1.0  # Width of the height sample grid [meters]
    num_samples_per_axis: int = 16  # Number of height samples per axis
    terrain_obs_num_samples: int = None
    
    # Terrain loading/saving
    load_terrain: bool = False  # Whether to load a pre-generated terrain
    save_terrain: bool = False  # Whether to save the generated terrain
    terrain_path: Optional[str] = None  # Path to load/save terrain
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        assert self.map_length > 0, "Map length must be positive"
        assert self.map_width > 0, "Map width must be positive"
        assert self.num_levels > 0, "Number of levels must be positive"
        assert self.num_terrains > 0, "Number of terrains must be positive"
        assert self.horizontal_scale > 0, "Horizontal scale must be positive"
        assert self.vertical_scale > 0, "Vertical scale must be positive"
        assert self.border_size >= 0, "Border size must be non-negative"
        assert self.spacing_between_scenes > 0, "Spacing between scenes must be positive"
        assert self.minimal_humanoid_spacing >= 0, "Minimal humanoid spacing must be positive"
        assert sum(self.terrain_proportions) <= 1.0, "Sum of terrain proportions must be less than or equal to 1.0"
        assert self.slope_threshold > 0, "Slope threshold must be positive"
        assert self.sample_width > 0, "Sample width must be positive"
        assert self.num_samples_per_axis > 0, "Number of samples per axis must be positive"
        
        if self.load_terrain:
            assert self.terrain_path is not None, "Terrain path must be specified when loading terrain"
        if self.save_terrain:
            assert self.terrain_path is not None, "Terrain path must be specified when saving terrain" 
            
        if self.terrain_obs_num_samples is None:
            self.terrain_obs_num_samples = self.num_samples_per_axis ** 2
