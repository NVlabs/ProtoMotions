# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for the base environment.

This module defines the configuration dataclasses for environment settings,
rewards, terminations, and observation components.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from protomotions.envs.obs.scene_obs import SceneObsConfig
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.control.base import ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.mdp_component import MdpComponent


@dataclass
class EnvConfig:
    """Main environment configuration."""

    max_episode_length: int = field(
        default=300,
        metadata={"help": "Maximum steps per episode before automatic reset.", "min": 1}
    )
    reset_grace_period: int = field(
        default=5,
        metadata={"help": "Steps after reset where grace period applies (for zeroing unreliable rewards).", "min": 0}
    )
    num_state_history_steps: int = field(
        default=0,
        metadata={"help": "Number of historical state steps to store. 0 = no history.", "min": 0}
    )

    _target_: str = "protomotions.envs.base_env.env.BaseEnv"

    scene_obs: SceneObsConfig = field(
        default_factory=SceneObsConfig,
        metadata={"help": "Scene observation configuration."}
    )

    motion_manager: MotionManagerConfig = field(
        default_factory=MotionManagerConfig,
        metadata={"help": "Motion manager for reference motion handling."}
    )

    ref_respawn_offset: float = field(
        default=0.05,
        metadata={"help": "Height offset for respawning relative to reference.", "min": 0.0}
    )
    ref_object_respawn_offset: float = field(
        default=0.0,
        metadata={"help": "Height offset for object respawning."}
    )
    ref_contact_smooth_window: int = field(
        default=0,
        metadata={"help": "Window length for smoothing contact labels. 0 = no smoothing.", "min": 0}
    )
    skip_correct_terrain_height_on_flat: bool = field(
        default=True,
        metadata={"help": "Skip terrain height correction when terrain is flat (optimization)."}
    )

    show_terrain_markers: bool = field(
        default=False,
        metadata={"help": "Show terrain markers during evaluation. Uses significant memory in IsaacGym."}
    )
    save_dir: str = field(
        default="",
        metadata={"help": "Directory for saving evaluation outputs."}
    )

    reward_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of named reward components. Each is a MdpComponent."}
    )
    
    control_components: Dict[str, ControlComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateful task/control managers."}
    )
    
    termination_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of termination functions. Each is a MdpComponent."}
    )
    
    observation_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of observation functions. Each is a MdpComponent."}
    )

    action_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Single action processing config dict with 'fn' key. Use make_pd_action_config() helper."}
    )

    # Odometer corruption parameters.  Used by corrupted_xy_offset_factory to
    # simulate per-session calibration error in the G1 leg-kinematics odometer.
    # The env samples odom_scale and a yaw_bias angle once per episode at reset.
    # Identity defaults mean no corruption when the factory is not used.
    odom_scale_range: tuple = field(
        default=(1.0, 1.0),
        metadata={
            "help": (
                "Per-episode odometer scale factor range (lo, hi) drawn from Uniform. "
                "Default (1.0, 1.0) = no scale corruption. "
                "Recommended for odom experiments: (0.7, 1.3)."
            )
        },
    )
    odom_yaw_range_deg: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-episode odometer yaw bias magnitude in degrees. "
                "The actual bias is drawn from Uniform(-deg, +deg). "
                "Default 0.0 = no yaw corruption. "
                "Recommended for odom experiments: 6.0."
            )
        },
    )
