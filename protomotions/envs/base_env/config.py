# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for the base environment.

This module defines the configuration dataclasses for environment settings,
rewards, terminations, and observation components.
"""

from typing import Optional, Dict
from dataclasses import dataclass, field

from protomotions.envs.action.config import ActionConfig
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.control.base import ControlComponentConfig
from protomotions.envs.sensors.config import SensorConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.rule import TerminationRule


@dataclass
class RecoveryResetConfig:
    """Reset-time sampling from a cache of physically settled fall poses."""

    recovery_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability that a reset uses a fall pose.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    fall_sim_steps: int = field(
        default=150,
        metadata={
            "help": "Physics steps used to generate settled fall poses.",
            "min": 0,
        },
    )


# The episode length inference runs with: long enough that no clip is cut short
# by the step budget. A plain int, not a sentinel, so the field type never changes.
INFERENCE_EPISODE_LENGTH = 1_000_000


@dataclass
class EnvConfig:
    """Main environment configuration."""

    max_episode_length: int = field(
        default=300,
        metadata={
            "help": "Maximum steps per episode before automatic reset.",
            "min": 1,
        },
    )
    reset_grace_period: int = field(
        default=5,
        metadata={
            "help": "Steps after reset where grace period applies (for zeroing unreliable rewards).",
            "min": 0,
        },
    )
    _target_: str = "protomotions.envs.base_env.env.BaseEnv"

    motion_manager: MotionManagerConfig = field(
        default_factory=MotionManagerConfig,
        metadata={"help": "Motion manager for reference motion handling."},
    )

    ref_respawn_offset: float = field(
        default=0.05,
        metadata={
            "help": "Height offset for respawning relative to reference.",
            "min": 0.0,
        },
    )
    ref_object_respawn_offset: float = field(
        default=0.0, metadata={"help": "Height offset for object respawning."}
    )
    ref_contact_smooth_window: int = field(
        default=0,
        metadata={
            "help": "Window length for smoothing contact labels. 0 = no smoothing.",
            "min": 0,
        },
    )
    recovery_reset: RecoveryResetConfig = field(
        default_factory=RecoveryResetConfig,
        metadata={"help": "Optional reset sampling from generated fall poses."},
    )
    skip_correct_terrain_height_on_flat: bool = field(
        default=True,
        metadata={
            "help": "Skip terrain height correction when terrain is flat (optimization)."
        },
    )

    show_terrain_markers: bool = field(
        default=False,
        metadata={
            "help": "Show terrain markers during evaluation. Uses significant memory in IsaacGym."
        },
    )
    save_dir: str = field(
        default="", metadata={"help": "Directory for saving evaluation outputs."}
    )

    reward_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={
            "help": "Dictionary of named reward components. Each is a MdpComponent."
        },
    )

    control_components: Dict[str, ControlComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateful task/control managers."},
    )

    termination_components: Dict[str, "TerminationRule"] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Dictionary of termination entries. Each is a TerminationRule over "
                "a kernel bound with from_kernel: no threshold for a boolean "
                "kernel, a float cut for a scalar one, truncation=True to end "
                "the episode without failing."
            )
        },
    )

    observation_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={
            "help": "Dictionary of observation functions. Each is a MdpComponent."
        },
    )

    action_config: Optional[ActionConfig] = field(
        default=None,
        metadata={
            "help": (
                "The action function bound to its parameters (an ActionConfig; "
                "None passes the policy action through). Use the make_*_action_config() "
                "helpers (make_pd_action_config, ...)."
            )
        },
    )

    sensors: Optional[SensorConfig] = None
    """Declared physical sensors; None exposes privileged observations only."""

    def apply_inference_defaults(self) -> None:
        """The environment settings every inference view starts from.

        Run a clip to its end rather than to the training step budget. The
        sensor switch to inference noise is applied by ``inference_defaults`` on
        ``self.sensors`` (when present), not here.
        """
        self.max_episode_length = INFERENCE_EPISODE_LENGTH
