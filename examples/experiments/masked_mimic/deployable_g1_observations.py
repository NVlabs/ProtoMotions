# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 sensor-only observation wiring, independent of the student architecture.

Call configure_observations from an experiment's env_config. Existing privileged
keys remain available to the teacher/posterior; actor routing must explicitly
select the new deployable keys. This is not a complete training experiment.
"""

from typing import Dict, Sequence, Union

import torch

from protomotions.envs.context_sources import sensed
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cast import to_float
from protomotions.envs.obs.humanoid import compute_humanoid_reduced_coords_observations
from protomotions.envs.obs.humanoid_historical import compute_historical_proprioception
from protomotions.envs.obs.masked_mimic import compute_target_poses_only
from protomotions.envs.obs.masked_mimic_velocity_command import (
    compute_masked_mimic_velocity_command_obs,
)
from protomotions.robot_configs.base import abstract_names_to_body_names
from protomotions.utils.baked_fk import BakedAnchorFK

HISTORY_STEPS = [8, 4, 3, 2, 1]


def configure_robot(robot_cfg):
    """Select the same five bodies before env_config builds goal masks."""
    robot_cfg.trackable_bodies_subset = abstract_names_to_body_names([
        "torso_body_name", "all_left_hand_bodies", "all_right_hand_bodies",
        "all_left_foot_bodies", "all_right_foot_bodies",
    ], robot_cfg)


def deployable_proprioception_observations(
    kinematic_info,
    bodies: torch.Tensor,
    anchor_idx: int,
    history_steps: Union[int, Sequence[int]] = tuple(HISTORY_STEPS),
) -> Dict[str, MdpComponent]:
    """The sensor-only current/history contract; configure enough history.

    Encoder/IMU-derived FK of ``bodies`` in the current heading frame, the
    reduced sensor state, the incoming raw action and one proprioception token
    per lag in ``history_steps`` (1-based, in token order; ``HISTORY_STEPS`` by
    default; the component checks them). Every state binding reads the noisy
    sensor estimates: no world position, body velocity or absolute height
    enters these keys. Model routing is caller-owned.
    """
    fk = BakedAnchorFK(kinematic_info, anchor_idx, bodies)
    return sensed(
        {
            "fk_max_coords_obs": MdpComponent.from_kernel(fk),
            "reduced_coords_obs": MdpComponent.from_kernel(
                compute_humanoid_reduced_coords_observations
            ),
            "last_action": MdpComponent.from_kernel(
                to_float, x=EnvContext.current_action
            ),
            "historical_proprioception": MdpComponent.from_kernel(
                compute_historical_proprioception, fk=fk, history_steps=history_steps
            ),
        },
        "noisy",
    )


def configure_observations(robot_cfg, env_cfg):
    """Add the explicit five-body G1 observation contract to an environment."""
    names = robot_cfg.trackable_bodies_subset
    if len(names) != 5 or len(set(names)) != 5:
        raise ValueError("Configure the G1 five-body subset before building the environment")
    bodies = torch.tensor(robot_cfg.body_indices(*names))
    anchor = robot_cfg.anchor_body_index
    env_cfg.observation_components.update(deployable_proprioception_observations(
        robot_cfg.kinematic_info, bodies, anchor, HISTORY_STEPS,
    ))
    # Reuse the sensor runtime's world FK; only remove absolute height here. The
    # anchor sub-mask pair is bound when the control sub-masks the anchor pose.
    submasks = {}
    if env_cfg.control_components["masked_mimic"].anchor_pose_submasking:
        submasks = {
            "anchor_translation_mask": EnvContext.masked_mimic.anchor_translation_mask,
            "anchor_rotation_mask": EnvContext.masked_mimic.anchor_rotation_mask,
        }
    goals = {
        "fk_masked_mimic_target_poses": MdpComponent.from_kernel(
            compute_target_poses_only,
            conditionable_body_ids=bodies,
            ground_independently=True,
            include_root_relative=False,
            **submasks,
        ),
        "masked_mimic_velocity_command": MdpComponent.from_kernel(
            compute_masked_mimic_velocity_command_obs
        ),
    }
    env_cfg.observation_components.update(sensed(goals, "noisy"))
    env_cfg.observation_components["masked_mimic_velocity_command_mask"] = (
        MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.velocity_command_mask
        )
    )
    return env_cfg
