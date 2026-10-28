# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduced target observations must only bind sensor channels they consume."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.base_env.publishers import prefix_names
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context_sources import sensed
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.target_poses import build_reduced_coords_target_poses
from protomotions.envs.state import BodyState
from protomotions.envs.state_names import body_state_names


def sensor_context():
    """A flat context of an encoder/IMU-only suite: no positions, no odometry."""
    rotations = torch.zeros(2, 3, 4)
    rotations[..., 3] = 1
    state = BodyState(
        rigid_body_rot=rotations,
        rigid_body_ang_vel=torch.zeros(2, 3, 3),
        dof_pos=torch.zeros(2, 2),
        dof_vel=torch.zeros(2, 2),
    )
    names = body_state_names(state, 1)
    return {
        **prefix_names("clean", names),
        **prefix_names("noisy", names),
        "mimic.future_anchor_rot": rotations,
        "mimic.future_anchor_pos": torch.ones(2, 3, 3),
        "mimic.future_dof_pos": torch.ones(2, 3, 2),
        "mimic.future_dof_vel": torch.ones(2, 3, 2),
        "mimic.future_anchor_vel": torch.zeros(2, 3, 3),
        "mimic.future_anchor_ang_vel": torch.zeros(2, 3, 3),
        "mimic.ref_anchor_pos": torch.zeros(2, 3),
    }


@pytest.mark.parametrize("source", ["clean", "noisy"])
@pytest.mark.parametrize("include_height", [False, True])
@pytest.mark.parametrize("zero_xy_offset", [False, True])
def test_target_without_measured_xy_drift_needs_no_localization(
    source, include_height, zero_xy_offset
):
    component = MdpComponent.from_kernel(
        build_reduced_coords_target_poses,
        include_height=include_height,
        include_xy_offset=zero_xy_offset,
        zero_xy_offset=zero_xy_offset,
    )
    component = sensed({"target": component}, source)["target"]
    context = sensor_context()
    ComponentManager(torch.device("cpu")).validate_bindings(
        {"target": component}, context, group="observation"
    )
    actual = component.compute(context)
    assert torch.isfinite(actual).all()
    assert actual.shape == (2, 3 * (6 + 4 + include_height + 2 * zero_xy_offset))


def test_real_xy_drift_rejects_suite_without_localization():
    # Measured XY drift: the recipe binds the anchor position, which the
    # sensor source then has to estimate.
    component = MdpComponent.from_kernel(
        build_reduced_coords_target_poses,
        include_xy_offset=True,
        current_state_anchor_pos=EnvContext.current.anchor_pos,
    )
    component = sensed({"target": component}, "noisy")["target"]
    with pytest.raises(ValueError, match="target.*noisy.anchor_pos"):
        ComponentManager(torch.device("cpu")).validate_bindings(
            {"target": component}, sensor_context(), group="observation"
        )


def test_public_l2c2_recipe_actor_and_clean_twin_execute_without_localization():
    from examples.experiments.mimic.mlp_bm_l2c2 import env_config
    from protomotions.robot_configs.g1 import G1RobotConfig

    config = env_config(G1RobotConfig(), SimpleNamespace())
    assert config.sensors.localization is None
    components = {
        name: component
        for name, component in config.observation_components.items()
        if name.startswith(("noisy_", "clean_"))
    }
    context = sensor_context()
    ComponentManager(torch.device("cpu")).validate_bindings(
        components, context, group="observation"
    )
    for component in components.values():
        assert torch.isfinite(component.compute(context)).all()
