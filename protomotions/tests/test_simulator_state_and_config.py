# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for simulator config validation and state containers."""

import pytest
import torch

from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DelayDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    ProjectileConfig,
    PushDomainRandomizationConfig,
    RobotNoiseConfig,
    SimParams,
    SimulatorConfig,
    get_matching_indices,
)
from protomotions.simulator.base_simulator.simulator_state import (
    BaseBatchedState,
    DataConversionMapping,
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _robot_state(state_conversion=StateConversion.COMMON) -> RobotState:
    return RobotState(
        state_conversion=state_conversion,
        fps=30.0,
        dof_pos=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        dof_vel=torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
        dof_forces=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        rigid_body_pos=torch.tensor(
            [
                [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 2.0], [1.0, 0.0, 3.0]],
            ]
        ),
        rigid_body_rot=_identity_quat(2, 2),
        rigid_body_vel=torch.ones(2, 2, 3),
        rigid_body_ang_vel=torch.ones(2, 2, 3) * 2.0,
        rigid_body_contacts=torch.tensor([[True, False], [False, True]]),
        rigid_body_contact_forces=torch.ones(2, 2, 3) * 3.0,
        local_rigid_body_rot=_identity_quat(2, 2),
    )


def _conversion(sim_w_last: bool = True) -> DataConversionMapping:
    return DataConversionMapping(
        body_convert_to_common=torch.tensor([1, 0]),
        body_convert_to_sim=torch.tensor([1, 0]),
        dof_convert_to_common=torch.tensor([1, 0]),
        dof_convert_to_sim=torch.tensor([1, 0]),
        sim_w_last=sim_w_last,
    )


def test_delay_domain_randomization_ramp_epochs():
    # Default (no ramp_epochs) = exactly the previous behavior: full configured range
    # from epoch 0, at every epoch.
    cfg_no_ramp = DelayDomainRandomizationConfig(
        action_delay_steps=(0, 2), observation_delay_steps=(0, 2)
    )
    for epoch in (0, 1, 750, 1500, 999999):
        assert cfg_no_ramp.effective_max_action_delay(epoch) == 2
        assert cfg_no_ramp.effective_max_observation_delay(epoch) == 2

    # ramp_epochs=1500: linear ramp of the effective max from 0 (epoch 0) up to the
    # configured max (epoch >= ramp_epochs), min(1, epoch/ramp_epochs) fraction.
    cfg = DelayDomainRandomizationConfig(
        action_delay_steps=(0, 2),
        observation_delay_steps=(0, 2),
        ramp_epochs=1500,
    )
    assert cfg.effective_max_action_delay(0) == 0  # round(2 * 0/1500) = 0
    assert cfg.effective_max_action_delay(750) == 1  # round(2 * 0.5) = 1 (halfway)
    assert cfg.effective_max_action_delay(1500) == 2  # round(2 * 1.0) = 2 (full)
    assert cfg.effective_max_action_delay(3000) == 2  # clamped, min(1, epoch/ramp)=1
    # Observation delay ramps identically (independent config, same math).
    assert cfg.effective_max_observation_delay(0) == 0
    assert cfg.effective_max_observation_delay(750) == 1
    assert cfg.effective_max_observation_delay(1500) == 2

    # Configured min is respected even mid-ramp: effective_max never drops below the
    # configured min (e.g. min=1 with a ramp that would otherwise compute max=0).
    cfg_min1 = DelayDomainRandomizationConfig(
        action_delay_steps=(1, 4),
        observation_delay_steps=(0, 4),
        ramp_epochs=1000,
    )
    assert cfg_min1.effective_max_action_delay(0) == 1  # round(4*0)=0, clamped up to min=1
    assert cfg_min1.effective_max_action_delay(500) == 2  # round(4*0.5)=2, >= min=1

    # ramp_epochs must be a positive int (or None).
    with pytest.raises(ValueError):
        DelayDomainRandomizationConfig(ramp_epochs=0)
    with pytest.raises(ValueError):
        DelayDomainRandomizationConfig(ramp_epochs=-5)


def test_simulator_config_matching_and_domain_randomization_validation():
    assert get_matching_indices(["left_foot", "right_foot", "head"], names_to_match=[".*foot"]) == [0, 1]
    assert get_matching_indices(["a", "b"], indices_to_match=[1]) == [1]
    with pytest.raises(AssertionError, match="Either names_to_match"):
        get_matching_indices(["a"])
    with pytest.raises(AssertionError, match="Only one"):
        get_matching_indices(["a"], names_to_match=["a"], indices_to_match=[0])
    with pytest.raises(AssertionError, match="Multiple regex"):
        get_matching_indices(["foot"], names_to_match=[".*", "foot"])

    ActionNoiseDomainRandomizationConfig(
        action_noise_range=(0.0, 1.0),
        dof_indices=[0],
    )
    for kwargs, message in [
        ({"action_noise_range": (0.0, 1.0), "dof_names": ["a"], "dof_indices": [0]}, "Only one"),
        ({"action_noise_range": (0.0, 1.0)}, "Either dof_names"),
        ({"action_noise_range": None, "dof_indices": [0]}, "action_noise_range"),
        ({"action_noise_range": (1.0, 1.0), "dof_indices": [0]}, "first value"),
    ]:
        with pytest.raises(ValueError, match=message):
            ActionNoiseDomainRandomizationConfig(**kwargs)

    FrictionDomainRandomizationConfig(body_indices=[0])
    with pytest.raises(ValueError, match="Either body_names"):
        FrictionDomainRandomizationConfig()
    with pytest.raises(ValueError, match="Only one"):
        FrictionDomainRandomizationConfig(body_names=[".*"], body_indices=[0])

    CenterOfMassDomainRandomizationConfig(com_range={"x": (-0.1, 0.1)}, body_names=["torso"])
    with pytest.raises(ValueError, match="valid keys"):
        CenterOfMassDomainRandomizationConfig(com_range=None, body_indices=[0])
    with pytest.raises(ValueError, match="valid keys"):
        CenterOfMassDomainRandomizationConfig(com_range={"bad": (0.0, 1.0)}, body_indices=[0])
    with pytest.raises(ValueError, match="Either body_names"):
        CenterOfMassDomainRandomizationConfig(com_range={"x": (-0.1, 0.1)})
    with pytest.raises(ValueError, match="Only one"):
        CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.1, 0.1)},
            body_names=["torso"],
            body_indices=[0],
        )

    assert RobotNoiseConfig().has_noise() is False
    assert RobotNoiseConfig(dof_pos_noise=[0.0, 0.1]).has_noise() is True
    assert PushDomainRandomizationConfig().has_push() is False
    assert PushDomainRandomizationConfig(max_linear_velocity=(0.0, 1.0, 0.0)).has_push() is True
    with pytest.raises(ValueError, match="positive"):
        PushDomainRandomizationConfig(push_interval_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="<="):
        PushDomainRandomizationConfig(push_interval_range=(2.0, 1.0))

    assert ProjectileConfig(num_projectiles=1, cube_half_size_range=(0.1, 0.3)).get_sizes() == [0.1]
    assert ProjectileConfig(num_projectiles=3, cube_half_size_range=(0.1, 0.3)).get_sizes() == [0.1, 0.2, 0.3]
    SimulatorConfig(
        _target_="fake.Sim",
        w_last=True,
        headless=True,
        num_envs=2,
        sim=SimParams(fps=60, decimation=2),
        experiment_name="unit",
    )
    with pytest.raises(AssertionError, match="SimulatorConfig._target_"):
        SimulatorConfig(w_last=True, headless=True, num_envs=1, sim=SimParams(), experiment_name="x")


def test_robot_state_properties_indexing_assignment_and_dict_roundtrip(capsys):
    state = _robot_state()

    assert StateConversion.from_str("COMMON") is StateConversion.COMMON
    with pytest.raises(ValueError, match="not a valid"):
        StateConversion.from_str("bad")
    assert torch.equal(state.root_pos, state.rigid_body_pos[:, 0, :])
    assert state.motion_dt == 1.0 / 30.0
    assert state.motion_length == (2 - 1) * (1.0 / 30.0)
    assert state.num_bodies == 2
    assert state.num_dofs == 2
    assert state.get_shape_mapping()["rigid_body_pos"] == (2, 3)
    assert state.get_shape_mapping(flattened=True)["rigid_body_pos"] == (6,)
    assert state.flatten_bodies("rigid_body_pos").shape == (2, 6)
    flattened_dof = state.flatten_bodies("dof_pos")
    assert torch.equal(flattened_dof, state.dof_pos)
    state.unflatten_bodies("rigid_body_pos", state.flatten_bodies("rigid_body_pos"))
    empty_state = RobotState(state_conversion=StateConversion.COMMON)
    assert empty_state.flatten_bodies("rigid_body_pos") is None
    assert empty_state.unflatten_bodies("rigid_body_pos", torch.empty(0)) is None

    clone = state.clone()
    clone.dof_pos[0, 0] = -99.0
    assert state.dof_pos[0, 0] != clone.dof_pos[0, 0]
    assert "dof_pos" in state
    assert "missing" not in state
    assert "rigid_body_pos" in list(iter(state))
    assert len(state) == len(state.to_dict())
    assert torch.equal(state["dof_pos"], state.dof_pos)
    with pytest.raises(KeyError):
        state["missing"]
    with pytest.raises(KeyError):
        state["missing"] = torch.ones(1)

    subset = state[torch.tensor([1])]
    assert subset.dof_pos.shape == (1, 2)
    state["dof_pos"] = torch.zeros_like(state.dof_pos)
    assert torch.equal(state.dof_pos, torch.zeros(2, 2))
    state[torch.tensor([0])] = subset
    assert torch.equal(state.dof_pos[0], subset.dof_pos[0])
    with pytest.raises(TypeError, match="BaseBatchedState"):
        state[torch.tensor([0])] = object()
    with pytest.raises(TypeError, match="Type mismatch"):
        state[torch.tensor([0])] = RootOnlyState(state_conversion=StateConversion.COMMON)

    restored = RobotState.from_dict(
        {"dof_pos": torch.ones(1, 2), "bad": torch.ones(1)},
        state_conversion=StateConversion.COMMON,
    )
    assert torch.equal(restored.dof_pos, torch.ones(1, 2))
    assert "Warning: Key 'bad'" in capsys.readouterr().out

    assert BaseBatchedState.motion_num_frames.fget(state) is None
    assert BaseBatchedState.convert_to_common(state, _conversion()) is None
    assert BaseBatchedState.convert_to_sim(state, _conversion()) is None


def test_robot_state_empty_properties_and_extra_conversion_branches():
    empty = RobotState(state_conversion=StateConversion.COMMON)
    assert empty.root_pos is None
    assert empty.root_rot is None
    assert empty.root_vel is None
    assert empty.root_ang_vel is None

    state = _robot_state(state_conversion=StateConversion.SIMULATOR)
    state.rigid_body_rot = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ]
    )
    state.convert_to_common(_conversion(sim_w_last=False))
    assert torch.equal(state.rigid_body_rot, _identity_quat(2, 2)[:, [1, 0]])
    state.convert_to_sim(_conversion(sim_w_last=False))
    assert state.state_conversion is StateConversion.SIMULATOR

    two_dim = RobotState(
        state_conversion=StateConversion.COMMON,
        dof_pos=torch.ones(2, 2),
        rigid_body_pos=torch.ones(2, 2, 3),
    )
    flattened = torch.arange(4.0).reshape(2, 2)
    assert torch.equal(two_dim.unflatten_bodies("dof_pos", flattened), flattened)

    target_common = RobotState(state_conversion=StateConversion.COMMON)
    target_common.data_conversion = _conversion()
    target_common.merge_fields_from(
        RobotState(
            state_conversion=StateConversion.SIMULATOR,
            dof_pos=torch.tensor([[1.0, 2.0]]),
        )
    )
    assert torch.equal(target_common.dof_pos, torch.tensor([[2.0, 1.0]]))

    target_sim = RobotState(state_conversion=StateConversion.SIMULATOR)
    target_sim.data_conversion = _conversion()
    target_sim.merge_fields_from(
        RobotState(
            state_conversion=StateConversion.COMMON,
            dof_pos=torch.tensor([[3.0, 4.0]]),
        )
    )
    assert torch.equal(target_sim.dof_pos, torch.tensor([[4.0, 3.0]]))

    with pytest.raises(AssertionError, match="dof_pos is not finite at indices"):
        RobotState(
            state_conversion=StateConversion.COMMON,
            dof_pos=torch.tensor([float("nan")]),
        )


def test_robot_state_conversion_translation_height_and_merge_paths():
    state = _robot_state(state_conversion=StateConversion.SIMULATOR)
    state.convert_to_common(_conversion(sim_w_last=True))

    assert state.state_conversion is StateConversion.COMMON
    assert torch.equal(state.dof_pos, torch.tensor([[1.0, 0.0], [3.0, 2.0]]))
    assert torch.equal(state.rigid_body_pos[:, 0], torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 3.0]]))
    state.convert_to_sim(_conversion(sim_w_last=True))
    assert state.state_conversion is StateConversion.SIMULATOR

    translated = _robot_state()
    translated.translate(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(translated.rigid_body_pos[0, 0], torch.tensor([1.0, 2.0, 2.0]))
    translated.translate(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    translated.translate(torch.zeros(2, 1, 3))
    with pytest.raises(ValueError, match="Invalid translation"):
        translated.translate(torch.zeros(1, 1, 1, 3))

    height_fixed = _robot_state()
    height_fixed.fix_height(z_up=True, height_offset=0.25)
    assert torch.isclose(height_fixed.rigid_body_pos[..., 2].min(), torch.tensor(0.25))
    per_frame = _robot_state()
    lifts = per_frame.fix_height_per_frame(z_up=True, height_offset=0.5)
    assert lifts.shape == (2, 3)
    assert torch.all(per_frame.rigid_body_pos[..., 2].min(dim=1)[0] >= 0.5)

    target = RobotState(state_conversion=StateConversion.COMMON, dof_pos=torch.ones(1, 2))
    source = RobotState(state_conversion=StateConversion.COMMON, dof_vel=torch.ones(1, 2))
    target.merge_fields_from(source)
    assert torch.equal(target.dof_vel, torch.ones(1, 2))
    with pytest.raises(AssertionError, match="already has a value"):
        target.merge_fields_from(RobotState(state_conversion=StateConversion.COMMON, dof_pos=torch.ones(1, 2)))

    with pytest.raises(AssertionError, match="rigid_body_pos is not finite"):
        RobotState(
            state_conversion=StateConversion.COMMON,
            rigid_body_pos=torch.tensor([[[float("nan"), 0.0, 0.0]]]),
        )


def test_root_reset_and_object_state_paths():
    conversion = _conversion(sim_w_last=False)
    root = RootOnlyState(
        state_conversion=StateConversion.SIMULATOR,
        fps=20.0,
        root_pos=torch.zeros(2, 3),
        root_rot=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        root_vel=torch.ones(2, 3),
        root_ang_vel=torch.ones(2, 3) * 2.0,
    )
    root.convert_to_common(conversion)
    assert torch.equal(root.root_rot, _identity_quat(2))
    root.convert_to_sim(conversion)
    assert torch.equal(
        root.root_rot,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    assert root.motion_num_frames == 2
    root.translate(torch.tensor([1.0, 2.0, 3.0]))
    root.translate(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="Invalid translation"):
        root.translate(torch.zeros(1, 1, 3))

    robot = _robot_state()
    reset = ResetState.from_robot_state(robot)
    assert torch.equal(reset.root_pos, robot.root_pos)
    assert reset.motion_num_frames == 2
    reset_no_fps = ResetState(state_conversion=StateConversion.COMMON)
    assert reset_no_fps.motion_num_frames is None
    reset.convert_to_sim(conversion)
    assert reset.state_conversion is StateConversion.SIMULATOR
    reset.convert_to_common(conversion)
    reset.translate(torch.tensor([0.0, 0.0, 1.0]))
    reset.translate(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]))
    with pytest.raises(ValueError, match="Invalid translation"):
        reset.translate(torch.zeros(1, 1, 3))

    obj = ObjectState(
        state_conversion=StateConversion.COMMON,
        fps=10.0,
        root_pos=torch.zeros(2, 2, 3),
        root_rot=_identity_quat(2, 2),
        root_vel=torch.ones(2, 2, 3),
        root_ang_vel=torch.ones(2, 2, 3) * 2.0,
        contact_forces=torch.ones(2, 2, 3) * 3.0,
    )
    assert obj.motion_num_frames == 2
    obj.translate(torch.tensor([1.0, 0.0, 0.0]))
    obj.translate(torch.zeros(2, 3))
    obj.translate(torch.zeros(2, 2, 3))
    with pytest.raises(ValueError, match="Invalid translation"):
        obj.translate(torch.zeros(1, 1, 1, 3))
    obj.convert_to_sim(conversion)
    assert obj.state_conversion is StateConversion.SIMULATOR
    obj.convert_to_common(conversion)
    assert obj.state_conversion is StateConversion.COMMON

    with pytest.raises(AssertionError, match="root_pos is not finite"):
        ObjectState(
            state_conversion=StateConversion.COMMON,
            root_pos=torch.tensor([[[float("inf"), 0.0, 0.0]]]),
        )
