# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded global prompt contracts; no simulator or checkpoint required."""

from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.masked_mimic_control import (
    FixedBodyCondition,
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.envs.obs.masked_mimic import compute_target_poses_only
from protomotions.tests.support.fake_reference import fake_reference
from protomotions.tests.support.task_inputs import (
    make_robot,
    make_task_inputs,
    motion_fakes,
)


def sample(profile=1, n=128, steps=5, bodies=5, visibility=0.8, fixed=None):
    weights = tuple(int(i == profile) for i in range(3))
    names = [f"body{i}" for i in range(bodies)]
    conditions = None
    if fixed is not None:
        conditions = [
            FixedBodyCondition(name, 1 if pos and rot else (0 if pos else 2))
            for name, (pos, rot) in zip(names, fixed.tolist())
            if pos or rot
        ]
    cfg = MaskedMimicControlConfig(
        target_profile_probabilities=weights,
        num_masked_future_steps=steps,
        visible_target_pose_prob=visibility,
        fixed_conditioning=conditions,
    )
    inputs = make_inputs(body_names=names, trackable=names)
    return MaskedMimicControl(cfg, inputs)._sample_target_plan(n)[:2]


@pytest.mark.parametrize(
    "profile,bounds", [(0, (0, 0.25)), (1, (0.25, 0.65)), (2, (0.65, 1))]
)
def test_density_spacing_and_coherent_properties(profile, bounds):
    torch.manual_seed(12)
    offsets, masks = sample(profile)
    gap = offsets[:, 0]
    assert (gap >= 0.1 + 1.9 * bounds[0]).all()
    assert (gap <= 0.1 + 1.9 * bounds[1]).all()
    torch.testing.assert_close(offsets, gap[:, None] * torch.arange(1, 6))
    template = masks.any(dim=1, keepdim=True)
    visible = masks.any(dim=-1).any(dim=-1)
    assert torch.equal(masks, template & visible[:, :, None, None])
    if profile == 0:
        assert masks.all()
    elif profile == 2:
        assert (visible.sum(1) == 1).all()
        assert (template.any(-1).sum(-1) == 1).all()


def test_reproducibility_and_empty_batch():
    torch.manual_seed(45)
    first = sample()
    torch.manual_seed(45)
    second = sample()
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    offsets, masks = sample(n=0)
    assert offsets.shape == (0, 5)
    assert masks.shape == (0, 5, 5, 2)


def test_profile_weight_cache_preserves_runtime_edits_and_rng():
    control = make_control()
    control._sample_target_plan(8)
    weights = control._profile_weights
    control._sample_target_plan(8)
    assert control._profile_weights is weights

    # Match a fresh controller after editing the live config, including RNG use.
    control.config.target_profile_probabilities = (0, 0, 1)
    fresh = make_control(
        MaskedMimicControlConfig(target_profile_probabilities=(0, 0, 1))
    )
    torch.manual_seed(17)
    expected = fresh._sample_target_plan(8)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(17)
    actual = control._sample_target_plan(8)
    assert control._profile_weights is not weights
    for old, new in zip(expected, actual):
        assert torch.equal(old, new)
    assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("profile", [0, 1, 2])
def test_fixed_conditions_override_profile_body_selection(profile):
    fixed = torch.tensor([[True, False], [False, True], [False, False]])
    _, masks = sample(profile, bodies=3, visibility=1, fixed=fixed)
    visible = masks.any(-1).any(-1)
    if profile == 2:
        assert (visible.sum(dim=1) == 1).all()
    else:
        assert visible.all()
    assert torch.equal(masks, fixed[None, None] & visible[:, :, None, None])


def test_unconditional_and_single_body_single_frame():
    assert not sample(1, visibility=0)[1].any()
    assert sample(0, visibility=0)[1].all()
    assert sample(2, steps=1, bodies=1)[1].any(-1).all()
    empty = torch.zeros(5, 2, dtype=torch.bool)
    assert not sample(0, fixed=empty)[1].any()


def test_minimum_spacing_is_one_control_step():
    control = make_control(
        MaskedMimicControlConfig(
            target_gap_min_seconds=0.001,
            target_gap_max_seconds=0.001,
        )
    )
    offsets, _, _ = control._sample_target_plan(8)
    torch.testing.assert_close(offsets, (0.02 * torch.arange(1, 6)).expand(8, -1))


@pytest.mark.parametrize("robot_name", ["g1", "soma23"])
def test_existing_robot_recipe_uses_structured_sampling_by_default(robot_name):
    from examples.experiments.masked_mimic import transformer
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config(robot_name)
    env_cfg = transformer.env_config(robot, SimpleNamespace(expert_model_path=None))
    cfg = env_cfg.control_components["masked_mimic"]
    assert cfg.target_profile_probabilities == (0.2, 0.6, 0.2)
    assert pickle.loads(pickle.dumps(cfg)) == cfg
    control = MaskedMimicControl(cfg, replace(make_inputs(), robot=robot))
    control.reset_rows(torch.arange(3), None)
    assert control.global_stream.body_masks.shape == (
        3,
        cfg.num_masked_future_steps * len(robot.trackable_bodies_subset) * 2,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_profile_probabilities": (0, 0, 0)},
        {"target_profile_probabilities": (-1, 1, 1)},
        {"target_profile_probabilities": (1, 1)},
        {"target_profile_probabilities": (float("nan"), 1, 1)},
        {"target_gap_min_seconds": 0},
        {"target_gap_max_seconds": 0.01},
        {"target_gap_max_seconds": float("inf")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MaskedMimicControlConfig(**kwargs).validate()


def make_inputs(
    num_envs=3,
    *,
    body_names=("torso", "helper", "hand"),
    trackable=("torso", "hand"),
    motion_ids=None,
    motion_times=None,
    motion_lib=None,
    placement_offset=None,
):
    """A ``TaskInputs`` over the sampling tests' motion fakes: three clips of
    10, 20 and 30 s, one env per clip (``motion_fakes(inputs)`` reaches the
    fakes afterwards)."""
    lengths = torch.tensor([10.0, 20.0, 30.0])
    motion = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        motion_manager=SimpleNamespace(
            motion_times=(
                torch.tensor([1.0, 2.0, 3.0]) if motion_times is None else motion_times
            ),
            motion_ids=torch.arange(num_envs) if motion_ids is None else motion_ids,
        ),
        motion_lib=(
            SimpleNamespace(get_motion_length=lambda ids: lengths[ids])
            if motion_lib is None
            else motion_lib
        ),
    )
    return make_task_inputs(
        num_envs,
        robot=make_robot(body_names, trackable_bodies_subset=trackable),
        reference=fake_reference(motion, placement_offset=placement_offset),
        dt=0.02,
    )


def make_control(config=None, inputs=None):
    config = config or MaskedMimicControlConfig()
    return MaskedMimicControl(config, inputs or make_inputs())


def test_partial_reset_and_step_do_not_resample_plan():
    torch.manual_seed(11)
    control = make_control()
    control.reset_rows(torch.arange(3), None)
    before = [
        x.clone()
        for x in (
            control.global_stream.times,
            control.global_stream.offsets,
            control.global_stream.body_masks,
            control.global_stream.frame_masks,
        )
    ]
    control.reset_rows(torch.tensor([1]), None)
    for old, new in zip(
        before,
        (
            control.global_stream.times,
            control.global_stream.offsets,
            control.global_stream.body_masks,
            control.global_stream.frame_masks,
        ),
    ):
        assert torch.equal(old[[0, 2]], new[[0, 2]])
    masks = control.global_stream.body_masks.clone()
    offsets = control.global_stream.offsets.clone()
    times = control.global_stream.times.clone()
    rng = torch.get_rng_state()
    motion_fakes(control.inputs).motion_manager.motion_times += 0.25
    control.advance(None)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(masks, control.global_stream.body_masks)
    now = motion_fakes(control.inputs).motion_manager.motion_times
    expected = torch.where((now >= times[:, 0])[:, None], offsets + now[:, None], times)
    torch.testing.assert_close(control.global_stream.times, expected)
    motion_fakes(control.inputs).motion_manager.motion_times[:] = torch.tensor(
        [10.0, 20.0, 30.0]
    )
    control.advance(None)
    assert torch.equal(
        control.global_stream.times,
        motion_fakes(control.inputs).motion_manager.motion_times[:, None].expand(-1, 5),
    )
    assert torch.isfinite(control.global_stream.times).all()
    control.reset_rows(torch.empty(0, dtype=torch.long), None)


def test_config_pickle_roundtrip():
    config = MaskedMimicControlConfig()
    assert pickle.loads(pickle.dumps(config)) == config


def test_fixed_body_resolution_and_invalid_entries():
    cfg = MaskedMimicControlConfig(
        target_profile_probabilities=(1, 0, 0),
        fixed_conditioning=[FixedBodyCondition("hand", 2)],
    )
    control = make_control(cfg)
    control.reset_rows(torch.arange(3), None)
    masks = control.global_stream.body_masks.reshape(3, 5, 2, 2)
    assert masks[:, :, 1, 1].all()
    assert masks.sum() == 15
    for entries in (
        [FixedBodyCondition("missing", 0)],
        [FixedBodyCondition("hand", 3)],
        [FixedBodyCondition("hand", 0), FixedBodyCondition("hand", 1)],
    ):
        cfg.fixed_conditioning = entries
        with pytest.raises(ValueError):
            make_control(cfg)


def test_plan_reaches_existing_context_and_observation_kernels(monkeypatch):
    control = make_control()
    control.reset_rows(torch.arange(3), None)
    monkeypatch.setattr(MimicControl, "publish", lambda self, env_ids: {})

    def state(ids, times):
        rot = torch.zeros(len(ids), 3, 4)
        rot[..., -1] = 1
        pos = times[:, None, None].expand(-1, 3, 3)
        return SimpleNamespace(
            rigid_body_pos=pos,
            rigid_body_rot=rot,
            rigid_body_vel=torch.zeros_like(pos),
            rigid_body_ang_vel=torch.zeros_like(pos),
            dof_pos=torch.zeros(len(ids), 2),
            dof_vel=torch.zeros(len(ids), 2),
        )

    motion_fakes(control.inputs).motion_lib.get_motion_state = state
    ctx = control.publish(None)
    # Disabled modalities contribute nothing, not even a placeholder.
    for name in (
        "local_ref_pos",
        "local_target_poses_masks",
        "anchor_translation_mask",
        "style_pose_samples",
        "average_velocity",
    ):
        assert f"masked_mimic.{name}" not in ctx, name
    torch.testing.assert_close(
        ctx["masked_mimic.time_offsets"], control.global_stream.offsets
    )
    torch.testing.assert_close(
        ctx["masked_mimic.ref_pos"][..., 0, 0], control.global_stream.times
    )
    # The observation must count down, not keep a constant lookahead.
    original_times = control.global_stream.times.clone()
    motion_fakes(control.inputs).motion_manager.motion_times += 0.01
    control.advance(None)
    ctx = control.publish(None)
    assert torch.equal(control.global_stream.times, original_times)
    torch.testing.assert_close(
        ctx["masked_mimic.time_offsets"], control.global_stream.offsets - 0.01
    )
    current = state(torch.arange(3), torch.zeros(3))
    obs = compute_target_poses_only(
        control.conditionable_body_ids,
        current.rigid_body_pos,
        current.rigid_body_rot,
        ctx["masked_mimic.ref_pos"],
        ctx["masked_mimic.ref_rot"],
        ctx["masked_mimic.target_bodies_masks"],
        anchor_body_index=0,
    ).reshape(3, 5, 2, 2, 12)
    masks = ctx["masked_mimic.target_bodies_masks"].reshape(3, 5, 2, 2)
    assert (obs[~masks] == 0).all()
    assert torch.isfinite(obs).all()
