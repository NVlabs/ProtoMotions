# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchor/local prompt geometry and lifecycle, independent of a simulator."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.body_selection import BodySelection, resolve_body_selections
from protomotions.envs.control.masked_mimic_control import (
    FixedBodyCondition,
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.masked_mimic import (
    compute_local_pose_samples,
    compute_local_target_poses,
    compute_target_masks_only,
    compute_target_poses_only,
    masked_mimic_goal_observations,
)
from protomotions.tests.support.task_inputs import motion_fakes
from protomotions.tests.test_structured_target_sampling import make_control, make_inputs
from protomotions.utils import rotations


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def poses(device, n=3, steps=4, bodies=5):
    torch.manual_seed(38)
    pos = torch.randn(n, steps, bodies, 3, device=device)
    rot = torch.randn(n, steps, bodies, 4, device=device)
    rot = torch.nn.functional.normalize(rot, dim=-1)
    ids = torch.tensor([4, 1, 3], device=device)
    return pos, rot, ids


def yaw(angle, device):
    angle = torch.tensor(angle, device=device)
    return rotations.quat_from_angle_axis(
        angle, torch.tensor([0.0, 0.0, 1.0], device=device), True
    )


@pytest.mark.parametrize(
    "future_steps,indices",
    [(None, [0, 1, 2, 3]), ((1, 2), [0, 1]), ((3, 1), [2, 0])],
)
@pytest.mark.parametrize("anchor_masks", [False, True])
@pytest.mark.parametrize("include_root_relative", [False, True])
def test_target_selection_keeps_pose_mask_and_anchor_layout_aligned(
    future_steps, indices, anchor_masks, include_root_relative
):
    pos, rot, ids = poses("cpu", n=2)
    masks = torch.rand(2, 4, 3, 2) > 0.5
    frames = torch.rand(2, 4) > 0.5
    trans, rotation = (torch.rand(2, 4, 2) > 0.5 for _ in range(2))
    inputs = (pos, rot, masks, frames, trans, rotation)
    originals = [value.clone() for value in inputs]
    anchor = {
        "anchor_translation_mask": trans if anchor_masks else None,
        "anchor_rotation_mask": rotation if anchor_masks else None,
    }

    def outputs(steps):
        return (
            compute_target_poses_only(
                ids,
                pos[:, 0],
                rot[:, 0],
                pos,
                rot,
                masks.flatten(1),
                1,
                future_steps=steps,
                include_root_relative=include_root_relative,
                **anchor,
            ),
            compute_target_masks_only(ids, masks.flatten(1), steps, **anchor),
            compute_local_target_poses(
                ids, pos, rot, masks.flatten(1), frames, 1, steps
            ),
        )

    for all_steps, selected in zip(outputs(None), outputs(future_steps)):
        expected = all_steps.reshape(2, 4, -1)[:, indices].flatten(1)
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    for value, original in zip(inputs, originals):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_local_targets_use_their_own_heading_ground_and_nonroot_anchor(device):
    pos, rot, ids = poses(device)
    obs = compute_local_pose_samples(pos, rot, ids, 1)
    torch.testing.assert_close(obs[:, :, 1, :2], torch.zeros_like(obs[:, :, 1, :2]))
    expected_height = pos[:, :, 1, 2] - pos[..., 2].amin(-1)
    torch.testing.assert_close(obs[:, :, 1, 2], expected_height)
    turn = yaw(1.23, device).expand_as(rot)
    shifted_pos = rotations.quat_rotate(turn, pos, True) + torch.tensor(
        [12.0, -4.0, 8.0], device=device
    )
    shifted_rot = rotations.quat_mul(turn, rot, True)
    torch.testing.assert_close(
        obs,
        compute_local_pose_samples(shifted_pos, shifted_rot, ids, 1),
        atol=5e-6,
        rtol=1e-5,
    )


def test_local_masks_hide_properties_and_select_matching_frames(device):
    pos, rot, ids = poses(device)
    body_masks = torch.ones(3, 4, 3, 2, dtype=torch.bool, device=device)
    body_masks[:, :, 0, 0] = False
    body_masks[:, :, 2, 1] = False
    frame_masks = torch.tensor([[True, False, True, True]], device=device).expand(3, -1)
    args = (ids, pos, rot, body_masks.flatten(1), frame_masks, 1)
    all_obs = compute_local_target_poses(*args).reshape(3, 4, 3, 9)
    assert (all_obs[:, 1] == 0).all()
    assert (all_obs[:, :, 0, :3] == 0).all()
    assert (all_obs[:, :, 2, 3:] == 0).all()
    # Change a hidden body's horizontal position and rotation, not the shared
    # anchor/ground frame that is deliberately used to canonicalize all bodies.
    changed_pos, changed_rot = pos.clone(), rot.clone()
    changed_pos[:, :, 4, :2] += 20
    changed_rot[:, :, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    torch.testing.assert_close(
        all_obs.flatten(1),
        compute_local_target_poses(ids, changed_pos, changed_rot, *args[3:]),
    )
    selected = compute_local_target_poses(*args, future_steps=(1, 3))
    torch.testing.assert_close(selected, all_obs[:, [0, 2]].flatten(1))


@pytest.mark.parametrize("extended", [False, True])
def test_anchor_xy_height_and_heading_only_do_not_leak_tilt(device, extended):
    pos = torch.zeros(2, 3, 4, 3, device=device)
    pos[:, :, 1] = torch.tensor([3.0, 4.0, 5.0], device=device)
    rot = torch.zeros(2, 3, 4, 4, device=device)
    rot[..., 3] = 1
    body_ids = torch.tensor([1, 3], device=device)
    current_pos, current_rot = pos[:, 0] * 0, rot[:, 0].clone()
    tilt = rotations.quat_from_angle_axis(
        torch.tensor(0.6, device=device),
        torch.tensor([1.0, 0.0, 0.0], device=device),
        True,
    )
    rot[:, :, 1] = rotations.quat_mul(yaw(0.4, device), tilt, True)
    trans = torch.tensor([[[True, False]], [[False, True]]], device=device).expand(
        -1, 3, -1
    )
    heading = torch.tensor([[[False, True]]], device=device).expand(2, 3, -1)
    masks = torch.ones(2, 12, dtype=torch.bool, device=device)
    args = (body_ids, current_pos, current_rot, pos, rot, masks)
    kwargs = dict(
        include_root_relative=extended,
        anchor_translation_mask=trans,
        anchor_rotation_mask=heading,
        anchor_body_index=1,
    )
    obs = compute_target_poses_only(*args, **kwargs).reshape(2, 3, 2, 2, -1)
    assert (obs[0, :, 0, 0, 2] == 0).all()
    assert (obs[1, :, 0, 0, :2] == 0).all()
    rot_without_tilt = rot.clone()
    rot_without_tilt[:, :, 1] = yaw(0.4, device)
    no_tilt = compute_target_poses_only(
        body_ids, current_pos, current_rot, pos, rot_without_tilt, masks, **kwargs
    )
    torch.testing.assert_close(obs[:, :, 0, 1], no_tilt.reshape_as(obs)[:, :, 0, 1])
    full = dict(kwargs, anchor_rotation_mask=heading.flip(-1))
    full_tilt = compute_target_poses_only(*args, **full)
    full_no_tilt = compute_target_poses_only(
        body_ids, current_pos, current_rot, pos, rot_without_tilt, masks, **full
    )
    assert not torch.allclose(full_tilt, full_no_tilt)
    # Visibility remains authoritative even if caller-supplied submasks are on.
    hidden = compute_target_poses_only(
        body_ids, current_pos, current_rot, pos, rot, masks * 0, **kwargs
    )
    assert (hidden == 0).all()
    subset = compute_target_poses_only(*args, **kwargs, future_steps=(3, 1))
    torch.testing.assert_close(subset, obs[:, [2, 0]].flatten(1))


def canvas(*bodies, anchor_pose_submasking=False):
    """A goal canvas over ``bodies``; only the fields the builder reads matter."""
    return GoalCanvas(
        future_steps=(1,),
        num_masked_future_steps=1,
        conditionable_bodies=tuple(bodies),
        anchor_body=bodies[0],
        anchor_pose_submasking=anchor_pose_submasking,
        has_local_stream=False,
        style_frames=None,
        has_velocity_command=False,
    )


class _Robot:
    """The one attribute a named body selection resolves against."""

    kinematic_info = SimpleNamespace(body_names=["a", "b", "c"])


def test_goal_observations_bind_anchor_and_submasks_to_both_kernels():
    plain = masked_mimic_goal_observations(canvas("a", "b"))
    assert set(plain) == {
        "masked_mimic_target_poses",
        "masked_mimic_target_masks",
        "masked_mimic_target_times",
        "masked_mimic_target_poses_masks",
        "masked_mimic_target_bodies_masks",
    }
    poses = plain["masked_mimic_target_poses"]
    assert str(poses.dynamic_vars["anchor_body_index"]) == "mimic.anchor_idx"
    assert str(poses.dynamic_vars["current_state_body_pos"]) == "current.rigid_body_pos"
    assert "anchor_translation_mask" not in poses.dynamic_vars
    # The bodies are the canvas's names; the build resolves them.
    for name in ("masked_mimic_target_poses", "masked_mimic_target_masks"):
        selection = plain[name].static_params["conditionable_body_ids"]
        assert selection == BodySelection.named("a", "b")
    with_anchor = masked_mimic_goal_observations(
        canvas("a", "b", anchor_pose_submasking=True)
    )
    for name in ("masked_mimic_target_poses", "masked_mimic_target_masks"):
        bound = with_anchor[name].dynamic_vars
        assert str(bound["anchor_translation_mask"]) == "masked_mimic.anchor_translation_mask"
        assert str(bound["anchor_rotation_mask"]) == "masked_mimic.anchor_rotation_mask"
    # The anchor is the kernel's own default binding: a recipe that binds the
    # kernel directly reads the mimic anchor without naming it.
    direct = MdpComponent.from_kernel(
        compute_target_poses_only, conditionable_body_ids=BodySelection.named("a")
    )
    assert str(direct.dynamic_vars["anchor_body_index"]) == "mimic.anchor_idx"
    assert "anchor_body_index" not in direct.static_params


def test_goal_observations_cover_the_same_frames():
    torch.manual_seed(3)
    envs, steps, bodies, ids = 2, 5, 3, torch.tensor([2, 0])
    rot = torch.nn.functional.normalize(torch.randn(envs, steps, bodies, 4), dim=-1)
    frame_masks = torch.tensor([[1, 0, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.bool)
    body_masks = torch.rand(envs, steps, len(ids), 2) < 0.5
    body_masks &= frame_masks[:, :, None, None]
    ctx = SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=torch.randn(envs, bodies, 3), rigid_body_rot=rot[:, 0]),
        mimic=SimpleNamespace(anchor_idx=0),
        masked_mimic=SimpleNamespace(
            ref_pos=torch.randn(envs, steps, bodies, 3),
            ref_rot=rot,
            time_offsets=torch.arange(steps).float().expand(envs, -1),
            target_poses_masks=frame_masks,
            target_bodies_masks=body_masks.flatten(1),
            anchor_translation_mask=frame_masks[..., None].expand(-1, -1, 2),
            anchor_rotation_mask=torch.zeros(envs, steps, 2, dtype=torch.bool),
        ),
    )
    components = resolve_body_selections(
        masked_mimic_goal_observations(canvas("c", "a", anchor_pose_submasking=True)),
        _Robot(),
    )
    assert torch.equal(
        components["masked_mimic_target_poses"].static_params["conditionable_body_ids"],
        ids,
    )
    obs = {name: component.compute(ctx) for name, component in components.items()}
    per_frame = {
        "masked_mimic_target_poses": len(ids) * 24,
        "masked_mimic_target_masks": len(ids) * 2 + 4,
        "masked_mimic_target_times": 1,
        "masked_mimic_target_poses_masks": 1,
        "masked_mimic_target_bodies_masks": len(ids) * 2,
    }
    for name, width in per_frame.items():
        assert obs[name].shape == (envs, steps * width), name
    # Hidden frames are zero in every masked output while the time stream keeps
    # all frames, so the per-frame outputs line up slot by slot.
    hidden = ~frame_masks
    for name, width in per_frame.items():
        if name != "masked_mimic_target_times":
            assert not obs[name].reshape(envs, steps, width)[hidden].any(), name
    torch.testing.assert_close(obs["masked_mimic_target_times"], ctx.masked_mimic.time_offsets)


def test_mask_features_append_four_anchor_bits_per_frame(device):
    body_masks = torch.zeros(2, 24, dtype=torch.bool, device=device)
    trans = torch.ones(2, 4, 2, dtype=torch.bool, device=device)
    rot = torch.zeros_like(trans)
    ids = torch.tensor([0, 1, 2], device=device)
    result = compute_target_masks_only(ids, body_masks, [3, 1], trans, rot).reshape(
        2, 2, 10
    )
    assert result[..., 6:8].all()
    assert not result[..., :6].any()
    assert not result[..., 8:].any()
    with pytest.raises(ValueError, match="together"):
        compute_target_masks_only(ids, body_masks, anchor_translation_mask=trans)


@pytest.mark.parametrize("global_prob,local_prob", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_global_and_local_enables_are_independent(global_prob, local_prob):
    cfg = MaskedMimicControlConfig(
        sparse_target_condition_probability=global_prob,
        local_target_condition_probability=local_prob,
        target_profile_probabilities=(1, 0, 0),
        anchor_pose_submasking=True,
    )
    control = make_control(cfg)
    control.reset_rows(torch.arange(3), None)
    assert control.global_stream.frame_masks.all().item() == bool(global_prob)
    assert (control.local_stream is None) == (local_prob == 0)
    if local_prob:
        assert control.local_stream.frame_masks.all()
    assert control.anchor_translation_mask.all().item() == bool(global_prob)
    assert control.anchor_rotation_mask[..., 0].all().item() == bool(global_prob)
    assert not control.anchor_rotation_mask[..., 1].any()


def test_dense_translation_only_anchor_keeps_both_xy_and_height():
    control = make_control(
        MaskedMimicControlConfig(
            anchor_pose_submasking=True,
            target_profile_probabilities=(1, 0, 0),
            fixed_conditioning=[FixedBodyCondition("torso", 0)],
        )
    )
    control.reset_rows(torch.arange(3), None)
    assert control.anchor_translation_mask.all()
    assert not control.anchor_rotation_mask.any()


def test_anchor_submodes_are_coherent_mutually_exclusive_and_visibility_gated():
    control = make_control(MaskedMimicControlConfig(anchor_pose_submasking=True))
    for seed in range(10):
        torch.manual_seed(seed)
        control.reset_rows(torch.arange(3), None)
        trans, rot = control.anchor_translation_mask, control.anchor_rotation_mask
        visible = control.global_stream.body_masks.reshape(3, 5, 2, 2)[:, :, 0]
        assert not (rot[..., 0] & rot[..., 1]).any()
        assert (trans[~visible[..., 0]] == 0).all()
        assert (rot[~visible[..., 1]] == 0).all()
        assert torch.equal(trans, trans.any(1, keepdim=True) & visible[..., 0:1])
        assert torch.equal(rot, rot.any(1, keepdim=True) & visible[..., 1:2])


def test_local_schedule_partial_reset_countdown_and_clip_end():
    cfg = MaskedMimicControlConfig(
        local_target_condition_probability=1, anchor_pose_submasking=True
    )
    control = make_control(cfg)
    control.reset_rows(torch.arange(3), None)
    assert not torch.equal(control.global_stream.times, control.local_stream.times)
    buffers = (
        control.local_stream.times,
        control.local_stream.offsets,
        control.local_stream.body_masks,
        control.local_stream.frame_masks,
        control.anchor_translation_mask,
        control.anchor_rotation_mask,
    )
    before = [x.clone() for x in buffers]
    control.reset_rows(torch.tensor([1]), None)
    for old, new in zip(before, buffers):
        assert torch.equal(old[[0, 2]], new[[0, 2]])
    before_times = control.local_stream.times.clone()
    before_global = control.global_stream.times.clone()
    before_masks = control.local_stream.body_masks.clone()
    # Force local due but global not due, proving separate advancement.
    manager = motion_fakes(control.inputs).motion_manager
    control.local_stream.times[:, 0] = manager.motion_times
    rng = torch.get_rng_state()
    control.advance(None)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(before_global, control.global_stream.times)
    assert torch.equal(before_masks, control.local_stream.body_masks)
    assert torch.equal(before_times, control.local_stream.times)
    manager = motion_fakes(control.inputs).motion_manager
    manager.motion_times[:] = torch.tensor([10.0, 20.0, 30.0])
    control.advance(None)
    assert torch.equal(
        control.local_stream.times,
        torch.tensor([10.0, 20.0, 30.0])[:, None].expand(3, 5),
    )
    control.reset_rows(torch.empty(0, dtype=torch.long), None)


@pytest.mark.parametrize("robot_name", ["g1", "soma23"])
def test_context_factory_and_robot_configs(robot_name, monkeypatch):
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config(robot_name)
    inputs = replace(make_inputs(), robot=robot)
    n_bodies = len(robot.kinematic_info.body_names)
    calls = []

    def get_state(ids, times):
        calls.append(times.clone())
        pos = torch.arange(n_bodies * 3).reshape(1, n_bodies, 3) * 0.02
        pos = pos.expand(len(times), -1, -1) + times[:, None, None]
        rot = torch.zeros(len(times), n_bodies, 4)
        rot[..., 3] = 1
        return SimpleNamespace(
            rigid_body_pos=pos,
            rigid_body_rot=rot,
            rigid_body_vel=torch.zeros_like(pos),
            rigid_body_ang_vel=torch.zeros_like(pos),
            dof_pos=torch.zeros(len(times), 2),
            dof_vel=torch.zeros(len(times), 2),
        )

    motion_fakes(inputs).motion_lib.get_motion_state = get_state
    monkeypatch.setattr(MimicControl, "publish", lambda self, env_ids: {})
    cfg = MaskedMimicControlConfig(
        local_target_condition_probability=1,
        anchor_pose_submasking=True,
        target_profile_probabilities=(1, 0, 0),
    )
    control = MaskedMimicControl(cfg, inputs)
    control.reset_rows(torch.arange(3), None)
    ctx = control.publish(torch.tensor([2, 0]))
    assert len(calls) == 2
    torch.testing.assert_close(
        ctx["masked_mimic.local_time_offsets"], control.local_stream.offsets[[2, 0]]
    )
    # The kernel's anchor default is the mimic control's published index, which
    # the patched-out MimicControl.publish does not supply here.
    ctx["mimic.anchor_idx"] = robot.anchor_body_index
    component = MdpComponent.from_kernel(
        compute_local_target_poses,
        conditionable_body_ids=control.conditionable_body_ids,
    )
    result = component.compute(ctx)
    assert result.shape == (2, 5 * len(robot.trackable_bodies_subset) * 9)
    # Current robot state cannot influence the local target kernel or binding.
    ctx["current.rigid_body_pos"] = torch.randn(2, n_bodies, 3) * 500
    torch.testing.assert_close(result, component.compute(ctx))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "name",
    ["sparse_target_condition_probability", "local_target_condition_probability"],
)
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_modality_probability(name, value):
    with pytest.raises(ValueError, match=name):
        make_control(MaskedMimicControlConfig(**{name: value}))


def test_anchor_must_be_conditionable():
    inputs = make_inputs()
    inputs.robot.anchor_body_index = 1  # "helper", which is not trackable
    with pytest.raises(ValueError, match="anchor"):
        MaskedMimicControl(
            MaskedMimicControlConfig(local_target_condition_probability=1), inputs
        )


def test_fractional_enables_sample_both_either_and_neither(device):
    control = make_control()
    torch.manual_seed(27)
    global_masks = torch.ones(4096, 2, 3, 2, device=device, dtype=torch.bool)
    local_masks = global_masks.clone()
    control._mask_disabled_stream(global_masks, 0.5)
    control._mask_disabled_stream(local_masks, 0.5)
    global_on = global_masks[:, 0, 0, 0]
    local_on = local_masks[:, 0, 0, 0]
    for event in (
        global_on & local_on,
        global_on & ~local_on,
        ~global_on & local_on,
        ~global_on & ~local_on,
    ):
        assert 0.20 < event.float().mean() < 0.30


def test_local_empty_batch_and_hidden_frame_inputs(device):
    pos, rot, ids = poses(device)
    masks = torch.ones(3, 24, dtype=torch.bool, device=device)
    frames = torch.ones(3, 4, dtype=torch.bool, device=device)
    empty = compute_local_target_poses(ids, pos[:0], rot[:0], masks[:0], frames[:0], 1)
    assert empty.shape == (0, 4 * 3 * 9)
    frames[:, 1] = False
    before = compute_local_target_poses(ids, pos, rot, masks, frames, 1)
    pos[:, 1] += torch.randn_like(pos[:, 1]) * 100
    rot[:, 1] = torch.nn.functional.normalize(torch.randn_like(rot[:, 1]), dim=-1)
    torch.testing.assert_close(
        before, compute_local_target_poses(ids, pos, rot, masks, frames, 1)
    )


def test_partial_anchor_translation_is_not_drawn_as_a_full_xyz_goal():
    inputs = replace(make_inputs(), headless=False)
    control = make_control(
        MaskedMimicControlConfig(
            anchor_pose_submasking=True,
            target_profile_probabilities=(1, 0, 0),
        ),
        inputs,
    )
    lib = motion_fakes(inputs).motion_lib
    lib.get_motion_state = lambda ids, times: SimpleNamespace(
        rigid_body_pos=torch.ones(len(ids), 3, 3),
    )
    control.reset_rows(torch.arange(3), None)
    control.anchor_translation_mask[..., 1] = False
    states = control.get_markers_state(None)
    assert all((state.translation[:, 0] > 90).all() for state in states.values())
    control.anchor_translation_mask[..., 1] = True
    states = control.get_markers_state(None)
    assert (
        torch.stack(
            [(state.translation[:, 0] < 10).all(-1) for state in states.values()]
        ).sum()
        == 3
    )
