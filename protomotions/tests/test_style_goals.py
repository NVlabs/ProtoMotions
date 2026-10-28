# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Untimed style geometry, reset lifecycle and per-frame encoder contract."""

import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.common import MaskedMean, ObsProcessor
from protomotions.agents.common.config import (
    MaskedMeanConfig,
    MLPWithConcatConfig,
    MLPLayerConfig,
    ObsProcessorConfig,
    ModuleOperationForwardConfig,
    ModuleOperationReshapeConfig,
)
from protomotions.agents.common.mlp import MLPWithConcat
from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
    MaskedMimicStyleConfig,
)
from protomotions.envs.control.mimic_control import MimicControl
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.cast import to_float
from protomotions.envs.obs.masked_mimic import (
    apply_style_condition_masks,
    compute_local_pose_samples,
    compute_style_pose_samples,
    compute_style_velocity_samples,
)
from protomotions.tests.support.task_inputs import motion_fakes
from protomotions.tests.test_structured_target_sampling import make_inputs
from protomotions.utils import rotations


def style_control(n=3, **kwargs):
    config = MaskedMimicControlConfig(style=MaskedMimicStyleConfig(**kwargs))
    queries = []

    def get_state(ids, times):
        queries.append(times.clone())
        pos = torch.zeros(len(times), 3, 3)
        pos[:, 0, 2] = times
        pos[:, 2, 0] = 2
        rot = torch.zeros(len(times), 3, 4)
        rot[..., 3] = 1
        return SimpleNamespace(
            rigid_body_pos=pos,
            rigid_body_rot=rot,
            rigid_body_vel=torch.ones_like(pos),
            rigid_body_ang_vel=2 * torch.ones_like(pos),
        )

    motion_lib = SimpleNamespace(
        get_motion_length=lambda ids: torch.full_like(ids, 10, dtype=torch.float),
        get_motion_state=get_state,
        queries=queries,
    )
    inputs = make_inputs(
        n,
        motion_ids=torch.zeros(n, dtype=torch.long),
        motion_times=torch.ones(n),
        motion_lib=motion_lib,
    )
    return MaskedMimicControl(config, inputs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_frames": 0},
        {"num_frames": 6},
        {"num_frames": 1.5},
        {"window_seconds": float("nan")},
        {"window_seconds": float("inf")},
        {"near_time_bias_power": 0.5},
        {"near_time_bias_power": float("nan")},
        {"velocity_probability": float("nan")},
        {"velocity_probability": 1.1},
        {"visible_frame_count_probabilities": (1, 2)},
        {"visible_frame_count_probabilities": (0,) * 6},
        {"visible_frame_count_probabilities": (-1, 1, 1, 1, 1, 1)},
        {"visible_frame_count_probabilities": (float("inf"), 1, 1, 1, 1, 1)},
    ],
)
def test_invalid_style_config(kwargs):
    with pytest.raises(ValueError):
        MaskedMimicStyleConfig(**kwargs)


def test_style_config_pickle_and_explicit_frame_count():
    cfg = MaskedMimicControlConfig(style=MaskedMimicStyleConfig())
    restored = pickle.loads(pickle.dumps(cfg))
    restored.validate()
    assert restored == cfg
    assert cfg.style.visible_frame_count_probabilities == (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)


def test_style_count_distribution_and_stable_body_masks():
    torch.manual_seed(991)
    control = style_control(20000)
    control._sample_style_plan(torch.arange(20000))
    count = control.style_frame_masks.sum(-1)
    observed = torch.bincount(count, minlength=6) / count.numel()
    torch.testing.assert_close(
        observed, torch.tensor([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]), atol=0.015, rtol=0
    )
    masks = control.style_condition_masks
    assert torch.equal(
        masks, masks.any(1, keepdim=True) & control.style_frame_masks[..., None]
    )
    offsets = control.style_time_offsets
    assert ((offsets >= 0) & (offsets <= 2)).all()
    assert abs(offsets.mean().item() - 2 / 3) < 0.015


def test_style_reset_partial_and_sliding_reference():
    torch.manual_seed(123)
    control = style_control(visible_frame_count_probabilities=(0, 0, 0, 0, 0, 1))
    control.reset_rows(torch.arange(3), None)
    offsets = control.style_time_offsets.clone()
    masks = control.style_condition_masks.clone()
    control._query_style_samples(None)
    first = motion_fakes(control.inputs).motion_lib.queries[-1]
    motion_fakes(control.inputs).motion_manager.motion_times += 0.2
    control._query_style_samples(None)
    torch.testing.assert_close(
        motion_fakes(control.inputs).motion_lib.queries[-1], first + 0.2
    )
    assert torch.equal(offsets, control.style_time_offsets)
    assert torch.equal(masks, control.style_condition_masks)
    control.reset_rows(torch.tensor([1]), None)
    assert torch.equal(offsets[[0, 2]], control.style_time_offsets[[0, 2]])
    assert torch.equal(masks[[0, 2]], control.style_condition_masks[[0, 2]])
    assert not torch.equal(offsets[1], control.style_time_offsets[1])
    control.reset_rows(torch.tensor([], dtype=torch.long), None)
    assert torch.equal(offsets[[0, 2]], control.style_time_offsets[[0, 2]])


def test_style_query_subset_end_clamp_and_invalidation():
    control = style_control(visible_frame_count_probabilities=(0, 0, 0, 0, 0, 1))
    control.reset_rows(torch.arange(3), None)
    motion_fakes(control.inputs).motion_manager.motion_times[:] = 9.9
    all_values = control._query_style_samples(None)
    assert (motion_fakes(control.inputs).motion_lib.queries[-1] <= 10).all()
    selected = control._query_style_samples(torch.tensor([2, 0]))
    for key, value in selected.items():
        torch.testing.assert_close(value, all_values[key][[2, 0]])
    motion_fakes(control.inputs).motion_manager.motion_times[:] = 11
    assert all(not value.any() for value in control._query_style_samples(None).values())


def test_hidden_style_never_queries_motion_and_has_fixed_widths():
    control = style_control(visible_frame_count_probabilities=(1, 0, 0, 0, 0, 0))
    control.reset_rows(torch.arange(3), None)
    pose, velocity, masks, frames = control._query_style_samples(None).values()
    assert not motion_fakes(control.inputs).motion_lib.queries
    assert pose.shape == (3, 5 * 2 * 9)
    assert velocity.shape == (3, 5 * 2 * 6)
    assert masks.shape == (3, 5, 5)
    assert frames.shape == (3, 5)
    assert all(not x.any() for x in (pose, velocity, masks, frames))


def test_style_geometry_and_velocity_heading_with_nonroot_anchor():
    torch.manual_seed(4)
    pos = torch.randn(2, 5, 4, 3)
    rot = torch.nn.functional.normalize(torch.randn(2, 5, 4, 4), dim=-1)
    vel, ang = torch.randn_like(pos), torch.randn_like(pos)
    ids = torch.tensor([3, 1, 0])
    pose = compute_style_pose_samples(
        pos, rot, anchor_body_index=1, conditionable_body_ids=ids
    )
    torch.testing.assert_close(
        pose, compute_local_pose_samples(pos, rot, ids, 1).flatten(1)
    )
    velocity = compute_style_velocity_samples(
        vel, ang, rot, anchor_body_index=1, conditionable_body_ids=ids
    )
    yaw = rotations.quat_from_angle_axis(
        torch.tensor(0.8), torch.tensor([0.0, 0.0, 1.0]), True
    ).expand_as(rot)
    turn = lambda x: rotations.quat_rotate(yaw, x, True)
    moved_rot = rotations.quat_mul(yaw, rot, True)
    torch.testing.assert_close(
        pose,
        compute_style_pose_samples(
            turn(pos) + 10, moved_rot, anchor_body_index=1, conditionable_body_ids=ids
        ),
        atol=3e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        velocity,
        compute_style_velocity_samples(
            turn(vel),
            turn(ang),
            moved_rot,
            anchor_body_index=1,
            conditionable_body_ids=ids,
        ),
    )
    # Absolute anchor velocity is retained, not subtracted from other bodies.
    assert velocity.reshape(2, 5, 3, 6)[:, :, 1].abs().sum() > 0


def test_pose_property_and_velocity_masks():
    pose, velocity = torch.ones(2, 2 * 3 * 9), torch.ones(2, 2 * 3 * 6)
    masks = torch.zeros(2, 2, 7, dtype=torch.bool)
    masks[:, 0, 0] = True  # body 0 translation
    masks[:, 0, 3] = True  # body 1 rotation
    masks[1, 0, -1] = True
    p, v = apply_style_condition_masks(pose, velocity, masks)
    p, v = p.reshape(2, 2, 3, 9), v.reshape(2, 2, 3, 6)
    assert p[:, 0, 0, :3].all() and not p[:, 0, 0, 3:].any()
    assert not p[:, 0, 1, :3].any() and p[:, 0, 1, 3:].all()
    assert not p[:, 1].any() and not p[:, :, 2].any()
    assert not v[0].any() and not v[:, 1].any()
    assert v[1, 0, :2].all() and not v[1, 0, 2].any()


def test_masked_mean_ignores_hidden_values_and_gradients():
    module = MaskedMean(
        MaskedMeanConfig(in_keys=["values", "mask"], out_keys=["pooled"])
    )
    values = torch.randn(3, 5, 8, requires_grad=True)
    mask = torch.tensor([[0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
    td = TensorDict({"values": values, "mask": mask}, batch_size=[3])
    result = module(td)["pooled"]
    assert not result[0].any()
    torch.testing.assert_close(result[1], values[1, [0, 2]].mean(0))
    result.sum().backward()
    assert not values.grad[~mask.bool()].any()
    changed = values.detach().clone()
    changed[~mask.bool()] = float("nan")
    td["values"] = changed
    torch.testing.assert_close(module(td)["pooled"], result)


@pytest.mark.parametrize("num_properties", [1, 3])
def test_masked_mean_reduces_property_masks(num_properties):
    pool = MaskedMean(MaskedMeanConfig(in_keys=["v", "m"], out_keys=["p"]))
    values = torch.randn(2, 4, 6, requires_grad=True)
    masks = torch.zeros(2, 4, num_properties, dtype=torch.bool)
    masks[1, 0, 0] = True
    masks[1, 2, -1] = True
    result = pool(TensorDict({"v": values, "m": masks}, [2]))["p"]
    assert not result[0].any()
    torch.testing.assert_close(result[1], values[1, [0, 2]].mean(0))
    result.sum().backward()
    assert not values.grad[~masks.any(-1)].any()


@pytest.mark.parametrize("mask_shape", [(2, 0), (2, 0, 1), (2, 0, 3)])
def test_masked_mean_empty_sequence(mask_shape):
    pool = MaskedMean(MaskedMeanConfig(in_keys=["v", "m"], out_keys=["p"]))
    values = torch.empty(2, 0, 6, requires_grad=True)
    masks = torch.empty(mask_shape, dtype=torch.bool)
    result = pool(TensorDict({"v": values, "m": masks}, [2]))["p"]
    torch.testing.assert_close(result, torch.zeros(2, 6))
    result.sum().backward()
    assert values.grad.shape == values.shape


def test_style_encoder_composition_zero_after_normalization_and_mlp_bias():
    # Match the recipe: normalize per-frame features, concatenate explicit masks,
    # encode each frame independently, then pool only visible frames.
    processor = ObsProcessor(
        ObsProcessorConfig(
            in_keys=["pose"],
            out_keys=["pose_seq"],
            normalize_obs=True,
            module_operations=[
                ModuleOperationReshapeConfig(new_shape=["batch_size", 5, -1]),
                ModuleOperationForwardConfig(),
            ],
        )
    )
    encoder = MLPWithConcat(
        MLPWithConcatConfig(
            in_keys=["pose_seq", "velocity", "conditions"],
            out_keys=["tokens"],
            normalize_obs=False,
            num_out=8,
            layers=[MLPLayerConfig(units=16, activation="silu")],
            module_operations=[ModuleOperationForwardConfig()],
        )
    )
    pool = MaskedMean(
        MaskedMeanConfig(in_keys=["tokens", "frames"], out_keys=["style"])
    )
    td = TensorDict(
        {
            "pose": torch.zeros(3, 90),
            "velocity": torch.zeros(3, 5, 12),
            "conditions": torch.zeros(3, 5, 5),
            "frames": torch.zeros(3, 5),
        },
        [3],
    )
    processor(td)
    processor.norm.running_obs_norm.mean.fill_(2)
    processor(td)
    encoder(td)
    assert td["tokens"].abs().sum() > 0
    assert not pool(td)["style"].any()


def style_observations():
    """The four untimed style inputs as a recipe binds them (the Metatron
    hybrid recipe): a float cast of each style name the control publishes."""
    return {
        "masked_mimic_style_poses": MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.style_pose_samples
        ),
        "masked_mimic_style_velocities": MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.style_velocity_samples
        ),
        "masked_mimic_style_masks": MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.style_condition_masks
        ),
        "masked_mimic_style_frame_masks": MdpComponent.from_kernel(
            to_float, x=EnvContext.masked_mimic.style_frame_masks
        ),
    }


def test_style_observations_are_float_casts_of_the_published_style_names():
    components = style_observations()
    assert len(components) == 4
    for name, component in components.items():
        assert "time" not in name
        assert component.compute_func(torch.tensor([True])).dtype == torch.float32
        assert component.dynamic_vars["x"].path.startswith("masked_mimic.style_")


@pytest.mark.parametrize("robot_name", ["g1", "soma23"])
def test_style_context_end_to_end(robot_name, monkeypatch):
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config(robot_name)
    inputs = replace(style_control().inputs, robot=robot)
    bodies = len(robot.kinematic_info.body_names)

    def get_state(ids, times):
        pos = (
            torch.arange(bodies * 3)
            .reshape(1, bodies, 3)
            .float()
            .expand(len(times), -1, -1)
        )
        rot = torch.zeros(len(times), bodies, 4)
        rot[..., 3] = 1
        return SimpleNamespace(
            rigid_body_pos=pos,
            rigid_body_rot=rot,
            rigid_body_vel=pos * 0.1,
            rigid_body_ang_vel=pos * 0.2,
            dof_pos=torch.zeros(len(times), 2),
            dof_vel=torch.zeros(len(times), 2),
        )

    motion_fakes(inputs).motion_lib.get_motion_state = get_state
    monkeypatch.setattr(MimicControl, "publish", lambda self, env_ids: {})
    control = MaskedMimicControl(
        MaskedMimicControlConfig(
            style=MaskedMimicStyleConfig(
                visible_frame_count_probabilities=(0, 0, 0, 0, 0, 1)
            ),
        ),
        inputs,
    )
    control.reset_rows(torch.arange(3), None)
    ctx = control.publish(torch.tensor([2, 0]))
    components = style_observations()
    obs = {key: value.compute(ctx) for key, value in components.items()}
    count = len(robot.trackable_bodies_subset)
    assert obs["masked_mimic_style_poses"].shape == (2, 5 * count * 9)
    assert obs["masked_mimic_style_velocities"].shape == (2, 5 * count * 6)
    assert obs["masked_mimic_style_masks"].shape == (2, 5, count * 2 + 1)
    assert all(torch.isfinite(value).all() for value in obs.values())


def test_empty_style_batch():
    control = style_control()
    pose, velocity, masks, frames = control._query_style_samples(
        torch.tensor([], dtype=torch.long)
    ).values()
    assert pose.shape == (0, 90)
    assert velocity.shape == (0, 60)
    assert masks.shape == (0, 5, 5)
    assert frames.shape == (0, 5)


@pytest.mark.parametrize(
    "value_shape,mask_shape",
    [((2, 3), (2, 3)), ((2, 3, 4), (2, 4)), ((2, 3, 4), (2, 3, 1, 1))],
)
def test_masked_mean_rejects_inconsistent_shapes(value_shape, mask_shape):
    pool = MaskedMean(MaskedMeanConfig(in_keys=["v", "m"], out_keys=["p"]))
    with pytest.raises(ValueError):
        pool(
            TensorDict(
                {"v": torch.zeros(value_shape), "m": torch.zeros(mask_shape)}, [2]
            )
        )
