# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed context views and MDP component factories."""

from types import SimpleNamespace

import torch

from protomotions.envs import component_factories as factories
from protomotions.envs import context_views
from protomotions.envs.context_views import (
    CurrentStateView,
    EnvContext,
    HistoricalView,
    MaskedMimicContext,
    MimicContext,
    PathContext,
    SteeringContext,
    TargetContext,
)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _robot_state(num_envs: int = 2, num_bodies: int = 3):
    rigid_body_pos = torch.arange(num_envs * num_bodies * 3, dtype=torch.float).reshape(
        num_envs, num_bodies, 3
    )
    rigid_body_rot = _identity_quat(num_envs, num_bodies)
    rigid_body_vel = torch.ones(num_envs, num_bodies, 3)
    rigid_body_ang_vel = torch.ones(num_envs, num_bodies, 3) * 2.0
    return SimpleNamespace(
        rigid_body_pos=rigid_body_pos,
        rigid_body_rot=rigid_body_rot,
        rigid_body_vel=rigid_body_vel,
        rigid_body_ang_vel=rigid_body_ang_vel,
        rigid_body_contacts=torch.tensor(
            [[True, False, True], [False, True, False]]
        ),
        rigid_body_contact_forces=torch.ones(num_envs, num_bodies, 3) * 3.0,
        dof_pos=torch.ones(num_envs, 4),
        dof_vel=torch.ones(num_envs, 4) * 2.0,
        dof_forces=torch.ones(num_envs, 4) * 3.0,
        local_rigid_body_rot=rigid_body_rot.clone(),
        root_pos=rigid_body_pos[:, 0, :],
        root_rot=rigid_body_rot[:, 0, :],
        root_vel=rigid_body_vel[:, 0, :],
        root_ang_vel=rigid_body_ang_vel[:, 0, :],
    )


def test_current_and_historical_views_precompute_accessors(monkeypatch):
    monkeypatch.setattr(
        context_views,
        "compute_local_ang_vel",
        lambda rot, ang_vel: ang_vel + 10.0,
    )
    state = _robot_state()

    current = CurrentStateView(state, anchor_idx=2)

    assert torch.equal(current.root_pos, state.root_pos)
    assert torch.equal(current.root_height, state.rigid_body_pos[:, 0, 2])
    assert torch.equal(current.anchor_pos, state.rigid_body_pos[:, 2, :])
    assert torch.equal(current.anchor_local_ang_vel, state.rigid_body_ang_vel[:, 2, :] + 10.0)
    assert EnvContext.current.anchor_pos.path == "current.anchor_pos"
    assert EnvContext.mimic.ref_state.rigid_body_pos.path == "mimic.ref_state.rigid_body_pos"

    buffer = SimpleNamespace(
        historical_rigid_body_pos=torch.ones(2, 2, 3, 3),
        historical_rigid_body_rot=_identity_quat(2, 2, 3),
        historical_rigid_body_vel=torch.ones(2, 2, 3, 3) * 2.0,
        historical_rigid_body_ang_vel=torch.ones(2, 2, 3, 3) * 3.0,
        historical_dof_pos=torch.ones(2, 2, 4),
        historical_dof_vel=torch.ones(2, 2, 4) * 4.0,
        historical_actions=torch.ones(2, 2, 5),
        historical_processed_actions=torch.ones(2, 2, 5) * 2.0,
        historical_ground_heights=torch.ones(2, 2) * 0.5,
        historical_body_contacts=torch.ones(2, 2, 3, dtype=torch.bool),
        historical_root_pos=torch.ones(2, 2, 3) * 5.0,
        historical_root_rot=_identity_quat(2, 2),
        historical_root_ang_vel=torch.ones(2, 2, 3) * 6.0,
        historical_anchor_pos=torch.ones(2, 2, 3) * 7.0,
        historical_anchor_rot=_identity_quat(2, 2),
        historical_anchor_vel=torch.ones(2, 2, 3) * 8.0,
        historical_anchor_ang_vel=torch.ones(2, 2, 3) * 9.0,
        noisy_historical_rigid_body_pos=torch.ones(2, 2, 3, 3) * -1.0,
        noisy_historical_rigid_body_rot=_identity_quat(2, 2, 3),
        noisy_historical_rigid_body_vel=torch.ones(2, 2, 3, 3) * -2.0,
        noisy_historical_rigid_body_ang_vel=torch.ones(2, 2, 3, 3) * -3.0,
        noisy_historical_dof_pos=torch.ones(2, 2, 4) * -4.0,
        noisy_historical_dof_vel=torch.ones(2, 2, 4) * -5.0,
        noisy_historical_ground_heights=torch.ones(2, 2) * -0.5,
        noisy_historical_root_pos=torch.ones(2, 2, 3) * -6.0,
        noisy_historical_root_rot=_identity_quat(2, 2),
        noisy_historical_root_ang_vel=torch.ones(2, 2, 3) * -7.0,
        noisy_historical_anchor_pos=torch.ones(2, 2, 3) * -8.0,
        noisy_historical_anchor_rot=_identity_quat(2, 2),
    )
    clean_history = HistoricalView(buffer, use_noisy=False)
    noisy_history = HistoricalView(buffer, use_noisy=True)

    assert torch.equal(clean_history.actions, buffer.historical_actions)
    assert torch.equal(clean_history.body_contacts, buffer.historical_body_contacts)
    assert torch.equal(clean_history.root_local_ang_vel, buffer.historical_root_ang_vel + 10.0)
    assert torch.equal(clean_history.anchor_ang_vel, buffer.historical_anchor_ang_vel)
    assert torch.equal(noisy_history.rigid_body_pos, buffer.noisy_historical_rigid_body_pos)
    assert noisy_history.actions is None
    assert noisy_history.anchor_vel is None


def test_control_contexts_and_env_context_store_optional_views():
    ref_state = _robot_state()
    future_pos = torch.ones(2, 4, 3, 3)
    future_rot = _identity_quat(2, 4, 3)
    future_vel = torch.ones(2, 4, 3, 3) * 2.0
    future_ang_vel = torch.ones(2, 4, 3, 3) * 3.0
    future_dof_pos = torch.ones(2, 4, 5)
    future_dof_vel = torch.ones(2, 4, 5) * 4.0
    mimic = MimicContext(
        ref_state=ref_state,
        future_pos=future_pos,
        future_rot=future_rot,
        future_vel=future_vel,
        future_ang_vel=future_ang_vel,
        future_dof_pos=future_dof_pos,
        future_dof_vel=future_dof_vel,
        anchor_idx=1,
        ref_lr=torch.ones(2, 5),
    )
    masked = MaskedMimicContext(
        mimic=mimic,
        ref_pos=future_pos,
        ref_rot=future_rot,
        target_times=torch.ones(2, 4),
        time_offsets=torch.arange(8, dtype=torch.float).reshape(2, 4),
        target_poses_masks=torch.ones(2, 4),
        target_bodies_masks=torch.ones(2, 4 * 3 * 2),
    )
    steering = SteeringContext(
        tar_dir=torch.ones(2, 2),
        tar_dir_theta=torch.ones(2),
        tar_speed=torch.ones(2) * 3.0,
        tar_face_dir=torch.ones(2, 2) * -1.0,
        prev_root_pos=torch.zeros(2, 3),
    )
    path = PathContext(
        tar_pos=torch.ones(2, 3),
        head_pos=torch.ones(2, 3) * 2.0,
        traj_samples=torch.ones(2, 5, 3),
        height_conditioned=True,
        head_body_id=2,
        progress_buf=torch.tensor([1, 2]),
    )
    target = TargetContext(torch.ones(2, 3), tar_proximity_threshold=0.5)
    current = CurrentStateView(ref_state, anchor_idx=1)
    env_context = EnvContext(
        current=current,
        noisy=current,
        dt=1.0 / 60.0,
        previous_action=torch.ones(2, 5),
        current_processed_action=torch.ones(2, 5) * 2.0,
        previous_processed_action=torch.ones(2, 5) * 3.0,
        ground_heights=torch.zeros(2),
        noisy_ground_heights=torch.ones(2),
        body_contacts=torch.ones(2, 3, dtype=torch.bool),
        current_contact_force_magnitudes=torch.ones(2, 3),
        prev_contact_force_magnitudes=torch.ones(2, 3) * 2.0,
        progress_buf=torch.tensor([4, 5]),
        contact_body_ids=torch.tensor([0, 2]),
        non_termination_contact_body_ids=torch.tensor([1]),
        odom_scale=torch.ones(2),
        odom_yaw_cos_sin=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        mimic=mimic,
        masked_mimic=masked,
        steering=steering,
        path=path,
        target=target,
    )

    assert torch.equal(mimic.future_root_pos, future_pos[:, :, 0, :])
    assert torch.equal(mimic.ref_anchor_pos, ref_state.rigid_body_pos[:, 1, :])
    assert torch.equal(masked.time_offsets, torch.arange(8, dtype=torch.float).reshape(2, 4))
    assert env_context.masked_mimic.mimic is mimic
    assert env_context.steering.tar_speed.tolist() == [3.0, 3.0]
    assert env_context.path.height_conditioned is True
    assert env_context.target.tar_proximity_threshold == 0.5


def _bindings(component):
    return component.get_bindings_dict()


def _params(component):
    return component.get_params()


def test_observation_factories_bind_expected_context_paths():
    max_coords = factories.max_coords_obs_factory(
        use_noisy=True,
        local_obs=False,
        root_height_obs=False,
        observe_contacts=True,
    )
    assert _bindings(max_coords)["body_pos"] == "noisy.rigid_body_pos"
    assert _bindings(max_coords)["ground_height"] == "noisy_ground_heights"
    assert _params(max_coords)["observe_contacts"] is True
    assert _params(max_coords)["local_obs"] is False

    reduced = factories.reduced_coords_obs_factory(
        root_height_obs=True,
        root_vel_obs=True,
    )
    assert _bindings(reduced)["root_pos"] == "current.root_pos"
    assert _bindings(reduced)["root_vel"] == "current.root_vel"

    hist_max = factories.historical_max_coords_obs_factory(
        use_noisy=True,
        observe_contacts=True,
        history_steps=[1, 3],
    )
    assert _bindings(hist_max)["historical_rigid_body_pos"] == "noisy_historical.rigid_body_pos"
    assert _params(hist_max)["history_steps"] == [1, 3]

    hist_reduced = factories.historical_reduced_coords_obs_factory(use_noisy=False)
    assert _bindings(hist_reduced)["historical_dof_pos"] == "historical.dof_pos"

    previous = factories.previous_actions_factory(history_steps=3, processed=True)
    assert _bindings(previous) == {"historical_actions": "historical.processed_actions"}
    assert _params(previous)["history_steps"] == 3

    max_target = factories.mimic_target_poses_max_coords_factory(
        use_noisy=True,
        with_velocities=False,
        with_relative=False,
        future_steps=2,
    )
    assert _bindings(max_target)["current_state_body_pos"] == "noisy.rigid_body_pos"
    assert _params(max_target)["future_steps"] == 2
    assert _params(max_target)["with_velocities"] is False

    future_rel = factories.mimic_target_poses_future_rel_factory(
        use_noisy=True,
        future_steps=4,
    )
    assert _bindings(future_rel)["current_state_body_rot"] == "noisy.rigid_body_rot"
    assert _params(future_rel)["future_steps"] == 4

    reduced_target = factories.mimic_target_poses_reduced_coords_factory(
        include_xy_offset=True,
        include_height=True,
        include_anchor_vel=True,
        include_anchor_ang_vel=True,
        zero_xy_offset=True,
    )
    assert _bindings(reduced_target)["mimic_ref_anchor_pos"] == "mimic.future_anchor_pos"
    assert _params(reduced_target)["zero_xy_offset"] is True

    deploy = factories.mimic_deploy_target_poses_factory(
        use_noisy=True,
        include_dof_vel=False,
        future_steps=[1, 4],
    )
    assert _bindings(deploy)["current_anchor_rot"] == "noisy.anchor_rot"
    assert _params(deploy)["include_dof_vel"] is False

    assert _bindings(factories.target_obs_factory())["tar_pos"] == "target.tar_pos"
    assert _bindings(factories.steering_obs_factory())["tar_speed"] == "steering.tar_speed"
    assert _bindings(factories.path_obs_factory())["traj_samples"] == "path.traj_samples"


def test_reward_factories_and_bundles_bind_expected_context_paths():
    smooth = factories.action_smoothness_factory(weight=-0.5)
    assert _bindings(smooth)["current_processed_action"] == "current_processed_action"
    assert _params(smooth)["weight"] == -0.5

    bundle = factories.mimic_tracking_rewards_factory(
        gt_weight=1.0,
        gr_weight=2.0,
        gv_weight=3.0,
        gav_weight=4.0,
        rh_weight=5.0,
    )
    assert set(bundle) == {"gt_rew", "gr_rew", "gv_rew", "gav_rew", "rh_rew"}
    assert _params(bundle["gt_rew"])["weight"] == 1.0
    assert _params(bundle["rh_rew"])["weight"] == 5.0

    rel = factories.gt_rel_rew_factory(body_indices=[0, 2])
    assert _bindings(rel)["anchor_idx"] == "mimic.anchor_idx"
    assert _params(rel)["body_indices"] == [0, 2]

    anchor_xy = factories.anchor_xy_rew_factory(weight=0.7, coefficient=-3.0)
    assert _bindings(anchor_xy)["current_anchor_pos"] == "current.anchor_pos"
    assert _params(anchor_xy) == {"weight": 0.7, "coefficient": -3.0}

    corrupted = factories.corrupted_xy_offset_factory(
        log_noise_std=0.2,
        soft_threshold=0.4,
    )
    assert _bindings(corrupted)["odom_yaw_cos_sin"] == "odom_yaw_cos_sin"
    assert _params(corrupted)["log_noise_std"] == 0.2

    power = factories.pow_rew_factory(
        weight=-2.0,
        min_value=None,
        use_torque_squared=True,
    )
    assert _bindings(power)["dof_forces"] == "current.dof_forces"
    assert "min_value" not in _params(power)
    assert _params(power)["use_torque_squared"] is True
    default_power = factories.pow_rew_factory(min_value=-0.25)
    assert _params(default_power)["min_value"] == -0.25

    contact_match = factories.contact_match_rew_factory(
        weight=-0.4,
        zero_during_grace_period=False,
    )
    assert _bindings(contact_match)["ref_contacts"] == "mimic.ref_state.rigid_body_contacts"
    assert _params(contact_match)["zero_during_grace_period"] is False

    force_change = factories.contact_force_change_rew_factory(
        min_value=None,
        threshold=5.0,
    )
    assert _bindings(force_change)["prev_contact_force_magnitudes"] == "prev_contact_force_magnitudes"
    assert "min_value" not in _params(force_change)
    default_force_change = factories.contact_force_change_rew_factory(min_value=-0.75)
    assert _params(default_force_change)["min_value"] == -0.75

    assert _bindings(factories.target_reward_factory())["tar_proximity_threshold"] == "target.tar_proximity_threshold"
    assert _bindings(factories.steering_reward_factory())["dt"] == "dt"
    assert _bindings(factories.path_following_reward_factory())["height_conditioned"] == "path.height_conditioned"

    for factory_fn in [
        factories.global_anchor_pos_rew_factory,
        factories.global_anchor_ori_rew_factory,
        factories.relative_body_pos_rew_factory,
        factories.relative_body_ori_rew_factory,
        factories.global_body_lin_vel_rew_factory,
        factories.global_body_ang_vel_rew_factory,
    ]:
        component = factory_fn(weight=0.9, sigma=1.3)
        assert _params(component) == {"weight": 0.9, "sigma": 1.3}


def test_termination_and_metric_factories_bind_metadata_and_wrappers():
    tracking_term = factories.tracking_error_term_factory(threshold=0.6)
    assert _bindings(tracking_term)["current_rigid_body_pos"] == "current.rigid_body_pos"
    assert _params(tracking_term)["threshold"] == 0.6
    assert "settle_steps" not in _params(tracking_term)
    assert (
        _params(factories.tracking_error_term_factory(settle_steps=25))["settle_steps"]
        == 25
    )

    fall = factories.fall_termination_factory(termination_height=0.2)
    assert _bindings(fall)["progress_buf"] == "progress_buf"
    assert _params(fall)["termination_height"] == 0.2

    assert _params(factories.anchor_pos_error_term_factory(threshold=0.1))["threshold"] == 0.1
    assert _params(factories.anchor_ori_error_term_factory(threshold=0.2))["threshold"] == 0.2
    assert _params(factories.relative_body_pos_error_term_factory(threshold=0.3))["threshold"] == 0.3
    assert _params(factories.anchor_height_error_term_factory(threshold=0.4))["threshold"] == 0.4

    no_threshold_metric = factories.gt_error_factory()
    assert _params(no_threshold_metric) == {}
    for factory_fn in [
        factories.gt_error_factory,
        factories.max_joint_error_factory,
        factories.gr_error_factory,
        factories.anchor_pos_metric_factory,
        factories.anchor_ori_metric_factory,
        factories.relative_body_pos_metric_factory,
        factories.anchor_height_error_metric_factory,
    ]:
        component = factory_fn(threshold=0.75)
        assert _params(component)["threshold"] == 0.75

    path_metric = factories.path_distance_error_factory(
        threshold=1.5,
        min_progress=3,
    )
    assert _bindings(path_metric)["target_pos"] == "path.tar_pos"
    assert _params(path_metric)["threshold"] == 0.5
    assert torch.equal(
        path_metric.get_compute_func()(
            head_pos=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            target_pos=torch.zeros(2, 3),
            fail_dist=1.5,
            progress_buf=torch.tensor([10, 10]),
            min_progress=3,
        ),
        torch.tensor([False, True]),
    )

    steering_metric = factories.steering_velocity_error_factory(
        speed_tolerance=0.5,
        direction_tolerance=0.7,
    )
    assert _bindings(steering_metric)["prev_root_pos"] == "steering.prev_root_pos"
    assert _params(steering_metric)["threshold"] == 0.5
    assert torch.equal(
        steering_metric.get_compute_func()(
            root_pos=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            prev_root_pos=torch.zeros(2, 3),
            tar_dir=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            tar_speed=torch.tensor([1.0, 1.0]),
            dt=1.0,
            speed_tolerance=0.5,
            direction_tolerance=0.7,
        ),
        torch.tensor([False, True]),
    )
