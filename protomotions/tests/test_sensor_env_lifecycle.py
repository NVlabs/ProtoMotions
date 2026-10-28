# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor lifecycle contracts through BaseEnv, with an independent CPU simulator."""

from dataclasses import fields
from unittest.mock import patch

import pytest
import torch

from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    LocalizationDrift,
    LocalizationSensor,
    SensorConfig,
)

pytest.importorskip("mujoco")


@pytest.fixture
def env():
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    sensors = SensorConfig(
        joint_encoders=JointEncoderSensor(0.04, 0.1),
        imu=IMUSensor(0.02, 0.05),
        localization=LocalizationSensor(
            body_name="pelvis",
            noise=0.03,
            drift=LocalizationDrift(scale=0.1, yaw=0.1, log_magnitude=0.02),
        ),
    )
    environment = make_sensor_test_env(sensors=sensors)
    yield environment
    environment.close()


def test_step_and_reset_reconstruct_once_and_context_reads_do_not_sample(env):
    with patch.object(
        env.sensors.fk, "body_kinematics", wraps=env.sensors.fk.body_kinematics
    ) as fk:
        env.step(torch.full((3, 2), 0.2))
        assert fk.call_count == 1
        assert fk.call_args.args[0].shape[0] == 2 * env.num_envs
        rng = env.sensors.generator.get_state().clone()
        env.observe(torch.arange(env.num_envs))
        _ = env.context["clean.dof_pos"]
        _ = env.context["noisy_historical.dof_pos"]
        torch.testing.assert_close(env.sensors.generator.get_state(), rng)
        assert fk.call_count == 1

        env.reset(torch.tensor([2, 0]))
        assert fk.call_count == 2
        assert fk.call_args.args[0].shape[0] == 4


def test_step_feeds_the_noisy_estimate_to_current_and_history(env):
    # Actor observations read EnvContext.noisy and noisy_historical. A step that
    # stored the clean estimate under "noisy", or a history that stopped
    # shifting the noisy source, keeps every shape valid, so the only symptom
    # would be a policy training on clean data while configured for noise.
    recorded = []
    sensor_step = env.sensors.step

    def recording_step(state):
        estimates = sensor_step(state)
        recorded.append(estimates)
        return estimates

    env.sensors.step = recording_step
    history_steps = env._state_history.num_history_steps
    for command in range(history_steps + 1):
        env.step(torch.full((env.num_envs, 2), 0.1 * (command + 1)))

    latest = recorded[-1]
    assert not torch.allclose(latest.noisy.dof_pos, latest.clean.dof_pos)
    torch.testing.assert_close(env.context["noisy.dof_pos"], latest.noisy.dof_pos)
    for k in range(1, history_steps + 1):
        torch.testing.assert_close(
            env.context["noisy_historical.dof_pos"][:, k - 1],
            recorded[-1 - k].noisy.dof_pos,
        )


def test_partial_reset_preserves_other_rows_and_seeds_noisy_history(env):
    env.step(torch.full((3, 2), 0.2))
    before = dict(env._state_history.capture().tensors())
    retained_context = env.context
    retained_dofs = retained_context["noisy.dof_pos"].clone()
    episode_before = env.sensors.save_state().episode
    ids = torch.tensor([2, 0])
    env.reset(ids)
    after = env._state_history.capture()
    for name, value in after.tensors():
        torch.testing.assert_close(value[1], before[name][1], msg=name)
    for field in fields(episode_before):
        torch.testing.assert_close(
            getattr(env.sensors.episode, field.name)[1],
            getattr(episode_before, field.name)[1],
            msg=field.name,
        )
    noisy = after.states["noisy"].dof_pos[ids]
    torch.testing.assert_close(noisy, noisy[:, :1].expand_as(noisy))
    assert not torch.equal(noisy, after.states["clean"].dof_pos[ids])
    assert torch.count_nonzero(after.raw_actions[ids]) == 0
    assert torch.count_nonzero(after.processed_actions[ids]) == 0
    torch.testing.assert_close(retained_context["noisy.dof_pos"], retained_dofs)


def test_restore_preserves_history_and_reproduces_next_sensor_draw(env):
    actions = torch.full((3, 2), 0.2)
    env.step(actions)
    snapshot = env.save_state()
    expected = env.sensors.step(env.simulator.get_robot_state())
    env.reset(torch.tensor([1]))
    env.step(-actions)

    with patch.object(env.simulator, "step", wraps=env.simulator.step) as step:
        with patch.object(
            env.sensors.fk, "body_kinematics", wraps=env.sensors.fk.body_kinematics
        ) as fk:
            env.restore_state(snapshot)
            assert step.call_count == 0
            assert fk.call_count == 0
    restored = dict(env._state_history.capture().tensors())
    for name, value in snapshot["history"].tensors():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(
        env.sensors.generator.get_state(), snapshot["sensors"].rng
    )
    actual = env.sensors.step(env.simulator.get_robot_state())
    for source in ("clean", "noisy"):
        expected_values = dict(getattr(expected, source).tensors())
        for name, value in getattr(actual, source).tensors():
            torch.testing.assert_close(
                value, expected_values[name], msg=f"{source}.{name}"
            )


def test_previous_raw_action_and_state_history_have_distinct_time_alignment(env):
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_actions_from_state,
    )
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import compute_action_rate_l2

    for command in (0.1, 0.2, 0.3):
        env.step(torch.full((3, 2), command))
    ctx = env.context
    torch.testing.assert_close(ctx["current_action"], torch.full((3, 2), 0.3))
    torch.testing.assert_close(ctx["previous_action"], torch.full((3, 2), 0.2))
    torch.testing.assert_close(
        ctx["noisy_historical.raw_actions"][:, 0], torch.full((3, 2), 0.2)
    )
    torch.testing.assert_close(
        ctx["raw_action_history"][:, :3, 0],
        torch.tensor([[0.3, 0.2, 0.1]]).expand(3, -1),
    )
    # Reward compares consecutive raw commands; policy feedback starts with
    # the command that produced the current state, not the preceding one.
    rate = MdpComponent.from_kernel(
        compute_action_rate_l2, meta=RewardMeta(weight=-0.1)
    )
    torch.testing.assert_close(rate.compute(ctx), torch.full((3,), 2 * 0.1**2))
    previous = MdpComponent.from_kernel(
        compute_historical_actions_from_state, history_steps=1
    )
    torch.testing.assert_close(previous.compute(ctx), torch.full((3, 2), 0.3))
