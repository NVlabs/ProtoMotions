# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseEnv's perturbation switch and context build count, on the CPU test double."""

from unittest.mock import patch

import pytest
import torch

pytest.importorskip("mujoco")

from protomotions.simulator.base_simulator.config import RobotNoiseConfig  # noqa: E402
from protomotions.tests.support.sensor_simulator import (  # noqa: E402
    SensorTestSimulator,
    make_sensor_test_env,
)


@pytest.fixture
def noisy_env():
    env = make_sensor_test_env(reset_noise=RobotNoiseConfig(dof_pos_noise=0.1))
    yield env
    env.close()


def test_step_builds_exactly_one_context_covering_every_env():
    env = make_sensor_test_env()
    try:
        env.reset()
        events = []
        publish = env._publish
        simulator_step = env.simulator.step

        def logged_build(env_ids=None):
            events.append(("build", env_ids))
            return publish(env_ids)

        def logged_step(*args, **kwargs):
            events.append(("physics", None))
            return simulator_step(*args, **kwargs)

        with patch.object(env, "_publish", logged_build):
            with patch.object(env.simulator, "step", logged_step):
                env.step(torch.full((3, 2), 0.2))
        # One context per step, built after physics and covering every env;
        # observations, rewards and terminations all read that one.
        assert [kind for kind, _ in events] == ["physics", "build"]
        assert events[1][1] is None
    finally:
        env.close()


def _reset_dof_pos(env, env_ids):
    env.reset(env_ids)
    return env.simulator.get_robot_state(env_ids).dof_pos.clone()


def test_perturbation_switch_gates_reset_noise_and_its_random_draws(noisy_env):
    env = noisy_env
    env_ids = torch.tensor([1])
    default_dof_pos = env._pose.default_reset_state[env_ids].dof_pos.clone()
    assert env.perturbations_enabled is True

    env.set_perturbations_enabled(False)
    assert env.perturbations_enabled is False
    torch.manual_seed(5)
    rng = torch.get_rng_state().clone()
    stage = env.randomness.generator("reset_noise")
    stage_before = stage.get_state().clone()
    first = _reset_dof_pos(env, env_ids)
    # No noise was applied; neither the reset-noise stage nor the global
    # generator moved.
    assert torch.equal(first, default_dof_pos)
    assert torch.equal(stage.get_state(), stage_before)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.manual_seed(5)
    assert torch.equal(_reset_dof_pos(env, env_ids), first)

    env.set_perturbations_enabled(True)
    assert env.perturbations_enabled is True
    torch.manual_seed(5)
    noisy = _reset_dof_pos(env, env_ids)
    assert not torch.equal(noisy, default_dof_pos)
    assert torch.all((noisy - default_dof_pos).abs() <= 0.1 + 1e-6)
    # The noise came from the reset-noise stage's generator (C6), which
    # moved; the global torch stream did not.
    assert not torch.equal(stage.get_state(), stage_before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_perturbation_switch_only_re_enables_pushes_the_simulator_has(noisy_env):
    env = noisy_env
    # The test double has no push randomization, so the env switch must not
    # try to turn pushes on when perturbations come back.
    assert env.simulator.push_enabled is False
    env.set_perturbations_enabled(False)
    env.set_perturbations_enabled(True)
    assert env.perturbations_enabled is True
    assert env.simulator.push_enabled is False


def test_fake_simulator_refuses_to_enable_pushes():
    sim = SensorTestSimulator(2)
    assert sim.push_enabled is False
    sim.set_push_enabled(False)
    with pytest.raises(ValueError, match="no push randomization"):
        sim.set_push_enabled(True)
    assert sim.push_enabled is False
