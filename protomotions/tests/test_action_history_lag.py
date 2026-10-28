# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin what the action-history observations hold in a real ``BaseEnv``.

The exporter tells deployments that history index ``j`` of each
action-history input holds the action from ``steps_back[j]`` policy calls
ago, and that a reset fills it with ``ACTION_HISTORY_RESET_FILL`` (both in
``deployment/tracker_inputs.py``).  That timing comes from the order of
operations in ``BaseEnv.step``, not from the history buffer alone, so this
steps a real environment: SMPL on the CPU MuJoCo backend, with numbered
actions, and checks which call's actions come back.  It covers the two
current inputs and the older ``historical.processed_actions`` binding that
released trackers were trained with.  If training changes any of these
facts, this fails until the constants, and with them every exported YAML,
are updated to match.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from deployment import tracker_inputs  # noqa: E402
from protomotions.components.motion_lib import MotionLib  # noqa: E402
from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.envs.action import make_bm_pd_action_config  # noqa: E402
from protomotions.envs.base_env.config import EnvConfig  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.context_views import EnvContext  # noqa: E402
from protomotions.envs.mdp_component import MdpComponent  # noqa: E402
from protomotions.envs.obs.humanoid_historical import (  # noqa: E402
    compute_historical_actions_from_state,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.base_simulator.utils import (  # noqa: E402
    convert_friction_for_simulator,
)
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402

HISTORY_SLOTS = 3
PROCESSED = "processed_action_history"
RAW = "raw_action_history"
LEGACY = tracker_inputs.LEGACY_ACTION_HISTORY_KEY

# Observation name -> the action-history key it reads.
OBSERVATIONS = {
    "processed": PROCESSED,
    "raw": RAW,
    "legacy_processed": LEGACY,
}


@pytest.fixture(scope="module")
def env():
    device = torch.device("cpu")
    robot_cfg = robot_config("smpl")
    sim_cfg = simulator_config(
        "mujoco",
        robot_cfg,
        headless=True,
        num_envs=1,
        experiment_name="action_history_lag",
    )
    action_config = make_bm_pd_action_config(robot_cfg)
    # SMPL's default pose is all zeros, which would make a zero fill and a
    # default-pose fill look the same.  A distinctive offset tells them apart.
    num_dofs = action_config.params["pd_action_offset"].numel()
    action_config.params["pd_action_offset"] = torch.linspace(-0.05, 0.05, num_dofs)
    env_cfg = EnvConfig(
        max_episode_length=1000,
        observation_components={
            "processed": MdpComponent.from_kernel(
                compute_historical_actions_from_state,
                history_steps=HISTORY_SLOTS,
                historical_actions=EnvContext.processed_action_history,
            ),
            "raw": MdpComponent.from_kernel(
                compute_historical_actions_from_state, history_steps=HISTORY_SLOTS
            ),
            # How configs bound the processed history before the current
            # inputs existed; released trackers were trained with this.
            "legacy_processed": MdpComponent(
                compute_func=compute_historical_actions_from_state,
                dynamic_vars={
                    "historical_actions": EnvContext.historical.processed_actions
                },
                static_params={"history_steps": HISTORY_SLOTS},
            ),
        },
        action_config=action_config,
    )
    terrain_cfg, sim_cfg = convert_friction_for_simulator(TerrainConfig(), sim_cfg)
    terrain = Terrain(config=terrain_cfg, num_envs=1, device=device)
    scene_lib = SceneLib.empty(num_envs=1, device=device, terrain=terrain)
    simulator = get_class(sim_cfg._target_)(
        config=sim_cfg,
        robot_config=robot_cfg,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
    )
    return BaseEnv(
        config=env_cfg,
        robot_config=robot_cfg,
        device=device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=MotionLib.empty(device=device),
        simulator=simulator,
    )


def _fill(env, key: str) -> torch.Tensor:
    offset = env.config.action_config.params["pd_action_offset"].float()
    spec = tracker_inputs.action_history_fill(key, offset.numpy())
    return torch.as_tensor(tracker_inputs.resolve_fill(spec, offset.numel()))


def _run(env, num_calls: int):
    """Reset, then step with numbered actions.

    Returns, for each observation and each decision d, what the policy would
    choose action d from; and, by graph output name, what each action
    produced.
    """
    config = env.config.action_config
    offset = config.params["pd_action_offset"].float()
    scale = config.params["action_scale"].float()
    env_ids = torch.arange(env.num_envs)
    obs, _ = env.reset(env_ids)
    observed = {
        name: [obs[name].reshape(HISTORY_SLOTS, -1).clone()] for name in OBSERVATIONS
    }
    outputs = {"actions": [], "joint_pos_targets": []}
    for call in range(num_calls):
        action = torch.full((1, offset.numel()), 0.01 * (call + 1))
        outputs["actions"].append(action[0].clone())
        outputs["joint_pos_targets"].append(offset + scale * action[0])
        obs, _, reset_buf, _, _ = env.step(action)
        assert not bool(reset_buf.any()), "the episode ended mid-test"
        for name in OBSERVATIONS:
            observed[name].append(obs[name].reshape(HISTORY_SLOTS, -1).clone())
    return observed, outputs


@pytest.mark.parametrize("name", sorted(OBSERVATIONS))
def test_observation_holds_outputs_from_the_declared_calls_back(env, name):
    key = OBSERVATIONS[name]
    observed, outputs = _run(env, num_calls=7)
    produced = outputs[tracker_inputs.action_history_output(key)]
    steps_back = tracker_inputs.action_history_steps_back(key, HISTORY_SLOTS)
    fill = _fill(env, key)
    for decision, obs in enumerate(observed[name]):
        for slot, back in enumerate(steps_back):
            source = decision - back
            want = produced[source] if source >= 0 else fill
            torch.testing.assert_close(
                obs[slot], want, msg=f"{name}: decision {decision}, slot {slot}"
            )


@pytest.mark.parametrize("name", sorted(OBSERVATIONS))
def test_a_reset_restores_the_declared_fill(env, name):
    key = OBSERVATIONS[name]
    _run(env, num_calls=5)
    observed, outputs = _run(env, num_calls=3)
    produced = outputs[tracker_inputs.action_history_output(key)]
    first = tracker_inputs.action_history_steps_back(key, 1)[0]
    fill = _fill(env, key)
    for decision in range(first):
        torch.testing.assert_close(observed[name][decision][0], fill)
    torch.testing.assert_close(observed[name][first][0], produced[0])


def test_the_lags_are_told_apart(env):
    """Guard against a vacuous pass: neighbouring calls must not match."""
    observed, outputs = _run(env, num_calls=4)
    targets = outputs["joint_pos_targets"]
    # One call back is right for the current input, wrong for the legacy one.
    assert np.allclose(observed["processed"][3][0].numpy(), targets[2].numpy())
    assert not np.allclose(
        observed["legacy_processed"][3][0].numpy(), targets[2].numpy()
    )
    assert not np.allclose(observed["processed"][3][0].numpy(), targets[1].numpy())
    # Raw and processed histories hold different values.
    assert not np.allclose(
        observed["raw"][3][0].numpy(), observed["processed"][3][0].numpy()
    )


def test_the_ring_states_the_slot_facts_the_deployment_contract_uses():
    """The history ring's own two slot facts are the ones the exporter tells
    deployments: the first slot of the two current action-history inputs is
    one policy call back (ACTION_LAG), and a reset fills the rings with zeros
    (RESET_FILL); the env test above measures both on a real step."""
    assert StateHistoryBuffer.RESET_FILL == tracker_inputs.ACTION_HISTORY_RESET_FILL
    for key in (PROCESSED, RAW):
        first = tracker_inputs.action_history_steps_back(key, 1)[0]
        assert first == StateHistoryBuffer.ACTION_LAG, key
