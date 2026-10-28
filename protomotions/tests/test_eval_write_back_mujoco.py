# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-MuJoCo round trip of the write-back that follows every evaluation.

After evaluation the trainer continues the interrupted episodes, so the first
training step must see exactly what it would have seen had evaluation never
run. Uses the SMPL humanoid and the example motion in ``examples/data``, both
plain git files, so this runs on the CPU CI job without LFS.
"""

from pathlib import Path

import pytest
import torch

mujoco = pytest.importorskip("mujoco")

import protomotions  # noqa: E402
import protomotions.envs.component_manager as component_manager  # noqa: E402
from protomotions.components.motion_lib import MotionLib, MotionLibConfig  # noqa: E402
from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.envs.base_env.config import EnvConfig  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.mdp_component import MdpComponent  # noqa: E402
from protomotions.envs.obs.humanoid import (  # noqa: E402
    compute_humanoid_max_coords_observations,
)
from protomotions.envs.obs.humanoid_historical import (  # noqa: E402
    compute_historical_actions_from_state,
)
from protomotions.envs.obs.target_poses import (  # noqa: E402
    build_max_coords_target_poses,
)
from protomotions.envs.rewards.meta import RewardMeta  # noqa: E402
from protomotions.envs.rewards.regularization import (  # noqa: E402
    compute_action_smoothness,
    compute_contact_force_change_rew,
)
from protomotions.envs.control.masked_mimic_control import (  # noqa: E402
    MaskedMimicControlConfig,
)
from protomotions.envs.control.mimic_control import MimicControlConfig  # noqa: E402
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.base_simulator.config import (  # noqa: E402
    DomainRandomizationConfig,
    LatencyDomainRandomizationConfig,
)
from protomotions.simulator.base_simulator.utils import (  # noqa: E402
    convert_friction_for_simulator,
)
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig  # noqa: E402
from protomotions.simulator.mujoco.simulator import MujocoSimulator  # noqa: E402

REPO_ROOT = Path(protomotions.__file__).resolve().parents[1]
MOTION_FILE = REPO_ROOT / "examples/data/smpl_humanoid_sit_armchair.motion"

SIM_RUNTIME = (
    "_common_actions",
    "_previous_actions",
    "_prev_prev_actions",
    "_steps_since_reset",
    "_sampled_action_latency_ms",
    "_action_latency_elapsed_ms",
)


@pytest.fixture(params=["mimic", "masked_mimic"])
def env(request, monkeypatch):
    # Eager kernels: compiling the MDP components dominates build time on CPU.
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", False)
    torch.manual_seed(0)
    device = torch.device("cpu")
    rc = robot_config("smpl")
    # Recorded any-partner labels require complete contact coverage.
    rc.update_fields(contact_bodies="all", contact_partners="all")
    sim_cfg = MujocoSimulatorConfig(
        num_envs=1,
        headless=True,
        experiment_name="test_eval_write_back",
        sim=rc.simulation_params.mujoco,
        # Both read the simulator's action history and steps-since-reset.
        pd_target_max_accel=0.05,
        domain_randomization=DomainRandomizationConfig(
            latency=LatencyDomainRandomizationConfig(max_latency_ms=10.0)
        ),
    )
    terrain_cfg = TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        border_size=40.0,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0] * 7 + [1.0],
    )
    terrain_cfg, sim_cfg = convert_friction_for_simulator(terrain_cfg, sim_cfg)
    terrain = Terrain(config=terrain_cfg, num_envs=1, device=device)
    scene_lib = SceneLib.empty(num_envs=1, device=device, terrain=terrain)
    motion_lib = MotionLib(MotionLibConfig(motion_file=str(MOTION_FILE)), device=device)
    if request.param == "mimic":
        control = {"mimic": MimicControlConfig(bootstrap_on_episode_end=True)}
    else:
        control = {
            "masked_mimic": MaskedMimicControlConfig(
                num_masked_future_steps=5, bootstrap_on_episode_end=True
            )
        }
    env_cfg = EnvConfig(
        max_episode_length=300,
        control_components=control,
        observation_components={
            "max_coords_obs": MdpComponent.from_kernel(
                compute_humanoid_max_coords_observations
            ),
            "previous_actions": MdpComponent.from_kernel(
                compute_historical_actions_from_state, history_steps=1
            ),
            "mimic_target_poses": MdpComponent.from_kernel(
                build_max_coords_target_poses, with_velocities=True
            ),
        },
        reward_components={
            "action_smoothness": MdpComponent.from_kernel(
                compute_action_smoothness, meta=RewardMeta(weight=-0.02)
            ),
            # Reads last step's contact forces.
            "contact_force_change": MdpComponent.from_kernel(
                compute_contact_force_change_rew,
                meta=RewardMeta(weight=-1e-3, zero_during_grace_period=False),
                threshold=0.0,
            ),
        },
        motion_manager=MimicMotionManagerConfig(init_start_prob=0.5),
    )
    sim = MujocoSimulator(
        config=sim_cfg,
        robot_config=rc,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
    )
    env = BaseEnv(
        config=env_cfg,
        robot_config=rc,
        device=device,
        simulator=sim,
        motion_lib=motion_lib,
        terrain=terrain,
        scene_lib=scene_lib,
    )
    yield env
    env.close()


def _actions(n, scale, seed, num_actions):
    gen = torch.Generator().manual_seed(seed)
    return [scale * torch.randn(1, num_actions, generator=gen) for _ in range(n)]


def _evaluate(env, actions):
    """What MimicEvaluator does to the shared env between save and restore."""
    env.reset(torch.arange(env.num_envs))
    for action in actions:
        # Simulator.step keeps the caller's tensor and later writes into it.
        env.step(action.clone())


def _write_back(env, snapshot, motion_state):
    """What MimicEvaluator.cleanup_after_evaluation does."""
    env.reference_motion.motion_manager.set_assignment(*motion_state)
    env.restore_state(snapshot)


def _control_tensors(env):
    return {
        (name, key): value.clone()
        for name, component in env.control_manager.components.items()
        for key, value in vars(component).items()
        if torch.is_tensor(value)
    }


def test_write_back_restores_what_the_first_post_eval_step_reads(env):
    n = env.robot_config.number_of_actions
    env.reset()
    for action in _actions(6, 0.3, 1, n):
        env.step(action.clone())

    snapshot = env.save_state()
    motion_state = (
        env.reference_motion.motion_manager.motion_ids.clone(),
        env.reference_motion.motion_manager.motion_times.clone(),
    )
    obs_before = env.get_obs()
    sim_before = {k: getattr(env.simulator, k).clone() for k in SIM_RUNTIME}
    env_tensors = {
        "prev_contact_force_magnitudes": env._contact_memory.prev_contact_force_magnitudes,
        "raw_action": env._action_stage.raw,
        "processed_action": env._action_stage.processed,
    }
    env_before = {k: tensor.clone() for k, tensor in env_tensors.items()}
    control_before = _control_tensors(env)

    _evaluate(env, _actions(15, 1.5, 2, n))
    _write_back(env, snapshot, motion_state)

    obs_after = env.get_obs()
    assert obs_after.keys() == obs_before.keys()
    for key, value in obs_before.items():
        assert torch.equal(obs_after[key], value), key
    for key, value in sim_before.items():
        assert torch.equal(getattr(env.simulator, key), value), key
    for key, value in env_before.items():
        assert torch.equal(env_tensors[key], value), key
    control_after = _control_tensors(env)
    assert control_after.keys() == control_before.keys()
    for key, value in control_before.items():
        assert torch.equal(control_after[key], value), key


def test_training_resumes_after_evaluation_as_if_never_interrupted(env):
    n = env.robot_config.number_of_actions
    env.reset()
    for action in _actions(6, 0.3, 1, n):
        env.step(action.clone())

    # After mj_step MuJoCo's body poses still describe the state before the
    # last integration, and get_robot_state reads the root pose from them.
    # Refresh so the snapshot and the live state agree.
    mujoco.mj_forward(env.simulator.model, env.simulator.data)
    snapshot = env.save_state()
    motion_state = (
        env.reference_motion.motion_manager.motion_ids.clone(),
        env.reference_motion.motion_manager.motion_times.clone(),
    )
    rng = torch.get_rng_state()
    continuation = _actions(4, 0.3, 3, n)

    def continue_training():
        # Latency (and MaskedMimic's masks) draw from the global generator.
        torch.set_rng_state(rng)
        out = []
        for action in continuation:
            # BaseAgent.fit: no env was done before evaluation.
            obs, _ = env.reset(torch.zeros(0, dtype=torch.long))
            next_obs, reward, done, _, _ = env.step(action.clone())
            assert not done.any()
            out.append((obs, next_obs, reward.clone()))
        return out

    uninterrupted = continue_training()
    _evaluate(env, _actions(15, 1.5, 2, n))
    _write_back(env, snapshot, motion_state)
    resumed = continue_training()

    for i, (expected, actual) in enumerate(zip(uninterrupted, resumed)):
        torch.testing.assert_close(
            actual[2], expected[2], atol=1e-4, rtol=0, msg=f"reward, step {i}"
        )
        for which in (0, 1):
            for key in expected[which]:
                torch.testing.assert_close(
                    actual[which][key],
                    expected[which][key],
                    atol=1e-3,
                    rtol=0,
                    msg=f"{key}, step {i}",
                )
