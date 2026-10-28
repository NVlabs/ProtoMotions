# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gaussian exploration contracts, including real multi-process CPU DDP."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel

from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import (
    PPOActorConfig,
    PPOAgentConfig,
    PPOModelConfig,
    StdMode,
)
from protomotions.agents.ppo.model import PPOActor, PPOModel
from protomotions.agents.ppo.utils.motion_index import MotionIndex


def make_actor(mode=StdMode.PER_MOTION):
    cfg = PPOActorConfig(
        mu_key="mu",
        in_keys=["obs"],
        num_out=2,
        num_motions=4,
        std_mode=mode,
        actor_logstd=-1.0,
        mu_model=MLPWithConcatConfig(
            in_keys=["obs"],
            out_keys=["mu"],
            num_out=2,
            layers=[MLPLayerConfig(units=8, activation="tanh")],
        ),
    )
    actor = PPOActor(cfg)
    actor.forward_deployable(TensorDict({"obs": torch.zeros(1, 3)}, batch_size=1))
    return actor


def sample(ids):
    return TensorDict(
        {"obs": torch.ones(len(ids), 3), "global_motion_id": torch.tensor(ids)},
        batch_size=len(ids),
    )


@pytest.mark.parametrize("mode", list(StdMode))
def test_distribution_and_deterministic_inference(mode):
    actor = make_actor(mode)
    td = sample([0, 1, 1])
    if mode != StdMode.PER_MOTION:
        del td["global_motion_id"]
    out = actor(td)
    assert out["action_std"].shape == (3, 2)
    assert actor.logstd.requires_grad == (mode != StdMode.FIXED)
    assert "global_motion_id" not in actor.in_keys
    assert "action_std" in actor.experience_buffer_keys()
    # Mean inference does not even read logstd, let alone need a training ID.
    with torch.no_grad():
        actor.logstd.fill_(float("nan"))
    mean = actor.forward_deployable(TensorDict({"obs": torch.ones(3, 3)}, batch_size=3))
    torch.testing.assert_close(mean["mean_action"], out["mean_action"])


def test_indexed_backward_accumulates_repeated_ids_and_entropy_updates():
    actor = make_actor()
    out = actor(sample([1, 1, 3]))
    loss = (
        -torch.distributions.Normal(out["mean_action"], out["action_std"])
        .entropy()
        .sum(-1)
        .mean()
    )
    loss.backward()
    torch.testing.assert_close(actor.logstd.grad[1], torch.full((2,), -2 / 3))
    torch.testing.assert_close(actor.logstd.grad[3], torch.full((2,), -1 / 3))
    assert torch.count_nonzero(actor.logstd.grad[[0, 2]]) == 0
    before = actor.logstd.detach().clone()
    torch.optim.Adam(actor.parameters(), lr=0.01).step()
    assert torch.all(actor.logstd[1] > before[1])


def test_integer_initial_value_still_creates_float_parameter():
    cfg = make_actor(StdMode.GLOBAL).config
    cfg.actor_logstd = -3
    assert PPOActor(cfg).logstd.is_floating_point()


def test_projection_can_move_inwards_and_does_not_clamp_fixed_std():
    actor = make_actor()
    with torch.no_grad():
        actor.logstd[0].fill_(100)
        actor.logstd[1].fill_(-100)
    actor.project_logstd()
    assert torch.all(actor.logstd[0] == actor.config.max_logstd)
    assert torch.all(actor.logstd[1] == actor.config.min_logstd)
    loss = actor.logstd[0].sum() - actor.logstd[1].sum()
    loss.backward()
    torch.optim.SGD(actor.parameters(), lr=0.01).step()
    actor.project_logstd()
    assert torch.all(actor.logstd[0] < actor.config.max_logstd)
    assert torch.all(actor.logstd[1] > actor.config.min_logstd)
    fixed = make_actor(StdMode.FIXED)
    with torch.no_grad():
        fixed.logstd.fill_(2)
    fixed.project_logstd()
    assert torch.all(fixed.logstd == 2)


def test_kl_uses_rollout_variance_not_current_variance_twice():
    agent = object.__new__(PPO)
    agent.fabric = SimpleNamespace(world_size=1)
    mu = torch.zeros(3, 2)
    assert agent._compute_kl(mu, mu, torch.ones_like(mu), torch.ones_like(mu) * 2) > 0
    assert agent._compute_kl(mu, mu, torch.ones_like(mu), torch.ones_like(mu)) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_forward_backward_and_projection():
    actor = make_actor().cuda()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    before = actor.logstd.detach().clone()
    for _ in range(3):
        td = sample([0, 0, 1, 3]).cuda()
        optimizer.zero_grad(set_to_none=True)
        output = actor(td)
        distribution = torch.distributions.Normal(
            output["mean_action"], output["action_std"]
        )
        loss = -distribution.log_prob(torch.zeros_like(output["mean_action"])).mean()
        loss -= 0.01 * distribution.entropy().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        actor.project_logstd()
        assert torch.isfinite(loss)
        assert torch.isfinite(actor.logstd.grad).all()
    assert not torch.equal(before, actor.logstd)
    assert (actor.logstd >= actor.config.min_logstd).all()
    assert (actor.logstd <= actor.config.max_logstd).all()


class Library:
    def __init__(self, path, names, pattern=None):
        self.motion_file = str(path)
        self.motion_files = tuple(names)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            motion_file=pattern or str(path), motion_file_shard_indices=[1, 0]
        )


def test_manifest_counts_unloaded_shards_and_maps_replicas(tmp_path):
    torch.save({"motion_files": ["walk", "run"]}, tmp_path / "motion_0.pt")
    torch.save({"motion_files": ["jump"]}, tmp_path / "motion_1.pt")
    lib = Library(
        tmp_path / "motion_0.pt", ["walk", "run"], str(tmp_path / "motion_slurmrank.pt")
    )
    index = MotionIndex.build(lib)
    assert index.num_motions == 3
    assert index.offset_for(lib) == 1  # Explicit configured shard order.
    assert index.offset_for(Library(tmp_path / "motion_1.pt", ["jump"])) == 0
    assert index.offset_for(deepcopy(lib)) == 1
    lib.motion_files = ("run", "walk")
    with pytest.raises(ValueError, match="ordering changed"):
        index.offset_for(lib)


def test_action_time_ids_are_copied_and_live_switch_rebinds(tmp_path):
    a = Library(tmp_path / "a.pt", ["a0", "a1"])
    b = Library(tmp_path / "b.pt", ["b0", "b1"])
    index = MotionIndex.from_sources(
        [
            (a.motion_file, None, a.motion_file, a.motion_files),
            (b.motion_file, None, b.motion_file, b.motion_files),
        ]
    )
    agent = object.__new__(PPO)
    agent.motion_index, agent._indexed_motion_lib = index, None
    agent.motion_lib = a
    agent.motion_manager = SimpleNamespace(motion_ids=torch.tensor([0, 1]))
    before_reset = agent.add_agent_info_to_obs({})["global_motion_id"]
    agent.motion_manager.motion_ids.fill_(1)
    assert before_reset.tolist() == [0, 1]
    agent.motion_lib = b
    assert agent.add_agent_info_to_obs({})["global_motion_id"].tolist() == [3, 3]


def make_agent(mode):
    actor = make_actor(mode)
    cfg = PPOModelConfig(
        actor=actor.config,
        critic=MLPWithConcatConfig(
            in_keys=["obs"], out_keys=["value"], num_out=1, layers=[]
        ),
    )
    model = PPOModel(cfg)
    model(sample([0, 1]))
    agent = object.__new__(PPO)
    agent.config = PPOAgentConfig(
        model=cfg, normalize_rewards=False, batch_size=4, training_max_steps=100
    )
    agent.model, agent.actor, agent.critic = model, model._actor, model._critic
    agent.motion_index = None
    agent.device = torch.device("cpu")
    agent.num_envs = 2
    agent.e_clip = agent.config.e_clip
    agent.current_epoch = 0
    agent.fabric = SimpleNamespace(world_size=1, global_rank=0)
    agent._setup_model_optimizer = lambda module, optimizer: (module, optimizer)
    return agent


def test_agent_loss_uses_selected_std_and_rollout_kl():
    agent = make_agent(StdMode.PER_MOTION)
    agent.config.adaptive_lr.enabled = True
    rollout = agent.actor(sample([0, 1])).detach().clone()
    batch = rollout.to_dict()
    batch["advantages"] = torch.zeros(2)
    with torch.no_grad():
        agent.actor.logstd[1].add_(0.1)
    loss, logs = agent.actor_step(batch)
    loss.backward()
    assert logs["actor/kl"] > 0
    assert torch.all(agent.actor.logstd.grad[:2] < 0)  # entropy only
    assert torch.count_nonzero(agent.actor.logstd.grad[2:]) == 0
    torch.testing.assert_close(batch["action_std"], rollout["action_std"])


def test_setup_counts_motions_only_for_training(tmp_path):
    agent = make_agent(StdMode.PER_MOTION)
    agent.config.model.actor.num_motions = None
    agent.motion_lib = Library(tmp_path / "motions.pt", ["a", "b", "c"])
    agent.training_setup = True
    agent._before_create_model()
    assert agent.config.model.actor.num_motions == 3
    assert agent.create_model()._actor.logstd.shape == (3, 2)
    agent.training_setup = False
    del agent.motion_lib
    agent._before_create_model()
    assert agent.motion_index is None
    assert agent.config.model.actor.num_motions == 3


def test_per_motion_actor_requires_setup_motion_count():
    config = PPOActorConfig(mu_key="mu", std_mode=StdMode.PER_MOTION)
    assert config.num_motions is None
    with pytest.raises(AssertionError, match="before actor construction"):
        PPOActor(config)


@pytest.mark.parametrize("mode", [StdMode.FIXED, StdMode.GLOBAL])
def test_shared_std_actor_needs_no_motion_count(mode):
    config = make_actor(mode).config
    config.num_motions = None
    assert PPOActor(config).logstd.shape == (2,)


def test_sidecar_matches_checkpoint_table(tmp_path, monkeypatch):
    from protomotions.agents.base_agent.agent import BaseAgent

    monkeypatch.setattr(BaseAgent, "save", lambda *args: None)
    agent = make_agent(StdMode.PER_MOTION)
    agent.motion_index = MotionIndex.build(
        Library(tmp_path / "motions.pt", list("abcd"))
    )
    agent.root_dir = tmp_path
    agent.step_count = 42
    agent.save()
    saved = torch.load(tmp_path / "motion_logstd.pt", weights_only=False)
    torch.testing.assert_close(saved["logstd"], agent.actor.logstd)
    assert saved["motion_index"] == agent.motion_index.state_dict()
    assert saved["step_count"] == 42


def test_unified_export_needs_no_motion_ids():
    from protomotions.utils.export_utils import UnifiedPipelineModule

    class Observations(torch.nn.Module):
        def get_output_keys(self):
            return ["obs"]

        def get_input_keys(self):
            return ["obs"]

        def forward(self, obs):
            return (obs,)

    class Actions(torch.nn.Module):
        def forward(self, action):
            return action, torch.ones_like(action), torch.ones_like(action)

    actor = make_actor()
    pipeline = UnifiedPipelineModule(Observations(), actor, Actions(), ["obs"])
    obs = torch.ones(2, 3)
    expected = actor.forward_deployable(TensorDict({"obs": obs}, batch_size=2))[
        "mean_action"
    ]
    actual = pipeline(obs)
    torch.testing.assert_close(actual[0], expected)


def test_warm_start_expands_shared_std_but_checks_manifest_before_loading(tmp_path):
    source = make_agent(StdMode.GLOBAL)
    target = make_agent(StdMode.PER_MOTION)
    lib = Library(tmp_path / "motions.pt", ["a", "b", "c", "d"])
    target.motion_index = MotionIndex.build(lib)
    checkpoint = {"model": deepcopy(source.model.state_dict())}
    target.load_parameters(checkpoint, load_training_state=False)
    assert checkpoint["model"]["_actor.logstd"].ndim == 1
    torch.testing.assert_close(target.actor.logstd, source.actor.logstd.expand(4, 2))
    obs = sample([0, 1])
    torch.testing.assert_close(
        source.actor(obs.clone())["mean_action"],
        target.actor(obs.clone())["mean_action"],
    )
    bad = {"model": deepcopy(target.model.state_dict()), "motion_index": []}
    before = deepcopy(target.model.state_dict())
    with pytest.raises(ValueError, match="Motion index differs"):
        target.load_parameters(bad, load_training_state=False)
    for key, tensor in before.items():
        torch.testing.assert_close(target.model.state_dict()[key], tensor)


def test_rollout_std_snapshot_survives_updates_and_shuffle():
    actor = make_actor()
    rollout = actor(sample([3, 0, 1])).detach().clone()
    saved_std = rollout["action_std"].clone()
    with torch.no_grad():
        actor.logstd[3].add_(0.2)
    order = torch.tensor([2, 0, 1])
    batch = rollout[order]
    new = actor(batch.clone())
    torch.testing.assert_close(batch["action_std"], saved_std[order])
    assert new["action_std"][1, 0] > batch["action_std"][1, 0]
    assert torch.equal(batch["global_motion_id"], torch.tensor([1, 3, 0]))


def test_adam_resume_next_update_is_identical(tmp_path):
    actor = make_actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    ids, obs, actions = torch.tensor([0, 1, 1]), torch.ones(3, 3), torch.zeros(3, 2)
    _optimization_loss(actor, ids, obs, actions).backward()
    optimizer.step()
    torch.save(
        {"model": actor.state_dict(), "optimizer": optimizer.state_dict()},
        tmp_path / "last.ckpt",
    )
    loaded = torch.load(tmp_path / "last.ckpt", weights_only=False)
    resumed = make_actor()
    resumed.load_state_dict(loaded["model"])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    for model, opt in [(actor, optimizer), (resumed, resumed_optimizer)]:
        opt.zero_grad(set_to_none=True)
        _optimization_loss(model, ids, obs, actions).backward()
        opt.step()
    for p, q in zip(actor.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_full_agent_checkpoint_restores_table_optimizer_and_progress(tmp_path):
    from protomotions.components.motion_lib import MotionFileSwitchMode

    agents = [make_agent(StdMode.PER_MOTION), make_agent(StdMode.PER_MOTION)]
    for agent in agents:
        agent.motion_lib = Library(tmp_path / "motions.pt", list("abcd"))
        agent.motion_lib.config.motion_file_switch_mode = MotionFileSwitchMode.FIXED
        agent.motion_index = MotionIndex.build(agent.motion_lib)
        agent.config.advantage_normalization.use_ema = False
        agent.evaluator = None
        agent.step_count = 17
        agent.fit_start_time = 0.0
        agent.best_evaluated_score = 0.8
        agent.create_optimizers(agent.model)
    source, target = agents
    loss = source.actor(sample([0, 2]))["action_std"].sum()
    loss.backward()
    source.actor_optimizer.step()
    source.current_epoch = 9
    torch.save(source.get_state_dict({}), tmp_path / "last.ckpt")
    target.load_parameters(torch.load(tmp_path / "last.ckpt", weights_only=False))
    assert target.current_epoch == 9
    assert target.step_count == 17
    assert target.best_evaluated_score == 0.8
    torch.testing.assert_close(target.actor.logstd, source.actor.logstd)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            target.actor_optimizer.state[target.actor.logstd][key],
            source.actor_optimizer.state[source.actor.logstd][key],
        )


def _optimization_loss(actor, ids, observations, actions):
    td = TensorDict({"obs": observations, "global_motion_id": ids}, batch_size=len(ids))
    out = actor(td)
    distribution = torch.distributions.Normal(out["mean_action"], out["action_std"])
    return (
        -distribution.log_prob(actions).sum(-1).mean()
        - 0.01 * distribution.entropy().sum(-1).mean()
    )


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        library = Library(f"/motions/group_{rank}.pt", [f"{rank}/a", f"{rank}/b"])
        index = MotionIndex.build(library)
        assert index.num_motions == 4
        assert index.offset_for(library) == rank * 2
        torch.manual_seed(7)
        actor = make_actor()
        baseline = deepcopy(actor)
        ddp = DistributedDataParallel(actor)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=0.001)
        baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.001)
        # Shared motion 1, disjoint 0/2, unvisited 3. Different sample counts.
        ids = torch.tensor([0, 1, 1, 1, 2, 2])
        obs = torch.arange(18).reshape(6, 3).float() / 18
        actions = torch.arange(12).reshape(6, 2).float() / 12
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            baseline_optimizer.zero_grad(set_to_none=True)
            sl = slice(rank * 3, (rank + 1) * 3)
            _optimization_loss(ddp, ids[sl], obs[sl], actions[sl]).backward()
            _optimization_loss(baseline, ids, obs, actions).backward()
            for p, q in zip(actor.parameters(), baseline.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            optimizer.step()
            baseline_optimizer.step()
            actor.project_logstd()
            baseline.project_logstd()
            for p, q in zip(actor.parameters(), baseline.parameters()):
                torch.testing.assert_close(p, q)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(
                        optimizer.state[p][key], baseline_optimizer.state[q][key]
                    )
    finally:
        dist.destroy_process_group()


def test_real_ddp_matches_combined_batch_with_adam_and_clipping(tmp_path):
    mp.spawn(_ddp_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2)
