# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The RLFT reference policy must survive adapter checkpoints.

Prior-constrained sampling restricts the fine-tuned policy to tokens inside a
reference policy's top-p set, and the KL penalty compares against the same
reference. These tests run the real agent ``setup()`` and ``load()`` on a tiny
prior, the way ``train_agent.py`` and ``inference_agent.py`` call them, and
check that loading a checkpoint restores the reference the run trained with.

The central check is parity: two processes with the same seed, one loading
``last.ckpt`` and one loading the adapter file, must sample identically, for plain
RLFT, PEFT+AMP and warm starts from SFT. Adapter files that store no reference
(SFT files, and RLFT files from older code) cannot match ``last.ckpt`` in
prior-constrained mode, so RLFT inference loading must fail before copying
weights. The SFT agent never samples against the reference, so it loads them.

A saved reference serves resume and inference. A new training run ignores it
and pins the policy it starts from, or the trained policy saved in the file
that ``reference_checkpoint`` names.

The reference keeps its own frozen copies of the FiLM input normalizer and the
conditioning network, so it stays fixed while the student's keep training.
Files saved before the reference had its own conditioning network are refused
until ``scripts/migrate_peft_reference_conditioning.py`` has copied the
student's network saved with them under the reference prefix.

Every test runs once with DoRA adapters and a FiLM input normalizer, and once
with LoRA adapters and no normalizer.
"""

import copy
import functools
import importlib.util
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.amp.config import DiscriminatorConfig
from protomotions.agents.common.autoregressive import DiscreteAutoregressiveTransformer
from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPLayerConfig,
    MLPWithConcatConfig,
    ModuleContainerConfig,
    ModuleOperationForwardConfig,
    ObsProcessorConfig,
    PretrainedModelConfig,
)
from protomotions.agents.common.discrete_latent import (
    DiscreteLatentDecoder,
    DiscreteLatentTargetEncoder,
    FSQTokenization,
)
from protomotions.agents.peft.model import DiscretePriorPEFTModel
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent
from protomotions.agents.peft.prior_amp_agent import DiscretePriorPEFTRLFTAMPAgent
from protomotions.agents.peft.prior_amp_config import (
    DiscretePriorPEFTRLFTAMPAgentConfig,
    DiscretePriorPEFTRLFTAMPModelConfig,
)
from protomotions.agents.peft.prior_amp_model import DiscretePriorPEFTRLFTAMPModel
from protomotions.agents.peft.prior_config import (
    DEFAULT_PEFT_CONDITION_KEY,
    DiscretePriorPEFTActorConfig,
    DiscretePriorPEFTConfig,
    DiscretePriorPEFTRLFTAgentConfig,
    DiscretePriorPEFTRLFTModelConfig,
    DiscretePriorPEFTSFTAgentConfig,
    DiscretePriorPEFTSFTModelConfig,
)
from protomotions.agents.peft.sft_agent import DiscretePriorPEFTSFTAgent
from protomotions.agents.peft.utils.adapter_state import (
    MIGRATION_SCRIPT,
    is_adapter_state_key,
)
from protomotions.agents.peft.utils.model_state import (
    optional_full_checkpoint_state_prefixes,
)
from protomotions.components.motion_lib import MotionFileSwitchMode
from protomotions.utils.config_utils import apply_overrides, parse_cli_overrides
from protomotions.utils.experiment_configs import ExperimentConfigs

NUM_ENVS = 32
VOCAB = 9
REFERENCE_PREFIX = "prior_with_peft.reference_"
ADAPTER_VARIANTS = {
    "dora_film_norm": {"peft_type": "dora", "film_input_norm": True},
    "lora_no_film_norm": {"peft_type": "lora", "film_input_norm": False},
}
SAMPLING_MODES = ["prior_constraint", "nucleus"]
AGENT_CLASSES = {
    "rlft": DiscretePriorPEFTRLFTAgent,
    "peft_amp": DiscretePriorPEFTRLFTAMPAgent,
}
AGENT_CONFIG_CLASSES = {
    "rlft": DiscretePriorPEFTRLFTAgentConfig,
    "peft_amp": DiscretePriorPEFTRLFTAMPAgentConfig,
}


class _Quantizer(nn.Module):
    num_fsq_levels = 3
    num_fsq_scalars = 4
    L = 3
    half_L = 1
    half_width = 0.5

    def quantize(self, z):
        return z.round()

    def codes_to_indices(self, codes):
        return codes.long() + 1

    def indices_to_codes(self, indices):
        return indices.float() - 1.0


class _Decoder(nn.Module):
    in_keys = ["max_coords_obs", "latent"]
    out_keys = ["decoded_action"]

    def forward(self, tensordict):
        tensordict["decoded_action"] = tensordict["latent"].sum(dim=-1, keepdim=True)
        return tensordict


class _TargetEncoder(nn.Module):
    """Maps target poses straight to FSQ codes, which SFT uses as its labels."""

    in_keys = ["mimic_target_poses"]
    out_keys = ["encoded"]

    def forward(self, tensordict):
        tensordict["encoded"] = tensordict["mimic_target_poses"].clamp(-1.0, 1.0)
        return tensordict


class _TinyPriorModel(nn.Module):
    """Smallest model exposing what the PEFT actor reads from a GPC prior."""

    def __init__(self):
        super().__init__()
        self.latent_tokenization = FSQTokenization(
            num_fsq_levels=3,
            num_fsq_scalars=4,
            fsq_scalars_per_prior_token=2,
        )
        self.latent_decoder = DiscreteLatentDecoder(
            decoder=_Decoder(),
            quantizer=_Quantizer(),
            latent_key="latent",
        )
        self.target_latent_encoder = DiscreteLatentTargetEncoder(
            encoder=_TargetEncoder(),
            quantizer=_Quantizer(),
            tokenization=self.latent_tokenization,
            target_key="target_latent",
        )
        self.prior = DiscreteAutoregressiveTransformer(
            DiscreteAutoregressiveTransformerConfig(
                in_keys=["max_coords_obs", "prior_tokens"],
                out_keys=["latent_logits"],
                context_key="max_coords_obs",
                token_key="prior_tokens",
                logits_key="latent_logits",
                generated_tokens_key="latent",
                logprob_key="latent_logprob",
                d_model=8,
                num_heads=2,
                num_layers=2,
                ff_size=16,
                dropout=0.0,
                num_tokens=2,
                vocab_size=VOCAB,
            )
        )


class _Fabric:
    device = torch.device("cpu")
    world_size = 1
    global_rank = 0

    def call(self, *args, **kwargs):
        pass

    def setup(self, module, optimizer):
        return module, optimizer


class _Env:
    def __init__(self, obs):
        self.obs = obs

    def get_obs(self):
        return {key: value.clone() for key, value in self.obs.items()}

    def get_task_id(self):
        return 0


class _Optimizer:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def _obs(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "max_coords_obs": torch.randn(NUM_ENVS, 2, generator=generator),
        "task_obs": torch.randn(NUM_ENVS, 3, generator=generator) * 2 + 1,
    }


@pytest.fixture(params=list(ADAPTER_VARIANTS))
def ctx(request, tmp_path):
    """A frozen prior and its checkpoint, shared by every simulated process."""
    torch.manual_seed(0)
    model = _TinyPriorModel()
    tokens = torch.zeros(2, 2, dtype=torch.long)
    model.prior(
        TensorDict(
            {
                "max_coords_obs": torch.randn(2, 2),
                "prior_tokens": F.one_hot(tokens, VOCAB).float(),
            },
            batch_size=2,
        )
    )
    path = tmp_path / "prior.ckpt"
    torch.save({"model": model.state_dict()}, path)
    return SimpleNamespace(
        prior_model=model,
        prior_path=path,
        tmp_path=tmp_path,
        adapter=ADAPTER_VARIANTS[request.param],
    )


def _amp_model_config(actor_config):
    """A PEFT+AMP model config with a small discriminator and disc critic."""

    def mlp(out_key):
        return MLPWithConcatConfig(
            in_keys=["max_coords_obs"],
            out_keys=[out_key],
            num_out=1,
            layers=[MLPLayerConfig(units=4, activation="relu")],
        )

    return DiscretePriorPEFTRLFTAMPModelConfig(
        actor=actor_config,
        critic=None,
        in_keys=["max_coords_obs"],
        out_keys=["disc_logits", "disc_value"],
        discriminator=DiscriminatorConfig(
            in_keys=["max_coords_obs"],
            out_keys=["disc_logits"],
            models=[mlp("disc_logits")],
        ),
        disc_critic=ModuleContainerConfig(
            in_keys=["max_coords_obs"],
            out_keys=["disc_value"],
            models=[mlp("disc_value")],
        ),
    )


def _setup_agent(
    ctx,
    *,
    seed,
    sampling_mode,
    clear_peft=False,
    agent_cls=DiscretePriorPEFTRLFTAgent,
    kl_coeff=0.1,
    reference_checkpoint=None,
    peft_model=None,
):
    """Build an RLFT agent and run its real ``setup()``, as each script does first.

    ``seed`` stands in for a separate process: it sets this process's random
    adapter initialization and the observations seen during setup. The PEFT+AMP
    agent gets the real PEFT+AMP model, discriminator included. The default
    ``kl_coeff`` makes every run need a reference, whatever its sampling mode.
    ``peft_model`` replaces the default conditioning network.
    """
    torch.manual_seed(seed)
    if sampling_mode == "prior_constraint":
        sampling = {
            "sampling_mode": "prior_constraint",
            "top_p": 1.0,
            "prior_top_p": 0.9,
        }
    else:
        sampling = {"sampling_mode": "nucleus", "top_p": 0.9, "prior_top_p": 1.0}
    actor_config = DiscretePriorPEFTActorConfig(
        in_keys=["task_obs"],
        peft=DiscretePriorPEFTConfig(
            rank=2,
            alpha=1.0,
            kl_coeff=kl_coeff,
            clear_peft=clear_peft,
            reference_checkpoint=reference_checkpoint,
            model=peft_model,
            **ctx.adapter,
            **sampling,
        ),
    )
    if agent_cls is DiscretePriorPEFTRLFTAMPAgent:
        model_config = _amp_model_config(actor_config)
        model_cls = DiscretePriorPEFTRLFTAMPModel
    else:
        model_config = DiscretePriorPEFTRLFTModelConfig(actor=actor_config, critic=None)
        model_cls = DiscretePriorPEFTModel
    model = model_cls(
        config=model_config,
        pretrained_prior_model=copy.deepcopy(ctx.prior_model),
    )

    agent = object.__new__(agent_cls)
    agent.fabric = _Fabric()
    agent.device = torch.device("cpu")
    agent.root_dir = ctx.tmp_path
    agent.env = _Env(_obs(seed))
    agent.num_envs = NUM_ENVS
    agent.add_agent_info_to_obs = lambda obs: obs
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(obs, batch_size=NUM_ENVS)
    agent._load_pretrained_modules = lambda: {"prior": ctx.prior_model}
    agent.create_model = lambda: model
    agent.create_optimizers = lambda created_model: None
    agent._print_param_info = lambda: None
    agent.config = SimpleNamespace(
        model=model_config,
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=str(ctx.prior_path),
                module_path="",
            )
        },
        normalize_rewards=False,
        advantage_normalization=SimpleNamespace(enabled=False),
        adaptive_lr=SimpleNamespace(enabled=False),
    )
    agent.motion_lib = SimpleNamespace(
        config=SimpleNamespace(motion_file_switch_mode=MotionFileSwitchMode.FIXED)
    )
    agent.evaluator = None
    agent.actor_optimizer = _Optimizer()
    agent.critic_optimizer = _Optimizer()
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent.setup()
    return agent


def _peft(agent):
    return agent.model._actor.prior_with_peft


@torch.no_grad()
def _train(agent, seed):
    """Stand in for optimizer steps by moving every student adapter tensor."""
    generator = torch.Generator().manual_seed(seed)
    for layer in _peft(agent).base_prior._transformer.layers:
        for name, parameter in layer.named_parameters():
            if not name.startswith("transformer_layer."):
                parameter.add_(torch.randn(parameter.shape, generator=generator))


def _fit_start(agent):
    """What ``fit()`` does before the first rollout: pin or check the reference."""
    agent._prepare_rlft_prior_reference()


def _save_last(agent, path, *, rlft=True):
    state = {"model": copy.deepcopy(agent.model.state_dict()), "actor_optimizer": {}}
    if rlft:
        state["critic_optimizer"] = {}
    torch.save(state, path)
    return path


def _save_adapter_checkpoint(agent, path):
    torch.save(type(agent).get_inference_state_dict(agent, {}), path)
    return path


def _is_reference_key(key):
    """True for a reference key in either kind of file.

    Adapter files store the reference's conditioning network under bare
    ``reference_peft_model.*`` keys; every other reference key has a dot before
    ``reference_``.
    """
    return ".reference_" in key or key.startswith("reference_")


def _drop_reference_keys(path):
    state = torch.load(path, weights_only=False)
    state["model"] = {
        key: value
        for key, value in state["model"].items()
        if not _is_reference_key(key)
    }
    torch.save(state, path)
    return path


def _probe_input(agent):
    actor = agent.model._actor
    tokens = torch.randint(
        0, VOCAB, (NUM_ENVS, 2), generator=torch.Generator().manual_seed(5)
    )
    return actor.build_prior_input(
        TensorDict(_obs(999), batch_size=NUM_ENVS), tokens=tokens
    )


@torch.no_grad()
def _reference_logits(agent):
    agent.model.eval()
    return _peft(agent).forward_prior(_probe_input(agent))


@torch.no_grad()
def _student_logits(agent):
    agent.model.eval()
    return _peft(agent)(_probe_input(agent))


@torch.no_grad()
def _frozen_prior_logits(agent):
    """Logits of the configured prior with every adapter residual switched off."""
    clone = copy.deepcopy(agent)
    _peft(clone).clear_peft()
    return _student_logits(clone)


@torch.no_grad()
def _sample(agent, seed=4):
    agent.model.eval()
    obs_td = TensorDict(_obs(123), batch_size=NUM_ENVS)
    torch.manual_seed(seed)
    return agent.model._actor.get_action_and_logp(obs_td)


def _assert_same_samples(actual, expected):
    assert torch.equal(actual["prior_tokens"], expected["prior_tokens"])
    assert torch.equal(actual["neglogp"], expected["neglogp"])


def _assert_same_policy(actual, expected):
    """Same student, same reference, and same sampled tokens and log-probs."""
    assert torch.equal(_student_logits(actual), _student_logits(expected))
    assert _peft(actual).reference_ready == _peft(expected).reference_ready
    if _peft(expected).reference_ready:
        assert torch.equal(_reference_logits(actual), _reference_logits(expected))
    _assert_same_samples(_sample(actual), _sample(expected))


def _inference_load(agent, path):
    """The call inference_agent.py makes after setup()."""
    agent.load(path, load_env=False, load_training_state=False, for_inference=True)


def _warm_start(agent, path):
    """The call train_agent.py makes after setup() to warm-start a new run."""
    agent.load(path, load_training_state=False)


def _warm_start_and_fit_start(agent, path):
    _warm_start(agent, path)
    _fit_start(agent)


LOADS = {"inference": _inference_load, "warm_start": _warm_start_and_fit_start}


def _fresh_run(ctx, sampling_mode, clear_peft=False, agent_cls=None, kl_coeff=0.1):
    """train_agent.py fresh mode: setup, load(None), fit start, then train.

    Also returns the reference's logits, or None when the run pins no reference.
    """
    run = _setup_agent(
        ctx,
        seed=10,
        sampling_mode=sampling_mode,
        clear_peft=clear_peft,
        agent_cls=agent_cls or DiscretePriorPEFTRLFTAgent,
        kl_coeff=kl_coeff,
    )
    run.load(None, load_training_state=False)
    _fit_start(run)
    # Judge from the config, so a run that should have pinned a reference and
    # didn't still fails here.
    needs_reference = sampling_mode == "prior_constraint" or kl_coeff > 0
    assert _peft(run).reference_ready == needs_reference
    reference = _reference_logits(run) if needs_reference else None
    _train(run, seed=1)
    return run, reference


def _load_full_and_adapter_checkpoint(ctx, last, adapter, load, **setup_kwargs):
    """Two processes with the same seed: one loads last.ckpt, one the adapter file."""
    via_last = _setup_agent(ctx, seed=20, **setup_kwargs)
    load(via_last, last)
    via_adapter_checkpoint = _setup_agent(ctx, seed=20, **setup_kwargs)
    load(via_adapter_checkpoint, adapter)
    return via_last, via_adapter_checkpoint


def _sft_run(ctx):
    """An SFT-trained adapter with no reference pinned, as an SFT run leaves it.

    The real SFT model never calls ``generate()``, so it never pins a reference
    itself. A nucleus-mode RLFT agent, which pins nothing during setup, stands
    in for it.
    """
    sft = _setup_agent(ctx, seed=50, sampling_mode="nucleus")
    _train(sft, seed=7)
    assert not _peft(sft).reference_ready
    return sft


def _save_sft_files(sft, tmp_path):
    """The two files an SFT run writes.

    The full checkpoint has no critic optimizer, which is how RLFT recognizes an
    SFT warm start. The adapter checkpoint is written by the SFT agent's own
    ``get_inference_state_dict``.
    """
    last = _save_last(sft, tmp_path / "sft_last.ckpt", rlft=False)
    adapter = tmp_path / "sft_inference_last.ckpt"
    torch.save(DiscretePriorPEFTSFTAgent.get_inference_state_dict(sft, {}), adapter)
    return last, adapter


def _sft_checkpoints(ctx):
    """The two files an SFT run writes, and the SFT policy's logits."""
    sft = _sft_run(ctx)
    last, adapter = _save_sft_files(sft, ctx.tmp_path)
    return last, adapter, _student_logits(sft)


def _sft_checkpoints_after_rlft(ctx):
    """The two files of a real SFT run that started from an RLFT last.ckpt.

    The RLFT file stores a reference, which the SFT run drops when it loads
    the file. Also returns the SFT policy's logits.
    """
    rlft, _ = _fresh_run(ctx, "nucleus")
    rlft_last = _save_last(rlft, ctx.tmp_path / "rlft_last.ckpt")
    sft = _setup_sft_agent(ctx, seed=50)
    sft.load(rlft_last, load_env=False, load_training_state=False)
    assert not _peft(sft).reference_ready
    _train(sft, seed=7)
    last, adapter = _save_real_sft_files(sft, ctx.tmp_path)
    return last, adapter, _student_logits(sft)


def _setup_sft_agent(ctx, *, seed, training=True):
    """Build the real SFT agent and run its real ``setup()``.

    Unlike ``_setup_agent``, the agent builds its own SFT model, frozen target
    encoder included, and its own optimizer. It is configured with
    sampling_mode="prior_constraint", as the shipped ``sft_target_prior_peft.py``
    is, although SFT never samples.
    """
    torch.manual_seed(seed)
    model_config = DiscretePriorPEFTSFTModelConfig(
        actor=DiscretePriorPEFTActorConfig(
            in_keys=["task_obs"],
            peft=DiscretePriorPEFTConfig(
                rank=2,
                alpha=1.0,
                sampling_mode="prior_constraint",
                top_p=0.9,
                prior_top_p=0.9,
                **ctx.adapter,
            ),
        ),
    )
    obs = _obs(seed)
    obs["mimic_target_poses"] = torch.randn(
        NUM_ENVS, 4, generator=torch.Generator().manual_seed(seed + 1)
    )

    agent = object.__new__(DiscretePriorPEFTSFTAgent)
    agent.fabric = _Fabric()
    agent.device = torch.device("cpu")
    agent.root_dir = ctx.tmp_path
    agent.env = _Env(obs)
    agent.num_envs = NUM_ENVS
    agent.add_agent_info_to_obs = lambda obs: obs
    agent.obs_dict_to_tensordict = lambda obs: TensorDict(obs, batch_size=NUM_ENVS)
    agent._load_pretrained_modules = lambda: {"prior": copy.deepcopy(ctx.prior_model)}
    agent.config = DiscretePriorPEFTSFTAgentConfig(
        batch_size=NUM_ENVS,
        training_max_steps=1,
        normalize_rewards=False,
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=str(ctx.prior_path),
                module_path="",
            )
        },
        model=model_config,
    )
    agent.motion_lib = SimpleNamespace(
        config=SimpleNamespace(motion_file_switch_mode=MotionFileSwitchMode.FIXED)
    )
    agent.evaluator = None
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent.setup(training=training)
    return agent


def _save_real_sft_files(sft, tmp_path):
    """Write ``last.ckpt`` and its adapter file the way ``BaseAgent.save`` does."""
    state = sft.get_state_dict({})
    last = tmp_path / "sft_last.ckpt"
    torch.save(state, last)
    adapter = tmp_path / "sft_inference_last.ckpt"
    torch.save(
        sft.get_inference_state_dict({}, model_state_dict=state["model"]), adapter
    )
    return last, adapter


def _has_reference_keys(path):
    model_state = torch.load(path, weights_only=False)["model"]
    return any(_is_reference_key(key) for key in model_state)


# ---------------------------------------------------------------------------
# What an adapter checkpoint stores, and restoring it
# ---------------------------------------------------------------------------


def test_adapter_checkpoint_stores_reference_adapters_but_not_frozen_weights(ctx):
    run, _ = _fresh_run(ctx, "prior_constraint")
    adapter = DiscretePriorPEFTRLFTAgent.get_inference_state_dict(run, {})["model"]

    reference_keys = {key for key in adapter if key.startswith(REFERENCE_PREFIX)}
    conditioning_keys = {
        key for key in adapter if key.startswith("reference_peft_model.")
    }
    student_prior_keys = {
        key for key in adapter if key.startswith("prior_with_peft.base_prior.")
    }
    assert all(
        is_adapter_state_key(key) or key in reference_keys | conditioning_keys
        for key in adapter
    )
    # The reference stores its own copy of the whole conditioning network.
    assert {
        key.replace("reference_peft_model.", "actor_peft_model.", 1)
        for key in conditioning_keys
    } == {key for key in adapter if key.startswith("actor_peft_model.")}
    # It also stores exactly the student's adapter tensors plus its own FiLM
    # input normalizer, and none of the frozen prior weights.
    assert {
        key.replace("reference_prior.", "base_prior.")
        for key in reference_keys
        if key.startswith("prior_with_peft.reference_prior.")
    } == student_prior_keys
    has_reference_norm = any(
        key.startswith("prior_with_peft.reference_film_input_norm.")
        for key in reference_keys
    )
    assert has_reference_norm == ctx.adapter["film_input_norm"]
    full_reference = [
        key
        for key in run.model.state_dict()
        if key.startswith("_actor.prior_with_peft.reference_prior.")
    ]
    assert len(full_reference) > len(student_prior_keys)
    assert not any("transformer_layer" in key or "_pos_emb" in key for key in adapter)


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_adapter_checkpoint_load_restores_fresh_run_reference_exactly(ctx, sampling_mode):
    run, training_reference = _fresh_run(ctx, sampling_mode)
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    inference = _setup_agent(ctx, seed=20, sampling_mode=sampling_mode)
    _inference_load(inference, adapter)

    assert torch.equal(_reference_logits(inference), training_reference)
    assert torch.equal(_student_logits(inference), _student_logits(run))
    assert not torch.allclose(_reference_logits(inference), _student_logits(run))
    _assert_same_samples(_sample(inference), _sample(run))


def test_adapter_checkpoint_load_restores_sft_reference_in_nucleus_mode(ctx):
    sft_last, _, sft_policy = _sft_checkpoints(ctx)
    run = _setup_agent(ctx, seed=60, sampling_mode="nucleus")
    _warm_start_and_fit_start(run, sft_last)
    training_reference = _reference_logits(run)
    assert torch.equal(training_reference, sft_policy)
    _train(run, seed=2)
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    inference = _setup_agent(ctx, seed=70, sampling_mode="nucleus")
    _inference_load(inference, adapter)

    assert torch.equal(_reference_logits(inference), training_reference)
    assert not torch.allclose(_reference_logits(inference), _student_logits(run))


def test_adapter_checkpoint_load_restores_non_trivial_reference_in_prior_constraint_mode(ctx):
    # Stage one trains from an SFT policy and saves last.ckpt.
    sft_last, _, _ = _sft_checkpoints(ctx)
    stage_one = _setup_agent(ctx, seed=60, sampling_mode="nucleus")
    _warm_start_and_fit_start(stage_one, sft_last)
    _train(stage_one, seed=2)
    stage_one_policy = _student_logits(stage_one)
    stage_one_last = _save_last(stage_one, ctx.tmp_path / "stage_one_last.ckpt")

    # Stage two continues with prior-constrained sampling from that last.ckpt,
    # so its reference is stage one's trained policy rather than the frozen prior.
    stage_two = _setup_agent(ctx, seed=61, sampling_mode="prior_constraint")
    _warm_start_and_fit_start(stage_two, stage_one_last)
    training_reference = _reference_logits(stage_two)
    assert torch.equal(training_reference, stage_one_policy)
    _train(stage_two, seed=3)
    adapter = _save_adapter_checkpoint(stage_two, ctx.tmp_path / "inference_last.ckpt")

    inference = _setup_agent(ctx, seed=70, sampling_mode="prior_constraint")
    _inference_load(inference, adapter)

    assert torch.equal(_reference_logits(inference), training_reference)
    assert not torch.allclose(
        _reference_logits(inference), _frozen_prior_logits(inference), atol=1e-3
    )
    assert not torch.allclose(_reference_logits(inference), _student_logits(stage_two))
    # The prior-constrained sampler then draws exactly what training drew.
    _assert_same_samples(_sample(inference), _sample(stage_two))


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_adapter_checkpoint_load_keeps_trained_adapters_when_clear_peft_is_set(ctx, sampling_mode):
    run, training_reference = _fresh_run(ctx, sampling_mode, clear_peft=True)
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    inference = _setup_agent(ctx, seed=20, sampling_mode=sampling_mode, clear_peft=True)
    _inference_load(inference, adapter)

    assert torch.equal(_student_logits(inference), _student_logits(run))
    assert torch.equal(_reference_logits(inference), training_reference)


def test_adapter_checkpoint_with_partial_reference_is_rejected(ctx):
    run, _ = _fresh_run(ctx, "prior_constraint")
    path = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    state = torch.load(path, weights_only=False)
    dropped = next(
        key
        for key in state["model"]
        if key.startswith("prior_with_peft.reference_prior.")
    )
    del state["model"][dropped]
    torch.save(state, path)

    inference = _setup_agent(ctx, seed=20, sampling_mode="prior_constraint")
    with pytest.raises(RuntimeError, match="missing_keys"):
        _inference_load(inference, path)


# ---------------------------------------------------------------------------
# Same seed, full checkpoint versus adapter file: identical sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("load", list(LOADS))
@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_full_and_adapter_checkpoint_rlft_checkpoints_sample_the_same(
    ctx, sampling_mode, clear_peft, load
):
    run, training_reference = _fresh_run(ctx, sampling_mode, clear_peft=clear_peft)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx,
        last,
        adapter,
        LOADS[load],
        sampling_mode=sampling_mode,
        clear_peft=clear_peft,
    )

    _assert_same_policy(via_adapter_checkpoint, via_last)
    trained = _student_logits(run)
    if load == "warm_start":
        # A new run pins the trained policy it loaded, then applies clear_peft.
        expected_reference = trained
        expected_student = (
            _frozen_prior_logits(via_adapter_checkpoint) if clear_peft else trained
        )
    else:
        expected_reference, expected_student = training_reference, trained
    assert torch.equal(_reference_logits(via_adapter_checkpoint), expected_reference)
    assert torch.equal(_student_logits(via_adapter_checkpoint), expected_student)


@pytest.mark.parametrize("load", list(LOADS))
@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_full_and_adapter_checkpoint_peft_amp_checkpoints_sample_the_same(ctx, sampling_mode, load):
    amp = DiscretePriorPEFTRLFTAMPAgent
    run, training_reference = _fresh_run(ctx, sampling_mode, agent_cls=amp)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    assert any(
        key.startswith("_discriminator.")
        for key in torch.load(last, weights_only=False)["model"]
    )
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx, last, adapter, LOADS[load], sampling_mode=sampling_mode, agent_cls=amp
    )

    _assert_same_policy(via_adapter_checkpoint, via_last)
    assert torch.equal(_student_logits(via_adapter_checkpoint), _student_logits(run))
    # A new run pins the trained policy it loaded; inference uses the saved one.
    expected_reference = (
        _student_logits(run) if load == "warm_start" else training_reference
    )
    assert torch.equal(_reference_logits(via_adapter_checkpoint), expected_reference)


@pytest.mark.parametrize("agent", list(AGENT_CLASSES))
@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_warm_start_from_sft_adapter_checkpoint_matches_full_sft_checkpoint(
    ctx, sampling_mode, clear_peft, agent
):
    """Training from an SFT adapter file starts exactly as from its last.ckpt.

    SFT adapter files store no reference, like RLFT adapter files written before
    they stored one. Warm-starting from either must pin the reference, and
    apply clear_peft, exactly as a warm start from the full SFT checkpoint.
    """
    sft_last, sft_adapter_checkpoint, sft_policy = _sft_checkpoints(ctx)
    assert not _has_reference_keys(sft_adapter_checkpoint)

    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx,
        sft_last,
        sft_adapter_checkpoint,
        _warm_start_and_fit_start,
        sampling_mode=sampling_mode,
        clear_peft=clear_peft,
        agent_cls=AGENT_CLASSES[agent],
    )

    _assert_same_policy(via_adapter_checkpoint, via_last)
    assert torch.equal(_reference_logits(via_adapter_checkpoint), sft_policy)
    expected_student = _frozen_prior_logits(via_adapter_checkpoint) if clear_peft else sft_policy
    assert torch.equal(_student_logits(via_adapter_checkpoint), expected_student)


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_warm_start_from_sft_that_loaded_rlft_pins_the_sft_policy(
    ctx, sampling_mode
):
    """An SFT run started from an RLFT last.ckpt drops that run's reference.

    Neither of its files stores a reference, so RLFT warm-started from either
    pins the SFT policy, exactly as from any other SFT run.
    """
    sft_last, sft_adapter_checkpoint, sft_policy = _sft_checkpoints_after_rlft(ctx)
    assert not _has_reference_keys(sft_last)
    assert not _has_reference_keys(sft_adapter_checkpoint)

    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx,
        sft_last,
        sft_adapter_checkpoint,
        _warm_start_and_fit_start,
        sampling_mode=sampling_mode,
    )

    _assert_same_policy(via_adapter_checkpoint, via_last)
    assert torch.equal(_reference_logits(via_adapter_checkpoint), sft_policy)


def test_adapter_checkpoint_load_restores_the_reference_film_input_normalizer(ctx):
    """The reference keeps its own FiLM input normalizer, not the student's."""
    if not ctx.adapter["film_input_norm"]:
        pytest.skip("this adapter variant has no FiLM input normalizer")
    # Warm-starting from SFT makes the reference the SFT policy, whose FiLM
    # weights are trained, so its FiLM input normalizer affects its logits.
    sft_last, _, _ = _sft_checkpoints(ctx)
    run = _setup_agent(ctx, seed=60, sampling_mode="nucleus")
    _warm_start_and_fit_start(run, sft_last)
    training_reference = _reference_logits(run)
    student_before = _student_logits(run)
    with torch.no_grad():
        for name, buffer in _peft(run).film_input_norm.named_buffers():
            if name.endswith("mean"):
                buffer.add_(0.5)
    # Only the student's normalizer moved.
    assert not torch.equal(_student_logits(run), student_before)
    assert torch.equal(_reference_logits(run), training_reference)
    _train(run, seed=2)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx, last, adapter, _inference_load, sampling_mode="nucleus"
    )

    _assert_same_policy(via_adapter_checkpoint, via_last)
    assert torch.equal(_reference_logits(via_adapter_checkpoint), training_reference)


# ---------------------------------------------------------------------------
# Adapter files without a reference: SFT files and RLFT files from older code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize("agent", list(AGENT_CLASSES))
def test_legacy_adapter_checkpoint_samples_like_last_ckpt_in_nucleus_mode(
    ctx, agent, clear_peft
):
    agent_cls = AGENT_CLASSES[agent]
    run, _ = _fresh_run(ctx, "nucleus", clear_peft=clear_peft, agent_cls=agent_cls)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    legacy = _drop_reference_keys(_save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt"))

    via_last, via_legacy = _load_full_and_adapter_checkpoint(
        ctx,
        last,
        legacy,
        _inference_load,
        sampling_mode="nucleus",
        clear_peft=clear_peft,
        agent_cls=agent_cls,
    )

    # Nucleus sampling never reads the reference, so none is pinned from the
    # legacy file, and clear_peft does not wipe the trained adapters.
    assert not _peft(via_legacy).reference_ready
    assert torch.equal(_student_logits(via_legacy), _student_logits(via_last))
    assert torch.equal(_student_logits(via_legacy), _student_logits(run))
    _assert_same_samples(_sample(via_legacy), _sample(via_last))


@pytest.mark.parametrize("agent", list(AGENT_CLASSES))
def test_adapter_checkpoint_without_reference_fails_constrained_inference(ctx, agent):
    agent_cls = AGENT_CLASSES[agent]
    run, training_reference = _fresh_run(
        ctx, "prior_constraint", clear_peft=True, agent_cls=agent_cls
    )
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    legacy = _drop_reference_keys(_save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt"))

    inference = _setup_agent(
        ctx,
        seed=20,
        sampling_mode="prior_constraint",
        clear_peft=True,
        agent_cls=agent_cls,
    )
    before = copy.deepcopy(inference.model.state_dict())
    assert not _peft(inference).reference_ready
    with pytest.raises(RuntimeError, match="no saved reference") as error:
        _inference_load(inference, legacy)
    for key, value in before.items():
        assert torch.equal(inference.model.state_dict()[key], value)
    assert not _peft(inference).reference_ready

    # The error offers two ways out, and both work. The first is last.ckpt.
    message = str(error.value)
    assert "last.ckpt" in message
    recovered = _setup_agent(
        ctx,
        seed=20,
        sampling_mode="prior_constraint",
        clear_peft=True,
        agent_cls=agent_cls,
    )
    _inference_load(recovered, last)
    assert torch.equal(_reference_logits(recovered), training_reference)

    # The second is an override that names a real field of the agent config...
    override = re.search(r"--overrides (\S+=\S+)", message)
    assert override is not None, message
    overrides = parse_cli_overrides([override.group(1)])
    config = AGENT_CONFIG_CLASSES[agent](batch_size=NUM_ENVS, training_max_steps=1)
    config.model.actor.peft.sampling_mode = "prior_constraint"
    apply_overrides(
        ExperimentConfigs(None, None, None, None, None, None, config), overrides
    )
    assert config.model.actor.peft.sampling_mode == "nucleus"

    # ...and with it applied the same file loads and keeps the trained student.
    unconstrained = _setup_agent(
        ctx,
        seed=20,
        sampling_mode=config.model.actor.peft.sampling_mode,
        clear_peft=True,
        agent_cls=agent_cls,
    )
    _inference_load(unconstrained, legacy)
    assert torch.equal(_student_logits(unconstrained), _student_logits(run))


@pytest.mark.parametrize("checkpoint", ["last", "adapter"])
def test_sft_run_files_load_for_inference_in_prior_constraint_mode(ctx, checkpoint):
    """SFT inference loads the run's own files, though neither stores a reference.

    The SFT model replays its target encoder's tokens instead of sampling, so
    prior_constraint never reads a reference there, and an SFT run never pins one.
    """
    sft = _setup_sft_agent(ctx, seed=50)
    _train(sft, seed=7)
    last, adapter = _save_real_sft_files(sft, ctx.tmp_path)
    path = last if checkpoint == "last" else adapter
    assert not _has_reference_keys(path)

    inference = _setup_sft_agent(ctx, seed=70, training=False)
    assert not torch.equal(_student_logits(inference), _student_logits(sft))
    _inference_load(inference, path)

    assert not _peft(inference).reference_ready
    trained = sft.model._actor.adapter_state_dict()
    loaded = inference.model._actor.adapter_state_dict()
    assert trained.keys() == loaded.keys()
    for key, value in trained.items():
        assert torch.equal(loaded[key], value), key
    assert torch.equal(_student_logits(inference), _student_logits(sft))
    # The step the evaluator runs needs no reference either.
    inference.eval()
    obs_td = TensorDict(inference.env.get_obs(), batch_size=NUM_ENVS)
    assert "action" in inference.inference_step(obs_td)


@pytest.mark.parametrize("agent", list(AGENT_CLASSES))
@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_training_from_legacy_adapter_checkpoint_pins_the_reference_at_fit_start(
    ctx, sampling_mode, clear_peft, agent
):
    """The load leaves the reference alone; fit start pins it as for a warm start."""
    agent_cls = AGENT_CLASSES[agent]
    run, _ = _fresh_run(ctx, sampling_mode, agent_cls=agent_cls)
    trained = _student_logits(run)
    legacy = _drop_reference_keys(_save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt"))

    warm = _setup_agent(
        ctx,
        seed=20,
        sampling_mode=sampling_mode,
        clear_peft=clear_peft,
        agent_cls=agent_cls,
    )
    assert not _peft(warm).reference_ready
    _warm_start(warm, legacy)
    assert torch.equal(_student_logits(warm), trained)
    _fit_start(warm)

    assert torch.equal(_reference_logits(warm), trained)
    expected_student = _frozen_prior_logits(warm) if clear_peft else trained
    assert torch.equal(_student_logits(warm), expected_student)


# ---------------------------------------------------------------------------
# last.ckpt, resume, and the PEFT+AMP agent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_last_ckpt_inference_and_resume_restore_saved_reference(ctx, sampling_mode):
    run, training_reference = _fresh_run(ctx, sampling_mode)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")

    inference = _setup_agent(ctx, seed=20, sampling_mode=sampling_mode)
    _inference_load(inference, last)
    resumed = _setup_agent(ctx, seed=30, sampling_mode=sampling_mode)
    resumed.load(last, load_env=False, load_training_state=True)

    assert torch.equal(_reference_logits(inference), training_reference)
    assert torch.equal(_reference_logits(resumed), training_reference)
    assert torch.equal(_student_logits(resumed), _student_logits(run))


def test_true_resume_still_requires_saved_reference(ctx):
    run, _ = _fresh_run(ctx, "nucleus")
    last = _drop_reference_keys(_save_last(run, ctx.tmp_path / "last.ckpt"))

    resumed = _setup_agent(ctx, seed=30, sampling_mode="nucleus")
    with pytest.raises(RuntimeError, match="no saved reference"):
        resumed.load(last, load_env=False, load_training_state=True)


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_peft_amp_agent_loads_adapter_checkpoint_of_a_non_amp_run(ctx, sampling_mode):
    run, training_reference = _fresh_run(ctx, sampling_mode, clear_peft=True)
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    inference = _setup_agent(
        ctx,
        seed=20,
        sampling_mode=sampling_mode,
        clear_peft=True,
        agent_cls=DiscretePriorPEFTRLFTAMPAgent,
    )
    fresh_disc = copy.deepcopy(inference.model._discriminator.state_dict())
    _inference_load(inference, adapter)

    assert torch.equal(_student_logits(inference), _student_logits(run))
    assert torch.equal(_reference_logits(inference), training_reference)
    _assert_same_samples(_sample(inference), _sample(run))
    for key, value in fresh_disc.items():
        assert torch.equal(inference.model._discriminator.state_dict()[key], value)


@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_peft_amp_warm_start_from_full_non_amp_checkpoint_pins_the_loaded_policy(
    ctx, sampling_mode, clear_peft
):
    """Adding AMP must not bypass PEFT's rule for a new run's reference."""
    # Stage one's saved reference is the SFT policy, not its trained student.
    sft_last, _, _ = _sft_checkpoints(ctx)
    stage_one = _setup_agent(ctx, seed=60, sampling_mode="nucleus")
    _warm_start_and_fit_start(stage_one, sft_last)
    _train(stage_one, seed=2)
    trained = _student_logits(stage_one)
    stage_one_last = _save_last(stage_one, ctx.tmp_path / "stage_one_last.ckpt")

    amp_run = _setup_agent(
        ctx,
        seed=61,
        sampling_mode=sampling_mode,
        clear_peft=clear_peft,
        agent_cls=DiscretePriorPEFTRLFTAMPAgent,
    )
    _warm_start_and_fit_start(amp_run, stage_one_last)

    assert torch.equal(_reference_logits(amp_run), trained)
    expected_student = _frozen_prior_logits(amp_run) if clear_peft else trained
    assert torch.equal(_student_logits(amp_run), expected_student)


@pytest.mark.parametrize("checkpoint_kind", ["full", "adapter"])
@pytest.mark.parametrize("section", ["student", "reference"])
@pytest.mark.parametrize("corruption", ["missing", "extra", "shape"])
def test_rejected_checkpoint_leaves_student_and_reference_unchanged(
    ctx, checkpoint_kind, section, corruption
):
    source, _ = _fresh_run(ctx, "prior_constraint")
    target, _ = _fresh_run(ctx, "prior_constraint")
    _train(source, seed=200)
    # Deliberately distinct target student AND reference so partial loads are visible.
    target.model._actor.reset_reference()
    _train(target, seed=201)
    _fit_start(target)
    before = copy.deepcopy(target.model.state_dict())
    state = (copy.deepcopy(source.model.state_dict()) if checkpoint_kind == "full"
             else source.get_inference_state_dict({})["model"])
    needle = ".reference_prior." if section == "reference" else ".base_prior."
    key = next(key for key in state if needle in key and ".lora." in key)
    if corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state[key + ".extra"] = torch.ones(1)
    else:
        state[key] = torch.ones(1, 1, 1)
    with pytest.raises(RuntimeError, match="Checkpoint mismatch"):
        target._load_model_state_dict(state)
    assert before.keys() == target.model.state_dict().keys()
    for key, value in before.items():
        assert torch.equal(target.model.state_dict()[key], value), key
    assert _peft(target).reference_ready


@pytest.mark.parametrize("checkpoint_kind", ["full", "adapter"])
@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
@pytest.mark.parametrize("kl_coeff", [0.0, 0.1])
def test_resume_reference_requirement_depends_on_consumers(
    ctx, checkpoint_kind, sampling_mode, kl_coeff
):
    source, _ = _fresh_run(ctx, sampling_mode)
    path = (_save_last(source, ctx.tmp_path / "last.ckpt") if checkpoint_kind == "full"
            else _save_adapter_checkpoint(source, ctx.tmp_path / "inference_last.ckpt"))
    _drop_reference_keys(path)
    target = _setup_agent(ctx, seed=45, sampling_mode=sampling_mode)
    target.model._actor.kl_coeff = kl_coeff
    before = copy.deepcopy(target.model.state_dict())
    if sampling_mode == "prior_constraint" or kl_coeff > 0:
        with pytest.raises(RuntimeError, match="no saved reference"):
            target.load(path, load_env=False, load_training_state=True)
        for key, value in before.items():
            assert torch.equal(target.model.state_dict()[key], value)
    elif checkpoint_kind == "full":
        target.load(path, load_env=False, load_training_state=True)
        _fit_start(target)
        assert not _peft(target).reference_ready
    else:
        # An adapter file is a weight warm start, not an optimizer resume.
        target.load(path, load_env=False, load_training_state=False)
        _fit_start(target)
        assert not _peft(target).reference_ready


def test_setup_and_generation_do_not_pin_a_reference(ctx):
    agent = _setup_agent(ctx, seed=25, sampling_mode="prior_constraint")
    assert not _peft(agent).reference_ready
    with pytest.raises(RuntimeError, match="requires a frozen reference"):
        _sample(agent)
    assert not _peft(agent).reference_ready


def test_loading_new_weights_drops_a_previous_reference(ctx):
    agent, _ = _fresh_run(ctx, "prior_constraint")
    sft, _, sft_policy = _sft_checkpoints(ctx)
    _warm_start(agent, sft)
    assert not _peft(agent).reference_ready
    _fit_start(agent)
    assert torch.equal(_reference_logits(agent), sft_policy)


def test_every_reference_network_is_optional_in_a_full_file(ctx):
    """A full file saved without a reference loads into a model that holds one."""
    run, _ = _fresh_run(ctx, "prior_constraint")
    optional = optional_full_checkpoint_state_prefixes(run.model)
    reference_keys = [key for key in run.model.state_dict() if _is_reference_key(key)]

    assert any(key.startswith("_actor.reference_peft_model.") for key in reference_keys)
    for key in reference_keys:
        assert key.startswith(optional), key


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
@pytest.mark.parametrize("adding_amp", [False, True])
def test_adding_or_removing_amp_keeps_full_and_adapter_checkpoint_policy_parity(
    ctx, sampling_mode, adding_amp
):
    plain, amp = DiscretePriorPEFTRLFTAgent, DiscretePriorPEFTRLFTAMPAgent
    source_cls, target_cls = (plain, amp) if adding_amp else (amp, plain)
    run, _ = _fresh_run(ctx, sampling_mode, agent_cls=source_cls)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")

    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx,
        last,
        adapter,
        _inference_load,
        sampling_mode=sampling_mode,
        agent_cls=target_cls,
    )

    _assert_same_policy(via_adapter_checkpoint, via_last)
    _assert_same_policy(via_last, run)


# ---------------------------------------------------------------------------
# Nucleus sampling without a KL penalty: nothing reads a reference
# ---------------------------------------------------------------------------


def _frozen_normalizers(agent):
    """Normalizers in the PEFT wrapper that no longer record statistics.

    The reference's own copies are left out, so after setup this is just the
    frozen prior's normalizers. The student conditioning network's normalizer
    lives outside the PEFT wrapper and is never frozen.
    """
    return {
        name
        for name, module in _peft(agent).named_modules()
        if isinstance(module, NormObsBase)
        and module._freeze_running
        and not name.startswith("reference_")
    }


def _conditioning_normalizers(agent):
    """The conditioning network's normalizers, mapped to whether each is frozen."""
    return {
        name: module._freeze_running
        for name, module in agent.model._actor.actor_peft_model.named_modules()
        if isinstance(module, NormObsBase)
    }


PRIOR_AGENT_LOGGER = "protomotions.agents.peft.prior_agent"


def _skipped_clear_peft_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.getMessage().startswith("clear_peft=True was skipped")
    ]


@torch.no_grad()
def _training_forward_moves_film_norm(agent):
    """Run one training-mode forward pass; report whether the FiLM statistics moved."""
    film_norm = _peft(agent).film_input_norm
    before = copy.deepcopy(film_norm.state_dict())
    agent.model.train()
    _peft(agent)(_probe_input(agent))
    agent.model.eval()
    after = film_norm.state_dict()
    return any(not torch.equal(after[key], value) for key, value in before.items())


@pytest.mark.parametrize("agent", list(AGENT_CLASSES))
def test_nucleus_without_kl_pins_no_reference(ctx, agent):
    run, reference = _fresh_run(
        ctx, "nucleus", agent_cls=AGENT_CLASSES[agent], kl_coeff=0.0
    )

    assert reference is None
    assert not _peft(run).reference_ready
    assert _peft(run).reference_prior is None
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    assert not _has_reference_keys(adapter)


@pytest.mark.parametrize("how", ["warm_start", "resume"])
def test_nucleus_without_kl_drops_a_loaded_reference(ctx, caplog, how):
    """A run that reads no reference holds none, so it never saves one again.

    The loaded file comes from a run that pinned a reference, standing in for
    a file that carries one into a run configured without a KL penalty.
    """
    source, _ = _fresh_run(ctx, "prior_constraint")
    path = _save_last(source, ctx.tmp_path / "source_last.ckpt")
    run = _setup_agent(
        ctx, seed=60, sampling_mode="nucleus", clear_peft=True, kl_coeff=0.0
    )
    run.load(path, load_env=False, load_training_state=how == "resume")
    assert _peft(run).reference_ready

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=PRIOR_AGENT_LOGGER):
        _fit_start(run)

    assert not _peft(run).reference_ready
    assert run.model._actor.reference_peft_model is None
    assert torch.equal(_student_logits(run), _student_logits(source))
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    assert not _has_reference_keys(last)
    assert not _has_reference_keys(adapter)
    # Only a new run would have cleared the adapters, so only it warns.
    assert len(_skipped_clear_peft_warnings(caplog)) == int(how == "warm_start")


def test_nucleus_without_kl_keeps_student_normalizers_learning(ctx):
    """Fit start freezes none of the student's normalizers, pinned reference or not."""
    if not ctx.adapter["film_input_norm"]:
        pytest.skip("this adapter variant has no FiLM input normalizer")
    run = _setup_agent(ctx, seed=10, sampling_mode="nucleus", kl_coeff=0.0)
    run.load(None, load_training_state=False)
    pinned = _setup_agent(ctx, seed=10, sampling_mode="nucleus", kl_coeff=0.1)
    pinned.load(None, load_training_state=False)

    _fit_start(run)
    _fit_start(pinned)

    assert not _peft(run).reference_ready
    assert _peft(pinned).reference_ready
    for agent in (run, pinned):
        # Only the frozen prior's normalizers stopped recording, at setup.
        frozen_prior = {
            name
            for name, module in _peft(agent).named_modules()
            if isinstance(module, NormObsBase) and name.startswith("base_prior.")
        }
        assert frozen_prior
        assert _frozen_normalizers(agent) == frozen_prior
        assert _training_forward_moves_film_norm(agent)
        # The conditioning network's own normalizer keeps updating too.
        conditioning = _conditioning_normalizers(agent)
        assert conditioning
        assert not any(conditioning.values())
    # The reference's copies are the ones that are frozen.
    assert run.model._actor.reference_peft_model is None
    reference = {
        name: module._freeze_running
        for name, module in pinned.model._actor.named_modules()
        if isinstance(module, NormObsBase)
        and (name.startswith("reference_") or ".reference_" in name)
    }
    assert "prior_with_peft.reference_film_input_norm" in reference
    assert any(name.startswith("reference_peft_model.") for name in reference)
    assert all(reference.values())


@pytest.mark.parametrize("clear_peft", [False, True])
def test_nucleus_without_kl_keeps_the_loaded_student_and_warns_about_clear_peft(
    ctx, caplog, clear_peft
):
    sft_last, _, sft_policy = _sft_checkpoints(ctx)
    run = _setup_agent(
        ctx, seed=60, sampling_mode="nucleus", clear_peft=clear_peft, kl_coeff=0.0
    )
    _warm_start(run, sft_last)
    # Clearing the adapters would visibly change the student.
    assert not torch.allclose(_frozen_prior_logits(run), sft_policy, atol=1e-3)

    with caplog.at_level(logging.WARNING, logger=PRIOR_AGENT_LOGGER):
        _fit_start(run)

    assert torch.equal(_student_logits(run), sft_policy)
    assert not _peft(run).reference_ready
    warnings = _skipped_clear_peft_warnings(caplog)
    assert len(warnings) == int(clear_peft)
    if clear_peft:
        assert "sampling_mode='nucleus' and kl_coeff=0" in warnings[0]
        assert "pins no reference policy" in warnings[0]


def test_resuming_nucleus_without_kl_does_not_warn_about_clear_peft(ctx, caplog):
    """A resume never clears the adapters in any mode, so it has nothing to report."""
    with caplog.at_level(logging.WARNING, logger=PRIOR_AGENT_LOGGER):
        run, _ = _fresh_run(ctx, "nucleus", clear_peft=True, kl_coeff=0.0)
    # The new run did warn, so the filter below matches the real message.
    assert len(_skipped_clear_peft_warnings(caplog)) == 1
    last = _save_last(run, ctx.tmp_path / "last.ckpt")

    caplog.clear()
    resumed = _setup_agent(
        ctx, seed=30, sampling_mode="nucleus", clear_peft=True, kl_coeff=0.0
    )
    resumed.load(last, load_env=False, load_training_state=True)
    with caplog.at_level(logging.WARNING, logger=PRIOR_AGENT_LOGGER):
        _fit_start(resumed)

    assert not _skipped_clear_peft_warnings(caplog)
    assert torch.equal(_student_logits(resumed), _student_logits(run))


# ---------------------------------------------------------------------------
# Which reference a run holds: a new run pins the policy it starts from, a
# resume keeps its saved one, and SFT runs and reference-free loads hold none
# ---------------------------------------------------------------------------


def _info_messages(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO
    ]


@pytest.mark.parametrize("agent", list(AGENT_CLASSES))
@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize(
    ("sampling_mode", "kl_coeff"), [("prior_constraint", 0.0), ("nucleus", 0.1)]
)
@pytest.mark.parametrize("checkpoint", ["last", "adapter"])
def test_warm_start_from_file_with_reference_pins_the_loaded_policy(
    ctx, caplog, checkpoint, sampling_mode, kl_coeff, clear_peft, agent
):
    """A new run ignores the reference stored in the file it starts from."""
    agent_cls = AGENT_CLASSES[agent]
    # The source run pinned its reference and then trained, so the reference
    # it saves differs from its trained policy.
    run, saved_reference = _fresh_run(
        ctx, sampling_mode, agent_cls=agent_cls, kl_coeff=kl_coeff
    )
    trained = _student_logits(run)
    assert not torch.allclose(saved_reference, trained)
    path = (
        _save_last(run, ctx.tmp_path / "last.ckpt")
        if checkpoint == "last"
        else _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    )

    warm = _setup_agent(
        ctx,
        seed=20,
        sampling_mode=sampling_mode,
        clear_peft=clear_peft,
        agent_cls=agent_cls,
        kl_coeff=kl_coeff,
    )
    _warm_start(warm, path)
    assert torch.equal(_reference_logits(warm), saved_reference)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=PRIOR_AGENT_LOGGER):
        _fit_start(warm)

    assert torch.equal(_reference_logits(warm), trained)
    expected_student = _frozen_prior_logits(warm) if clear_peft else trained
    assert torch.equal(_student_logits(warm), expected_student)
    assert not _skipped_clear_peft_warnings(caplog)
    assert any(
        "stored in the loaded checkpoint is not used" in message
        for message in _info_messages(caplog)
    )


@pytest.mark.parametrize(
    ("sampling_mode", "kl_coeff"), [("prior_constraint", 0.0), ("nucleus", 0.1)]
)
def test_resume_keeps_saved_reference_and_never_clears(
    ctx, caplog, sampling_mode, kl_coeff
):
    run, saved_reference = _fresh_run(
        ctx, sampling_mode, clear_peft=True, kl_coeff=kl_coeff
    )
    last = _save_last(run, ctx.tmp_path / "last.ckpt")

    resumed = _setup_agent(
        ctx, seed=30, sampling_mode=sampling_mode, clear_peft=True, kl_coeff=kl_coeff
    )
    resumed.load(last, load_env=False, load_training_state=True)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=PRIOR_AGENT_LOGGER):
        _fit_start(resumed)

    assert torch.equal(_reference_logits(resumed), saved_reference)
    # clear_peft applied when the run started, not again on resume.
    assert torch.equal(_student_logits(resumed), _student_logits(run))
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any(
        "keeps the reference policy saved" in message
        for message in _info_messages(caplog)
    )


def test_student_film_norm_keeps_updating_after_pinning(ctx):
    """The student's FiLM input normalizer keeps learning; the reference's does not."""
    if not ctx.adapter["film_input_norm"]:
        pytest.skip("this adapter variant has no FiLM input normalizer")
    # Warm-starting from SFT pins a reference with trained FiLM weights, so its
    # logits depend on its FiLM input normalizer.
    sft_last, _, _ = _sft_checkpoints(ctx)
    run = _setup_agent(ctx, seed=60, sampling_mode="prior_constraint")
    _warm_start_and_fit_start(run, sft_last)
    reference = _reference_logits(run)

    assert _training_forward_moves_film_norm(run)
    assert torch.equal(_reference_logits(run), reference)


@pytest.mark.parametrize(
    ("checkpoint", "how"),
    [("last", "load"), ("adapter", "load"), ("adapter", "load_adapter_checkpoint")],
)
def test_sft_drops_a_loaded_reference(ctx, checkpoint, how):
    """SFT never samples against a reference, so it neither keeps nor saves one."""
    rlft, _ = _fresh_run(ctx, "prior_constraint")
    path = (
        _save_last(rlft, ctx.tmp_path / "rlft_last.ckpt")
        if checkpoint == "last"
        else _save_adapter_checkpoint(rlft, ctx.tmp_path / "rlft_inference_last.ckpt")
    )
    assert _has_reference_keys(path)

    sft = _setup_sft_agent(ctx, seed=50)
    if how == "load":
        sft.load(path, load_env=False, load_training_state=False)
    else:
        sft.load_adapter_checkpoint(path)

    assert not _peft(sft).reference_ready
    assert torch.equal(_student_logits(sft), _student_logits(rlft))
    for saved in _save_real_sft_files(sft, ctx.tmp_path):
        assert not _has_reference_keys(saved)


def test_load_adapter_checkpoint_without_reference_clears_previous_reference(ctx):
    """New adapter weights must not run against an earlier policy's reference."""
    agent, _ = _fresh_run(ctx, "prior_constraint")
    _, sft_adapter_checkpoint, sft_policy = _sft_checkpoints(ctx)
    assert _peft(agent).reference_ready
    assert not _has_reference_keys(sft_adapter_checkpoint)

    agent.load_adapter_checkpoint(sft_adapter_checkpoint)

    assert not _peft(agent).reference_ready
    assert agent.model._actor.reference_peft_model is None
    assert torch.equal(_student_logits(agent), sft_policy)
    with pytest.raises(RuntimeError, match="requires a frozen reference"):
        _sample(agent)


# ---------------------------------------------------------------------------
# The reference's own frozen copy of the conditioning network
# ---------------------------------------------------------------------------


@torch.no_grad()
def _drift_student_conditioning(agent, steps=40):
    """Move the student conditioning network's normalizer, as PPO minibatches do.

    Each training-mode pass sees shifted task observations.
    """
    agent.model.train()
    for step in range(steps):
        obs = _obs(1000 + step)
        obs["task_obs"] = obs["task_obs"] * 3 + 4
        agent.model._actor.build_prior_input(TensorDict(obs, batch_size=NUM_ENVS))
    agent.model.eval()


def _pinned_sft_run(ctx, sampling_mode="prior_constraint"):
    """A new run warm-started from SFT, so its reference has trained FiLM weights.

    A reference with fresh adapters ignores its condition, so only a trained
    one shows whether the condition it reads moves.
    """
    sft_last, _, _ = _sft_checkpoints(ctx)
    run = _setup_agent(ctx, seed=60, sampling_mode=sampling_mode)
    _warm_start_and_fit_start(run, sft_last)
    return run


def _drop_conditioning_copy(path):
    """Strip the copy, as in files written before the reference kept one."""
    state = torch.load(path, weights_only=False)
    state["model"] = {
        key: value
        for key, value in state["model"].items()
        if "reference_peft_model." not in key
    }
    torch.save(state, path)
    return path


@functools.lru_cache(maxsize=None)
def _migration_script():
    """Import the migration script by path: it is a script, not a package."""
    path = Path(__file__).resolve().parents[2] / MIGRATION_SCRIPT
    spec = importlib.util.spec_from_file_location("_migrate_peft_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("sampling_mode", SAMPLING_MODES)
def test_reference_ignores_student_conditioning_drift(ctx, sampling_mode):
    run = _pinned_sft_run(ctx, sampling_mode)
    reference, student = _reference_logits(run), _student_logits(run)

    _drift_student_conditioning(run)

    assert not torch.equal(_student_logits(run), student)
    assert torch.equal(_reference_logits(run), reference)


def test_full_and_adapter_files_store_the_same_conditioning_copy(ctx):
    """Both files store the copy pinned at training start, not the student's network."""
    run = _pinned_sft_run(ctx)
    pinned = copy.deepcopy(run.model._actor.actor_peft_model.state_dict())
    _drift_student_conditioning(run)
    full = torch.load(_save_last(run, ctx.tmp_path / "last.ckpt"), weights_only=False)
    adapter = torch.load(
        _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt"),
        weights_only=False,
    )

    for prefix, model_state in (("_actor.", full["model"]), ("", adapter["model"])):
        copy_keys = {
            key
            for key in model_state
            if key.startswith(prefix + "reference_peft_model.")
        }
        assert copy_keys == {prefix + "reference_peft_model." + key for key in pinned}
        for key, value in pinned.items():
            assert torch.equal(model_state[prefix + "reference_peft_model." + key], value)
        # The student's own network kept learning after the pin.
        assert any(
            not torch.equal(model_state[prefix + "actor_peft_model." + key], value)
            for key, value in pinned.items()
        )


@pytest.mark.parametrize(
    ("checkpoint", "how"),
    [("last", "inference"), ("adapter", "inference"), ("last", "resume")],
)
def test_older_files_are_refused_until_the_migration_script_runs(ctx, checkpoint, how):
    """A file without the copy is refused, naming the script; the migrated
    file holds exactly the reference older code computed.

    Older code evaluated the reference with the student's live conditioning
    network, so a file it saved holds the network that reference read at
    save time. The refusal changes nothing in the loading agent; the script
    copies that network under the reference prefix into a new file, loading
    it reproduces that reference, and from then on the reference no longer
    drifts.
    """
    run = _pinned_sft_run(ctx)
    _drift_student_conditioning(run)
    _train(run, seed=2)
    older = copy.deepcopy(run)
    older_actor = older.model._actor
    older_actor.reference_peft_model.load_state_dict(
        older_actor.actor_peft_model.state_dict()
    )
    older_reference = _reference_logits(older)
    assert not torch.equal(older_reference, _reference_logits(run))
    path = _drop_conditioning_copy(
        _save_last(run, ctx.tmp_path / "last.ckpt")
        if checkpoint == "last"
        else _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    )

    loaded = _setup_agent(ctx, seed=70, sampling_mode="prior_constraint")

    def load(file):
        if how == "inference":
            _inference_load(loaded, file)
        else:
            loaded.load(file, load_env=False, load_training_state=True)
            _fit_start(loaded)

    before = copy.deepcopy(loaded.model.state_dict())
    with pytest.raises(RuntimeError, match=re.escape(MIGRATION_SCRIPT)):
        load(path)
    after = loaded.model.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key], value)
    assert not _peft(loaded).reference_ready

    migrated = ctx.tmp_path / "migrated.ckpt"
    _migration_script().main([str(path), str(migrated)])
    load(migrated)
    assert torch.equal(_reference_logits(loaded), older_reference)
    _drift_student_conditioning(loaded)
    assert torch.equal(_reference_logits(loaded), older_reference)
    # The script writes a new file only: nothing is overwritten.
    with pytest.raises(SystemExit, match="already exists"):
        _migration_script().main([str(path), str(migrated)])


@pytest.mark.parametrize("checkpoint", ["last", "adapter", "sft"])
def test_the_migration_script_refuses_a_file_that_needs_none(ctx, checkpoint):
    """A file whose reference has its own conditioning network, or that stores
    no reference at all (an SFT file), is left alone."""
    if checkpoint == "sft":
        path, _, _ = _sft_checkpoints(ctx)
    else:
        run, _ = _fresh_run(ctx, "prior_constraint")
        path = (
            _save_last(run, ctx.tmp_path / "last.ckpt")
            if checkpoint == "last"
            else _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
        )
    out = ctx.tmp_path / "migrated.ckpt"
    with pytest.raises(SystemExit, match="needs no migration"):
        _migration_script().main([str(path), str(out)])
    assert not out.exists()


@pytest.mark.parametrize("checkpoint", ["last", "adapter"])
def test_partial_conditioning_copy_is_rejected(ctx, checkpoint):
    """A file with part of the copy is a key mismatch, not an older file."""
    run, _ = _fresh_run(ctx, "prior_constraint")
    path = (
        _save_last(run, ctx.tmp_path / "last.ckpt")
        if checkpoint == "last"
        else _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    )
    state = torch.load(path, weights_only=False)
    dropped = next(key for key in state["model"] if "reference_peft_model." in key)
    del state["model"][dropped]
    torch.save(state, path)

    inference = _setup_agent(ctx, seed=20, sampling_mode="prior_constraint")
    with pytest.raises(RuntimeError, match=re.escape(f"missing_keys=['{dropped}']")):
        _inference_load(inference, path)
    assert not _peft(inference).reference_ready


@pytest.mark.parametrize("checkpoint", ["last", "adapter"])
def test_conditioning_copy_without_the_rest_of_the_reference_is_rejected(
    ctx, checkpoint
):
    """A file holding only part of a reference fails before any weights change."""
    run, _ = _fresh_run(ctx, "prior_constraint")
    path = (
        _save_last(run, ctx.tmp_path / "last.ckpt")
        if checkpoint == "last"
        else _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    )
    state = torch.load(path, weights_only=False)
    state["model"] = {
        key: value
        for key, value in state["model"].items()
        if "reference_peft_model." in key or not _is_reference_key(key)
    }
    torch.save(state, path)

    warm = _setup_agent(ctx, seed=20, sampling_mode="prior_constraint")
    before = copy.deepcopy(warm.model.state_dict())
    with pytest.raises(
        RuntimeError, match=r"missing_keys=\[[^\]]*prior_with_peft\.reference_prior\."
    ):
        _warm_start(warm, path)
    for key, value in before.items():
        assert torch.equal(warm.model.state_dict()[key], value)
    assert not _peft(warm).reference_ready


def _stateless_conditioning_network():
    """A conditioning network that saves no state: no normalizer and no weights."""
    return ModuleContainerConfig(
        in_keys=["task_obs"],
        out_keys=[DEFAULT_PEFT_CONDITION_KEY],
        models=[
            ObsProcessorConfig(
                in_keys=["task_obs"],
                out_keys=[DEFAULT_PEFT_CONDITION_KEY],
                normalize_obs=False,
                module_operations=[ModuleOperationForwardConfig()],
            )
        ],
    )


@pytest.mark.parametrize("checkpoint", ["last", "adapter"])
def test_conditioning_network_without_state_is_not_mistaken_for_an_older_file(
    ctx, checkpoint
):
    """A copy with no state saves no keys, which must not read as a missing copy
    (the file loads; the loaders refuse an older file, the script migrates it)."""
    peft_model = _stateless_conditioning_network()
    run = _setup_agent(
        ctx, seed=10, sampling_mode="prior_constraint", peft_model=peft_model
    )
    run.load(None, load_training_state=False)
    _fit_start(run)
    _train(run, seed=1)
    assert not run.model._actor.actor_peft_model.state_dict()
    path = (
        _save_last(run, ctx.tmp_path / "last.ckpt")
        if checkpoint == "last"
        else _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    )

    loaded = _setup_agent(
        ctx, seed=20, sampling_mode="prior_constraint", peft_model=peft_model
    )
    _inference_load(loaded, path)
    with pytest.raises(SystemExit, match="needs no migration"):
        _migration_script().main([str(path), str(ctx.tmp_path / "migrated.ckpt")])

    _assert_same_policy(loaded, run)


# ---------------------------------------------------------------------------
# reference_checkpoint: a new run anchors to another file's trained policy
# ---------------------------------------------------------------------------


def _anchor_file(ctx, anchor):
    """A file to anchor to, and the logits of the trained policy it saved.

    ``anchor`` names the run that wrote the file, SFT or RLFT, and the kind of
    file. The RLFT run started from SFT and then trained, its conditioning
    network included, so the reference it stores is not its trained policy.
    """
    run_kind, file_kind = anchor.split("_")
    if run_kind == "sft":
        last, adapter, policy = _sft_checkpoints(ctx)
        return (last if file_kind == "last" else adapter), policy
    rlft = _pinned_sft_run(ctx)
    _drift_student_conditioning(rlft)
    _train(rlft, seed=2)
    policy = _student_logits(rlft)
    assert not torch.allclose(_reference_logits(rlft), policy)
    path = (
        _save_last(rlft, ctx.tmp_path / "anchor_last.ckpt")
        if file_kind == "last"
        else _save_adapter_checkpoint(rlft, ctx.tmp_path / "anchor_inference_last.ckpt")
    )
    return path, policy


@pytest.mark.parametrize("clear_peft", [False, True])
@pytest.mark.parametrize("anchor", ["sft_last", "sft_adapter", "rlft_last", "rlft_adapter"])
def test_reference_checkpoint_pins_that_files_trained_policy(
    ctx, caplog, anchor, clear_peft
):
    """A new run anchors to the file's trained policy, not to its stored reference."""
    anchor_path, anchor_policy = _anchor_file(ctx, anchor)
    # The run starts from a third policy, so neither the reference nor the
    # student can match by accident.
    start, _ = _fresh_run(ctx, "prior_constraint")
    start_last = _save_last(start, ctx.tmp_path / "start_last.ckpt")
    run = _setup_agent(
        ctx,
        seed=80,
        sampling_mode="prior_constraint",
        clear_peft=clear_peft,
        reference_checkpoint=str(anchor_path),
    )
    _warm_start(run, start_last)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=PRIOR_AGENT_LOGGER):
        _fit_start(run)

    assert torch.equal(_reference_logits(run), anchor_policy)
    expected_student = _frozen_prior_logits(run) if clear_peft else _student_logits(start)
    assert torch.equal(_student_logits(run), expected_student)
    assert any(str(anchor_path) in message for message in _info_messages(caplog))

    # Both files the run writes restore that reference.
    _train(run, seed=9)
    last = _save_last(run, ctx.tmp_path / "last.ckpt")
    adapter = _save_adapter_checkpoint(run, ctx.tmp_path / "inference_last.ckpt")
    via_last, via_adapter_checkpoint = _load_full_and_adapter_checkpoint(
        ctx, last, adapter, _inference_load, sampling_mode="prior_constraint"
    )
    _assert_same_policy(via_adapter_checkpoint, via_last)
    assert torch.equal(_reference_logits(via_adapter_checkpoint), anchor_policy)

    # A resume keeps its saved reference and never opens the option's file.
    resumed = _setup_agent(
        ctx,
        seed=30,
        sampling_mode="prior_constraint",
        clear_peft=clear_peft,
        reference_checkpoint=str(ctx.tmp_path / "missing.ckpt"),
    )
    resumed.load(last, load_env=False, load_training_state=True)
    _fit_start(resumed)
    assert torch.equal(_reference_logits(resumed), anchor_policy)
    assert torch.equal(_student_logits(resumed), _student_logits(run))


@pytest.mark.parametrize("problem", ["missing_file", "not_a_peft_file", "wrong_shape"])
def test_unusable_reference_checkpoint_fails_before_anything_is_pinned(ctx, problem):
    """The error names the option, and the run's weights stay as they were."""
    sft_last, sft_adapter, _ = _sft_checkpoints(ctx)
    if problem == "missing_file":
        path, reason = ctx.tmp_path / "missing.ckpt", "No such file"
    elif problem == "not_a_peft_file":
        path, reason = ctx.prior_path, "holds no PEFT adapter"
    else:
        state = torch.load(sft_adapter, weights_only=False)
        key = next(key for key in state["model"] if ".lora." in key)
        state["model"][key] = torch.ones(1, 1, 1)
        path, reason = ctx.tmp_path / "wrong_shape.ckpt", "shape_mismatches"
        torch.save(state, path)
    run = _setup_agent(
        ctx, seed=80, sampling_mode="prior_constraint", reference_checkpoint=str(path)
    )
    _warm_start(run, sft_last)
    before = copy.deepcopy(run.model.state_dict())

    message = re.escape(
        f"agent.model.actor.peft.reference_checkpoint={path} cannot be used as "
        "the reference policy: "
    )
    with pytest.raises(RuntimeError, match=f"{message}.*{reason}"):
        _fit_start(run)

    assert not _peft(run).reference_ready
    assert before.keys() == run.model.state_dict().keys()
    for key, value in before.items():
        assert torch.equal(run.model.state_dict()[key], value), key


def test_reference_checkpoint_is_ignored_with_a_warning_when_nothing_reads_it(
    ctx, caplog
):
    # The file does not exist: a run that reads no reference never opens it.
    missing = ctx.tmp_path / "missing.ckpt"
    run = _setup_agent(
        ctx,
        seed=80,
        sampling_mode="nucleus",
        kl_coeff=0.0,
        reference_checkpoint=str(missing),
    )
    run.load(None, load_training_state=False)
    with caplog.at_level(logging.WARNING, logger=PRIOR_AGENT_LOGGER):
        _fit_start(run)

    assert not _peft(run).reference_ready
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"reference_checkpoint={str(missing)!r} was ignored")
    assert "sampling_mode='nucleus' and kl_coeff=0" in warnings[0]
