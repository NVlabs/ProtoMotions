# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for discrete-latent prior configuration."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import nn
from torch.nn.parameter import UninitializedParameter

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.supervised.latent_prior_config import (
    DiscreteAutoregressiveLatentSupervisedAgentConfig,
    DiscreteAutoregressiveLatentPriorModelConfig,
)
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.common.autoregressive import (
    DiscreteAutoregressiveTransformer,
)
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPLayerConfig,
    MLPWithConcatConfig,
    ModuleContainerConfig,
    PretrainedModelConfig,
)
from protomotions.agents.common.discrete_latent import FSQTokenization
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.common.supervision import SupervisionLossType
from protomotions.agents.common.pretrained import load_pretrained_model_module
from protomotions.agents.supervised.latent_prior_model import (
    DiscreteAutoregressiveLatentPriorModel,
)
from protomotions.agents.optimizer.muon import MuonWithAuxAdam


class DummyLatentEncoder(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["target"]
        self.out_keys = ["latent"]
        self.proj = nn.Linear(4, 4)

    def forward(self, tensordict):
        tensordict["latent"] = self.proj(tensordict["target"])
        return tensordict


class DummyLazyLatentEncoder(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["target"]
        self.out_keys = ["latent"]
        self.proj = nn.LazyLinear(4)

    def forward(self, tensordict):
        tensordict["latent"] = self.proj(tensordict["target"])
        return tensordict


class DummyLatentDecoder(TensorDictModuleBase):
    def __init__(self):
        super().__init__()
        self.in_keys = ["state", "latent"]
        self.out_keys = ["action"]
        self.proj = nn.Linear(4, 4)

    def forward(self, tensordict):
        tensordict["action"] = tensordict["state"] + self.proj(tensordict["latent"])
        return tensordict


class DummyDiscreteQuantizer(nn.Module):
    num_fsq_levels = 3
    num_fsq_scalars = 4

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def quantize(self, latent):
        return latent * self.scale

    def codes_to_indices(self, codes):
        return codes.long().clamp(min=0, max=self.num_fsq_levels - 1)

    def indices_to_codes(self, indices):
        return indices.float()


class DummyDiscreteCodec(nn.Module):
    latent_key = "latent"

    def __init__(self):
        super().__init__()
        self.encoder = DummyLatentEncoder()
        self.decoder = DummyLatentDecoder()
        self.quantizer = DummyDiscreteQuantizer()


class DummyCheckpointActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = DummyDiscreteCodec()


class DummyCheckpointModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.actor = DummyCheckpointActor()

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        return tensordict


@dataclass
class DummyDiscreteCodecConfig:
    _target_: str = (
        "protomotions.tests.test_discrete_latent_prior_config."
        "ConfigurableDummyDiscreteCodec"
    )


class ConfigurableDummyDiscreteCodec(DummyDiscreteCodec):
    def __init__(self, config):
        super().__init__()


class ConfigurableDummyLazyDiscreteCodec(DummyDiscreteCodec):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.encoder = DummyLazyLatentEncoder()
        self.decoder = DummyLatentDecoder()
        self.quantizer = DummyDiscreteQuantizer()


@dataclass
class DummyCheckpointModelConfig:
    _target_: str = (
        "protomotions.tests.test_discrete_latent_prior_config."
        "DummyCheckpointModel"
    )


@dataclass
class DummyAgentConfig:
    model: DummyCheckpointModelConfig


@dataclass
class DummyLazyDiscreteCodecConfig:
    _target_: str = (
        "protomotions.tests.test_discrete_latent_prior_config."
        "ConfigurableDummyLazyDiscreteCodec"
    )


def _small_prior_config(checkpoint_path) -> DiscreteAutoregressiveLatentPriorModelConfig:
    config = DiscreteAutoregressiveLatentPriorModelConfig(
        latent_decoder=PretrainedModelConfig(
            checkpoint_path=str(checkpoint_path),
            module_path="actor.mu",
        ),
        prior=DiscreteAutoregressiveTransformerConfig(
            in_keys=["state", "prior_tokens"],
            out_keys=[LATENT_LOGITS_KEY],
            context_key="state",
            token_key="prior_tokens",
            logits_key=LATENT_LOGITS_KEY,
            d_model=8,
            num_heads=2,
            num_layers=1,
            ff_size=16,
            dropout=0.0,
            num_tokens=2,
            vocab_size=9,
        ),
        fsq_scalars_per_prior_token=2,
    )
    config.prior.context_encoder.models[0].normalize_obs = True
    return config


def test_prior_agent_config_does_not_own_muon_settings():
    field_names = set(DiscreteAutoregressiveLatentSupervisedAgentConfig.__dataclass_fields__)

    assert "muon" not in field_names
    assert (
        DiscreteAutoregressiveLatentSupervisedAgentConfig._target_
        == "protomotions.agents.supervised.agent.SupervisedAgent"
    )


def test_prior_model_uses_generic_optimizer_field():
    model_fields = DiscreteAutoregressiveLatentPriorModelConfig.__dataclass_fields__

    assert "optimizer" in model_fields
    assert "actor_optimizer" not in model_fields
    assert "latent_decoder" in model_fields
    assert "codec" not in model_fields
    assert "autoencoder" not in model_fields
    assert "autoencoder_checkpoint" not in model_fields
    assert "tracker_checkpoint" not in model_fields
    assert "actor" not in model_fields
    assert "context_normalizer_source" not in model_fields


def test_prior_context_encoder_owns_context_normalization(tmp_path):
    config = _small_prior_config(tmp_path / "missing.ckpt")
    config.latent_decoder.module_config = DummyDiscreteCodecConfig()
    model = DiscreteAutoregressiveLatentPriorModel(config)
    model.eval()
    context_encoder = model.prior._context_encoder.models[0]

    td = TensorDict(
        {
            "state": torch.zeros(2, 4),
            "target": torch.zeros(2, 4),
        },
        batch_size=2,
    )

    out = model.materialize(td)

    assert "action" in out.keys()
    assert not hasattr(model, "context_normalizer")
    assert context_encoder.norm.running_obs_norm._initialized
    assert context_encoder.norm.running_obs_norm.mean.shape == (4,)


def test_prior_derives_multiple_context_inputs_from_prior_config(tmp_path):
    config = _small_prior_config(tmp_path / "missing.ckpt")
    config.latent_decoder.module_config = DummyDiscreteCodecConfig()
    config.prior.in_keys = ["state", "phase", "prior_tokens"]
    config.prior.context_encoder = ModuleContainerConfig(
        in_keys=["state", "phase"],
        out_keys=[config.prior.context_embedding_key],
        models=[
            MLPWithConcatConfig(
                in_keys=["state", "phase"],
                out_keys=[config.prior.context_embedding_key],
                num_out=8,
                normalize_obs=True,
                layers=[MLPLayerConfig(units=8, activation="gelu")],
            )
        ],
    )
    model = DiscreteAutoregressiveLatentPriorModel(config)
    model.eval()

    td = TensorDict(
        {
            "state": torch.zeros(2, 4),
            "phase": torch.zeros(2, 2),
            "target": torch.zeros(2, 4),
        },
        batch_size=2,
    )

    out = model.materialize(td)

    assert "action" in out.keys()
    assert model.prior_context_keys == ["state", "phase"]
    assert model.in_keys == ["state", "phase"]


def test_prior_setup_materialize_initializes_embedded_target_encoder(tmp_path):
    config = _small_prior_config(tmp_path / "missing.ckpt")
    config.latent_decoder.module_config = DummyLazyDiscreteCodecConfig()
    model = DiscreteAutoregressiveLatentPriorModel(config)
    model.eval()

    assert isinstance(
        model.target_latent_encoder.encoder.proj.weight,
        UninitializedParameter,
    )

    td = TensorDict(
        {
            "state": torch.zeros(2, 4),
            "target": torch.zeros(2, 4),
        },
        batch_size=2,
    )

    model.materialize(td)

    assert not isinstance(
        model.target_latent_encoder.encoder.proj.weight,
        UninitializedParameter,
    )


def test_prior_model_uses_strict_standard_state_dict_loading():
    from protomotions.agents.supervised.latent_prior_model import (
        DiscreteAutoregressiveLatentPriorModel,
    )

    assert "load_state_dict" not in DiscreteAutoregressiveLatentPriorModel.__dict__


def test_pretrained_loader_requires_checkpoint_and_resolved_config(tmp_path):
    model_config = DummyCheckpointModelConfig()
    model = DummyCheckpointModel(model_config)
    checkpoint_path = tmp_path / "tracker.ckpt"
    torch.save(
        {"agent": DummyAgentConfig(model=model_config)},
        tmp_path / "resolved_configs.pt",
    )
    torch.save({"model": model.state_dict()}, checkpoint_path)

    config = PretrainedModelConfig(
        checkpoint_path=str(checkpoint_path),
        module_path="actor.mu",
    )

    module = load_pretrained_model_module(config, device=torch.device("cpu"))

    assert isinstance(module, DummyDiscreteCodec)


def test_prior_model_crashes_when_latent_decoder_checkpoint_is_missing(
    tmp_path,
):
    config = _small_prior_config(tmp_path / "missing.ckpt")

    try:
        DiscreteAutoregressiveLatentPriorModel(config)
    except FileNotFoundError as error:
        assert "Pretrained model checkpoint not found" in str(error)
    else:
        raise AssertionError("Expected missing pretrained decoder checkpoint to fail")


def test_prior_model_instantiates_embedded_latent_decoder_without_checkpoint(tmp_path):
    config = _small_prior_config(tmp_path / "missing.ckpt")
    config.latent_decoder.module_config = DummyDiscreteCodecConfig()

    model = DiscreteAutoregressiveLatentPriorModel(config)

    assert model.latent_decoder.num_fsq_levels == DummyDiscreteQuantizer.num_fsq_levels
    assert model.latent_decoder.num_fsq_scalars == DummyDiscreteQuantizer.num_fsq_scalars
    assert model.target_latent_encoder.in_keys == ["target"]


def test_prior_model_instantiates_embedded_latent_decoder_without_checkpoint_path():
    config = _small_prior_config("")
    config.latent_decoder.module_config = DummyDiscreteCodecConfig()

    model = DiscreteAutoregressiveLatentPriorModel(config)

    assert model.latent_decoder.num_fsq_levels == DummyDiscreteQuantizer.num_fsq_levels
    assert model.latent_decoder.num_fsq_scalars == DummyDiscreteQuantizer.num_fsq_scalars


def test_prior_config_embeds_latent_decoder_module_config_for_inference(tmp_path):
    tracker_module_config = DummyDiscreteCodecConfig()
    tracker_model_config = SimpleNamespace(
        actor=SimpleNamespace(mu_model=tracker_module_config)
    )
    tracker_checkpoint = tmp_path / "tracker.ckpt"
    torch.save(
        {"agent": DummyAgentConfig(model=tracker_model_config)},
        tmp_path / "resolved_configs.pt",
    )

    config = _small_prior_config(tracker_checkpoint)
    config.latent_decoder.module_config_path = "agent.model.actor.mu_model"

    config.prepare_inference_config_for_save()

    assert isinstance(config.latent_decoder.module_config, DummyDiscreteCodecConfig)
    assert config.latent_decoder.module_config._target_ == tracker_module_config._target_
    assert config.latent_decoder.checkpoint_path == ""


def test_prior_model_rejects_missing_decoder_source_and_token_only_prior(tmp_path):
    config = _small_prior_config("")
    config.latent_decoder.module_config = None

    try:
        DiscreteAutoregressiveLatentPriorModel(config)
    except ValueError as error:
        assert "checkpoint_path or module_config must be set" in str(error)
    else:
        raise AssertionError("Expected missing latent decoder source to fail")

    config = _small_prior_config(tmp_path / "missing.ckpt")
    config.latent_decoder.module_config = DummyDiscreteCodecConfig()
    config.prior.in_keys = [config.prior.token_key]

    try:
        DiscreteAutoregressiveLatentPriorModel(config)
    except ValueError as error:
        assert "at least one non-token input key" in str(error)
    else:
        raise AssertionError("Expected token-only prior inputs to fail")


def test_prior_model_runtime_paths_with_embedded_discrete_codec(tmp_path):
    config = _small_prior_config(tmp_path / "missing.ckpt")
    config.latent_decoder.module_config = DummyDiscreteCodecConfig()
    config.rollout_action_std = 0.05
    model = DiscreteAutoregressiveLatentPriorModel(config)
    model.eval()
    state = torch.zeros(2, 4)
    target = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    assert model.reset_rollout_context() is None
    assert model.optimization_module() is model.prior
    assert model.get_inference_in_keys() == ["state"]

    model.train(True)
    assert model.training
    assert model.prior.training
    assert not model.latent_decoder.training
    assert not model.target_latent_encoder.training
    model.eval()

    forced = model(
        TensorDict(
            {
                "state": state.clone(),
                TARGET_LATENT_KEY: target.clone(),
            },
            batch_size=2,
        )
    )
    assert forced["action"].shape == (2, 4)
    assert forced["mean_action"].shape == (2, 4)
    assert forced["neglogp"].shape == (2,)
    assert forced[LATENT_LOGITS_KEY].shape == (2, 2, 9)
    assert torch.equal(forced[TARGET_LATENT_KEY], target)

    prefilled_action = torch.full((2, 4), 9.0)
    forced_with_action = model._teacher_forced(
        TensorDict(
            {
                "state": state.clone(),
                "action": prefilled_action.clone(),
                TARGET_LATENT_KEY: target.clone(),
            },
            batch_size=2,
        )
    )
    assert torch.equal(forced_with_action["action"], prefilled_action)
    assert "neglogp" not in forced_with_action.keys()

    generated = model(TensorDict({"state": state.clone()}, batch_size=2))
    assert generated["action"].shape == (2, 4)
    assert generated["neglogp"].shape == (2,)
    assert torch.equal(generated["neglogp"], torch.zeros(2))
    assert generated[LATENT_KEY].shape == (2, 2)
    assert generated[LATENT_LOGITS_KEY].shape == (2, 2, 9)

    action, payload = model.generate(TensorDict({"state": state.clone()}, batch_size=2))
    assert torch.equal(action, payload["action"])
    assert payload[LATENT_KEY].shape == (2, 2)
    assert payload[LATENT_LOGITS_KEY].shape == (2, 2, 9)

    reconstructed_action, reconstruction = model.reconstruct(
        TensorDict(
            {
                "state": state.clone(),
                TARGET_LATENT_KEY: target.clone(),
            },
            batch_size=2,
        )
    )
    assert torch.equal(reconstructed_action, reconstruction["action"])
    assert torch.equal(reconstruction[LATENT_KEY], target)

    expert = model.collect_expert_rollout(
        TensorDict(
            {
                "state": state.clone(),
                "target": torch.zeros(2, 4),
            },
            batch_size=2,
        )
    )
    assert expert["action"].shape == (2, 4)
    assert expert["neglogp"].shape == (2,)
    assert expert[TARGET_LATENT_KEY].shape == (2, 2)

    acted = model.act(TensorDict({"state": state.clone()}, batch_size=2), mean=False)
    assert acted["action"].shape == (2, 4)


def test_prior_supervised_loss_matches_discrete_latents():
    config = DiscreteAutoregressiveLatentSupervisedAgentConfig(
        batch_size=1,
        training_max_steps=1,
    )

    assert config.rollout_actor is RolloutActor.EXPERT
    assert config.loss.loss_type == SupervisionLossType.DISCRETE_CROSS_ENTROPY
    assert config.loss.prediction_key == LATENT_LOGITS_KEY
    assert config.loss.target_key == TARGET_LATENT_KEY
    assert config.loss.label_smoothing > 0


def test_prior_supervised_config_does_not_override_base_training_defaults():
    field_names = set(DiscreteAutoregressiveLatentSupervisedAgentConfig.__dataclass_fields__)

    assert "num_steps" in field_names
    assert "num_mini_epochs" in field_names
    assert "task_reward_w" in field_names
    assert (
        "num_steps"
        not in DiscreteAutoregressiveLatentSupervisedAgentConfig.__dict__.get(
            "__annotations__",
            {},
        )
    )
    assert (
        "num_mini_epochs"
        not in DiscreteAutoregressiveLatentSupervisedAgentConfig.__dict__.get(
            "__annotations__",
            {},
        )
    )
    assert (
        "task_reward_w"
        not in DiscreteAutoregressiveLatentSupervisedAgentConfig.__dict__.get(
            "__annotations__",
            {},
        )
    )


def test_prior_forward_generates_when_no_teacher_forcing_sequence_in_training():
    model = object.__new__(DiscreteAutoregressiveLatentPriorModel)
    torch.nn.Module.__init__(model)
    model.training = True
    model._teacher_forced = lambda td: td.set("path", "teacher_forced")
    model._generate = lambda td: td.set("path", "generated")

    td = TensorDict({"state": torch.randn(2, 3)}, batch_size=2)

    out = DiscreteAutoregressiveLatentPriorModel.forward(model, td)

    assert out["path"] == "generated"


def test_prior_forward_teacher_forces_when_target_sequence_is_available_in_eval():
    model = object.__new__(DiscreteAutoregressiveLatentPriorModel)
    torch.nn.Module.__init__(model)
    model.training = False
    model._teacher_forced = lambda td: td.set("path", "teacher_forced")
    model._generate = lambda td: td.set("path", "generated")

    td = TensorDict(
        {
            "state": torch.randn(2, 3),
            TARGET_LATENT_KEY: torch.zeros(2, 4, dtype=torch.long),
        },
        batch_size=2,
    )

    out = DiscreteAutoregressiveLatentPriorModel.forward(model, td)

    assert out["path"] == "teacher_forced"


def test_prior_forward_does_not_treat_action_as_latent_supervision():
    model = object.__new__(DiscreteAutoregressiveLatentPriorModel)
    torch.nn.Module.__init__(model)
    model.training = False
    model._teacher_forced = lambda td: td.set("path", "teacher_forced")
    model._generate = lambda td: td.set("path", "generated")

    td = TensorDict({"state": torch.randn(2, 3)}, batch_size=2)

    out = DiscreteAutoregressiveLatentPriorModel.forward(model, td)

    assert out["path"] == "generated"


def test_fsq_tokenization_indices_round_trip():
    tokenization = FSQTokenization(
        num_fsq_levels=5,
        num_fsq_scalars=6,
        fsq_scalars_per_prior_token=3,
    )

    fsq_indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0],
            [4, 3, 2, 1, 0, 4],
        ]
    )

    prior_tokens = tokenization.fsq_indices_to_prior_tokens(fsq_indices)
    round_trip = tokenization.prior_tokens_to_fsq_indices(prior_tokens)

    assert torch.equal(round_trip, fsq_indices)


def test_prior_muon_optimizer_hyperparameters_are_config_driven():
    prior = nn.Sequential(
        nn.Linear(3, 4),
        nn.Linear(4, 4),
        nn.LayerNorm(4),
        nn.Linear(4, 2),
    )
    optimizer = MuonWithAuxAdam(
        params=prior,
        lr=0.012,
        weight_decay=0.34,
        momentum=0.91,
        adam_lr=0.056,
        adam_betas=(0.8, 0.88),
        adam_eps=1e-6,
        adam_weight_decay=0.78,
    )
    hidden_group = next(group for group in optimizer.param_groups if group["use_muon"])
    adam_group = next(group for group in optimizer.param_groups if not group["use_muon"])

    assert hidden_group["lr"] == 0.012
    assert hidden_group["weight_decay"] == 0.34
    assert hidden_group["momentum"] == 0.91
    assert adam_group["lr"] == 0.056
    assert adam_group["betas"] == (0.8, 0.88)
    assert adam_group["eps"] == 1e-6
    assert adam_group["weight_decay"] == 0.78


def test_prior_muon_optimizer_keeps_token_and_output_projections_auxiliary():
    prior = DiscreteAutoregressiveTransformer(
        DiscreteAutoregressiveTransformerConfig(
            in_keys=["prior_context", "prior_tokens"],
            out_keys=[LATENT_LOGITS_KEY],
            context_key="prior_context",
            token_key="prior_tokens",
            logits_key=LATENT_LOGITS_KEY,
            d_model=8,
            num_heads=2,
            num_layers=1,
            ff_size=16,
            dropout=0.0,
            num_tokens=4,
            vocab_size=3,
        )
    )

    optimizer = MuonWithAuxAdam(
        params=prior,
    )
    adam_group = next(group for group in optimizer.param_groups if not group["use_muon"])

    adam_params = set(adam_group["params"])
    assert all(param in adam_params for param in prior._token_encoder.parameters())
    assert all(param in adam_params for param in prior._output_head.parameters())
