# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for discrete-prior PEFT actor/model helpers."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.autoregressive import DiscreteAutoregressiveTransformer
from protomotions.agents.common.config import (
    DiscreteAutoregressiveTransformerConfig,
    MLPWithConcatConfig,
)
from protomotions.agents.common.discrete_latent import (
    DiscreteLatentDecoder,
    DiscreteLatentTargetEncoder,
    FSQTokenization,
)
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGPROB_KEY,
)
from protomotions.agents.peft import actor as actor_module
from protomotions.agents.peft.prior_config import (
    DiscretePriorPEFTConfig,
    DiscretePriorPEFTActorConfig,
)
from protomotions.agents.peft.actor import DiscretePriorPEFTActor
from protomotions.agents.peft.model import DiscretePriorPEFTModel
from protomotions.agents.peft.prior_with_peft import DiscretePriorWithPEFT


class _FakeDiscretePriorWithPEFT(nn.Module):
    instances = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.init_calls = 0
        self.init_warmup_obs = None
        self.generate_calls = []
        self.base_prior = kwargs["prior"]
        self.adapter = nn.Module()
        self.adapter.lora = nn.Linear(1, 1, bias=False)
        self.adapter.gamma = nn.Linear(1, 1)
        self.adapter.beta = nn.Linear(1, 1)
        self.adapter.m = nn.Parameter(torch.ones(1))
        self.frozen = nn.Linear(1, 1)
        _FakeDiscretePriorWithPEFT.instances.append(self)

    @staticmethod
    def _context(input_dict):
        return input_dict.get("max_coords_obs", input_dict.get("self_obs"))

    def init_peft(self, warmup_obs=None):
        self.init_calls += 1
        self.init_warmup_obs = warmup_obs

    def forward(self, input_dict):
        condition_key = self.kwargs.get("condition_key", "task_cond")
        return self._context(input_dict) + input_dict[condition_key]

    def generate(self, prior_dict, return_logits=True, return_logprob=False):
        self.generate_calls.append((prior_dict, return_logits, return_logprob))
        context = self._context(prior_dict)
        indices = torch.tensor([[0, 1], [2, 0]], device=context.device)
        logits = torch.tensor(
            [
                [[3.0, 0.0, -1.0], [0.0, 4.0, -2.0]],
                [[-1.0, 0.0, 5.0], [2.0, 0.0, -1.0]],
            ],
            device=context.device,
        )
        if return_logprob:
            return indices, torch.distributions.Categorical(logits=logits).log_prob(
                indices
            )
        return indices, logits


class _FakePEFTInput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_keys = config.in_keys
        self.out_keys = config.out_keys
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        output = torch.cat(
            [tensordict[key] for key in self.in_keys],
            dim=-1,
        ) * self.scale
        if "task_cond" in self.out_keys:
            tensordict["task_cond"] = output
        for key in self.out_keys:
            if key == "task_cond" or key in tensordict:
                continue
            tensordict[key] = output
        return tensordict


class _Quantizer(nn.Module):
    num_fsq_levels = 3
    num_fsq_scalars = 4
    L = 3
    half_L = 1
    half_width = 0.5

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def quantize(self, z):
        return z.round()

    def codes_to_indices(self, codes):
        return codes.long() + 1

    def indices_to_codes(self, indices):
        return indices.float() - 1.0


class _Encoder(nn.Module):
    in_keys = ["mimic_target_poses"]
    out_keys = ["encoded"]

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        tensordict["encoded"] = tensordict["mimic_target_poses"] * self.weight + 0.25
        return tensordict


class _Decoder(nn.Module):
    in_keys = ["max_coords_obs", "latent"]
    out_keys = ["decoded_action"]

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, tensordict):
        tensordict["decoded_action"] = (
            tensordict["latent"].sum(dim=-1, keepdim=True) * self.weight
        )
        return tensordict


def _prior_transformer():
    return DiscreteAutoregressiveTransformer(
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
            num_layers=1,
            ff_size=16,
            dropout=0.0,
            num_tokens=2,
            vocab_size=9,
        )
    )


class _PriorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_tokenization = FSQTokenization(
            num_fsq_levels=3,
            num_fsq_scalars=4,
            fsq_scalars_per_prior_token=2,
        )
        quantizer = _Quantizer()
        self.latent_decoder = DiscreteLatentDecoder(
            decoder=_Decoder(),
            quantizer=quantizer,
            latent_key="latent",
        )
        self.target_latent_encoder = DiscreteLatentTargetEncoder(
            encoder=_Encoder(),
            quantizer=quantizer,
            tokenization=self.latent_tokenization,
            target_key="target_latent",
        )
        self.prior = _prior_transformer()


class _DummyPriorActor:
    def __init__(self, config, pretrained_prior_model, mimic_target_poses_dim):
        self.config = config
        self.pretrained_prior_model = pretrained_prior_model
        self.mimic_target_poses_dim = mimic_target_poses_dim
        self.in_keys = ["max_coords_obs"]
        self.out_keys = ["action", "mean_action", "neglogp", "prior_tokens"]
        self.forward_calls = []

    def __call__(self, input_dict):
        self.forward_calls.append(input_dict)
        return {"teacher_logits": input_dict["teacher"]}

    def generate(self, tensordict):
        batch_size = (
            tensordict.batch_size[0]
            if isinstance(tensordict, TensorDict)
            else tensordict["max_coords_obs"].shape[0]
        )
        action = torch.ones(batch_size, 1)
        return action, {
            "action": action,
            LATENT_KEY: torch.zeros(batch_size, 2, dtype=torch.long),
        }

    def get_action_and_logp(self, tensordict):
        action, output = self.generate(tensordict)
        for key, value in output.items():
            tensordict[key] = value
        tensordict["mean_action"] = action
        tensordict["neglogp"] = torch.zeros(action.shape[0])
        tensordict["prior_tokens"] = output[LATENT_KEY]
        return tensordict


def _actor_peft_model_config(**kwargs):
    defaults = {
        "_target_": "protomotions.tests.test_prior_peft_model._FakePEFTInput",
        "in_keys": [
            "max_coords_obs",
            *_DEFAULT_CONDITIONING_KEYS,
            "custom_terrain",
        ],
        "out_keys": ["task_cond"],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _peft_config(**kwargs):
    defaults = {
        "model": None,
        "condition_key": "task_cond",
        "rank": 2,
        "alpha": 4.0,
        "peft_type": "lora",
        "temperature": 0.75,
        "top_p": 0.8,
        "sampling_mode": "prior_constraint",
        "prior_top_p": 0.6,
        "kl_coeff": 0.25,
        "film_input_norm": False,
        "film_input_norm_clamp": 5.0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


_DEFAULT_CONDITIONING_KEYS = (
    "task_obs",
    "masked_mimic_target_poses",
    "masked_mimic_target_masks",
    "masked_mimic_target_times",
    "historical_pose_obs",
)


def _actor(monkeypatch, mimic_target_poses_dim=2):
    _FakeDiscretePriorWithPEFT.instances.clear()
    monkeypatch.setattr(actor_module, "DiscretePriorWithPEFT", _FakeDiscretePriorWithPEFT)
    actor_peft_model = _actor_peft_model_config()
    return DiscretePriorPEFTActor(
        config=SimpleNamespace(
            peft=_peft_config(model=actor_peft_model),
            in_keys=["max_coords_obs", *_DEFAULT_CONDITIONING_KEYS, "custom_terrain"],
            out_keys=["action", "mean_action", "neglogp", "prior_tokens"],
        ),
        pretrained_prior_model=_PriorModel(),
        mimic_target_poses_dim=mimic_target_poses_dim,
    )


def _adaptive_actor(monkeypatch, task_in_keys=("task_obs",), mimic_target_poses_dim=2):
    _FakeDiscretePriorWithPEFT.instances.clear()
    monkeypatch.setattr(actor_module, "DiscretePriorWithPEFT", _FakeDiscretePriorWithPEFT)
    return DiscretePriorPEFTActor(
        config=DiscretePriorPEFTActorConfig(
            in_keys=list(task_in_keys),
            out_keys=["action", "mean_action", "neglogp", "prior_tokens"],
            peft=DiscretePriorPEFTConfig(
                rank=2,
                alpha=4.0,
                peft_type="lora",
                temperature=0.75,
                top_p=0.8,
                sampling_mode="prior_constraint",
                prior_top_p=0.6,
                kl_coeff=0.25,
            ),
        ),
        pretrained_prior_model=_PriorModel(),
        mimic_target_poses_dim=mimic_target_poses_dim,
    )


def _td():
    return TensorDict(
        {
            "max_coords_obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "task_obs": torch.tensor([[0.1], [0.2]]),
            "masked_mimic_target_poses": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "masked_mimic_target_masks": torch.tensor([[1.0], [0.0]]),
            "masked_mimic_target_times": torch.tensor([[0.5], [0.75]]),
            "historical_pose_obs": torch.tensor([[9.0], [8.0]]),
            "mimic_target_poses": torch.tensor(
                [
                    [-1.25, -1.25, -0.25, -1.25],
                    [0.75, -1.25, -1.25, -1.25],
                ]
            ),
            "custom_terrain": torch.ones(2, 16),
        },
        batch_size=2,
    )


def test_prior_with_peft_reference_is_frozen_complete_snapshot():
    prior = _prior_transformer()
    peft = DiscretePriorWithPEFT(
        prior,
        conditioning_dim=3,
        peft_type="lora",
        film_input_norm=True,
        condition_key="task_cond",
    )
    warmup = {
        "max_coords_obs": torch.ones(2, 2),
        "task_cond": torch.tensor([[0.5], [1.5]]),
    }

    peft.init_peft(warmup_obs=warmup)
    assert peft.film_input_norm._freeze_running is False

    peft.train(True)
    before_mean = peft.film_input_norm.running_obs_norm.mean.clone()
    peft.forward(
        {
            "max_coords_obs": torch.ones(2, 2),
            "task_cond": torch.tensor([[10.0], [12.0]]),
            "prior_tokens": torch.tensor([[0, 1], [2, 3]]),
        }
    )
    assert not torch.allclose(peft.film_input_norm.running_obs_norm.mean, before_mean)

    assert peft.capture_reference() is True
    assert peft.film_input_norm._freeze_running is True
    assert peft.capture_reference() is False
    assert peft.reference_ready is True
    assert peft.reference_prior is not peft.base_prior
    assert all(not p.requires_grad for p in peft.reference_prior.parameters())

    input_dict = {
        **warmup,
        "tokens": torch.tensor([[0, 1], [2, 3]]),
    }
    before = peft.forward_prior(input_dict)
    with torch.no_grad():
        for parameter in peft.base_prior._output_head.parameters():
            parameter.add_(10.0)
            break
    after = peft.forward_prior(input_dict)

    assert torch.allclose(after, before)


def test_prior_peft_actor_initializes_and_freezes_pretrained_components(monkeypatch):
    actor = _actor(monkeypatch, mimic_target_poses_dim=2)
    peft = _FakeDiscretePriorWithPEFT.instances[-1]

    assert actor.latent_decoder.training is False
    assert actor.target_latent_encoder.training is False
    assert actor.target_latent_encoder.encoder.training is False
    assert actor.latent_decoder.decoder.training is False
    assert actor.latent_decoder.quantizer.training is False
    assert peft.kwargs["prior"].training is False
    assert all(not p.requires_grad for p in actor.latent_decoder.decoder.parameters())
    assert all(
        not p.requires_grad for p in actor.target_latent_encoder.encoder.parameters()
    )
    assert all(not p.requires_grad for p in peft.kwargs["prior"].parameters())
    assert torch.equal(actor.latent_tokenization.basis, torch.tensor([1, 3]))
    assert actor.actor_peft_model.in_keys == [
        "max_coords_obs",
        *_DEFAULT_CONDITIONING_KEYS,
        "custom_terrain",
    ]
    assert actor.in_keys == [
        "max_coords_obs",
        *_DEFAULT_CONDITIONING_KEYS,
        "custom_terrain",
    ]
    assert actor.out_keys == ["action", "mean_action", "neglogp", "prior_tokens"]
    assert actor.kl_coeff == 0.25
    assert peft.kwargs["rank"] == 2
    assert peft.kwargs["condition_key"] == "task_cond"
    assert "conditioning_dim" not in peft.kwargs
    assert "extra_conditioning_dim" not in peft.kwargs
    assert isinstance(peft.kwargs["prior"], DiscreteAutoregressiveTransformer)

    warmup_obs = _td()
    actor.init_peft(warmup_obs=warmup_obs)
    assert peft.init_calls == 1
    assert torch.equal(
        peft.init_warmup_obs["max_coords_obs"],
        warmup_obs["max_coords_obs"],
    )
    assert "task_cond" in peft.init_warmup_obs
    actor.train(True)
    assert actor.target_latent_encoder.encoder.training is False


def test_prior_peft_actor_handles_missing_encoder_and_preserves_frozen_train_modes(
    monkeypatch,
):
    actor = _actor(monkeypatch, mimic_target_poses_dim=0)

    assert actor.target_latent_encoder is None
    actor.train(True)
    assert actor.latent_decoder.decoder.training is False
    assert actor.latent_decoder.quantizer.training is False
    assert actor.latent_decoder.training is False
    assert actor.prior_with_peft.training is True
    actor.train(False)
    assert actor.prior_with_peft.training is False


def test_prior_peft_actor_registers_frozen_modules_once_under_canonical_paths(
    monkeypatch,
):
    actor = _actor(monkeypatch)

    assert "decoder" not in actor._modules
    assert "quantizer" not in actor._modules
    assert "encoder" not in actor._modules
    assert "prior" not in actor._modules

    state_keys = set(actor.state_dict())
    forbidden_prefixes = ("decoder.", "quantizer.", "encoder.", "prior.")
    assert not any(key.startswith(forbidden_prefixes) for key in state_keys)
    assert any(key.startswith("latent_decoder.decoder.") for key in state_keys)
    assert any(key.startswith("latent_decoder.quantizer.") for key in state_keys)
    assert any(key.startswith("target_latent_encoder.encoder.") for key in state_keys)
    assert not any(key == "obs_normalizer" for key in vars(actor))


def test_prior_peft_actor_fsq_index_helpers_round_trip(monkeypatch):
    actor = _actor(monkeypatch)
    codes = torch.tensor([[0.1, 1.9, -1.2, 0.0]])
    fsq_indices = torch.tensor([[0, 1, 2, 0]])
    prior_tokens = torch.tensor([[3, 2]])

    assert torch.equal(actor.quantize(codes), codes.round())
    assert torch.equal(actor.fsq_codes_to_fsq_indices(codes), codes.long() + 1)
    assert torch.equal(actor.fsq_indices_to_codes(fsq_indices), fsq_indices.float() - 1)
    assert torch.equal(actor.fsq_indices_to_prior_tokens(fsq_indices), prior_tokens)
    assert torch.equal(actor.prior_tokens_to_fsq_indices(prior_tokens), fsq_indices)
    assert torch.equal(
        actor.one_hot_prior_tokens(prior_tokens),
        torch.nn.functional.one_hot(prior_tokens, 9).float(),
    )


def test_prior_peft_actor_token_perturbation_modes_cover_replace_neighbor_and_mixed(
    monkeypatch,
):
    actor = _actor(monkeypatch)
    tokens = torch.tensor([[0, 1], [2, 0]])

    assert (
        actor.perturb_tokens(tokens, rate=0.0, mode="replace").data_ptr()
        == tokens.data_ptr()
    )

    torch.manual_seed(0)
    replaced = actor.perturb_tokens(tokens, rate=1.0, mode="replace")
    assert replaced.shape == tokens.shape
    assert (replaced >= 0).all()
    assert (replaced < actor.prior_token_vocab_size).all()

    torch.manual_seed(1)
    neighbor = actor.perturb_tokens(tokens, rate=1.0, mode="neighbor")
    assert neighbor.shape == tokens.shape
    assert (neighbor >= 0).all()
    assert (neighbor < actor.prior_token_vocab_size).all()

    torch.manual_seed(2)
    mixed = actor.perturb_tokens(tokens, rate=1.0, mode="mixed")
    assert mixed.shape == tokens.shape


def test_prior_peft_actor_token_perturbation_returns_original_when_mask_empty(
    monkeypatch,
):
    actor = _actor(monkeypatch)
    tokens = torch.tensor([[0, 1], [2, 0]])

    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, **kwargs: torch.ones(*shape, device=kwargs.get("device")),
    )

    perturbed = actor.perturb_tokens(tokens, rate=0.5, mode="replace")

    assert perturbed.data_ptr() == tokens.data_ptr()


def test_prior_peft_actor_passes_raw_max_coords_obs_to_frozen_prior(monkeypatch):
    actor = _actor(monkeypatch)
    td = _td()
    raw_obs = td["max_coords_obs"].clone()

    prior_input = actor.build_prior_input(td.clone())
    assert torch.equal(prior_input["max_coords_obs"], raw_obs)

    actor.get_action_and_logp(td.clone())
    rollout_prior_dict = actor.prior_with_peft.generate_calls[-1][0]
    assert torch.equal(rollout_prior_dict["max_coords_obs"], raw_obs)


def test_prior_peft_actor_encode_decode_and_target_prediction(monkeypatch):
    actor = _actor(monkeypatch)
    td = _td()

    assert torch.equal(actor._encode(td), (td["mimic_target_poses"] + 0.25).round())
    assert torch.equal(
        actor._decode(td, torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        torch.tensor([[3.0], [7.0]]),
    )
    assert torch.equal(
        actor.predict_target_prior_tokens(td),
        actor.fsq_indices_to_prior_tokens(actor.fsq_codes_to_fsq_indices(actor._encode(td))),
    )


def test_prior_peft_actor_runs_configured_actor_peft_model(monkeypatch):
    actor = _actor(monkeypatch)
    td = _td()

    task_cond = actor.build_prior_input(td)["task_cond"]
    assert torch.equal(
        task_cond,
        torch.cat(
            [
                td["max_coords_obs"],
                td["task_obs"],
                td["masked_mimic_target_poses"],
                td["masked_mimic_target_masks"],
                td["masked_mimic_target_times"],
                td["historical_pose_obs"],
                td["custom_terrain"],
            ],
            dim=-1,
        ),
    )
    with pytest.raises(ValueError, match="actor PEFT model in_keys"):
        actor.build_prior_input(td.select("max_coords_obs"))


def test_prior_peft_actor_default_conditioning_uses_task_and_prior_context_inputs(
    monkeypatch,
):
    actor = _adaptive_actor(monkeypatch)
    td = _td()

    assert actor.frozen_prior_input_keys == ["max_coords_obs"]
    assert actor.actor_peft_model.in_keys == ["task_obs"]
    assert actor.actor_peft_model.out_keys == ["task_cond"]
    assert sum(p.numel() for p in actor.actor_peft_model.parameters()) == 0
    assert actor.in_keys == ["task_obs", "max_coords_obs"]
    actor.actor_peft_model.eval()
    prior_input = actor.build_prior_input(td)
    assert torch.allclose(prior_input["task_cond"], td["task_obs"])

    assert torch.allclose(prior_input["task_cond"], td["task_obs"])
    assert torch.equal(
        prior_input["max_coords_obs"],
        td["max_coords_obs"],
    )


def test_prior_peft_actor_uses_only_configured_actor_peft_model_inputs(monkeypatch):
    _FakeDiscretePriorWithPEFT.instances.clear()
    monkeypatch.setattr(actor_module, "DiscretePriorWithPEFT", _FakeDiscretePriorWithPEFT)
    actor_peft_model = _actor_peft_model_config(
        in_keys=["max_coords_obs", *_DEFAULT_CONDITIONING_KEYS],
    )
    actor = DiscretePriorPEFTActor(
        config=SimpleNamespace(
            peft=_peft_config(model=actor_peft_model),
            in_keys=["max_coords_obs", *_DEFAULT_CONDITIONING_KEYS],
            out_keys=["action", "mean_action", "neglogp", "prior_tokens"],
        ),
        pretrained_prior_model=_PriorModel(),
        mimic_target_poses_dim=2,
    )

    td = _td()

    prior_input = actor.build_prior_input(td)
    assert "extra_cond" not in prior_input
    assert torch.equal(
        prior_input["task_cond"],
        torch.cat([td[key] for key in actor_peft_model.in_keys], dim=-1),
    )


def test_prior_peft_actor_requires_actor_peft_model_task_cond(monkeypatch):
    _FakeDiscretePriorWithPEFT.instances.clear()
    monkeypatch.setattr(actor_module, "DiscretePriorWithPEFT", _FakeDiscretePriorWithPEFT)

    with pytest.raises(AssertionError, match="task_cond"):
        DiscretePriorPEFTActor(
            config=SimpleNamespace(
                peft=_peft_config(
                    model=_actor_peft_model_config(
                        in_keys=["max_coords_obs"],
                        out_keys=["max_coords_obs"],
                    ),
                ),
                in_keys=["max_coords_obs", *_DEFAULT_CONDITIONING_KEYS],
                out_keys=["action", "mean_action", "neglogp", "prior_tokens"],
            ),
            pretrained_prior_model=_PriorModel(),
            mimic_target_poses_dim=2,
        )


def test_prior_peft_actor_adds_frozen_prior_inputs_to_runtime_inputs(monkeypatch):
    _FakeDiscretePriorWithPEFT.instances.clear()
    monkeypatch.setattr(actor_module, "DiscretePriorWithPEFT", _FakeDiscretePriorWithPEFT)

    actor = DiscretePriorPEFTActor(
        config=SimpleNamespace(
            peft=_peft_config(model=_actor_peft_model_config(in_keys=["task_obs"])),
            in_keys=["task_obs"],
            out_keys=["action", "mean_action", "neglogp", "prior_tokens"],
        ),
        pretrained_prior_model=_PriorModel(),
        mimic_target_poses_dim=2,
    )

    assert actor.in_keys == ["task_obs", "max_coords_obs"]


def test_prior_peft_actor_teacher_forces_adapter_prior(monkeypatch):
    actor = _actor(monkeypatch)
    teacher_logits = actor(
        {
            "max_coords_obs": torch.ones(2, 1),
            "task_cond": torch.ones(2, 1) * 2,
        }
    )

    assert torch.equal(teacher_logits, torch.ones(2, 1) * 3)


def test_prior_peft_actor_get_action_and_logp_writes_rollout_keys(monkeypatch):
    actor = _actor(monkeypatch)
    td = _td()

    out = actor.get_action_and_logp(td)
    expected_logprob = torch.distributions.Categorical(
        logits=torch.tensor(
            [
                [[3.0, 0.0, -1.0], [0.0, 4.0, -2.0]],
                [[-1.0, 0.0, 5.0], [2.0, 0.0, -1.0]],
            ]
        )
    ).log_prob(torch.tensor([[0, 1], [2, 0]]))

    assert torch.equal(out["action"], torch.tensor([[-3.0], [-2.0]]))
    assert torch.equal(out["mean_action"], out["action"])
    assert torch.equal(out[LATENT_KEY], torch.tensor([[0, 1], [2, 0]]))
    assert torch.allclose(out[LATENT_LOGPROB_KEY], expected_logprob)
    assert torch.allclose(out["neglogp"], -expected_logprob)


def test_prior_peft_actor_extracts_and_loads_adapter_only_state(monkeypatch):
    actor = _actor(monkeypatch)

    adapter_state = actor.adapter_state_dict()

    assert adapter_state
    assert all(
        key.startswith("actor_peft_model.")
        or any(part in key for part in (".lora.", ".gamma.", ".beta.", ".m"))
        for key in adapter_state
    )
    assert not any("frozen" in key for key in adapter_state)
    assert not any("_anchor_transformer" in key for key in adapter_state)
    assert not any("reference_prior" in key for key in adapter_state)
    assert not any("reference_film_input_norm" in key for key in adapter_state)

    updated = {
        f"_actor.{key}": torch.full_like(value, 3.0)
        for key, value in adapter_state.items()
    }
    updated[
        "_actor.prior_with_peft.base_prior._transformer.layers.0.transformer_layer.weight"
    ] = torch.tensor([9.0])

    result = actor.load_adapter_state_dict(updated, strict=True)

    assert result == {"missing_keys": [], "unexpected_keys": []}
    for value in actor.adapter_state_dict().values():
        assert torch.equal(value, torch.full_like(value, 3.0))


def test_prior_peft_model_uses_rollout_forward_only():
    config = SimpleNamespace(
        actor=SimpleNamespace(
            _target_="protomotions.tests.test_prior_peft_model._DummyPriorActor"
        ),
        critic=None,
    )
    model = DiscretePriorPEFTModel(
        config=config,
        pretrained_prior_model=object(),
        mimic_target_poses_dim=4,
    )

    assert model._actor.mimic_target_poses_dim == 4
    assert model._critic is None
    model.reset_rollout_context(num_envs=2, device=torch.device("cpu"))

    td = TensorDict({"max_coords_obs": torch.ones(2, 2)}, batch_size=2)
    out_td = model(td.clone())
    assert torch.equal(out_td["action"], torch.ones(2, 1))
    assert torch.equal(out_td["mean_action"], torch.ones(2, 1))
    assert torch.equal(out_td["prior_tokens"], out_td[LATENT_KEY])
    assert torch.equal(out_td["neglogp"], torch.zeros(2))
    assert LATENT_KEY in out_td.keys()
    assert model.experience_buffer_keys() == [
        "action",
        "mean_action",
        "neglogp",
        "prior_tokens",
    ]

    with pytest.raises(TypeError, match="TensorDict rollout input"):
        model({"max_coords_obs": torch.ones(2, 2)})


def test_prior_peft_model_builds_optional_critic():
    config = SimpleNamespace(
        actor=SimpleNamespace(
            _target_="protomotions.tests.test_prior_peft_model._DummyPriorActor"
        ),
        critic=MLPWithConcatConfig(
            in_keys=["max_coords_obs"],
            out_keys=["value"],
            num_out=1,
            layers=[],
        ),
    )

    model = DiscretePriorPEFTModel(
        config=config,
        pretrained_prior_model=object(),
    )

    assert model._critic is not None
    assert model._critic.in_keys == ["max_coords_obs"]
    assert model._critic.out_keys == ["value"]

    td = TensorDict({"max_coords_obs": torch.ones(2, 2)}, batch_size=2)
    out_td = model(td)
    assert "value" in out_td.keys()
    assert out_td["value"].shape == (2, 1)
    assert model.experience_buffer_keys() == [
        "action",
        "mean_action",
        "neglogp",
        "prior_tokens",
        "value",
    ]


def test_prior_peft_amp_model_adds_discriminator_and_disc_critic_outputs():
    from protomotions.agents.amp.config import DiscriminatorConfig
    from protomotions.agents.common.config import ModuleContainerConfig, MLPLayerConfig
    from protomotions.agents.peft.prior_amp_model import DiscretePriorPEFTRLFTAMPModel

    config = SimpleNamespace(
        actor=SimpleNamespace(
            _target_="protomotions.tests.test_prior_peft_model._DummyPriorActor"
        ),
        critic=MLPWithConcatConfig(
            in_keys=["max_coords_obs"],
            out_keys=["value"],
            num_out=1,
            layers=[],
        ),
        in_keys=["max_coords_obs", "historical_max_coords_obs"],
        out_keys=[
            "action",
            "mean_action",
            "neglogp",
            "prior_tokens",
            "value",
            "disc_logits",
            "disc_value",
        ],
        discriminator=DiscriminatorConfig(
            in_keys=["historical_max_coords_obs"],
            out_keys=["disc_logits"],
            models=[
                MLPWithConcatConfig(
                    in_keys=["historical_max_coords_obs"],
                    out_keys=["disc_logits"],
                    num_out=1,
                    layers=[MLPLayerConfig(units=4, activation="relu")],
                )
            ],
        ),
        disc_critic=ModuleContainerConfig(
            in_keys=["max_coords_obs", "historical_max_coords_obs"],
            out_keys=["disc_value"],
            models=[
                MLPWithConcatConfig(
                    in_keys=["max_coords_obs", "historical_max_coords_obs"],
                    out_keys=["disc_value"],
                    num_out=1,
                    layers=[MLPLayerConfig(units=4, activation="relu")],
                )
            ],
        ),
    )
    model = DiscretePriorPEFTRLFTAMPModel(
        config=config,
        pretrained_prior_model=object(),
    )

    td = TensorDict(
        {
            "max_coords_obs": torch.ones(2, 2),
            "historical_max_coords_obs": torch.ones(2, 3),
        },
        batch_size=2,
    )

    out_td = model(td)

    assert out_td["action"].shape == (2, 1)
    assert out_td["value"].shape == (2, 1)
    assert out_td["disc_logits"].shape == (2, 1)
    assert out_td["disc_value"].shape == (2, 1)
    assert model.experience_buffer_keys() == [
        "action",
        "mean_action",
        "neglogp",
        "prior_tokens",
        "value",
        "disc_logits",
        "disc_value",
    ]

    model._disc_critic.out_keys = ["value"]
    with pytest.raises(ValueError, match="must not reuse"):
        model.experience_buffer_keys()
