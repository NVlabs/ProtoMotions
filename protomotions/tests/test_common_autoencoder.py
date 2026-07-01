# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for common autoencoder modules."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.autoencoder import AutoEncoder
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.agents.common.fsq import (
    FSQAutoEncoder,
    FiniteScalarQuantizer,
)
from protomotions.agents.common.common import MODULE_INTERNALS_KEY
from protomotions.agents.common.pretrained import load_pretrained_model_module
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGVAR_KEY,
    LATENT_MU_KEY,
    PRIVILEGED_LATENT_KEY,
    PRIVILEGED_LATENT_LOGVAR_KEY,
    PRIVILEGED_LATENT_MU_KEY,
    VAE_LATENT_KEY,
)
from protomotions.agents.common.autoencoder.config import AutoEncoderConfig
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    MLPLayerConfig,
    ModuleContainerConfig,
    PretrainedModelConfig,
)
from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig
from protomotions.agents.supervised.masked_mimic_config import MaskedMimicModelConfig
from protomotions.utils.hydra_replacement import get_class


def _mlp(in_keys, out_key, num_out):
    return MLPWithConcatConfig(
        in_keys=in_keys,
        out_keys=[out_key],
        normalize_obs=False,
        num_out=num_out,
        layers=[MLPLayerConfig(units=4, activation="relu")],
    )


def test_autoencoder_identity_bottleneck_and_empty_internal_logs():
    config = AutoEncoderConfig(
        encoder=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["encoded"],
            models=[_mlp(["obs"], "encoded", 2)],
        ),
        decoder=ModuleContainerConfig(
            in_keys=["obs", "latent"],
            out_keys=["decoded"],
            models=[_mlp(["obs", "latent"], "decoded", 1)],
        ),
    )
    model = AutoEncoder(config)
    td = TensorDict({"obs": torch.randn(2, 3)}, batch_size=2)
    latent = torch.ones(2, 2)

    out = model(td.clone(), log_internals=True)

    assert model.bottleneck(latent, td) is latent
    assert model.internal_logs(latent, td) == {}
    assert out["decoded"].shape == (2, 1)
    assert MODULE_INTERNALS_KEY not in out.keys()
    assert model.in_keys == ["obs"]
    assert model.out_keys == ["decoded"]


def test_fsq_autoencoder_quantizes_logs_and_exposes_quantizer_api():
    with pytest.raises(ValueError, match="odd number"):
        FiniteScalarQuantizer(num_fsq_levels=4, num_fsq_scalars=2)

    config = FSQAutoEncoderConfig(
        encoder=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["encoded"],
            models=[_mlp(["obs"], "encoded", 3)],
        ),
        decoder=ModuleContainerConfig(
            in_keys=["obs", "quantized_latent"],
            out_keys=["decoded"],
            models=[_mlp(["obs", "quantized_latent"], "decoded", 1)],
        ),
        latent_key="quantized_latent",
        num_fsq_levels=5,
        num_fsq_scalars=3,
    )
    model = FSQAutoEncoder(config)
    td = TensorDict({"obs": torch.randn(2, 3)}, batch_size=2)

    out = model(td, log_internals=True)
    codes = torch.tensor([[-2.0, 0.0, 2.0]])
    indices = model.codes_to_indices(codes)

    assert model.num_fsq_levels == 5
    assert model.num_fsq_scalars == 3
    assert model.L.shape == (3,)
    assert model.half_width.tolist() == [2.0, 2.0, 2.0]
    assert model.half_L.shape == (3,)
    assert torch.equal(model.indices_to_codes(indices), codes)
    assert torch.equal(model.round_ste(torch.tensor([1.2])), torch.tensor([1.0]))
    assert torch.all(model.bound(torch.full((1, 3), 100.0)) <= model.half_L)
    assert torch.all(out["quantized_latent"].frac() == 0)
    assert out["decoded"].shape == (2, 1)
    assert out[MODULE_INTERNALS_KEY]["perplexity"].shape == (2,)
    assert model.calculate_perplexity(out["quantized_latent"], skip=True).item() == 0.0


def test_fsq_autoencoder_rejects_encoder_output_dim_mismatch():
    config = FSQAutoEncoderConfig(
        encoder=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["encoded"],
            models=[_mlp(["obs"], "encoded", 2)],
        ),
        decoder=ModuleContainerConfig(
            in_keys=["obs", "quantized_latent"],
            out_keys=["decoded"],
            models=[_mlp(["obs", "quantized_latent"], "decoded", 1)],
        ),
        latent_key="quantized_latent",
        num_fsq_levels=5,
        num_fsq_scalars=3,
    )

    with pytest.raises(ValueError, match="encoder output dim .* num_fsq_scalars"):
        FSQAutoEncoder(config)


def test_fsq_autoencoder_config_targets_common_fsq_module():
    fsq_module = importlib.import_module(
        "protomotions.agents.common.fsq"
    )

    assert get_class(FSQAutoEncoderConfig()._target_) is fsq_module.FSQAutoEncoder


class _CheckpointActor(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder_class = get_class(config.decoder._target_)
        self.mu = nn.Module()
        self.mu.decoder = decoder_class(config=config.decoder)


class _CheckpointModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self._actor = _CheckpointActor(config.actor)

    def forward(self, tensordict: TensorDict, log_internals: bool = False):
        return self._actor.mu.decoder(tensordict)


def _write_checkpoint_model(tmp_path, decoder_config, source_td):
    source_config = SimpleNamespace(
        _target_="protomotions.tests.test_common_autoencoder._CheckpointModel",
        actor=SimpleNamespace(decoder=decoder_config),
    )
    source_model = _CheckpointModel(source_config)
    source_model._actor.mu.decoder(source_td)

    checkpoint_path = tmp_path / "model.ckpt"
    torch.save({"model": source_model.state_dict()}, checkpoint_path)
    torch.save(
        {"agent": SimpleNamespace(model=source_config)},
        tmp_path / "resolved_configs.pt",
    )
    return checkpoint_path, source_model


def test_masked_mimic_model_owns_resettable_noise_and_returns_it():
    latent_dim = 2
    config = SimpleNamespace(
        encoder=ModuleContainerConfig(
            in_keys=["priv_obs"],
            out_keys=["encoder_mu", "encoder_logvar"],
            models=[
                _mlp(["priv_obs"], "encoder_mu", latent_dim),
                _mlp(["priv_obs"], "encoder_logvar", latent_dim),
            ],
        ),
        prior=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["prior_mu", "prior_logvar"],
            models=[
                _mlp(["obs"], "prior_mu", latent_dim),
                _mlp(["obs"], "prior_logvar", latent_dim),
            ],
        ),
        trunk=ModuleContainerConfig(
            in_keys=["obs", "vae_latent"],
            out_keys=["decoded_action"],
            models=[_mlp(["obs", "vae_latent"], "decoded_action", 1)],
        ),
        vae=SimpleNamespace(vae_latent_dim=latent_dim, vae_noise_type="zeros"),
    )
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=3, device="cpu")

    obs = {
        "obs": torch.randn(3, 3),
        "priv_obs": torch.randn(3, 2),
    }
    td = TensorDict(obs, batch_size=3)
    out = model(td)

    assert out["action"].shape == (3, 1)
    assert out["privileged_action"].shape == (3, 1)
    assert out["vae_noise"].shape == (3, latent_dim)
    assert torch.all(out["vae_noise"] == 0)
    assert out[LATENT_KEY].shape == (3, latent_dim)
    assert out[PRIVILEGED_LATENT_KEY].shape == (3, latent_dim)
    assert torch.equal(out[LATENT_MU_KEY], out["prior_mu"])
    assert torch.equal(out[LATENT_LOGVAR_KEY], out["prior_logvar"])
    assert torch.equal(
        out[PRIVILEGED_LATENT_MU_KEY],
        out["prior_mu"] + out["encoder_mu"],
    )
    assert torch.equal(out[PRIVILEGED_LATENT_LOGVAR_KEY], out["encoder_logvar"])
    assert "vae_noise" in model.out_keys
    assert LATENT_KEY in model.out_keys
    assert PRIVILEGED_LATENT_KEY in model.out_keys


def test_masked_mimic_kl_uses_configured_module_out_keys():
    latent_dim = 2
    config = SimpleNamespace(
        encoder=ModuleContainerConfig(
            in_keys=["priv_obs"],
            out_keys=["posterior_delta_mu", "posterior_logvar"],
            models=[
                _mlp(["priv_obs"], "posterior_delta_mu", latent_dim),
                _mlp(["priv_obs"], "posterior_logvar", latent_dim),
            ],
        ),
        prior=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["context_mu", "context_logvar"],
            models=[
                _mlp(["obs"], "context_mu", latent_dim),
                _mlp(["obs"], "context_logvar", latent_dim),
            ],
        ),
        trunk=ModuleContainerConfig(
            in_keys=["obs", "vae_latent"],
            out_keys=["decoded_action"],
            models=[_mlp(["obs", "vae_latent"], "decoded_action", 1)],
        ),
        vae=SimpleNamespace(vae_latent_dim=latent_dim, vae_noise_type="zeros"),
    )
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=3, device="cpu")

    td = TensorDict(
        {
            "obs": torch.randn(3, 3),
            "priv_obs": torch.randn(3, 2),
        },
        batch_size=3,
    )
    out = model(td)
    expected_kl = 0.5 * (
        out["context_logvar"]
        - out["posterior_logvar"]
        + torch.exp(out["posterior_logvar"]) / torch.exp(out["context_logvar"])
        + out["posterior_delta_mu"] ** 2 / torch.exp(out["context_logvar"])
        - 1
    )

    assert torch.allclose(model.kl_loss(out), expected_kl)


def test_masked_mimic_forward_inference_uses_prior_only():
    latent_dim = 2
    config = SimpleNamespace(
        encoder=ModuleContainerConfig(
            in_keys=["priv_obs"],
            out_keys=["encoder_mu", "encoder_logvar"],
            models=[
                _mlp(["priv_obs"], "encoder_mu", latent_dim),
                _mlp(["priv_obs"], "encoder_logvar", latent_dim),
            ],
        ),
        prior=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["prior_mu", "prior_logvar"],
            models=[
                _mlp(["obs"], "prior_mu", latent_dim),
                _mlp(["obs"], "prior_logvar", latent_dim),
            ],
        ),
        trunk=ModuleContainerConfig(
            in_keys=["obs", "terrain", VAE_LATENT_KEY],
            out_keys=["decoded_action"],
            models=[_mlp(["obs", "terrain", VAE_LATENT_KEY], "decoded_action", 1)],
        ),
        vae=SimpleNamespace(vae_latent_dim=latent_dim, vae_noise_type="zeros"),
    )
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=2, device="cpu")

    out = model.forward_inference(
        TensorDict(
            {
                "obs": torch.randn(2, 3),
                "terrain": torch.randn(2, 1),
                "priv_obs": torch.randn(2, 2),
            },
            batch_size=2,
        )
    )

    assert out["action"].shape == (2, 1)
    assert "privileged_action" not in out.keys()
    assert torch.equal(out[LATENT_MU_KEY], out["prior_mu"])
    assert torch.equal(out[LATENT_LOGVAR_KEY], out["prior_logvar"])
    assert model.get_inference_in_keys() == ["obs", "terrain"]


def test_masked_mimic_reset_resamples_only_done_env_noise():
    latent_dim = 2
    config = SimpleNamespace(
        encoder=ModuleContainerConfig(
            in_keys=["priv_obs"],
            out_keys=["encoder_mu", "encoder_logvar"],
            models=[
                _mlp(["priv_obs"], "encoder_mu", latent_dim),
                _mlp(["priv_obs"], "encoder_logvar", latent_dim),
            ],
        ),
        prior=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["prior_mu", "prior_logvar"],
            models=[
                _mlp(["obs"], "prior_mu", latent_dim),
                _mlp(["obs"], "prior_logvar", latent_dim),
            ],
        ),
        trunk=ModuleContainerConfig(
            in_keys=["obs", "vae_latent"],
            out_keys=["decoded_action"],
            models=[_mlp(["obs", "vae_latent"], "decoded_action", 1)],
        ),
        vae=SimpleNamespace(vae_latent_dim=latent_dim, vae_noise_type="normal"),
    )
    model = MaskedMimicModel(config)

    torch.manual_seed(0)
    model.reset_rollout_context(num_envs=3, device="cpu")
    initial_noise = model.vae_noise.clone()

    torch.manual_seed(1)
    model.reset_rollout_context(env_ids=torch.tensor([1]))

    assert torch.equal(model.vae_noise[0], initial_noise[0])
    assert not torch.equal(model.vae_noise[1], initial_noise[1])
    assert torch.equal(model.vae_noise[2], initial_noise[2])


def test_masked_mimic_uses_recorded_noise_when_provided():
    latent_dim = 2
    config = SimpleNamespace(
        encoder=ModuleContainerConfig(
            in_keys=["priv_obs"],
            out_keys=["encoder_mu", "encoder_logvar"],
            models=[
                _mlp(["priv_obs"], "encoder_mu", latent_dim),
                _mlp(["priv_obs"], "encoder_logvar", latent_dim),
            ],
        ),
        prior=ModuleContainerConfig(
            in_keys=["obs"],
            out_keys=["prior_mu", "prior_logvar"],
            models=[
                _mlp(["obs"], "prior_mu", latent_dim),
                _mlp(["obs"], "prior_logvar", latent_dim),
            ],
        ),
        trunk=ModuleContainerConfig(
            in_keys=["obs", "vae_latent"],
            out_keys=["decoded_action"],
            models=[_mlp(["obs", "vae_latent"], "decoded_action", 1)],
        ),
        vae=SimpleNamespace(vae_latent_dim=latent_dim, vae_noise_type="zeros"),
    )
    model = MaskedMimicModel(config)
    model.reset_rollout_context(num_envs=3, device="cpu")

    provided_noise = torch.ones(3, latent_dim)
    td = TensorDict(
        {
            "obs": torch.randn(3, 3),
            "priv_obs": torch.randn(3, 2),
            "vae_noise": provided_noise,
        },
        batch_size=3,
    )
    out = model(td)

    assert torch.equal(out["vae_noise"], provided_noise)


def test_masked_mimic_model_config_targets_named_student():
    config = MaskedMimicModelConfig()
    assert get_class(config._target_) is MaskedMimicModel
    assert not issubclass(MaskedMimicModel, AutoEncoder)


def test_load_pretrained_model_module_resolves_nested_private_paths(tmp_path):
    decoder_config = ModuleContainerConfig(
        in_keys=["obs", "vae_latent"],
        out_keys=["decoded_action"],
        models=[_mlp(["obs", "vae_latent"], "decoded_action", 1)],
    )
    source_td = TensorDict(
        {
            "obs": torch.randn(3, 3),
            "vae_latent": torch.randn(3, 2),
        },
        batch_size=3,
    )
    checkpoint_path, source_model = _write_checkpoint_model(
        tmp_path,
        decoder_config,
        source_td,
    )

    decoder = load_pretrained_model_module(
        PretrainedModelConfig(
            checkpoint_path=str(checkpoint_path),
            module_path="actor.mu.decoder",
        ),
        device=torch.device("cpu"),
    )

    for key, value in source_model._actor.mu.decoder.state_dict().items():
        assert torch.equal(decoder.state_dict()[key], value)
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
