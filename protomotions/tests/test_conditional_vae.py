# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder normalization, spherical sampling and residual posterior contracts."""

from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.common.common import NormObsBase
from protomotions.agents.common.config import (
    MLPLayerConfig,
    MLPWithConcatConfig,
    ModuleContainerConfig,
)
from protomotions.agents.supervised.masked_mimic_config import (
    LatentGeometry,
    MaskedMimicModelConfig,
    MaskedMimicVAEConfig,
)
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel
from protomotions.utils.config_utils import apply_overrides
from protomotions.utils.experiment_configs import ExperimentConfigs


def test_latent_geometry_enum_override_and_saved_config():
    from protomotions.utils.config_schema import from_tree, to_tree

    vae = MaskedMimicVAEConfig()
    assert vae.latent_geometry is LatentGeometry.GAUSSIAN
    agent = SimpleNamespace(model=SimpleNamespace(vae=vae))
    apply_overrides(
        ExperimentConfigs(None, None, None, None, None, None, agent),
        {"agent.model.vae.latent_geometry": "sphere"},
    )
    assert vae.latent_geometry is LatentGeometry.SPHERE
    vae.__post_init__()

    # The enum survives the JSON saved form: it is tagged by member name and
    # rebuilt through the real constructor (B4), not pickled.
    tree = to_tree(vae, where="agent.model.vae")
    restored = from_tree(tree, type(vae), where="agent.model.vae")
    assert restored.latent_geometry is LatentGeometry.SPHERE


def _mlp(inputs, output, normalize=False):
    return MLPWithConcatConfig(
        in_keys=inputs,
        out_keys=[output],
        num_out=3,
        layers=[MLPLayerConfig(units=8, activation="silu")],
        normalize_obs=normalize,
    )


def _model(geometry=LatentGeometry.SPHERE, normalize=False):
    prior = ModuleContainerConfig(
        in_keys=["goals"],
        out_keys=["prior_mu", "prior_logvar"],
        models=[
            _mlp(["goals"], "prior_mu"),
            _mlp(["goals"], "prior_logvar"),
        ],
    )
    encoder = ModuleContainerConfig(
        in_keys=["expert_obs"],
        out_keys=["residual_mu", "posterior_logvar"],
        models=[
            _mlp(["expert_obs"], "residual_mu"),
            _mlp(["expert_obs"], "posterior_logvar"),
        ],
    )
    trunk = ModuleContainerConfig(
        in_keys=["self_obs", "vae_latent"],
        out_keys=["decoded_action"],
        models=[
            _mlp(
                ["self_obs", "vae_latent"],
                "decoded_action",
                normalize,
            ),
        ],
    )
    return MaskedMimicModel(
        MaskedMimicModelConfig(
            prior=prior,
            encoder=encoder,
            trunk=trunk,
            vae=MaskedMimicVAEConfig(vae_latent_dim=3, latent_geometry=geometry),
        )
    )


def _batch():
    return TensorDict(
        {
            "self_obs": torch.randn(5, 4),
            "goals": torch.randn(5, 6),
            "expert_obs": torch.randn(5, 7),
            "vae_noise": torch.randn(5, 3),
        },
        batch_size=5,
    )


def test_inference_needs_only_prior_and_decoder_observations():
    model = _model().eval()
    assert model.in_keys == ["goals", "expert_obs", "self_obs"]
    assert model.deployable_in_keys == ["goals", "self_obs"]
    batch = _batch()
    model(batch.clone())  # Materialize both paths before exporting/checkpointing.
    with patch.object(
        model._encoder, "forward", side_effect=AssertionError("teacher used")
    ):
        first = model.forward_deployable(batch.exclude("expert_obs"))
        second = model.forward_deployable(
            batch.clone().set("expert_obs", torch.full((5, 7), float("nan")))
        )
    torch.testing.assert_close(first["action"], second["action"])
    changed = batch.clone().set("goals", batch["goals"] + 5)
    assert not torch.allclose(
        first["action"], model.forward_deployable(changed)["action"]
    )


@pytest.mark.parametrize("geometry", list(LatentGeometry))
def test_full_and_inference_forward_share_prior_outputs_and_gradients(geometry):
    model = _model(geometry, normalize=True).eval()
    batch = _batch()
    full = model(batch.clone())
    full["action"].square().sum().backward()
    expected_grads = {
        name: None if param.grad is None else param.grad.clone()
        for name, param in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)

    inference = model.forward_deployable(batch.exclude("expert_obs").clone())
    for key in ("action", "latent", "latent_mu", "latent_logvar", "vae_noise"):
        torch.testing.assert_close(full[key], inference[key], rtol=0, atol=0)
    inference["action"].square().sum().backward()
    for name, param in model.named_parameters():
        if expected_grads[name] is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, expected_grads[name], rtol=0, atol=0)


@pytest.mark.parametrize("geometry", list(LatentGeometry))
def test_geometry_and_gradients(geometry):
    model = _model(geometry)
    out = model(_batch())
    if geometry == LatentGeometry.SPHERE:
        for key in ("latent", "privileged_latent"):
            torch.testing.assert_close(out[key].norm(dim=-1), torch.ones(5))
    loss, _ = model.compute_model_loss(out, 0, out["action"].sum() * 0)
    (loss + out["privileged_action"].square().mean()).backward()
    for module in (model._prior, model._encoder, model._trunk):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_sphere_zero_mean_has_defined_direction_and_gaussian_is_unchanged():
    model = _model()
    zero = torch.zeros(5, 3)
    result = model._sample_latent(zero, zero, zero)
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]).expand(5, -1))
    model.config.vae.latent_geometry = LatentGeometry.GAUSSIAN
    mu, logvar, noise = torch.randn(3, 5, 3)
    torch.testing.assert_close(
        model._sample_latent(mu, logvar, noise), mu + (logvar / 2).exp() * noise
    )


def test_sphere_zero_noise_inference_is_unit_norm_and_deterministic():
    model = _model()
    model.config.vae.vae_noise_type = "zeros"
    model.reset_rollout_context(num_envs=5, device="cpu")
    model.eval()
    batch = _batch().exclude("vae_noise", "expert_obs")
    first = model.forward_deployable(batch.clone())
    second = model.forward_deployable(batch.clone())
    torch.testing.assert_close(first["latent"].norm(dim=-1), torch.ones(5))
    torch.testing.assert_close(first["action"], second["action"])


def test_residual_kl_matches_gaussian_distribution_kl():
    model = _model()
    out = model(_batch())
    prior = torch.distributions.Normal(out["prior_mu"], (out["prior_logvar"] / 2).exp())
    posterior = torch.distributions.Normal(
        out["prior_mu"] + out["residual_mu"], (out["posterior_logvar"] / 2).exp()
    )
    torch.testing.assert_close(
        model.kl_loss(out), torch.distributions.kl_divergence(posterior, prior)
    )


def test_unit_norm_penalty_without_kl_does_not_require_posterior():
    model = _model()
    model.config.vae.kld_schedule = None
    model.config.vae.prior_mean_unit_norm_coeff = 0.25
    from protomotions.agents.common.latent import LATENT_MU_KEY

    means = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
    loss, _ = model.compute_model_loss(
        TensorDict({LATENT_MU_KEY: means}, batch_size=1), 0, means.sum() * 0
    )
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    assert torch.isfinite(means.grad).all()


def test_decoder_records_normalizer_once_and_restores_flags_on_error():
    model = _model(normalize=True)
    normalizer = next(m for m in model._trunk.modules() if isinstance(m, NormObsBase))
    norm = normalizer.running_obs_norm
    with patch.object(norm, "record_moments", wraps=norm.record_moments) as record:
        model(_batch())
        assert record.call_count == 1
    assert not normalizer._freeze_running
    with pytest.raises(RuntimeError, match="decode error"):
        with model._freeze_trunk_normalizer_updates():
            assert normalizer._freeze_running
            raise RuntimeError("decode error")
    assert not normalizer._freeze_running
    normalizer._freeze_running = True
    with model._freeze_trunk_normalizer_updates():
        pass
    assert normalizer._freeze_running


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_geometry": "invalid"},
        {"latent_geometry": "sphere"},
        {"latent_geometry": "gaussian"},
        {"vae_latent_dim": 0},
        {"prior_mean_unit_norm_coeff": -1},
        {"prior_mean_unit_norm_coeff": float("nan")},
        {"prior_mean_unit_norm_coeff": 1},
    ],
)
def test_invalid_vae_config(kwargs):
    with pytest.raises(ValueError):
        MaskedMimicVAEConfig(**kwargs)
