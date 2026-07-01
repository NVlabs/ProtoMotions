# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FSQ tracker metric tensors."""

from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict

from protomotions.agents.common.common import MODULE_INTERNALS_KEY
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    MLPLayerConfig,
)
from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig
from protomotions.agents.common.fsq import (
    FSQAutoEncoder,
    FiniteScalarQuantizer,
)
from protomotions.agents.optimizer.muon import MuonWithAuxAdam


def test_finite_scalar_quantizer_rejects_even_levels():
    with pytest.raises(ValueError, match="odd number"):
        FiniteScalarQuantizer(num_fsq_levels=4, num_fsq_scalars=2)


def test_finite_scalar_quantizer_indices_and_codes_round_trip():
    quantizer = FiniteScalarQuantizer(num_fsq_levels=5, num_fsq_scalars=3)
    indices = torch.tensor([[0, 2, 4], [4, 3, 1]])

    codes = quantizer.indices_to_codes(indices)
    round_trip = quantizer.codes_to_indices(codes)

    assert torch.equal(codes, torch.tensor([[-2.0, 0.0, 2.0], [2.0, 1.0, -1.0]]))
    assert torch.equal(round_trip, indices)


def test_finite_scalar_quantizer_perplexity_tracks_token_usage():
    quantizer = FiniteScalarQuantizer(num_fsq_levels=3, num_fsq_scalars=2)
    codes = quantizer.indices_to_codes(
        torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 1],
                [0, 2],
            ]
        )
    )

    assert torch.equal(quantizer.calculate_perplexity(codes, skip=True), torch.tensor(0.0))
    assert torch.allclose(quantizer.calculate_perplexity(codes), torch.tensor(1.5))


def test_perplexity_metric_can_be_accumulated_in_place():
    """Perplexity should not be an expanded zero-stride view."""
    module = FSQAutoEncoder(
        FSQAutoEncoderConfig(
            num_fsq_levels=5,
            num_fsq_scalars=2,
            encoder_out_keys=["latent"],
            decoder_out_keys=["mu"],
            encoder=MLPWithConcatConfig(
                in_keys=["max_coords_obs", "mimic_target_poses"],
                out_keys=["latent"],
                normalize_obs=False,
                num_out=2,
                layers=[
                    MLPLayerConfig(units=4, activation="relu"),
                    MLPLayerConfig(units=4, activation="relu"),
                ],
            ),
            decoder=MLPWithConcatConfig(
                in_keys=["max_coords_obs", "latent"],
                out_keys=["mu"],
                normalize_obs=False,
                num_out=2,
                layers=[
                    MLPLayerConfig(units=4, activation="relu"),
                    MLPLayerConfig(units=4, activation="relu"),
                ],
            ),
        )
    )
    td = TensorDict(
        {
            "max_coords_obs": torch.randn(4, 3),
            "mimic_target_poses": torch.randn(4, 2),
        },
        batch_size=4,
    )

    out = module(td, log_internals=True)
    accumulator = out[MODULE_INTERNALS_KEY]["perplexity"].detach()

    accumulator += out[MODULE_INTERNALS_KEY]["perplexity"].detach()
    assert accumulator.shape == (4,)


def test_muon_optimizer_hyperparameters_are_config_driven():
    module = FSQAutoEncoder(
        FSQAutoEncoderConfig(
            num_fsq_levels=5,
            num_fsq_scalars=2,
            encoder_out_keys=["latent"],
            decoder_out_keys=["mu"],
            encoder=MLPWithConcatConfig(
                in_keys=["max_coords_obs", "mimic_target_poses"],
                out_keys=["latent"],
                normalize_obs=False,
                num_out=2,
                layers=[
                    MLPLayerConfig(units=4, activation="relu"),
                    MLPLayerConfig(units=4, activation="relu"),
                ],
            ),
            decoder=MLPWithConcatConfig(
                in_keys=["max_coords_obs", "latent"],
                out_keys=["mu"],
                normalize_obs=False,
                num_out=2,
                layers=[
                    MLPLayerConfig(units=4, activation="relu"),
                    MLPLayerConfig(units=4, activation="relu"),
                ],
            ),
        )
    )
    td = TensorDict(
        {
            "max_coords_obs": torch.randn(4, 3),
            "mimic_target_poses": torch.randn(4, 2),
        },
        batch_size=4,
    )
    module(td)

    optimizer = MuonWithAuxAdam(
        params=module,
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


def test_fsq_tracker_experiment_encoder_uses_targets_and_decoder_context():
    from examples.experiments.mimic.fsq import agent_config

    robot_config = SimpleNamespace(
        number_of_actions=3,
    )
    args = SimpleNamespace(batch_size=1, training_max_steps=1)

    config = agent_config(robot_config, None, args)
    fsq = config.model.actor.mu_model

    assert fsq.encoder.in_keys == ["mimic_target_poses"]
    assert fsq.decoder.in_keys == ["max_coords_obs", "latent"]
    assert fsq.decoder.normalize_obs is True
