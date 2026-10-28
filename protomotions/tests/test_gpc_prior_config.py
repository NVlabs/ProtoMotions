# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPC prior experiment configuration."""

from argparse import Namespace
from types import SimpleNamespace

from examples.experiments.gpc import prior


def test_gpc_prior_loads_tracker_future_steps(monkeypatch, tmp_path):
    # The saved bundle rebuilds through the real constructors; these SimpleNamespace
    # stand-ins can't round-trip, so patch the bundle loader to return them.
    import protomotions.utils.experiment_configs as experiment_configs
    from protomotions.utils.experiment_configs import ExperimentConfigs

    checkpoint_path = tmp_path / "score_based.ckpt"
    checkpoint_path.write_bytes(b"")

    bundle = ExperimentConfigs(
        robot=None,
        simulator=None,
        terrain=None,
        scene_lib=None,
        motion_lib=None,
        env=SimpleNamespace(
            control_components={"mimic": SimpleNamespace(future_steps=[2, 4, 9])},
        ),
        agent=None,
    )
    monkeypatch.setattr(
        experiment_configs,
        "load_experiment_configs_from_checkpoint",
        lambda ckpt: bundle,
    )

    future_steps = prior._tracker_future_steps(str(checkpoint_path))

    # The saved list comes back as the normalised step tuple.
    assert future_steps == (2, 4, 9)


def test_gpc_prior_context_encoder_owns_normalization(monkeypatch):
    # agent_config embeds the tracker's mu_model at build; fake the cached read.
    monkeypatch.setattr(
        prior,
        "_tracker",
        lambda ckpt: SimpleNamespace(
            agent=SimpleNamespace(
                model=SimpleNamespace(actor=SimpleNamespace(mu_model="fsq"))
            )
        ),
    )
    config = prior.agent_config(
        robot_config=None,
        env_config=None,
        args=Namespace(
            tracker_checkpoint="tracker.ckpt",
            batch_size=1,
            training_max_steps=1,
            no_ema=True,
        ),
    )

    context_encoder = config.model.prior.context_encoder.models[0]

    assert "context_obs_key" not in config.model.__dataclass_fields__
    assert not hasattr(config.model, "context_normalizer_source")
    assert config.model.prior.in_keys == [
        "max_coords_obs",
        "prior_tokens",
    ]
    assert context_encoder.in_keys == ["max_coords_obs"]
    assert context_encoder.out_keys == ["context_embedding"]
    assert config.model.prior.context_embedding_key == "context_embedding"
    assert context_encoder.normalize_obs
    assert config.loss.label_smoothing == 0.01
