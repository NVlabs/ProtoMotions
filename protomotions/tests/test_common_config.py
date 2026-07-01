# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared common config branch boundaries."""

from protomotions.agents.common import config as common_config


def test_common_config_exposes_fsq_configs_but_not_later_configs():
    assert hasattr(common_config, "PretrainedModelConfig")
    assert not hasattr(common_config, "AutoEncoderConfig")
    assert not hasattr(common_config, "FSQAutoEncoderConfig")
    assert hasattr(common_config, "DiscreteAutoregressiveTransformerConfig")
    assert not hasattr(common_config, "VoxelEncoderConfig")
    assert not hasattr(common_config, "VoxelEncoder2DConfig")
