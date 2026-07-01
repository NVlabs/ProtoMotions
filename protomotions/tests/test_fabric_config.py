# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.utils.fabric_config import FabricConfig


class NoDeepcopy:
    def __deepcopy__(self, memo):
        raise AssertionError("Fabric kwargs must not deepcopy live objects")


def test_fabric_config_kwargs_are_shallow():
    logger = NoDeepcopy()
    callback = NoDeepcopy()

    config = FabricConfig(
        strategy=None,
        loggers=[logger],
        callbacks=[callback],
    )

    kwargs = config.as_kwargs()

    assert kwargs["loggers"][0] is logger
    assert kwargs["callbacks"][0] is callback


def test_fabric_config_instantiates_mapping_entries():
    config = FabricConfig(
        strategy={"_target_": "types.SimpleNamespace", "name": "strategy"},
        loggers=[{"_target_": "types.SimpleNamespace", "name": "logger"}],
        callbacks=[{"_target_": "types.SimpleNamespace", "name": "callback"}],
    )

    assert config.strategy.name == "strategy"
    assert config.loggers[0].name == "logger"
    assert config.callbacks[0].name == "callback"


def test_fabric_config_loggable_dict_summarizes_live_objects():
    logger = NoDeepcopy()

    config = FabricConfig(
        strategy=None,
        loggers=[logger],
        callbacks=[],
    )

    summary = config.as_loggable_dict()

    assert summary["loggers"] == ["NoDeepcopy"]
