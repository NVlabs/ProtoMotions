# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for standard config construction and inference import helpers."""

import sys
import types
from types import SimpleNamespace

import pytest

from protomotions.utils.config_builder import build_standard_configs
from protomotions.utils.inference_utils import apply_all_inference_overrides
from protomotions.utils.simulator_imports import import_simulator_before_torch


def test_build_standard_configs_calls_factories_in_order_and_includes_agent(monkeypatch):
    import protomotions.robot_configs.factory as robot_factory
    import protomotions.simulator.factory as simulator_factory

    calls = []
    args = SimpleNamespace(
        robot_name="testbot",
        simulator="newton",
        headless=True,
        num_envs=8,
        experiment_name="unit",
    )

    def robot_config(robot_name):
        calls.append(("robot", robot_name))
        return SimpleNamespace(name=robot_name, configured=False)

    def simulator_config(simulator, robot_cfg, headless, num_envs, experiment_name):
        calls.append(("simulator", simulator, robot_cfg.name, headless, num_envs, experiment_name))
        return SimpleNamespace(name=simulator, configured=False)

    def configure(robot_cfg, simulator_cfg, received_args):
        calls.append(("configure", robot_cfg.name, simulator_cfg.name, received_args))
        robot_cfg.configured = True
        simulator_cfg.configured = True

    monkeypatch.setattr(robot_factory, "robot_config", robot_config)
    monkeypatch.setattr(simulator_factory, "simulator_config", simulator_config)

    configs = build_standard_configs(
        args,
        terrain_config_fn=lambda received_args: calls.append(("terrain", received_args)) or "terrain",
        scene_lib_config_fn=lambda received_args: calls.append(("scene", received_args)) or "scene",
        motion_lib_config_fn=lambda received_args: calls.append(("motion", received_args)) or "motion",
        env_config_fn=lambda robot_cfg, received_args: calls.append(("env", robot_cfg.name, received_args)) or "env",
        configure_robot_and_simulator_fn=configure,
        agent_config_fn=lambda robot_cfg, env_cfg, received_args: calls.append(
            ("agent", robot_cfg.name, env_cfg, received_args)
        )
        or "agent",
    )

    assert configs == {
        "robot": configs["robot"],
        "simulator": configs["simulator"],
        "terrain": "terrain",
        "scene_lib": "scene",
        "motion_lib": "motion",
        "env": "env",
        "agent": "agent",
    }
    assert configs["robot"].configured is True
    assert configs["simulator"].configured is True
    assert calls == [
        ("robot", "testbot"),
        ("simulator", "newton", "testbot", True, 8, "unit"),
        ("configure", "testbot", "newton", args),
        ("terrain", args),
        ("scene", args),
        ("motion", args),
        ("env", "testbot", args),
        ("agent", "testbot", "env", args),
    ]


def test_build_standard_configs_allows_omitted_optional_hooks(monkeypatch):
    import protomotions.robot_configs.factory as robot_factory
    import protomotions.simulator.factory as simulator_factory

    args = SimpleNamespace(
        robot_name="testbot",
        simulator="newton",
        headless=False,
        num_envs=2,
        experiment_name="unit",
    )
    robot_cfg = SimpleNamespace(name="robot")
    simulator_cfg = SimpleNamespace(name="sim")
    monkeypatch.setattr(robot_factory, "robot_config", lambda robot_name: robot_cfg)
    monkeypatch.setattr(
        simulator_factory,
        "simulator_config",
        lambda simulator, robot_config, headless, num_envs, experiment_name: simulator_cfg,
    )

    configs = build_standard_configs(
        args,
        terrain_config_fn=lambda received_args: None,
        scene_lib_config_fn=lambda received_args: "scene",
        motion_lib_config_fn=lambda received_args: "motion",
        env_config_fn=lambda robot_config, received_args: "env",
    )

    assert configs["robot"] is robot_cfg
    assert configs["simulator"] is simulator_cfg
    assert configs["terrain"] is None
    assert configs["agent"] is None


def test_apply_all_inference_overrides_delegates_to_experiment_hook(caplog):
    configs = [SimpleNamespace(name=str(index)) for index in range(7)]
    args = SimpleNamespace(mode="eval")
    received = []

    def apply_inference_overrides(*values):
        received.append(values)
        values[0].name = "updated"

    experiment_module = SimpleNamespace(
        apply_inference_overrides=apply_inference_overrides
    )

    apply_all_inference_overrides(*configs, experiment_module=experiment_module, args=args)

    assert received == [tuple(configs + [args])]
    assert configs[0].name == "updated"


def test_apply_all_inference_overrides_ignores_missing_inputs_and_logs_failures(caplog):
    configs = [SimpleNamespace() for _ in range(7)]
    called = False

    def apply_inference_overrides(*values):
        nonlocal called
        called = True

    apply_all_inference_overrides(
        *configs,
        experiment_module=SimpleNamespace(apply_inference_overrides=apply_inference_overrides),
        args=None,
    )
    assert called is False

    def failing_override(*values):
        raise RuntimeError("bad override")

    apply_all_inference_overrides(
        *configs,
        experiment_module=SimpleNamespace(apply_inference_overrides=failing_override),
        args=SimpleNamespace(),
    )

    assert "Failed to apply experiment inference overrides: bad override" in caplog.text


def test_import_simulator_before_torch_handles_supported_names(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacgym", types.ModuleType("isaacgym"))

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_app = types.ModuleType("isaaclab.app")

    class _AppLauncher:
        pass

    isaaclab_app.AppLauncher = _AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)

    assert import_simulator_before_torch("isaacgym") is None
    assert import_simulator_before_torch("isaaclab") is _AppLauncher
    assert import_simulator_before_torch("newton") is None
    assert import_simulator_before_torch(None) is None


def test_import_simulator_before_torch_surfaces_missing_required_packages(monkeypatch):
    monkeypatch.delitem(sys.modules, "isaacgym", raising=False)

    with pytest.raises(ModuleNotFoundError):
        import_simulator_before_torch("isaacgym")
