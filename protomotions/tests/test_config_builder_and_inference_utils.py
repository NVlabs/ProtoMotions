# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for standard config construction and inference import helpers."""

import sys
import types
from types import SimpleNamespace

import pytest

from protomotions.utils.config_builder import build_configs
from protomotions.utils.experiment_configs import (
    ExperimentConfigs,
    load_experiment_configs,
    load_experiment_configs_from_checkpoint,
    save_experiment_configs,
)
from protomotions.utils.simulator_imports import (
    _validate_isaaclab_version,
    import_simulator_before_torch,
)


def _build_env(monkeypatch, *, robot=None, simulator=None):
    """Install fake robot/simulator factories and a no-op binding check.

    build_configs runs validate_anchor_body(), motion_lib.validate() and
    run_config_binding_check after building; the fakes keep those harmless so a
    unit test can focus on the build sequence.
    """
    import protomotions.robot_configs.factory as robot_factory
    import protomotions.simulator.factory as simulator_factory
    import protomotions.envs.context_check as context_check

    robot = robot or SimpleNamespace(
        name="testbot", configured=False, validate_anchor_body=lambda: None
    )
    simulator = simulator or SimpleNamespace(name="newton", configured=False)
    monkeypatch.setattr(robot_factory, "robot_config", lambda robot_name: robot)
    monkeypatch.setattr(
        simulator_factory,
        "simulator_config",
        lambda simulator_name, robot_cfg, headless, num_envs, experiment_name: simulator,
    )
    monkeypatch.setattr(
        context_check, "run_config_binding_check", lambda *a, **k: None
    )
    return robot, simulator


def _args():
    return SimpleNamespace(
        robot_name="testbot",
        simulator="newton",
        headless=True,
        num_envs=8,
        experiment_name="unit",
    )


def test_build_configs_runs_the_build_sequence_in_order(monkeypatch):
    """build_configs calls robot, simulator, configure, the four component hooks
    and agent in order and returns a typed ExperimentConfigs read by attribute."""
    import protomotions.robot_configs.factory as robot_factory
    import protomotions.simulator.factory as simulator_factory

    calls = []
    robot = SimpleNamespace(
        name="testbot", configured=False, validate_anchor_body=lambda: None
    )
    simulator = SimpleNamespace(name="newton", configured=False)
    motion = SimpleNamespace(name="motion", validate=lambda: None)
    _build_env(monkeypatch, robot=robot, simulator=simulator)
    monkeypatch.setattr(
        robot_factory,
        "robot_config",
        lambda robot_name: calls.append(("robot", robot_name)) or robot,
    )
    monkeypatch.setattr(
        simulator_factory,
        "simulator_config",
        lambda simulator_name, robot_cfg, headless, num_envs, experiment_name: calls.append(
            ("simulator", simulator_name, robot_cfg.name, headless, num_envs, experiment_name)
        )
        or simulator,
    )

    def configure(robot_cfg, simulator_cfg, received_args):
        calls.append(("configure", robot_cfg.name, simulator_cfg.name))
        robot_cfg.configured = True
        simulator_cfg.configured = True

    experiment = SimpleNamespace(
        __file__="<fake>",
        configure_robot_and_simulator=configure,
        terrain_config=lambda a: calls.append(("terrain",)) or "terrain",
        scene_lib_config=lambda a: calls.append(("scene",)) or "scene",
        motion_lib_config=lambda a: calls.append(("motion",)) or motion,
        env_config=lambda robot_cfg, a: calls.append(("env", robot_cfg.name)) or "env",
        agent_config=lambda robot_cfg, env_cfg, a: calls.append(
            ("agent", robot_cfg.name, env_cfg)
        )
        or SimpleNamespace(name="agent"),
    )

    configs = build_configs(experiment, _args())

    assert isinstance(configs, ExperimentConfigs)
    assert configs.robot is robot and configs.simulator is simulator
    assert configs.terrain == "terrain" and configs.scene_lib == "scene"
    assert configs.motion_lib is motion and configs.env == "env"
    assert configs.agent.name == "agent"
    assert configs.robot.configured and configs.simulator.configured
    assert calls == [
        ("robot", "testbot"),
        ("simulator", "newton", "testbot", True, 8, "unit"),
        ("configure", "testbot", "newton"),
        ("terrain",),
        ("scene",),
        ("motion",),
        ("env", "testbot"),
        ("agent", "testbot", "env"),
    ]


def test_build_configs_runs_the_after_robot_hook_between_configure_and_terrain(
    monkeypatch,
):
    """after_robot_hook runs after the experiment's configure and before terrain."""
    _build_env(monkeypatch)
    order = []
    experiment = SimpleNamespace(
        __file__="<fake>",
        configure_robot_and_simulator=lambda r, s, a: order.append("configure"),
        terrain_config=lambda a: order.append("terrain") or "terrain",
        scene_lib_config=lambda a: "scene",
        motion_lib_config=lambda a: SimpleNamespace(validate=lambda: None),
        env_config=lambda r, a: "env",
    )
    build_configs(
        experiment,
        _args(),
        after_robot_hook=lambda r, s, a: order.append("after_robot_hook"),
    )
    assert order == ["configure", "after_robot_hook", "terrain"]


def test_build_configs_omits_agent_and_configure_when_absent(monkeypatch):
    """A recipe without configure_robot_and_simulator or agent_config builds with
    agent None."""
    _build_env(monkeypatch)
    experiment = SimpleNamespace(
        __file__="<fake>",
        terrain_config=lambda a: "terrain",
        scene_lib_config=lambda a: "scene",
        motion_lib_config=lambda a: SimpleNamespace(validate=lambda: None),
        env_config=lambda robot_cfg, a: "env",
    )
    configs = build_configs(experiment, _args())
    assert configs.agent is None
    assert configs.terrain == "terrain"


def test_build_configs_rejects_none_terrain(monkeypatch):
    """terrain_config returning None fails loudly (use TerrainConfig() for flat)."""
    _build_env(monkeypatch)
    experiment = SimpleNamespace(
        __file__="recipe.py",
        terrain_config=lambda a: None,
        scene_lib_config=lambda a: "scene",
        motion_lib_config=lambda a: SimpleNamespace(validate=lambda: None),
        env_config=lambda robot_cfg, a: "env",
    )
    with pytest.raises(TypeError, match=r"terrain_config\(\) in recipe.py returned None"):
        build_configs(experiment, _args())


def test_build_configs_names_a_missing_required_hook(monkeypatch):
    """A recipe missing a required build function is named at build time."""
    _build_env(monkeypatch)
    experiment = SimpleNamespace(
        __file__="recipe.py",
        terrain_config=lambda a: "terrain",
        scene_lib_config=lambda a: "scene",
        # motion_lib_config is missing
        env_config=lambda robot_cfg, a: "env",
    )
    with pytest.raises(AttributeError, match=r"recipe.py must define motion_lib_config"):
        build_configs(experiment, _args())


def test_build_configs_applies_the_iteration_limit_when_given(monkeypatch):
    """--training-max-iterations lands on the agent config."""
    _build_env(monkeypatch)
    agent = SimpleNamespace(name="agent", training_max_iterations=None)
    experiment = SimpleNamespace(
        __file__="<fake>",
        terrain_config=lambda a: "terrain",
        scene_lib_config=lambda a: "scene",
        motion_lib_config=lambda a: SimpleNamespace(validate=lambda: None),
        env_config=lambda robot_cfg, a: "env",
        agent_config=lambda robot_cfg, env_cfg, a: agent,
    )
    args = _args()
    args.training_max_iterations = 42
    build_configs(experiment, args)
    assert agent.training_max_iterations == 42


def _tiny_bundle(**overrides):
    """A bundle whose roots are real round-trippable configs.

    Each root is a ``MotionLibConfig`` so ``load_experiment_configs`` rebuilds it
    through the real constructor; the field types are not what is under test here,
    the save/load machinery is.
    """
    from protomotions.components.motion_lib import MotionLibConfig

    fields = {
        root: MotionLibConfig(motion_file=root) for root in ExperimentConfigs.ROOTS
    }
    fields["agent"] = None
    fields.update(overrides)
    return ExperimentConfigs(**fields)


def test_experiment_configs_bundle_round_trips(tmp_path):
    """save_experiment_configs / load_experiment_configs preserve the bundle."""
    configs = _tiny_bundle()
    path = save_experiment_configs(configs, tmp_path)
    assert path.name == "resolved_configs.json"
    back = load_experiment_configs(path)
    assert isinstance(back, ExperimentConfigs)
    assert back.robot.motion_file == "robot"
    assert back.motion_lib.motion_file == "motion_lib"
    assert back.agent is None


def test_load_experiment_configs_rejects_a_foreign_file(tmp_path):
    """A JSON file that is not a bundle fails naming the regenerate step."""
    import json

    path = tmp_path / "resolved_configs.json"
    path.write_text(json.dumps({"env": {"num_envs": 2}}))
    with pytest.raises(ValueError, match="format_version"):
        load_experiment_configs(path)


def test_load_experiment_configs_from_checkpoint_reads_sibling(tmp_path):
    """The loader reads the bundle beside a checkpoint, or raises naming the path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_bytes(b"unused")
    save_experiment_configs(_tiny_bundle(), run_dir)

    loaded = load_experiment_configs_from_checkpoint(str(checkpoint))
    assert loaded.robot.motion_file == "robot"

    with pytest.raises(FileNotFoundError, match="resolved configs not found"):
        load_experiment_configs_from_checkpoint(str(tmp_path / "missing" / "model.pt"))


def test_import_simulator_before_torch_handles_supported_names(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacgym", types.ModuleType("isaacgym"))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "12.0.0")

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


def test_validate_isaaclab_version_rejects_pre_migration_runtime(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "11.9.0")

    with pytest.raises(RuntimeError, match="isaaclab>=12.0.0"):
        _validate_isaaclab_version()


def test_import_simulator_before_torch_surfaces_missing_required_packages(monkeypatch):
    monkeypatch.delitem(sys.modules, "isaacgym", raising=False)

    with pytest.raises(ModuleNotFoundError):
        import_simulator_before_torch("isaacgym")
