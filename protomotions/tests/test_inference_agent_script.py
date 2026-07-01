# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script-level tests for inference_agent.py with runtime boundaries faked."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


INFERENCE_AGENT_PATH = str(Path(__file__).resolve().parents[1] / "inference_agent.py")


class _FakeFabricConfig:
    def __init__(self, accelerator="gpu", devices=1, num_nodes=1, loggers=None, callbacks=None):
        self.accelerator = accelerator
        self.devices = devices
        self.num_nodes = num_nodes
        self.loggers = loggers or []
        self.callbacks = callbacks or []

    def as_kwargs(self):
        return {
            "accelerator": self.accelerator,
            "devices": self.devices,
            "num_nodes": self.num_nodes,
            "loggers": self.loggers,
            "callbacks": self.callbacks,
        }


class _FakeFabric:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = torch.device("cpu")
        self.launched = False
        _FakeFabric.instances.append(self)

    def launch(self):
        self.launched = True


def _install_inference_import_fakes(monkeypatch, app_launcher=None):
    simulator_imports = ModuleType("protomotions.utils.simulator_imports")
    simulator_imports.import_simulator_before_torch = lambda simulator: app_launcher
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.simulator_imports",
        simulator_imports,
    )

    fabric_config = ModuleType("protomotions.utils.fabric_config")
    fabric_config.FabricConfig = _FakeFabricConfig
    monkeypatch.setitem(sys.modules, "protomotions.utils.fabric_config", fabric_config)

    lightning = ModuleType("lightning")
    lightning.__path__ = []
    lightning_fabric = ModuleType("lightning.fabric")
    lightning_fabric.Fabric = _FakeFabric
    lightning.fabric = lightning_fabric
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "lightning.fabric", lightning_fabric)


def _load_inference_agent_globals(monkeypatch, checkpoint, simulator="newton", extra_args=None):
    _FakeFabric.instances.clear()
    _install_inference_import_fakes(monkeypatch)
    argv = [
        "inference_agent.py",
        "--checkpoint",
        str(checkpoint),
        "--simulator",
        simulator,
    ]
    if extra_args:
        argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)
    return runpy.run_path(INFERENCE_AGENT_PATH, run_name="inference_agent_unit")


def _write_resolved_configs(run_dir):
    robot_config = SimpleNamespace(name="robot")
    simulator_config = SimpleNamespace(
        _target_="protomotions.simulator.isaacgym.simulator.IsaacGymSimulator",
        num_envs=1,
        headless=False,
    )
    terrain_config = SimpleNamespace(name="terrain")
    scene_lib_config = SimpleNamespace(scene_file="old_scene.yaml", asset_root="/old/root")
    motion_lib_config = SimpleNamespace(motion_file="old.motion")
    env_config = SimpleNamespace(_target_="env.Target", save_dir="weights")
    agent_config = SimpleNamespace(_target_="agent.Target")
    torch.save(
        {
            "robot": robot_config,
            "simulator": simulator_config,
            "terrain": terrain_config,
            "scene_lib": scene_lib_config,
            "motion_lib": motion_lib_config,
            "env": env_config,
            "agent": agent_config,
        },
        run_dir / "resolved_configs_inference.pt",
    )
    return {
        "robot": robot_config,
        "simulator": simulator_config,
        "terrain": terrain_config,
        "scene_lib": scene_lib_config,
        "motion_lib": motion_lib_config,
        "env": env_config,
        "agent": agent_config,
    }


def test_inference_parser_requires_checkpoint_and_parses_options(monkeypatch, tmp_path):
    checkpoint = tmp_path / "last.ckpt"
    module = _load_inference_agent_globals(monkeypatch, checkpoint)

    parser = module["create_parser"]()
    parsed = parser.parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--simulator",
            "mujoco",
            "--full-eval",
            "--headless",
            "--num-envs",
            "4",
            "--motion-file",
            "motion.pt",
            "--scenes-file",
            "scene.yaml",
            "--command-source",
            "target=keyboard",
            "--overrides",
            "env.max_episode_length=5",
        ]
    )

    assert parsed.checkpoint == str(checkpoint)
    assert parsed.simulator == "mujoco"
    assert parsed.full_eval is True
    assert parsed.headless is True
    assert parsed.num_envs == 4
    assert parsed.motion_file == "motion.pt"
    assert parsed.scenes_file == "scene.yaml"
    assert parsed.command_source == ["target=keyboard"]
    assert parsed.overrides == ["env.max_episode_length=5"]


def test_inference_command_source_override_sets_target_keyboard_source(
    monkeypatch,
    tmp_path,
):
    from protomotions.envs.control.target_control import (
        KeyboardTargetCommandSourceConfig,
        RandomTargetCommandSourceConfig,
        TargetControlConfig,
    )

    checkpoint = tmp_path / "last.ckpt"
    module = _load_inference_agent_globals(monkeypatch, checkpoint)
    env_config = SimpleNamespace(
        control_components={
            "target": TargetControlConfig(
                command_source=RandomTargetCommandSourceConfig(
                    tar_change_time_min=2.0,
                    tar_change_time_max=8.0,
                    tar_dist_max=6.0,
                )
            )
        }
    )

    module["apply_command_source_overrides"](env_config, ["target=keyboard"])

    command_source = env_config.control_components["target"].command_source
    assert isinstance(command_source, KeyboardTargetCommandSourceConfig)
    assert [binding.key for binding in command_source.key_bindings] == [
        "W",
        "S",
        "A",
        "D",
    ]


def _install_minimal_inference_runtime(monkeypatch, module, captured):
    main_globals = module["main"].__globals__

    class FakeEnv:
        def __init__(self, **kwargs):
            captured["env_config"] = kwargs["config"]
            self.simulator = SimpleNamespace()

    class FakeEvaluator:
        def simple_test_policy(self, collect_metrics=True):
            captured["simple_test"] = collect_metrics

    class FakeAgent:
        def __init__(self, config, env, fabric, **kwargs):
            captured["agent_config"] = config
            self.evaluator = FakeEvaluator()

        def setup(self):
            pass

        def load(self, checkpoint, load_env=True, load_training_state=True):
            captured["load"] = (checkpoint, load_env, load_training_state)

    def fake_build_all_components(**kwargs):
        return {
            "terrain": None,
            "scene_lib": None,
            "motion_lib": None,
            "simulator": SimpleNamespace(),
        }

    simulator_utils = ModuleType("protomotions.simulator.base_simulator.utils")
    simulator_utils.convert_friction_for_simulator = (
        lambda terrain, simulator_config: (terrain, simulator_config)
    )
    component_builder = ModuleType("protomotions.utils.component_builder")
    component_builder.build_all_components = fake_build_all_components
    base_env = ModuleType("protomotions.envs.base_env.env")
    base_env.BaseEnv = object

    monkeypatch.setitem(
        main_globals,
        "get_class",
        lambda target: FakeEnv if target == "env.Target" else FakeAgent,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.simulator.base_simulator.utils",
        simulator_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.component_builder",
        component_builder,
    )
    monkeypatch.setitem(sys.modules, "protomotions.envs.base_env.env", base_env)


def test_inference_main_applies_pretrained_checkpoint_cli_override(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    configs = _write_resolved_configs(run_dir)
    stale_prior_path = "/remote/old/exp-20260508_200038/results/prior/last.ckpt"
    prior_checkpoint = tmp_path / "prior" / "last.ckpt"
    prior_checkpoint.parent.mkdir()
    prior_checkpoint.write_text("prior")
    configs["agent"].pretrained_modules = {
        "prior": SimpleNamespace(checkpoint_path=stale_prior_path)
    }
    torch.save(configs, run_dir / "resolved_configs_inference.pt")
    module = _load_inference_agent_globals(
        monkeypatch,
        checkpoint,
        simulator="isaacgym",
        extra_args=[
            "--overrides",
            f"agent.pretrained_modules.prior.checkpoint_path={prior_checkpoint}",
        ],
    )
    captured = {}
    _install_minimal_inference_runtime(monkeypatch, module, captured)

    module["main"]()

    assert (
        captured["agent_config"].pretrained_modules["prior"].checkpoint_path
        == str(prior_checkpoint)
    )


def test_inference_main_keeps_random_command_source_by_default(
    monkeypatch,
    tmp_path,
):
    from protomotions.envs.control.target_control import (
        RandomTargetCommandSourceConfig,
        TargetControlConfig,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    configs = _write_resolved_configs(run_dir)
    configs["env"].control_components = {
        "target": TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                tar_change_time_min=2.0,
                tar_change_time_max=8.0,
                tar_dist_max=6.0,
            )
        )
    }
    torch.save(configs, run_dir / "resolved_configs_inference.pt")
    module = _load_inference_agent_globals(
        monkeypatch,
        checkpoint,
        simulator="isaacgym",
    )
    captured = {}
    _install_minimal_inference_runtime(monkeypatch, module, captured)

    module["main"]()

    command_source = captured["env_config"].control_components["target"].command_source
    assert isinstance(command_source, RandomTargetCommandSourceConfig)
    assert command_source.tar_change_time_min == 2.0
    assert command_source.tar_change_time_max == 8.0
    assert command_source.tar_dist_max == 6.0


def test_inference_main_overrides_command_source_to_keyboard(
    monkeypatch,
    tmp_path,
):
    from protomotions.envs.control.target_control import (
        KeyboardTargetCommandSourceConfig,
        RandomTargetCommandSourceConfig,
        TargetControlConfig,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    configs = _write_resolved_configs(run_dir)
    configs["env"].control_components = {
        "target": TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                tar_change_time_min=2.0,
                tar_change_time_max=8.0,
                tar_dist_max=6.0,
            )
        )
    }
    torch.save(configs, run_dir / "resolved_configs_inference.pt")
    module = _load_inference_agent_globals(
        monkeypatch,
        checkpoint,
        simulator="isaacgym",
        extra_args=["--command-source", "target=keyboard"],
    )
    captured = {}
    _install_minimal_inference_runtime(monkeypatch, module, captured)

    module["main"]()

    command_source = captured["env_config"].control_components["target"].command_source
    assert isinstance(command_source, KeyboardTargetCommandSourceConfig)
    assert [binding.key for binding in command_source.key_bindings] == [
        "W",
        "S",
        "A",
        "D",
    ]


def test_inference_main_does_not_infer_pretrained_checkpoint_paths(
    monkeypatch,
    tmp_path,
):
    exps_dir = tmp_path / "exps"
    run_dir = exps_dir / "exp-20260510_185815"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    configs = _write_resolved_configs(run_dir)
    stale_prior_path = (
        "/remote/exps/user/exp-20260508_200038/"
        "results/200038_gpc_prior/last.ckpt"
    )
    local_prior_checkpoint = exps_dir / "exp-20260508_200038" / "last.ckpt"
    local_prior_checkpoint.parent.mkdir()
    local_prior_checkpoint.write_text("prior")
    configs["agent"].pretrained_modules = {
        "prior": SimpleNamespace(checkpoint_path=stale_prior_path)
    }
    torch.save(configs, run_dir / "resolved_configs_inference.pt")
    module = _load_inference_agent_globals(monkeypatch, checkpoint, simulator="isaacgym")
    captured = {}
    _install_minimal_inference_runtime(monkeypatch, module, captured)

    module["main"]()

    assert (
        captured["agent_config"].pretrained_modules["prior"].checkpoint_path
        == stale_prior_path
    )


def test_inference_pretrained_override_parser_supports_multiple_modules(
    monkeypatch,
    tmp_path,
):
    from protomotions.utils.config_utils import (
        apply_config_overrides,
        parse_cli_overrides,
    )

    checkpoint = tmp_path / "last.ckpt"
    _load_inference_agent_globals(monkeypatch, checkpoint)
    agent_config = SimpleNamespace(
        pretrained_modules={
            "prior": SimpleNamespace(checkpoint_path="old-prior.ckpt"),
            "tracker": SimpleNamespace(checkpoint_path="old-tracker.ckpt"),
        }
    )
    overrides = parse_cli_overrides(
        [
            "agent.pretrained_modules.prior.checkpoint_path=new-prior.ckpt",
            "agent.pretrained_modules.tracker.checkpoint_path=new-tracker.ckpt",
        ]
    )

    apply_config_overrides(
        overrides,
        env_config=SimpleNamespace(),
        simulator_config=SimpleNamespace(),
        robot_config=SimpleNamespace(),
        agent_config=agent_config,
    )

    assert agent_config.pretrained_modules["prior"].checkpoint_path == "new-prior.ckpt"
    assert (
        agent_config.pretrained_modules["tracker"].checkpoint_path == "new-tracker.ckpt"
    )


def test_inference_main_full_eval_switches_simulator_and_applies_cli_overrides(
    monkeypatch,
    tmp_path,
    capsys,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    configs = _write_resolved_configs(run_dir)
    module = _load_inference_agent_globals(
        monkeypatch,
        checkpoint,
        simulator="newton",
        extra_args=[
            "--full-eval",
            "--headless",
            "--num-envs",
            "3",
            "--motion-file",
            "override.motion",
            "--scenes-file",
            "none",
            "--overrides",
            "env.foo=1",
        ],
    )
    main_globals = module["main"].__globals__
    calls = []

    class FakeEnv:
        def __init__(self, **kwargs):
            calls.append(("env_init", kwargs))
            self.simulator = kwargs["simulator"]

    class FakeEvaluator:
        def __init__(self):
            self.eval_count = 5

        def evaluate(self):
            calls.append(("evaluate", self.eval_count))
            return {"score/a": 1.25, "score/b": 2.5}, 3.75, 2

        def simple_test_policy(self, collect_metrics=True):
            calls.append(("simple_test", collect_metrics))

    class FakeAgent:
        def __init__(self, config, env, fabric, **kwargs):
            calls.append(("agent_init", kwargs))
            self.evaluator = FakeEvaluator()

        def setup(self):
            calls.append(("agent_setup", None))

        def load(self, checkpoint, load_env=True, load_training_state=True):
            calls.append(("agent_load", checkpoint, load_env, load_training_state))

    simulator = SimpleNamespace(
        shutdown=lambda: calls.append(("shutdown", None)),
    )

    def fake_build_all_components(**kwargs):
        calls.append(("build_components", kwargs))
        return {
            "terrain": "terrain",
            "scene_lib": "scene",
            "motion_lib": "motion",
            "simulator": simulator,
        }

    simulator_factory = ModuleType("protomotions.simulator.factory")
    simulator_factory.update_simulator_config_for_test = (
        lambda current_simulator_config, new_simulator, robot_config: SimpleNamespace(
            _target_="protomotions.simulator.newton.simulator.NewtonSimulator",
            num_envs=current_simulator_config.num_envs,
            headless=current_simulator_config.headless,
            switched_to=new_simulator,
        )
    )
    config_utils = ModuleType("protomotions.utils.config_utils")
    config_utils.parse_cli_overrides = lambda overrides: {"env.foo": 1}
    config_utils.apply_config_overrides = (
        lambda overrides, env, simulator, robot, agent, terrain, motion, scene: setattr(
            env,
            "overridden",
            overrides,
        )
    )
    simulator_utils = ModuleType("protomotions.simulator.base_simulator.utils")
    simulator_utils.convert_friction_for_simulator = (
        lambda terrain, simulator_config: (terrain, simulator_config)
    )
    component_builder = ModuleType("protomotions.utils.component_builder")
    component_builder.build_all_components = fake_build_all_components
    base_env = ModuleType("protomotions.envs.base_env.env")
    base_env.BaseEnv = object

    monkeypatch.setitem(
        main_globals,
        "get_class",
        lambda target: FakeEnv if target == "env.Target" else FakeAgent,
    )
    monkeypatch.setitem(sys.modules, "protomotions.simulator.factory", simulator_factory)
    monkeypatch.setitem(sys.modules, "protomotions.utils.config_utils", config_utils)
    monkeypatch.setitem(
        sys.modules,
        "protomotions.simulator.base_simulator.utils",
        simulator_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.component_builder",
        component_builder,
    )
    monkeypatch.setitem(sys.modules, "protomotions.envs.base_env.env", base_env)

    module["main"]()

    output = capsys.readouterr().out
    assert "EVALUATION RESULTS" in output
    assert "Overall Score: 3.750000" in output
    assert _FakeFabric.instances[0].kwargs["accelerator"] == "gpu"
    assert configs["simulator"].num_envs == 1
    build_call = next(call for call in calls if call[0] == "build_components")
    assert build_call[1]["simulator_config"].num_envs == 3
    assert build_call[1]["simulator_config"].headless is True
    assert build_call[1]["simulator_config"].switched_to == "newton"
    assert build_call[1]["motion_lib_config"].motion_file == "override.motion"
    assert build_call[1]["scene_lib_config"].scene_file is None
    assert build_call[1]["scene_lib_config"].asset_root is None
    env_call = next(call for call in calls if call[0] == "env_init")
    assert env_call[1]["config"].overridden == {"env.foo": 1}
    assert ("evaluate", 0) in calls
    assert ("agent_load", str(checkpoint), False, False) in calls
    assert ("shutdown", None) in calls


def test_inference_main_simple_test_recomputes_scene_asset_root(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    _write_resolved_configs(run_dir)
    scene_file = tmp_path / "assets" / "scenes" / "scene.yaml"
    scene_file.parent.mkdir(parents=True)
    scene_file.write_text("scene")
    module = _load_inference_agent_globals(
        monkeypatch,
        checkpoint,
        simulator="isaacgym",
        extra_args=["--scenes-file", str(scene_file)],
    )
    main_globals = module["main"].__globals__
    calls = []

    class FakeEnv:
        def __init__(self, **kwargs):
            self.simulator = SimpleNamespace()

    class FakeEvaluator:
        def simple_test_policy(self, collect_metrics=True):
            calls.append(("simple_test", collect_metrics))

    class FakeAgent:
        def __init__(self, config, env, fabric, **kwargs):
            self.evaluator = FakeEvaluator()

        def setup(self):
            pass

        def load(self, checkpoint, load_env=True, load_training_state=True):
            calls.append(("load", load_env, load_training_state))

    def fake_build_all_components(**kwargs):
        calls.append(("asset_root", kwargs["scene_lib_config"].asset_root))
        return {
            "terrain": None,
            "scene_lib": None,
            "motion_lib": None,
            "simulator": SimpleNamespace(),
        }

    simulator_utils = ModuleType("protomotions.simulator.base_simulator.utils")
    simulator_utils.convert_friction_for_simulator = (
        lambda terrain, simulator_config: (terrain, simulator_config)
    )
    component_builder = ModuleType("protomotions.utils.component_builder")
    component_builder.build_all_components = fake_build_all_components
    base_env = ModuleType("protomotions.envs.base_env.env")
    base_env.BaseEnv = object

    monkeypatch.setitem(
        main_globals,
        "get_class",
        lambda target: FakeEnv if target == "env.Target" else FakeAgent,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.simulator.base_simulator.utils",
        simulator_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.component_builder",
        component_builder,
    )
    monkeypatch.setitem(sys.modules, "protomotions.envs.base_env.env", base_env)

    module["main"]()

    assert ("asset_root", str(tmp_path / "assets")) in calls
    assert ("simple_test", True) in calls
    assert ("load", False, False) in calls


def test_inference_main_requires_resolved_configs(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    module = _load_inference_agent_globals(monkeypatch, checkpoint)

    with pytest.raises(AssertionError, match="Could not find resolved configs"):
        module["main"]()
