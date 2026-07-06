# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for train_agent.py helper paths."""

import json
import os
import runpy
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from protomotions.utils.cli_utils import parse_bool


TRAIN_AGENT_PATH = str(Path(__file__).resolve().parents[1] / "train_agent.py")


@dataclass
class _TinyConfig:
    _target_: str = "unit.Target"
    value: int = 1


class _TinyFabricConfig:
    def as_loggable_dict(self):
        return {"devices": 1, "num_nodes": 1}


def _install_train_agent_import_fakes(monkeypatch):
    simulator_imports = ModuleType("protomotions.utils.simulator_imports")
    simulator_imports.import_simulator_before_torch = lambda simulator: None
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.simulator_imports",
        simulator_imports,
    )

    wandb = ModuleType("wandb")
    wandb.run = SimpleNamespace(id="wandb-unit")
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    class FakeWandbLogger:
        def __init__(self):
            self.logged = None

        def log_hyperparams(self, params):
            self.logged = params

    lightning = ModuleType("lightning")
    lightning_pytorch = ModuleType("lightning.pytorch")
    lightning_loggers = ModuleType("lightning.pytorch.loggers")
    lightning_loggers.WandbLogger = FakeWandbLogger
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", lightning_pytorch)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.loggers", lightning_loggers)

    utils_pkg = ModuleType("utils")
    torch_utils = ModuleType("utils.torch_utils")
    torch_utils.seeding = lambda seed, torch_deterministic=False: None
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.torch_utils", torch_utils)

    return FakeWandbLogger


def _load_train_agent_globals(monkeypatch, tmp_path):
    _install_train_agent_import_fakes(monkeypatch)
    experiment = tmp_path / "experiment.py"
    experiment.write_text("VALUE = 1\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_agent.py",
            "--robot-name",
            "g1",
            "--simulator",
            "isaacgym",
            "--num-envs",
            "4",
            "--batch-size",
            "8",
            "--motion-file",
            "motions.pt",
            "--experiment-path",
            str(experiment),
            "--experiment-name",
            "unit",
            "--headless",
            "off",
            "--overrides",
            "env.max_episode_length=12",
        ],
    )
    return runpy.run_path(TRAIN_AGENT_PATH, run_name="train_agent_unit")


def test_train_agent_parser_and_bool_helpers(monkeypatch, tmp_path):
    module = _load_train_agent_globals(monkeypatch, tmp_path)

    assert module["args"].headless is False
    assert module["args"].overrides == ["env.max_episode_length=12"]
    assert parse_bool(True) is True
    assert parse_bool("yes") is True
    assert parse_bool("0") is False

    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("maybe")

    parser = module["create_parser"]()
    parsed = parser.parse_args(
        [
            "--robot-name",
            "g1",
            "--simulator",
            "newton",
            "--num-envs",
            "2",
            "--batch-size",
            "4",
            "--motion-file",
            "m.pt",
            "--experiment-path",
            "exp.py",
            "--experiment-name",
            "parser",
            "--headless",
            "false",
            "--torch-deterministic",
            "--create-config-only",
        ]
    )
    assert parsed.headless is False
    assert parsed.torch_deterministic is True
    assert parsed.create_config_only is True


def test_wbc_collective_fix_env_is_default_off(monkeypatch, tmp_path):
    for name in (
        "PG_TIMEOUT_SEC",
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "FIX_WBC_MATERIALIZE_COLLECTIVE",
        "FIX_WBC_RMS_COLLECTIVE_SCHEDULE",
        "FIX_WBC_METRIC_COLLECTIVE_SCHEDULE",
        "FIX_WBC_GRAD_CLIP_COLLECTIVE_SCHEDULE",
    ):
        monkeypatch.delenv(name, raising=False)

    module = _load_train_agent_globals(monkeypatch, tmp_path)

    assert module["_enable_wbc_collective_fixes_for_experiment"]("unit") is False
    assert "FIX_WBC_RMS_COLLECTIVE_SCHEDULE" not in os.environ


def test_wbc_collective_fix_env_enabled_for_masked_and_superdr(monkeypatch, tmp_path):
    module = _load_train_agent_globals(monkeypatch, tmp_path)

    for experiment_name in (
        "h1_2_masked_mimic_teleop",
        "h1_2_superdrmegaperturb_scratch",
    ):
        for name in module["_WBC_COLLECTIVE_FIX_ENV"]:
            monkeypatch.delenv(name, raising=False)
        for name in module["_WBC_MASKED_MIMIC_ONLY_FIX_ENV"]:
            monkeypatch.delenv(name, raising=False)

        assert (
            module["_enable_wbc_collective_fixes_for_experiment"](experiment_name)
            is True
        )
        for name, value in module["_WBC_COLLECTIVE_FIX_ENV"].items():
            assert os.environ[name] == value
        if experiment_name == "h1_2_masked_mimic_teleop":
            for name, value in module["_WBC_MASKED_MIMIC_ONLY_FIX_ENV"].items():
                assert os.environ[name] == value
        else:
            for name in module["_WBC_MASKED_MIMIC_ONLY_FIX_ENV"]:
                assert name not in os.environ


def test_detect_checkpoint_mode_handles_fresh_warm_start_and_resume(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    detect_checkpoint_mode = module["detect_checkpoint_mode"]

    fresh_args = SimpleNamespace(checkpoint=None)
    assert detect_checkpoint_mode(fresh_args, tmp_path / "fresh") == (
        "fresh",
        None,
        None,
    )

    warm_args = SimpleNamespace(checkpoint=str(tmp_path / "weights.ckpt"))
    assert detect_checkpoint_mode(warm_args, tmp_path / "warm") == (
        "warm_start",
        tmp_path / "weights.ckpt",
        None,
    )

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "last.ckpt").write_text("checkpoint")
    (resume_dir / "config.yaml").write_text(
        json.dumps(
            {
                "robot_name": "smpl",
                "batch_size": 64,
                "wandb_id": "wandb-resume",
            }
        )
    )
    resume_args = SimpleNamespace(checkpoint=None, robot_name="g1")

    mode, checkpoint, wandb_id = detect_checkpoint_mode(resume_args, resume_dir)

    assert mode == "resume"
    assert checkpoint == resume_dir / "last.ckpt"
    assert wandb_id == "wandb-resume"
    assert resume_args.robot_name == "smpl"
    assert resume_args.batch_size == 64

    missing_config_dir = tmp_path / "missing_config"
    missing_config_dir.mkdir()
    (missing_config_dir / "last.ckpt").write_text("checkpoint")
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        detect_checkpoint_mode(SimpleNamespace(checkpoint=None), missing_config_dir)


def test_load_experiment_module_loads_file_and_rejects_missing_paths(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    experiment = tmp_path / "custom_experiment.py"
    experiment.write_text("VALUE = 7\n")

    loaded = module["load_experiment_module"](experiment)

    assert loaded.VALUE == 7
    with pytest.raises(FileNotFoundError, match="Experiment file not found"):
        module["load_experiment_module"](tmp_path / "missing.py")


def test_save_configs_writes_json_pickle_yaml_and_experiment_copy(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    save_dir = tmp_path / "results"
    experiment = tmp_path / "experiment.py"
    experiment.write_text("# experiment\n")
    args = SimpleNamespace(use_wandb=True, experiment_name="unit", seed=123)
    config = _TinyConfig(value=5)

    module["save_configs"](
        save_dir,
        args,
        config,
        config,
        config,
        config,
        config,
        config,
        config,
        _TinyFabricConfig(),
        experiment_source_path=experiment,
    )

    checkpoint_config = json.loads((save_dir / "config.yaml").read_text())
    assert checkpoint_config["wandb_id"] == "wandb-unit"
    assert checkpoint_config["seed"] == 123
    assert (save_dir / "resolved_configs.pt").exists()
    assert (save_dir / "resolved_configs.yaml").read_text()
    assert (save_dir / "experiment_config.py").read_text() == "# experiment\n"

    loaded = torch.load(
        save_dir / "resolved_configs.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert loaded["robot"].value == 5


def test_save_configs_tolerates_missing_wandb_id_and_yaml_failure(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    module["wandb"].run = object()
    save_dir = tmp_path / "results"
    experiment = tmp_path / "experiment.py"
    experiment.write_text("# experiment\n")
    args = SimpleNamespace(use_wandb=True, experiment_name="unit")
    config = SimpleNamespace(value=5)

    module["save_configs"](
        save_dir,
        args,
        config,
        config,
        config,
        config,
        config,
        config,
        config,
        _TinyFabricConfig(),
        experiment_source_path=experiment,
        file_name="no_yaml",
    )

    checkpoint_config = json.loads((save_dir / "config.yaml").read_text())
    assert checkpoint_config["wandb_id"] is None
    assert (save_dir / "no_yaml.pt").exists()
    assert not (save_dir / "no_yaml.yaml").exists()


def test_prepare_inference_configs_for_save_calls_optional_hooks(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)

    class ConfigWithHook:
        called = False

        def prepare_inference_config_for_save(self):
            self.called = True

    config_with_hook = ConfigWithHook()
    config_without_hook = object()

    module["prepare_inference_configs_for_save"](
        config_without_hook,
        config_with_hook,
    )

    assert config_with_hook.called


def test_try_log_hyperparams_to_wandb_logs_only_matching_logger(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    logger = module["WandbLogger"]()
    fabric = SimpleNamespace(loggers=[object(), logger])
    config = _TinyConfig(value=9)

    module["try_log_hyperparams_to_wandb"](
        fabric,
        config,
        config,
        config,
        config,
        config,
        config,
        config,
        _TinyFabricConfig(),
    )

    assert logger.logged["robot"]["value"] in (9, "9")
    assert logger.logged["fabric"] in (
        {"devices": 1, "num_nodes": 1},
        {"devices": "1", "num_nodes": "1"},
    )


def test_try_log_hyperparams_to_wandb_tolerates_logger_failures(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)

    class FailingWandbLogger(module["WandbLogger"]):
        def log_hyperparams(self, params):
            raise RuntimeError("wandb is offline")

    module["try_log_hyperparams_to_wandb"](
        SimpleNamespace(loggers=[FailingWandbLogger()]),
        _TinyConfig(value=9),
        _TinyConfig(value=9),
        _TinyConfig(value=9),
        _TinyConfig(value=9),
        _TinyConfig(value=9),
        _TinyConfig(value=9),
        _TinyConfig(value=9),
        _TinyFabricConfig(),
    )


def test_create_config_only_saves_training_and_inference_configs(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)

    save_calls = []
    override_calls = []

    def fake_save_configs(
        save_dir,
        args,
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
        fabric_config,
        experiment_source_path,
        file_name,
    ):
        save_calls.append(
            {
                "file_name": file_name,
                "robot_value": robot_config.value,
                "fabric": fabric_config.as_loggable_dict(),
                "experiment_source_path": experiment_source_path,
            }
        )

    def fake_apply_all_inference_overrides(
        robot_config,
        simulator_config,
        env_config,
        agent_config,
        terrain_config,
        motion_lib_config,
        scene_lib_config,
        experiment_module,
        args,
    ):
        override_calls.append((experiment_module, args.experiment_name))
        robot_config.value = 99

    fabric_config_module = ModuleType("protomotions.utils.fabric_config")

    class FakeFabricConfig:
        def __init__(self, devices, num_nodes, loggers, callbacks):
            self.devices = devices
            self.num_nodes = num_nodes
            self.loggers = loggers
            self.callbacks = callbacks

        def as_loggable_dict(self):
            return {"devices": self.devices, "num_nodes": self.num_nodes}

    fabric_config_module.FabricConfig = FakeFabricConfig
    inference_utils = ModuleType("protomotions.utils.inference_utils")
    inference_utils.apply_all_inference_overrides = fake_apply_all_inference_overrides

    monkeypatch.setitem(
        module["_handle_create_config_only"].__globals__,
        "save_configs",
        fake_save_configs,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.fabric_config",
        fabric_config_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.inference_utils",
        inference_utils,
    )

    args = SimpleNamespace(ngpu=2, nodes=3, experiment_name="config-only")
    experiment = tmp_path / "experiment.py"
    config = _TinyConfig(value=5)

    module["_handle_create_config_only"](
        args,
        tmp_path / "results",
        experiment,
        experiment_module=SimpleNamespace(name="experiment"),
        robot_config=config,
        simulator_config=config,
        terrain_config=config,
        scene_lib_config=config,
        motion_lib_config=config,
        env_config=config,
        agent_config=config,
    )

    assert [call["file_name"] for call in save_calls] == [
        "resolved_configs",
        "resolved_configs_inference",
    ]
    assert save_calls[0]["robot_value"] == 5
    assert save_calls[1]["robot_value"] == 99
    assert save_calls[0]["fabric"]["devices"] == 2
    assert save_calls[0]["fabric"]["num_nodes"] == 3
    assert override_calls == [(SimpleNamespace(name="experiment"), "config-only")]


def test_main_create_config_only_builds_configs_and_exits_before_training(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    main_globals = module["main"].__globals__
    config = _TinyConfig(value=11)
    args = SimpleNamespace(
        experiment_name="main-config-only",
        experiment_path=str(tmp_path / "experiment.py"),
        create_config_only=True,
        checkpoint=None,
        overrides=[],
        ngpu=1,
        nodes=1,
    )
    fake_experiment = SimpleNamespace(
        terrain_config=lambda: "terrain",
        scene_lib_config=lambda: "scene",
        motion_lib_config=lambda: "motion",
        env_config=lambda: "env",
        configure_robot_and_simulator=lambda *args, **kwargs: None,
        agent_config=lambda: "agent",
    )
    handled = []

    def fake_build_standard_configs(**kwargs):
        assert kwargs["terrain_config_fn"] is fake_experiment.terrain_config
        assert kwargs["agent_config_fn"] is fake_experiment.agent_config
        return {
            "robot": config,
            "simulator": config,
            "terrain": config,
            "scene_lib": config,
            "motion_lib": config,
            "env": config,
            "agent": config,
        }

    config_builder = ModuleType("protomotions.utils.config_builder")
    config_builder.build_standard_configs = fake_build_standard_configs

    def fake_handle_create_config_only(
        parsed_args,
        save_dir,
        experiment_source_path,
        experiment_module,
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
    ):
        handled.append(
            {
                "args": parsed_args,
                "save_dir": save_dir,
                "experiment_source_path": experiment_source_path,
                "experiment_module": experiment_module,
                "robot_config": robot_config,
                "agent_config": agent_config,
            }
        )

    monkeypatch.setitem(main_globals, "args", args)
    monkeypatch.setitem(
        main_globals,
        "parser",
        SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setitem(
        main_globals,
        "load_experiment_module",
        lambda path: fake_experiment,
    )
    monkeypatch.setitem(
        main_globals,
        "_handle_create_config_only",
        fake_handle_create_config_only,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.config_builder",
        config_builder,
    )

    module["main"]()

    assert len(handled) == 1
    assert handled[0]["save_dir"] == Path("results") / "main-config-only"
    assert handled[0]["experiment_source_path"] == Path(args.experiment_path)
    assert handled[0]["experiment_module"] is fake_experiment
    assert handled[0]["robot_config"] is config
    assert handled[0]["agent_config"] is config


def test_main_config_only_registers_custom_args_and_applies_cli_overrides(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    main_globals = module["main"].__globals__
    parser_calls = []
    override_calls = []
    config = _TinyConfig(value=11)
    args = SimpleNamespace(
        experiment_name="main-config-overrides",
        experiment_path=str(tmp_path / "experiment.py"),
        create_config_only=True,
        checkpoint=None,
        overrides=["env.value=17"],
        ngpu=1,
        nodes=1,
    )

    fake_parser = SimpleNamespace(
        add_argument=lambda *items, **kwargs: parser_calls.append((items, kwargs)),
        parse_args=lambda: args,
    )
    fake_experiment = SimpleNamespace(
        additional_experiment_arguments=lambda parser: parser.add_argument(
            "--custom-scale", type=float, default=1.0
        ),
        terrain_config=lambda: "terrain",
        scene_lib_config=lambda: "scene",
        motion_lib_config=lambda: "motion",
        env_config=lambda: "env",
    )

    def fake_build_standard_configs(**kwargs):
        assert kwargs["configure_robot_and_simulator_fn"] is None
        assert kwargs["agent_config_fn"] is None
        return {
            "robot": config,
            "simulator": config,
            "terrain": config,
            "scene_lib": config,
            "motion_lib": config,
            "env": config,
            "agent": config,
        }

    config_builder = ModuleType("protomotions.utils.config_builder")
    config_builder.build_standard_configs = fake_build_standard_configs

    monkeypatch.setitem(main_globals, "args", args)
    monkeypatch.setitem(main_globals, "parser", fake_parser)
    monkeypatch.setitem(
        main_globals,
        "load_experiment_module",
        lambda path: fake_experiment,
    )
    monkeypatch.setitem(
        main_globals,
        "_handle_create_config_only",
        lambda *items, **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.config_builder",
        config_builder,
    )
    monkeypatch.setattr(
        sys.modules["protomotions.utils.config_utils"],
        "parse_cli_overrides",
        lambda raw: {"env.value": 17},
    )
    monkeypatch.setattr(
        sys.modules["protomotions.utils.config_utils"],
        "apply_config_overrides",
        lambda overrides, env, simulator, robot, agent, **kwargs: override_calls.append(
            (overrides, env, simulator, robot, agent, kwargs)
        ),
    )

    module["main"]()

    assert parser_calls == [(("--custom-scale",), {"type": float, "default": 1.0})]
    assert len(override_calls) == 1
    assert override_calls[0][0] == {"env.value": 17}
    assert override_calls[0][1] is config
    assert override_calls[0][5]["terrain_config"] is config


def test_main_fresh_training_path_wires_fabric_components_agent_and_saves(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    main_globals = module["main"].__globals__
    calls = []
    args = SimpleNamespace(
        experiment_name="fresh",
        experiment_path=str(tmp_path / "experiment.py"),
        create_config_only=False,
        checkpoint=None,
        overrides=[],
        ngpu=2,
        nodes=1,
        use_wandb=True,
        use_slurm=True,
        simulator="isaacgym",
        headless=True,
        seed=10,
        torch_deterministic=True,
    )
    robot_config = SimpleNamespace(_target_="robot.Target")
    simulator_config = SimpleNamespace(_target_="sim.Target")
    terrain_config = SimpleNamespace(_target_="terrain.Target")
    scene_lib_config = SimpleNamespace(_target_="scene.Target")
    motion_lib_config = SimpleNamespace(_target_="motion.Target")
    env_config = SimpleNamespace(_target_="env.Target", save_dir="weights")
    agent_config = SimpleNamespace(_target_="agent.Target")
    fake_experiment = SimpleNamespace(
        terrain_config=lambda: "terrain",
        scene_lib_config=lambda: "scene",
        motion_lib_config=lambda: "motion",
        env_config=lambda: "env",
        configure_robot_and_simulator=lambda *args, **kwargs: None,
        agent_config=lambda: "agent",
    )

    class FakeFabricConfig:
        def __init__(self, devices, num_nodes, loggers, callbacks, **kwargs):
            self.devices = devices
            self.num_nodes = num_nodes
            self.loggers = loggers
            self.callbacks = callbacks

        def as_kwargs(self):
            return {
                "devices": self.devices,
                "num_nodes": self.num_nodes,
                "loggers": self.loggers,
                "callbacks": self.callbacks,
            }

        def as_loggable_dict(self):
            return {"devices": self.devices, "num_nodes": self.num_nodes}

    class FakeFabric:
        def __init__(self, **kwargs):
            calls.append(("fabric_init", kwargs))
            self.device = torch.device("cpu")
            self.world_size = 1
            self.global_rank = 0
            self.local_rank = 0
            self.strategy = SimpleNamespace(
                barrier=lambda: calls.append(("agent_barrier", None))
            )
            self.loggers = []

        def launch(self):
            calls.append(("launch", None))

        def seed_everything(self, seed):
            calls.append(("seed_everything", seed))

        def call(self, name, *args, **kwargs):
            calls.append(("fabric_call", name))

    class FakeEnv:
        def __init__(self, **kwargs):
            calls.append(("env_init", kwargs))
            self.simulator = kwargs["simulator"]

    class FakeAgent:
        def __init__(self, config, env, fabric):
            calls.append(("agent_init", config, env))
            self.fabric = fabric
            self.loaded = None

        def setup(self):
            calls.append(("agent_setup", None))

        def load(self, checkpoint, load_training_state: bool = True):
            self.loaded = checkpoint
            calls.append(("agent_load", checkpoint, load_training_state))

        def fit(self):
            calls.append(("agent_fit", None))

    def fake_build_standard_configs(**kwargs):
        return {
            "robot": robot_config,
            "simulator": simulator_config,
            "terrain": terrain_config,
            "scene_lib": scene_lib_config,
            "motion_lib": motion_lib_config,
            "env": env_config,
            "agent": agent_config,
        }

    def fake_build_all_components(**kwargs):
        calls.append(("build_components", kwargs))
        return {
            "terrain": "terrain",
            "scene_lib": "scene_lib",
            "motion_lib": "motion_lib",
            "simulator": SimpleNamespace(name="simulator"),
        }

    def fake_save_configs(*args, **kwargs):
        calls.append(("save_configs", kwargs["file_name"]))

    def fake_apply_all_inference_overrides(*args, **kwargs):
        calls.append(("inference_overrides", kwargs["args"].experiment_name))

    fabric_config_module = ModuleType("protomotions.utils.fabric_config")
    fabric_config_module.FabricConfig = FakeFabricConfig
    lightning_fabric = ModuleType("lightning.fabric")
    lightning_fabric.Fabric = FakeFabric
    sys.modules["lightning"].__path__ = []
    sys.modules["lightning"].fabric = lightning_fabric
    config_builder = ModuleType("protomotions.utils.config_builder")
    config_builder.build_standard_configs = fake_build_standard_configs
    component_builder = ModuleType("protomotions.utils.component_builder")
    component_builder.build_all_components = fake_build_all_components
    inference_utils = ModuleType("protomotions.utils.inference_utils")
    inference_utils.apply_all_inference_overrides = fake_apply_all_inference_overrides
    simulator_utils = ModuleType("protomotions.simulator.base_simulator.utils")
    simulator_utils.convert_friction_for_simulator = (
        lambda terrain, simulator: (terrain, simulator)
    )
    base_env = ModuleType("protomotions.envs.base_env.env")
    base_env.BaseEnv = object

    monkeypatch.setitem(main_globals, "args", args)
    monkeypatch.setitem(
        main_globals,
        "parser",
        SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setitem(
        main_globals,
        "detect_checkpoint_mode",
        lambda parsed_args, save_dir: ("fresh", None, None),
    )
    monkeypatch.setitem(
        main_globals,
        "load_experiment_module",
        lambda path: fake_experiment,
    )
    monkeypatch.setitem(
        main_globals,
        "get_class",
        lambda target: FakeEnv if target == "env.Target" else FakeAgent,
    )
    monkeypatch.setitem(main_globals, "save_configs", fake_save_configs)
    monkeypatch.setitem(
        main_globals,
        "try_log_hyperparams_to_wandb",
        lambda *args, **kwargs: calls.append(("try_log_hparams", None)),
    )
    monkeypatch.setitem(
        main_globals,
        "seeding",
        lambda seed, torch_deterministic=False: calls.append(
            ("seeding", seed, torch_deterministic)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.fabric_config",
        fabric_config_module,
    )
    monkeypatch.setitem(sys.modules, "lightning.fabric", lightning_fabric)
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.config_builder",
        config_builder,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.component_builder",
        component_builder,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.inference_utils",
        inference_utils,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.simulator.base_simulator.utils",
        simulator_utils,
    )
    monkeypatch.setitem(sys.modules, "protomotions.envs.base_env.env", base_env)

    module["main"]()

    assert ("launch", None) in calls
    assert ("seed_everything", 10) in calls
    assert ("seeding", 10, True) in calls
    assert ("fabric_call", "on_app_start") in calls
    assert ("fabric_call", "on_env_init_start") in calls
    assert ("fabric_call", "on_env_init_end") in calls
    assert ("agent_setup", None) in calls
    assert ("agent_load", None, False) in calls
    assert ("agent_fit", None) in calls
    assert ("try_log_hparams", None) in calls
    assert ("save_configs", "resolved_configs") in calls
    assert ("save_configs", "resolved_configs_inference") in calls
    build_call = next(call for call in calls if call[0] == "build_components")
    assert build_call[1]["save_dir"] == "weights"
    fabric_call = next(call for call in calls if call[0] == "fabric_init")
    assert len(fabric_call[1]["loggers"]) == 2
    assert len(fabric_call[1]["callbacks"]) == 1


def test_main_resume_isaaclab_uses_saved_configs_launcher_and_skip_flag(
    monkeypatch,
    tmp_path,
):
    module = _load_train_agent_globals(monkeypatch, tmp_path)
    main_globals = module["main"].__globals__
    monkeypatch.chdir(tmp_path)
    calls = []
    resume_dir = tmp_path / "results" / "resume"
    resume_dir.mkdir(parents=True)
    checkpoint_path = resume_dir / "last.ckpt"
    checkpoint_path.write_text("checkpoint")
    (resume_dir / "config.yaml").write_text(
        json.dumps(
            {
                "experiment_name": "resume",
                "experiment_path": str(tmp_path / "ignored_experiment.py"),
                "create_config_only": False,
                "checkpoint": None,
                "overrides": ["saved.override=1"],
                "ngpu": 1,
                "nodes": 2,
                "use_wandb": False,
                "use_slurm": False,
                "simulator": "isaaclab",
                "headless": False,
                "seed": None,
                "torch_deterministic": False,
            }
        )
    )
    robot_config = SimpleNamespace(_target_="robot.Target")
    simulator_config = SimpleNamespace(_target_="sim.Target")
    terrain_config = SimpleNamespace(_target_="terrain.Target")
    scene_lib_config = SimpleNamespace(_target_="scene.Target")
    motion_lib_config = SimpleNamespace(_target_="motion.Target")
    env_config = SimpleNamespace(_target_="env.Target", save_dir="resume_weights")
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
        resume_dir / "resolved_configs.pt",
    )

    args = SimpleNamespace(
        experiment_name="resume",
        experiment_path=str(tmp_path / "ignored_experiment.py"),
        create_config_only=False,
        checkpoint=None,
        overrides=["user.override=2"],
        ngpu=99,
        nodes=99,
    )

    class FakeAppLauncher:
        def __init__(self, flags):
            calls.append(("app_launcher", flags))
            self.app = "simulation-app"

    class FakeFabricConfig:
        def __init__(self, devices, num_nodes, loggers, callbacks, **kwargs):
            self.devices = devices
            self.num_nodes = num_nodes
            self.loggers = loggers
            self.callbacks = callbacks

        def as_kwargs(self):
            return {"devices": self.devices, "num_nodes": self.num_nodes}

        def as_loggable_dict(self):
            return {"devices": self.devices, "num_nodes": self.num_nodes}

    class FakeFabric:
        def __init__(self, **kwargs):
            calls.append(("fabric_init", kwargs))
            self.device = torch.device("cpu")
            self.world_size = 2
            self.global_rank = 3
            self.local_rank = 1
            self.strategy = SimpleNamespace(
                barrier=lambda: calls.append(("agent_barrier", None))
            )
            self.loggers = []

        def launch(self):
            calls.append(("launch", None))

        def call(self, name, *args, **kwargs):
            calls.append(("fabric_call", name))

    class FakeEnv:
        def __init__(self, **kwargs):
            calls.append(("env_init", kwargs))

    class FakeAgent:
        def __init__(self, config, env, fabric):
            self.fabric = fabric
            self._skip_next_policy_update = False
            calls.append(("agent_init", config, env))

        def setup(self):
            calls.append(("agent_setup", None))

        def load(self, checkpoint, load_training_state: bool = True):
            calls.append(("agent_load", checkpoint, load_training_state))

        def fit(self):
            calls.append(("agent_fit", self._skip_next_policy_update))

    def fake_build_all_components(**kwargs):
        calls.append(("build_components", kwargs))
        return {
            "terrain": "terrain",
            "scene_lib": "scene_lib",
            "motion_lib": "motion_lib",
            "simulator": "simulator",
        }

    fabric_config_module = ModuleType("protomotions.utils.fabric_config")
    fabric_config_module.FabricConfig = FakeFabricConfig
    lightning_fabric = ModuleType("lightning.fabric")
    lightning_fabric.Fabric = FakeFabric
    sys.modules["lightning"].__path__ = []
    sys.modules["lightning"].fabric = lightning_fabric
    component_builder = ModuleType("protomotions.utils.component_builder")
    component_builder.build_all_components = fake_build_all_components
    simulator_utils = ModuleType("protomotions.simulator.base_simulator.utils")
    simulator_utils.convert_friction_for_simulator = (
        lambda terrain, simulator: (terrain, simulator)
    )
    base_env = ModuleType("protomotions.envs.base_env.env")
    base_env.BaseEnv = object
    omni = ModuleType("omni")
    omni.__path__ = []
    omni_log_module = ModuleType("omni.log")
    omni_log_module.SettingBehavior = SimpleNamespace(OVERRIDE="override")
    omni_log_module.get_log = lambda: SimpleNamespace(
        set_channel_enabled=lambda *items: calls.append(("omni_log", items))
    )
    omni.log = omni_log_module

    monkeypatch.setitem(main_globals, "args", args)
    monkeypatch.setitem(main_globals, "AppLauncher", FakeAppLauncher)
    monkeypatch.setitem(
        main_globals,
        "load_experiment_module",
        lambda path: pytest.fail("resume should not load experiment module"),
    )
    monkeypatch.setitem(
        main_globals,
        "get_class",
        lambda target: FakeEnv if target == "env.Target" else FakeAgent,
    )
    monkeypatch.setitem(
        main_globals,
        "save_configs",
        lambda *items, **kwargs: pytest.fail("resume should not save configs"),
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.fabric_config",
        fabric_config_module,
    )
    monkeypatch.setitem(sys.modules, "lightning.fabric", lightning_fabric)
    monkeypatch.setitem(
        sys.modules,
        "protomotions.utils.component_builder",
        component_builder,
    )
    monkeypatch.setitem(
        sys.modules,
        "protomotions.simulator.base_simulator.utils",
        simulator_utils,
    )
    monkeypatch.setitem(sys.modules, "protomotions.envs.base_env.env", base_env)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.log", omni_log_module)

    module["main"]()

    assert (
        "agent_load",
        Path("results") / "resume" / "last.ckpt",
        True,
    ) in calls
    assert ("agent_fit", True) in calls
    assert ("app_launcher", {"headless": False, "device": "cpu", "distributed": True}) in calls
    assert os.environ["LOCAL_RANK"] == "1"
    assert os.environ["RANK"] == "3"
    build_call = next(call for call in calls if call[0] == "build_components")
    assert build_call[1]["simulation_app"] == "simulation-app"
    assert build_call[1]["save_dir"] == "resume_weights"
    assert len([call for call in calls if call[0] == "omni_log"]) == 2
