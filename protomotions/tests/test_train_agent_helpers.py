# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the training entry point.

The shim ``protomotions/train_agent.py`` is import-safe; the build sequence and
run body live in ``protomotions.launch.training`` and ``config_builder``, so
these tests import those functions directly instead of running the script.
"""

import argparse
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import protomotions.launch.training as launch_training
from protomotions.components.motion_lib import MotionFileSwitchMode
from protomotions.launch.training import (
    build_wandb_logger_config,
    detect_checkpoint_mode,
    load_motion_shard_cycle,
    resolve_resume_motion_shard_cycle,
    run,
    save_configs,
    save_training_configs,
    try_log_hyperparams_to_wandb,
)
from protomotions.launch.training_cli import create_training_parser
from protomotions.utils.cli_utils import parse_bool
from protomotions.utils.config_builder import load_experiment_module
from protomotions.utils.experiment_configs import (
    BUNDLE_FILENAME,
    ExperimentConfigs,
    load_experiment_configs,
)


@dataclass
class _TinyConfig:
    _target_: str = "unit.Target"
    value: int = 1
    # The env root's field build_env_from_configs reads for the curriculum weights.
    save_dir: str = ""

    def validate(self):
        self.validate_calls = getattr(self, "validate_calls", 0) + 1

    def validate_anchor_body(self):
        pass


def _bundle(**overrides):
    """An ExperimentConfigs whose seven roots are a shared tiny config."""
    config = overrides.pop("config", _TinyConfig(value=5))
    fields = dict(
        robot=config,
        simulator=config,
        terrain=config,
        scene_lib=config,
        motion_lib=config,
        env=config,
        agent=config,
    )
    fields.update(overrides)
    return ExperimentConfigs(**fields)


class _FakeFabricConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def as_loggable_dict(self):
        return {"devices": 1, "num_nodes": 1}

    def as_kwargs(self):
        return {}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_training_parser_and_bool_helpers():
    """create_training_parser declares the training flags and parse_bool parses."""
    assert parse_bool(True) is True
    assert parse_bool("yes") is True
    assert parse_bool("0") is False
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("maybe")

    parser = create_training_parser()
    required = [
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
    ]
    parsed = parser.parse_args(
        required
        + ["--headless", "false", "--torch-deterministic", "--create-config-only"]
    )
    assert parsed.headless is False
    assert parsed.torch_deterministic is True
    assert parsed.wandb_project == "physical_animation"
    assert parsed.create_config_only is True
    assert parsed.training_max_iterations is None

    assert (
        parser.parse_args(required + ["--wandb-project", "custom"]).wandb_project
        == "custom"
    )
    assert (
        parser.parse_args(
            required + ["--training-max-iterations", "30"]
        ).training_max_iterations
        == 30
    )

    with pytest.raises(SystemExit):
        parser.parse_args(
            required
            + ["--training-max-steps", "100", "--training-max-iterations", "10"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(required + ["--training-max-iterations", "0"])


# ---------------------------------------------------------------------------
# Motion shard cycle
# ---------------------------------------------------------------------------


def test_load_motion_shard_cycle_reads_required_checkpoint_state(tmp_path):
    """The shard cycle is read from the checkpoint, and a missing key raises."""
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"motion_shard_cycle": 4}, checkpoint)
    assert load_motion_shard_cycle(checkpoint) == 4

    torch.save({}, checkpoint)
    with pytest.raises(KeyError, match="motion_shard_cycle"):
        load_motion_shard_cycle(checkpoint)


@pytest.mark.parametrize("bad_cycle", [1.5, True, "4"])
def test_load_motion_shard_cycle_rejects_non_plain_integer_values(tmp_path, bad_cycle):
    """A non-int shard cycle is rejected."""
    checkpoint = tmp_path / "last.ckpt"
    torch.save({"motion_shard_cycle": bad_cycle}, checkpoint)
    with pytest.raises(TypeError, match="plain Python int"):
        load_motion_shard_cycle(checkpoint)


def test_resume_motion_shard_cycle_fails_all_ranks_before_broadcast(
    monkeypatch, tmp_path
):
    """A rank-0 load failure is raised across all ranks before any broadcast."""
    calls = []
    fabric = SimpleNamespace(
        global_rank=0,
        device=torch.device("cpu"),
        broadcast=lambda value: calls.append(("broadcast", value)),
    )
    monkeypatch.setattr(
        launch_training,
        "load_motion_shard_cycle",
        lambda checkpoint: (_ for _ in ()).throw(OSError("checkpoint unavailable")),
    )
    with pytest.raises(
        RuntimeError, match="Motion shard cycle checkpoint load"
    ) as error:
        resolve_resume_motion_shard_cycle(
            fabric,
            tmp_path / "last.ckpt",
            SimpleNamespace(motion_file_switch_mode=MotionFileSwitchMode.LIVE),
        )
    assert isinstance(error.value.__cause__, OSError)
    assert calls == []


def test_fixed_resume_motion_shard_cycle_bypasses_checkpoint_and_broadcast(
    monkeypatch, tmp_path
):
    """FIXED mode returns 0 without reading the checkpoint or broadcasting."""
    fabric = SimpleNamespace(
        global_rank=0,
        device=torch.device("cpu"),
        broadcast=lambda value: pytest.fail("fixed mode must not broadcast"),
    )
    monkeypatch.setattr(
        launch_training,
        "load_motion_shard_cycle",
        lambda checkpoint: pytest.fail("fixed mode must not load checkpoint"),
    )
    assert (
        resolve_resume_motion_shard_cycle(
            fabric,
            tmp_path / "ckpt.ckpt",
            SimpleNamespace(motion_file_switch_mode=MotionFileSwitchMode.FIXED),
        )
        == 0
    )


def test_healthy_resume_rank_fails_before_broadcast_when_remote_load_fails(
    monkeypatch, tmp_path
):
    """A non-zero rank fails via the collective check, never loading or broadcasting."""
    from protomotions.agents.utils import distributed

    calls = []
    fabric = SimpleNamespace(
        global_rank=1,
        device=torch.device("cpu"),
        broadcast=lambda value: calls.append(("broadcast", value)),
    )
    monkeypatch.setattr(
        launch_training,
        "load_motion_shard_cycle",
        lambda checkpoint: pytest.fail("healthy rank must not load checkpoint"),
    )
    monkeypatch.setattr(
        distributed,
        "raise_if_any_rank_failed",
        lambda error, operation, device: (_ for _ in ()).throw(
            RuntimeError("Motion shard cycle checkpoint load failed on another rank")
        ),
    )
    with pytest.raises(RuntimeError, match="another rank"):
        resolve_resume_motion_shard_cycle(
            fabric,
            tmp_path / "last.ckpt",
            SimpleNamespace(motion_file_switch_mode=MotionFileSwitchMode.LIVE),
        )
    assert calls == []


# ---------------------------------------------------------------------------
# Checkpoint mode, wandb logger, experiment module
# ---------------------------------------------------------------------------


def test_build_wandb_logger_config_uses_cli_project(tmp_path):
    """The wandb logger config carries the CLI name, project and wandb id."""
    config = build_wandb_logger_config(
        SimpleNamespace(experiment_name="unit-run", wandb_project="custom-project"),
        tmp_path,
        "wandb-id",
    )
    assert config["name"] == "unit-run"
    assert config["project"] == "custom-project"
    assert config["save_dir"] == tmp_path
    assert config["id"] == "wandb-id"


def test_detect_checkpoint_mode_handles_fresh_warm_start_and_resume(tmp_path):
    """detect_checkpoint_mode classifies fresh, warm start and resume runs."""
    assert detect_checkpoint_mode(
        SimpleNamespace(checkpoint=None), tmp_path / "fresh"
    ) == (
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
        json.dumps({"robot_name": "smpl", "batch_size": 64, "wandb_id": "wandb-resume"})
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


def test_resume_keeps_kit_thread_count_given_on_the_command_line(tmp_path):
    """A command-line --kit-thread-count wins over the saved value on resume."""
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    (resume_dir / "last.ckpt").write_text("checkpoint")

    def resume(saved, cli):
        (resume_dir / "config.yaml").write_text(
            json.dumps({"batch_size": 64, "kit_thread_count": saved})
        )
        args = SimpleNamespace(checkpoint=None, batch_size=8, kit_thread_count=cli)
        assert detect_checkpoint_mode(args, resume_dir)[0] == "resume"
        assert args.batch_size == 64
        return args.kit_thread_count

    assert resume(saved=None, cli=1) == 1
    assert resume(saved=4, cli=1) == 1
    assert resume(saved=1, cli=None) == 1
    assert resume(saved=None, cli=None) is None


def test_load_experiment_module_loads_file_and_rejects_missing_paths(tmp_path):
    """load_experiment_module imports a file and names a missing path."""
    experiment = tmp_path / "custom_experiment.py"
    experiment.write_text("VALUE = 7\n")
    assert load_experiment_module(experiment).VALUE == 7
    with pytest.raises(FileNotFoundError, match="Experiment file not found"):
        load_experiment_module(tmp_path / "missing.py")


def test_load_experiment_module_prefers_the_running_tree_over_the_recipe_root(
    monkeypatch, tmp_path
):
    """A recipe copy inside another checkout imports the running tree's recipes.

    The copy's own project root is appended after the running tree, so it only
    serves what the running tree lacks (here a top-level helper module).
    """
    import protomotions

    running_root = Path(protomotions.__file__).resolve().parents[1]
    foreign = tmp_path / "other_checkout"
    for package in ("examples", "examples/experiments", "examples/experiments/mimic"):
        (foreign / package).mkdir(parents=True, exist_ok=True)
        (foreign / package / "__init__.py").write_text("")
    (foreign / "examples/experiments/mimic/mlp.py").write_text("SENTINEL = 1\n")
    (foreign / "recipe_helpers_only_here.py").write_text("HELPER = 'recipe root'\n")
    run_dir = foreign / "results" / "run"
    run_dir.mkdir(parents=True)
    copy = run_dir / "experiment_config.py"
    copy.write_text(
        "from examples.experiments.mimic import mlp\n"
        "import recipe_helpers_only_here\n"
        "MLP_FILE = mlp.__file__\n"
        "HELPER = recipe_helpers_only_here.HELPER\n"
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "recipe_helpers_only_here", raising=False)

    module = load_experiment_module(copy)

    assert str(Path(module.MLP_FILE).resolve()).startswith(str(running_root))
    assert module.HELPER == "recipe root"
    assert sys.path[0] != str(foreign)
    assert sys.path.index(str(foreign)) > sys.path.index(str(running_root))


# ---------------------------------------------------------------------------
# Saving the bundle
# ---------------------------------------------------------------------------


def test_save_configs_writes_config_yaml_the_bundle_and_experiment_copy(
    monkeypatch, tmp_path
):
    """save_configs writes config.yaml, the typed JSON bundle and an experiment copy."""
    wandb = SimpleNamespace(run=SimpleNamespace(id="wandb-unit"))
    monkeypatch.setattr(launch_training, "wandb", wandb)
    save_dir = tmp_path / "results"
    experiment = tmp_path / "experiment.py"
    experiment.write_text("# experiment\n")
    args = SimpleNamespace(use_wandb=True, experiment_name="unit", seed=123)

    save_configs(save_dir, args, _bundle(), experiment_source_path=experiment)

    checkpoint_config = json.loads((save_dir / "config.yaml").read_text())
    assert checkpoint_config["wandb_id"] == "wandb-unit"
    assert checkpoint_config["seed"] == 123
    assert (save_dir / BUNDLE_FILENAME).exists()
    assert (save_dir / "experiment_config.py").read_text() == "# experiment\n"

    # The bundle is a tagged JSON tree rebuilt through the real constructor.
    loaded = load_experiment_configs(save_dir / BUNDLE_FILENAME)
    assert isinstance(loaded, ExperimentConfigs)
    assert loaded.robot.value == 5


def test_save_configs_tolerates_a_missing_wandb_id(monkeypatch, tmp_path):
    """A missing wandb id leaves wandb_id None and still writes the bundle."""
    monkeypatch.setattr(launch_training, "wandb", SimpleNamespace(run=object()))
    save_dir = tmp_path / "results"
    experiment = tmp_path / "experiment.py"
    experiment.write_text("# experiment\n")
    args = SimpleNamespace(use_wandb=True, experiment_name="unit")

    save_configs(
        save_dir,
        args,
        _bundle(),
        experiment_source_path=experiment,
        bundle_filename="alt_bundle.json",
    )

    checkpoint_config = json.loads((save_dir / "config.yaml").read_text())
    assert checkpoint_config["wandb_id"] is None
    assert (save_dir / "alt_bundle.json").exists()


def test_try_log_hyperparams_to_wandb_logs_only_matching_logger(monkeypatch):
    """Only a WandbLogger is logged to, with one entry per config root plus fabric."""

    class FakeWandbLogger:
        def __init__(self):
            self.logged = None

        def log_hyperparams(self, params):
            self.logged = params

    monkeypatch.setattr(launch_training, "WandbLogger", FakeWandbLogger)
    logger = FakeWandbLogger()
    fabric = SimpleNamespace(loggers=[object(), logger])

    try_log_hyperparams_to_wandb(
        fabric, _bundle(config=_TinyConfig(value=9)), _FakeFabricConfig()
    )

    assert logger.logged["robot"]["value"] in (9, "9")
    assert logger.logged["fabric"] in (
        {"devices": 1, "num_nodes": 1},
        {"devices": "1", "num_nodes": "1"},
    )


def test_try_log_hyperparams_to_wandb_tolerates_logger_failures(monkeypatch):
    """A logger that raises during logging does not propagate."""

    class FailingWandbLogger:
        def log_hyperparams(self, params):
            raise RuntimeError("wandb is offline")

    monkeypatch.setattr(launch_training, "WandbLogger", FailingWandbLogger)
    try_log_hyperparams_to_wandb(
        SimpleNamespace(loggers=[FailingWandbLogger()]),
        _bundle(config=_TinyConfig(value=9)),
        _FakeFabricConfig(),
    )


def test_save_training_configs_saves_one_bundle_and_dry_runs_inference(
    monkeypatch, tmp_path
):
    """Only the training bundle is written, then the inference derivation is dry-run."""
    saved = []

    def fake_save_configs(
        save_dir, args, configs, experiment_source_path, bundle_filename=BUNDLE_FILENAME
    ):
        saved.append((bundle_filename, configs.robot.value))

    monkeypatch.setattr(launch_training, "save_configs", fake_save_configs)

    dry_runs = []
    inference_utils = SimpleNamespace(
        derive_inference_configs=lambda training,
        args,
        overrides,
        hooks: dry_runs.append((training, overrides, hooks)),
        recipe_hooks=lambda checkpoint_dir, inference_config: ["hook"],
    )
    monkeypatch.setitem(
        sys.modules, "protomotions.utils.inference_utils", inference_utils
    )

    configs = _bundle()
    save_training_configs(
        tmp_path,
        SimpleNamespace(experiment_name="unit"),
        configs,
        experiment_source_path="experiment.py",
    )

    assert [name for name, _ in saved] == [
        BUNDLE_FILENAME
    ]  # one bundle, no inference copy
    assert len(dry_runs) == 1  # the derivation ran once as a dry-run
    training, overrides, hooks = dry_runs[0]
    assert training is configs and overrides == {} and hooks == ["hook"]
    assert configs.robot.value == 5  # the training bundle is untouched


# ---------------------------------------------------------------------------
# The run() flow
# ---------------------------------------------------------------------------


class _FakeFabric:
    def __init__(self, calls):
        self.calls = calls
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")
        self.strategy = SimpleNamespace(barrier=lambda: calls.append(("barrier",)))
        self.loggers = []

    def launch(self):
        self.calls.append(("launch",))

    def seed_everything(self, seed):
        self.calls.append(("seed_everything", seed))

    def call(self, hook, *args, **kwargs):
        self.calls.append(("fabric_call", hook))

    def broadcast(self, value):
        return value


def _install_run_fakes(monkeypatch, calls, configs):
    """Replace run()'s heavy collaborators with recording fakes.

    run() constructs the env through ``build_env_from_configs`` and the agent
    through ``get_class(config._target_)`` (the imported ``BaseEnv``/``BaseAgent``
    are only type hints), so the fakes are wired through ``get_class`` where
    each module binds it: ``component_builder`` for the env, ``hydra_replacement``
    for the agent.
    """
    # Import the env/agent modules now so the ones that bind get_class at module
    # scope (e.g. base_env/reset/motion.py) hold the real get_class, not the fake
    # below -- otherwise a later test that builds a real env would get our fake.
    import protomotions.envs.base_env.env  # noqa: F401
    import protomotions.agents.base_agent.agent  # noqa: F401

    fabric = _FakeFabric(calls)
    agents = []
    monkeypatch.setattr(
        "protomotions.utils.fabric_config.FabricConfig", _FakeFabricConfig
    )
    monkeypatch.setattr("lightning.fabric.Fabric", lambda **kwargs: fabric)
    monkeypatch.setattr(
        "protomotions.utils.torch_utils.seeding",
        lambda seed, torch_deterministic=False: calls.append(
            ("seeding", seed, torch_deterministic)
        ),
    )
    monkeypatch.setattr(
        "protomotions.utils.component_builder.convert_friction_for_simulator",
        lambda terrain, simulator: (terrain, simulator),
    )
    monkeypatch.setattr(
        "protomotions.utils.component_builder.build_all_components",
        lambda **kwargs: {
            "terrain": "terrain",
            "scene_lib": "scene",
            "motion_lib": "motion",
            "simulator": "simulator",
        },
    )

    class _FakeEnv:
        def __init__(self, **kwargs):
            self.simulator = SimpleNamespace()

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.fabric = fabric
            agents.append(self)

        def setup(self):
            calls.append(("agent_setup",))

        def load(self, checkpoint, load_training_state=True):
            calls.append(("agent_load", checkpoint, load_training_state))

        def fit(self):
            calls.append(("agent_fit",))

    def fake_get_class(target):
        return _FakeEnv if "Env" in target else _FakeAgent

    monkeypatch.setattr(
        "protomotions.utils.component_builder.get_class", fake_get_class
    )
    monkeypatch.setattr(
        "protomotions.utils.hydra_replacement.get_class", fake_get_class
    )
    monkeypatch.setattr(
        "protomotions.envs.context_check.run_config_binding_check",
        lambda *a, **k: calls.append(("binding_check",)),
    )
    monkeypatch.setattr(
        launch_training, "build_configs", lambda module, args, overrides: configs
    )
    monkeypatch.setattr(
        launch_training,
        "load_experiment_module",
        lambda path: SimpleNamespace(__file__=str(path)),
    )
    monkeypatch.setattr(
        launch_training,
        "save_training_configs",
        lambda *a, **k: calls.append(("save",)),
    )
    monkeypatch.setattr(
        launch_training,
        "try_log_hyperparams_to_wandb",
        lambda *a, **k: calls.append(("hparams",)),
    )
    return fabric, agents


def _flow_bundle(agent=...):
    """A bundle whose env and agent carry distinct _target_s for get_class."""
    env = _TinyConfig(value=1, _target_="fake.EnvClass")
    if agent is ...:
        agent = _TinyConfig(value=7, _target_="fake.AgentClass")
    return _bundle(env=env, agent=agent)


def _fresh_argv(experiment, name="fresh-run"):
    return [
        "--robot-name",
        "g1",
        "--simulator",
        "newton",
        "--num-envs",
        "4",
        "--batch-size",
        "8",
        "--motion-file",
        "motions.pt",
        "--experiment-path",
        str(experiment),
        "--experiment-name",
        name,
    ]


def test_run_fresh_training_wires_fabric_components_agent_and_saves(
    monkeypatch, tmp_path
):
    """A fresh run builds configs, launches fabric, constructs env+agent, saves and fits."""
    monkeypatch.chdir(tmp_path)
    calls = []
    configs = _flow_bundle()
    _install_run_fakes(monkeypatch, calls, configs)
    experiment = tmp_path / "experiment.py"
    experiment.write_text("VALUE = 1\n")

    run(_fresh_argv(experiment), app_launcher=None)

    assert ("launch",) in calls
    assert ("seed_everything", 0) in calls
    assert ("seeding", 0, False) in calls
    assert ("fabric_call", "on_app_start") in calls
    assert ("fabric_call", "on_env_init_start") in calls
    assert ("fabric_call", "on_env_init_end") in calls
    assert ("agent_setup",) in calls
    assert (
        "agent_load",
        None,
        False,
    ) in calls  # fresh: no checkpoint, no training state
    assert ("agent_fit",) in calls
    assert ("save",) in calls


def test_run_create_config_only_saves_and_exits_before_training(monkeypatch, tmp_path):
    """--create-config-only saves both bundles and returns before building fabric."""
    monkeypatch.chdir(tmp_path)
    calls = []
    configs = _flow_bundle()
    _install_run_fakes(monkeypatch, calls, configs)
    handled = []
    monkeypatch.setattr(
        launch_training,
        "_handle_create_config_only",
        lambda args, save_dir, src, cfgs: handled.append((save_dir, cfgs)),
    )
    experiment = tmp_path / "experiment.py"
    experiment.write_text("VALUE = 1\n")

    run(
        _fresh_argv(experiment, "cfg-only") + ["--create-config-only"],
        app_launcher=None,
    )

    assert len(handled) == 1
    assert handled[0][0] == Path("results") / "cfg-only"
    assert handled[0][1] is configs
    assert ("launch",) not in calls  # exited before fabric
    assert ("agent_fit",) not in calls


def test_run_requires_an_agent_config(monkeypatch, tmp_path):
    """Training refuses a recipe with no agent_config()."""
    monkeypatch.chdir(tmp_path)
    calls = []
    _install_run_fakes(monkeypatch, calls, _flow_bundle(agent=None))
    experiment = tmp_path / "experiment.py"
    experiment.write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="defines no agent_config"):
        run(_fresh_argv(experiment, "no-agent"), app_launcher=None)


def test_run_resume_uses_saved_bundle_and_sets_skip_flag(monkeypatch, tmp_path):
    """Resume loads the saved bundle, loads training state and skips the next update."""
    monkeypatch.chdir(tmp_path)
    calls = []
    configs = _flow_bundle()
    # Resume reads the switch mode and validates the motion lib from the bundle.
    configs.motion_lib = SimpleNamespace(
        motion_file_switch_mode=MotionFileSwitchMode.FIXED, validate=lambda: None
    )
    _, agents = _install_run_fakes(monkeypatch, calls, configs)
    # build_configs must not run on resume: the bundle is loaded from disk.
    monkeypatch.setattr(
        launch_training,
        "build_configs",
        lambda *a, **k: pytest.fail("resume must not rebuild configs"),
    )
    monkeypatch.setattr(
        launch_training, "load_experiment_configs", lambda path: configs
    )

    save_dir = tmp_path / "results" / "resume-run"
    save_dir.mkdir(parents=True)
    (save_dir / "last.ckpt").write_text("checkpoint")
    (save_dir / BUNDLE_FILENAME).write_text("bundle")  # existence is checked
    (save_dir / "config.yaml").write_text(
        json.dumps({"batch_size": 8, "wandb_id": None})
    )

    run(_fresh_argv(tmp_path / "experiment.py", "resume-run"), app_launcher=None)

    load_calls = [call for call in calls if call[0] == "agent_load"]
    assert load_calls and load_calls[0][2] is True  # resume loads training state
    assert load_calls[0][1].name == "last.ckpt"  # from the saved run, not --checkpoint
    assert ("save",) not in calls  # resume does not re-save configs
    assert ("agent_fit",) in calls
    assert agents[0]._skip_next_policy_update is True


def test_run_warm_start_saves_configs_with_a_json_serializable_checkpoint(
    monkeypatch, tmp_path
):
    """Warm start keeps args.checkpoint a string so save_configs' json.dump works.

    Regression: assigning the Path from detect_checkpoint_mode left
    args.checkpoint a PosixPath, and save_configs' json.dump(vars(args)) then
    raised on the first run, after the env and agent were already built. The
    other run() tests fake the save, so warm start needs the real save path.
    """
    monkeypatch.chdir(tmp_path)
    calls = []
    configs = _flow_bundle()
    _install_run_fakes(monkeypatch, calls, configs)
    # Exercise the real save (that is where the crash was); only wandb and the
    # inference-derivation dry-run are faked (the dry-run would otherwise need a
    # real recipe and real config classes, not the tiny stand-ins).
    monkeypatch.setattr(
        launch_training,
        "save_training_configs",
        save_training_configs,
    )
    monkeypatch.setattr(
        launch_training, "wandb", SimpleNamespace(run=SimpleNamespace(id="warm"))
    )
    inference_utils = SimpleNamespace(
        derive_inference_configs=lambda *a, **k: None,
        recipe_hooks=lambda *a, **k: [],
    )
    monkeypatch.setitem(
        sys.modules, "protomotions.utils.inference_utils", inference_utils
    )

    experiment = tmp_path / "experiment.py"
    experiment.write_text("VALUE = 1\n")
    checkpoint = tmp_path / "pretrained" / "weights.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("weights")

    run(
        _fresh_argv(experiment, "warm-run") + ["--checkpoint", str(checkpoint)],
        app_launcher=None,
    )

    # The string path reached agent.load unchanged, without training state.
    load_calls = [call for call in calls if call[0] == "agent_load"]
    assert load_calls and load_calls[0][1] == str(checkpoint)
    assert load_calls[0][2] is False

    # The real save ran and config.yaml holds the checkpoint as a string.
    save_dir = tmp_path / "results" / "warm-run"
    saved_config = json.loads((save_dir / "config.yaml").read_text())
    assert saved_config["checkpoint"] == str(checkpoint)
    assert isinstance(saved_config["checkpoint"], str)
    assert (save_dir / BUNDLE_FILENAME).exists()
    assert ("agent_fit",) in calls


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_importing_the_shims_pulls_no_heavy_dependency():
    """Importing the train/inference shims imports neither torch nor wandb."""
    script = textwrap.dedent(
        """
        import sys
        import protomotions.train_agent
        import protomotions.inference_agent
        leaked = [m for m in ("torch", "wandb", "lightning") if m in sys.modules]
        assert not leaked, leaked
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
