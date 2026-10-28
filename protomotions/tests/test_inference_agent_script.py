# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the inference entry point.

The shim ``protomotions/inference_agent.py`` is import-safe; the run body lives
in ``protomotions.launch.inference``, so these tests import ``run`` and
``create_inference_parser`` directly and fake the runtime collaborators.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from protomotions.components.motion_lib import MotionLibConfig
from protomotions.launch.inference import create_inference_parser, run
from protomotions.utils.experiment_configs import ExperimentConfigs

MASKED_MIMIC_INFERENCE_CONFIG = str(
    Path(__file__).resolve().parents[2] / "examples/inference/masked_mimic.py"
)
TARGET_KEYBOARD_INFERENCE_CONFIG = str(
    Path(__file__).resolve().parents[2] / "examples/inference/target_keyboard.py"
)


class _FakeFabricConfig:
    def __init__(
        self, accelerator="gpu", devices=1, num_nodes=1, loggers=None, callbacks=None
    ):
        self.accelerator = accelerator
        self.devices = devices

    def as_kwargs(self):
        return {"accelerator": self.accelerator, "devices": self.devices}


class _FakeFabric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = torch.device("cpu")

    def launch(self):
        pass


def _noop_defaults():
    """A picklable no-op apply_inference_defaults for the env/simulator stand-ins."""


def _noop_agent_defaults(env):
    """A picklable no-op apply_inference_defaults(env) for the agent stand-in."""


def _bundle(**overrides):
    """An inference bundle: an ExperimentConfigs of tiny stand-in configs.

    The stand-in configs carry no-op apply_inference_defaults so the real
    derivation can run its framework-defaults step over them (the defaults
    themselves are exercised by test_inference_derivation); they are
    module-level functions so the bundle still pickles.
    """
    fields = dict(
        robot=SimpleNamespace(name="robot"),
        simulator=SimpleNamespace(
            _target_="protomotions.simulator.isaacgym.simulator.IsaacGymSimulator",
            num_envs=1,
            headless=False,
            sim=SimpleNamespace(fps=60, decimation=2),
            domain_randomization=None,
            apply_inference_defaults=_noop_defaults,
        ),
        terrain=SimpleNamespace(name="terrain"),
        scene_lib=SimpleNamespace(scene_file="old_scene.yaml", asset_root="/old/root"),
        motion_lib=MotionLibConfig(motion_file="old.motion"),
        env=SimpleNamespace(
            _target_="env.Target",
            save_dir="weights",
            sensors=None,
            apply_inference_defaults=_noop_defaults,
        ),
        agent=SimpleNamespace(
            _target_="agent.Target", apply_inference_defaults=_noop_agent_defaults
        ),
    )
    fields.update(overrides)
    return ExperimentConfigs(**fields)


def _write_bundle(monkeypatch, run_dir, bundle):
    """Make run()'s bundle loader return ``bundle`` for a checkpoint under run_dir.

    The saved form rebuilds every root through its real constructor, which the
    tiny SimpleNamespace stand-ins here can't satisfy, so there is no JSON file
    to write: patch the loader to hand back the in-memory bundle instead. The
    "no bundle beside the checkpoint" test does not call this, so it still
    exercises the real loader's missing-file error.
    """
    import protomotions.launch.inference as inference

    run_dir = Path(run_dir)

    def _load(checkpoint):
        if Path(checkpoint).parent != run_dir:
            raise FileNotFoundError(f"resolved configs not found: {checkpoint}")
        return bundle

    monkeypatch.setattr(inference, "load_experiment_configs_from_checkpoint", _load)
    # recipe_hooks() loads the recipe from here; a no-hook recipe is enough.
    (run_dir / "experiment_config.py").write_text("# inference test recipe (no hook)\n")


def _argv(checkpoint, simulator="newton", extra_args=None):
    argv = ["--checkpoint", str(checkpoint), "--simulator", simulator]
    if extra_args:
        argv.extend(extra_args)
    return argv


def _install_runtime(monkeypatch, captured):
    """Fake the runtime collaborators ``run()`` imports locally.

    ``run()`` constructs the env through ``build_env_from_configs`` and the
    agent through ``get_class(config._target_)``; the fakes map ``env.Target``
    to the env and ``agent.Target`` to the agent, through ``get_class`` where
    each module binds it (``component_builder`` and ``hydra_replacement``).
    """
    # Import the env/agent modules now so the ones that bind get_class at module
    # scope (e.g. base_env/reset/motion.py) hold the real get_class, not the fake
    # below -- otherwise a later test that builds a real env would get our fake.
    import protomotions.envs.base_env.env  # noqa: F401
    import protomotions.agents.base_agent.agent  # noqa: F401

    monkeypatch.setattr(
        "protomotions.utils.fabric_config.FabricConfig", _FakeFabricConfig
    )
    monkeypatch.setattr(
        "lightning.fabric.Fabric", lambda **kwargs: _FakeFabric(**kwargs)
    )
    monkeypatch.setattr(
        "protomotions.utils.component_builder.convert_friction_for_simulator",
        lambda terrain, simulator: (terrain, simulator),
    )
    monkeypatch.setattr(
        "protomotions.envs.context_check.run_config_binding_check",
        lambda *a, **k: captured.setdefault("binding_check", []).append(a),
    )

    class _FakeEvaluator:
        def __init__(self):
            self.eval_count = 5

        def simple_test_policy(self):
            captured["simple_test"] = True

    class _FakeEnv:
        def __init__(self, **kwargs):
            captured["env_config"] = kwargs["config"]
            self.simulator = kwargs["simulator"]

    class _FakeAgent:
        def __init__(self, config, env, fabric, **kwargs):
            captured["agent_config"] = config
            self.evaluator = captured.get("evaluator") or _FakeEvaluator()

        def setup(self, *, training=True):
            assert not training

        def load(
            self,
            checkpoint,
            load_env=True,
            load_training_state=True,
            *,
            for_inference=False,
        ):
            assert for_inference is True
            captured["load"] = (checkpoint, load_env, load_training_state)

    def fake_build_all_components(**kwargs):
        captured["build_kwargs"] = kwargs
        return {
            "terrain": None,
            "scene_lib": None,
            "motion_lib": None,
            "simulator": captured.get("simulator") or SimpleNamespace(),
        }

    monkeypatch.setattr(
        "protomotions.utils.component_builder.build_all_components",
        fake_build_all_components,
    )

    def fake_get_class(target):
        return _FakeEnv if target == "env.Target" else _FakeAgent

    monkeypatch.setattr(
        "protomotions.utils.component_builder.get_class", fake_get_class
    )
    monkeypatch.setattr(
        "protomotions.utils.hydra_replacement.get_class", fake_get_class
    )


# ---------------------------------------------------------------------------
# Command line and inference-config helpers
# ---------------------------------------------------------------------------


def test_inference_parser_requires_checkpoint_and_parses_options():
    """create_inference_parser requires --checkpoint and parses the options."""
    parser = create_inference_parser()
    parsed = parser.parse_args(
        [
            "--checkpoint",
            "ckpt",
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
            "--overrides",
            "env.max_episode_length=5",
        ]
    )
    assert parsed.checkpoint == "ckpt"
    assert parsed.simulator == "mujoco"
    assert parsed.full_eval is True
    assert parsed.headless is True
    assert parsed.num_envs == 4
    assert parsed.motion_file == "motion.pt"
    assert parsed.scenes_file == "scene.yaml"
    assert parsed.overrides == ["env.max_episode_length=5"]

    with pytest.raises(SystemExit):
        parser.parse_args(["--simulator", "mujoco"])  # --checkpoint is required


@pytest.mark.parametrize(
    "source, error, message",
    [
        ("value = 1", ValueError, "must define apply_inference_overrides"),
        (
            "def apply_inference_overrides(configs, args):\n    raise RuntimeError('invalid demo')",
            RuntimeError,
            "invalid demo",
        ),
        (
            "raise ImportError('missing demo dependency')",
            ImportError,
            "missing demo dependency",
        ),
    ],
)
def test_inference_config_errors_are_not_swallowed(
    monkeypatch, tmp_path, source, error, message
):
    """A broken --inference-config hook's error surfaces from run() unchanged.

    The missing-hook and import errors come from loading it (recipe_hooks); a
    hook that raises in its body surfaces when the derivation calls it.
    """
    _write_bundle(monkeypatch, tmp_path, _bundle())
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    hook = tmp_path / "demo.py"
    hook.write_text(source)
    _install_runtime(monkeypatch, {})

    with pytest.raises(error, match=message):
        run(
            _argv(checkpoint, "isaacgym", ["--inference-config", str(hook)]),
            app_launcher=None,
        )


def test_inference_config_requires_an_existing_file(monkeypatch, tmp_path):
    """A missing --inference-config path is named, not swallowed."""
    _write_bundle(monkeypatch, tmp_path, _bundle())
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    _install_runtime(monkeypatch, {})

    with pytest.raises(FileNotFoundError, match="Experiment file not found"):
        run(
            _argv(
                checkpoint, "isaacgym", ["--inference-config", str(tmp_path / "x.py")]
            ),
            app_launcher=None,
        )


# ---------------------------------------------------------------------------
# The run() flow
# ---------------------------------------------------------------------------


def test_run_fails_loud_when_no_bundle_sits_beside_the_checkpoint(tmp_path):
    """Inference fails loud, naming both candidates, when no bundle is present."""
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    with pytest.raises(FileNotFoundError, match="resolved configs not found"):
        run(_argv(checkpoint, "isaacgym"), app_launcher=None)


def test_run_loads_the_one_training_bundle(monkeypatch, tmp_path):
    """Inference derives its view from the one (training) bundle beside the checkpoint."""
    _write_bundle(monkeypatch, tmp_path, _bundle())
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(_argv(checkpoint, "isaacgym"), app_launcher=None)

    assert captured["env_config"]._target_ == "env.Target"  # reached construction


def test_run_inference_config_is_policy_agnostic_and_cli_takes_precedence(
    monkeypatch, tmp_path
):
    """An --inference-config hook runs, and --overrides and scalar flags win over it."""
    _write_bundle(monkeypatch, tmp_path, _bundle())
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    hook = tmp_path / "demo.py"
    # The hook touches a field no override touches; a hook that undid an override
    # would be refused (see test_inference_derivation).
    hook.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Settings:\n    enabled: bool = True\n"
        "def apply_inference_overrides(configs, args):\n"
        "    configs.agent.demo_enabled = Settings().enabled\n"
    )
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(
        _argv(
            checkpoint,
            "isaacgym",
            [
                "--inference-config",
                str(hook),
                "--num-envs",
                "2",
                "--overrides",
                "env.save_dir=cli",
            ],
        ),
        app_launcher=None,
    )

    assert captured["env_config"].save_dir == "cli"  # --overrides applied
    assert (
        captured["build_kwargs"]["simulator_config"].num_envs == 2
    )  # --num-envs applied
    assert captured["agent_config"].demo_enabled is True  # the hook ran


def _anchor_mask_passthrough(mask):
    return mask


def test_run_native_task_cli_replaces_control_before_general_overrides(
    monkeypatch, tmp_path
):
    """A masked-mimic inference config swaps the control class and configures its task.

    B2 applies --overrides before the hooks, so a task that exists only on the
    swapped control is configured inside the hook, not by a later override.
    """
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
    from protomotions.envs.control.masked_mimic_tasks.controller import (
        MaskedMimicTaskControlConfig,
    )
    from protomotions.envs.mdp_component import MdpComponent

    bundle = _bundle()
    # The goal canvas is derived from the robot and checked against what the
    # observations bind: a bundle trained with anchor sub-masking binds the
    # anchor masks.
    bundle.robot.kinematic_info = SimpleNamespace(body_names=["root", "hand"])
    bundle.robot.anchor_body_index = 0
    bundle.robot.trackable_bodies_subset = ["root", "hand"]
    bundle.env.control_components = {
        "masked_mimic": MaskedMimicControlConfig(
            anchor_pose_submasking=True, local_target_condition_probability=0.5
        )
    }
    bundle.env.observation_components = {
        "anchor": MdpComponent(
            _anchor_mask_passthrough,
            {"mask": EnvContext.masked_mimic.anchor_translation_mask},
        )
    }
    bundle.env.termination_components = {"tracking": "unused"}
    _write_bundle(monkeypatch, tmp_path, bundle)
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    captured = {}
    _install_runtime(monkeypatch, captured)

    hook = tmp_path / "reach.py"
    hook.write_text(
        "from protomotions.envs.control.masked_mimic_tasks.controller import (\n"
        "    configure_masked_mimic_task,\n"
        ")\n"
        "def apply_inference_overrides(configs, args):\n"
        "    configure_masked_mimic_task(configs.env, configs.robot, 'reaching')\n"
        "    configs.env.control_components['masked_mimic'].reach_body_name = 'hand'\n"
    )

    run(
        _argv(checkpoint, "isaacgym", ["--inference-config", str(hook)]),
        app_launcher=None,
    )

    env = captured["env_config"]
    assert isinstance(
        env.control_components["masked_mimic"], MaskedMimicTaskControlConfig
    )
    assert env.control_components["masked_mimic"].task == "reaching"
    assert env.control_components["masked_mimic"].reach_body_name == "hand"
    assert env.termination_components == {}


def test_run_native_tasks_reject_full_tracking_evaluation(monkeypatch, tmp_path):
    """The masked-mimic task config refuses --full-eval."""
    _write_bundle(monkeypatch, tmp_path, _bundle())
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    _install_runtime(monkeypatch, {})
    with pytest.raises(ValueError, match="full evaluation"):
        run(
            _argv(
                checkpoint,
                "isaacgym",
                ["--inference-config", MASKED_MIMIC_INFERENCE_CONFIG, "--full-eval"],
            ),
            app_launcher=None,
        )


def test_run_keeps_random_command_source_by_default(monkeypatch, tmp_path):
    """Without the keyboard inference config the steering command source stays random."""
    from protomotions.envs.control.command_source import (
        RandomTargetCommandSourceConfig,
    )
    from protomotions.envs.control.target_control import (
        TargetControlConfig,
    )

    bundle = _bundle()
    bundle.env.control_components = {
        "target": TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                tar_change_time_min=2.0, tar_change_time_max=8.0, tar_dist_max=6.0
            )
        )
    }
    _write_bundle(monkeypatch, tmp_path, bundle)
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(_argv(checkpoint, "isaacgym"), app_launcher=None)

    command_source = captured["env_config"].control_components["target"].command_source
    assert isinstance(command_source, RandomTargetCommandSourceConfig)
    assert command_source.tar_change_time_min == 2.0
    assert command_source.tar_change_time_max == 8.0
    assert command_source.tar_dist_max == 6.0
    assert captured["binding_check"][0][0] is captured["env_config"]


def test_run_target_keyboard_inference_config_sets_keyboard_source(
    monkeypatch, tmp_path
):
    """The keyboard inference config swaps the steering command source to the keyboard one."""
    from protomotions.envs.control.command_source import (
        KeyboardTargetCommandSourceConfig,
        RandomTargetCommandSourceConfig,
    )
    from protomotions.envs.control.target_control import (
        TargetControlConfig,
    )

    bundle = _bundle()
    bundle.env.control_components = {
        "target": TargetControlConfig(command_source=RandomTargetCommandSourceConfig())
    }
    _write_bundle(monkeypatch, tmp_path, bundle)
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_text("checkpoint")
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(
        _argv(
            checkpoint,
            "isaacgym",
            ["--inference-config", TARGET_KEYBOARD_INFERENCE_CONFIG],
        ),
        app_launcher=None,
    )

    command_source = captured["env_config"].control_components["target"].command_source
    assert isinstance(command_source, KeyboardTargetCommandSourceConfig)


def test_run_applies_pretrained_checkpoint_cli_override(monkeypatch, tmp_path):
    """--overrides can repoint a pretrained module's checkpoint path."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    prior_checkpoint = tmp_path / "prior" / "last.ckpt"
    prior_checkpoint.parent.mkdir()
    prior_checkpoint.write_text("prior")
    bundle = _bundle()
    bundle.agent.pretrained_modules = {
        "prior": SimpleNamespace(checkpoint_path="/stale/prior/last.ckpt")
    }
    _write_bundle(monkeypatch, run_dir, bundle)
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(
        _argv(
            checkpoint,
            "isaacgym",
            [
                "--overrides",
                f"agent.pretrained_modules.prior.checkpoint_path={prior_checkpoint}",
            ],
        ),
        app_launcher=None,
    )

    assert captured["agent_config"].pretrained_modules["prior"].checkpoint_path == str(
        prior_checkpoint
    )


def test_run_does_not_infer_pretrained_checkpoint_paths(monkeypatch, tmp_path):
    """Without an override a stale pretrained checkpoint path is left untouched."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    bundle = _bundle()
    bundle.agent.pretrained_modules = {
        "prior": SimpleNamespace(checkpoint_path="/stale/prior/last.ckpt")
    }
    _write_bundle(monkeypatch, run_dir, bundle)
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(_argv(checkpoint, "isaacgym"), app_launcher=None)

    assert (
        captured["agent_config"].pretrained_modules["prior"].checkpoint_path
        == "/stale/prior/last.ckpt"
    )


def test_run_full_eval_switches_simulator_and_applies_cli_overrides(
    monkeypatch, tmp_path, capsys
):
    """--full-eval switches the backend, applies the flags and prints the metrics."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    bundle = _bundle()
    _write_bundle(monkeypatch, run_dir, bundle)
    calls = []

    class _EvalEvaluator:
        def __init__(self):
            self.eval_count = 5

        def evaluate(self):
            from protomotions.agents.evaluators.manager import EvaluationResult

            calls.append(("evaluate", self.eval_count))
            return EvaluationResult(
                logs={"score/a": 1.25, "score/b": 2.5}, score=3.75, num_items=2
            )

    simulator = SimpleNamespace(shutdown=lambda: calls.append(("shutdown",)))
    captured = {"evaluator": _EvalEvaluator(), "simulator": simulator}
    _install_runtime(monkeypatch, captured)

    monkeypatch.setattr(
        "protomotions.simulator.factory.update_simulator_config_for_test",
        lambda current_simulator_config, new_simulator, robot_config: SimpleNamespace(
            _target_="protomotions.simulator.newton.simulator.NewtonSimulator",
            num_envs=current_simulator_config.num_envs,
            headless=current_simulator_config.headless,
            sim=SimpleNamespace(fps=120, decimation=4),
            switched_to=new_simulator,
            apply_inference_defaults=_noop_defaults,
        ),
    )
    run(
        _argv(
            checkpoint,
            "newton",
            [
                "--full-eval",
                "--headless",
                "--num-envs",
                "3",
                "--motion-file",
                "override.motion",
                "--scenes-file",
                "none",
                "--overrides",
                "env.save_dir=cli",
            ],
        ),
        app_launcher=None,
    )

    output = capsys.readouterr().out
    assert "EVALUATION RESULTS" in output
    assert "score/a: 1.250000" in output
    assert "Items Evaluated: 2" in output
    assert "Overall Score: 3.750000" in output
    build = captured["build_kwargs"]
    assert build["simulator_config"].num_envs == 3
    assert build["simulator_config"].headless is True
    assert build["simulator_config"].switched_to == "newton"
    assert build["motion_lib_config"].motion_file == "override.motion"
    assert build["scene_lib_config"].scene_file is None
    assert build["scene_lib_config"].asset_root is None
    assert captured["env_config"].save_dir == "cli"  # a general --override applied
    assert ("evaluate", 0) in calls
    assert captured["load"] == (str(checkpoint), False, False)
    assert ("shutdown",) in calls


def test_run_explicit_motion_file_clears_stale_shard_subset(monkeypatch, tmp_path):
    """--motion-file replaces the saved motion and clears its shard selection."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    bundle = _bundle(
        motion_lib=MotionLibConfig(
            motion_file="/tmp/pack_slurmrank.pt", motion_file_shard_indices=[1, 3]
        )
    )
    _write_bundle(monkeypatch, run_dir, bundle)
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(
        _argv(checkpoint, "isaacgym", ["--motion-file", "replacement.pt"]),
        app_launcher=None,
    )

    motion = captured["build_kwargs"]["motion_lib_config"]
    assert motion.motion_file == "replacement.pt"
    assert motion.motion_file_shard_indices is None


def test_run_simple_test_recomputes_scene_asset_root(monkeypatch, tmp_path):
    """--scenes-file points scene_lib at the new file and re-derives its asset root."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.ckpt"
    checkpoint.write_text("checkpoint")
    _write_bundle(monkeypatch, run_dir, _bundle())
    scene_file = tmp_path / "assets" / "scenes" / "scene.yaml"
    scene_file.parent.mkdir(parents=True)
    scene_file.write_text("scene")
    captured = {}
    _install_runtime(monkeypatch, captured)

    run(
        _argv(checkpoint, "isaacgym", ["--scenes-file", str(scene_file)]),
        app_launcher=None,
    )

    assert captured["build_kwargs"]["scene_lib_config"].asset_root == str(
        tmp_path / "assets"
    )
    assert captured["simple_test"] is True
    assert captured["load"] == (str(checkpoint), False, False)
