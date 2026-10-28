# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A shipped recipe through the stages a training run and its checkpoint go
through, one function per stage.

Two tests enumerate recipe directories over these stages, with the same
verdict tables beside them: ``protomotions/tests/test_recipe_stages.py`` for
``examples/experiments``, and its twin in the internal experiment tree for the
recipes there. Every verdict is for ``--robot-name g1`` (an IMU body, no
contact bodies) unless the test's robot table names another robot for the
recipe (the Metatron SOMA recipes require ``soma23``); another robot can
change a verdict for a recipe that binds a robot-dependent context name. The
build stage takes the robot's motion file (:data:`MOTION_FILES`).

* :func:`stage_config`: ``build_configs`` from the fake training arguments
  (:func:`fake_args`) and the config-only binding check, what ``train_agent.py``
  runs before building anything. The report's unknown lines may only be the
  names the data decides (the motion file's contact labels, the scene
  library's objects), both checked at build.
* :func:`stage_bundle`: ``save_experiment_configs`` then
  ``load_experiment_configs``, an equal tree per root, what resume and
  inference load.
* :func:`stage_inference`: ``derive_inference_configs`` with the recipe's
  ``apply_inference_overrides`` hook executed, not only arity-checked.
* :func:`stage_build` (minutes, behind ``PROTOMOTIONS_RECIPE_BUILD_SLOW=1``,
  :data:`SLOW`): the MuJoCo one-env build from the training command line
  (:func:`training_argv`, :func:`build_env`), ``env.reset``, an ``env.step``
  with finite observations and rewards (:func:`finite_step`), a reset of every
  row of the stepped env and a second step (``fit()``'s steady state, which
  clears live buffers and records on a mixed history ring),
  ``agent.setup(training=True)`` as the training entry point runs it, the
  agent's ``_before_fit`` hook (what ``fit()`` does once at training start,
  e.g. PEFT pinning its reference policy) and
  :func:`registration_forward`, the experience-buffer setup ``fit()`` then
  runs on the post-reset observations, through the agent's own
  ``setup_experience_buffer``: the model's ``forward`` (for an actor-critic,
  the actor and the critic) and ``_register_model_output_keys``, so every key
  the model declares for the buffer is present and a declared rollout state
  has the shape and dtype of its spec, then the rewards, the dones and the
  algorithm's own key registrations; every output finite.

MuJoCo has no body-mass randomisation, so a build of a recipe that configures
it stands in ``None`` for ``BodyMassDomainRandomizationConfig``
(:func:`eager_mujoco_build`), as a MuJoCo smoke run of that recipe has to.
:func:`write_checkpoint` writes an untrained checkpoint of a recipe and its
bundle through ``agent.save``, as training writes them, for the recipes that
read another run's checkpoint while their configs build; a session fixture
removes them with :func:`remove_checkpoints` when the run ends (a prior's are
gigabytes). Those recipes open only the bundle beside the path while their
configs build (``load_experiment_configs_from_checkpoint``), so
:func:`write_bundle` writes just that, the offline configs' bundle beside a
checkpoint path that is never written, in seconds and without a simulator; a
fixture available in every run writes the same chain that way
(:func:`write_bundles`), and the config, bundle and inference stages of those
recipes run in CI. The build stage loads the weights and takes the checkpoints.
"""

import argparse
import importlib
import io
import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

import protomotions
import protomotions.envs.component_manager as component_manager
import protomotions.simulator.base_simulator.config as sim_config
from protomotions.envs.context_check import (
    ConfigBindingReport,
    check_bindings_against_config,
)

REPO_ROOT = Path(protomotions.__file__).resolve().parents[1]
# The motion file the build stage trains each robot on; both are Git LFS files
# (:func:`has_payload` tells a fetched payload from the pointer left in its
# place). g1's is the default and the gate for the whole build stage.
MOTION_FILES = {
    "g1": REPO_ROOT / "data/motion_for_trackers/g1_bones_seed_mini.pt",
    "soma23": REPO_ROOT / "data/motion_for_trackers/soma23_bones_seed_mini.pt",
}
MOTION_FILE = MOTION_FILES["g1"]
LFS_POINTER_PREFIX = b"version https://git-lfs"
SLOW = os.environ.get("PROTOMOTIONS_RECIPE_BUILD_SLOW") == "1"

# The config-only check leaves these to the build: the motion file's contact
# labels and the scene library's objects are not in the configs.
DATA_DEPENDENT_UNKNOWN = ("mimic.ref_state.rigid_body_contacts", "context path 'scene.")
# The recipe packages whose modules bind the body-mass config class at import.
RECIPE_PACKAGES = ("examples.experiments", "internal.experiments")


def has_payload(path: Path) -> bool:
    """False for a missing file or a Git LFS pointer left in place of one."""
    return path.is_file() and not path.read_bytes()[:40].startswith(LFS_POINTER_PREFIX)


def experiment_names(experiments: Path) -> List[str]:
    """The dotted module names of every ``.py`` under ``experiments``, sorted."""
    return sorted(
        ".".join(path.relative_to(experiments).with_suffix("").parts)
        for path in experiments.rglob("*.py")
        if path.name != "__init__.py"
    )


def recipe_path(name: str, package: str) -> str:
    """``package.name`` as a path under the repository root."""
    return f"{package.replace('.', '/')}/{name.replace('.', '/')}.py"


# =============================================================================
# Configs offline
# =============================================================================


def fake_args(module, robot_name: str = "g1", **extra) -> argparse.Namespace:
    """The CLI namespace the experiment functions read, with every extra
    argument an experiment declares at its parser default and ``extra`` set
    over it (a recipe that reads another run's checkpoint gets its path here).
    The checkpoint arguments are seeded so that no declared default resolves to
    a shipped file. The robot is g1 unless a test asks otherwise."""
    args = argparse.Namespace(
        robot_name=robot_name,
        simulator="isaacgym",
        headless=True,
        num_envs=4,
        experiment_name="config-check",
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=64,
        training_max_steps=1000,
        training_max_iterations=None,
        tracker_checkpoint=None,
        expert_model_path=None,
        # The PEFT recipes declare the shipped prior as their default; a fast
        # stage that started reading it while the configs build would open the
        # shipped bundle and pass, so the seed is a path that never resolves.
        # (Not None: the recipes put it into a str field the saver checks.)
        prior_checkpoint="missing/last.ckpt",
        checkpoint=None,
    )
    declare = getattr(module, "additional_experiment_arguments", None)
    if declare is not None:
        parser = argparse.ArgumentParser()
        declare(parser)
        for action in parser._actions:
            if action.dest != "help" and not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def build_experiment_configs(
    module_name: str, package: str, robot_name: str = "g1", **extra
):
    """The experiment's configs exactly as ``build_configs`` builds them for
    training, from fake CLI arguments; nothing else is built. ``build_configs``
    runs the config-only binding check as its last step, so a recipe the check
    refuses (or one that cannot build offline) raises here, as it would at
    training."""
    from protomotions.utils.config_builder import build_configs

    module = importlib.import_module(f"{package}.{module_name}")
    args = fake_args(module, robot_name, **extra)
    with redirect_stdout(io.StringIO()):  # the robot factories print
        return build_configs(module, args)


def check_experiment(
    module_name: str, package: str, robot_name: str = "g1", **extra
) -> ConfigBindingReport:
    """Build one experiment's configs offline and return the binding report,
    asserting the check draws no random number. A recipe the check refuses
    raises from ``build_configs``; callers that expect a refusal use
    ``pytest.raises`` instead."""
    configs = build_experiment_configs(module_name, package, robot_name, **extra)
    rng = torch.get_rng_state()
    report = check_bindings_against_config(
        configs.env, configs.robot, configs.terrain, configs.agent
    )
    assert torch.equal(torch.get_rng_state(), rng)
    return report


def stage_config(name: str, package: str, **extra):
    """Build the configs and run the config-only check; the report's unknown
    lines may only be the data-dependent names."""
    configs = build_experiment_configs(name, package, **extra)
    report = check_bindings_against_config(
        configs.env, configs.robot, configs.terrain, configs.agent
    )
    assert report.errors == [], "\n".join(report.errors)
    for line in report.unknown:
        assert any(fragment in line for fragment in DATA_DEPENDENT_UNKNOWN), line
    return configs


def stage_bundle(configs, workdir: Path):
    """Save the bundle and load it back; every root rebuilds to an equal tree."""
    from protomotions.utils.config_schema import to_tree
    from protomotions.utils.experiment_configs import (
        load_experiment_configs,
        save_experiment_configs,
    )

    loaded = load_experiment_configs(save_experiment_configs(configs, workdir))
    unequal = [
        root
        for root in configs.ROOTS
        if to_tree(getattr(configs, root), where=root)
        != to_tree(getattr(loaded, root), where=root)
    ]
    assert unequal == [], f"the bundle round trip differs in {unequal}"
    return loaded


def stage_inference(name: str, package: str, loaded, **extra):
    """Derive the inference view, executing the recipe's hook if it has one."""
    from protomotions.utils.inference_utils import (
        _check_hook_arity,
        derive_inference_configs,
    )

    module = importlib.import_module(f"{package}.{name}")
    args = fake_args(module, **extra)
    args.inference_config = None
    hooks = []
    hook = getattr(module, "apply_inference_overrides", None)
    if hook is not None:
        _check_hook_arity(hook)
        hooks.append(hook)
    with redirect_stdout(io.StringIO()):
        derive_inference_configs(loaded, args, {}, hooks)


# =============================================================================
# The MuJoCo one-env build
# =============================================================================


def training_argv(
    experiment: str,
    motion_file: Path = MOTION_FILE,
    extra_argv: Iterable[str] = (),
    robot_name: str = "g1",
) -> List[str]:
    """The training command line for one MuJoCo env of ``experiment`` (a recipe
    path under the repository root), with the recipe's own flags appended."""
    return [
        "--robot-name",
        robot_name,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--batch-size",
        "64",
        "--motion-file",
        str(motion_file),
        "--experiment-path",
        str(REPO_ROOT / experiment),
        "--experiment-name",
        "build_smoke",
        "--training-max-steps",
        "1000",
        "--headless",
        "true",
        *extra_argv,
    ]


def declared_arguments(name: str, package: str):
    """The destinations of the arguments ``package.name`` declares with
    ``additional_experiment_arguments`` (``tracker_checkpoint`` for
    ``--tracker-checkpoint``)."""
    module = importlib.import_module(f"{package}.{name}")
    declare = getattr(module, "additional_experiment_arguments", None)
    if declare is None:
        return set()
    parser = argparse.ArgumentParser()
    declare(parser)
    return {action.dest for action in parser._actions if action.dest != "help"}


def accepted_extra_argv(name: str, package: str, extra: Dict[str, object]) -> List[str]:
    """``--flag value`` pairs of ``extra`` that the recipe's own parser declares
    (argparse exits 2 on an unknown flag; the main parser's arguments, such as
    the simulator, are already on the training argv)."""
    declared = declared_arguments(name, package)
    argv = []
    for dest, value in extra.items():
        if dest in declared:
            argv += ["--" + dest.replace("_", "-"), str(value)]
    return argv


def checkpoint_extras(
    reads: Dict[str, str], written, simulator: Optional[str] = "mujoco"
) -> Dict[str, str]:
    """``fake_args`` extras handing a recipe the fixture's checkpoints or
    bundles: ``reads`` maps the recipe's argument to the key in ``written``.
    Empty when ``written`` is None (the checkpoint fixture outside the slow run)
    or the recipe reads nothing. The checkpoints are MuJoCo builds and a recipe
    may refuse a teacher trained on another simulator, so the simulator comes
    with them; the sim-free bundles hold the offline configs' own simulator and
    pass ``simulator=None``."""
    if written is None or not reads:
        return {}
    extra = {dest: str(written[key]) for dest, key in reads.items()}
    if simulator is not None:
        extra["simulator"] = simulator
    return extra


class NoBodyMass:
    """``BodyMassDomainRandomizationConfig(...)`` -> ``None`` (MuJoCo lacks it)."""

    def __new__(cls, **kwargs):
        return None


def eager_mujoco_build(monkeypatch) -> None:
    """What a MuJoCo build on the CPU needs patched: eager kernels (compiling
    the components dominates the build time) and no body-mass randomisation,
    replaced where the recipes look the class up: the config module (fresh
    recipe loads) and every recipe module already imported, which bound the
    real class at import."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", False)
    monkeypatch.setattr(sim_config, "BodyMassDomainRandomizationConfig", NoBodyMass)
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith(RECIPE_PACKAGES) and hasattr(
            module, "BodyMassDomainRandomizationConfig"
        ):
            monkeypatch.setattr(module, "BodyMassDomainRandomizationConfig", NoBodyMass)


def build_env(configs):
    """The environment of resolved configs on the CPU, built as the entry
    points build it (``build_env_from_configs``; the friction conversion is
    written back into ``configs``)."""
    from protomotions.utils.component_builder import build_env_from_configs

    return build_env_from_configs(configs, torch.device("cpu"))


def build_reset_step(experiment: str, motion_file: Path = MOTION_FILE):
    """Build ``experiment``'s environment on MuJoCo with one env, reset, step once.

    The configs come from the training command line (``build_experiment``), so
    the recipe's ``configure_robot_and_simulator`` and ``env_config`` run as
    they do for a run. Returns the env, the first step's observations and its
    rewards.
    """
    from protomotions.utils.config_builder import build_experiment

    configs = build_experiment(training_argv(experiment, motion_file))
    env = build_env(configs)
    env.reset()
    actions = torch.zeros(env.num_envs, configs.robot.number_of_actions)
    obs, rewards, _, _, _ = env.step(actions)
    return env, obs, rewards


_FABRIC = None


def cpu_fabric():
    """One launched single-process CPU Fabric for every agent a test builds."""
    global _FABRIC
    if _FABRIC is None:
        from lightning.fabric import Fabric

        from protomotions.utils.fabric_config import FabricConfig

        _FABRIC = Fabric(
            **FabricConfig(
                accelerator="cpu", devices=1, num_nodes=1, loggers=[], callbacks=[]
            ).as_kwargs()
        )
        _FABRIC.launch()
    return _FABRIC


def build_agent(configs, env, root_dir: Path):
    """The recipe's agent over a built env, as ``train_agent.py`` constructs it."""
    from protomotions.utils.hydra_replacement import get_class

    return get_class(configs.agent._target_)(
        config=configs.agent, env=env, fabric=cpu_fabric(), root_dir=root_dir
    )


def registration_forward(agent, obs: Dict[str, torch.Tensor]):
    """``fit()``'s experience-buffer setup on the observations ``env.reset``
    returned, through the agent's own ``setup_experience_buffer``: the buffer,
    the env keys registered from the observations, the model's own ``forward``
    in eval mode under ``no_grad``, ``_register_model_output_keys`` (every key
    the model declares for the buffer is indexed and a declared rollout state's
    shape and dtype checked against its spec), then the rewards, the dones and
    the algorithm's own key registrations. The model must declare a key of its
    own and every output tensor must be finite. Returns the output."""
    output = agent.setup_experience_buffer(obs)
    assert agent.model_output_keys, "the model declares no key of its own"
    for key, value in output.items(True, True):
        assert not isinstance(value, torch.Tensor) or torch.isfinite(value).all(), key
    return output


def finite_step(env, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """One ``env.step``: finite rewards and observations, a bool reset flag.
    Returns the observations."""
    obs, rewards, resets, _, _ = env.step(actions)
    assert torch.isfinite(rewards).all(), rewards
    assert resets.dtype == torch.bool, resets.dtype
    for key, value in obs.items():
        assert torch.isfinite(value).all(), key
    return obs


def stage_build(
    experiment: str,
    workdir: Path,
    extra_argv: Iterable[str] = (),
    robot_name: str = "g1",
):
    """MuJoCo one-env build from the training argv (the robot's motion file),
    reset, a step with finite observations and rewards, a reset of every row of
    the stepped env and a second step (``fit()``'s steady state),
    ``agent.setup(training=True)``, the agent's ``_before_fit`` hook and
    ``fit()``'s experience-buffer setup on the first reset's observations.
    Closes the env."""
    from protomotions.utils.config_builder import build_experiment

    argv = training_argv(experiment, MOTION_FILES[robot_name], extra_argv, robot_name)
    with redirect_stdout(io.StringIO()):
        configs = build_experiment(argv)
    env = build_env(configs)
    try:
        agent = build_agent(configs, env, workdir)
        obs, _ = env.reset()
        actions = torch.zeros(env.num_envs, configs.robot.number_of_actions)
        finite_step(env, actions)
        # fit()'s steady state is step -> reset(done_ids) -> step. A reset of a
        # stepped env clears live buffers (the history ring holding a pushed
        # state, the controls after an advance, the contact memory, the action
        # stage) and the next record lands on a mixed ring; a reset at start
        # cannot show either.
        env.reset(torch.arange(env.num_envs))
        finite_step(env, actions)
        agent.setup(training=True)
        assert sum(p.numel() for p in agent.model.parameters()) > 0
        agent._before_fit()
        registration_forward(agent, obs)
    finally:
        env.close()


def write_checkpoint(
    experiment: str,
    out_dir: Path,
    extra_argv: Iterable[str] = (),
    robot_name: str = "g1",
) -> Path:
    """An untrained checkpoint of ``experiment`` on ``robot_name`` and its
    bundle in ``out_dir``, written through ``agent.save`` as training writes
    them, for a recipe that reads another run's checkpoint while its configs
    build. The caller applies :func:`eager_mujoco_build`."""
    from protomotions.utils.config_builder import build_experiment
    from protomotions.utils.experiment_configs import save_experiment_configs

    argv = training_argv(experiment, MOTION_FILES[robot_name], extra_argv, robot_name)
    with redirect_stdout(io.StringIO()):
        configs = build_experiment(argv)
    save_experiment_configs(configs, out_dir)
    env = build_env(configs)
    try:
        agent = build_agent(configs, env, out_dir)
        agent.setup(training=True)
        with redirect_stdout(io.StringIO()):
            agent.save(checkpoint_name="last.ckpt")
    finally:
        env.close()
    return out_dir / "last.ckpt"


def with_motion_payload(recipes):
    """The entries of ``recipes`` (the shape :func:`write_checkpoints` takes)
    whose robot's motion file has its payload; the others cannot build. A
    recipe that reads a checkpoint left out this way runs on the same robot
    and its build skips for the same reason."""
    return {
        key: (package, name, reads, robot)
        for key, (package, name, reads, robot) in recipes.items()
        if has_payload(MOTION_FILES[robot])
    }


def write_checkpoints(recipes, mktemp) -> Dict[str, Path]:
    """The checkpoints of ``recipes`` in order,
    ``{key: (package, name, reads, robot)}`` where ``reads`` maps a recipe
    argument to the key of an earlier entry whose checkpoint it takes (a prior
    reads its tracker's) and ``robot`` is the robot the recipe trains. Each is
    written into ``mktemp(key)``; the caller applies :func:`eager_mujoco_build`
    and passes only entries whose motion payload is present
    (:func:`with_motion_payload`). An entry that fails leaves nothing behind:
    the checkpoints already written (gigabytes for a prior) and its own
    directory are removed before the error propagates."""
    written: Dict[str, Path] = {}
    out_dir: Optional[Path] = None
    try:
        for key, (package, name, reads, robot) in recipes.items():
            out_dir = mktemp(key)
            extra = {dest: str(written[other]) for dest, other in reads.items()}
            written[key] = write_checkpoint(
                recipe_path(name, package),
                out_dir,
                accepted_extra_argv(name, package, extra),
                robot,
            )
    except BaseException:
        remove_checkpoints(written)
        # The failing entry's directory; when ``mktemp`` itself raised, this is
        # the previous entry's, just removed above.
        if out_dir is not None and out_dir.exists():
            shutil.rmtree(out_dir)
        raise
    return written


def write_bundle(experiment: str, package: str, out_dir: Path, **extra) -> Path:
    """The bundle of ``experiment``'s offline configs (:func:`stage_config`:
    ``fake_args``, no env, no simulator) in ``out_dir``, and the path of the
    checkpoint that is not written beside it. A recipe that reads another run's
    checkpoint while its configs build opens only the bundle beside the path
    (``load_experiment_configs_from_checkpoint``), so this is what its config,
    bundle and inference stages need; its build loads the weights and takes
    :func:`write_checkpoint`'s."""
    from protomotions.utils.experiment_configs import save_experiment_configs

    save_experiment_configs(stage_config(experiment, package, **extra), out_dir)
    return out_dir / "last.ckpt"


def write_bundles(recipes, mktemp) -> Dict[str, Path]:
    """The bundles of ``recipes`` in order, the shape :func:`write_checkpoints`
    takes, each in ``mktemp(key)``, a later entry reading the earlier bundles
    it names. Seconds on the CPU, so a fixture can write them in every run."""
    written: Dict[str, Path] = {}
    for key, (package, name, reads, robot) in recipes.items():
        extra = checkpoint_extras(reads, written, simulator=None)
        written[key] = write_bundle(
            name, package, mktemp(key), robot_name=robot, **extra
        )
    return written


def remove_checkpoints(written: Dict[str, Path]) -> None:
    """Delete the directories :func:`write_checkpoints` filled (the checkpoint,
    its inference twin, the env checkpoint and the bundle: gigabytes for a
    prior), for a session fixture's teardown. A delete that fails raises, so a
    leftover is a teardown error rather than a silent few gigabytes."""
    for path in written.values():
        shutil.rmtree(path.parent)
