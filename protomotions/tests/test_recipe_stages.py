# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every example experiment through the stages a training run and its
checkpoint go through (``protomotions/tests/support/recipe_stages.py``, which
names the stages); the internal experiment tree carries the same test over its
own recipes.

The fast stages (config, bundle, inference) run in CI for every recipe, 22 / 22:
the four that read another run's checkpoint while their configs build
(``CONFIGS_NEED_CHECKPOINT``) take the ``bundles`` fixture, the checkpoint
recipes' offline configs saved as bundles beside checkpoint paths that are never
written (``write_bundles``, seconds, in every run), which is all those stages
open. The build stage (``PROTOMOTIONS_RECIPE_BUILD_SLOW=1``, minutes) builds
each on MuJoCo with one env, resets, steps, resets the stepped env and steps
again, sets the agent up and runs the forward ``fit()`` runs before the first
rollout; the recipes whose build needs data the tree does not ship are listed in
``NEEDS_DATA_TO_BUILD`` with the exception they raise. Eleven recipes read
another run's checkpoint (``READS_CHECKPOINT``): the fast stages run the other
seven without one, which is what keeps them out of ``CONFIGS_NEED_CHECKPOINT``;
the build stage loads the weights, so the slow run writes untrained checkpoints
through ``agent.save`` in dependency order (``CHECKPOINT_RECIPES``: three
trackers, then the prior built on one of them) and hands those to all eleven.
Every verdict is for ``--robot-name g1`` except the recipes ``ROBOT`` names,
the two Metatron SOMA recipes, which require ``soma23``: their fast stages run
on that robot in every run, and their build takes its motion file, whose LFS
payload the checkout must have (``MOTION_FILES``; the build skips otherwise).
"""

import importlib

import pytest

from protomotions.tests.support import recipe_stages
from protomotions.tests.support.recipe_stages import (
    MOTION_FILE,
    MOTION_FILES,
    REPO_ROOT,
    SLOW,
    accepted_extra_argv,
    build_experiment_configs,
    checkpoint_extras,
    declared_arguments,
    eager_mujoco_build,
    experiment_names,
    has_payload,
    recipe_path,
    remove_checkpoints,
    stage_build,
    stage_bundle,
    stage_config,
    stage_inference,
    with_motion_payload,
    write_bundles,
    write_checkpoints,
)

PACKAGE = "examples.experiments"
EXPERIMENTS = REPO_ROOT / "examples" / "experiments"

# Modules under examples/experiments that are not experiments: shared building
# blocks other recipes import, and the signature template ``format`` (the one
# helper with an ``env_config``).
HELPER_MODULES = {
    "format",
    "gpc.prior_context",
    "masked_mimic.deployable_g1_observations",
}

# The checkpoints the slow run writes, in order: a key, the recipe (package and
# name), the checkpoints (by key) that recipe reads in turn, and its robot.
CHECKPOINT_RECIPES = {
    # The FSQ tracker whose decoder the GPC prior freezes (40 FSQ scalars, five
    # per prior token).
    "tracker": (PACKAGE, "mimic.fsq", {}, "g1"),
    # The five-body G1 tracker Metatron distils.
    "five_body_tracker": (PACKAGE, "metatron.g1_teacher", {}, "g1"),
    # The SOMA23 animation expert the SOMA student distils.
    "soma_teacher": (PACKAGE, "metatron.soma_teacher", {}, "soma23"),
    # The prior the PEFT recipes adapt.
    "prior": (PACKAGE, "gpc.prior", {"tracker_checkpoint": "tracker"}, "g1"),
}
# Recipes that read another run's checkpoint: the argument that takes it and
# the checkpoint above.
_PRIOR = {"prior_checkpoint": "prior"}
READS_CHECKPOINT = {
    "gpc.prior": {"tracker_checkpoint": "tracker"},
    "gpc.sft_target_prior_peft": {"tracker_checkpoint": "tracker", **_PRIOR},
    "gpc.steering_prior_peft": _PRIOR,
    "gpc.target_prior_peft": _PRIOR,
    "gpc.task_steering_headvel_prior_peft": _PRIOR,
    "gpc.task_steering_headvel_prior_peft_amp": _PRIOR,
    "gpc.task_target_prior_peft": _PRIOR,
    "gpc.task_target_prior_peft_amp": _PRIOR,
    "metatron.g1_student": {"expert_model_path": "five_body_tracker"},
    "metatron.soma_student": {"expert_model_path": "soma_teacher"},
    # MaskedMimic distils a full-body tracker: its loss target is the expert's
    # actions, which fit()'s buffer setup refuses to start without (the build
    # stage runs that setup). With the path the recipe also reads the expert's
    # bundle while its configs build, to copy the expert's observations.
    "masked_mimic.transformer": {"expert_model_path": "tracker"},
}
# Of those, the recipes that read the checkpoint while their configs build (the
# fast stages hand them the sim-free bundles), with the exception they raise
# without one, so a recipe that starts failing for another reason fails the
# table test. The others read it at build only.
CONFIGS_NEED_CHECKPOINT = {
    "gpc.prior": (TypeError, "NoneType"),  # Path(None)
    "gpc.sft_target_prior_peft": (TypeError, "NoneType"),
    "metatron.g1_student": (ValueError, "expert-model-path"),
    "metatron.soma_student": (ValueError, "expert-model-path"),
}
# The robot a recipe requires when it is not g1 (every other verdict is for
# ``--robot-name g1``): the Metatron SOMA recipes refuse any other character.
ROBOT = {
    "metatron.soma_teacher": "soma23",
    "metatron.soma_student": "soma23",
}
# Recipes whose configs the bundle saver refuses today (the exception and a
# fragment of its message; the entry goes when the saver takes the value).
# Empty: Metatron's deployable observations pass one ``BakedAnchorFK`` both as
# a kernel and as the ``fk`` parameter of ``compute_historical_proprioception``,
# and the saver writes a kernel object in either position (item 57).
BUNDLE_REFUSED_TODAY = {}
# Recipes whose configs build, round-trip and derive, but whose MuJoCo build
# needs something the tree does not ship: the reason, the exception the build
# raises today and a fragment of its message.
NEEDS_DATA_TO_BUILD = {}


def _experiment_names():
    return [n for n in experiment_names(EXPERIMENTS) if n not in HELPER_MODULES]


def recipe_robot(name: str) -> str:
    """The robot ``name`` runs on in these tests: ``ROBOT``'s entry, else g1."""
    return ROBOT.get(name, "g1")


def _robots_of(keys):
    """The robots of the ``CHECKPOINT_RECIPES`` entries ``keys`` name."""
    return {CHECKPOINT_RECIPES[key][3] for key in keys}


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory):
    """``CHECKPOINT_RECIPES`` written as untrained checkpoints with their bundles,
    through ``agent.save`` as training writes them; ``None`` outside the slow
    run."""
    if not SLOW or not has_payload(MOTION_FILE):
        yield None
        return
    with pytest.MonkeyPatch.context() as monkeypatch:
        eager_mujoco_build(monkeypatch)
        written = write_checkpoints(
            with_motion_payload(CHECKPOINT_RECIPES), tmp_path_factory.mktemp
        )
    yield written
    remove_checkpoints(written)


@pytest.mark.parametrize("fails", ["write", "mktemp"])
def test_write_checkpoints_leaves_nothing_behind_when_an_entry_fails(
    monkeypatch, tmp_path, fails
):
    """A later entry that fails (its write, or its ``mktemp``) removes the
    checkpoints already written (a prior's run directory is gigabytes) and its
    own directory before the error propagates, so the session fixture has
    nothing to tear down. The failing write had been handed the earlier
    checkpoint through the recipe's flag."""
    extra_argvs = {}

    def write_checkpoint(experiment, out_dir, extra_argv=(), robot_name="g1"):
        extra_argvs[out_dir.name] = list(extra_argv)
        (out_dir / "last.ckpt").write_bytes(b"ckpt")
        if out_dir.name == "prior":
            raise RuntimeError("the prior's write failed")
        return out_dir / "last.ckpt"

    monkeypatch.setattr(recipe_stages, "write_checkpoint", write_checkpoint)
    made = []

    def mktemp(key):
        if key == "prior" and fails == "mktemp":
            raise RuntimeError("the prior's mktemp failed")
        made.append(tmp_path / key)
        made[-1].mkdir()
        return made[-1]

    recipes = {key: CHECKPOINT_RECIPES[key] for key in ("tracker", "prior")}
    with pytest.raises(RuntimeError, match=f"the prior's {fails} failed"):
        write_checkpoints(recipes, mktemp)
    if fails == "write":
        assert [path.name for path in made] == ["tracker", "prior"]
        assert extra_argvs["prior"] == [
            "--tracker-checkpoint",
            str(tmp_path / "tracker" / "last.ckpt"),
        ]
    else:
        assert [path.name for path in made] == ["tracker"]
    assert not any(path.exists() for path in made)


@pytest.fixture(scope="session")
def bundles(tmp_path_factory):
    """``CHECKPOINT_RECIPES`` as sim-free bundles beside checkpoint paths that
    are never written, in every run: what the fast stages of a recipe that
    reads another run's checkpoint open."""
    return write_bundles(CHECKPOINT_RECIPES, tmp_path_factory.mktemp)


@pytest.mark.parametrize("name", _experiment_names())
def test_configs_build_round_trip_and_derive(name, bundles, tmp_path):
    robot = recipe_robot(name)
    # Only the recipes that read the checkpoint while their configs build take
    # the bundles; the others run without one, which is what keeps them out of
    # CONFIGS_NEED_CHECKPOINT.
    reads = READS_CHECKPOINT[name] if name in CONFIGS_NEED_CHECKPOINT else {}
    extra = checkpoint_extras(reads, bundles, simulator=None)
    configs = stage_config(name, PACKAGE, robot_name=robot, **extra)
    if name in BUNDLE_REFUSED_TODAY:
        exception, match = BUNDLE_REFUSED_TODAY[name]
        with pytest.raises(exception, match=match):
            stage_bundle(configs, tmp_path / "bundle")
        # The inference view still derives from the built configs.
        stage_inference(name, PACKAGE, configs, robot_name=robot, **extra)
        return
    loaded = stage_bundle(configs, tmp_path / "bundle")
    stage_inference(name, PACKAGE, loaded, robot_name=robot, **extra)


@pytest.mark.skipif(not SLOW, reason="set PROTOMOTIONS_RECIPE_BUILD_SLOW=1 (minutes)")
@pytest.mark.skipif(
    not has_payload(MOTION_FILE),
    reason="needs the LFS payload of data/motion_for_trackers/g1_bones_seed_mini.pt",
)
@pytest.mark.parametrize("name", _experiment_names())
def test_builds_resets_steps_sets_up_and_forwards_on_mujoco(
    name, checkpoints, monkeypatch, tmp_path
):
    robot = recipe_robot(name)
    if not has_payload(MOTION_FILES[robot]):
        motion_file = MOTION_FILES[robot].relative_to(REPO_ROOT)
        pytest.skip(f"needs the LFS payload of {motion_file}")
    eager_mujoco_build(monkeypatch)
    extra = checkpoint_extras(READS_CHECKPOINT.get(name, {}), checkpoints)
    extra_argv = accepted_extra_argv(name, PACKAGE, extra)
    experiment = recipe_path(name, PACKAGE)
    if name in NEEDS_DATA_TO_BUILD:
        reason, exception, match = NEEDS_DATA_TO_BUILD[name]
        with pytest.raises(exception, match=match):
            stage_build(experiment, tmp_path, extra_argv, robot_name=robot)
        return
    stage_build(experiment, tmp_path, extra_argv, robot_name=robot)


def test_the_tables_name_real_experiments():
    names = set(_experiment_names())
    assert set(READS_CHECKPOINT) <= names
    assert set(CONFIGS_NEED_CHECKPOINT) <= set(READS_CHECKPOINT)
    assert set(BUNDLE_REFUSED_TODAY) <= names
    assert set(NEEDS_DATA_TO_BUILD) <= names
    assert set(ROBOT) <= names
    # Every robot a recipe requires has a motion file for its build stage.
    assert set(ROBOT.values()) <= set(MOTION_FILES)
    # Every checkpoint a recipe reads is written, through a flag the recipe
    # declares (accepted_extra_argv drops a flag the recipe does not declare),
    # by a recipe of the same robot (with_motion_payload leaves a robot's
    # entries out together, and the reader's build skips for the same reason).
    written = set(CHECKPOINT_RECIPES)
    for package, name, reads, robot in CHECKPOINT_RECIPES.values():
        assert name in experiment_names(REPO_ROOT.joinpath(*package.split("."))), name
        assert set(reads.values()) <= written, name
        assert set(reads) <= declared_arguments(name, package), name
        assert _robots_of(reads.values()) <= {robot}, name
    for name, reads in READS_CHECKPOINT.items():
        assert set(reads.values()) <= written, name
        assert set(reads) <= declared_arguments(name, PACKAGE), name
        assert _robots_of(reads.values()) <= {recipe_robot(name)}, name
    assert HELPER_MODULES <= set(experiment_names(EXPERIMENTS))
    # A helper is a building block without an ``env_config``, or the
    # signature template ``format``.
    for name in HELPER_MODULES:
        module = importlib.import_module(f"{PACKAGE}.{name}")
        assert name == "format" or not hasattr(module, "env_config"), name
    # Without its checkpoint each recipe still fails to build its configs, for
    # the recorded reason.
    for name, (exception, match) in CONFIGS_NEED_CHECKPOINT.items():
        with pytest.raises(exception, match=match):
            build_experiment_configs(name, PACKAGE, recipe_robot(name))


def test_a_build_only_recipe_cannot_read_its_checkpoint_in_the_fast_stages(
    monkeypatch,
):
    """A recipe outside ``CONFIGS_NEED_CHECKPOINT`` that starts reading its
    checkpoint while its configs build fails the fast stage instead of opening
    the shipped bundle its argument defaults to: ``fake_args`` seeds the
    checkpoint arguments with paths that never resolve."""
    from protomotions.utils.experiment_configs import (
        load_experiment_configs_from_checkpoint,
    )

    recipe = importlib.import_module(f"{PACKAGE}.gpc.target_prior_peft")
    agent_config = recipe.agent_config

    def reads_the_prior_while_the_configs_build(robot_config, env_config, args):
        load_experiment_configs_from_checkpoint(args.prior_checkpoint)
        return agent_config(robot_config, env_config, args)

    monkeypatch.setattr(recipe, "agent_config", reads_the_prior_while_the_configs_build)
    with pytest.raises(FileNotFoundError, match="missing"):
        stage_config("gpc.target_prior_peft", PACKAGE)
