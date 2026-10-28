# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the shipped pretrained bundles into the v2 self-contained layout.

The bundles shipped as the v1 ``resolved_configs.pt`` / ``.yaml`` pickle form,
which the v2 loader (``resolved_configs.json``, see
``protomotions/utils/experiment_configs.py``) cannot read. This tool rebuilds each
kept directory from the current recipe through the launcher's own build + save.

Run it from the repo root: the self-referential ``experiment_path`` that
``config.yaml`` records is relative, and the launcher resolves it against the
working directory.

Reproducibility is at the *config* level: ``rebuild_bundle_into`` rebuilds
``config.yaml`` / ``resolved_configs.json`` / ``experiment_config.py`` from the
written ``config.yaml`` byte-identically (the accompanying gate test asserts it).
The full end-to-end ``regenerate`` is not re-runnable as-is, because it reads the
v1 ``last.ckpt`` to slim the inference checkpoint and then deletes it, so a second
``regenerate`` on an already-migrated directory has no ``last.ckpt`` to read.

For each kept directory it writes, through ``launch.training.save_configs``:
``resolved_configs.json`` (the full training-form bundle, the training simulator
preserved), ``config.yaml`` (the launcher's ``vars(args)`` dump) and a fresh copy
of the current recipe as ``experiment_config.py`` (the saved v1 copy imported the
removed ``envs.component_factories`` and no longer builds). It then writes a
model-only ``inference_last.ckpt`` (the structure of
``BaseAgent.get_inference_state_dict``: ``model`` plus epoch / step_count /
run_start_time / best_evaluated_score, no optimizer or training state; for the AMP
tracker g1-bones-deploy the discriminator is dropped too) and a ``provenance.json``,
and removes the stale ``last.ckpt`` / ``resolved_configs.pt`` / ``.yaml`` files. A
directory that ships a second checkpoint (soma-bones' IsaacLab-trained variant) gets
it in the same model-only form (``inference_last_lab.ckpt``) with its own
``provenance_lab.json`` paired by suffix; the full source file is removed too.

The recipe for each directory is the current public recipe whose model strict-loads
the shipped checkpoint (verified by the accompanying needs-LFS gate test); the saved
v1 recipe is not used because it no longer imports.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_DISCRIMINATOR_PREFIXES = ("_discriminator", "_disc_critic")
_INFERENCE_META_KEYS = ("epoch", "step_count", "run_start_time", "best_evaluated_score")


@dataclass
class SecondaryCheckpoint:
    """A second checkpoint shipped in the same directory beside the primary one.

    The same config and model trained a different way (soma-bones ships the
    IsaacLab-trained variant beside the IsaacGym-trained primary). It is slimmed to
    its own model-only ``inference_<...>.ckpt`` and gets its own
    ``provenance_<...>.json`` with the identical flat schema, paired to the inference
    file by the suffix; the full source file is removed with the other stale
    artifacts. ``trained_on`` is written into that provenance only when set (the
    primary omits the key, as the model card says its training simulator cannot be
    recovered).
    """

    source: str  # the full checkpoint shipped in v1, e.g. "last_lab.ckpt"
    inference_name: str  # the slimmed model-only file, e.g. "inference_last_lab.ckpt"
    provenance_file: str  # e.g. "provenance_lab.json"
    trained_on: Optional[str] = None


@dataclass
class BundleSpec:
    """How to regenerate one pretrained directory."""

    directory: str  # relative to the repo root
    recipe: str  # current public recipe whose model matches the shipped checkpoint
    robot: str
    simulator: str  # the training simulator, preserved in the bundle
    motion_file: str
    num_envs: int
    batch_size: int
    experiment_name: str
    provenance_name: str
    overrides: List[str] = field(default_factory=list)
    drop_discriminator: bool = False  # AMP model slimmed into a PPO recipe
    secondary_checkpoints: List[SecondaryCheckpoint] = field(default_factory=list)
    # (find, replace) edits applied to the copied recipe before the build, for a
    # shipped checkpoint whose training diverged from the current public recipe in a
    # way no CLI override reaches (e.g. an action-scaling default changed after the
    # weights were trained). Each find string must be present or regeneration fails.
    recipe_substitutions: List[Tuple[str, str]] = field(default_factory=list)
    # (flag, value) pairs for a recipe's own ``additional_experiment_arguments`` (the
    # gpc prior's required ``--tracker-checkpoint``). The launcher dumps them into
    # config.yaml like any other arg, so _config_to_argv round-trips them.
    extra_args: List[Tuple[str, str]] = field(default_factory=list)


# The kept public directories. The two SMPL trackers, the AMP FSQ tracker
# soma_bones_fsq_amp_muon and masked_mimic/smpl were removed when the trackers were
# regenerated and are not listed here.
SPECS: List[BundleSpec] = [
    BundleSpec(
        directory="data/pretrained_models/motion_tracker/soma_bones_fsq",
        recipe="examples/experiments/mimic/fsq.py",
        robot="soma23",
        simulator="isaaclab",
        motion_file="data/motion_for_trackers/soma23_bones_seed_mini.pt",
        num_envs=8192,
        batch_size=16384,
        experiment_name="soma23_fsq_tracker",
        provenance_name="SOMA FSQ motion tracker",
    ),
    BundleSpec(
        directory="data/pretrained_models/motion_tracker/soma-bones",
        recipe="examples/experiments/mimic/mlp.py",
        robot="soma23",
        simulator="isaaclab",
        motion_file="data/motion_for_trackers/soma23_bones_seed_mini.pt",
        num_envs=1,
        batch_size=8192,
        experiment_name="soma23_tracker",
        provenance_name="SOMA motion tracker",
        # The old bundle trained with robot friction 0.5 (resolved_configs.yaml:
        # default_robot_friction: 0.5); the public mlp.py recipe leaves the default
        # 1.0, so the training knob is restored as an override. fsq.py sets 0.5 in the
        # recipe itself, so soma_bones_fsq needs none.
        overrides=["simulator.default_robot_friction=0.5"],
        secondary_checkpoints=[
            SecondaryCheckpoint(
                source="last_lab.ckpt",
                inference_name="inference_last_lab.ckpt",
                provenance_file="provenance_lab.json",
                trained_on="IsaacLab",
            )
        ],
    ),
    BundleSpec(
        directory="data/pretrained_models/motion_tracker/g1-bones-deploy",
        recipe="examples/experiments/mimic/mlp_bm_l2c2.py",
        robot="g1",
        simulator="isaaclab",
        motion_file="data/motion_for_trackers/g1_bones_seed_mini.pt",
        num_envs=8192,
        batch_size=16384,
        experiment_name="g1_bones_tracker",
        provenance_name="G1 BONES motion tracker (AMP, discriminator-slimmed)",
        drop_discriminator=True,
        # The shipped G1 weights (2026-08-10) predate commit a347b28b1 (2026-10-15),
        # which made make_bm_pd_action_config scale by effort_fraction * effort_limit
        # / stiffness with effort_fraction defaulting to 0.25 -- a quarter of the
        # action scale these weights trained with (the old pickle kept its config
        # insulated). effort_fraction=1.0 restores the trained scale; no CLI override
        # reaches action_scale, so the one-line recipe divergence is carried here.
        recipe_substitutions=[
            (
                "        action_config=make_bm_pd_action_config(robot_cfg),\n",
                "        # Divergence from the public recipe for this shipped "
                "checkpoint:\n"
                "        # effort_fraction=1.0 restores the pre-a347b28b1 action "
                "scale the\n"
                "        # 2026-08-10 G1 weights trained with (the public default is "
                "0.25).\n"
                "        action_config=make_bm_pd_action_config(\n"
                "            robot_cfg, effort_fraction=1.0\n"
                "        ),\n",
            )
        ],
    ),
    BundleSpec(
        directory="data/pretrained_models/gpc_prior/soma_bones",
        recipe="examples/experiments/gpc/prior.py",
        robot="soma23",
        simulator="isaaclab",
        motion_file="data/motion_for_trackers/soma23_bones_seed_mini.pt",
        num_envs=1024,
        batch_size=1024,
        experiment_name="soma23_gpc_prior",
        provenance_name="SOMA GPC prior",
        # prior.py sets default_robot_friction = 0.5 in the recipe itself, so no
        # override is needed. It requires --tracker-checkpoint (the frozen FSQ tracker
        # whose decoder config it embeds); that directory's resolved_configs.json is
        # regenerated in the same release, so the build reads the v2 bundle.
        extra_args=[
            (
                "--tracker-checkpoint",
                "data/pretrained_models/motion_tracker/soma_bones_fsq/"
                "inference_last.ckpt",
            )
        ],
    ),
]

# Stale v1 artifacts removed from each regenerated directory once the new bundle is
# written. There is no resolved_configs_inference.json in v2: inference derives its
# view from the one bundle at load.
_STALE_FILES = (
    "last.ckpt",
    "resolved_configs.pt",
    "resolved_configs.yaml",
    "resolved_configs_inference.pt",
    "resolved_configs_inference.yaml",
)


def _argv(spec: BundleSpec, experiment_path: str) -> List[str]:
    """The training command line that rebuilds this bundle's configs.

    ``experiment_path`` is the directory's own ``experiment_config.py`` (relative to
    the repo root), so ``config.yaml`` records the self-referential path and the
    bundle reproduces from the directory's own files.
    """
    argv = [
        "--robot-name",
        spec.robot,
        "--simulator",
        spec.simulator,
        "--num-envs",
        str(spec.num_envs),
        "--batch-size",
        str(spec.batch_size),
        "--motion-file",
        spec.motion_file,
        "--experiment-path",
        experiment_path,
        "--experiment-name",
        spec.experiment_name,
        "--create-config-only",
    ]
    for flag, value in spec.extra_args:
        argv += [flag, value]
    if spec.overrides:
        argv += ["--overrides", *spec.overrides]
    return argv


# config.yaml (the launcher's vars(args) dump) maps back to these flags; a field at
# its default is still emitted because the dump records it, so the round-trip is
# byte-identical. Mirrors the training parser in launch/training_cli.py.
_CONFIG_VALUE_FLAGS = [
    ("robot_name", "--robot-name"),
    ("simulator", "--simulator"),
    ("num_envs", "--num-envs"),
    ("batch_size", "--batch-size"),
    ("motion_file", "--motion-file"),
    ("experiment_path", "--experiment-path"),
    ("experiment_name", "--experiment-name"),
    ("wandb_project", "--wandb-project"),
    ("ngpu", "--ngpu"),
    ("nodes", "--nodes"),
    ("seed", "--seed"),
    ("training_max_steps", "--training-max-steps"),
]
_CONFIG_OPTIONAL_FLAGS = [
    ("scenes_file", "--scenes-file"),
    ("checkpoint", "--checkpoint"),
    ("kit_thread_count", "--kit-thread-count"),
    ("training_max_iterations", "--training-max-iterations"),
    # Recipe-specific args a recipe declares via additional_experiment_arguments.
    ("tracker_checkpoint", "--tracker-checkpoint"),
]
_CONFIG_STORE_TRUE_FLAGS = [
    ("use_wandb", "--use-wandb"),
    ("use_slurm", "--use-slurm"),
    ("torch_deterministic", "--torch-deterministic"),
    ("create_config_only", "--create-config-only"),
]


def _config_to_argv(config: dict) -> List[str]:
    """Reconstruct the training argv from a saved ``config.yaml``."""
    argv: List[str] = []
    for key, flag in _CONFIG_VALUE_FLAGS:
        if config.get(key) is not None:
            argv += [flag, str(config[key])]
    for key, flag in _CONFIG_OPTIONAL_FLAGS:
        if config.get(key) is not None:
            argv += [flag, str(config[key])]
    for key, flag in _CONFIG_STORE_TRUE_FLAGS:
        if config.get(key):
            argv += [flag]
    argv += ["--headless", str(config.get("headless", True))]
    if config.get("overrides"):
        argv += ["--overrides", *config["overrides"]]
    return argv


def _save_bundle(argv: List[str], experiment_path: str, out_dir: Path) -> None:
    """Build the configs from ``experiment_path`` and save the three bundle files to
    ``out_dir`` through the launcher's own ``save_configs``.

    ``out_dir`` must differ from the directory holding ``experiment_path`` so
    ``save_configs``'s copy of the experiment source does not hit ``SameFileError``
    (the self-referential path makes source == destination otherwise).
    """
    from protomotions.launch.training import save_configs
    from protomotions.launch.training_cli import parse_training_args
    from protomotions.utils.config_builder import (
        build_experiment,
        load_experiment_module,
    )

    with redirect_stdout(io.StringIO()):
        configs = build_experiment(argv)
        experiment = load_experiment_module(experiment_path)
        args = parse_training_args(argv, experiment)
        save_configs(
            out_dir,
            args,
            configs,
            experiment_source_path=experiment_path,
            portable=True,
        )


def regenerate_configs(spec: BundleSpec, repo_root: Path) -> Path:
    """Rebuild the self-contained v2 bundle in the directory.

    ``config.yaml`` records the directory's OWN ``experiment_config.py`` as the
    experiment path, so the bundle reproduces from its own files and does not drift
    with the public recipe. The saved v1 ``experiment_config.py`` imports the removed
    ``envs.component_factories``, so the current recipe is copied in first; the
    launcher's save then runs in a scratch directory (so its experiment-source copy
    does not hit ``SameFileError`` against the self-referential path) and the three
    written files are moved into the directory.
    """
    import tempfile

    directory = repo_root / spec.directory
    directory.mkdir(parents=True, exist_ok=True)
    dir_recipe_rel = f"{spec.directory}/experiment_config.py"
    # The directory's experiment_config.py becomes the current public recipe, with
    # any recorded divergences applied (a find string absent means the recipe moved
    # out from under the divergence, so fail loudly rather than ship the wrong config).
    dir_recipe = repo_root / dir_recipe_rel
    shutil.copy(repo_root / spec.recipe, dir_recipe)
    if spec.recipe_substitutions:
        text = dir_recipe.read_text()
        for find, replace in spec.recipe_substitutions:
            if find not in text:
                raise ValueError(
                    f"{spec.directory}: recipe substitution target not found in "
                    f"{spec.recipe}; the public recipe has drifted:\n{find!r}"
                )
            text = text.replace(find, replace)
        dir_recipe.write_text(text)
    argv = _argv(spec, dir_recipe_rel)
    scratch = Path(tempfile.mkdtemp())
    _save_bundle(argv, dir_recipe_rel, scratch)
    for name in ("config.yaml", "resolved_configs.json", "experiment_config.py"):
        shutil.move(str(scratch / name), str(directory / name))
    return directory / "resolved_configs.json"


def rebuild_bundle_into(directory: Path, scratch: Path) -> None:
    """Rebuild the bundle from the directory's ``config.yaml`` into ``scratch``.

    The idempotence check: ``scratch``'s ``config.yaml`` / ``resolved_configs.json`` /
    ``experiment_config.py`` must be byte-identical to the directory's. Reads only the
    argv fields ``config.yaml`` records; the self-referential experiment path means the
    build loads the directory's own ``experiment_config.py``.
    """
    config = json.loads((directory / "config.yaml").read_text())
    _save_bundle(_config_to_argv(config), config["experiment_path"], scratch)


def to_inference_checkpoint(full_checkpoint: dict, drop_discriminator: bool) -> dict:
    """The model-only inference checkpoint from a full training checkpoint.

    Keeps the model (optionally without the AMP discriminator) and the light
    metadata ``BaseAgent.get_inference_state_dict`` keeps; drops every optimizer and
    training-state entry. Tensors are detached, moved to CPU and cloned, as the
    canonical writer does.
    """
    import torch

    model = full_checkpoint["model"]
    if drop_discriminator:
        model = {
            key: value
            for key, value in model.items()
            if not any(p in key for p in _DISCRIMINATOR_PREFIXES)
        }
    inference = {
        "model": {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in model.items()
        }
    }
    for key in _INFERENCE_META_KEYS:
        if key in full_checkpoint:
            inference[key] = full_checkpoint[key]
    return inference


def _slim_checkpoint(
    directory: Path, source: str, out_name: str, drop_discriminator: bool
) -> Path:
    """Slim one full checkpoint in ``directory`` to its model-only inference form."""
    import torch

    full = torch.load(directory / source, map_location="cpu", weights_only=False)
    inference = to_inference_checkpoint(full, drop_discriminator)
    out = directory / out_name
    torch.save(inference, out)
    return out


def write_inference_checkpoint(spec: BundleSpec, repo_root: Path) -> Path:
    """Slim the shipped ``last.ckpt`` (and any secondary checkpoint) to its model-only
    inference form and return the primary ``inference_last.ckpt`` path.

    A secondary checkpoint is the same model trained differently (soma-bones' IsaacLab
    variant); it shares the directory's model, so the same discriminator policy applies.
    """
    directory = repo_root / spec.directory
    primary = _slim_checkpoint(
        directory, "last.ckpt", "inference_last.ckpt", spec.drop_discriminator
    )
    for secondary in spec.secondary_checkpoints:
        _slim_checkpoint(
            directory,
            secondary.source,
            secondary.inference_name,
            spec.drop_discriminator,
        )
    return primary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_provenance(
    directory: Path,
    name: str,
    robot: str,
    inference_name: str,
    provenance_file: str,
    trained_on: Optional[str] = None,
) -> Path:
    """Write one flat provenance file for one inference checkpoint.

    ``trained_on`` is written only when set; one checkpoint per file, no list form.
    """
    import torch

    checkpoint = directory / inference_name
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    provenance = {
        "name": name,
        "robot": robot,
        "inference_checkpoint": inference_name,
        "inference_checkpoint_sha256": _sha256(checkpoint),
        "inference_checkpoint_bytes": checkpoint.stat().st_size,
        "model_tensor_count": len(model),
        "resumable": False,
    }
    if trained_on is not None:
        provenance["trained_on"] = trained_on
    out = directory / provenance_file
    out.write_text(json.dumps(provenance, indent=2) + "\n")
    return out


def write_provenance(spec: BundleSpec, repo_root: Path) -> Path:
    """Write provenance.json for the primary checkpoint and provenance_<...>.json for
    each secondary one; return the primary provenance path.

    The primary omits ``trained_on`` (its training simulator cannot be recovered); a
    secondary writes its own, paired to its inference file by the suffix.
    """
    directory = repo_root / spec.directory
    primary = _write_provenance(
        directory,
        spec.provenance_name,
        spec.robot,
        "inference_last.ckpt",
        "provenance.json",
    )
    for secondary in spec.secondary_checkpoints:
        _write_provenance(
            directory,
            spec.provenance_name,
            spec.robot,
            secondary.inference_name,
            secondary.provenance_file,
            secondary.trained_on,
        )
    return primary


def remove_stale_files(spec: BundleSpec, repo_root: Path) -> List[Path]:
    """Delete the v1 artifacts the v2 layout replaces (including each secondary
    checkpoint's full source file); return what was removed."""
    directory = repo_root / spec.directory
    removed = []
    stale = list(_STALE_FILES) + [s.source for s in spec.secondary_checkpoints]
    for name in stale:
        path = directory / name
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def regenerate(spec: BundleSpec, repo_root: Path, *, remove_stale: bool = True) -> None:
    """Regenerate one directory end to end: bundle, inference checkpoint, provenance."""
    write_inference_checkpoint(spec, repo_root)
    regenerate_configs(spec, repo_root)
    write_provenance(spec, repo_root)
    if remove_stale:
        remove_stale_files(spec, repo_root)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        action="append",
        help="Regenerate only this directory (repeatable); default is all specs.",
    )
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help="Do not delete the v1 last.ckpt / resolved_configs.pt / .yaml files.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    specs = SPECS
    if args.directory:
        wanted = set(args.directory)
        specs = [
            s
            for s in SPECS
            if s.directory in wanted or Path(s.directory).name in wanted
        ]
    for spec in specs:
        print(f"Regenerating {spec.directory} from {spec.recipe}")
        regenerate(spec, repo_root, remove_stale=not args.keep_stale)
        print("  wrote resolved_configs.json, config.yaml, experiment_config.py,")
        print("  inference_last.ckpt, provenance.json")
        for secondary in spec.secondary_checkpoints:
            print(f"  wrote {secondary.inference_name}, {secondary.provenance_file}")


if __name__ == "__main__":
    main()
