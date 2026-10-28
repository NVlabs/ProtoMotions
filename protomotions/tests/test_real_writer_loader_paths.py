# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundle entry paths on real files, nothing faked (open item 5).

Elsewhere the writer and the loaders are stood in for: ``train_agent``'s
``save_configs`` and ``save_training_configs`` in ``test_train_agent_helpers``,
``load_experiment_configs_from_checkpoint`` in the inference-script, pretrained-
module and teacher-reading tests, ``load_pretrained_model_module`` in the PEFT
helper tests. Those doubles isolate the code around the files; this module is
the one place each entry path runs end to end on files the real writer wrote:

* ``--create-config-only``: ``_handle_create_config_only`` on a real recipe's
  configs writes ``config.yaml``, ``resolved_configs.json`` and the recipe copy,
  and dry-runs the inference derivation with the copied recipe's hook;
* inference: ``load_experiment_configs_from_checkpoint`` beside a checkpoint,
  ``recipe_hooks`` from the copied recipe, ``derive_inference_configs``, as
  ``launch/inference.py`` runs them;
* a PEFT agent's prior: ``_load_pretrained_modules`` through the real
  ``load_pretrained_model_module`` over a bundle beside a saved checkpoint.
"""

import importlib
import inspect
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from protomotions.agents.base_agent.config import BaseModelConfig
from protomotions.agents.common.config import PretrainedModelConfig
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.launch.training import _handle_create_config_only
from protomotions.tests.support.recipe_stages import (
    REPO_ROOT,
    build_experiment_configs,
    fake_args,
)
from protomotions.utils.config_schema import to_tree
from protomotions.utils.experiment_configs import (
    BUNDLE_FILENAME,
    ExperimentConfigs,
    load_experiment_configs,
    save_experiment_configs,
)
from protomotions.utils.inference_utils import derive_inference_configs, recipe_hooks

PACKAGE = "examples.experiments"
RECIPE = "mimic.mlp"  # has an inference hook (tracking_playback) and terminations
RECIPE_PATH = REPO_ROOT / "examples/experiments/mimic/mlp.py"


def _same_tree(a: ExperimentConfigs, b: ExperimentConfigs) -> None:
    for root in ExperimentConfigs.ROOTS:
        assert to_tree(getattr(a, root), where=root) == to_tree(
            getattr(b, root), where=root
        ), root


@pytest.fixture(params=[True, False], ids=["bytecode", "no-bytecode"])
def written(tmp_path, monkeypatch, request):
    """A run directory the real ``--create-config-only`` path wrote."""
    monkeypatch.setattr(sys, "dont_write_bytecode", not request.param)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    module = importlib.import_module(f"{PACKAGE}.{RECIPE}")
    args = fake_args(module)
    args.use_wandb = False
    args.create_config_only = True
    args.experiment_path = str(RECIPE_PATH)
    args.inference_config = None
    configs = build_experiment_configs(RECIPE, PACKAGE)
    save_dir = tmp_path / "config-only"
    _handle_create_config_only(args, save_dir, RECIPE_PATH, configs)
    return save_dir, args, configs


def test_create_config_only_writes_the_bundle_and_dry_runs_the_recipe_hook(written):
    save_dir, args, configs = written
    # The dry-run imports the saved recipe. Python may cache its bytecode;
    # the three persisted configuration artifacts are unchanged.
    expected_entries = [
        "config.yaml",
        "experiment_config.py",
        BUNDLE_FILENAME,
    ]
    if not sys.dont_write_bytecode:
        expected_entries.append("__pycache__")
        recipe_cache = Path(
            importlib.util.cache_from_source(str(save_dir / "experiment_config.py"))
        )
        assert list((save_dir / "__pycache__").iterdir()) == [recipe_cache]
    assert sorted(p.name for p in save_dir.iterdir()) == sorted(expected_entries)
    cli = yaml.safe_load((save_dir / "config.yaml").read_text())
    assert cli["experiment_name"] == args.experiment_name and cli["wandb_id"] is None
    assert (save_dir / "experiment_config.py").read_text() == RECIPE_PATH.read_text()
    # The bundle rebuilds through the real constructors to the configs written.
    loaded = load_experiment_configs(save_dir / BUNDLE_FILENAME)
    _same_tree(loaded, configs)
    # The dry-run ran the copied recipe's hook; run the same derivation on the
    # reloaded bundle and look for the hook's work (tracking_playback clears the
    # terminations and starts every clip at its first frame), leaving the
    # training bundle as written.
    hooks = recipe_hooks(save_dir, None)
    assert [h.__name__ for h in hooks] == ["apply_inference_overrides"]
    assert Path(inspect.getfile(hooks[0])) == save_dir / "experiment_config.py"
    view = derive_inference_configs(loaded, args, {}, hooks)
    assert loaded.env.termination_components and view.env.termination_components == {}
    assert loaded.env.motion_manager.init_start_prob == 0.2
    assert view.env.motion_manager.init_start_prob == 1.0
    _same_tree(loaded, load_experiment_configs(save_dir / BUNDLE_FILENAME))


def test_a_run_directory_without_the_recipe_copy_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="create-config-only"):
        recipe_hooks(tmp_path, None)


def test_inference_loads_the_bundle_beside_the_checkpoint_and_derives(
    written, tmp_path
):
    """``launch/inference.py``'s config half on the files training wrote: the
    bundle beside the checkpoint, the hooks from the copied recipe, the view.
    Works on its own copy of the run directory, leaving the original bundle
    unchanged."""
    import protomotions.launch.inference as inference

    save_dir, args, configs = written
    run_dir = tmp_path / "run"
    shutil.copytree(save_dir, run_dir)
    checkpoint = run_dir / "last.ckpt"
    torch.save({"model": {}}, checkpoint)
    training = inference.load_experiment_configs_from_checkpoint(checkpoint)
    _same_tree(training, configs)
    view = derive_inference_configs(
        training, args, {}, recipe_hooks(checkpoint.parent, args.inference_config)
    )
    assert view.env.termination_components == {}
    assert training.env.termination_components  # the derivation copied
    with pytest.raises(FileNotFoundError):
        inference.load_experiment_configs_from_checkpoint(
            run_dir / "elsewhere" / "last.ckpt"
        )


@dataclass
class _TinyPriorConfig(BaseModelConfig):
    _target_: str = "protomotions.tests.test_real_writer_loader_paths._TinyPrior"
    width: int = 3
    inference_defaults_applied: bool = False


@dataclass
class _TinyAgentConfig(PPOAgentConfig):
    """A real agent config (a ``BaseAgentConfig``) around the tiny prior, so the
    loader's agent-defaults branch runs on the loaded bundle; its hook leaves a
    mark the rebuilt module records, which proves the branch ran."""

    def apply_inference_defaults(self, env) -> None:
        self.model.inference_defaults_applied = True


class _TinyPrior(nn.Module):
    def __init__(self, config: _TinyPriorConfig):
        super().__init__()
        self.linear = nn.Linear(config.width, 1)
        self.defaults_applied = config.inference_defaults_applied

    def materialize_from_state_dict(self, state_dict) -> None:  # the loader's contract
        pass

    def forward(self, x):
        return self.linear(x)


def test_peft_prior_loads_through_the_real_bundle_beside_its_checkpoint(tmp_path):
    """``_load_pretrained_modules`` with nothing faked: the prior's bundle is
    written by the real writer beside a checkpoint holding the prior's weights,
    and the real ``load_pretrained_model_module`` rebuilds the module from the
    bundle's ``agent.model`` (after the agent-side inference defaults, which
    a real agent config receives) and loads the weights, frozen and in eval
    mode."""
    from protomotions.components.motion_lib import MotionLibConfig

    prior_dir = tmp_path / "prior"
    prior_dir.mkdir()
    filler = {
        root: MotionLibConfig(motion_file=root) for root in ExperimentConfigs.ROOTS
    }
    filler["agent"] = _TinyAgentConfig(
        model=_TinyPriorConfig(width=3), batch_size=4, training_max_steps=1
    )
    assert not filler["agent"].model.inference_defaults_applied  # as saved
    save_experiment_configs(ExperimentConfigs(**filler), prior_dir)
    trained = _TinyPrior(_TinyPriorConfig(width=3))
    with torch.no_grad():
        trained.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
        trained.linear.bias.fill_(-4.0)
    checkpoint = prior_dir / "last.ckpt"
    torch.save({"model": trained.state_dict()}, checkpoint)

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.fabric = SimpleNamespace(device=torch.device("cpu"))
    agent.config = SimpleNamespace(
        pretrained_modules={
            "prior": PretrainedModelConfig(
                checkpoint_path=str(checkpoint), module_path=""
            )
        }
    )
    modules = DiscretePriorPEFTRLFTAgent._load_pretrained_modules(agent)
    prior = modules["prior"]
    assert isinstance(prior, _TinyPrior)
    assert prior.defaults_applied  # the loader ran apply_inference_defaults
    assert torch.equal(prior.linear.weight, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(prior.linear.bias, torch.tensor([-4.0]))
    assert not prior.training and not any(p.requires_grad for p in prior.parameters())
