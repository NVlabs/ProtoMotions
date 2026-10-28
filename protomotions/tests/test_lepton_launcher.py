# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for train_lepton.py command construction and guards.

These cover the parts a reviewer flagged as untested: the train_agent.py argument
tail is preserved as real argv (not re-split), the container command wraps the
resume loop and passes the stall timeout, and --nodes > 1 is refused rather than
warned. They import the launcher module directly (no `lep`, no cluster) the same
way test_train_slurm_site.py loads train_slurm.py."""

import importlib.util
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest


TRAIN_LEPTON_PATH = Path(__file__).resolve().parents[1] / "train_lepton.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_lepton_under_test", TRAIN_LEPTON_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tl = _load_module()


def _args(**over):
    """A minimal args namespace for the command builders."""
    base = dict(
        robot_name="g1",
        simulator="isaaclab",
        motion_file="data/motion_for_trackers/m.pt",
        ngpu=4,
        nodes=1,
        experiment_name="my_run",
        experiment_path="examples/experiments/mimic/mlp.py",
        num_envs=4096,
        batch_size=32768,
        training_max_iterations=None,
        training_max_steps=100,
        scenes_file=None,
        use_wandb=False,
        wandb_project="physical_animation",
        checkpoint=None,
        overrides=[],
        max_runs=75,
        # build_lep_job_create fields
        image=None,
        resource_shape=None,
        node_group=None,
        registry_secret=None,
        stall_timeout_min=30,
        no_progress_limit=3,
        repeat_limit=3,
        shared_memory_mib=65536,
        preemptible=False,
        max_job_retries=10,
        ttl_seconds=259200,
        queue_priority=4,
        visibility="private",
        user="tester",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _site(**over):
    base = dict(
        container_images={"isaaclab": "nvcr.io/x/isaaclab:0.1"},
        node_group="ng-1",
        registry_secret="reg-secret",
        workspace_name="ws-name",
        storage_name="amlfs",
    )
    base.update(over)
    return base


# --- parse_launcher_args: own flags parsed, training tail forwarded -----------


def test_parse_launcher_args_forwards_the_training_tail_and_reads_facts():
    # The launcher declares only its own flags, so the whole training command --
    # minus --user, which is its own -- forwards verbatim, and the few values the
    # submission needs are read off that tail.
    tail = [
        "--robot-name=g1",
        "--simulator=isaaclab",
        "--num-envs=64",
        "--batch-size=256",
        "--motion-file=data/m.pt",
        "--experiment-path=examples/x.py",
        "--experiment-name=run7",
        "--ngpu=4",
        "--nodes=2",
        "--use-wandb",
        "--seed",
        "3",
    ]
    args, passthrough = tl.parse_launcher_args(["--user=me", *tail])
    assert "--user=me" not in passthrough
    assert passthrough == tail
    assert args.simulator == "isaaclab"
    assert args.experiment_name == "run7"
    assert args.ngpu == 4 and args.nodes == 2
    assert args.use_wandb is True


# --- build_container_command: lossless round-trip, wraps the loop -------------


def test_container_command_round_trips_a_spaced_argument():
    # The container shell splits the command string back into argv. shlex.split
    # models that split; a value with a space must survive as one token, the
    # interpreter "cmd -flag" must stay a single positional, and the forwarded
    # training tail must come through verbatim after the five loop positionals.
    forwarded = ["--robot-name=g1", "--note", "hello world", "--overrides", "a=b"]
    cmd = tl.build_container_command(
        "/workspace/isaaclab/isaaclab.sh -p",
        _args(),
        "/mnt/exp-1",
        passthrough=forwarded,
    )
    body = cmd.split("lepton_train_loop.sh ", 1)[1]
    toks = shlex.split(body)
    assert toks[0] == "/workspace/isaaclab/isaaclab.sh -p"  # one positional
    assert toks[1:5] == ["isaaclab", "my_run", "/mnt/exp-1", "75"]
    # The forwarded tail follows, verbatim and lossless (the spaced value intact).
    assert toks[5:] == forwarded
    assert "hello world" in toks


def test_container_command_wraps_the_resume_loop_and_train_agent():
    forwarded = [
        "--experiment-path=examples/experiments/mimic/mlp.py",
        "--num-envs=4096",
    ]
    cmd = tl.build_container_command(
        "python", _args(), "/mnt/exp-2", passthrough=forwarded
    )
    assert "bash scripts/cluster/lepton_train_loop.sh" in cmd
    assert cmd.startswith("set -euo pipefail; cd ")
    body = cmd.split("lepton_train_loop.sh ", 1)[1]
    toks = shlex.split(body)
    # positionals 1..5 then the forwarded train_agent.py tail, which the loop runs.
    assert toks[:5] == ["python", "isaaclab", "my_run", "/mnt/exp-2", "75"]
    assert toks[5:] == forwarded


# --- build_lep_job_create: the stall timeout is actually passed ---------------


def test_lep_job_create_passes_stall_timeout_env():
    cmd = tl.build_lep_job_create(
        _args(stall_timeout_min=45), _site(), "/mnt/exp", "true"
    )
    assert "--env" in cmd and "STALL_TIMEOUT_MIN=45" in cmd
    # the container command is the last --command value
    assert cmd[cmd.index("--command") + 1] == "true"


def test_lep_job_create_forwards_disabled_stall_timeout():
    cmd = tl.build_lep_job_create(
        _args(stall_timeout_min=0), _site(), "/mnt/exp", "true"
    )
    assert "STALL_TIMEOUT_MIN=0" in cmd


def test_lep_job_create_sets_expandable_segments_alloc_env():
    # A launcher default: expandable_segments cuts CUDA allocator fragmentation.
    # It changes no result but rescued the fragmentation-OOM arm on these runs.
    cmd = tl.build_lep_job_create(_args(), _site(), "/mnt/exp", "true")
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in cmd


def test_lep_job_create_passes_no_progress_limit_env():
    # lepton_train_loop.sh maps --no-progress-limit onto RETRY_MAX_ATTEMPTS (the
    # shared run_with_retry knob); the launcher must let a flaky scene build raise it
    # above the default 3, or three ordinary aborts kill the job.
    cmd = tl.build_lep_job_create(
        _args(no_progress_limit=8), _site(), "/mnt/exp", "true"
    )
    assert "RETRY_MAX_ATTEMPTS=8" in cmd


def test_lep_job_create_passes_repeat_limit_env():
    # --repeat-limit feeds RETRY_REPEAT_LIMIT: the fast-fail-on-python threshold the
    # Lepton loop previously lacked. The launcher must forward it.
    cmd = tl.build_lep_job_create(
        _args(repeat_limit=5), _site(), "/mnt/exp", "true"
    )
    assert "RETRY_REPEAT_LIMIT=5" in cmd


def test_lep_job_create_missing_image_fails_cleanly():
    with pytest.raises(SystemExit):
        tl.build_lep_job_create(
            _args(simulator="newton", image=None),
            _site(container_images={"isaaclab": "x"}),  # no newton image
            "/mnt/exp",
            "true",
        )


# --- guards ------------------------------------------------------------------


def test_multinode_is_refused_not_warned():
    with pytest.raises(SystemExit):
        tl.reject_unqualified_multinode(2)
    # single node is fine
    tl.reject_unqualified_multinode(1)


def test_lepton_job_name_sanitizes_to_rfc1123():
    assert tl._lepton_job_name("lepton_smoke") == "lepton-smoke"
    assert tl._lepton_job_name("__weird__Name__") == "weird-name"
    # never empty
    assert tl._lepton_job_name("___") == "job"


def test_unconfigured_values_flags_placeholders():
    # The shipped defaults are all placeholders -> everything is unset.
    assert tl.unconfigured_values(dict(tl.SITE_DEFAULTS))
    # A fully-real site reports nothing unset.
    real = {
        "workspace_id": "u5",
        "workspace_name": "ws",
        "node_group": "ng",
        "storage_name": "amlfs",
        "base_dir": "/mnt/users/u5/exp",
        "teleport_cluster": "tp",
        "pod_name": "pod",
        "registry_secret": "secret",
        "container_images": {"isaaclab": "nvcr.io/x:1"},
        "python_executables": {"isaaclab": "python"},
    }
    assert tl.unconfigured_values(real) == []
