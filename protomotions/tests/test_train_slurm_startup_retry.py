# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wiring tests for the SLURM launcher's opt-in startup retry.

The retry LOGIC itself lives in scripts/cluster/retry_lib.sh and is exercised by
test_cluster_retry_lib.py. These tests only check that train_slurm.py wires it up
correctly: with the feature off the emitted script is byte-identical to the
no-retry path, and with it on the batch layer sources the shared lib, wraps srun in
run_with_retry, and exports the RETRY_* knobs (break-on-progress, the checkpoint
path, the exception fast-fail limit). They load train_slurm.py directly the same
way test_train_slurm_site.py does -- no SSH, no cluster."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


TRAIN_SLURM_PATH = Path(__file__).resolve().parents[1] / "train_slurm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_slurm_startup_under_test", TRAIN_SLURM_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = _load_module()


# --- the acceptance gate: default flags emit a byte-identical script ---------

GOLDEN_DEFAULT_SCRIPT = """#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --account=acct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=/exp/exp-TS/slurm_output.log
#SBATCH --error=/exp/exp-TS/slurm_output.log
#SBATCH --array=0-5%1

# Job array enables automatic resume: if job times out, next array task continues

stdbuf -oL -eL srun --container-image=/img.sqsh --no-container-mount-home --container-mounts=/scratch:/scratch:rw /bin/bash -c 'JOB_CMD_PLACEHOLDER'
"""


def _slurm_args(**over):
    base = dict(
        account="acct",
        partition="gpu",
        nodes=1,
        ngpu=8,
        slurm_time="4:00:00",
        array_size=5,
        use_wandb=False,
        experiment_name="myexp",
        startup_max_attempts=1,
        startup_repeat_limit=3,
        startup_retry_backoff_sec=15,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_default_flags_emit_byte_identical_script():
    script, _ = ts.generate_slurm_script(
        _slurm_args(), "/exp/exp-TS", "JOB_CMD_PLACEHOLDER", "/img.sqsh", "jobname"
    )
    assert "run_with_retry" not in script
    assert "retry_lib.sh" not in script
    assert script == GOLDEN_DEFAULT_SCRIPT


def test_site_default_startup_knobs_keep_the_feature_off():
    # The byte-identical guarantee holds only if the RESOLVED default is off. The
    # golden test above passes hardcoded args; guard the SITE_DEFAULTS value too so a
    # change that flips the shipped default (e.g. startup_max_attempts: 2) -- which
    # would wrap srun on a plain submission -- is caught here. startup_stall_min is
    # gone (one watchdog now), so assert it is not silently reintroduced.
    assert ts.SITE_DEFAULTS["startup_max_attempts"] == 1
    assert "startup_stall_min" not in ts.SITE_DEFAULTS


def test_feature_on_sources_the_lib_wraps_srun_and_exports_the_knobs():
    script, _ = ts.generate_slurm_script(
        _slurm_args(startup_max_attempts=8, startup_repeat_limit=5, startup_retry_backoff_sec=20),
        "/exp/exp-TS",
        "JOB_CMD_PLACEHOLDER",
        "/img.sqsh",
        "jobname",
    )
    # The batch layer sources the SHARED lib (not an inlined copy) and wraps srun.
    assert "source /exp/exp-TS/scripts/cluster/retry_lib.sh" in script
    assert "run_with_retry stdbuf -oL -eL srun" in script
    # SLURM hands off to the array on progress rather than resuming in place.
    assert "RETRY_ON_PROGRESS=break" in script
    # The checkpoint the loop watches for progress is results/<experiment_name>.
    assert "CKPT=/exp/exp-TS/results/myexp/last.ckpt" in script
    assert "RETRY_MAX_ATTEMPTS=8" in script
    assert "RETRY_REPEAT_LIMIT=5" in script
    assert "RETRY_BACKOFF_SEC=20" in script


def test_build_job_command_sources_the_lib_and_wraps_the_watchdog():
    # The always-on inner watchdog is the ONE watchdog: build_job_command sources
    # retry_lib.sh (after its cd into the snapshot) and runs training under
    # run_with_stall_watchdog. No startup-only watchdog is emitted anywhere.
    args = _slurm_args(
        simulator="newton",
        robot_name="g1",
        motion_file="/m.pt",
        num_envs=4096,
        batch_size=32768,
        experiment_path="examples/experiments/mimic/mlp.py",
        training_max_iterations=None,
        training_max_steps=100,
        scenes_file=None,
        checkpoint=None,
        overrides=[],
        stall_timeout_min=30,
        seed=0,
        torch_deterministic=False,
        user="tester",
        wandb_project="p",
    )
    job_cmd = ts.build_job_command(args, "/exp/exp-TS", "python")
    assert "source /exp/exp-TS/scripts/cluster/retry_lib.sh" in job_cmd
    assert "run_with_stall_watchdog python -u protomotions/train_agent.py" in job_cmd
    assert "STALL_TIMEOUT_MIN=30" in job_cmd
