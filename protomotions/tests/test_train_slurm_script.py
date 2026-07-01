# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Script-level tests for the public train_slurm.py template."""

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TRAIN_SLURM_PATH = str(Path(__file__).resolve().parents[1] / "train_slurm.py")


def _base_argv(experiment_path):
    return [
        "train_slurm.py",
        "--robot-name",
        "g1",
        "--simulator",
        "isaacgym",
        "--num-envs",
        "16",
        "--batch-size",
        "32",
        "--motion-file",
        "motions.pt",
        "--experiment-path",
        str(experiment_path),
        "--experiment-name",
        "unit",
        "--user",
        "remote",
    ]


def test_train_slurm_upload_only_syncs_code_and_exits(
    tmp_path,
    monkeypatch,
    capsys,
):
    experiment = tmp_path / "experiment.py"
    experiment.write_text("# config\n")
    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        _base_argv(experiment) + ["--only-upload-code"],
    )

    runpy.run_path(TRAIN_SLURM_PATH, run_name="__main__")

    assert len(calls) == 2
    assert calls[0][0].startswith("ssh remote@YOUR_CLUSTER_LOGIN_NODE")
    assert calls[1][0].startswith("rsync -az")
    assert "Code uploaded. Exiting" in capsys.readouterr().out


def test_train_slurm_submission_builds_sbatch_script_and_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    experiment = tmp_path / "agent_experiment.py"
    experiment.write_text("# config\n")
    calls = []

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 12345\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        _base_argv(experiment)
        + [
            "--scenes-file",
            "scenes.yaml",
            "--ngpu",
            "2",
            "--nodes",
            "1",
            "--checkpoint",
            "last.ckpt",
            "--overrides",
            "env.max_episode_length=42",
        ],
    )

    runpy.run_path(TRAIN_SLURM_PATH, run_name="__main__")

    output = capsys.readouterr().out
    script_files = list((tmp_path / "tmp").glob("slurm_*.sh"))
    assert len(script_files) == 1
    script = script_files[0].read_text()
    assert "--container-image=/path/to/containers/isaacgym.sqsh" in script
    assert "protomotions/train_agent.py" in script
    assert "--num-envs=16" in script
    assert "--batch-size=32" in script
    assert "--scenes-file=scenes.yaml" in script
    assert "--checkpoint=last.ckpt" in script
    assert "--overrides env.max_episode_length=42" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=2" in script
    assert "JOB SUBMITTED" in output
    assert any(cmd.startswith("scp ") for cmd, _ in calls)
    assert any("sbatch" in cmd for cmd, _ in calls)


def test_train_slurm_subprocess_run_raises_on_error(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment.py"
    experiment.write_text("# config\n")

    def _run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        _base_argv(experiment) + ["--only-upload-code"],
    )

    with pytest.raises(Exception, match="Command failed"):
        runpy.run_path(TRAIN_SLURM_PATH, run_name="__main__")


def test_train_slurm_wandb_local_key_is_passed_when_remote_missing(
    tmp_path,
    monkeypatch,
):
    experiment = tmp_path / "wandb_experiment.py"
    experiment.write_text("# config\n")

    def _run(cmd, **kwargs):
        if "grep -q 'machine api.wandb.ai'" in cmd or "test -n" in cmd:
            return SimpleNamespace(returncode=0, stdout="not_found\n")
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 333\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "example-wandb-key")
    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        _base_argv(experiment) + ["--use-wandb"],
    )

    runpy.run_path(TRAIN_SLURM_PATH, run_name="__main__")

    script = next((tmp_path / "tmp").glob("slurm_*.sh")).read_text()
    assert "WANDB_API_KEY=example-wandb-key" in script
    assert "--use-wandb" in script


def test_train_slurm_wandb_remote_netrc_skips_api_key(
    tmp_path,
    monkeypatch,
    capsys,
):
    experiment = tmp_path / "wandb_experiment.py"
    experiment.write_text("# config\n")

    def _run(cmd, **kwargs):
        if "grep -q 'machine api.wandb.ai'" in cmd:
            return SimpleNamespace(returncode=0, stdout="found\n")
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 444\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        _base_argv(experiment) + ["--use-wandb"],
    )

    runpy.run_path(TRAIN_SLURM_PATH, run_name="__main__")

    script = next((tmp_path / "tmp").glob("slurm_*.sh")).read_text()
    assert "WANDB_API_KEY=" not in script
    assert "WANDB credentials found in ~/.netrc" in capsys.readouterr().out
