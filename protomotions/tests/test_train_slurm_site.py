# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for train_slurm.py site-config resolution, including multi-cluster
selection: a site file may list several clusters that share one allocation, and
--cluster (or default_cluster) picks one."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


TRAIN_SLURM_PATH = Path(__file__).resolve().parents[1] / "train_slurm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_slurm_under_test", TRAIN_SLURM_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = _load_module()


def _write(tmp_path, text):
    path = tmp_path / "site.yaml"
    path.write_text(text)
    return str(path)


FLAT_SITE = """
login_node: login.example.edu
base_dir: /scratch/{account}/exps
account: my_alloc
partition: gpu
container_mounts: /scratch:/scratch:rw
container_images:
  isaacgym: /containers/isaacgym.sqsh
  isaaclab: /containers/isaaclab.sqsh
  newton: /containers/newton.sqsh
"""

TWO_CLUSTERS = """
account: my_alloc
base_dir: /scratch/{account}/exps
container_mounts: /scratch:/scratch:rw
default_cluster: a100
clusters:
  a100:
    login_node: a100-login.example.edu
    partition: gpu
    container_images:
      isaacgym: /containers/isaacgym.sqsh
      isaaclab: /containers/isaaclab.sqsh
    python_executables:
      isaaclab: /workspace/isaaclab/isaaclab.sh -p
  l40:
    login_node: l40-login.example.edu
    partition: batch
    container_images:
      isaaclab: /containers/isaaclab.sqsh
    python_executables:
      isaaclab: /workspace/isaaclab/isaaclab.sh -p
"""


def test_flat_site_has_no_cluster_and_no_placeholders(tmp_path):
    site = ts.load_site_config(_write(tmp_path, FLAT_SITE))
    assert site["_cluster"] is None
    assert site["_clusters_available"] == []
    assert site["login_node"] == "login.example.edu"
    assert site["account"] == "my_alloc"
    assert ts.unconfigured_values(site) == []


def test_flat_site_rejects_cluster_argument(tmp_path):
    with pytest.raises(ValueError, match="defines no 'clusters'"):
        ts.load_site_config(_write(tmp_path, FLAT_SITE), cluster="a100")


def test_default_cluster_used_when_none_requested(tmp_path):
    site = ts.load_site_config(_write(tmp_path, TWO_CLUSTERS))
    assert site["_cluster"] == "a100"
    assert site["_clusters_available"] == ["a100", "l40"]
    assert site["login_node"] == "a100-login.example.edu"
    assert site["partition"] == "gpu"
    # Shared top-level account applies whichever cluster is chosen.
    assert site["account"] == "my_alloc"
    assert site["base_dir"] == "/scratch/{account}/exps"
    assert ts.unconfigured_values(site) == []


def test_explicit_cluster_overrides_default(tmp_path):
    site = ts.load_site_config(_write(tmp_path, TWO_CLUSTERS), cluster="l40")
    assert site["_cluster"] == "l40"
    assert site["login_node"] == "l40-login.example.edu"
    assert site["partition"] == "batch"
    assert site["account"] == "my_alloc"


def test_cluster_container_images_replace_not_merge(tmp_path):
    """A cluster listing only isaaclab must not inherit the other simulators'
    placeholder images -- otherwise an isaaclab-only submission is wrongly
    refused as unconfigured."""
    site = ts.load_site_config(_write(tmp_path, TWO_CLUSTERS), cluster="l40")
    assert set(site["container_images"]) == {"isaaclab"}
    assert "isaacgym" not in site["container_images"]
    assert ts.unconfigured_values(site) == []


def test_unknown_cluster_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown cluster 'tpu'"):
        ts.load_site_config(_write(tmp_path, TWO_CLUSTERS), cluster="tpu")


def test_multiple_clusters_without_selection_is_rejected(tmp_path):
    text = TWO_CLUSTERS.replace("default_cluster: a100\n", "")
    with pytest.raises(ValueError, match="none was selected"):
        ts.load_site_config(_write(tmp_path, text))


def test_single_cluster_is_auto_selected(tmp_path):
    text = """
account: my_alloc
base_dir: /scratch/{account}/exps
container_mounts: /scratch:/scratch:rw
clusters:
  only:
    login_node: only-login.example.edu
    partition: gpu
    container_images:
      isaaclab: /containers/isaaclab.sqsh
"""
    site = ts.load_site_config(_write(tmp_path, text))
    assert site["_cluster"] == "only"
    assert site["login_node"] == "only-login.example.edu"


def test_cluster_may_override_shared_account(tmp_path):
    text = TWO_CLUSTERS.replace(
        "    partition: batch\n",
        "    partition: batch\n    account: other_alloc\n",
    )
    site = ts.load_site_config(_write(tmp_path, text), cluster="l40")
    assert site["account"] == "other_alloc"
    # The unchanged cluster keeps the shared account.
    a100 = ts.load_site_config(_write(tmp_path, text), cluster="a100")
    assert a100["account"] == "my_alloc"


def test_unknown_top_level_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        ts.load_site_config(_write(tmp_path, FLAT_SITE + "nonsense: 1\n"))


def test_unknown_cluster_key_is_rejected(tmp_path):
    text = TWO_CLUSTERS.replace(
        "    partition: batch\n",
        "    partition: batch\n    bogus: 1\n",
    )
    with pytest.raises(ValueError, match="unknown key"):
        ts.load_site_config(_write(tmp_path, text), cluster="l40")


def test_default_cluster_without_clusters_is_rejected(tmp_path):
    text = FLAT_SITE + "default_cluster: a100\n"
    with pytest.raises(ValueError, match="no 'clusters'"):
        ts.load_site_config(_write(tmp_path, text))


# --- main() level: the fixes that a bad submission must fail before any I/O ---


def _run_main(monkeypatch, tmp_path, site_path, extra_argv):
    """Run train_slurm.main() with subprocess faked; return the recorded calls."""
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 1\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ts.subprocess, "run", _run)
    argv = [
        "train_slurm.py",
        "--robot-name",
        "g1",
        "--motion-file",
        "motions.pt",
        "--experiment-path",
        "exp.py",
        "--experiment-name",
        "unit",
        "--user",
        "remote",
        "--num-envs",
        "8",
        "--batch-size",
        "32",
        "--site-config",
        site_path,
        *extra_argv,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    ts.main()
    return calls


def test_unsupported_simulator_fails_before_any_rsync(tmp_path, monkeypatch):
    """cs004-style cluster stages only isaaclab; asking for newton must raise
    before sync_code_to_cluster does its ssh/rsync."""
    site = _write(tmp_path, TWO_CLUSTERS)  # l40 has only the isaaclab image
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ts.subprocess, "run", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_slurm.py",
            "--robot-name",
            "g1",
            "--motion-file",
            "motions.pt",
            "--experiment-path",
            "exp.py",
            "--experiment-name",
            "unit",
            "--user",
            "remote",
            "--num-envs",
            "8",
            "--batch-size",
            "32",
            "--site-config",
            site,
            "--cluster",
            "l40",
            "--simulator",
            "newton",
        ],
    )
    with pytest.raises(ValueError, match="No container configured for simulator"):
        ts.main()
    assert calls == []  # nothing was synced


def test_cli_account_overrides_missing_site_account(tmp_path, monkeypatch):
    """A site file that omits the shared top-level account must still submit
    when --account supplies a real one, rather than being refused as a
    placeholder."""
    no_account = TWO_CLUSTERS.replace("account: my_alloc\n", "")
    calls = _run_main(
        monkeypatch,
        tmp_path,
        _write(tmp_path, no_account),
        ["--cluster", "a100", "--simulator", "isaaclab", "--account", "real_alloc"],
    )
    assert any("sbatch" in c for c in calls)


def test_startup_site_keys_are_accepted_and_flag_beats_site(tmp_path, monkeypatch, capsys):
    # A site file may set the three startup knobs; the loader (which rejects unknown
    # keys) must accept them because they are in SITE_DEFAULTS.
    site_on = FLAT_SITE + (
        "startup_max_attempts: 8\n"
        "startup_repeat_limit: 4\n"
        "startup_retry_backoff_sec: 5\n"
    )
    loaded = ts.load_site_config(_write(tmp_path, site_on))
    assert loaded["startup_max_attempts"] == 8
    assert loaded["startup_repeat_limit"] == 4

    # With no flag, the site value turns the feature on: the emitted script wraps
    # srun in the shared run_with_retry and the worst-case line is printed.
    _run_main(monkeypatch, tmp_path, _write(tmp_path, site_on), ["--simulator", "isaaclab"])
    out = capsys.readouterr().out
    assert "up to 8 attempts" in out

    # An explicit flag beats the site value: turning it off leaves the default,
    # byte-identical (unwrapped) script -- no worst-case line.
    _run_main(
        monkeypatch,
        tmp_path,
        _write(tmp_path, site_on),
        ["--simulator", "isaaclab", "--startup-max-attempts", "1"],
    )
    out = capsys.readouterr().out
    assert "up to" not in out


# ---------------------------------------------------------------------------
# Regression tests: capabilities the site-config rewrite dropped, which made
# GPC-prior / distillation runs unsubmittable and silently changed seed/array.
# ---------------------------------------------------------------------------


def _agent_argv():
    return [
        "--robot-name=g1",
        "--simulator=isaaclab",
        "--num-envs=64",
        "--batch-size=256",
        "--motion-file=data/m.pt",
        "--experiment-path=examples/x.py",
        "--experiment-name=e",
        "--user=me",
    ]


def test_unknown_experiment_flags_are_forwarded():
    # gpc/prior.py contributes --tracker-checkpoint via
    # additional_experiment_arguments; train_agent.py parses it, so the launcher
    # must not reject it -- it must forward it verbatim.
    args, passthrough = ts.create_parser().parse_known_args(
        _agent_argv() + ["--tracker-checkpoint", "/tmp/t.ckpt"]
    )
    assert passthrough == ["--tracker-checkpoint", "/tmp/t.ckpt"]
    cmd = ts.build_job_command(args, "/exp", "python", passthrough)
    assert "--tracker-checkpoint /tmp/t.ckpt" in cmd


def test_overrides_stay_after_passthrough():
    # train_agent's --overrides is nargs="*" and greedily consumes the rest, so
    # passthrough flags must precede it or be swallowed as override tokens.
    args, passthrough = ts.create_parser().parse_known_args(
        _agent_argv() + ["--tracker-checkpoint", "/tmp/t.ckpt", "--overrides", "a=1"]
    )
    cmd = ts.build_job_command(args, "/exp", "python", passthrough)
    assert cmd.index("--overrides") > cmd.index("--tracker-checkpoint")


def test_seed_is_forwarded():
    args, passthrough = ts.create_parser().parse_known_args(
        _agent_argv() + ["--seed", "3"]
    )
    cmd = ts.build_job_command(args, "/exp", "python", passthrough)
    assert "--seed=3 " in cmd


def test_torch_deterministic_is_forwarded():
    args, passthrough = ts.create_parser().parse_known_args(
        _agent_argv() + ["--torch-deterministic"]
    )
    cmd = ts.build_job_command(args, "/exp", "python", passthrough)
    assert "--torch-deterministic " in cmd


def test_srun_repeat_is_an_alias_for_array_size():
    args, _ = ts.create_parser().parse_known_args(
        _agent_argv() + ["--srun-repeat", "50"]
    )
    assert args.array_size == 50
    args2, _ = ts.create_parser().parse_known_args(
        _agent_argv() + ["--array-size", "42"]
    )
    assert args2.array_size == 42
    # Neither given -> None, so main() can fall back to the site's array_size.
    args3, _ = ts.create_parser().parse_known_args(_agent_argv())
    assert args3.array_size is None


def test_headless_passes_through_not_swallowed():
    # Not declared on the launcher any more, so it flows to train_agent (which
    # parses the bool) instead of being silently consumed.
    _, passthrough = ts.create_parser().parse_known_args(
        _agent_argv() + ["--headless", "False"]
    )
    assert passthrough == ["--headless", "False"]


def test_rsync_excludes_the_dangerous_dirs(monkeypatch):
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ts, "subprocess_run", fake_run)
    ts.sync_code_to_cluster("me", "/exp", Path("/repo"), extra_excludes=["robojudo"])
    rsync = next(c for c in captured if c.startswith("rsync"))
    for pattern in [
        ".worktrees",
        ".venv**",
        "**/*.mp4",
        "data/pretrained_models",
        "robojudo",
    ]:
        assert f'--exclude="{pattern}"' in rsync, pattern
