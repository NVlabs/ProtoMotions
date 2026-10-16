# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the SLURM stall watchdog and process-tree kill.

train_slurm.py embeds a stall watchdog in the job command: run_with_stall_watchdog
runs training and kills its whole process tree (kill_tree) if the SLURM log stops
growing. The bash for both is produced by _stall_watchdog_bash() / _kill_tree_bash()
so it can be exercised here against fake child processes -- no SLURM, no GPU. The
watchdog's poll interval (STALL_POLL_SEC) and no-growth limit (STALL_QUIET_LIMIT)
are overridden to sub-second / tiny values so the suite runs in a couple of seconds
rather than minutes; with the defaults they reduce to the production 'sleep 60;
quiet >= STALL_TIMEOUT_MIN'."""

import importlib.util
import os
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


TRAIN_SLURM_PATH = Path(__file__).resolve().parents[1] / "train_slurm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_slurm_watchdog_under_test", TRAIN_SLURM_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ts = _load_module()

# The bash under test: the two function definitions, sourced into every snippet.
if hasattr(ts, "_retry_lib_source"):
    # The development launcher sources shared primitives; the public launcher
    # embeds them. Exercise the implementation used by the loaded launcher.
    WATCHDOG_BASH = ts._retry_lib_source(
        shlex.quote(str(TRAIN_SLURM_PATH.parents[1]))
    )
else:
    WATCHDOG_BASH = ts._kill_tree_bash() + "\n" + ts._stall_watchdog_bash()

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("bash") is None
    or shutil.which("pgrep") is None,
    reason="needs bash + pgrep + GNU stat (the Linux CI runner has them)",
)


def _bash(body, env=None, timeout=30):
    full = dict(os.environ)
    if env:
        full.update(env)
    return subprocess.run(
        ["bash", "-c", WATCHDOG_BASH + "\n" + body],
        env=full,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _exit_code(stdout):
    for line in stdout.splitlines():
        if line.startswith("EXIT="):
            return int(line.split("=", 1)[1])
    raise AssertionError(f"no EXIT= line in:\n{stdout}")


def test_watchdog_kills_a_silent_child(tmp_path):
    log = tmp_path / "slurm_output.log"
    log.write_text("")  # never grows
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", "STALL_QUIET_LIMIT": "3"}
    r = _bash('run_with_stall_watchdog sleep 100\necho "EXIT=$?"', env=env, timeout=15)
    # Returned at all (no TimeoutExpired) => the 100s sleep was killed early, and a
    # killed process yields a nonzero status.
    assert _exit_code(r.stdout) != 0, r.stdout + r.stderr


def test_watchdog_lets_an_active_child_finish(tmp_path):
    log = tmp_path / "slurm_output.log"
    log.write_text("")
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", "STALL_QUIET_LIMIT": "4"}
    body = (
        "run_with_stall_watchdog bash -c "
        "'for i in $(seq 1 15); do echo tick >> \"$LOGFILE\"; sleep 0.03; done'\n"
        'echo "EXIT=$?"'
    )
    r = _bash(body, env=env, timeout=15)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


@pytest.mark.parametrize("disable_env", ["STALL_QUIET_LIMIT=0", "STALL_TIMEOUT_MIN=0"])
def test_watchdog_disabled_does_not_kill(tmp_path, disable_env):
    log = tmp_path / "slurm_output.log"
    log.write_text("")
    key, val = disable_env.split("=")
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", key: val}
    r = _bash(
        "run_with_stall_watchdog bash -c 'sleep 0.4'\n" 'echo "EXIT=$?"',
        env=env,
        timeout=15,
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


def test_watchdog_does_not_kill_just_under_threshold(tmp_path):
    log = tmp_path / "slurm_output.log"
    log.write_text("")
    # threshold = 5 polls * 0.1s = 0.5s; child stays silent for 0.25s then exits 0.
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.1", "STALL_QUIET_LIMIT": "5"}
    r = _bash(
        "run_with_stall_watchdog bash -c 'sleep 0.25'\n" 'echo "EXIT=$?"',
        env=env,
        timeout=15,
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


def test_kill_tree_reaps_the_whole_subtree(tmp_path):
    # A truly stuck training process holds its ranks (and their GPUs) several
    # levels down; kill_tree must reap the whole tree so nothing is orphaned.
    pids = tmp_path / "pids"
    body = (
        f'export PIDS="{pids}"\n'
        ': > "$PIDS"\n'
        "bash -c 'sleep 100 & echo $! >> \"$PIDS\"; sleep 100 & echo $! >> \"$PIDS\"; wait' &\n"
        "root=$!\n"
        "sleep 0.4\n"
        'kill_tree "$root"\n'
        "sleep 0.3\n"
        "alive=0\n"
        'while read -r p; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done < "$PIDS"\n'
        'kill -0 "$root" 2>/dev/null && alive=$((alive+1))\n'
        'echo "ALIVE=$alive"'
    )
    r = _bash(body, timeout=15)
    assert "ALIVE=0" in r.stdout, r.stdout + r.stderr
