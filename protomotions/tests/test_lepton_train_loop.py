# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for scripts/cluster/lepton_train_loop.sh.

The loop's risky logic -- the stall watchdog, the process-tree kill, argument
preservation, the crash-restart decision and the give-up sentinel -- is exercised
here without a GPU, a cluster, or `lep`, using fake child processes and a fake
"python". Two hooks in the script make this possible:

  * LEPTON_LOOP_SOURCE_ONLY  -- `source` the script to get its functions only,
                                so the watchdog / kill_tree run against fake
                                children.
  * LEPTON_LOOP_SKIP_SETUP   -- run the whole loop but skip the pip install +
                                IsaacLab warmup, so a fake python drives it.

The watchdog poll interval (STALL_POLL_SEC) and no-growth limit (STALL_QUIET_LIMIT
/ RETRY_MAX_ATTEMPTS) are overridden to sub-second / tiny values, so the suite runs
in a couple of seconds rather than minutes."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "lepton_train_loop.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("bash") is None
    or shutil.which("pgrep") is None,
    reason="needs bash + pgrep + GNU stat (the Linux CI runner has them)",
)


def _bash(snippet, env=None, timeout=30):
    full = dict(os.environ)
    if env:
        full.update(env)
    return subprocess.run(
        ["bash", "-c", snippet],
        env=full,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _exit_code(stdout):
    """Parse the EXIT=<n> line a snippet prints for run_with_stall_watchdog."""
    for line in stdout.splitlines():
        if line.startswith("EXIT="):
            return int(line.split("=", 1)[1])
    raise AssertionError(f"no EXIT= line in:\n{stdout}")


def _source_snippet(body):
    # Fail loudly if the script cannot be sourced, so a wrong path can never look
    # like a passing test (an undefined function would otherwise print EXIT=127).
    return (
        "export LEPTON_LOOP_SOURCE_ONLY=1\n"
        f'source "{SCRIPT}" || {{ echo "SOURCE_FAILED"; exit 99; }}\n'
        f"{body}\n"
    )


# --- stall watchdog ----------------------------------------------------------


def test_watchdog_kills_a_silent_child(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("")  # never grows
    snippet = _source_snippet(
        f'export LOGFILE="{log}" STALL_POLL_SEC=0.05 STALL_QUIET_LIMIT=3\n'
        "run_with_stall_watchdog sleep 100\n"
        'echo "EXIT=$?"'
    )
    r = _bash(snippet, timeout=15)
    # Returned at all (no TimeoutExpired) => the 100s sleep was killed early, and
    # a killed process yields a nonzero status.
    assert _exit_code(r.stdout) != 0, r.stdout + r.stderr


def test_watchdog_lets_an_active_child_finish(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("")
    snippet = _source_snippet(
        f'export LOGFILE="{log}" STALL_POLL_SEC=0.05 STALL_QUIET_LIMIT=4\n'
        "run_with_stall_watchdog bash -c "
        "'for i in $(seq 1 15); do echo tick >> \"$LOGFILE\"; sleep 0.03; done'\n"
        'echo "EXIT=$?"'
    )
    r = _bash(snippet, timeout=15)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


@pytest.mark.parametrize("disable_env", ["STALL_QUIET_LIMIT=0", "STALL_TIMEOUT_MIN=0"])
def test_watchdog_disabled_does_not_kill(tmp_path, disable_env):
    log = tmp_path / "train.log"
    log.write_text("")
    snippet = _source_snippet(
        f'export LOGFILE="{log}" STALL_POLL_SEC=0.05 {disable_env}\n'
        "run_with_stall_watchdog bash -c 'sleep 0.4'\n"  # silent, but must survive
        'echo "EXIT=$?"'
    )
    r = _bash(snippet, timeout=15)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


def test_watchdog_does_not_kill_just_under_threshold(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("")
    # threshold = 5 polls * 0.1s = 0.5s; child stays silent for 0.25s then exits 0.
    snippet = _source_snippet(
        f'export LOGFILE="{log}" STALL_POLL_SEC=0.1 STALL_QUIET_LIMIT=5\n'
        "run_with_stall_watchdog bash -c 'sleep 0.25'\n"
        'echo "EXIT=$?"'
    )
    r = _bash(snippet, timeout=15)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


# --- process-tree termination ------------------------------------------------


def test_watchdog_kills_the_whole_process_tree(tmp_path):
    # The childless-sleep test above proves "silent -> killed", but a bare sleep
    # has no descendants, so it would still pass if the watchdog reverted from
    # kill_tree to a plain kill. Here the watched process spawns two grandchildren
    # (as real training spawns rank workers); after the watchdog fires, none must
    # survive -- otherwise a reparented worker keeps its GPU/MASTER_PORT and the
    # immediate resume collides.
    log = tmp_path / "train.log"
    log.write_text("")  # never grows -> watchdog fires
    pids = tmp_path / "pids"
    snippet = _source_snippet(
        f'export LOGFILE="{log}" STALL_POLL_SEC=0.05 STALL_QUIET_LIMIT=3 PIDS="{pids}"\n'
        ': > "$PIDS"\n'
        "run_with_stall_watchdog bash -c "
        "'sleep 100 & echo $! >> \"$PIDS\"; sleep 100 & echo $! >> \"$PIDS\"; wait'\n"
        'echo "EXIT=$?"\n'
        "sleep 0.3\n"
        "alive=0\n"
        'while read -r p; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done < "$PIDS"\n'
        'echo "ALIVE=$alive"'
    )
    r = _bash(snippet, timeout=15)
    assert _exit_code(r.stdout) != 0, r.stdout + r.stderr  # the wedged tree was killed
    assert "ALIVE=0" in r.stdout, r.stdout + r.stderr  # no grandchild orphaned


def test_kill_tree_reaps_the_whole_subtree(tmp_path):
    pids = tmp_path / "pids"
    snippet = _source_snippet(
        f'export PIDS="{pids}"\n'
        ': > "$PIDS"\n'
        # inner bash spawns two grandchildren and waits, so the tree is 3 deep
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
    r = _bash(snippet, timeout=15)
    assert "ALIVE=0" in r.stdout, r.stdout + r.stderr


# --- full loop: argv preservation, restart, sentinel -------------------------


def _fake_python(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/bash\n" + body + "\n")
    p.chmod(0o755)
    return p


def _run_loop(script_python, exp_dir, max_runs, tail, extra_env=None, timeout=30):
    env = {"LEPTON_LOOP_SKIP_SETUP": "1", "STALL_TIMEOUT_MIN": "0"}
    if extra_env:
        env.update(extra_env)
    return _bash_argv(
        ["bash", str(SCRIPT), str(script_python), "newton", "myexp", str(exp_dir), str(max_runs), *tail],
        env=env,
        timeout=timeout,
    )


def _bash_argv(argv, env=None, timeout=30):
    full = dict(os.environ)
    if env:
        full.update(env)
    return subprocess.run(argv, env=full, capture_output=True, text=True, timeout=timeout)


def test_train_args_reach_python_intact(tmp_path):
    out = tmp_path / "rec.out"
    rec = _fake_python(tmp_path, "rec.sh", f'printf "%s\\n" "$@" >> "{out}"\nexit 0')
    exp = tmp_path / "exp"
    exp.mkdir()
    r = _run_loop(rec, exp, 1, ["--robot-name=g1", "--overrides", "a=b", "c d"])
    assert r.returncode == 0, r.stdout + r.stderr
    got = out.read_text().splitlines()
    # The loop runs: <python> -u protomotions/train_agent.py <tail...>
    assert got == [
        "-u",
        "protomotions/train_agent.py",
        "--robot-name=g1",
        "--overrides",
        "a=b",
        "c d",  # the spaced token survived "$@", not split into "c" and "d"
    ], got


def test_deterministic_failure_gives_up_and_writes_sentinel(tmp_path):
    always_fail = _fake_python(tmp_path, "fail.sh", "exit 1")
    exp = tmp_path / "exp"
    exp.mkdir()
    r = _run_loop(
        always_fail, exp, 5, ["--robot-name=g1"], extra_env={"RETRY_MAX_ATTEMPTS": "2"}
    )
    assert r.returncode != 0, r.stdout + r.stderr  # the job is marked failed
    assert (exp / ".lepton_loop_gave_up").exists(), "give-up sentinel not written"


def test_crash_after_progress_then_success(tmp_path):
    counter = tmp_path / "counter"
    prog = _fake_python(
        tmp_path,
        "prog.sh",
        f'n=0; [ -f "{counter}" ] && n=$(cat "{counter}")\n'
        'if [ "$n" -eq 0 ]; then\n'
        f'  echo 1 > "{counter}"\n'
        "  mkdir -p results/myexp; touch results/myexp/last.ckpt\n"  # advance progress
        "  exit 1\n"  # crash after saving
        "fi\n"
        "exit 0",  # second run completes
    )
    exp = tmp_path / "exp"
    exp.mkdir()
    r = _run_loop(
        prog, exp, 5, ["--robot-name=g1"], extra_env={"RETRY_MAX_ATTEMPTS": "2"}
    )
    # A crash that advanced the checkpoint is treated as transient, and the next
    # run completes, so the job succeeds. (This case exercises the progress path
    # but does not by itself prove the consecutive-counter RESET matters -- that
    # is what the interleaved test below pins down.)
    assert r.returncode == 0, r.stdout + r.stderr


def test_progress_crash_resets_consecutive_no_progress_counter(tmp_path):
    # The give-up rule is "RETRY_MAX_ATTEMPTS runs IN A ROW with no checkpoint
    # progress". A progressing crash must RESET that consecutive count -- otherwise
    # interleaved transient failures around real progress would be miscounted as a
    # deterministic failure and the job would give up while it is actually
    # advancing. With RETRY_MAX_ATTEMPTS=2, this fake python never produces two
    # no-progress crashes in a row (no-progress, progress, no-progress, progress,
    # success), so the run must succeed. Deleting the reset makes it give up at the
    # third run -- so this test, unlike the one above, actually depends on it.
    counter = tmp_path / "counter"
    prog = _fake_python(
        tmp_path,
        "interleaved.sh",
        f'n=0; [ -f "{counter}" ] && n=$(cat "{counter}"); n=$((n+1)); echo "$n" > "{counter}"\n'
        "mkdir -p results/myexp\n"
        "case \"$n\" in\n"
        "  1) exit 1 ;;\n"  # no progress
        "  2) touch -m -d @1000000100 results/myexp/last.ckpt; exit 1 ;;\n"  # progress
        "  3) exit 1 ;;\n"  # no progress (checkpoint unchanged)
        "  4) touch -m -d @1000000200 results/myexp/last.ckpt; exit 1 ;;\n"  # progress
        "  *) exit 0 ;;\n"  # success
        "esac",
    )
    exp = tmp_path / "exp"
    exp.mkdir()
    r = _run_loop(
        prog, exp, 9, ["--robot-name=g1"], extra_env={"RETRY_MAX_ATTEMPTS": "2"}
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (exp / ".lepton_loop_gave_up").exists(), "gave up despite never 2-in-a-row"


def test_multiword_interpreter_word_splits_to_command_and_flag(tmp_path):
    # The interpreter positional can be "cmd -flag" (the default is
    # "isaaclab.sh -p"); the loop runs it UNQUOTED so it word-splits back into a
    # command and its flag. A recorder that insists on receiving -p as its first
    # arg fails (exit 127, "no such file 'wrap.sh -p'") if the loop ever quotes
    # "$PYTHON" -- so this guards that deliberate word-split end to end.
    out = tmp_path / "wrap.out"
    wrap = _fake_python(
        tmp_path,
        "wrap.sh",
        f'[ "$1" = "-p" ] || {{ echo "interpreter flag not word-split" >&2; exit 3; }}\n'
        "shift\n"
        f'printf "%s\\n" "$@" >> "{out}"\nexit 0',
    )
    exp = tmp_path / "exp"
    exp.mkdir()
    r = _run_loop(f"{wrap} -p", exp, 1, ["--robot-name=g1"])
    assert r.returncode == 0, r.stdout + r.stderr
    got = out.read_text().splitlines()
    assert got == ["-u", "protomotions/train_agent.py", "--robot-name=g1"], got


def test_prior_give_up_sentinel_fails_fast_without_training(tmp_path):
    # A relaunch that finds a give-up sentinel must exit NONZERO (marked failed),
    # not silently exit 0, and must not run training again.
    ran = tmp_path / "ran.marker"
    py = _fake_python(tmp_path, "py.sh", f'touch "{ran}"\nexit 0')
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / ".lepton_loop_gave_up").write_text("")
    r = _run_loop(py, exp, 1, ["--robot-name=g1"])
    assert r.returncode != 0, r.stdout + r.stderr  # NOT a false success
    assert not ran.exists(), "training ran despite the give-up sentinel"
