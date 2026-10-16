# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for scripts/cluster/retry_lib.sh -- the shared retry/watchdog
primitives both train_slurm.py and lepton_train_loop.sh source.

Everything is exercised with fake commands (a small script that bumps a counter,
optionally writes an exception line to $LOGFILE, optionally advances $CKPT's mtime,
and exits with a chosen code) so no GPU, cluster or real training is needed. The
watchdog's poll interval and no-growth limit are overridden to sub-second values,
and RETRY_BACKOFF_SEC is 0, so the suite runs in a couple of seconds.

The three primitives under test:
  kill_tree                 reaps a whole process tree.
  run_with_stall_watchdog   THE one watchdog: kills a wedged tree on log silence.
  run_with_retry            the unified loop: break (SLURM) / continue (Lepton) on
                            checkpoint progress, fast-fail a repeated Python
                            exception, bounded, optional give-up sentinel."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


RETRY_LIB = Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "retry_lib.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("bash") is None
    or shutil.which("pgrep") is None,
    reason="needs bash + pgrep + GNU date/stat/wc (the Linux CI runner has them)",
)


def _bash(body, env=None, timeout=40):
    full = dict(os.environ)
    if env:
        full.update(env)
    # Fail loudly if the lib cannot be sourced, so a wrong path can never look like
    # a passing test (an undefined function would otherwise print EXIT=127).
    snippet = f'source "{RETRY_LIB}" || {{ echo "SOURCE_FAILED"; exit 99; }}\n{body}'
    return subprocess.run(
        ["bash", "-c", snippet],
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


def _fake_cmd(tmp_path, name, body):
    """A fake training command: bump the attempt counter into $n, record a RUN
    line, then run `body` (which may read $n, write $LOGFILE/$CKPT, and exit)."""
    p = tmp_path / name
    p.write_text(
        "#!/bin/bash\n"
        'n=0; [ -f "$COUNTER" ] && n=$(cat "$COUNTER"); n=$((n + 1)); echo "$n" > "$COUNTER"\n'
        'echo RUN >> "$RUNS"\n' + body + "\n"
    )
    p.chmod(0o755)
    return p


def _runs(tmp_path):
    r = tmp_path / "runs"
    return len(r.read_text().splitlines()) if r.exists() else 0


def _env(tmp_path, log, **over):
    e = {
        "LOGFILE": str(log),
        "CKPT": str(tmp_path / "last.ckpt"),
        "COUNTER": str(tmp_path / "counter"),
        "RUNS": str(tmp_path / "runs"),
        "RETRY_MAX_ATTEMPTS": "3",
        "RETRY_REPEAT_LIMIT": "3",
        "RETRY_BACKOFF_SEC": "0",
        "RETRY_ON_PROGRESS": "continue",
    }
    e.update({k: str(v) for k, v in over.items()})
    return e


def _run(tmp_path, cmd, env):
    return _bash(f'run_with_retry "{cmd}"\necho "EXIT=$?"', env=env)


# A body fragment that advances the checkpoint mtime to a distinct, increasing
# value per attempt (second-granularity mtime can't otherwise change within a test
# that runs in <1s). Used for the "training reached / made progress" attempts.
_ADVANCE_CKPT = 'touch -m -d @$((1000000000 + n)) "$CKPT"'


# --- run_with_retry: basic accounting ----------------------------------------


def test_success_on_first_attempt_runs_once(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "ok.sh", "exit 0")
    r = _run(tmp_path, cmd, _env(tmp_path, log))
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 1


def test_passes_multiple_arguments_to_the_command(tmp_path):
    # Production calls `run_with_retry run_with_stall_watchdog <python> -u ...` /
    # `run_with_retry stdbuf -oL -eL srun ...` -- many argv elements. Run the whole
    # argv, not just $1.
    log = tmp_path / "log"
    log.write_text("")
    env = _env(tmp_path, log)
    r = _bash(
        'run_with_retry bash -c \'echo RUN >> "$RUNS"; exit 0\'\necho "EXIT=$?"',
        env=env,
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 1


def test_max_below_one_still_runs_once(tmp_path):
    # A 0/negative cap must not skip the command and report a false success.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "ok.sh", "exit 0")
    r = _run(tmp_path, cmd, _env(tmp_path, log, RETRY_MAX_ATTEMPTS=0))
    assert _exit_code(r.stdout) == 0
    assert _runs(tmp_path) == 1


# --- break mode (SLURM): stop the moment training is reached ------------------


def test_break_mode_stops_once_a_checkpoint_is_written(tmp_path):
    # A scene build that reached training (wrote a checkpoint) then died -- like a
    # walltime kill mid-run. In break mode the array element resumes from the
    # checkpoint; retrying srun in place is wrong.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "midrun.sh", f'{_ADVANCE_CKPT}\nexit 1')
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, RETRY_ON_PROGRESS="break", RETRY_MAX_ATTEMPTS=5)
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 1


def test_break_mode_retries_pretraining_abort_up_to_the_cap(tmp_path):
    # A native abort before training (no checkpoint, no Python traceback) is flaky:
    # retry it, bounded by the cap, then return the last exit code.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "abort.sh", 'echo "double free or corruption" >> "$LOGFILE"\nexit 250')
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, RETRY_ON_PROGRESS="break", RETRY_MAX_ATTEMPTS=4)
    )
    assert _exit_code(r.stdout) == 250
    assert _runs(tmp_path) == 4


# --- continue mode (Lepton): resume until done -------------------------------


def test_continue_mode_resumes_on_progress_then_completes(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    # Attempt 1 advances the checkpoint then crashes; attempt 2 completes.
    cmd = _fake_cmd(
        tmp_path, "prog.sh", f'if [ "$n" -eq 1 ]; then {_ADVANCE_CKPT}; exit 1; fi\nexit 0'
    )
    r = _run(
        tmp_path, cmd, _env(tmp_path, log, RETRY_ON_PROGRESS="continue", RETRY_MAX_ATTEMPTS=2)
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 2


def test_continue_mode_gives_up_after_consecutive_no_progress(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "fail.sh", "exit 1")
    r = _run(
        tmp_path,
        cmd,
        _env(
            tmp_path,
            log,
            RETRY_ON_PROGRESS="continue",
            RETRY_MAX_ATTEMPTS=2,
            RETRY_TOTAL_ATTEMPTS=9,
            GAVE_UP=str(tmp_path / "gave_up"),
        ),
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 2  # gave up at the no-progress cap, not 9
    assert (tmp_path / "gave_up").exists(), "give-up sentinel not written"


def test_continue_mode_total_cap_bounds_a_never_completing_run(tmp_path):
    # Every attempt advances the checkpoint (progress) then crashes, so the
    # no-progress streak keeps resetting and never trips its cap. RETRY_TOTAL_ATTEMPTS
    # is the backstop that stops it looping forever; and because it exhausted the
    # budget while still making progress (not deterministic), NO sentinel is written.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "slowprog.sh", f'{_ADVANCE_CKPT}\nexit 1')
    r = _run(
        tmp_path,
        cmd,
        _env(
            tmp_path,
            log,
            RETRY_ON_PROGRESS="continue",
            RETRY_MAX_ATTEMPTS=2,
            RETRY_TOTAL_ATTEMPTS=4,
            GAVE_UP=str(tmp_path / "gave_up"),
        ),
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 4
    assert not (tmp_path / "gave_up").exists(), "sentinel written despite progress"


def test_prior_give_up_sentinel_short_circuits_without_running(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    gave_up = tmp_path / "gave_up"
    gave_up.write_text("")
    cmd = _fake_cmd(tmp_path, "ok.sh", "exit 0")
    r = _run(tmp_path, cmd, _env(tmp_path, log, GAVE_UP=str(gave_up)))
    assert _exit_code(r.stdout) != 0, r.stdout + r.stderr  # not a false success
    assert _runs(tmp_path) == 0, "command ran despite the give-up sentinel"


# --- the Python fast-fail: retry flaky, refuse deterministic -----------------


def test_repeated_identical_exception_fast_fails(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "det.sh", 'echo "AssertionError: boom" >> "$LOGFILE"\nexit 1')
    r = _run(
        tmp_path,
        cmd,
        _env(tmp_path, log, RETRY_MAX_ATTEMPTS=10, RETRY_REPEAT_LIMIT=3, GAVE_UP=str(tmp_path / "g")),
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 3  # stopped at the repeat limit, not 10
    assert "deterministic, not flaky" in r.stderr
    assert "AssertionError: boom" in r.stderr  # the traceback is surfaced
    assert (tmp_path / "g").exists()  # a deterministic give-up writes the sentinel


def test_message_less_exception_fast_fails(tmp_path):
    # A bare `assert` prints just "AssertionError" -- no colon, no message.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "bare.sh", 'echo "AssertionError" >> "$LOGFILE"\nexit 1')
    r = _run(tmp_path, cmd, _env(tmp_path, log, RETRY_MAX_ATTEMPTS=10, RETRY_REPEAT_LIMIT=3))
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 3
    assert "deterministic, not flaky" in r.stderr


def test_same_class_varying_message_fast_fails(tmp_path):
    # Same class, different message each attempt (a device id): the bug is one, so
    # the class-only comparison must still count it.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "vary.sh", 'echo "RuntimeError: CUDA error on device $n" >> "$LOGFILE"\nexit 1')
    r = _run(tmp_path, cmd, _env(tmp_path, log, RETRY_MAX_ATTEMPTS=10, RETRY_REPEAT_LIMIT=3))
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 3


def test_native_abort_without_traceback_uses_full_budget(tmp_path):
    # exit 250 with an abort message but NO Python exception line -> the fast-fail
    # must not trip; this is the heap-corruption bug and it must keep retrying.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "abort.sh", 'echo "double free or corruption (out)" >> "$LOGFILE"\nexit 250')
    r = _run(tmp_path, cmd, _env(tmp_path, log, RETRY_MAX_ATTEMPTS=4, RETRY_REPEAT_LIMIT=3))
    assert _exit_code(r.stdout) == 250
    assert _runs(tmp_path) == 4


def test_alternating_exceptions_do_not_fast_fail(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path,
        "alt.sh",
        'if [ $((n % 2)) -eq 1 ]; then echo "ValueError: odd" >> "$LOGFILE"; '
        'else echo "KeyError: even" >> "$LOGFILE"; fi\nexit 1',
    )
    r = _run(tmp_path, cmd, _env(tmp_path, log, RETRY_MAX_ATTEMPTS=5, RETRY_REPEAT_LIMIT=3))
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 5  # never two identical in a row -> full budget


def test_fast_fail_reads_only_the_current_attempt_slice(tmp_path):
    # Odd attempts emit the exception, even attempts emit nothing. Only if each
    # attempt's exception is read from ITS OWN log slice does this stay under the
    # limit -- reading the whole accumulated log would see the previous attempt's
    # line during a silent attempt and stop early. Guards the _mark slicing.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path, "slice.sh", 'case "$n" in 1|3|5) echo "ValueError: boom" >> "$LOGFILE";; esac\nexit 1'
    )
    r = _run(tmp_path, cmd, _env(tmp_path, log, RETRY_MAX_ATTEMPTS=5, RETRY_REPEAT_LIMIT=2))
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 5  # never two same-class in a row -> whole budget


def test_progress_resets_the_repeat_counter(tmp_path):
    # A deterministic-looking exception on either side of a real progress step must
    # NOT accumulate across the progress: the reset of prev_cls/repeat on progress
    # means attempts 1 and 3 (same class, but with progress between them) do not
    # fast-fail. With REPEAT_LIMIT=2 and a progress step at attempt 2, the run
    # survives to completion at attempt 4.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path,
        "reset.sh",
        "case \"$n\" in\n"
        '  1) echo "ValueError: x" >> "$LOGFILE"; exit 1 ;;\n'  # no progress, exc
        f'  2) {_ADVANCE_CKPT}; exit 1 ;;\n'  # progress -> resets counters
        '  3) echo "ValueError: x" >> "$LOGFILE"; exit 1 ;;\n'  # no progress, same class again
        f'  4) {_ADVANCE_CKPT}; exit 0 ;;\n'  # completes
        "esac",
    )
    r = _run(
        tmp_path,
        cmd,
        _env(
            tmp_path,
            log,
            RETRY_ON_PROGRESS="continue",
            RETRY_MAX_ATTEMPTS=2,
            RETRY_REPEAT_LIMIT=2,
            RETRY_TOTAL_ATTEMPTS=9,
            GAVE_UP=str(tmp_path / "g"),
        ),
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert not (tmp_path / "g").exists(), "fast-failed despite progress between the two"


def test_one_time_epoch0_checkpoint_does_not_defeat_fast_fail(tmp_path):
    # Models base_agent's epoch-0 save: last.ckpt is created ONCE on the first
    # attempt (advancing the mtime), then a deterministic bug crashes every attempt
    # with the same exception and NO further checkpoint progress (base_agent writes
    # the epoch-0 ckpt only when absent, so resume attempts do not re-touch it). The
    # single early "progress" must not mask the deterministic crash -- it fast-fails a
    # couple attempts later, not after the whole 75-run budget. This is the contract
    # base_agent's only-if-absent save relies on; the unconditional save it replaced
    # advanced the mtime EVERY attempt and burned the whole budget instead.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path,
        "epoch0.sh",
        f'if [ "$n" -eq 1 ]; then {_ADVANCE_CKPT}; fi\n'
        'echo "RuntimeError: boom" >> "$LOGFILE"\nexit 1',
    )
    r = _run(
        tmp_path,
        cmd,
        _env(
            tmp_path,
            log,
            RETRY_ON_PROGRESS="continue",
            RETRY_MAX_ATTEMPTS=9,
            RETRY_REPEAT_LIMIT=3,
            RETRY_TOTAL_ATTEMPTS=75,
            GAVE_UP=str(tmp_path / "g"),
        ),
    )
    assert _exit_code(r.stdout) == 1
    # attempt 1: progress -> reset. attempts 2,3,4: same class -> repeat 1,2,3 -> stop.
    assert _runs(tmp_path) == 4, r.stdout + r.stderr
    assert (tmp_path / "g").exists()


def test_total_attempts_below_no_progress_cap_is_honored(tmp_path):
    # An explicit small total (Lepton --max-runs below --no-progress-limit) is a hard
    # cap: the loop stops at total, not silently at the larger no-progress cap. Guards
    # the clamp that used to raise total up to nomax.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(tmp_path, "fail.sh", "exit 1")
    r = _run(
        tmp_path,
        cmd,
        _env(
            tmp_path,
            log,
            RETRY_ON_PROGRESS="continue",
            RETRY_MAX_ATTEMPTS=3,
            RETRY_TOTAL_ATTEMPTS=2,
        ),
    )
    assert _exit_code(r.stdout) == 1
    assert _runs(tmp_path) == 2  # honored total=2, not the no-progress cap of 3


# --- run_with_stall_watchdog + kill_tree (THE one watchdog) ------------------


def test_watchdog_kills_a_silent_child(tmp_path):
    log = tmp_path / "log"
    log.write_text("")  # never grows
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", "STALL_QUIET_LIMIT": "3"}
    r = _bash('run_with_stall_watchdog sleep 100\necho "EXIT=$?"', env=env, timeout=15)
    assert _exit_code(r.stdout) != 0, r.stdout + r.stderr


def test_watchdog_lets_an_active_child_finish(tmp_path):
    log = tmp_path / "log"
    log.write_text("")
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", "STALL_QUIET_LIMIT": "4"}
    body = (
        "run_with_stall_watchdog bash -c "
        "'for i in $(seq 1 15); do echo tick >> \"$LOGFILE\"; sleep 0.03; done'\n"
        'echo "EXIT=$?"'
    )
    r = _bash(body, env=env, timeout=15)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


def test_failed_attempt_leaves_no_orphaned_ranks(tmp_path):
    """A failed attempt must not leave its surviving ranks running.

    When one rank aborts, the launcher does not always reap its siblings; those
    survivors reparent to init, keeping their GPU contexts, their MASTER_PORT and
    -- as seen on a real Lepton job -- spamming NCCL errors once a second through
    every later attempt. kill_tree cannot reach them (its pgrep -P walk stops at
    the vanished parent), so the attempt runs in its own process group and the
    group is reaped afterwards.

    Shape mirrors production: run_with_retry wraps run_with_stall_watchdog, which
    is where the process group is established (lepton_train_loop.sh does exactly
    this).
    """
    marker = tmp_path / "rank"
    log = tmp_path / "log"
    log.write_text("")
    env = {
        "LOGFILE": str(log),
        "MARK": str(marker),
        "CKPT": str(tmp_path / "nonexistent.ckpt"),
        "RETRY_MAX_ATTEMPTS": "2",
        "RETRY_BACKOFF_SEC": "0",
        "RETRY_REPEAT_LIMIT": "0",
        "STALL_QUIET_LIMIT": "0",  # watchdog off: isolate the crash path
        "REAP_GRACE_SEC": "1",
    }
    # Three "ranks" that outlive the attempt; the attempt then exits nonzero.
    body = (
        "attempt() {\n"
        "  for r in 1 2 3; do\n"
        '    ( for i in $(seq 1 100); do echo alive >> "${MARK}.$r"; sleep 0.2; done ) &\n'
        "  done\n"
        "  sleep 0.4\n"
        "  exit 250\n"
        "}\n"
        "export -f attempt\n"
        "run_with_retry run_with_stall_watchdog bash -c attempt >/dev/null 2>&1\n"
        'echo "EXIT=$?"\n'
    )
    r = _bash(body, env=env, timeout=60)
    # Sanity: the ranks really did start, so a green result cannot mean "nothing ran".
    assert any(
        (tmp_path / f"rank.{i}").exists() for i in (1, 2, 3)
    ), f"no rank markers written; test did not exercise the path: {r.stdout}{r.stderr}"

    counts = {}
    for i in (1, 2, 3):
        p = tmp_path / f"rank.{i}"
        counts[i] = len(p.read_text().splitlines()) if p.exists() else 0
    time.sleep(1.0)
    still_running = []
    for i in (1, 2, 3):
        p = tmp_path / f"rank.{i}"
        now = len(p.read_text().splitlines()) if p.exists() else 0
        if now > counts[i]:
            still_running.append(i)
    assert not still_running, (
        f"ranks {still_running} outlived the retry loop -- orphan leak. "
        f"{r.stdout}{r.stderr}"
    )


def test_watchdog_kills_a_child_emitting_only_dead_rank_noise(tmp_path):
    """A half-dead DDP job must be killed even though its log keeps growing.

    When a rank DIES (rather than wedging), the survivors stop blocking at the
    collective and emit NCCL heartbeat / TCPStore "Broken pipe" errors several
    lines a second, forever. A watchdog keyed on raw file size therefore never
    fires -- the symptom of the failure is what keeps it asleep. Measured on a
    real 4-rank job whose ranks 0 and 1 vanished silently at epoch 1; it would
    have idled 4 A100s for the full job TTL.
    """
    log = tmp_path / "log"
    log.write_text("")
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", "STALL_QUIET_LIMIT": "3"}
    # Keep the payload free of quotes: it is nested inside bash -c '...' and a
    # stray quote makes the CHILD die of a syntax error, which would make this
    # test pass for the wrong reason (it did, on the first draft).
    body = (
        "run_with_stall_watchdog bash -c "
        "'for i in $(seq 1 400); do "
        "echo [rank2]: ProcessGroupNCCL Broken pipe should dump >> \"$LOGFILE\"; "
        "sleep 0.02; done'\n"
        'echo "EXIT=$?"'
    )
    r = _bash(body, env=env, timeout=30)
    # Guard against the child having failed for its own reasons: it must have
    # actually written the noise before being killed.
    assert log.read_text().count("ProcessGroupNCCL") > 5, log.read_text()[:400]
    assert _exit_code(r.stdout) != 0, r.stdout + r.stderr


def test_watchdog_spares_a_child_mixing_progress_with_noise(tmp_path):
    """Noise must not mask real progress: a job still logging epochs survives."""
    log = tmp_path / "log"
    log.write_text("")
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", "STALL_QUIET_LIMIT": "4"}
    body = (
        "run_with_stall_watchdog bash -c "
        "'for i in $(seq 1 15); do "
        'echo "[rank2]: ProcessGroupNCCL Broken pipe" >> "$LOGFILE"; '
        'echo "Epoch $i, training... 100% 0:00:47" >> "$LOGFILE"; '
        "sleep 0.03; done'\n"
        'echo "EXIT=$?"'
    )
    r = _bash(body, env=env, timeout=15)
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


@pytest.mark.parametrize("disable_env", ["STALL_QUIET_LIMIT=0", "STALL_TIMEOUT_MIN=0"])
def test_watchdog_disabled_does_not_kill(tmp_path, disable_env):
    log = tmp_path / "log"
    log.write_text("")
    key, val = disable_env.split("=")
    env = {"LOGFILE": str(log), "STALL_POLL_SEC": "0.05", key: val}
    r = _bash(
        "run_with_stall_watchdog bash -c 'sleep 0.4'\n" 'echo "EXIT=$?"',
        env=env,
        timeout=15,
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr


def test_kill_tree_reaps_the_whole_subtree(tmp_path):
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


def test_watchdog_inside_retry_kills_hang_then_retry_succeeds(tmp_path):
    # The production composition: run_with_retry wraps run_with_stall_watchdog. A
    # hung first attempt (silent forever) is killed by the watchdog, run_with_retry
    # sees the nonzero exit and -- no checkpoint, no exception -> flaky -- retries;
    # the second attempt completes.
    log = tmp_path / "log"
    log.write_text("")
    cmd = _fake_cmd(
        tmp_path, "hang.sh", 'case "$n" in 1) sleep 100;; *) exit 0;; esac'
    )
    env = _env(
        tmp_path,
        log,
        RETRY_MAX_ATTEMPTS=3,
        STALL_POLL_SEC="0.1",
        STALL_QUIET_LIMIT="3",
    )
    r = _bash(
        f'run_with_retry run_with_stall_watchdog "{cmd}"\necho "EXIT=$?"',
        env=env,
        timeout=30,
    )
    assert _exit_code(r.stdout) == 0, r.stdout + r.stderr
    assert _runs(tmp_path) == 2  # attempt 1 killed by the watchdog, attempt 2 ok
