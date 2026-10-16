# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared cluster retry/watchdog primitives, SOURCED by both launchers so their
# retry logic cannot drift:
#   - train_slurm.py's generated sbatch sources this at the batch layer (to wrap
#     srun in run_with_retry) and again inside srun (to wrap train_agent.py in
#     run_with_stall_watchdog).
#   - lepton_train_loop.sh sources it once (in-job) for both.
#
# This file is only ever sourced, never executed. It defines four primitives:
#
#   kill_tree <pid>
#       Kill a process AND all descendants, children first. Only reaches processes
#       still reachable through parent/child links.
#
#   reap_process_group <pgid>
#       Kill whatever is left in a process group, TERM then KILL. Complements
#       kill_tree, which cannot follow a process that has already been reparented
#       to init -- a reparented process keeps its process group, so this catches
#       the ranks a crashed attempt leaves behind.
#
#   run_with_stall_watchdog <cmd...>       -- THE one watchdog.
#       Run <cmd...>; if $LOGFILE stops growing for a while, kill its whole tree
#       so the caller sees an exit instead of a hang. This is the ONLY liveness
#       check: a hung scene build is caught here (it is armed from process start,
#       before training), and for a known-flaky scene build you lower its timeout
#       (STALL_TIMEOUT_MIN) rather than adding a second, startup-only watchdog.
#
#   run_with_retry <cmd...>
#       Run <cmd...> repeatedly. <cmd...> is expected to itself run training under
#       run_with_stall_watchdog (Lepton passes it inline; SLURM's srun wraps it),
#       so a hang is already killed by that watchdog and this loop only ever sees
#       the resulting exit code. Per attempt:
#         exit 0                             -> done.
#         $CKPT's mtime advanced this attempt (training saved a checkpoint):
#             RETRY_ON_PROGRESS=break        -> stop; the SLURM array's next task
#                                               resumes from the checkpoint.
#             RETRY_ON_PROGRESS=continue     -> reset the no-progress streak and
#                                               loop; Lepton resumes in place.
#         no progress, SAME Python exception class RETRY_REPEAT_LIMIT times in a
#         row                                -> deterministic bug: give up, surface
#                                               it (this is the Python fast-fail).
#         no progress, native abort / new exception
#                                            -> flaky: retry, bounded.
#
# Env knobs for run_with_retry (all optional):
#   RETRY_MAX_ATTEMPTS   (1)   consecutive no-progress attempts before giving up.
#   RETRY_TOTAL_ATTEMPTS (=MAX_ATTEMPTS)  hard cap on total attempts; only matters
#                              in continue mode, where progress resets the
#                              no-progress streak and could otherwise loop forever.
#   RETRY_REPEAT_LIMIT   (3, 0=off)  same-exception-class attempts before fast-fail.
#   RETRY_BACKOFF_SEC    (15)  sleep between no-progress retries.
#   RETRY_ON_PROGRESS    (continue|break)  see above.
#   CKPT                       last.ckpt path; its mtime is the progress signal.
#   LOGFILE                    log to slice for the exception-class check.
#   GAVE_UP                    optional sentinel path. When set, a give-up touches
#                              it and a pre-existing one short-circuits to a nonzero
#                              exit (so a relaunch is marked failed, not falsely
#                              successful). Unset -> no sentinel (the SLURM array
#                              re-runs and fast-fails again per task instead).
#
# Env knobs for run_with_stall_watchdog: STALL_TIMEOUT_MIN (30, 0=off) is the
# user-facing one (no-growth minutes); STALL_POLL_SEC (60) and STALL_QUIET_LIMIT
# (=STALL_TIMEOUT_MIN) let a test drive it in fractions of a second.
# STALL_IGNORE_REGEX overrides which lines are treated as dead-rank noise rather
# than progress, and REAP_GRACE_SEC (5) is how long survivors get between the
# reaper's SIGTERM and its SIGKILL.
#
# Test hook: RETRY_LIB_SOURCE_ONLY is accepted for symmetry with
# lepton_train_loop.sh, but since this file has no main path it is a no-op here --
# a test just sources the file and calls the functions.

ckpt_stamp() {
    # $CKPT's mtime in epoch-seconds, or "none" while it does not exist yet.
    # base_agent writes last.ckpt at loop entry (epoch 0), so this advances the
    # moment training is reached and a long-first-epoch run still leaves a resume
    # point before its first periodic save.
    [ -n "${CKPT:-}" ] && [ -f "$CKPT" ] && stat -c %Y "$CKPT" 2>/dev/null || echo "none"
}

kill_tree() {
    # Kill a process AND all its descendants, children first. The training python
    # and its Fabric worker ranks live several levels below the process we
    # background; killing only the parent reparents the grandchildren to init --
    # they keep the GPUs and MASTER_PORT, and the immediate resume then collides.
    # Recurse depth-first so a parent is never killed before its children.
    local root=$1 child
    for child in $(pgrep -P "$root" 2>/dev/null); do
        kill_tree "$child"
    done
    kill -9 "$root" 2>/dev/null
}

reap_process_group() {
    # Kill anything still alive in process group $1 (== the attempt's pid, because
    # run_with_stall_watchdog starts it under "set -m"). Complements kill_tree:
    # kill_tree walks the parent/child links and so cannot reach a process that
    # has already been reparented to init, whereas a reparented process keeps its
    # process group. Safe to call when the group is already empty.
    #
    # SIGTERM first so a rank can release its GPU context, then SIGKILL whatever
    # ignored it. Without the grace period a killed rank can leave the device in a
    # state the next attempt trips over.
    local pgid=$1
    kill -- -"$pgid" 2>/dev/null || return 0
    local waited=0
    while kill -0 -- -"$pgid" 2>/dev/null && [ "$waited" -lt "${REAP_GRACE_SEC:-5}" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    kill -9 -- -"$pgid" 2>/dev/null
    return 0
}

run_with_stall_watchdog() {
    # Run "$@", and if $LOGFILE stops growing for STALL_QUIET_LIMIT consecutive
    # polls of STALL_POLL_SEC seconds, kill the whole process tree so the caller
    # restarts instead of hanging. A wedged rank -- one stuck at setup while the
    # others block at a distributed barrier -- emits no output and no exit code,
    # but a blocked collective silences the HEALTHY ranks too, so "the durable log
    # stopped growing" is a reliable whole-job-wedge signal. Defaults poll every
    # 60s with the limit in minutes (STALL_TIMEOUT_MIN); a limit <= 0 disables it.
    #
    # That "silences the healthy ranks too" assumption holds when a rank is WEDGED
    # (alive but stuck). It does NOT hold when a rank DIES: the survivors then stop
    # blocking and start emitting NCCL heartbeat / TCPStore "Broken pipe" errors
    # several lines a second, forever. Measured on a real 4-rank job where ranks 0
    # and 1 vanished silently and the log kept growing at that rate -- so the raw
    # file size never stopped changing and the watchdog could never fire. The one
    # failure mode most needing a kill was the one it slept through.
    #
    # So measure PROGRESS, not bytes: count only lines that are not dead-rank
    # noise. A half-dead job then looks correctly quiet, while a healthy job (whose
    # progress lines are not filtered) is unaffected. Checkpoint mtime is not usable
    # as the liveness signal here -- last.ckpt does not exist during the scene build
    # this watchdog exists to cover, and save_last_checkpoint_every defaults to 10
    # epochs, far coarser than the timeout.
    local poll="${STALL_POLL_SEC:-60}"
    local quiet_limit="${STALL_QUIET_LIMIT:-${STALL_TIMEOUT_MIN:-30}}"
    # Overridable so a site with different noise can extend it, and so tests can
    # drive it. Anchored where possible to avoid filtering real content.
    local noise="${STALL_IGNORE_REGEX:-ProcessGroupNCCL|TCPStore|Broken pipe|^frame #|^Exception raised from|should dump}"
    # Run the attempt in its OWN PROCESS GROUP ("set -m" makes a background job a
    # group leader, so PGID == pid). When one rank aborts, the launcher does not
    # always reap its siblings; those survivors reparent to init, and kill_tree's
    # pgrep -P walk cannot follow them once the direct child is gone. A process
    # group can: orphans keep their PGID.
    #
    # Measured leak this closes: on a Lepton job, attempts 1, 2 and 4 each left
    # ranks behind, so by attempt 5 several stale IsaacLab processes were still
    # holding GPU contexts and PhysX buffers -- and spamming NCCL errors once a
    # second, which is what drowned the healthy run's output. kill_tree's own
    # comment already predicted this ("they keep the GPUs and MASTER_PORT, and the
    # immediate resume then collides"); the guard was simply never wired into the
    # crash path, only into the watchdog's.
    set -m
    "$@" &
    local pid=$!
    set +m
    if [ "${quiet_limit}" -le 0 ]; then
        wait "$pid"
        local status_no_wd=$?
        reap_process_group "$pid"
        return "$status_no_wd"
    fi
    (
        last=-1
        quiet=0
        while kill -0 "$pid" 2>/dev/null; do
            sleep "$poll"
            # Count of non-noise lines, not file size: dead-rank NCCL spam must not
            # read as progress. grep -a so a binary-looking log still counts.
            size=$(grep -acvE "$noise" "$LOGFILE" 2>/dev/null || echo 0)
            if [ "$size" != "$last" ]; then
                last=$size
                quiet=0
            else
                quiet=$((quiet + 1))
            fi
            if [ "$quiet" -ge "${quiet_limit}" ]; then
                echo "[watchdog] ${LOGFILE} has produced no non-noise output for" \
                    "${quiet_limit} poll(s)" \
                    "of ${poll}s -- training looks wedged (a rank stuck at" \
                    "setup/barrier?). Killing pid ${pid} and its descendants so the" \
                    "caller restarts instead of burning the walltime." >&2
                kill_tree "$pid"
                break
            fi
        done
    ) &
    local watchdog=$!
    wait "$pid"
    local status=$?
    kill "$watchdog" 2>/dev/null
    wait "$watchdog" 2>/dev/null
    reap_process_group "$pid"
    return "$status"
}

run_with_retry() {
    local nomax="${RETRY_MAX_ATTEMPTS:-1}"
    local total="${RETRY_TOTAL_ATTEMPTS:-$nomax}"
    local repeat_limit="${RETRY_REPEAT_LIMIT:-3}"
    local backoff="${RETRY_BACKOFF_SEC:-15}"
    local on_progress="${RETRY_ON_PROGRESS:-continue}"
    # Always run at least once: a 0/negative value must not skip the command and
    # report a false success. Guard the total the same way, but do NOT raise it to
    # nomax -- an explicit small total (e.g. Lepton's --max-runs below
    # --no-progress-limit) is a legitimate hard cap to honour, whichever bound trips
    # first.
    [ "$nomax" -lt 1 ] && nomax=1
    [ "$total" -lt 1 ] && total=1

    # A give-up sentinel from a PRIOR attempt (deterministic failure, no progress).
    # Exit NONZERO so this relaunch is marked failed rather than falsely
    # successful: exiting 0 here would let a job's final state read as "completed"
    # even though training never succeeded, which is dangerous for automation
    # watching the job status.
    if [ -n "${GAVE_UP:-}" ] && [ -f "$GAVE_UP" ]; then
        echo "[retry] a previous attempt gave up (deterministic failure; see the log" \
            "above). Exiting NONZERO so this job is marked failed, not falsely" \
            "successful. Remove ${GAVE_UP} to allow retries." >&2
        return 1
    fi

    local attempt=0 no_progress=0 status=0 prev_cls="" repeat=0
    local before after _mark _exc _cls
    while [ "$attempt" -lt "$total" ]; do
        attempt=$((attempt + 1))
        before=$(ckpt_stamp)
        # Read only THIS attempt's slice of the shared log for the exception check:
        # everything from line _mark+1 on.
        if [ -f "${LOGFILE:-}" ]; then _mark=$(wc -l < "$LOGFILE"); else _mark=0; fi

        "$@"
        status=$?

        if [ "$status" -eq 0 ]; then
            echo "[retry] attempt ${attempt} exited 0 -- done."
            return 0
        fi

        after=$(ckpt_stamp)
        if [ "$after" != "$before" ]; then
            # Progress: training saved a checkpoint during this attempt, so it got
            # past setup. A failure now is a mid-run crash / walltime kill, resumed
            # from the checkpoint -- never a flaky scene build to bang on in place.
            if [ "$on_progress" = "break" ]; then
                echo "[retry] attempt ${attempt} exited ${status} after a checkpoint was" \
                    "written -- training was reached; not retrying in place (the array" \
                    "element resumes from the checkpoint)." >&2
                return "$status"
            fi
            no_progress=0
            prev_cls=""
            repeat=0
            echo "[retry] attempt ${attempt} crashed (exit ${status}); checkpoint" \
                "advanced -- resuming."
            continue
        fi

        # No progress. Classify: a native abort/hang prints no Python traceback, so
        # retry it freely (bounded). A DETERMINISTIC bug prints the same exception
        # every attempt -- stop and surface it rather than hide it behind retries.
        # Match message-LESS exceptions too (a bare `assert` prints just
        # "AssertionError", no colon -- a common deterministic startup bug). Compare
        # only the exception CLASS (strip the message): a message that embeds a
        # path, CUDA device id or rank varies between attempts, but the bug is one.
        no_progress=$((no_progress + 1))
        # ${LOGFILE:-} (not bare $LOGFILE) so this cannot abort under `set -u` if a
        # caller ever leaves LOGFILE unset -- it just yields an empty slice, routing
        # to the native-abort (retry) path. Mirrors the guard on the _mark read above.
        _exc=$(tail -n +$((_mark + 1)) "${LOGFILE:-}" 2>/dev/null | grep -oE '^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)(: .*)?$' | tail -1)
        _cls="${_exc%%:*}"
        if [ -n "$_exc" ] && [ "$_cls" = "$prev_cls" ]; then
            repeat=$((repeat + 1))
        elif [ -n "$_exc" ]; then
            repeat=1
        fi
        prev_cls="$_cls"
        if [ -n "$_exc" ] && [ "$repeat_limit" -gt 0 ] && [ "$repeat" -ge "$repeat_limit" ]; then
            echo "[retry] same exception ${repeat} attempts in a row -- this is" \
                "deterministic, not flaky:" >&2
            echo "[retry]   ${_exc}" >&2
            [ -n "${GAVE_UP:-}" ] && touch "$GAVE_UP"
            return "$status"
        fi
        if [ "$no_progress" -ge "$nomax" ]; then
            echo "[retry] ${no_progress} attempt(s) in a row made no progress -- almost" \
                "certainly a deterministic failure. Giving up -- see the traceback(s)" \
                "above." >&2
            [ -n "${GAVE_UP:-}" ] && touch "$GAVE_UP"
            return "$status"
        fi
        echo "[retry] attempt ${attempt} failed (exit ${status}) with no progress;" \
            "retrying in ${backoff}s in case it was transient." >&2
        sleep "$backoff"
    done

    # Ran out of the total budget while still making occasional progress but never
    # completing. Not necessarily deterministic, so do NOT write the sentinel -- a
    # fresh relaunch should be free to continue from the last checkpoint.
    echo "[retry] exhausted ${total} attempt(s) without completing." >&2
    return "$status"
}
