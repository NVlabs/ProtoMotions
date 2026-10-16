#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# In-job setup + self-resuming training loop for ProtoMotions on DGX Cloud Lepton.
#
# On SLURM, a 4h walltime forces a job array: each array task is a fresh srun that
# resumes from last.ckpt, and train_agent.py runs with --use-slurm so it stops
# itself at ~3.5h before the walltime kill. Lepton jobs have NO walltime, so none
# of that applies -- train_agent.py runs straight to completion, and this loop
# only restarts it after a *crash*.
#
# Resume is filesystem-based and needs no flags: train_agent.py finds
# results/<exp>/last.ckpt in the (unchanged) snapshot cwd and continues. It saves
# last.ckpt at loop entry (epoch 0) and every K epochs thereafter regardless of
# SLURM, which is what makes a plain re-invocation safe. So each loop iteration
# runs the identical command; the first run starts fresh, later runs auto-resume.
#
# The crash-restart policy itself (retry flaky failures, fast-fail a repeated
# Python exception, key progress off last.ckpt's mtime) lives in the shared
# run_with_retry in retry_lib.sh, so it stays in lockstep with train_slurm.py.
#
# Positional args (main path):
#   1  PYTHON      interpreter invocation, e.g. "/workspace/isaaclab/isaaclab.sh -p"
#   2  SIMULATOR   isaacgym|isaaclab|newton (selects install path + IsaacLab warmup)
#   3  EXP_NAME    experiment name; checkpoints live at results/<EXP_NAME>/last.ckpt
#   4  EXP_FOLDER  code snapshot on AMLFS; cwd for training, holds results/
#   5  MAX_RUNS    hard cap on total loop iterations (RETRY_TOTAL_ATTEMPTS)
#   6+ the train_agent.py argument tail, forwarded verbatim as "$@"
#
# Each positional is shlex-quoted individually by train_lepton.py, so the
# container shell splits the command back into exactly these arguments and the
# train_agent.py tail reaches "$@" without a value being word-split on a space.
#
# Env knobs (defaults in parentheses; passed by train_lepton.py):
#   STALL_TIMEOUT_MIN (30)   kill training if the durable log stops growing for
#                            this many minutes; 0 disables.
#   RETRY_MAX_ATTEMPTS (3)   give up after this many consecutive runs that do not
#                            advance last.ckpt (deterministic failure).
#   RETRY_REPEAT_LIMIT (3)   give up sooner if the SAME Python exception repeats
#                            this many runs in a row (the fast-fail-on-python path).
# Test-only knobs (see protomotions/tests/test_lepton_train_loop.py):
#   LEPTON_LOOP_SOURCE_ONLY  when set, `source` this file to get the shared
#                            functions (kill_tree/run_with_stall_watchdog/
#                            run_with_retry/...) WITHOUT running the main path.
#   LEPTON_LOOP_SKIP_SETUP   skip the per-container install + IsaacLab warmup, so
#                            the loop itself can be exercised with a fake python.
#   STALL_POLL_SEC (60)      watchdog poll interval; sub-second in tests.
#   STALL_QUIET_LIMIT        watchdog no-growth polls before killing (default:
#                            STALL_TIMEOUT_MIN, i.e. the user knob is in minutes).

set -uo pipefail

# ===========================================================================
# Shared retry/watchdog primitives.
# ===========================================================================
# kill_tree, run_with_stall_watchdog, run_with_retry and ckpt_stamp live in
# retry_lib.sh so this loop and the SLURM launcher share one implementation.
# Source by absolute path from THIS script's own directory so it resolves no
# matter the cwd, and so a LEPTON_LOOP_SOURCE_ONLY test that sources this file
# gets those functions transitively.
_LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=retry_lib.sh
source "${_LOOP_DIR}/retry_lib.sh"

# When a test sources this file just for the functions above, stop here -- do not
# parse positionals, install, or train.
if [ -n "${LEPTON_LOOP_SOURCE_ONLY:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

# ===========================================================================
# main
# ===========================================================================

PYTHON="${1:?python invocation}"
SIMULATOR="${2:?simulator}"
EXP_NAME="${3:?experiment name}"
EXP_FOLDER="${4:?exp folder}"
MAX_RUNS="${5:?max runs}"
shift 5  # remaining "$@" is the train_agent.py argument tail

cd "$EXP_FOLDER" || { echo "[lepton-loop] cannot cd to $EXP_FOLDER" >&2; exit 1; }

# Durable log on AMLFS: lep's live logs are tail-only and vanish once the
# container exits, so mirror all output to a file under the snapshot too (the
# SLURM path has slurm_output.log). No background readers here, so tee gets EOF
# and exits cleanly with the script. Appended across loop iterations and across
# any preemption relaunch. It is also the growth signal the stall watchdog reads.
LOGFILE="${EXP_FOLDER}/lepton_train.log"
exec > >(tee -a "$LOGFILE") 2>&1

# A give-up sentinel from a PRIOR attempt (deterministic failure, no progress).
# Exit NONZERO so this relaunch is marked failed rather than falsely successful:
# exiting 0 here would let the job's final state read as "completed" even though
# training never succeeded, which is dangerous for any automation watching the job
# status. train_lepton.py clears this sentinel when you deliberately resume with
# --exp-folder; remove it by hand otherwise to allow retries. Checked here, before
# the expensive install+warmup, so a known-dead relaunch exits fast (run_with_retry
# re-checks it too, but only after setup).
GAVE_UP="${EXP_FOLDER}/.lepton_loop_gave_up"
if [ -f "$GAVE_UP" ]; then
    echo "[lepton-loop] a previous attempt gave up (deterministic failure; see the" \
        "log above). Exiting NONZERO so this job is marked failed, not falsely" \
        "successful. Remove ${GAVE_UP} (or resubmit via train_lepton.py --exp-folder," \
        "which clears it) to allow retries." >&2
    exit 1
fi

# Map the Lepton knobs onto the shared run_with_retry environment. Continue on
# progress: with no walltime, resume in place until training completes, capped at
# MAX_RUNS total iterations. RETRY_BACKOFF_SEC=0 preserves the original
# immediate-resume timing.
CKPT="results/${EXP_NAME}/last.ckpt"
export CKPT LOGFILE GAVE_UP
export RETRY_MAX_ATTEMPTS="${RETRY_MAX_ATTEMPTS:-3}"
export RETRY_TOTAL_ATTEMPTS="${MAX_RUNS}"
export RETRY_REPEAT_LIMIT="${RETRY_REPEAT_LIMIT:-3}"
export RETRY_ON_PROGRESS="continue"
export RETRY_BACKOFF_SEC="${RETRY_BACKOFF_SEC:-0}"
export STALL_TIMEOUT_MIN="${STALL_TIMEOUT_MIN:-30}"

echo "[lepton-loop] host=$(hostname) exp=${EXP_NAME}"
echo "[lepton-loop] folder=${EXP_FOLDER}"
echo "[lepton-loop] durable log: ${LOGFILE}"
echo "[lepton-loop] python=${PYTHON} simulator=${SIMULATOR} max_runs=${MAX_RUNS}" \
    "no_progress_limit=${RETRY_MAX_ATTEMPTS} repeat_limit=${RETRY_REPEAT_LIMIT}" \
    "stall_timeout_min=${STALL_TIMEOUT_MIN}"

# --- one-time per-container setup -------------------------------------------
# Import the staged snapshot as an editable install; runtime deps already live in
# the image (--no-deps). IsaacLab's bare python/pip are not runnable, so pip must
# go through the isaaclab.sh wrapper -- a plain `pip install` there exits 127.
# Skipped under LEPTON_LOOP_SKIP_SETUP so a test can drive the loop with a fake
# python that needs no install.
if [ -z "${LEPTON_LOOP_SKIP_SETUP:-}" ]; then
    export PYTHONUNBUFFERED=1
    echo "[lepton-loop] installing protomotions from the snapshot..."
    if [ "$SIMULATOR" = "isaaclab" ]; then
        /workspace/isaaclab/isaaclab.sh -p -m pip uninstall -y protomotions || true
        /workspace/isaaclab/isaaclab.sh -p -m pip install -e . --no-deps || {
            echo "[lepton-loop] protomotions install failed" >&2; exit 1; }
        # NB: do NOT `pip install --upgrade wandb` here. The image's wandb already
        # satisfies protomotions' pin (<0.24); upgrading pulls wandb 0.29 (violates the
        # pin) and drags click to 8.5, which conflicts with isaacsim-kernel's click==8.1.7.
    else
        pip uninstall -y protomotions 2>/dev/null || true
        pip install -e . --no-dependencies || {
            echo "[lepton-loop] protomotions install failed" >&2; exit 1; }
    fi

    # IsaacLab lazily bootstraps a shared pip env the first time AppLauncher starts.
    # Warm it once here so the Fabric-spawned worker ranks (multi-GPU) do not race on
    # it. A single process runs this loop per node, so no rank coordination is needed
    # -- unlike the SLURM path, where srun launches one task per GPU.
    if [ "$SIMULATOR" = "isaaclab" ]; then
        echo "[lepton-loop] warming IsaacLab AppLauncher..."
        /workspace/isaaclab/isaaclab.sh -p -c \
            "from isaaclab.app import AppLauncher; app = AppLauncher({'headless': True}).app; app.close()" || {
            echo "[lepton-loop] IsaacLab warmup failed" >&2; exit 1; }
    fi
else
    echo "[lepton-loop] LEPTON_LOOP_SKIP_SETUP set -- skipping install + warmup (test)."
    export PYTHONUNBUFFERED=1
fi

# --- distributed rendezvous (multi-node) ------------------------------------
# Lightning Fabric's environment reads MASTER_ADDR / MASTER_PORT / NODE_RANK; on
# Lepton those are not set for us, so derive them from the Lepton worker env when
# --nodes > 1 (one worker == one node). Single-node binds to loopback -- sibling
# ranks cannot resolve the container hostname on Lepton. NB: train_lepton.py
# currently REFUSES --nodes > 1, so in practice NNODES is 1 here; this branch is
# groundwork kept ready for when multi-node is qualified.
NNODES="${LEPTON_JOB_TOTAL_WORKERS:-1}"
export NODE_RANK="${LEPTON_JOB_WORKER_INDEX:-0}"
if [ "$NNODES" -gt 1 ]; then
    export MASTER_ADDR="${MASTER_ADDR:-${LEPTON_JOB_WORKER_HOSTNAME_PREFIX}-0.${LEPTON_SUBDOMAIN}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
    # GPU Direct RDMA is off by default on these nodes; without it cross-node NCCL
    # runs ~9x slower. SYS engages GDR via dmabuf. NB: this only helps when both
    # workers land in the SAME network group -- a cross-group 2-node job still
    # hangs/degrades, which the job's node selection (not this env) must prevent.
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
else
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
echo "[lepton-loop] rendezvous: NNODES=${NNODES} NODE_RANK=${NODE_RANK}" \
    "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# --- run --------------------------------------------------------------------
# run_with_retry resumes after a crash; run_with_stall_watchdog (inside it) kills a
# hung train_agent.py so the loop sees an exit. $PYTHON is "cmd -flag" -- the
# word-split is intended; the train tail is quoted "$@" so each argument survives.
# shellcheck disable=SC2086
run_with_retry run_with_stall_watchdog $PYTHON -u protomotions/train_agent.py "$@"
exit $?
