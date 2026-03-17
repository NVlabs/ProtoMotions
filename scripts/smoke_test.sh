#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Smoke test for experiment configs. Runs training for 60s, verifies it
# starts successfully, then kills the process and cleans up.
#
# Runs in the FOREGROUND — output streams to your terminal in real-time.
# Ctrl+C aborts immediately. After TIMEOUT_SEC the process is auto-killed.
#
# Usage:
#   ./scripts/smoke_test.sh <python> <experiment_path> <robot> <simulator> [motion_file]
#
# Examples:
#   ./scripts/smoke_test.sh /home/xxx/code/protomotions/env_isaaclab/bin/python \
#     examples/experiments/mimic/my_config.py g1 isaaclab
#
#   ./scripts/smoke_test.sh /home/xxx/code/protomotions/env_isaaclab/bin/python \
#     examples/experiments/mimic/my_config.py g1 isaaclab /path/to/motions.pt
#
# Exit codes:
#   0 = PASS (training reached Epoch 0, or setup completed without errors)
#   1 = FAIL (crash, import error, or no training output within timeout)

set -o pipefail

PYTHON="$1"
EXPERIMENT_PATH="$2"
ROBOT="$3"
SIMULATOR="$4"

# Default motion files per robot
declare -A DEFAULT_MOTION_FILES=(
    ["g1"]="data/motion_for_trackers/g1_random_subset_tiny.pt"
    ["h1_2"]="data/motion_for_trackers/h1_2_random_subset_tiny.pt"
    ["smpl"]="examples/data/amass_isaac_simple_humanoid_v2_subset_tiny.pt"
)

MOTION_FILE="${5:-${DEFAULT_MOTION_FILES[$ROBOT]:-}}"
if [ -z "$MOTION_FILE" ]; then
    echo "ERROR: No motion file specified and no default for robot '$ROBOT'"
    exit 1
fi

TEST_NAME="smoke_test_$(basename "$EXPERIMENT_PATH" .py)_$$"
CONFIG_BASENAME=$(basename "$EXPERIMENT_PATH")
TIMEOUT_SEC=60
LOG_FILE=$(mktemp /tmp/smoke_test_XXXXXX.log)

echo "=== Smoke Test: $CONFIG_BASENAME ==="
echo "  Python:    $PYTHON"
echo "  Robot:     $ROBOT"
echo "  Simulator: $SIMULATOR"
echo "  Motion:    $MOTION_FILE"
echo "  Timeout:   ${TIMEOUT_SEC}s"
echo "  Log:       $LOG_FILE"
echo ""

# Kill any stale train_agent processes first
STALE_PIDS=$(ps aux | grep "[t]rain_agent" | awk '{print $2}' || true)
if [ -n "$STALE_PIDS" ]; then
    echo "Killing stale train_agent processes: $STALE_PIDS"
    echo "$STALE_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Run training with a hard timeout.
#
# 'timeout --kill-after=5 N cmd' sends SIGTERM after N seconds, then SIGKILL
# after 5 more seconds if the process is still alive. Because the command runs
# as a direct child of 'timeout' (foreground), Ctrl+C reaches the whole
# pipeline naturally and kills everything immediately.
#
# 'tee' captures the full output to LOG_FILE so pass/fail detection works
# even if the process crashes with an unfiltered error message.
timeout --kill-after=5 "$TIMEOUT_SEC" \
    "$PYTHON" -u protomotions/train_agent.py \
        --robot-name "$ROBOT" \
        --simulator "$SIMULATOR" \
        --experiment-path "$EXPERIMENT_PATH" \
        --experiment-name "$TEST_NAME" \
        --motion-file "$MOTION_FILE" \
        --num-envs 1024 \
        --batch-size 8192 \
    2>&1 | tee "$LOG_FILE"

TIMEOUT_EXIT=${PIPESTATUS[0]}
# 0   = process exited cleanly within timeout
# 124 = timed out (SIGTERM sent) — normal for a healthy long-running training
# 137 = killed by SIGKILL (sent by --kill-after=5 after SIGTERM was ignored)

# Force-kill any lingering subprocesses (e.g. IsaacGym/torchrun workers that
# survive SIGTERM on their own). Grep by experiment name which is unique per run.
pkill -9 -f "$TEST_NAME" 2>/dev/null || true

# Clean up results directory
if [ -d "results/$TEST_NAME" ]; then
    rm -rf "results/$TEST_NAME"
fi

echo ""

# Pass/fail detection
if grep -q "Epoch 0, training" "$LOG_FILE" 2>/dev/null; then
    echo "PASS: $CONFIG_BASENAME"
    rm -f "$LOG_FILE"
    exit 0
fi

if [ "$TIMEOUT_EXIT" -eq 0 ] || [ "$TIMEOUT_EXIT" -eq 124 ] || [ "$TIMEOUT_EXIT" -eq 137 ]; then
    if grep -q "Setup complete" "$LOG_FILE" 2>/dev/null; then
        if ! grep -qi "Traceback\|Exception\|CUDA out of memory" "$LOG_FILE" 2>/dev/null; then
            echo "PASS: $CONFIG_BASENAME (setup complete, no errors)"
            rm -f "$LOG_FILE"
            exit 0
        fi
    fi
fi

echo "FAIL: $CONFIG_BASENAME"
echo "Log: $LOG_FILE"
exit 1
