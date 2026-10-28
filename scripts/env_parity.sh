#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# BaseEnv parity check between a base git ref and this checkout.
#
#   scripts/env_parity.sh <base-ref> [scenario ...]
#
# Exports the base code (protomotions + examples, no LFS payloads) under
# $TMPDIR and copies THIS checkout's protomotions/tests/parity, tests/support
# and tests/fixtures into it: the harness, the fake simulator and its MJCF are
# harness code, and both sides must run the same double. Then, for every
# scenario (default: all), records the base twice and requires the two to be
# fully identical, values AND call order ("BASE NOT DETERMINISTIC" otherwise),
# records the head once and compares base vs head. One line per scenario:
#
#   IDENTICAL           values and call order agree
#   CALL ORDER CHANGED  values agree, the order of instrumented calls differs
#   DIVERGENCE          a value differs
#
# Exit 0 when every scenario is IDENTICAL, 1 when any value diverged, 2 when
# only call orders changed. A compare that crashes is reported as "compare
# failed" and fails the script (it is not non-determinism). The three traces
# of every non-identical scenario are kept under
# ${PARITY_KEEP_DIR:-$PWD/parity_traces}/<scenario>/; the base export and
# everything else are deleted on exit. PYTHON overrides the interpreter;
# PARITY_COMPARE_ARGS passes extra flags to every compare (e.g. "--atol 1e-6"
# for the GPU mode). PARITY_BASE_OWN_SCENARIOS=1 keeps the base's own
# scenarios.py (see the note by the harness copy below) for a PR that changes
# the harness's build call; default 0 copies the head's as usual. Under it, a
# scenario the base's own scenarios.py does not know (one the PR adds) is
# recorded on the head once, so it is known to run, reported as HEAD-ONLY and
# not compared; a scenario only the base knows (one the PR renamed or removed)
# is reported as BASE-ONLY and not run; neither changes the exit code, but a
# run in which no scenario was compared at all fails (exit 1), so a PR that
# renames every scenario cannot be green with zero comparisons.
# PARITY_BASE_OWN_SUPPORT=1 likewise keeps the base's own tests/support and
# tests/fixtures for a PR that changes the shared double's API; default 0.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <base-ref> [scenario ...]" >&2
    exit 2
fi
BASE_REF=$1
shift

PYTHON=${PYTHON:-python}
HEAD_DIR=$(pwd)
MIN_FREE_GB=60
TMP_ROOT=${TMPDIR:-/tmp}
KEEP_ROOT=${PARITY_KEEP_DIR:-$HEAD_DIR/parity_traces}
read -r -a compare_args <<< "${PARITY_COMPARE_ARGS:-}"

free_gb=$(df -BG --output=avail "$TMP_ROOT" | tail -1 | tr -dc '0-9')
if [[ -z "$free_gb" || "$free_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "refusing: only ${free_gb:-?} GB free on the filesystem of $TMP_ROOT, need ${MIN_FREE_GB} GB" >&2
    exit 1
fi

if [[ ! -d "$HEAD_DIR/protomotions/tests/parity" ]]; then
    echo "run from the repository root (protomotions/tests/parity not found)" >&2
    exit 1
fi

BASE_DIR=$(mktemp -d "$TMP_ROOT/env_parity_base.XXXXXX")
TRACE_DIR=$(mktemp -d "$TMP_ROOT/env_parity_traces.XXXXXX")
trap 'rm -rf "$BASE_DIR" "$TRACE_DIR"' EXIT

export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
mkdir -p "$TRACE_DIR/mpl"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$TRACE_DIR/mpl}

echo "exporting $BASE_REF ($(git rev-parse --short "$BASE_REF")) to $BASE_DIR"
git -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false \
    archive "$BASE_REF" protomotions examples | tar -x -C "$BASE_DIR"
# The base tree predates (or differs in) the harness and the test doubles it
# runs on; both sides must run this checkout's copies.
echo "copying protomotions/tests/{parity,support,fixtures} from the head into the base export"
for harness_dir in protomotions/tests/parity protomotions/tests/support protomotions/tests/fixtures; do
    rm -rf "$BASE_DIR/$harness_dir"
    cp -R "$HEAD_DIR/$harness_dir" "$BASE_DIR/$harness_dir"
done

# PARITY_BASE_OWN_SCENARIOS=1: restore the base's own scenarios.py after the copy
# above. Use this ONLY for a PR that changes the harness's own build CALL (e.g.
# build_standard_configs -> build_experiment): head's scenarios.py then cannot
# import against the base code, so each side builds its own way while the rest of
# the harness (recorder, spies, support, fixtures) stays identical from the head.
# The comparison is still valid because only the build call differs. Default (0)
# is unchanged; name this flag in the PR body when you use it.
if [[ "${PARITY_BASE_OWN_SCENARIOS:-0}" == "1" ]]; then
    echo "PARITY_BASE_OWN_SCENARIOS=1: the base keeps its own scenarios.py (harness build-call change)"
    git show "$BASE_REF:protomotions/tests/parity/scenarios.py" \
        > "$BASE_DIR/protomotions/tests/parity/scenarios.py"
fi

# PARITY_BASE_OWN_SUPPORT=1: restore the base's own tests/support and tests/fixtures
# after the harness copy above. Use this ONLY for a PR that changes the shared
# double's API (e.g. the sensor config classes make_sensor_test_env builds): the
# base then builds its fake env through its own double while the recorder, spies
# and the rest of protomotions/tests/parity still come from the head. The fake
# simulator's physics and the scenario definitions are unchanged, so the
# comparison stays valid. Default (0) is unchanged; name this flag in the PR body.
if [[ "${PARITY_BASE_OWN_SUPPORT:-0}" == "1" ]]; then
    echo "PARITY_BASE_OWN_SUPPORT=1: the base keeps its own tests/support and tests/fixtures (shared-double API change)"
    for support_dir in protomotions/tests/support protomotions/tests/fixtures; do
        rm -rf "$BASE_DIR/$support_dir"
        git -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false \
            archive "$BASE_REF" "$support_dir" | tar -x -C "$BASE_DIR"
    done
fi

explicit_list=0
if [[ $# -gt 0 ]]; then
    scenarios=("$@")
    explicit_list=1
else
    # Captured into a variable first so the interpreter's failure propagates.
    names=$(cd "$HEAD_DIR" && PYTHONPATH=$HEAD_DIR "$PYTHON" -m protomotions.tests.parity list --names)
    mapfile -t scenarios <<< "$names"
fi
if [[ ${#scenarios[@]} -eq 0 || -z "${scenarios[0]}" ]]; then
    echo "no scenarios to run" >&2
    exit 1
fi
# With the base's own scenarios.py, the base may not know a scenario the head
# adds; those are recorded on the head only (see the loop).
base_names=""
if [[ "${PARITY_BASE_OWN_SCENARIOS:-0}" == "1" ]]; then
    base_names=$(cd "$BASE_DIR" && PYTHONPATH=$BASE_DIR "$PYTHON" -m protomotions.tests.parity list --names)
fi

# `python -m` puts the current directory first on sys.path, so each side runs
# from its own tree; PYTHONPATH alone would not pick the base over the head.
record_in() {
    local dir=$1 scenario=$2 out=$3
    (cd "$dir" && PYTHONPATH=$dir "$PYTHON" -m protomotions.tests.parity record \
        --scenario "$scenario" --out "$out" > "$out.log" 2>&1) \
        || { echo "record failed ($dir, $scenario); last lines of $out.log:" >&2; tail -20 "$out.log" >&2; return 1; }
}

# Prints the verdict line and returns compare's exit code: 0 identical,
# 1 value divergence, 2 call order only, 3 the compare itself failed (its
# traceback is then shown).
compare_in() {
    local a=$1 b=$2 err=$TRACE_DIR/compare.err out status=0
    out=$(cd "$HEAD_DIR" && PYTHONPATH=$HEAD_DIR "$PYTHON" -m protomotions.tests.parity compare \
        "$a" "$b" ${compare_args[@]+"${compare_args[@]}"} 2>"$err") || status=$?
    if [[ $status -eq 3 || -z "$out" ]]; then
        cat "$err" >&2
        status=3
    fi
    printf '%s\n' "$out"
    return $status
}

keep_traces() {
    local scenario=$1 dir=$KEEP_ROOT/$scenario
    mkdir -p "$dir"
    mv "$TRACE_DIR/$scenario.base1.pt" "$TRACE_DIR/$scenario.base2.pt" "$TRACE_DIR/$scenario.head.pt" "$dir/"
    echo "[$scenario] traces kept: $dir/$scenario.base1.pt $dir/$scenario.base2.pt $dir/$scenario.head.pt"
}

summary=()
status=0
compared=0
for scenario in "${scenarios[@]}"; do
    base1=$TRACE_DIR/$scenario.base1.pt
    base2=$TRACE_DIR/$scenario.base2.pt
    head=$TRACE_DIR/$scenario.head.pt
    if [[ "${PARITY_BASE_OWN_SCENARIOS:-0}" == "1" ]] && ! grep -qxF "$scenario" <<< "$base_names"; then
        record_in "$HEAD_DIR" "$scenario" "$head"
        echo "[$scenario] head-only: the base ref's scenarios.py has no scenario of this name; recorded on the head once, not compared"
        echo "[$scenario] head record: $(tail -n 1 "$head.log")"
        summary+=("$scenario: HEAD-ONLY (recorded on the head, not compared)")
        rm -f "$head" "$head.log"
        continue
    fi
    record_in "$BASE_DIR" "$scenario" "$base1"
    record_in "$BASE_DIR" "$scenario" "$base2"
    base_status=0
    base_check=$(compare_in "$base1" "$base2") || base_status=$?
    echo "[$scenario] base twice: $base_check"
    if [[ $base_status -eq 3 ]]; then
        echo "[$scenario] compare failed -- see the traceback above"
        exit 1
    elif [[ $base_status -ne 0 ]]; then
        echo "[$scenario] BASE NOT DETERMINISTIC -- the base ref does not reproduce itself (values or call order); the head is not to blame"
        exit 1
    fi
    record_in "$HEAD_DIR" "$scenario" "$head"
    head_status=0
    head_check=$(compare_in "$base1" "$head") || head_status=$?
    echo "[$scenario] base vs head: $head_check"
    compared=$((compared + 1))
    case $head_status in
        0)
            summary+=("$scenario: IDENTICAL")
            ;;
        1)
            summary+=("$scenario: DIVERGENCE -- $head_check")
            status=1
            keep_traces "$scenario"
            ;;
        2)
            summary+=("$scenario: CALL ORDER CHANGED -- $head_check")
            [[ $status -eq 1 ]] || status=2
            keep_traces "$scenario"
            ;;
        *)
            echo "[$scenario] compare failed -- see the traceback above"
            exit 1
            ;;
    esac
    rm -f "$base1" "$base2" "$head" "$base1.log" "$base2.log" "$head.log"
done

# Without an explicit list, a scenario only the base knows (renamed or removed
# by the PR) is reported, not silently skipped.
if [[ $explicit_list -eq 0 && -n "$base_names" ]]; then
    while IFS= read -r base_scenario; do
        [[ -z "$base_scenario" ]] && continue
        if ! printf '%s\n' "${scenarios[@]}" | grep -qxF "$base_scenario"; then
            echo "[$base_scenario] base-only: the head's scenarios.py has no scenario of this name; not run"
            summary+=("$base_scenario: BASE-ONLY (not run)")
        fi
    done <<< "$base_names"
fi

echo "---- env parity: $BASE_REF vs head ----"
printf '%s\n' "${summary[@]}"
if [[ $compared -eq 0 ]]; then
    echo "no scenario was compared (every scenario was HEAD-ONLY or BASE-ONLY)" >&2
    exit 1
fi
exit $status
