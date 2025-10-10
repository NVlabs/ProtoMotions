#!/usr/bin/env bash

# End-to-end helper script that converts custom SMPL outputs into a packaged
# MotionLib state ready for ProtoMotions training. Update the configuration
# section before running.

set -euo pipefail

###############################################################################
# Configuration
###############################################################################

PYTHON_BIN=${PYTHON_BIN:-python}


SEQUENCE_NAME="football_high_res"
PREPROCESS_DIR="/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res"
WORK_ROOT="/scratch/izar/cizinsky/zurihack/data/${SEQUENCE_NAME}"

FPS=5
GENDER="neutral"
HUMANOID_TYPE="smpl"
ROBOT_TYPE="h1"

# Leave empty to convert every detected track.
TRACK_IDS=(0)

###############################################################################
# Derived paths (feel free to adjust)
###############################################################################

AMASS_EXPORT_DIR="${WORK_ROOT}/amass_export"
ISAAC_EXPORT_DIR="${WORK_ROOT}/isaac_export"
MOTION_DESC_PATH="${WORK_ROOT}/motion_descriptors/${SEQUENCE_NAME}.yaml"
PACKAGED_OUTPUT="${WORK_ROOT}/motion_states/${SEQUENCE_NAME}.pt"

# The conversion script sanitizes the sequence name; mirror the same logic here.
SANITIZED_SEQUENCE_NAME=${SEQUENCE_NAME// /_}
SANITIZED_SEQUENCE_NAME=${SANITIZED_SEQUENCE_NAME//(/}
SANITIZED_SEQUENCE_NAME=${SANITIZED_SEQUENCE_NAME//)/}
SANITIZED_SEQUENCE_NAME=${SANITIZED_SEQUENCE_NAME//[/}
SANITIZED_SEQUENCE_NAME=${SANITIZED_SEQUENCE_NAME//]/}
CONVERTED_SUBDIR="${SANITIZED_SEQUENCE_NAME}-${ROBOT_TYPE}"

###############################################################################
# Create working directories
###############################################################################

mkdir -p "${AMASS_EXPORT_DIR}"
mkdir -p "${ISAAC_EXPORT_DIR}"
mkdir -p "$(dirname "${MOTION_DESC_PATH}")"
mkdir -p "$(dirname "${PACKAGED_OUTPUT}")"

###############################################################################
# Step 1: Custom SMPL -> AMASS-style .npz
###############################################################################

echo "[1/4] Exporting AMASS-style clips..."
TRACK_ARGS=()
if ((${#TRACK_IDS[@]} > 0)); then
  for tid in "${TRACK_IDS[@]}"; do
    TRACK_ARGS+=(--track-id "${tid}")
  done
fi

"${PYTHON_BIN}" data/scripts/custom_smpl_to_amass.py \
  "${PREPROCESS_DIR}" \
  "${AMASS_EXPORT_DIR}" \
  --sequence-name "${SEQUENCE_NAME}" \
  --fps "${FPS}" \
  --gender "${GENDER}" \
  "${TRACK_ARGS[@]}"

###############################################################################
# Step 2: AMASS -> Isaac/poselib npy
###############################################################################

echo "[2/4] Converting AMASS clips to Isaac format..."
"${PYTHON_BIN}" data/scripts/convert_amass_to_isaac.py \
  --amass-root-dir "${AMASS_EXPORT_DIR}" \
  --output-dir "${ISAAC_EXPORT_DIR}" \
  --humanoid-type "${HUMANOID_TYPE}" \
  --robot-type "${ROBOT_TYPE}" \
  --force-remake \
  --force-retarget

###############################################################################
# Step 3: Build motion descriptor YAML
###############################################################################

echo "[3/4] Creating motion descriptor..."
"${PYTHON_BIN}" data/scripts/create_motion_descriptor.py \
  "${ISAAC_EXPORT_DIR}" \
  "${MOTION_DESC_PATH}" \
  --fps "${FPS}" \
  --sequence-subdir "${CONVERTED_SUBDIR}"

###############################################################################
# Step 4: Package MotionLib state
###############################################################################

echo "[4/4] Packaging MotionLib state..."
"${PYTHON_BIN}" data/scripts/package_motion_lib.py \
  motion_file="${MOTION_DESC_PATH}" \
  amass_data_path="${ISAAC_EXPORT_DIR}" \
  outpath="${PACKAGED_OUTPUT}" \
  humanoid_type="${HUMANOID_TYPE}"

echo "[DONE] Packaged motion saved to ${PACKAGED_OUTPUT}"
