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
WORK_ROOT="/scratch/izar/cizinsky/zurihack/data/"

FPS=30
GENDER="neutral"
HUMANOID_TYPE="smpl"
ROBOT_TYPE="smpl"

# Leave empty to convert every detected track.
TRACK_IDS=(0)

###############################################################################
# Derived paths (feel free to adjust)
###############################################################################

AMASS_EXPORT_DIR="${WORK_ROOT}/amass_export"
MOTION_DESC_PATH="${WORK_ROOT}/motion_descriptors/${SEQUENCE_NAME}.yaml"
PACKAGED_OUTPUT="${WORK_ROOT}/motion_states/${SEQUENCE_NAME}.pt"
DATASET_REPO_ROOT="/scratch/izar/cizinsky/zurihack/"

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
RETARGET_FLAGS=()
if [[ "${ROBOT_TYPE}" == "h1" || "${ROBOT_TYPE}" == "g1" ]]; then
  RETARGET_FLAGS+=(--force-retarget)
fi

"${PYTHON_BIN}" data/scripts/convert_amass_to_isaac.py \
  "${AMASS_EXPORT_DIR}" \
  --robot-type "${ROBOT_TYPE}" \
  --humanoid-type "${HUMANOID_TYPE}" \
  --output-dir "${AMASS_EXPORT_DIR}" \
  --force-remake \
  "${RETARGET_FLAGS[@]}"

###############################################################################
# Step 3: Build motion descriptor YAML
###############################################################################

echo "[3/4] Creating motion descriptor..."
"${PYTHON_BIN}" data/scripts/create_motion_descriptor.py \
  "${AMASS_EXPORT_DIR}" \
  "${MOTION_DESC_PATH}" \
  --fps "${FPS}" \
  --sequence-subdir "${CONVERTED_SUBDIR}"

###############################################################################
# Step 4: Package MotionLib state
###############################################################################

echo "[4/4] Packaging MotionLib state..."
"${PYTHON_BIN}" data/scripts/package_motion_lib.py \
  "${MOTION_DESC_PATH}" \
  "${AMASS_EXPORT_DIR}" \
  "${PACKAGED_OUTPUT}" \
  --humanoid-type "${HUMANOID_TYPE}"

###############################################################################
# Step 5: Commit and push dataset repo to Hugging Face
###############################################################################

if [[ -d "${DATASET_REPO_ROOT}/.git" ]]; then
  echo "[5/5] Syncing dataset repo with Hugging Face..."
  pushd "${DATASET_REPO_ROOT}" >/dev/null
  git add .
  if git diff --cached --quiet; then
    echo "[5/5] No dataset changes to commit."
  else
    COMMIT_MSG="Update ${SEQUENCE_NAME} dataset ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
    git commit -m "${COMMIT_MSG}"
    git push
  fi
  popd >/dev/null
else
  echo "[5/5] Skipping dataset sync; no Git repository found at ${DATASET_REPO_ROOT}"
fi

echo "[DONE] Packaged motion saved to ${PACKAGED_OUTPUT}"
