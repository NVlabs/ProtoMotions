#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to retarget AMASS SMPL motions to a robot (G1 or H1_2)
#
# IMPORTANT: ProtoMotions and PyRoki require separate Python environments.
# You must provide paths to both Python interpreters.
#
# Usage: ./scripts/retarget_amass_to_robot.sh <proto_python> <pyroki_python> <amass_pt_file> <robot_type> [skip_freq]
#
# Example:
#   ./scripts/retarget_amass_to_robot.sh \
#       ~/miniconda3/envs/protomotions/bin/python \
#       ~/miniconda3/envs/pyroki/bin/python \
#       /path/to/amass.pt g1 15
#
# Arguments:
#   proto_python:  Path to Python interpreter with ProtoMotions installed
#   pyroki_python: Path to Python interpreter with PyRoki installed
#   amass_pt_file: Path to packaged AMASS MotionLib .pt file (outputs saved in same directory)
#   robot_type:    Target robot: 'g1' or 'h1_2'
#   skip_freq:     (Optional) Skip every N motions for subset processing (default: 1 = all motions)

set -e  # Exit on error

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <proto_python> <pyroki_python> <amass_pt_file> <robot_type> [skip_freq]"
    echo ""
    echo "Arguments:"
    echo "  proto_python   Path to Python interpreter with ProtoMotions installed"
    echo "  pyroki_python  Path to Python interpreter with PyRoki installed"
    echo "  amass_pt_file  Path to packaged AMASS MotionLib .pt file (outputs saved in same dir)"
    echo "  robot_type     Target robot: 'g1' or 'h1_2'"
    echo "  skip_freq      (Optional) Skip every N motions (default: 1 = all motions)"
    echo ""
    echo "Example:"
    echo "  $0 ~/miniconda3/envs/protomotions/bin/python ~/miniconda3/envs/pyroki/bin/python /data/amass.pt g1 15"
    exit 1
fi

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
AMASS_PT_FILE="$3"
ROBOT_TYPE="$4"
SKIP_FREQ="${5:-1}"

# Validate robot type
if [ "$ROBOT_TYPE" != "g1" ] && [ "$ROBOT_TYPE" != "h1_2" ]; then
    echo "Error: robot_type must be 'g1' or 'h1_2'"
    exit 1
fi

# Validate Python interpreters exist
if [ ! -f "$PROTO_PYTHON" ]; then
    echo "Error: ProtoMotions Python not found: $PROTO_PYTHON"
    exit 1
fi

if [ ! -f "$PYROKI_PYTHON" ]; then
    echo "Error: PyRoki Python not found: $PYROKI_PYTHON"
    exit 1
fi

# Validate input file exists
if [ ! -f "$AMASS_PT_FILE" ]; then
    echo "Error: AMASS .pt file not found: $AMASS_PT_FILE"
    exit 1
fi

# Output directories are in the same location as input (follows rigv1-vaulting convention)
OUTPUT_DIR="$(dirname "$AMASS_PT_FILE")"
KEYPOINTS_DIR="${OUTPUT_DIR}/keypoints-for-retarget"
RETARGETED_DIR="${OUTPUT_DIR}/pyroki-retargeted-${ROBOT_TYPE}"
CONTACTS_DIR="${OUTPUT_DIR}/contacts"
PROTO_DIR="${OUTPUT_DIR}/proto-${ROBOT_TYPE}"
FINAL_PT="${OUTPUT_DIR}/proto-${ROBOT_TYPE}.pt"

echo "=============================================="
echo "Retargeting AMASS to ${ROBOT_TYPE^^}"
echo "=============================================="
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Input:               $AMASS_PT_FILE"
echo "Output dir:          $OUTPUT_DIR"
echo "Skip freq:           $SKIP_FREQ (1 = all motions)"
echo "=============================================="

# Step 1: Extract keypoints from packaged MotionLib (uses ProtoMotions)
echo ""
echo "[Step 1/5] Extracting keypoints from SMPL motions..."
$PROTO_PYTHON data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    "$AMASS_PT_FILE" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format smpl \
    --start-idx 0 \
    --skip-freq "$SKIP_FREQ"

# Step 2: Run PyRoki retargeting (uses PyRoki)
echo ""
echo "[Step 2/5] Running PyRoki retargeting to ${ROBOT_TYPE^^}..."
if [ "$ROBOT_TYPE" == "g1" ]; then
    $PYROKI_PYTHON pyroki/batch_retarget_to_g1_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --output-dir "$RETARGETED_DIR" \
        --no-visualize \
        --skip-existing
else
    $PYROKI_PYTHON pyroki/batch_retarget_to_h1_2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --output-dir "$RETARGETED_DIR" \
        --no-visualize \
        --skip-existing
fi

# Step 3: Extract contact labels from source motions (uses PyRoki)
echo ""
echo "[Step 3/5] Extracting foot contact labels from source SMPL motions..."
if [ "$ROBOT_TYPE" == "g1" ]; then
    $PYROKI_PYTHON pyroki/batch_retarget_to_g1_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR" \
        --skip-existing
else
    $PYROKI_PYTHON pyroki/batch_retarget_to_h1_2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR" \
        --skip-existing
fi

# Step 4: Convert to ProtoMotions format with contact labels (uses ProtoMotions)
echo ""
echo "[Step 4/5] Converting to ProtoMotions format..."
$PROTO_PYTHON data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir "$RETARGETED_DIR" \
    --output-dir "$PROTO_DIR" \
    --robot-type "$ROBOT_TYPE" \
    --contact-labels-dir "$CONTACTS_DIR" \
    --apply-motion-filter \
    --force-remake

# Step 5: Package into MotionLib (uses ProtoMotions)
echo ""
echo "[Step 5/5] Packaging into MotionLib..."
$PROTO_PYTHON protomotions/components/motion_lib.py \
    --motion-path "$PROTO_DIR" \
    --output-file "$FINAL_PT"

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Output MotionLib: $FINAL_PT"
echo ""
echo "To verify the result:"
echo "  python examples/motion_libs_visualizer.py --motion_files $FINAL_PT --robot $ROBOT_TYPE --simulator isaacgym"
echo ""
