#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to retarget a single SMPL .motion file to a robot (G1 or H1_2)
#
# IMPORTANT: ProtoMotions and PyRoki require separate Python environments.
# You must provide paths to both Python interpreters.
#
# Usage: ./scripts/retarget_single_motion_to_robot.sh <proto_python> <pyroki_python> <motion_file> <output_dir> <robot_type>
#
# Example:
#   ./scripts/retarget_single_motion_to_robot.sh \
#       ~/miniconda3/envs/protomotions/bin/python \
#       ~/miniconda3/envs/pyroki/bin/python \
#       /path/to/motion.motion /path/to/output g1
#
# Arguments:
#   proto_python:  Path to Python interpreter with ProtoMotions installed
#   pyroki_python: Path to Python interpreter with PyRoki installed
#   motion_file:   Path to input .motion file (SMPL format)
#   output_dir:    Directory where all intermediate and final outputs will be saved
#   robot_type:    Target robot: 'g1' or 'h1_2'

set -e  # Exit on error

# Parse arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <proto_python> <pyroki_python> <motion_file> <output_dir> <robot_type>"
    echo ""
    echo "Arguments:"
    echo "  proto_python   Path to Python interpreter with ProtoMotions installed"
    echo "  pyroki_python  Path to Python interpreter with PyRoki installed"
    echo "  motion_file    Path to input .motion file (SMPL format)"
    echo "  output_dir     Directory where all outputs will be saved"
    echo "  robot_type     Target robot: 'g1' or 'h1_2'"
    echo ""
    echo "Example:"
    echo "  $0 ~/miniconda3/envs/protomotions/bin/python ~/miniconda3/envs/pyroki/bin/python /data/walk.motion /data/retargeted g1"
    exit 1
fi

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
MOTION_FILE="$3"
OUTPUT_DIR="$4"
ROBOT_TYPE="$5"

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

# Validate input file exists and is a .motion file
if [ ! -f "$MOTION_FILE" ]; then
    echo "Error: Motion file not found: $MOTION_FILE"
    exit 1
fi

if [[ "$MOTION_FILE" != *.motion ]]; then
    echo "Error: Input file must be a .motion file: $MOTION_FILE"
    exit 1
fi

# Get the motion filename without extension for naming outputs
MOTION_BASENAME=$(basename "$MOTION_FILE" .motion)

# Create output directories
KEYPOINTS_DIR="${OUTPUT_DIR}/keypoints"
RETARGETED_DIR="${OUTPUT_DIR}/retargeted_${ROBOT_TYPE}"
CONTACTS_DIR="${OUTPUT_DIR}/contacts"
PROTO_DIR="${OUTPUT_DIR}/retargeted_${ROBOT_TYPE}_proto"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Retargeting Single Motion to ${ROBOT_TYPE^^}"
echo "=============================================="
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Input:               $MOTION_FILE"
echo "Output dir:          $OUTPUT_DIR"
echo "=============================================="

# Step 1: Extract keypoints from single motion (uses ProtoMotions)
echo ""
echo "[Step 1/5] Extracting keypoints from SMPL motion..."
$PROTO_PYTHON data/scripts/extract_keypoints_from_single_motion.py \
    "$MOTION_FILE" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format smpl \
    --force-remake

# Step 2: Run PyRoki retargeting (uses PyRoki)
echo ""
echo "[Step 2/5] Running PyRoki retargeting to ${ROBOT_TYPE^^}..."
if [ "$ROBOT_TYPE" == "g1" ]; then
    $PYROKI_PYTHON pyroki/batch_retarget_to_g1_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --output-dir "$RETARGETED_DIR" \
        --no-visualize
else
    $PYROKI_PYTHON pyroki/batch_retarget_to_h1_2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --output-dir "$RETARGETED_DIR" \
        --no-visualize
fi

# Step 3: Extract contact labels from source motion (uses PyRoki)
echo ""
echo "[Step 3/5] Extracting foot contact labels from source SMPL motion..."
if [ "$ROBOT_TYPE" == "g1" ]; then
    $PYROKI_PYTHON pyroki/batch_retarget_to_g1_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR"
else
    $PYROKI_PYTHON pyroki/batch_retarget_to_h1_2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR"
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

# Step 5: Find and report the output file
echo ""
echo "[Step 5/5] Locating output file..."
OUTPUT_MOTION=$(find "$PROTO_DIR" -name "*.motion" -type f | head -1)

if [ -z "$OUTPUT_MOTION" ]; then
    echo "Error: No output .motion file found in $PROTO_DIR"
    echo "The motion may have been filtered out. Check the logs above."
    exit 1
fi

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Output motion: $OUTPUT_MOTION"
echo ""
echo "To visualize the result:"
echo "  python examples/motion_libs_visualizer.py --motion_files $OUTPUT_MOTION --robot $ROBOT_TYPE --simulator isaacgym"
echo ""

