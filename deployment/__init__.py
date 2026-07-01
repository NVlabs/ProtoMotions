# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment utilities for ProtoMotions policies.

This package provides a minimal deployment stack for running trained policies
outside of the full ProtoMotions training framework:

    deployment/
        state_utils.py   -- derive anchor_rot / root_local_ang_vel from raw sim state
        motion_utils.py  -- MotionPlayer: interpolation + 50fps caching
        export_bm_tracker_onnx.py -- ONNX export for BeyondMimic tracker policies
        test_tracker_mujoco.py -- standalone MuJoCo inference loop

Typical workflow
----------------
1. Export the ONNX model once::

    python deployment/export_bm_tracker_onnx.py \\
        --checkpoint results/my_exp/last.ckpt \\
        --output deployment/models/

2. Test with MuJoCo (first run also caches the motion at 50fps)::

    python deployment/test_tracker_mujoco.py \\
        --onnx  deployment/models/unified_pipeline.onnx \\
        --motion data/motions/walk.pt \\
        --cache-motion  # write walk.50fps.pt next to walk.pt

3. Subsequent test runs are PyTorch-computation-free::

    python deployment/test_tracker_mujoco.py \\
        --onnx  deployment/models/unified_pipeline.onnx \\
        --motion data/motions/walk.50fps.pt
"""
