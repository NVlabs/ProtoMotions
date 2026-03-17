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
"""Deployment utilities for ProtoMotions policies.

This package provides a minimal deployment stack for running trained policies
outside of the full ProtoMotions training framework:

    deployment/
        state_utils.py   -- derive anchor_rot / root_local_ang_vel from raw sim state
        motion_utils.py  -- MotionPlayer: interpolation + 50fps caching
        export_bm_tracker_onnx.py -- ONNX export for BeyondMimic trackers (no simulator required)
        test_tracker_mujoco.py -- standalone MuJoCo inference loop

Typical workflow
----------------
1. Export the ONNX model once::

    python deployment/export_bm_tracker_onnx.py \\
        --checkpoint results/my_exp/last.ckpt

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
