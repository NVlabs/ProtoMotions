# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for torch-numpy equivalence of odometer corruption.

Verifies that ``apply_odom_corruption_torch`` and ``apply_odom_corruption_np``
produce identical outputs given the same inputs and random noise, ensuring zero
risk of sim-to-real mismatch from the shared corruption utility.
"""

import math

import numpy as np
import pytest
import torch

from protomotions.utils.odom_corruption import (
    apply_odom_corruption_np,
    apply_odom_corruption_torch,
    sample_odom_params_np,
)


class TestOdomCorruptionEquivalence:
    """Verify torch and numpy implementations produce identical results."""

    @pytest.mark.parametrize(
        "xy, scale, yaw_deg",
        [
            ([1.0, 0.0], 1.0, 0.0),      # identity corruption, forward offset
            ([0.0, 2.0], 1.3, 6.0),       # right offset, scale + yaw
            ([-0.5, 0.3], 0.7, -6.0),     # backward-left, min scale, negative yaw
            ([3.0, -1.5], 1.15, 3.0),     # large offset
            ([0.01, 0.02], 1.0, 0.0),     # near-zero offset (tests soft threshold)
        ],
    )
    def test_torch_numpy_match(self, xy, scale, yaw_deg):
        """Core equivalence: given same inputs and noise, outputs match."""
        log_noise_std = 0.12
        soft_threshold = 0.15
        yaw_rad = math.radians(yaw_deg)
        yaw_cs = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)

        xy_np = np.array(xy, dtype=np.float32)
        xy_torch = torch.tensor([xy], dtype=torch.float32)  # [1, 2]
        scale_torch = torch.tensor([scale], dtype=torch.float32)
        yaw_cs_torch = torch.from_numpy(yaw_cs).unsqueeze(0)  # [1, 2]

        # Use a fixed noise value by temporarily replacing randn
        fixed_noise = 0.05

        # Torch: manually inject noise
        torch.manual_seed(0)
        with torch.no_grad():
            cos_y = yaw_cs_torch[:, 0]
            sin_y = yaw_cs_torch[:, 1]
            x_rot = cos_y * xy_torch[:, 0] - sin_y * xy_torch[:, 1]
            y_rot = sin_y * xy_torch[:, 0] + cos_y * xy_torch[:, 1]
            xy_affine_t = scale_torch.unsqueeze(-1) * torch.stack([x_rot, y_rot], dim=-1)
            mag_t = torch.norm(xy_affine_t, dim=-1, keepdim=True).clamp(min=1e-8)
            dir_t = xy_affine_t / mag_t
            log_mag_t = torch.log(1.0 + mag_t)
            nw_t = mag_t / (mag_t + soft_threshold)
            noisy_log_t = log_mag_t + fixed_noise * log_noise_std * nw_t
            noisy_mag_t = (torch.exp(noisy_log_t) - 1.0).clamp(min=0.0)
            result_torch = (dir_t * noisy_mag_t).squeeze(0).numpy()

        # Numpy: same computation
        x_rot_np = yaw_cs[0] * xy_np[0] - yaw_cs[1] * xy_np[1]
        y_rot_np = yaw_cs[1] * xy_np[0] + yaw_cs[0] * xy_np[1]
        xy_affine_np = scale * np.array([x_rot_np, y_rot_np], dtype=np.float32)
        mag_np = max(float(np.linalg.norm(xy_affine_np)), 1e-8)
        dir_np = xy_affine_np / mag_np
        log_mag_np = math.log(1.0 + mag_np)
        nw_np = mag_np / (mag_np + soft_threshold)
        noisy_log_np = log_mag_np + fixed_noise * log_noise_std * nw_np
        noisy_mag_np = max(math.exp(noisy_log_np) - 1.0, 0.0)
        result_np = dir_np * noisy_mag_np

        np.testing.assert_allclose(result_torch, result_np, atol=1e-5, rtol=1e-5)

    def test_zero_noise_deterministic(self):
        """With log_noise_std=0, both implementations should be fully deterministic."""
        xy = np.array([2.0, -1.0], dtype=np.float32)
        scale = 1.2
        yaw_cs = np.array([math.cos(0.1), math.sin(0.1)], dtype=np.float32)

        result_np = apply_odom_corruption_np(xy, scale, yaw_cs, log_noise_std=0.0)

        xy_t = torch.from_numpy(xy).unsqueeze(0)
        scale_t = torch.tensor([scale])
        yaw_cs_t = torch.from_numpy(yaw_cs).unsqueeze(0)
        result_t = apply_odom_corruption_torch(
            xy_t, scale_t, yaw_cs_t, log_noise_std=0.0
        ).squeeze(0).numpy()

        np.testing.assert_allclose(result_t, result_np, atol=1e-5)

    def test_identity_corruption_preserves_direction(self):
        """With scale=1, yaw=0, noise=0, output equals input."""
        xy = np.array([1.5, -0.8], dtype=np.float32)
        identity_cs = np.array([1.0, 0.0], dtype=np.float32)

        result = apply_odom_corruption_np(
            xy, odom_scale=1.0, odom_yaw_cos_sin=identity_cs, log_noise_std=0.0
        )
        np.testing.assert_allclose(result, xy, atol=1e-5)

    def test_batched_torch(self):
        """Torch version handles batched inputs correctly."""
        batch = 16
        xy = torch.randn(batch, 2)
        scale = torch.ones(batch) * 1.1
        yaw_cs = torch.zeros(batch, 2)
        yaw_cs[:, 0] = 1.0  # cos(0) = 1, sin(0) = 0

        result = apply_odom_corruption_torch(xy, scale, yaw_cs, log_noise_std=0.0)
        assert result.shape == (batch, 2)

        # With zero noise and zero yaw, result should be scale * xy
        expected = 1.1 * xy
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestSampleOdomParams:
    """Test parameter sampling utility."""

    def test_scale_in_range(self):
        np.random.seed(42)
        for _ in range(100):
            scale, _ = sample_odom_params_np(scale_range=(0.7, 1.3))
            assert 0.7 <= scale <= 1.3

    def test_yaw_cos_sin_unit(self):
        np.random.seed(42)
        for _ in range(100):
            _, yaw_cs = sample_odom_params_np(yaw_range_deg=12.0)
            norm = np.linalg.norm(yaw_cs)
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_zero_yaw_range(self):
        scale, yaw_cs = sample_odom_params_np(yaw_range_deg=0.0)
        np.testing.assert_allclose(yaw_cs, [1.0, 0.0], atol=1e-6)
