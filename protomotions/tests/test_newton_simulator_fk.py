# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test for Newton simulator forward kinematics integration.

This test validates that:
1. Setting simulator state with root and DOF values works correctly
2. Forward kinematics computation is accurate
3. Retrieved body states match the expected states from MotionLib

Usage:
    python protomotions/tests/test_newton_simulator_fk.py --motion-file <path_to_motion.motion> [--robot g1|h1|smpl] [--frame-idx 50]

NOTE: there seems to be some small difference in the results between the Newton simulator and the MotionLib.
This is the main reason why this test is here. But you can also use this to validate your Newton installation along with ProtoMotions.

"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_G1_MOTION_FILE = REPO_ROOT / "examples/data/g1_crouch_to_run.motion"


class NewtonSimulatorFKHarness:
    """Test suite for Newton simulator forward kinematics."""

    def __init__(
        self,
        robot_name: str = "g1",
        motion_file: Optional[str] = None,
        num_envs: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the test suite.

        Args:
            robot_name: Name of the robot (g1, h1, smpl, etc.)
            motion_file: Path to motion file (.motion or .pt)
            num_envs: Number of parallel environments
            device: Device to run on
        """
        self.robot_name = robot_name
        self.motion_file = motion_file
        self.num_envs = num_envs
        self.device = torch.device(device)

        print(f"\n{'='*80}")
        print("Initializing Newton Simulator FK Test")
        print(f"{'='*80}")
        print(f"Robot: {robot_name}")
        print(f"Motion file: {motion_file}")
        print(f"Num envs: {num_envs}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

        # Initialize components
        self._setup_robot_config()
        self._setup_motion_lib()
        self._setup_simulator()

    def _setup_robot_config(self):
        """Load robot configuration."""
        from protomotions.robot_configs.factory import robot_config

        print(f"Loading robot config for {self.robot_name}...")
        self.robot_cfg = robot_config(self.robot_name)
        print(f"  ✓ Loaded robot with {self.robot_cfg.kinematic_info.num_dofs} DOFs")
        print(f"  ✓ Number of bodies: {self.robot_cfg.kinematic_info.num_bodies}")

    def _setup_motion_lib(self):
        """Load motion library."""
        if self.motion_file is None:
            print("Warning: No motion file provided. Using default motion.")
            # Use default motion based on robot type
            if self.robot_name == "g1":
                self.motion_file = "examples/data/g1_crouch_to_run.motion"
            elif self.robot_name == "smpl":
                self.motion_file = "examples/data/smpl_humanoid_sit_armchair.motion"
            else:
                raise ValueError(
                    f"No default motion available for robot {self.robot_name}. Please provide --motion-file"
                )

        print(f"Loading motion library from {self.motion_file}...")
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig

        # Check if file exists
        if not os.path.exists(self.motion_file):
            raise FileNotFoundError(f"Motion file not found: {self.motion_file}")

        motion_lib_config = MotionLibConfig(
            motion_file=self.motion_file,
            get_motion_state_use_blend=False,  # Use exact frames for testing
        )

        self.motion_lib = MotionLib(config=motion_lib_config, device=str(self.device))

        num_motions = self.motion_lib.num_motions()
        total_frames = sum(self.motion_lib.motion_num_frames).item()
        print(f"  ✓ Loaded {num_motions} motion(s)")
        print(f"  ✓ Total frames: {total_frames}")
        print(f"  ✓ Motion length(s): {self.motion_lib.motion_lengths.tolist()}")

    def _setup_simulator(self):
        """Initialize the Newton simulator."""
        from protomotions.components.scene_lib import SceneLib
        from protomotions.components.terrains.config import TerrainConfig
        from protomotions.components.terrains.terrain import Terrain
        from protomotions.simulator.newton.config import (
            NewtonSimParams,
            NewtonSimulatorConfig,
        )
        from protomotions.simulator.newton.simulator import NewtonSimulator

        print("Initializing Newton simulator...")

        # Create a simple flat terrain
        terrain_config = TerrainConfig(
            map_length=20.0,
            map_width=20.0,
            border_size=40.0,
            num_levels=1,
            num_terrains=1,
            terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 100% flat
            horizontal_scale=0.1,
            vertical_scale=0.005,
        )

        self.terrain = Terrain(
            config=terrain_config, num_envs=self.num_envs, device=self.device
        )

        # Create empty scene lib (no objects for this test)
        self.scene_lib = SceneLib.empty(
            num_envs=self.num_envs, device=str(self.device), terrain=self.terrain
        )

        # Create simulator config
        sim_config = NewtonSimulatorConfig(
            num_envs=self.num_envs,
            headless=True,
            sim=NewtonSimParams(
                fps=30,
                decimation=1,
            ),
            experiment_name="newton_simulator_fk_test",
        )

        # Initialize simulator
        self.simulator = NewtonSimulator(
            config=sim_config,
            robot_config=self.robot_cfg,
            terrain=self.terrain,
            device=self.device,
            scene_lib=self.scene_lib,
        )

        # Finalize simulator initialization (no visualization markers for this test)
        self.simulator._initialize_with_markers({})

        print(f"  ✓ Simulator initialized with {self.num_envs} environments")
        print(f"  ✓ Simulation dt: {self.simulator.sim_dt:.4f}s")
        print(f"  ✓ Frame dt: {self.simulator.frame_dt:.4f}s")

    def run_single_frame_fk(self, motion_id: int = 0, frame_idx: int = 0) -> dict:
        """
        Test FK for a single frame from the motion library.

        Args:
            motion_id: ID of the motion to test
            frame_idx: Frame index within the motion

        Returns:
            dict: Test results containing errors and statistics
        """
        print(f"\n{'='*80}")
        print(f"Testing FK for Motion {motion_id}, Frame {frame_idx}")
        print(f"{'='*80}\n")

        # Get motion state from MotionLib (ground truth)
        motion_ids = torch.tensor([motion_id] * self.num_envs, device=self.device)
        frame_indices = torch.tensor([frame_idx] * self.num_envs, device=self.device)

        gt_state = self.motion_lib.get_motion_state_exact_frame(
            motion_ids, frame_indices
        )

        print("Ground truth state from MotionLib:")
        print(f"  Root pos shape: {gt_state.root_pos.shape}")
        print(f"  Root rot shape: {gt_state.root_rot.shape}")
        print(f"  DOF pos shape: {gt_state.dof_pos.shape}")
        print(f"  DOF vel shape: {gt_state.dof_vel.shape}")
        print(f"  Rigid body pos shape: {gt_state.rigid_body_pos.shape}")
        print(f"  Rigid body rot shape: {gt_state.rigid_body_rot.shape}")

        # Set simulator state (reset_envs handles conversion internally)
        print("\nSetting simulator state...")
        self.simulator.reset_envs(gt_state)

        # Get state back from simulator (this includes FK computation)
        # get_robot_state() returns both body and DOF state in common coordinates
        print("Retrieving state from simulator (after FK)...")
        sim_state = self.simulator.get_robot_state()

        # Compute errors
        print("\nComputing errors...")
        results = self._compute_errors(gt_state, sim_state)

        # Print results
        self._print_results(results)

        return results

    @staticmethod
    def _compute_errors(gt_state: RobotState, sim_state: RobotState) -> dict:
        """
        Compute errors between ground truth and simulator states.

        Args:
            gt_state: Ground truth state from MotionLib
            sim_state: State retrieved from simulator

        Returns:
            dict: Error statistics
        """
        results = {}

        # Root position error (in meters)
        root_pos_error = torch.norm(gt_state.root_pos - sim_state.root_pos, dim=-1)
        results["root_pos_error_mean"] = root_pos_error.mean().item()
        results["root_pos_error_max"] = root_pos_error.max().item()
        results["root_pos_error_std"] = root_pos_error.std().item()

        # Root rotation error (quaternion distance)
        # Compute quaternion difference: q_error = q_gt * q_sim^-1
        # Then convert to angle
        gt_quat = gt_state.root_rot
        sim_quat = sim_state.root_rot

        # Quaternion conjugate (inverse for unit quaternions)
        sim_quat_inv = sim_quat.clone()
        sim_quat_inv[..., :3] *= -1  # Negate x, y, z components

        # Quaternion multiplication: q1 * q2
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1[..., 3], q1[..., 0], q1[..., 1], q1[..., 2]
            w2, x2, y2, z2 = q2[..., 3], q2[..., 0], q2[..., 1], q2[..., 2]

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

            return torch.stack([x, y, z, w], dim=-1)

        q_error = quat_mul(gt_quat, sim_quat_inv)

        # Convert to angle: angle = 2 * arccos(|w|)
        root_rot_error = 2 * torch.acos(
            torch.clamp(torch.abs(q_error[..., 3]), -1.0, 1.0)
        )
        root_rot_error = torch.rad2deg(root_rot_error)

        results["root_rot_error_mean"] = root_rot_error.mean().item()
        results["root_rot_error_max"] = root_rot_error.max().item()
        results["root_rot_error_std"] = root_rot_error.std().item()

        # DOF position error (in radians)
        dof_pos_error = torch.abs(gt_state.dof_pos - sim_state.dof_pos)
        results["dof_pos_error_mean"] = dof_pos_error.mean().item()
        results["dof_pos_error_max"] = dof_pos_error.max().item()
        results["dof_pos_error_std"] = dof_pos_error.std().item()

        # Rigid body position error (in meters)
        body_pos_error = torch.norm(
            gt_state.rigid_body_pos - sim_state.rigid_body_pos, dim=-1
        )
        results["body_pos_error_mean"] = body_pos_error.mean().item()
        results["body_pos_error_max"] = body_pos_error.max().item()
        results["body_pos_error_std"] = body_pos_error.std().item()
        results["body_pos_error_per_body_mean"] = body_pos_error.mean(dim=0)

        # Rigid body rotation error (quaternion distance in degrees)
        body_gt_quat = gt_state.rigid_body_rot
        body_sim_quat = sim_state.rigid_body_rot

        body_sim_quat_inv = body_sim_quat.clone()
        body_sim_quat_inv[..., :3] *= -1

        body_q_error = quat_mul(body_gt_quat.view(-1, 4), body_sim_quat_inv.view(-1, 4))
        body_q_error = body_q_error.view(body_gt_quat.shape)

        body_rot_error = 2 * torch.acos(
            torch.clamp(torch.abs(body_q_error[..., 3]), -1.0, 1.0)
        )
        body_rot_error = torch.rad2deg(body_rot_error)

        results["body_rot_error_mean"] = body_rot_error.mean().item()
        results["body_rot_error_max"] = body_rot_error.max().item()
        results["body_rot_error_std"] = body_rot_error.std().item()
        results["body_rot_error_per_body_mean"] = body_rot_error.mean(dim=0)

        return results

    def _print_results(self, results: dict):
        """Print test results in a formatted manner."""
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)

        print("\n📍 Root Position Error (meters):")
        print(f"  Mean:   {results['root_pos_error_mean']:.6f} m")
        print(f"  Max:    {results['root_pos_error_max']:.6f} m")
        print(f"  Std:    {results['root_pos_error_std']:.6f} m")

        print("\n🔄 Root Rotation Error (degrees):")
        print(f"  Mean:   {results['root_rot_error_mean']:.4f}°")
        print(f"  Max:    {results['root_rot_error_max']:.4f}°")
        print(f"  Std:    {results['root_rot_error_std']:.4f}°")

        print("\n🦾 DOF Position Error (radians):")
        print(f"  Mean:   {results['dof_pos_error_mean']:.6f} rad")
        print(f"  Max:    {results['dof_pos_error_max']:.6f} rad")
        print(f"  Std:    {results['dof_pos_error_std']:.6f} rad")

        print("\n🤖 Rigid Body Position Error (meters):")
        print(f"  Mean:   {results['body_pos_error_mean']:.6f} m")
        print(f"  Max:    {results['body_pos_error_max']:.6f} m")
        print(f"  Std:    {results['body_pos_error_std']:.6f} m")

        print("\n🤖 Rigid Body Rotation Error (degrees):")
        print(f"  Mean:   {results['body_rot_error_mean']:.4f}°")
        print(f"  Max:    {results['body_rot_error_max']:.4f}°")
        print(f"  Std:    {results['body_rot_error_std']:.4f}°")

        # Determine pass/fail
        print("\n" + "=" * 80)
        print("PASS/FAIL CRITERIA")
        print("=" * 80)

        # Define tolerances
        tolerances = {
            "root_pos_error_mean": 1e-4,  # 0.1mm
            "root_rot_error_mean": 1e-2,  # 0.01 degrees
            "dof_pos_error_mean": 1e-4,  # 0.0001 radians
            "body_pos_error_mean": 1e-3,  # 1mm
            "body_rot_error_mean": 0.1,  # 0.1 degrees
        }

        all_passed = True
        for metric, tolerance in tolerances.items():
            passed = results[metric] < tolerance
            all_passed = all_passed and passed
            status = "✓ PASS" if passed else "✗ FAIL"
            print(
                f"  {status}: {metric} = {results[metric]:.6f} (tolerance: {tolerance})"
            )

        print("\n" + "=" * 80)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 80 + "\n")

        return all_passed

    def run_multiple_frames(self, num_frames: int = 10) -> list:
        """
        Test FK for multiple frames from the motion library.

        Args:
            num_frames: Number of frames to test

        Returns:
            list: List of test results for each frame
        """
        print(f"\n{'='*80}")
        print(f"Testing FK for Multiple Frames ({num_frames} frames)")
        print(f"{'='*80}\n")

        # Get total number of frames in the first motion
        num_available_frames = self.motion_lib.motion_num_frames[0].item()

        # Sample frames uniformly
        frame_indices = np.linspace(0, num_available_frames - 1, num_frames, dtype=int)

        all_results = []
        for i, frame_idx in enumerate(frame_indices):
            print(f"\n--- Testing Frame {i+1}/{num_frames} (frame_idx={frame_idx}) ---")
            results = self.run_single_frame_fk(motion_id=0, frame_idx=frame_idx)
            all_results.append(results)

        # Aggregate results
        print(f"\n{'='*80}")
        print(f"AGGREGATE RESULTS ACROSS {num_frames} FRAMES")
        print(f"{'='*80}\n")

        metrics = [
            "root_pos_error_mean",
            "root_rot_error_mean",
            "dof_pos_error_mean",
            "body_pos_error_mean",
            "body_rot_error_mean",
        ]

        for metric in metrics:
            values = [r[metric] for r in all_results]
            print(f"{metric}:")
            print(f"  Mean across frames: {np.mean(values):.6f}")
            print(f"  Max across frames:  {np.max(values):.6f}")
            print(f"  Min across frames:  {np.min(values):.6f}")

        return all_results


def _make_robot_state() -> RobotState:
    rigid_body_pos = torch.tensor(
        [
            [[0.0, 0.1, 0.2], [1.0, 2.0, 3.0]],
            [[0.3, 0.4, 0.5], [4.0, 5.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    rigid_body_rot = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    dof_pos = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=torch.float32
    )

    return RobotState(
        state_conversion=StateConversion.COMMON,
        dof_pos=dof_pos,
        dof_vel=torch.zeros_like(dof_pos),
        rigid_body_pos=rigid_body_pos,
        rigid_body_rot=rigid_body_rot,
    )


def test_compute_errors_reports_zero_for_identical_states():
    state = _make_robot_state()

    results = NewtonSimulatorFKHarness._compute_errors(state, state)

    scalar_metrics = [
        "root_pos_error_mean",
        "root_pos_error_max",
        "root_rot_error_mean",
        "root_rot_error_max",
        "dof_pos_error_mean",
        "dof_pos_error_max",
        "body_pos_error_mean",
        "body_pos_error_max",
        "body_rot_error_mean",
        "body_rot_error_max",
    ]
    for metric in scalar_metrics:
        assert results[metric] == pytest.approx(0.0, abs=1e-6)

    assert torch.allclose(
        results["body_pos_error_per_body_mean"], torch.zeros(2), atol=1e-6
    )
    assert torch.allclose(
        results["body_rot_error_per_body_mean"], torch.zeros(2), atol=1e-6
    )


def test_compute_errors_treats_negated_quaternions_as_same_rotation():
    gt_state = _make_robot_state()
    sim_state = gt_state.clone()
    sim_state.rigid_body_rot = -gt_state.rigid_body_rot

    results = NewtonSimulatorFKHarness._compute_errors(gt_state, sim_state)

    assert results["root_rot_error_max"] == pytest.approx(0.0, abs=1e-6)
    assert results["body_rot_error_max"] == pytest.approx(0.0, abs=1e-6)


def test_newton_simulator_fk_matches_motionlib_default_frame():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    if not torch.cuda.is_available():
        pytest.skip("Newton FK integration requires CUDA")
    if not DEFAULT_G1_MOTION_FILE.exists():
        pytest.skip(f"Default G1 motion file is missing: {DEFAULT_G1_MOTION_FILE}")

    harness = NewtonSimulatorFKHarness(
        robot_name="g1",
        motion_file=str(DEFAULT_G1_MOTION_FILE),
        num_envs=1,
        device="cuda",
    )

    results = harness.run_single_frame_fk(motion_id=0, frame_idx=0)

    assert results["root_pos_error_mean"] < 1e-4
    assert results["root_rot_error_mean"] < 1e-2
    assert results["dof_pos_error_mean"] < 1e-4
    assert results["body_pos_error_mean"] < 1e-3
    assert results["body_rot_error_mean"] < 0.1


def main():
    parser = argparse.ArgumentParser(
        description="Test Newton simulator forward kinematics"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Path to motion file (.motion or .pt)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        choices=["g1", "h1", "h1_2", "smpl", "smplx", "amp", "rigv1"],
        help="Robot type",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument("--frame-idx", type=int, default=50, help="Frame index to test")
    parser.add_argument(
        "--test-multiple", action="store_true", help="Test multiple frames"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of frames to test (if --test-multiple)",
    )

    args = parser.parse_args()

    # Initialize test
    test = NewtonSimulatorFKHarness(
        robot_name=args.robot,
        motion_file=args.motion_file,
        num_envs=args.num_envs,
        device=args.device,
    )

    # Run tests
    if args.test_multiple:
        test.run_multiple_frames(num_frames=args.num_frames)
    else:
        test.run_single_frame_fk(motion_id=0, frame_idx=args.frame_idx)


if __name__ == "__main__":
    main()
