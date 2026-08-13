# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone deployment input protocol (``deployment/tracker_inputs.py``).

These cover the assembly that runs on hardware, where there is no simulator to
supply a consistent world frame.  The load-bearing property is that every
reference quantity handed to the policy ends up in the *robot's* frame, because
the policy differences reference positions against each other and against the
robot's own odometer.

The critical fixture detail: the robot starts somewhere other than the motion's
recorded origin, with a heading offset.  ``deployment/test_tracker_mujoco.py``
starts the robot exactly on the motion's first frame, which collapses the
alignment transform to the identity and hides any frame mismatch.  Real hardware
never does that -- the robot stands where it stands and the clip was captured
somewhere else -- so these tests deliberately keep the two apart.

Run with:  pytest protomotions/tests/test_deploy_tracker_inputs.py -v

No torch, mujoco or onnxruntime needed; ``tracker_inputs`` is pure NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from deployment.state_utils import quat_rotate_np
from deployment.tracker_inputs import (
    align_future_motion_refs,
    build_tracker_onnx_inputs,
    compute_odom_disp_start,
    get_tracker_input_requirements,
    init_tracker_alignment,
)

NUM_BODIES = 6
NUM_DOFS = 4
ANCHOR_IDX = 3
N_FUTURE = 4

# The clip was captured a long way from the origin, as mocap clips are.
MOTION_START_XY = np.array([-9.0, -18.0], dtype=np.float32)
MOTION_HEIGHT = 0.75
MOTION_YAW_DEG = -25.0

# The robot is powered on near the origin, facing somewhere else entirely.
ROBOT_START = np.array([0.0, 0.0, 0.75], dtype=np.float32)
ROBOT_YAW_DEG = 40.0


def _yaw_quat(deg: float) -> np.ndarray:
    """Yaw-only quaternion in xyzw."""
    half = np.deg2rad(deg) / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _body_array(anchor_pos: np.ndarray) -> np.ndarray:
    """A full body-position array whose anchor slice is exactly ``anchor_pos``."""
    offsets = np.zeros((NUM_BODIES, 3), dtype=np.float32)
    offsets[:, 0] = np.linspace(-0.2, 0.2, NUM_BODIES)
    offsets[:, 2] = np.linspace(-0.4, 0.4, NUM_BODIES)
    offsets[ANCHOR_IDX] = 0.0
    return (anchor_pos[None, :] + offsets).astype(np.float32)


@pytest.fixture
def motion() -> dict:
    """A reference clip that walks +1 m along its own +x between now and the horizon."""
    motion_rot = np.tile(_yaw_quat(MOTION_YAW_DEG), (NUM_BODIES, 1)).astype(np.float32)

    start_anchor = np.array(
        [MOTION_START_XY[0], MOTION_START_XY[1], MOTION_HEIGHT], dtype=np.float32
    )
    # The reference has already walked 1 m before the step under test.
    current_anchor = start_anchor + np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Future steps advance 0.25 m each, so the last one is exactly +1 m from now.
    future_pos = np.stack(
        [
            _body_array(
                current_anchor + np.array([0.25 * (k + 1), 0.0, 0.0], np.float32)
            )
            for k in range(N_FUTURE)
        ]
    ).astype(np.float32)

    return {
        "start_anchor_pos": start_anchor,
        "start_anchor_rot": _yaw_quat(MOTION_YAW_DEG),
        "current_body_pos": _body_array(current_anchor),
        "future_refs": {
            "body_pos": future_pos,
            "body_rot": np.tile(motion_rot, (N_FUTURE, 1, 1)).astype(np.float32),
            "dof_pos": np.zeros((N_FUTURE, NUM_DOFS), dtype=np.float32),
            "dof_vel": np.zeros((N_FUTURE, NUM_DOFS), dtype=np.float32),
        },
    }


@pytest.fixture
def alignment(motion):
    """Alignment for a robot that starts away from the clip, with a heading offset."""
    return init_tracker_alignment(
        robot_anchor_pos=ROBOT_START,
        robot_anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        motion_anchor_pos=motion["start_anchor_pos"],
        motion_anchor_rot=motion["start_anchor_rot"],
    )


def _aligned_current_ref_anchor(motion, alignment) -> np.ndarray:
    """The current reference anchor, aligned exactly as the MuJoCo runner does."""
    motion_disp = motion["current_body_pos"] - alignment.motion_anchor_start_pos
    aligned = alignment.robot_anchor_start_pos + quat_rotate_np(
        alignment.heading_offset, motion_disp
    )
    return aligned[ANCHOR_IDX].astype(np.float32)


# ---------------------------------------------------------------------------
# Frame consistency -- the regression this file exists for
# ---------------------------------------------------------------------------


def _travel_from_onnx_inputs(
    onnx_inputs: dict, ref_anchor_pos: np.ndarray
) -> np.ndarray:
    """Reproduce the subtraction ``BakedTargetFK`` performs on these two inputs."""
    future_anchor_pos = onnx_inputs["mimic_future_anchor_pos"][0]  # [n_future, 3]
    return future_anchor_pos[:, :2] - ref_anchor_pos[None, :2]


@pytest.mark.parametrize(
    "onnx_keys",
    [
        pytest.param(
            {
                "mimic_future_anchor_pos": "mimic.future_anchor_pos",
                "mimic_ref_anchor_pos": "mimic.ref_anchor_pos",
            },
            id="anchor_only",  # what the baked-FK export actually requests
        ),
        pytest.param(
            {
                "mimic_future_anchor_pos": "mimic.future_anchor_pos",
                "mimic_future_pos": "mimic.future_pos",
                "mimic_ref_anchor_pos": "mimic.ref_anchor_pos",
            },
            id="anchor_plus_full_body",
        ),
    ],
)
def test_future_and_current_reference_anchors_share_a_frame(
    motion, alignment, onnx_keys
):
    """The travel the policy reads must equal the reference's real travel.

    The anchor-only case is the one that regressed: the assembler used to align
    positions only when the *full* body array was an ONNX input, so an export that
    asked for the anchor alone got its future anchor in the motion's recorded frame
    while the current reference anchor was in the robot's frame.  The difference of
    the two was then the offset between the frames -- roughly 21 m here -- instead
    of the 1 m the reference actually travels.
    """
    ref_anchor_pos = _aligned_current_ref_anchor(motion, alignment)
    aligned_refs = align_future_motion_refs(motion["future_refs"], alignment)

    onnx_inputs = build_tracker_onnx_inputs(
        onnx_input_names=list(onnx_keys),
        onnx_name_to_key=onnx_keys,
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=aligned_refs,
        anchor_body_index=ANCHOR_IDX,
        num_dofs=NUM_DOFS,
        ref_anchor_pos=ref_anchor_pos,
    )

    travel = _travel_from_onnx_inputs(onnx_inputs, ref_anchor_pos)

    # Truth: the clip advances 0.25*(k+1) m along its own +x, seen through the
    # start-heading offset that maps the clip's frame onto the robot's.
    for k in range(N_FUTURE):
        motion_frame_step = np.array([0.25 * (k + 1), 0.0, 0.0], dtype=np.float32)
        expected = quat_rotate_np(alignment.heading_offset, motion_frame_step)[:2]
        np.testing.assert_allclose(travel[k], expected, atol=1e-4)

    # And the last step really is the 1 m we set up, not a coincidence of scale.
    assert np.linalg.norm(travel[-1]) == pytest.approx(1.0, abs=1e-4)


def test_alignment_is_the_identity_when_the_robot_starts_on_the_motion(motion):
    """Documents why the MuJoCo runner cannot catch a frame mismatch.

    ``set_initial_pose`` places the robot on the motion's frame 0, which makes the
    alignment transform the identity.  Aligned and un-aligned data are then equal,
    so the runner reports zero error either way.  This is the blind spot the tests
    above exist to cover.
    """
    on_motion = init_tracker_alignment(
        robot_anchor_pos=motion["start_anchor_pos"],
        robot_anchor_rot=motion["start_anchor_rot"],
        motion_anchor_pos=motion["start_anchor_pos"],
        motion_anchor_rot=motion["start_anchor_rot"],
    )

    np.testing.assert_allclose(
        on_motion.heading_offset, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6
    )
    aligned = align_future_motion_refs(motion["future_refs"], on_motion)
    np.testing.assert_allclose(
        aligned["body_pos"], motion["future_refs"]["body_pos"], atol=1e-5
    )


def test_alignment_actually_moves_data_when_the_robot_starts_elsewhere(
    motion, alignment
):
    """Guards the fixture itself: a degenerate setup would make the tests vacuous."""
    aligned = align_future_motion_refs(motion["future_refs"], alignment)
    shift = np.abs(aligned["body_pos"] - motion["future_refs"]["body_pos"]).max()
    assert shift > 1.0, "fixture no longer separates the robot from the motion"


def test_odometer_offset_and_reference_anchor_share_a_frame(motion, alignment):
    """A stationary robot at its start pose is offset from the reference by the gap.

    ``compute_odom_offset_local`` differences the reference anchor against the
    robot's believed position, so the reference anchor has to be in the robot's
    frame for that to mean anything.
    """
    odom_disp_start = compute_odom_disp_start(ROBOT_START[:2], alignment)
    np.testing.assert_allclose(odom_disp_start, np.zeros(2), atol=1e-6)

    ref_anchor_pos = _aligned_current_ref_anchor(motion, alignment)
    # The reference has walked 1 m along the clip's +x since the shared start,
    # so in the robot's frame it sits that far away, rotated by the heading offset.
    expected = (
        ROBOT_START[:2]
        + quat_rotate_np(
            alignment.heading_offset, np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )[:2]
    )
    np.testing.assert_allclose(ref_anchor_pos[:2], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Requirement detection and error reporting
# ---------------------------------------------------------------------------


def test_anchor_only_export_is_recognised_as_needing_reference_positions():
    """The regression's root cause: anchor-only exports were not flagged."""
    requirements = get_tracker_input_requirements(
        {
            "mimic_future_anchor_pos": "mimic.future_anchor_pos",
            "mimic_ref_anchor_pos": "mimic.ref_anchor_pos",
        }
    )
    assert requirements.future_anchor_pos
    assert not requirements.future_pos  # the full body array is genuinely absent
    assert requirements.needs_reference_positions
    assert requirements.needs_spatial_start


def test_export_reading_no_reference_positions_needs_none():
    requirements = get_tracker_input_requirements(
        {
            "current_dof_pos": "current.dof_pos",
            "mimic_future_anchor_rot": "mimic.future_anchor_rot",
        }
    )
    assert not requirements.needs_reference_positions
    assert not requirements.needs_spatial_start


def test_reference_data_the_policy_never_reads_may_be_omitted(motion, alignment):
    """A deployment only has to carry what its policy actually consumes."""
    rotations_only = {
        "body_rot": motion["future_refs"]["body_rot"],
        "dof_pos": motion["future_refs"]["dof_pos"],
    }
    aligned = align_future_motion_refs(rotations_only, alignment)
    assert "body_pos" not in aligned

    onnx_inputs = build_tracker_onnx_inputs(
        onnx_input_names=["mimic_future_anchor_rot", "mimic_future_dof_pos"],
        onnx_name_to_key={
            "mimic_future_anchor_rot": "mimic.future_anchor_rot",
            "mimic_future_dof_pos": "mimic.future_dof_pos",
        },
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=aligned,
        anchor_body_index=ANCHOR_IDX,
        num_dofs=NUM_DOFS,
    )
    assert set(onnx_inputs) == {"mimic_future_anchor_rot", "mimic_future_dof_pos"}


def test_an_input_with_no_source_is_reported_by_name(motion, alignment):
    """Silently dropping it just defers an opaque failure to onnxruntime."""
    aligned = align_future_motion_refs(motion["future_refs"], alignment)

    with pytest.raises(KeyError) as excinfo:
        build_tracker_onnx_inputs(
            onnx_input_names=["mimic_future_anchor_pos", "current_anchor_pos"],
            onnx_name_to_key={
                "mimic_future_anchor_pos": "mimic.future_anchor_pos",
                "current_anchor_pos": "current.anchor_pos",
            },
            dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
            dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
            anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
            root_local_ang_vel=np.zeros(3, dtype=np.float32),
            future_refs=aligned,
            anchor_body_index=ANCHOR_IDX,
            num_dofs=NUM_DOFS,
        )

    message = str(excinfo.value)
    assert "current_anchor_pos" in message
    assert "current.anchor_pos" in message
