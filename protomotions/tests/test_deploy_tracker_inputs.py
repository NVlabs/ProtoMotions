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

from deployment import tracker_inputs
from deployment.state_utils import quat_rotate_np
from deployment.tracker_inputs import (
    LEGACY_ACTION_HISTORY_KEY,
    ActionFeedback,
    action_history_entries,
    action_history_fill,
    action_history_output,
    action_history_steps_back,
    align_future_motion_refs,
    build_tracker_onnx_inputs,
    compute_odom_disp_start,
    get_tracker_input_requirements,
    init_tracker_alignment,
    resolve_fill,
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


def _contract(
    onnx_keys, num_dofs=NUM_DOFS, anchor_idx=ANCHOR_IDX, num_bodies=NUM_BODIES
):
    """A minimal valid DeploymentContract whose inputs are ``onnx_keys``.

    The reference-host consumers read only the inputs and the robot block, so the
    other blocks carry the smallest values that pass the contract's validation.
    ``onnx_keys`` maps each ONNX input name to its host key, as the exporter's
    _runtime.onnx_name_to_in_key does.
    """
    from deployment.contract import (
        SCHEMA,
        ControlBlock,
        DeploymentContract,
        InputSpec,
        MotionBlock,
        OutputSpec,
        RobotBlock,
        RuntimeBlock,
        TimingBlock,
    )

    joint_names = tuple(f"j{i}" for i in range(num_dofs))
    body_names = tuple(f"b{i}" for i in range(num_bodies))
    inputs = tuple(
        InputSpec(
            onnx_name=name,
            key=key,
            source_namespace=key.split(".", 1)[0],
            kind="x",
            provider="simulator_truth",
            shape=(1, num_dofs),
            dtype="float32",
            element_names=(),
            future_steps=None,
            history=None,
        )
        for name, key in onnx_keys.items()
    )
    return DeploymentContract(
        schema=SCHEMA,
        inputs=inputs,
        outputs=(OutputSpec("actions", "actions", (1, num_dofs), None),),
        robot=RobotBlock(
            "robot.xml",
            body_names,
            joint_names,
            body_names[anchor_idx],
            body_names[0],
        ),
        control=ControlBlock(
            "fn",
            (1.0,) * num_dofs,
            (0.1,) * num_dofs,
            (10.0,) * num_dofs,
            None,
            None,
            None,
        ),
        timing=TimingBlock(0.02, 0.005),
        motion=MotionBlock((1, 2)),
        runtime=RuntimeBlock(
            tuple(onnx_keys),
            ("actions",),
            dict(onnx_keys),
            (),
            tuple(onnx_keys.values()),
        ),
        sensors=None,
        checkpoint="c.ckpt",
        control_type="BUILT_IN_PD",
        onnx_sha256=None,
    )


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
        _contract(onnx_keys),
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=aligned_refs,
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
        _contract(
            {
                "mimic_future_anchor_pos": "mimic.future_anchor_pos",
                "mimic_ref_anchor_pos": "mimic.ref_anchor_pos",
            }
        )
    )
    assert requirements.future_anchor_pos
    assert not requirements.future_pos  # the full body array is genuinely absent
    assert requirements.needs_reference_positions
    assert requirements.needs_spatial_start


def test_export_reading_no_reference_positions_needs_none():
    requirements = get_tracker_input_requirements(
        _contract(
            {
                "current_dof_pos": "current.dof_pos",
                "mimic_future_anchor_rot": "mimic.future_anchor_rot",
            }
        )
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
        _contract(
            {
                "mimic_future_anchor_rot": "mimic.future_anchor_rot",
                "mimic_future_dof_pos": "mimic.future_dof_pos",
            }
        ),
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=aligned,
    )
    assert set(onnx_inputs) == {"mimic_future_anchor_rot", "mimic_future_dof_pos"}


def test_an_input_with_no_source_is_reported_by_name(motion, alignment):
    """Silently dropping it just defers an opaque failure to onnxruntime."""
    aligned = align_future_motion_refs(motion["future_refs"], alignment)

    with pytest.raises(KeyError) as excinfo:
        build_tracker_onnx_inputs(
            _contract(
                {
                    "mimic_future_anchor_pos": "mimic.future_anchor_pos",
                    "current_anchor_pos": "current.anchor_pos",
                }
            ),
            dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
            dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
            anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
            root_local_ang_vel=np.zeros(3, dtype=np.float32),
            future_refs=aligned,
        )

    message = str(excinfo.value)
    assert "current_anchor_pos" in message
    assert "current.anchor_pos" in message


# ---------------------------------------------------------------------------
# Action-history inputs
#
# Training feeds processed_action_history and raw_action_history the actions
# from one call back, and the older historical.processed_actions the joint
# targets from two calls back (zeros before that, on this code).
# test_action_history_lag.py checks those constants against a real
# environment; these tests check that the deploy helpers turn them into the
# right input.
# ---------------------------------------------------------------------------

DEFAULT_POSE = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
PROCESSED = "processed_action_history"
RAW = "raw_action_history"


def _numbered(call: int) -> np.ndarray:
    """Values that identify the call that produced them."""
    return np.full(NUM_DOFS, 0.01 * (call + 1), dtype=np.float32)


def _fed_at(call: int, steps_back: int, fill: np.ndarray) -> np.ndarray:
    return _numbered(call - steps_back) if call >= steps_back else fill


def test_each_key_names_its_output_and_lag():
    assert action_history_output(PROCESSED) == "joint_pos_targets"
    assert action_history_output(RAW) == "actions"
    assert action_history_output(LEGACY_ACTION_HISTORY_KEY) == "joint_pos_targets"
    assert action_history_steps_back(PROCESSED, 3) == [1, 2, 3]
    assert action_history_steps_back(RAW, 1) == [1]
    assert action_history_steps_back(LEGACY_ACTION_HISTORY_KEY, 3) == [2, 3, 4]
    with pytest.raises(ValueError, match="not an action-history input"):
        action_history_steps_back("historical.actions", 1)


@pytest.mark.parametrize("steps_back", [1, 2])
def test_action_feedback_feeds_values_from_steps_back(steps_back):
    feedback = ActionFeedback(NUM_DOFS, (steps_back,))
    zeros = np.zeros(NUM_DOFS, dtype=np.float32)
    for call in range(6):
        np.testing.assert_array_equal(
            feedback.as_input()[0], _fed_at(call, steps_back, zeros)
        )
        feedback.record(_numbered(call))


def test_action_feedback_reset_restores_the_fill():
    feedback = ActionFeedback(NUM_DOFS, (2,), fill=DEFAULT_POSE)
    bound = feedback.as_input()
    for call in range(4):
        feedback.record(_numbered(call))
    feedback.reset()
    assert feedback.as_input() is bound
    for call in range(4):
        np.testing.assert_array_equal(bound[0], _fed_at(call, 2, DEFAULT_POSE))
        feedback.record(_numbered(call))


@pytest.mark.parametrize("key", [PROCESSED, LEGACY_ACTION_HISTORY_KEY])
def test_action_feedback_window_is_newest_first(key):
    steps = action_history_steps_back(key, 3)
    feedback = ActionFeedback(NUM_DOFS, steps)
    zeros = np.zeros(NUM_DOFS, dtype=np.float32)
    for call in range(7):
        for slot, steps_back in enumerate(steps):
            np.testing.assert_array_equal(
                feedback.as_input()[slot], _fed_at(call, steps_back, zeros)
            )
        feedback.record(_numbered(call))


@pytest.mark.parametrize(
    "key, first", [(PROCESSED, 1), (RAW, 1), (LEGACY_ACTION_HISTORY_KEY, 2)]
)
@pytest.mark.parametrize("reset_fill", ["zeros", "pd_action_offset"])
def test_yaml_entry_without_lag_fields_gets_the_keys_original_convention(
    monkeypatch, key, first, reset_fill
):
    """Old YAMLs read the same whatever this code's own reset convention is."""
    monkeypatch.setattr(tracker_inputs, "ACTION_HISTORY_RESET_FILL", reset_fill)
    entry = {"key": key, "shape": [1, 2, NUM_DOFS], "kind": "last_actions"}
    feedback = ActionFeedback.from_input_entry(
        entry, NUM_DOFS, pd_action_offset=DEFAULT_POSE
    )
    assert feedback.steps_back == (first, first + 1)
    assert feedback.output_name == action_history_output(key)
    np.testing.assert_array_equal(feedback.fill, np.zeros(NUM_DOFS))


def test_yaml_entry_lag_and_reset_value_fields_are_read():
    entry = {
        "key": PROCESSED,
        "shape": [1, 2, NUM_DOFS],
        "history": 2,
        "first_action_lag": 3,
        "reset_value": 0.5,
    }
    feedback = ActionFeedback.from_input_entry(entry, NUM_DOFS)
    assert feedback.steps_back == (3, 4)
    np.testing.assert_array_equal(feedback.fill, np.full(NUM_DOFS, 0.5))
    # steps_back and fill_before_available say the same thing in more detail
    # and win if both are present.
    both = ActionFeedback.from_input_entry(
        {**entry, "steps_back": [1, 2], "fill_before_available": "zeros"}, NUM_DOFS
    )
    assert both.steps_back == (1, 2)
    np.testing.assert_array_equal(both.fill, np.zeros(NUM_DOFS))


def test_yaml_entry_fields_are_honoured_and_checked():
    entry = {
        "key": LEGACY_ACTION_HISTORY_KEY,
        "shape": [1, 2, NUM_DOFS],
        "steps_back": [2, 3],
        "fill_before_available": DEFAULT_POSE.tolist(),
    }
    feedback = ActionFeedback.from_input_entry(entry, NUM_DOFS)
    assert feedback.steps_back == (2, 3)
    np.testing.assert_allclose(
        feedback.as_input(), np.stack([DEFAULT_POSE, DEFAULT_POSE])
    )
    by_name = ActionFeedback.from_input_entry(
        {**entry, "fill_before_available": "pd_action_offset"},
        NUM_DOFS,
        pd_action_offset=DEFAULT_POSE,
    )
    np.testing.assert_allclose(by_name.fill, DEFAULT_POSE)
    with pytest.raises(ValueError):
        ActionFeedback.from_input_entry({**entry, "steps_back": [2]}, NUM_DOFS)
    with pytest.raises(ValueError):
        ActionFeedback.from_input_entry(
            {"key": PROCESSED, "shape": [1, 1, NUM_DOFS + 1]}, NUM_DOFS
        )
    with pytest.raises(ValueError, match="not an action-history input"):
        ActionFeedback.from_input_entry({"shape": [1, 1, NUM_DOFS]}, NUM_DOFS)
    with pytest.raises(ValueError):
        resolve_fill("default_pose", NUM_DOFS)
    with pytest.raises(ValueError):
        resolve_fill("pd_action_offset", NUM_DOFS)
    with pytest.raises(ValueError):
        resolve_fill([0.0, float("nan"), 0.0, 0.0], NUM_DOFS)


def test_action_history_fill_follows_the_reset_convention(monkeypatch):
    monkeypatch.setattr(tracker_inputs, "ACTION_HISTORY_RESET_FILL", "zeros")
    assert action_history_fill(PROCESSED, DEFAULT_POSE) == "zeros"
    monkeypatch.setattr(tracker_inputs, "ACTION_HISTORY_RESET_FILL", "pd_action_offset")
    for key in (PROCESSED, LEGACY_ACTION_HISTORY_KEY):
        assert action_history_fill(key, DEFAULT_POSE) == pytest.approx(
            DEFAULT_POSE.tolist()
        )
        assert action_history_fill(key, None) == "zeros"
    # Raw actions reset to zeros either way.
    assert action_history_fill(RAW, DEFAULT_POSE) == "zeros"


def test_from_history_spec_reads_slot_facts_from_the_typed_contract():
    """The reference host builds a feedback from the typed HistorySlotSpec.

    The per-slot call counts are derived from the typed ``first_action_lag`` and
    ``slots`` (slot ``i`` is ``first_action_lag + i`` calls back), not the legacy
    ``steps_back`` duplicate. Every spec here carries only the four typed fields
    (``steps_back`` left at its default ``()``) to prove the derivation.
    """
    from types import SimpleNamespace

    from deployment.contract import HistorySlotSpec

    # The legacy processed history: lag 2, default-pose fill, only typed fields.
    processed = ActionFeedback.from_history_spec(
        SimpleNamespace(
            history=HistorySlotSpec(
                output_key="joint_pos_targets",
                slots=3,
                first_action_lag=2,
                reset_fill=DEFAULT_POSE.tolist(),
            )
        ),
        NUM_DOFS,
    )
    assert processed.output_name == "joint_pos_targets"
    assert processed.steps_back == (2, 3, 4)
    np.testing.assert_allclose(processed.fill, DEFAULT_POSE)

    # A "pd_action_offset" fill resolves against the supplied offset.
    by_name = ActionFeedback.from_history_spec(
        SimpleNamespace(
            history=HistorySlotSpec(
                output_key="joint_pos_targets",
                slots=1,
                first_action_lag=1,
                reset_fill="pd_action_offset",
            )
        ),
        NUM_DOFS,
        pd_action_offset=DEFAULT_POSE,
    )
    assert by_name.steps_back == (1,)
    np.testing.assert_allclose(by_name.fill, DEFAULT_POSE)

    # The raw history feeds the policy's own outputs back with a zeros reset.
    raw = ActionFeedback.from_history_spec(
        SimpleNamespace(
            history=HistorySlotSpec(
                output_key="actions",
                slots=1,
                first_action_lag=1,
                reset_fill="zeros",
            )
        ),
        NUM_DOFS,
    )
    assert raw.output_name == "actions"
    assert raw.steps_back == (1,)
    np.testing.assert_array_equal(raw.fill, np.zeros(NUM_DOFS))


def test_action_history_entries_match_on_the_semantic_key():
    meta = {
        "policy_inputs": [
            {"name": "current_dof_pos", "key": "current.dof_pos"},
            {"name": "renamed_history", "key": PROCESSED},
            {"name": "raw", "key": RAW},
        ]
    }
    entries = action_history_entries(meta)
    assert sorted(entries) == [PROCESSED, RAW]
    assert entries[PROCESSED]["name"] == "renamed_history"
    assert action_history_entries({"policy_inputs": []}) == {}


@pytest.mark.parametrize("key", [PROCESSED, LEGACY_ACTION_HISTORY_KEY])
@pytest.mark.parametrize("history", ["window", "flat"])
def test_build_inputs_accepts_a_window_or_a_single_slot(
    motion, alignment, key, history
):
    aligned = align_future_motion_refs(motion["future_refs"], alignment)
    processed = ActionFeedback(NUM_DOFS, (1, 2) if history == "window" else (1,))
    raw = ActionFeedback(NUM_DOFS, (1,), output_name="actions")
    for call in range(4):
        processed.record(_numbered(call))
        raw.record(-_numbered(call))
    prev_actions = (
        processed.as_input() if history == "window" else processed.as_input()[0]
    )
    onnx_inputs = build_tracker_onnx_inputs(
        _contract({"history": key, "raw": RAW}),
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=aligned,
        prev_actions=prev_actions,
        prev_raw_actions=raw.as_input(),
    )
    fed = onnx_inputs["history"]
    slots = 2 if history == "window" else 1
    assert fed.shape == (1, slots, NUM_DOFS)
    np.testing.assert_array_equal(fed[0, 0], _numbered(3))
    if history == "window":
        np.testing.assert_array_equal(fed[0, 1], _numbered(2))
    np.testing.assert_array_equal(onnx_inputs["raw"], -_numbered(3)[None, None])


def test_build_inputs_feeds_each_action_history_its_own_window(motion, alignment):
    """Two processed inputs at different lags must not share one window."""
    aligned = align_future_motion_refs(motion["future_refs"], alignment)
    current = ActionFeedback(NUM_DOFS, action_history_steps_back(PROCESSED, 1))
    legacy = ActionFeedback(
        NUM_DOFS, action_history_steps_back(LEGACY_ACTION_HISTORY_KEY, 1)
    )
    for call in range(4):
        current.record(_numbered(call))
        legacy.record(_numbered(call))
    onnx_inputs = build_tracker_onnx_inputs(
        _contract({"current": PROCESSED, "legacy": LEGACY_ACTION_HISTORY_KEY}),
        dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
        dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
        anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
        root_local_ang_vel=np.zeros(3, dtype=np.float32),
        future_refs=aligned,
        action_histories={
            PROCESSED: current.as_input(),
            LEGACY_ACTION_HISTORY_KEY: legacy.as_input(),
        },
    )
    np.testing.assert_array_equal(onnx_inputs["current"][0, 0], _numbered(3))
    np.testing.assert_array_equal(onnx_inputs["legacy"][0, 0], _numbered(2))
    with pytest.raises(ValueError, match="not an action-history input"):
        build_tracker_onnx_inputs(
            _contract({}),
            dof_pos=np.zeros(NUM_DOFS, dtype=np.float32),
            dof_vel=np.zeros(NUM_DOFS, dtype=np.float32),
            anchor_rot=_yaw_quat(ROBOT_YAW_DEG),
            root_local_ang_vel=np.zeros(3, dtype=np.float32),
            future_refs=aligned,
            action_histories={"historical.raw_actions": current.as_input()},
        )
