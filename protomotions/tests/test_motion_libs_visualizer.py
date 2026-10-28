# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clip lookup in examples/motion_libs_visualizer.py, without a simulator.

Evaluation saves a predicted motion library whose failed clips can stop at the
failing frame and whose unplayed clips have no frames. Shown next to the
ground truth, each library must stay inside its own copy of the clip.
"""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig


VISUALIZER_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "motion_libs_visualizer.py"
)


@pytest.fixture
def visualizer(monkeypatch):
    # The script parses its arguments at import time. Newton needs no
    # simulator import before torch, so nothing heavy is loaded here.
    monkeypatch.setattr(
        sys,
        "argv",
        [VISUALIZER_PATH.name, "--motion_files", "unused.pt", "--simulator", "newton"],
    )
    return runpy.run_path(str(VISUALIZER_PATH), run_name="motion_libs_visualizer_unit")


def _motion_lib(tmp_path, name, num_frames, contacts=False):
    """Packed library whose dof_pos holds each frame's global index.

    With ``contacts``, body 0 touches the ground on even stored frames and
    body 1 on odd ones.
    """
    num_frames = torch.tensor(num_frames, dtype=torch.long)
    total = int(num_frames.sum())
    length_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long), num_frames.cumsum(0)[:-1]]
    )
    frame_ids = torch.arange(total, dtype=torch.float32)
    rot = torch.zeros(total, 2, 4)
    rot[..., 3] = 1.0
    contact_labels = None
    if contacts:
        even = frame_ids % 2 == 0
        contact_labels = torch.stack([even, ~even], dim=-1)
    path = tmp_path / f"{name}.pt"
    torch.save(
        {
            "gts": torch.zeros(total, 2, 3),
            "grs": rot,
            "gvs": torch.zeros(total, 2, 3),
            "gavs": torch.zeros(total, 2, 3),
            "dps": frame_ids[:, None],
            "dvs": torch.zeros(total, 1),
            "length_starts": length_starts,
            "motion_num_frames": num_frames,
            "motion_lengths": num_frames.float() / 30,
            "motion_dt": torch.full((len(num_frames),), 1 / 30),
            "motion_weights": torch.ones(len(num_frames)),
            "motion_files": tuple(f"clip_{i}" for i in range(len(num_frames))),
            "contacts": contact_labels,
        },
        path,
    )
    return MotionLib(MotionLibConfig(motion_file=str(path)))


def _frame_id(visualizer, motion_lib, motion, frame):
    state = visualizer["_exact_frame_state"](motion_lib, torch.tensor([motion]), frame)
    return None if state is None else int(state.dof_pos[0, 0])


def test_short_clip_holds_its_own_last_frame(visualizer, tmp_path):
    ground_truth = _motion_lib(tmp_path, "gt", [3, 2])
    # Clip 0 failed after one frame; clip 1 played in full.
    predicted = _motion_lib(tmp_path, "pred", [1, 2])

    assert _frame_id(visualizer, ground_truth, 0, 2) == 2
    # Unclamped, frame 2 of clip 0 is global frame 2: clip 1's second frame.
    assert _frame_id(visualizer, predicted, 0, 2) == 0
    assert _frame_id(visualizer, predicted, 1, 1) == 2


def test_short_last_clip_stays_inside_the_stored_data(visualizer, tmp_path):
    predicted = _motion_lib(tmp_path, "pred", [3, 1])

    # Unclamped, frame 1 of the last clip is global frame 4, past the end.
    assert _frame_id(visualizer, predicted, 1, 1) == 3


def test_clip_with_no_frames_is_left_out(visualizer, tmp_path):
    # Clip 0 never played. Frame -1 would wrap round to the last stored frame.
    predicted = _motion_lib(tmp_path, "pred", [0, 2])

    assert _frame_id(visualizer, predicted, 0, 0) is None
    assert _frame_id(visualizer, predicted, 1, 0) == 0


def test_playback_length_is_the_longest_copy(visualizer, tmp_path):
    ground_truth = _motion_lib(tmp_path, "gt", [3, 2])
    predicted = _motion_lib(tmp_path, "pred", [0, 1])
    longest = visualizer["_longest_clip_length"]

    # The order of --motion_files no longer decides how long a clip plays.
    assert longest([predicted, ground_truth], 0) == 3
    assert longest([ground_truth, predicted], 1) == 2
    assert longest([predicted], 0) == 0


def _playback_without_simulator(visualizer, motion_libs):
    """A visualizer with only the state that pose lookup and smoothness read."""
    cls = visualizer["MotionVisualizerSmoothness"]
    playback = cls.__new__(cls)
    playback.motion_libs = motion_libs
    playback.device = torch.device("cpu")
    playback.kinematic_info = SimpleNamespace(num_bodies=2, num_dofs=1)
    playback.library_origins_xy = [
        torch.tensor([float(i), 0.0]) for i in range(len(motion_libs))
    ]
    playback.window_frames = 4
    playback.use_data_vel = False
    playback.metric = "nj"
    playback.current_frame = 0
    return playback


def _show_motion(visualizer, playback, motion, frame):
    playback.current_motion_idx = motion
    playback.current_motion_length = visualizer["_longest_clip_length"](
        playback.motion_libs, motion
    )
    playback._precompute_motion_smoothness()
    playback.current_frame = frame
    dof_pos, rigid_body_pos, _, _ = playback._get_current_pose()
    return dof_pos[:, 0].tolist(), rigid_body_pos[:, 0, 2].tolist()


def test_library_without_the_clip_is_parked_out_of_sight(visualizer, tmp_path):
    predicted = _motion_lib(tmp_path, "pred", [0, 2, 0])
    ground_truth = _motion_lib(tmp_path, "gt", [6, 5, 0])
    playback = _playback_without_simulator(visualizer, [predicted, ground_truth])

    # Predicted never played clip 0: its robot is parked, ground truth plays on.
    dof_pos, root_z = _show_motion(visualizer, playback, 0, 5)
    assert dof_pos == [0.0, 5.0]
    assert root_z == [-100.0, 0.0]
    assert playback.precomputed_smoothness.shape == (6, 2, 2)

    # Clip 1 failed after two frames: predicted holds its last frame.
    dof_pos, root_z = _show_motion(visualizer, playback, 1, 4)
    assert dof_pos == [1.0, 10.0]
    assert root_z == [0.0, 0.0]

    # No library has clip 2: nothing to precompute, and both robots are parked.
    dof_pos, root_z = _show_motion(visualizer, playback, 2, 0)
    assert playback.precomputed_smoothness is None
    assert root_z == [-100.0, -100.0]


def test_contact_markers_come_from_each_librarys_own_frame(visualizer, tmp_path):
    # Clip 0: predicted failed after one frame, ground truth has four, and the
    # third library never played it.
    predicted = _motion_lib(tmp_path, "pred", [1, 2], contacts=True)
    ground_truth = _motion_lib(tmp_path, "gt", [4, 2], contacts=True)
    unplayed = _motion_lib(tmp_path, "unplayed", [0, 2], contacts=True)
    playback = _playback_without_simulator(
        visualizer, [predicted, ground_truth, unplayed]
    )
    playback.contact_marker_name = "contacts"
    rigid_body_rot = torch.zeros(3, 2, 4)
    rigid_body_rot[..., 3] = 1.0
    bodies = SimpleNamespace(
        rigid_body_pos=torch.ones(3, 2, 3), rigid_body_rot=rigid_body_rot
    )
    playback.simulator = SimpleNamespace(get_bodies_state=lambda: bodies)
    playback.current_motion_idx = 0
    playback.current_motion_length = visualizer["_longest_clip_length"](
        playback.motion_libs, 0
    )
    playback.current_frame = 3

    markers = playback._update_contact_markers()["contacts"]

    # Shown markers sit on the body (z = 1); hidden ones are parked at -100.
    # Predicted holds its only frame (even: body 0), ground truth is on frame 3
    # (odd: body 1), and the unplayed library shows nothing.
    assert markers.translation[..., 2].tolist() == [
        [1.0, -100.0],
        [-100.0, 1.0],
        [-100.0, -100.0],
    ]


def test_libraries_with_unplayed_clips_can_be_moved_to_their_origin(tmp_path):
    # The visualizer moves every library to its origin before anything else.
    for num_frames in ([2, 0], [0, 2], [0, 0]):
        predicted = _motion_lib(tmp_path, "pred", num_frames)
        predicted.translate_all_motions_to_origin(torch.tensor([1.0, 0.0]))
        assert torch.all(predicted.gts[..., 0] == 1.0)
