# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for protomotions.components.motion_lib MotionLib query helpers and
the underlying calc_frame_blend interpolation utility.

Avoids file IO by either using the empty Null Object pattern
(``MotionLib.empty()``) or hand-populating tensor fields on a stub.
"""

from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path

import pytest
import torch

from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.utils.motion_interpolation_utils import calc_frame_blend


def _run_motion_lib_main():
    module_path = Path(sys.modules[MotionLib.__module__].__file__)
    runpy.run_path(str(module_path), run_name="__main__")


# ---------- calc_frame_blend ---------------------------------------------------


def test_calc_frame_blend_returns_lower_frame_and_blend_for_midpoint():
    """At time = 0.5 * length on a 6-frame motion, frame_idx0 = 2 and blend = 0.5."""
    time = torch.tensor([0.5])
    length = torch.tensor([1.0])
    num_frames = torch.tensor([6])
    dt = torch.tensor([1.0 / 5])  # 6 frames span [0, 5*dt]

    frame_idx0, frame_idx1, blend = calc_frame_blend(time, length, num_frames, dt)

    assert frame_idx0.item() == 2
    assert frame_idx1.item() == 3
    assert math.isclose(blend.item(), 0.5, abs_tol=1e-5)


def test_calc_frame_blend_clamps_overshoot_time_to_last_frame():
    """time outside [0, length] clamps indices and blend to boundary frames."""
    time = torch.tensor([5.0, -1.0])
    length = torch.tensor([1.0, 1.0])
    num_frames = torch.tensor([4, 4])
    dt = torch.tensor([1.0 / 3, 1.0 / 3])

    frame_idx0, frame_idx1, blend = calc_frame_blend(time, length, num_frames, dt)

    # Overshoot: phase=1.0, frame_idx0 = num_frames-1 = 3, frame_idx1 = min(4, 3) = 3.
    assert frame_idx0[0].item() == 3
    assert frame_idx1[0].item() == 3
    assert math.isclose(blend[0].item(), 0.0, abs_tol=1e-5)
    # Negative time: phase=0.0, frame_idx0 = 0, frame_idx1 = 1.
    assert frame_idx0[1].item() == 0
    assert frame_idx1[1].item() == 1
    assert math.isclose(blend[1].item(), 0.0, abs_tol=1e-5)


def test_calc_frame_blend_blend_is_zero_at_exact_frame_boundaries():
    """At exact frame times, blend should be (close to) 0 within float tolerance."""
    num_frames = torch.tensor([5])
    dt = torch.tensor([0.25])  # 5 frames, frame times {0, 0.25, 0.5, 0.75, 1.0}
    length = torch.tensor([1.0])

    for k in range(4):
        time = torch.tensor([k * 0.25])
        idx0, idx1, blend = calc_frame_blend(time, length, num_frames, dt)
        assert idx0.item() == k
        assert idx1.item() == min(k + 1, 4)
        assert math.isclose(blend.item(), 0.0, abs_tol=1e-5)


# ---------- MotionLib.empty() and counting helpers -----------------------------


def test_empty_motion_lib_reports_zero_motions_and_zero_total_length():
    motion_lib = MotionLib.empty(device="cpu")
    assert motion_lib.num_motions() == 0
    # Empty sum returns scalar tensor or python 0.
    total = motion_lib.get_total_length()
    if isinstance(total, torch.Tensor):
        assert total.item() == 0
    else:
        assert total == 0


def test_empty_motion_lib_get_motion_length_with_none_returns_empty_tensor():
    motion_lib = MotionLib.empty(device="cpu")
    lengths = motion_lib.get_motion_length(None)
    assert lengths.shape == (0,)


def test_empty_motion_lib_get_motion_num_frames_with_none_returns_empty_tensor():
    motion_lib = MotionLib.empty(device="cpu")
    n_frames = motion_lib.get_motion_num_frames(None)
    assert n_frames.shape == (0,)


def test_empty_motion_lib_has_goal_states_returns_false():
    motion_lib = MotionLib.empty(device="cpu")
    assert motion_lib.has_goal_states() is False


def test_empty_motion_lib_get_goal_state_times_raises_when_unloaded():
    motion_lib = MotionLib.empty(device="cpu")
    with pytest.raises(RuntimeError):
        motion_lib.get_goal_state_times(torch.tensor([0]))


def test_empty_motion_lib_default_config_init_round_trip():
    """Constructing via MotionLibConfig(motion_file=None) matches the empty
    factory: same field shapes, same Null Object behavior."""
    direct = MotionLib(MotionLibConfig(motion_file=None), device="cpu")
    assert direct.num_motions() == 0
    assert direct.has_goal_states() is False
    assert direct.lrs is None
    assert direct.goal_states is None
    assert direct.motion_files == ()


# ---------- Hand-populated MotionLib query paths -------------------------------


def _populate_motion_lib(motion_lib, motion_lengths, motion_num_frames):
    """Stamp the minimum tensor fields needed for length / index queries."""
    motion_lib.motion_lengths = torch.tensor(motion_lengths, dtype=torch.float32)
    motion_lib.motion_num_frames = torch.tensor(motion_num_frames, dtype=torch.long)
    motion_lib.motion_dt = torch.tensor(
        [length / max(n - 1, 1) for length, n in zip(motion_lengths, motion_num_frames)],
        dtype=torch.float32,
    )
    motion_lib.length_starts = torch.tensor(
        [sum(motion_num_frames[:i]) for i in range(len(motion_num_frames))],
        dtype=torch.long,
    )


def _identity_quat(num_frames: int, num_bodies: int) -> torch.Tensor:
    quats = torch.zeros(num_frames, num_bodies, 4, dtype=torch.float32)
    quats[..., 3] = 1.0
    return quats


def _populated_motion_lib(*, contacts: str | None = "bool", include_lrs: bool = False):
    motion_lib = MotionLib.empty(device="cpu")
    _populate_motion_lib(motion_lib, motion_lengths=[0.2, 0.1], motion_num_frames=[3, 2])
    motion_lib.motion_weights = torch.tensor([0.25, 0.75], dtype=torch.float32)
    motion_lib.motion_files = ("walk.motion", "turn.motion")

    total_frames = 5
    num_bodies = 2
    dof_dim = 3

    base_pos = torch.arange(
        total_frames * num_bodies * 3, dtype=torch.float32
    ).reshape(total_frames, num_bodies, 3)
    base_dof = torch.arange(total_frames * dof_dim, dtype=torch.float32).reshape(
        total_frames, dof_dim
    )

    motion_lib.gts = base_pos.clone()
    motion_lib.grs = _identity_quat(total_frames, num_bodies)
    motion_lib.gvs = base_pos + 100.0
    motion_lib.gavs = base_pos + 200.0
    motion_lib.dps = base_dof.clone()
    motion_lib.dvs = base_dof + 50.0
    motion_lib.goal_states = None
    motion_lib.lrs = _identity_quat(total_frames, num_bodies) if include_lrs else None

    if contacts == "bool":
        motion_lib.contacts = torch.tensor(
            [
                [False, False],
                [True, False],
                [False, True],
                [True, True],
                [False, True],
            ],
            dtype=torch.bool,
        )
    elif contacts == "float":
        motion_lib.contacts = torch.tensor(
            [
                [0.0, 0.25],
                [1.0, 0.75],
                [0.5, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    elif contacts == "zeros":
        motion_lib.contacts = torch.zeros(total_frames, num_bodies, dtype=torch.bool)
    elif contacts is None:
        motion_lib.contacts = None
    else:
        raise ValueError(f"Unexpected contacts mode: {contacts}")

    return motion_lib


def _motion_file_payload(
    num_frames: int,
    *,
    offset: float = 0.0,
    fps: float = 10.0,
    contacts: torch.Tensor | None = None,
    include_lrs: bool = True,
):
    num_bodies = 2
    dof_dim = 3
    rigid_body_pos = torch.arange(
        num_frames * num_bodies * 3, dtype=torch.float32
    ).reshape(num_frames, num_bodies, 3)
    dof_pos = torch.arange(num_frames * dof_dim, dtype=torch.float32).reshape(
        num_frames, dof_dim
    )

    payload = {
        "fps": fps,
        "rigid_body_pos": rigid_body_pos + offset,
        "rigid_body_rot": _identity_quat(num_frames, num_bodies),
        "rigid_body_vel": rigid_body_pos + offset + 100.0,
        "rigid_body_ang_vel": rigid_body_pos + offset + 200.0,
        "dof_pos": dof_pos + offset,
        "dof_vel": dof_pos + offset + 50.0,
    }
    if contacts is not None:
        payload["rigid_body_contacts"] = contacts
    if include_lrs:
        payload["local_rigid_body_rot"] = _identity_quat(num_frames, num_bodies)
    return payload


def test_get_motion_length_indexes_specific_motion_ids():
    motion_lib = MotionLib.empty(device="cpu")
    _populate_motion_lib(motion_lib, motion_lengths=[1.0, 2.0, 3.0], motion_num_frames=[4, 6, 10])

    lengths = motion_lib.get_motion_length(torch.tensor([0, 2]))

    assert torch.allclose(lengths, torch.tensor([1.0, 3.0]))
    # Pass-through with None returns the full tensor.
    assert torch.allclose(motion_lib.get_motion_length(None), torch.tensor([1.0, 2.0, 3.0]))


def test_get_motion_num_frames_indexes_specific_motion_ids():
    motion_lib = MotionLib.empty(device="cpu")
    _populate_motion_lib(motion_lib, motion_lengths=[1.0, 2.0], motion_num_frames=[5, 9])

    n_frames = motion_lib.get_motion_num_frames(torch.tensor([1]))
    assert n_frames.tolist() == [9]


def test_calc_closest_frame_rounds_to_nearest_index_within_motion():
    motion_lib = MotionLib.empty(device="cpu")
    _populate_motion_lib(motion_lib, motion_lengths=[1.0], motion_num_frames=[5])
    motion_lib.device = "cpu"

    # Times [0, 0.2, 0.4, 0.6, 0.8, 1.0] → frames [0, 1, 2, 3, 3, 4]
    # (frame_idx = round(t / length * (n-1)) = round(t * 4))
    for t, expected in [
        (0.0, 0),
        (0.2, 1),
        (0.4, 2),
        (0.6, 2),  # banker's rounding edge: 0.6*4 = 2.4 → 2
        (0.8, 3),
        (1.0, 4),
    ]:
        idx = motion_lib._calc_closest_frame(
            torch.tensor([0]), torch.tensor([t], dtype=torch.float32)
        )
        assert idx.item() == expected, f"t={t}: expected {expected}, got {idx.item()}"


def test_calc_closest_frame_clips_negative_and_overshoot_times():
    motion_lib = MotionLib.empty(device="cpu")
    _populate_motion_lib(motion_lib, motion_lengths=[1.0], motion_num_frames=[5])
    motion_lib.device = "cpu"

    idx = motion_lib._calc_closest_frame(
        torch.tensor([0, 0]), torch.tensor([-0.5, 5.0], dtype=torch.float32)
    )
    # Negative clipped to 0 → frame 0; overshoot clipped to length=1 → frame n-1=4.
    assert idx.tolist() == [0, 4]


def test_calc_frame_blend_from_id_and_time_consistent_with_pure_helper():
    """The MotionLib-level _calc_frame_blend_from_id_and_time should agree with
    the standalone calc_frame_blend on a clipped time inside the motion."""
    motion_lib = MotionLib.empty(device="cpu")
    _populate_motion_lib(motion_lib, motion_lengths=[1.0], motion_num_frames=[5])

    times = torch.tensor([0.5])
    motion_ids = torch.tensor([0])

    idx0, idx1, blend = motion_lib._calc_frame_blend_from_id_and_time(motion_ids, times)
    expected = calc_frame_blend(
        times,
        motion_lib.motion_lengths[motion_ids],
        motion_lib.motion_num_frames[motion_ids],
        motion_lib.motion_dt[motion_ids],
    )

    assert torch.equal(idx0, expected[0])
    assert torch.equal(idx1, expected[1])
    assert torch.allclose(blend, expected[2])


def test_motion_state_exact_frame_uses_motion_offsets_and_optional_fields():
    motion_lib = _populated_motion_lib(include_lrs=True)
    motion_ids = torch.tensor([0, 1])
    frame_indices = torch.tensor([2, 1])
    sample_indices = torch.tensor([2, 4])

    state = motion_lib.get_motion_state_exact_frame(motion_ids, frame_indices)

    assert torch.equal(state.rigid_body_pos, motion_lib.gts[sample_indices])
    assert torch.equal(state.rigid_body_rot, motion_lib.grs[sample_indices])
    assert torch.equal(state.rigid_body_vel, motion_lib.gvs[sample_indices])
    assert torch.equal(state.rigid_body_ang_vel, motion_lib.gavs[sample_indices])
    assert torch.equal(state.dof_pos, motion_lib.dps[sample_indices])
    assert torch.equal(state.dof_vel, motion_lib.dvs[sample_indices])
    assert torch.equal(state.local_rigid_body_rot, motion_lib.lrs[sample_indices])
    assert torch.equal(state.rigid_body_contacts, motion_lib.contacts[sample_indices])


def test_get_motion_state_interpolates_positions_dofs_and_bool_contacts():
    motion_lib = _populated_motion_lib()

    state = motion_lib.get_motion_state(torch.tensor([0]), torch.tensor([0.05]))

    assert torch.allclose(
        state.rigid_body_pos[0], (motion_lib.gts[0] + motion_lib.gts[1]) / 2.0
    )
    assert torch.allclose(
        state.rigid_body_vel[0], (motion_lib.gvs[0] + motion_lib.gvs[1]) / 2.0
    )
    assert torch.allclose(
        state.rigid_body_ang_vel[0], (motion_lib.gavs[0] + motion_lib.gavs[1]) / 2.0
    )
    assert torch.allclose(
        state.dof_pos[0], (motion_lib.dps[0] + motion_lib.dps[1]) / 2.0
    )
    assert torch.allclose(
        state.dof_vel[0], (motion_lib.dvs[0] + motion_lib.dvs[1]) / 2.0
    )
    assert torch.equal(
        state.rigid_body_contacts[0], motion_lib.contacts[0] | motion_lib.contacts[1]
    )


def test_get_motion_state_averages_smoothed_float_contacts():
    motion_lib = _populated_motion_lib(contacts="float")

    state = motion_lib.get_motion_state(torch.tensor([0]), torch.tensor([0.05]))

    assert torch.allclose(
        state.rigid_body_contacts[0],
        (motion_lib.contacts[0] + motion_lib.contacts[1]) / 2.0,
    )


def test_get_motion_state_uses_local_rotations_for_expmap_dof_positions():
    motion_lib = _populated_motion_lib(include_lrs=True)

    state = motion_lib.get_motion_state(torch.tensor([0]), torch.tensor([0.05]))

    assert torch.allclose(state.dof_pos, torch.zeros(1, 3), atol=1e-6)
    assert not torch.allclose(
        (motion_lib.dps[0:1] + motion_lib.dps[1:2]) / 2.0, state.dof_pos
    )


def test_goal_state_times_and_batched_cache():
    motion_lib = _populated_motion_lib()
    motion_lib.goal_states = torch.tensor([False, True, True, False, False])

    goal_times = motion_lib.get_goal_state_times(torch.tensor([0, 1]))
    assert torch.allclose(goal_times, torch.tensor([0.2, -1.0]))

    cached = motion_lib.get_goal_state_times_batched(torch.tensor([1, 0, 1]))
    assert torch.allclose(cached, torch.tensor([-1.0, 0.2, -1.0]))

    motion_lib.goal_states[:] = False
    cached_again = motion_lib.get_goal_state_times_batched(torch.tensor([0, 1]))
    assert torch.allclose(cached_again, torch.tensor([0.2, -1.0]))


def test_process_packaged_motion_file_name_multi_gpu_selects_rank_file(
    tmp_path, monkeypatch
):
    (tmp_path / "chunk_0.pt").write_text("rank0")
    (tmp_path / "chunk_2.pt").write_text("rank2")
    motion_lib = MotionLib.empty(device="cpu")

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    selected = motion_lib.process_packaged_motion_file_name_multi_gpu(
        str(tmp_path / "chunk_slurmrank.pt")
    )

    assert selected == str(tmp_path / "chunk_2.pt")
    assert motion_lib.different_motion_files_across_ranks is True
    assert motion_lib.process_packaged_motion_file_name_multi_gpu("plain.pt") == "plain.pt"


def test_fetch_motion_files_handles_yaml_single_files_directories_and_invalid(
    tmp_path,
):
    motion_lib = MotionLib.empty(device="cpu")
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.motion").write_text("a")
    (nested / "b.motion").write_text("b")
    (tmp_path / "array.npz").write_text("npz")
    yaml_path = tmp_path / "motions.yaml"
    yaml_path.write_text(
        """
motions:
  - file: a.motion
    weight: 2.5
  - file: nested/b.motion
"""
    )

    motion_files, weights = motion_lib._fetch_motion_files(str(yaml_path))
    assert motion_files == [str(tmp_path / "a.motion"), str(nested / "b.motion")]
    assert weights == [2.5, 1.0]

    assert motion_lib._fetch_motion_files(str(tmp_path / "a.motion")) == (
        [str(tmp_path / "a.motion")],
        [1.0],
    )
    assert motion_lib._fetch_motion_files(str(tmp_path / "array.npz")) == (
        [str(tmp_path / "array.npz")],
        [1.0],
    )

    directory_files, directory_weights = motion_lib._fetch_motion_files(str(tmp_path))
    assert set(directory_files) == {
        str(tmp_path / "a.motion"),
        str(nested / "b.motion"),
    }
    assert directory_weights == [1.0, 1.0]

    with pytest.raises(AssertionError, match="Motion file must be"):
        motion_lib._fetch_motion_files(str(tmp_path / "missing.txt"))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(AssertionError, match="No motion files found"):
        motion_lib._fetch_motion_files(str(empty_dir))


def test_packaged_motion_lib_save_load_constructor_and_zero_contact_discard(
    tmp_path, caplog
):
    motion_lib = _populated_motion_lib(contacts="zeros")
    package_path = tmp_path / "packed.pt"

    motion_lib.save_to_file(package_path)
    with pytest.raises(AssertionError, match="ends with .pt"):
        motion_lib.save_to_file(tmp_path / "packed.bin")

    loaded = MotionLib.empty(device="cpu")
    with caplog.at_level("WARNING"):
        loaded.load_from_file(package_path)

    assert loaded.contacts is None
    assert "Discarding contacts" in caplog.text
    assert torch.equal(loaded.gts, motion_lib.gts)
    assert loaded.lrs is None
    assert loaded.goal_states is None

    constructed = MotionLib(MotionLibConfig(motion_file=str(package_path)), device="cpu")
    assert torch.equal(constructed.gts, motion_lib.gts)
    assert constructed.contacts is None
    assert constructed.motion_file == str(package_path)


def test_smooth_contacts_validates_inputs_and_respects_motion_boundaries():
    no_contacts = _populated_motion_lib(contacts=None)
    no_contacts.smooth_contacts(3)
    assert no_contacts.contacts is None

    motion_lib = _populated_motion_lib()
    with pytest.raises(ValueError, match="positive"):
        motion_lib.smooth_contacts(0)
    with pytest.raises(ValueError, match="odd"):
        motion_lib.smooth_contacts(2)

    bad_contacts = _populated_motion_lib(contacts="float")
    with pytest.raises(ValueError, match="must be binary"):
        bad_contacts.smooth_contacts(3)

    smooth = _populated_motion_lib()
    smooth.contacts = torch.tensor(
        [
            [False, False],
            [False, False],
            [True, False],
            [False, False],
            [False, False],
        ],
        dtype=torch.bool,
    )
    smooth.smooth_contacts(3)

    assert smooth.contacts.dtype == torch.float32
    assert torch.all((smooth.contacts >= 0.0) & (smooth.contacts <= 1.0))
    assert torch.allclose(smooth.contacts[:3, 0], torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0]))
    assert torch.equal(smooth.contacts[3:, 0], torch.zeros(2))


def test_translate_all_motions_to_origin_moves_each_motion_independently():
    motion_lib = _populated_motion_lib()
    motion_lib.gts[0, 0, :2] = torch.tensor([10.0, -5.0])
    motion_lib.gts[3, 0, :2] = torch.tensor([-2.0, 4.0])
    before = motion_lib.gts.clone()
    target_xy = torch.tensor([1.0, 2.0])

    motion_lib.translate_all_motions_to_origin(target_xy)

    for start, num_frames in zip(motion_lib.length_starts, motion_lib.motion_num_frames):
        start_idx = start.item()
        end_idx = start_idx + num_frames.item()
        translation_xy = target_xy - before[start_idx, 0, :2]
        assert torch.allclose(motion_lib.gts[start_idx, 0, :2], target_xy)
        assert torch.allclose(
            motion_lib.gts[start_idx:end_idx, :, :2],
            before[start_idx:end_idx, :, :2] + translation_xy.reshape(1, 1, 2),
        )
        assert torch.allclose(
            motion_lib.gts[start_idx:end_idx, :, 2],
            before[start_idx:end_idx, :, 2],
        )

    zeroed = _populated_motion_lib()
    zeroed.translate_all_motions_to_origin()
    assert torch.allclose(zeroed.gts[0, 0, :2], torch.zeros(2))
    assert torch.allclose(zeroed.gts[3, 0, :2], torch.zeros(2))


def test_load_motions_from_yaml_packs_fields_weights_contacts_and_lrs(tmp_path):
    first_path = tmp_path / "first.motion"
    second_path = tmp_path / "second.motion"
    torch.save(
        _motion_file_payload(
            3,
            offset=10.0,
            contacts=torch.tensor(
                [[True, False], [False, True], [True, True]], dtype=torch.bool
            ),
        ),
        first_path,
    )
    torch.save(
        _motion_file_payload(
            2,
            offset=40.0,
            contacts=torch.tensor([[False, False], [True, False]], dtype=torch.bool),
        ),
        second_path,
    )
    yaml_path = tmp_path / "motions.yaml"
    yaml_path.write_text(
        """
motions:
  - file: first.motion
    weight: 0.25
  - file: second.motion
    weight: 0.75
"""
    )

    motion_lib = MotionLib(MotionLibConfig(motion_file=str(yaml_path)), device="cpu")

    assert motion_lib.motion_files == (str(first_path), str(second_path))
    assert torch.equal(motion_lib.motion_num_frames, torch.tensor([3, 2]))
    assert torch.equal(motion_lib.length_starts, torch.tensor([0, 3]))
    assert torch.allclose(motion_lib.motion_dt, torch.tensor([0.1, 0.1]))
    assert torch.allclose(motion_lib.motion_lengths, torch.tensor([0.2, 0.1]))
    assert torch.allclose(motion_lib.motion_weights, torch.tensor([0.25, 0.75]))
    assert motion_lib.gts.shape == (5, 2, 3)
    assert motion_lib.lrs.shape == (5, 2, 4)
    assert motion_lib.contacts.dtype == torch.bool
    assert torch.equal(motion_lib.contacts[0], torch.tensor([True, False]))


def test_load_motions_clips_long_motion_and_discards_all_zero_contacts(
    tmp_path, caplog
):
    motion_path = tmp_path / "long.motion"
    torch.save(
        _motion_file_payload(
            10,
            contacts=torch.zeros(10, 2, dtype=torch.bool),
            include_lrs=False,
        ),
        motion_path,
    )

    with caplog.at_level("WARNING"):
        motion_lib = MotionLib(
            MotionLibConfig(
                motion_file=str(motion_path),
                max_seconds=0.4,
                clip_delta=0.0,
            ),
            device="cpu",
        )

    assert torch.equal(motion_lib.motion_num_frames, torch.tensor([5]))
    assert torch.allclose(motion_lib.motion_lengths, torch.tensor([0.4]))
    assert motion_lib.gts.shape[0] == 5
    assert motion_lib.lrs is None
    assert motion_lib.contacts is None
    assert "Discarding contacts" in caplog.text


def test_load_motions_handles_absent_contact_labels(tmp_path):
    motion_path = tmp_path / "no_contacts.motion"
    torch.save(_motion_file_payload(2, contacts=None), motion_path)

    motion_lib = MotionLib(MotionLibConfig(motion_file=str(motion_path)), device="cpu")

    assert motion_lib.contacts is None
    assert motion_lib.lrs.shape == (2, 2, 4)


def test_motion_lib_cli_repackages_pt_motion_file(tmp_path, monkeypatch):
    input_lib = _populated_motion_lib()
    input_path = tmp_path / "input.pt"
    output_path = tmp_path / "output.pt"
    input_lib.save_to_file(input_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion_lib.py",
            "--motion-path",
            str(input_path),
            "--output-file",
            str(output_path),
            "--device",
            "cpu",
            "--max-seconds",
            "1.0",
            "--clip-delta",
            "0.25",
            "--clip-seed",
            "7",
        ],
    )

    _run_motion_lib_main()

    repacked = MotionLib(MotionLibConfig(motion_file=str(output_path)), device="cpu")
    assert torch.equal(repacked.gts, input_lib.gts)
    assert torch.equal(repacked.motion_num_frames, input_lib.motion_num_frames)
    assert repacked.motion_files == input_lib.motion_files


def test_motion_lib_cli_validates_yaml_relative_motion_reference(tmp_path, monkeypatch):
    yaml_path = tmp_path / "missing_motion.yaml"
    yaml_path.write_text(
        """
motions:
  - file: missing.motion
"""
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "motion_lib.py",
            "--motion-path",
            str(yaml_path),
            "--output-file",
            str(tmp_path / "output.pt"),
        ],
    )

    with pytest.raises(FileNotFoundError, match="Did you forget to copy"):
        _run_motion_lib_main()
