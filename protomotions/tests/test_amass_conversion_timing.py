# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for progress reporting when AMASS conversions slice a motion."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_amass_slicing_preserves_the_run_start_time(monkeypatch, tmp_path, capsys):
    """Clip bounds control slicing without replacing the progress timer."""
    scripts = Path(__file__).resolve().parents[2] / "data" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "_amass_conversion_timing", scripts / "convert_amass_to_proto.py"
    )
    conversion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion)
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    poses = np.arange(5 * 72).reshape(5, 72)
    translations = np.arange(5 * 3).reshape(5, 3)
    np.savez(clip_dir / "clip.npz", poses=poses, trans=translations, mocap_framerate=1)
    config = tmp_path / "timing.yaml"
    config.write_text(
        "motions:\n- file: clips/clip.motion\n"
        "  sub_motions:\n  - timings: {start: 1, end: 3}\n"
    )
    monkeypatch.setattr(conversion, "extract_kinematic_info", lambda path: object())
    times = iter([100.0, 102.0, 102.0])
    monkeypatch.setattr(
        conversion,
        "time",
        SimpleNamespace(
            time=lambda: next(times),
            localtime=lambda seconds: seconds,
            strftime=lambda format, value: "clock",
        ),
    )
    processed = []
    monkeypatch.setattr(
        conversion, "process_motion_segment", lambda *args: processed.append(args)
    )
    conversion.main(tmp_path, motion_configs=[str(config)])

    assert len(processed) == 1
    np.testing.assert_array_equal(processed[0][0], poses[1:3])
    np.testing.assert_array_equal(processed[0][1], translations[1:3])
    assert "Average time per file: 0:00:02" in capsys.readouterr().out
