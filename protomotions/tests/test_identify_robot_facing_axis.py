# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the simulator-free parts of the robot facing-axis helper."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "identify_robot_facing_axis.py"
)


def _load_script_module():
    spec = spec_from_file_location("identify_robot_facing_axis", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_facing_axis_is_perpendicular_to_left_right_pair_in_xy():
    module = _load_script_module()

    facing = module._facing_from_left_right(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 2.0, 9.0]),
    )

    np.testing.assert_allclose(facing, np.array([-1.0, 0.0]))


def test_helper_explains_why_forward_axis_is_two_dimensional(capsys):
    module = _load_script_module()

    module._print_method_explanation()

    output = capsys.readouterr().out
    assert "Z-up" in output
    assert "2D" in output
    assert "left" in output
    assert "right" in output


def test_viewer_arrow_starts_at_root_and_follows_inferred_facing():
    module = _load_script_module()

    start, end = module._arrow_endpoints(
        root_pos=np.array([1.0, 2.0, 3.0]),
        facing_xy=np.array([0.0, -2.0]),
        length=0.75,
    )

    np.testing.assert_allclose(start, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(end, np.array([1.0, 1.25, 3.0]))
