# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for termination kernels' Python source and TorchScript exports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from protomotions.envs.terminations import base as scripted_base
from protomotions.envs.terminations import task as scripted_task


def _load_with_python_jit(module_name: str, relative_path: str):
    module_path = Path(__file__).parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None

    original_script = torch.jit.script
    torch.jit.script = lambda fn=None, **_: fn if fn is not None else (lambda f: f)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        torch.jit.script = original_script

    return module


python_base = _load_with_python_jit(
    "_termination_base_python",
    "protomotions/envs/terminations/base.py",
)
python_task = _load_with_python_jit(
    "_termination_task_python",
    "protomotions/envs/terminations/task.py",
)


def test_base_termination_python_source_matches_scripted_exports():
    contacts = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, True, False],
        ]
    )
    pos = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.05], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.50], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.05], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float,
    )
    termination_heights = torch.tensor([0.1, 0.1, 0.1])
    allowed = torch.tensor([0], dtype=torch.long)
    progress = torch.tensor([5, 5, 1])

    assert torch.equal(
        python_base.check_fall_contact_term(contacts, allowed, progress),
        scripted_base.check_fall_contact_term(contacts, allowed, progress),
    )
    assert torch.equal(
        python_base.check_height_term(pos, termination_heights, allowed),
        scripted_base.check_height_term(pos, termination_heights, allowed),
    )
    assert torch.equal(
        python_base.check_max_length_term(progress, max_episode_length=5.5),
        scripted_base.check_max_length_term(progress, max_episode_length=5.5),
    )
    assert torch.equal(
        python_base.combine_fall_termination(
            contacts, pos, termination_heights, allowed, progress
        ),
        scripted_base.combine_fall_termination(
            contacts, pos, termination_heights, allowed, progress
        ),
    )


def test_task_termination_python_source_matches_scripted_exports():
    head = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [3.0, 4.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = torch.tensor(
        [
            [3.0, 4.0, 1.6],
            [3.0, 4.0, 1.6],
            [0.0, 0.0, 1.5],
        ]
    )
    progress = torch.tensor([11, 11, 10])

    assert torch.equal(
        python_task.check_path_distance_term(
            head, target, fail_dist=4.9, progress_buf=progress, min_progress=10
        ),
        scripted_task.check_path_distance_term(
            head, target, fail_dist=4.9, progress_buf=progress, min_progress=10
        ),
    )
    assert torch.equal(
        python_task.check_path_height_term(
            head,
            target,
            fail_height_dist=0.5,
            progress_buf=progress,
            min_progress=10,
        ),
        scripted_task.check_path_height_term(
            head,
            target,
            fail_height_dist=0.5,
            progress_buf=progress,
            min_progress=10,
        ),
    )

    root = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    prev = torch.zeros_like(root)
    tar_dir = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    tar_speed = torch.tensor([1.0, 1.0, 1.0])

    assert torch.equal(
        python_task.check_steering_velocity_error(
            root,
            prev,
            tar_dir,
            tar_speed,
            dt=0.1,
            speed_tolerance=0.2,
            direction_tolerance=0.5,
        ),
        scripted_task.check_steering_velocity_error(
            root,
            prev,
            tar_dir,
            tar_speed,
            dt=0.1,
            speed_tolerance=0.2,
            direction_tolerance=0.5,
        ),
    )
