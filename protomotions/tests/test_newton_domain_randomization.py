# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from protomotions.simulator.newton.randomization_utils import (
    move_friction_tables_to_device,
)


def _cpu_friction_dr(num_buckets: int = 4, num_bodies: int = 2) -> dict:
    """Friction DR dict as produced by Simulator._process_friction_domain_randomization,
    whose bucket tables are always sampled on CPU."""
    return {
        "body_indices": [0, 1],
        "static_friction": torch.rand(num_buckets, num_bodies),
        "dynamic_friction": None,
        "restitution": torch.rand(num_buckets, num_bodies),
    }


def test_move_friction_tables_preserves_values_and_none_entries():
    friction_dr = _cpu_friction_dr()

    moved = move_friction_tables_to_device(friction_dr, torch.device("cpu"))

    assert moved["body_indices"] == friction_dr["body_indices"]
    assert moved["dynamic_friction"] is None
    assert torch.equal(moved["static_friction"], friction_dr["static_friction"])
    assert torch.equal(moved["restitution"], friction_dr["restitution"])
    # The input dict must not be mutated: other simulators share it and expect
    # the tables to stay on CPU.
    assert friction_dr["static_friction"].device.type == "cpu"
    assert friction_dr["restitution"].device.type == "cpu"


def test_moved_tables_support_bucket_indexing_on_target_device():
    friction_dr = _cpu_friction_dr(num_buckets=4, num_bodies=2)
    device = torch.device("cpu")

    moved = move_friction_tables_to_device(friction_dr, device)
    bucket_ids = torch.randint(0, 4, (8,), device=device)

    values = moved["static_friction"][bucket_ids, 1]
    assert values.shape == (8,)
    assert values.device == moved["static_friction"].device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bucket_indexing_after_move():
    """Regression test: the Newton simulator indexes the CPU-sampled bucket
    tables with CUDA bucket ids, which raises a device-mismatch RuntimeError
    unless the tables are moved to the simulation device first."""
    friction_dr = _cpu_friction_dr(num_buckets=4, num_bodies=2)
    device = torch.device("cuda")
    bucket_ids = torch.randint(0, 4, (8,), device=device)

    with pytest.raises(RuntimeError):
        friction_dr["static_friction"][bucket_ids, 0]

    moved = move_friction_tables_to_device(friction_dr, device)
    for key in ("static_friction", "restitution"):
        assert moved[key].device.type == "cuda"
        values = moved[key][bucket_ids, 0]
        assert values.device.type == "cuda"
