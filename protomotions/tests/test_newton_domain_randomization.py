# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from protomotions.simulator.base_simulator.utils import (
    get_friction_bucket_count,
    get_friction_table,
)
from protomotions.simulator.newton.randomization_utils import (
    move_friction_tables_to_device,
)


def _friction_dr(
    num_buckets: int = 4,
    num_bodies: int = 2,
    *,
    include_static: bool = True,
    include_dynamic: bool = True,
    include_restitution: bool = True,
) -> dict:
    return {
        "body_indices": list(range(num_bodies)),
        "static_friction": (
            torch.rand(num_buckets, num_bodies) if include_static else None
        ),
        "dynamic_friction": (
            torch.rand(num_buckets, num_bodies) if include_dynamic else None
        ),
        "restitution": (
            torch.rand(num_buckets, num_bodies) if include_restitution else None
        ),
    }


def test_move_friction_tables_preserves_values_and_does_not_mutate_input():
    friction_dr = _friction_dr()

    moved = move_friction_tables_to_device(friction_dr, torch.device("cpu"))

    assert moved is not friction_dr
    assert moved["body_indices"] == friction_dr["body_indices"]
    for key in ("static_friction", "dynamic_friction", "restitution"):
        assert torch.equal(moved[key], friction_dr[key])
        assert friction_dr[key].device.type == "cpu"


def test_moved_tables_support_bucket_indexing_on_target_device():
    friction_dr = _friction_dr()
    device = torch.device("cpu")

    moved = move_friction_tables_to_device(friction_dr, device)
    bucket_ids = torch.randint(0, 4, (8,), device=device)

    for key in ("static_friction", "dynamic_friction", "restitution"):
        values = moved[key][bucket_ids, 1]
        assert values.shape == (8,)
        assert values.device == moved[key].device


def test_bucket_count_uses_any_configured_table():
    assert get_friction_bucket_count(_friction_dr(num_buckets=4)) == 4
    assert (
        get_friction_bucket_count(
            _friction_dr(
                num_buckets=3,
                include_static=False,
                include_dynamic=False,
                include_restitution=True,
            )
        )
        == 3
    )
    assert (
        get_friction_bucket_count(
            _friction_dr(
                num_buckets=5,
                include_static=False,
                include_dynamic=True,
                include_restitution=False,
            )
        )
        == 5
    )
    assert (
        get_friction_bucket_count(
            _friction_dr(
                include_static=False,
                include_dynamic=False,
                include_restitution=False,
            )
        )
        == 0
    )


def test_single_friction_table_prefers_static_then_dynamic():
    friction_dr = _friction_dr(include_restitution=False)
    assert torch.equal(
        get_friction_table(friction_dr), friction_dr["static_friction"]
    )

    friction_dr = _friction_dr(include_static=False, include_restitution=False)
    assert torch.equal(
        get_friction_table(friction_dr), friction_dr["dynamic_friction"]
    )
    assert (
        get_friction_table(_friction_dr(include_static=False, include_dynamic=False))
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_bucket_indexing_after_move():
    """Regression test for CPU tables indexed by CUDA bucket ids."""
    friction_dr = _friction_dr()
    device = torch.device("cuda")
    bucket_ids = torch.randint(0, 4, (8,), device=device)

    with pytest.raises(RuntimeError):
        friction_dr["static_friction"][bucket_ids, 0]

    moved = move_friction_tables_to_device(friction_dr, device)
    for key in ("static_friction", "dynamic_friction", "restitution"):
        assert moved[key].device.type == "cuda"
        values = moved[key][bucket_ids, 0]
        assert values.device.type == "cuda"
