# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-process CPU check: missing results must not change collective ordering."""

from types import SimpleNamespace

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from protomotions.agents.utils.training import aggregate_evaluation_results


def _rank_check(rank, rendezvous):
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        fabric = SimpleNamespace(world_size=2)
        local = ({}, None, 0) if rank == 0 else ({"eval/success_rate": 0.75}, 0.75, 4)
        logs, score = aggregate_evaluation_results(*local, fabric)
        assert logs == {"eval/success_rate": 0.75} and score == 0.75
        assert aggregate_evaluation_results({}, None, 0, fabric) == ({}, None)
        local = ({"task_a": 2.0}, None, 2) if rank == 0 else ({"task_b": 4.0}, 0.5, 6)
        logs, score = aggregate_evaluation_results(*local, fabric)
        assert logs == {"task_a": 2.0, "task_b": 4.0} and score == 0.5

    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_evaluation_with_empty_and_different_task_results(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        _rank_check, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )
