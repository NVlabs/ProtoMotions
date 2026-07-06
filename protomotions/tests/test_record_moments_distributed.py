# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-rank gloo tests for RunningMeanStd synchronization boundaries.

Regression guard for the resume deadlock in
``wbc_push/hang_evidence_deadlock``: ranks were caught in different parts of
``RunningMeanStd.record_moments`` (all_gather, broadcast, env reset, and DDP
optimization). The root fix is to make ``record_moments`` local-only and sync
normalizers only at explicit rank-uniform training-loop boundaries.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace

from protomotions.agents.utils.normalization import (
    RunningMeanStd,
    sync_running_mean_std_modules,
)


class _GlooFabric:
    """Minimal Fabric shim backed by a real gloo process group.

    Only the surface the explicit sync helper touches: world_size, global_rank,
    and device.
    """

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.device = torch.device("cpu")


def _worker(rank: int, world_size: int, port: int, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Guard: assert we never touch the gloo object-broadcast path.
        called = {"object_broadcast": 0}
        orig_obj = dist.broadcast_object_list

        def _spy(obj_list, *a, **k):
            called["object_broadcast"] += 1
            return orig_obj(obj_list, *a, **k)

        dist.broadcast_object_list = _spy

        rms = RunningMeanStd(None, shape=(1,), device="cpu")
        rms.count.zero_()

        # Rank-skewed execution: rank 0 records twice while rank 1 is still in
        # reset-like work. This used to deadlock because record_moments itself
        # issued collectives. It is now purely local, so skew is safe.
        if rank == 0:
            rms.record_moments(torch.tensor([[1.0], [3.0]]))
            rms.record_moments(torch.tensor([[5.0], [7.0]]))
        else:
            import time

            time.sleep(0.25)
            rms.record_moments(torch.tensor([[11.0], [13.0]]))

        local_before_sync = {
            "mean": float(rms.mean.item()),
            "var": float(rms.var.item()),
            "count": int(rms.count.item()),
        }

        fabric = _GlooFabric()
        sync_running_mean_std_modules([("rms", rms)], fabric)

        ret[rank] = {
            "mean": float(rms.mean.item()),
            "var": float(rms.var.item()),
            "count": int(rms.count.item()),
            "local_before_sync": local_before_sync,
            "object_broadcast": called["object_broadcast"],
        }
    finally:
        dist.destroy_process_group()


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_record_moments_is_local_and_explicit_sync_is_tensor_only():
    world_size = 2
    port = _free_port()
    manager = mp.Manager()
    ret = manager.dict()

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port, ret))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=55)

    for rank, p in enumerate(procs):
        if p.is_alive():
            p.terminate()
            pytest.fail(f"rank {rank} hung in record_moments (deadlock regression)")
        assert p.exitcode == 0, f"rank {rank} exited with {p.exitcode}"

    assert set(ret.keys()) == {0, 1}

    # The local-only updates intentionally diverge before the explicit sync.
    assert ret[0]["local_before_sync"] != ret[1]["local_before_sync"]

    # Both ranks must agree exactly after the explicit rank-uniform sync.
    assert ret[0]["mean"] == pytest.approx(ret[1]["mean"])
    assert ret[0]["var"] == pytest.approx(ret[1]["var"])
    assert ret[0]["count"] == ret[1]["count"]

    # Combined statistics over all six samples.
    all_values = torch.tensor([1.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    assert ret[0]["mean"] == pytest.approx(float(all_values.mean()))
    assert ret[0]["var"] == pytest.approx(float(all_values.var(unbiased=False)))
    assert ret[0]["count"] == 6

    # The fix's core invariant: no object collective on either rank.
    assert ret[0]["object_broadcast"] == 0
    assert ret[1]["object_broadcast"] == 0


if __name__ == "__main__":
    test_record_moments_is_local_and_explicit_sync_is_tensor_only()
    print("PASSED")
