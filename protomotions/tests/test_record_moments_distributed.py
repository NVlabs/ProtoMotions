# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-rank gloo test for RunningMeanStd.record_moments' distributed buffer sync.

Regression guard for the resume deadlock in
``wbc_push/hang_evidence_run2seed_futex``: the original code broadcast the
combined mean/var/count with ``Fabric.broadcast``, which the Lightning DDP
strategy implements via ``torch.distributed.broadcast_object_list`` — a
gloo/CPU, pickle-based *object* collective. The all_gathers in the same method
run on the NCCL/GPU backend, so record_moments issued a cross-backend
(NCCL then gloo) sequence inside a DDP forward, interleaved with DDP's own NCCL
collectives. The relative completion order of the two backends is timing
dependent across ranks and could wedge the process group on resume.

The fix keeps the whole sync on the same (tensor) backend via
``torch.distributed.broadcast``. This test spins up a real 2-rank gloo process
group, feeds each rank a DIFFERENT shard, and asserts every rank ends with the
SAME combined statistics — i.e. the collectives ran in lockstep and did not
hang. It also asserts ``broadcast_object_list`` is never called.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace

from protomotions.agents.utils.normalization import RunningMeanStd


class _GlooFabric:
    """Minimal Fabric shim backed by a real gloo process group.

    Only the surface RunningMeanStd.record_moments touches: world_size,
    global_rank, and all_gather (broadcast now goes straight through
    torch.distributed inside record_moments).
    """

    def __init__(self):
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

    def all_gather(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        value = value.contiguous()
        out = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(out, value)
        return torch.stack(out)


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

        fabric = _GlooFabric()
        rms = RunningMeanStd(fabric, shape=(1,), device="cpu")
        rms.count.zero_()

        # Each rank contributes a different shard (values 2*rank+1, 2*rank+3).
        base = 2.0 * rank
        shard = torch.tensor([[base + 1.0], [base + 3.0]])
        rms.record_moments(shard)

        ret[rank] = {
            "mean": float(rms.mean.item()),
            "var": float(rms.var.item()),
            "count": int(rms.count.item()),
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


def test_record_moments_2rank_gloo_lockstep_and_no_object_broadcast():
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

    # Both ranks must agree exactly (lockstep collectives, no desync).
    assert ret[0]["mean"] == pytest.approx(ret[1]["mean"])
    assert ret[0]["var"] == pytest.approx(ret[1]["var"])
    assert ret[0]["count"] == ret[1]["count"]

    # Combined statistics over all four samples {1,3, 3,5}.
    all_values = torch.tensor([1.0, 3.0, 3.0, 5.0])
    assert ret[0]["mean"] == pytest.approx(float(all_values.mean()))
    assert ret[0]["var"] == pytest.approx(float(all_values.var(unbiased=False)))
    assert ret[0]["count"] == 4

    # The fix's core invariant: no gloo object-broadcast on either rank.
    assert ret[0]["object_broadcast"] == 0
    assert ret[1]["object_broadcast"] == 0


if __name__ == "__main__":
    test_record_moments_2rank_gloo_lockstep_and_no_object_broadcast()
    print("PASSED")
