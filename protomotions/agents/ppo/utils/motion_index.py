# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-only, stable row allocation for per-motion Gaussian exploration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from protomotions.components.motion_lib import discover_motion_shards
from protomotions.agents.utils.distributed import raise_if_any_rank_failed


def names_digest(names):
    return hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()


@dataclass(frozen=True)
class MotionShard:
    path: str
    offset: int
    count: int
    names_digest: str


@dataclass
class MotionIndex:
    shards: list[MotionShard]

    @property
    def num_motions(self):
        return sum(shard.count for shard in self.shards)

    def state_dict(self):
        return [asdict(shard) for shard in self.shards]

    def offset_for(self, motion_lib):
        path = str(Path(motion_lib.motion_file).resolve())
        shard = next(shard for shard in self.shards if shard.path == path)
        if (
            len(motion_lib.motion_files) != shard.count
            or names_digest(motion_lib.motion_files) != shard.names_digest
        ):
            raise ValueError(
                f"Motion ordering changed in {path}; rebuild exploration state"
            )
        return shard.offset

    @classmethod
    def from_sources(cls, sources):
        """Read each packaged shard's metadata once; mmap avoids loading poses."""
        shards = []
        offset = 0
        seen = set()
        for pattern, indices, loaded_path, loaded_names in sources:
            if "slurmrank" in pattern:
                paths = discover_motion_shards(pattern, indices).values()
            else:
                paths = [loaded_path]
            for path in paths:
                path = str(Path(path).resolve())
                if path in seen:
                    continue
                seen.add(path)
                if path == str(Path(loaded_path).resolve()):
                    names = loaded_names
                else:
                    names = torch.load(
                        path, map_location="cpu", weights_only=False, mmap=True
                    )["motion_files"]
                shards.append(
                    MotionShard(path, offset, len(names), names_digest(names))
                )
                offset += len(names)
        if not offset:
            raise ValueError("Per-motion std requires a nonempty motion library")
        return cls(shards)

    @classmethod
    def build(cls, motion_lib):
        source = (
            motion_lib.config.motion_file,
            motion_lib.config.motion_file_shard_indices,
            motion_lib.motion_file,
            motion_lib.motion_files,
        )
        sources = [source]
        if dist.is_initialized():
            sources = [None] * dist.get_world_size()
            dist.all_gather_object(sources, source)
        # All ranks construct the same ordered union, including different task
        # groups. Only rank zero reads metadata for shards not currently loaded.
        result = [None]
        error = None
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                result[0] = cls.from_sources(sources)
        except Exception as caught:
            error = caught
        # A missing/broken shard must fail all ranks, not strand them in broadcast.
        raise_if_any_rank_failed(error, "Motion exploration index", motion_lib.device)
        if dist.is_initialized():
            dist.broadcast_object_list(result)
        return result[0]
