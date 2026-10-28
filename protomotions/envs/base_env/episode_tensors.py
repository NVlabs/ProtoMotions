# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-env state registry the control components, the episode buffers and
the contact memory allocate through.

An owner's episode state (a control component's, the episode buffers', the
contact memory's) is a set of tensors with the env count as first dimension.
Allocating them through :class:`EpisodeTensors` gives the three things every
owner on the env's walks needs without writing them per owner:
row reset (``reset_rows`` puts the reset rows back to each slot's reset
value), snapshot (``save_state`` clones every slot) and restore
(``load_state`` copies a snapshot back in place, refusing a changed shape or
key set). Four kinds of entry:

- ``add(name, shape, dtype, reset_value)``: a per-env slot
  ``[num_envs, *shape]``, row-reset, snapshotted.
- ``shared(name, tensor)``: a tensor without an env row (a ring cursor),
  snapshotted, never row-reset.
- ``constant(name, tensor)``: neither; registered so the control's attribute
  guard knows it is deliberate (a step table, body ranks).
- ``child(name)``: a nested registry for a sub-object (a target stream, a
  path generator), walked recursively.

Names are unique across the four kinds. ``owns`` answers the control's
attribute guard; ``tensors`` lists every snapshotted tensor for the
completeness test.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple

import torch
from torch import Tensor


class EpisodeTensors:
    """Per-env slots, shared tensors, constants and child registries."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        self._slots: Dict[str, Tensor] = {}
        self._reset_values: Dict[str, Tensor] = {}
        self._shared: Dict[str, Tensor] = {}
        self._constants: Dict[str, Tensor] = {}
        self._children: Dict[str, "EpisodeTensors"] = {}

    def _refuse_taken(self, name: str) -> None:
        if (
            name in self._slots
            or name in self._shared
            or name in self._constants
            or name in self._children
        ):
            raise ValueError(f"state slot '{name}' already exists")

    def add(
        self, name: str, shape: Tuple[int, ...], dtype: torch.dtype, reset_value: Any
    ) -> Tensor:
        """Allocate ``[num_envs, *shape]`` filled with ``reset_value`` (a scalar,
        or a tensor of ``shape``), which ``reset_rows`` writes back per row."""
        self._refuse_taken(name)
        value = torch.as_tensor(reset_value, dtype=dtype, device=self.device)
        if value.ndim == 0:
            tensor = torch.full(
                (self.num_envs, *shape), reset_value, dtype=dtype, device=self.device
            )
        else:
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"state slot '{name}': reset value of shape {tuple(value.shape)} "
                    f"does not match the slot shape {tuple(shape)}"
                )
            tensor = value.unsqueeze(0).expand(self.num_envs, *shape).clone()
        self._slots[name] = tensor
        self._reset_values[name] = value
        return tensor

    def shared(self, name: str, tensor: Tensor) -> Tensor:
        """Register a tensor without an env row: snapshotted, never row-reset."""
        self._refuse_taken(name)
        self._shared[name] = tensor
        return tensor

    def constant(self, name: str, tensor: Tensor) -> Tensor:
        """Register a constant (neither reset nor snapshotted) so the attribute
        guard accepts it; a constant may be re-registered under its name."""
        if name not in self._constants:
            self._refuse_taken(name)
        self._constants[name] = tensor
        return tensor

    def child(self, name: str) -> "EpisodeTensors":
        """A nested registry, walked by reset, snapshot and restore."""
        self._refuse_taken(name)
        registry = EpisodeTensors(self.num_envs, self.device)
        self._children[name] = registry
        return registry

    def reset_rows(self, env_ids: Tensor) -> None:
        """Put the rows of ``env_ids`` back to every slot's reset value."""
        for name, tensor in self._slots.items():
            tensor[env_ids] = self._reset_values[name]
        for registry in self._children.values():
            registry.reset_rows(env_ids)

    def save_state(self) -> Dict[str, Any]:
        """Owned clones of every slot and shared tensor; children nested."""
        snapshot: Dict[str, Any] = {name: t.clone() for name, t in self._slots.items()}
        snapshot.update({name: t.clone() for name, t in self._shared.items()})
        snapshot.update({name: r.save_state() for name, r in self._children.items()})
        return snapshot

    def load_state(self, snapshot: Dict[str, Any]) -> None:
        """Copy a ``save_state`` result back in place."""
        expected = set(self._slots) | set(self._shared) | set(self._children)
        if set(snapshot) != expected:
            raise ValueError(
                f"state snapshot keys {sorted(snapshot)} do not match the registry's "
                f"{sorted(expected)}"
            )
        for name, tensor in (*self._slots.items(), *self._shared.items()):
            saved = snapshot[name]
            if tuple(saved.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"state slot '{name}' shape changed: saved {tuple(saved.shape)} vs "
                    f"live {tuple(tensor.shape)}"
                )
            tensor.copy_(saved)
        for name, registry in self._children.items():
            registry.load_state(snapshot[name])

    def owns(self, tensor: Tensor) -> bool:
        """Whether ``tensor`` is one registered here (any kind, any depth)."""
        registered = (
            *self._slots.values(),
            *self._shared.values(),
            *self._constants.values(),
        )
        if any(tensor is candidate for candidate in registered):
            return True
        return any(registry.owns(tensor) for registry in self._children.values())

    def tensors(self) -> Iterator[Tuple[str, Tensor]]:
        """Every snapshotted tensor as ``(path, tensor)``, children prefixed."""
        yield from self._slots.items()
        yield from self._shared.items()
        for name, registry in self._children.items():
            for path, tensor in registry.tensors():
                yield f"{name}.{path}", tensor


__all__ = ["EpisodeTensors"]
