# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-name context handoff for generated ProtoMotions policies.

ProtoMotions observation modules address values by semantic TensorDict/context
keys, while an ONNX graph uses sanitized names.  The manifest is the explicit
mapping between the two; the cycle only validates fixed shape/dtype and updates
the preallocated feed map.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


class ProtoMotionsGlobalContext:
    """Name-addressed, fixed-shape policy context.

    ``input_keys`` are semantic context keys and ``input_names`` are the graph
    names in the same order.  They are intentionally allowed to differ because
    dotted context paths are sanitized for ONNX identifiers.  This is an
    explicit artifact mapping, not a positional or compatibility fallback.
    """

    __slots__ = (
        "_keys",
        "_names",
        "_shapes",
        "_dtypes",
        "_indices",
        "_values",
        "_feed",
    )

    def __init__(
        self,
        input_keys: Iterable[str],
        input_names: Iterable[str],
        input_shapes: Iterable[Iterable[int]],
        input_dtypes: Iterable[str],
    ) -> None:
        self._keys = tuple(str(key) for key in input_keys)
        names = tuple(str(name) for name in input_names)
        if len(self._keys) != len(names):
            raise ValueError("policy context input key/name lengths do not match")
        self._names = names
        self._shapes = tuple(tuple(int(dim) for dim in shape) for shape in input_shapes)
        self._dtypes = tuple(np.dtype(dtype) for dtype in input_dtypes)
        if not (len(self._keys) == len(self._shapes) == len(self._dtypes)):
            raise ValueError("policy context metadata lengths do not match")
        if len(set(self._keys)) != len(self._keys):
            raise ValueError("policy context input keys must be unique")
        if len(set(self._names)) != len(self._names):
            raise ValueError("policy context input names must be unique")
        self._indices = {key: index for index, key in enumerate(self._keys)}
        self._values: list[np.ndarray | None] = [None] * len(self._keys)
        # ORT accepts a mapping. Keep the mapping object stable across cycles;
        # only its fixed-name array views are replaced.
        self._feed: dict[str, np.ndarray] = {}

    @property
    def keys(self) -> tuple[str, ...]:
        """The semantic context keys in model order."""

        return self._keys

    @property
    def names(self) -> tuple[str, ...]:
        """The ONNX graph input names in model order."""

        return self._names

    def set(self, key: str, value: np.ndarray) -> None:
        """Publish one named value, rejecting silent reshapes/conversions."""

        try:
            index = self._indices[key]
        except KeyError as exc:
            raise KeyError(f"policy does not declare context input {key!r}") from exc
        array = np.asarray(value)
        if array.shape != self._shapes[index]:
            raise ValueError(
                f"context input {key!r} has shape {array.shape}; "
                f"expected {self._shapes[index]}"
            )
        if array.dtype != self._dtypes[index]:
            raise ValueError(
                f"context input {key!r} has dtype {array.dtype}; "
                f"expected {self._dtypes[index]}"
            )
        self._values[index] = array

    def set_many(self, values: Mapping[str, np.ndarray]) -> None:
        """Publish a complete named context without accepting unknown fields."""

        unknown = set(values) - set(self._keys)
        missing = set(self._keys) - set(values)
        if unknown or missing:
            raise ValueError(
                f"policy context fields differ from manifest; unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}"
            )
        for key in self._keys:
            self.set(key, values[key])

    def feed(self) -> dict[str, np.ndarray]:
        """Return the fixed-name ORT feed after all declared fields are set."""

        for index, key in enumerate(self._keys):
            value = self._values[index]
            if value is None:
                raise RuntimeError(f"policy context input {key!r} was not published")
            self._feed[self._names[index]] = value.reshape((1, *self._shapes[index]))
        return self._feed
