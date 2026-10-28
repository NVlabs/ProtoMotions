# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-divergence comparison of two recorded traces.

Values come first, call order second. Every key except ``calls`` is a VALUE
key (``compile_path`` included: a changed compile path is a real divergence);
``calls`` is compared only after all values agreed, so a refactor that moves
an internal call without changing a single number is reported as "call order
changed", not as a divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

Trace = Dict[str, Any]
CALLS_KEY = "calls"


def save_trace(trace: Trace, path) -> None:
    torch.save(trace, str(path))


def load_trace(path) -> Trace:
    return torch.load(str(path), weights_only=False)


@dataclass
class ComparisonResult:
    """Outcome of :func:`compare`.

    ``values_identical`` covers every key except ``calls``; ``calls_identical``
    covers the ``calls`` lists; ``identical`` is both. ``first_divergence``
    names the first record whose values differ (index, tag and every differing
    key of that record; ``index``/``tag``/``key``/``keys`` carry the same
    information for programmatic use, ``key`` being the first of ``keys``).
    ``first_call_divergence`` names the first record whose call order differs
    (``call_index``/``call_tag``). ``worst_diff`` is the largest float
    difference seen before stopping (inf for an exact-compared value that
    differed).
    """

    identical: bool
    values_identical: bool
    calls_identical: bool
    first_divergence: Optional[str]
    first_call_divergence: Optional[str]
    worst_diff: float
    num_records: int
    index: Optional[int] = None
    tag: Optional[str] = None
    key: Optional[str] = None
    keys: Tuple[str, ...] = ()
    call_index: Optional[int] = None
    call_tag: Optional[str] = None


def _is_exact_dtype(tensor: torch.Tensor) -> bool:
    return not (tensor.is_floating_point() or tensor.is_complex())


def _compare_tensors(
    x: torch.Tensor, y: torch.Tensor, atol: float
) -> Tuple[bool, float, str]:
    if x.shape != y.shape or x.dtype != y.dtype:
        return (
            False,
            math.inf,
            f"shape/dtype {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}",
        )
    x, y = x.detach().cpu(), y.detach().cpu()
    if torch.equal(x, y):
        return True, 0.0, ""
    if _is_exact_dtype(x):
        return False, math.inf, "values differ (exact dtype)"
    diff = (x.to(torch.float64) - y.to(torch.float64)).abs()
    diff[torch.isnan(x) & torch.isnan(y)] = 0.0
    worst = diff.max().item()
    if math.isnan(worst):
        return False, math.inf, "NaN on one side only"
    return worst <= atol, worst, f"maxdiff {worst:g}"


def _compare_sequences(x, y) -> Tuple[bool, float, str]:
    for position, (a, b) in enumerate(zip(x, y)):
        if a != b:
            return False, math.inf, f"entry {position}: {a!r} vs {b!r}"
    if len(x) != len(y):
        return False, math.inf, f"length {len(x)} vs {len(y)}"
    return True, 0.0, ""


def _compare_values(x: Any, y: Any, atol: float) -> Tuple[bool, float, str]:
    if torch.is_tensor(x) and torch.is_tensor(y):
        return _compare_tensors(x, y, atol)
    if torch.is_tensor(x) or torch.is_tensor(y):
        return False, math.inf, f"type {type(x).__name__} vs {type(y).__name__}"
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return _compare_sequences(x, y)
    if x == y:
        return True, 0.0, ""
    return False, math.inf, f"{x!r} vs {y!r}"


def compare(
    a: Trace,
    b: Trace,
    *,
    atol: float = 0.0,
    ignore_prefixes: Sequence[str] = (),
) -> ComparisonResult:
    """Compare two traces record by record.

    First every value key (everything but ``calls``) across all records,
    stopping at the first record whose values differ; then the ``calls``
    lists, stopping at the first record whose call order differs. Bool/int
    tensors, ``list[str]`` and bool/str values are compared exactly; float
    tensors within ``atol``. A key-set, shape or dtype mismatch is a value
    divergence, as is a different record count. Keys starting with any of
    ``ignore_prefixes`` are skipped (``"calls"`` skips the call-order pass).
    """
    records_a, records_b = a["records"], b["records"]
    num_common = min(len(records_a), len(records_b))
    worst = 0.0

    def ignored(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in ignore_prefixes)

    # --- values ----------------------------------------------------------
    value_divergence: Optional[Dict[str, Any]] = None
    for index in range(num_common):
        record_a, record_b = records_a[index], records_b[index]
        tag = str(record_a.get("tag", "?"))
        keys_a = [k for k in record_a if k != CALLS_KEY and not ignored(k)]
        keys_b = [k for k in record_b if k != CALLS_KEY and not ignored(k)]
        set_b = set(keys_b)
        differing: List[Tuple[str, str]] = []
        for key in keys_a:
            if key not in set_b:
                differing.append((key, "only in a"))
                continue
            ok, diff, detail = _compare_values(record_a[key], record_b[key], atol)
            if math.isfinite(diff):
                worst = max(worst, diff)
            if not ok:
                worst = max(worst, diff)
                differing.append((key, detail))
        set_a = set(keys_a)
        differing.extend((key, "only in b") for key in keys_b if key not in set_a)
        if differing:
            keys = tuple(key for key, _ in differing)
            value_divergence = {
                "index": index,
                "tag": tag,
                "key": keys[0],
                "keys": keys,
                "text": f"record {index} ({tag}) keys [{', '.join(keys)}]: "
                f"{differing[0][1]}",
            }
            break
    if value_divergence is None and len(records_a) != len(records_b):
        longer = records_a if len(records_a) > len(records_b) else records_b
        tag = str(longer[num_common].get("tag", "?"))
        value_divergence = {
            "index": num_common,
            "tag": tag,
            "key": None,
            "keys": (),
            "text": f"record {num_common} ({tag}): record count differs: "
            f"{len(records_a)} vs {len(records_b)}",
        }

    # --- call order --------------------------------------------------------
    call_divergence: Optional[Dict[str, Any]] = None
    if not ignored(CALLS_KEY):
        for index in range(num_common):
            record_a, record_b = records_a[index], records_b[index]
            tag = str(record_a.get("tag", "?"))
            calls_a, calls_b = record_a.get(CALLS_KEY), record_b.get(CALLS_KEY)
            if calls_a is None or calls_b is None:
                if calls_a is None and calls_b is None:
                    continue
                detail = f"calls missing in {'a' if calls_a is None else 'b'}"
            else:
                ok, _, detail = _compare_sequences(calls_a, calls_b)
                if ok:
                    continue
            call_divergence = {
                "index": index,
                "tag": tag,
                "text": f"record {index} ({tag}): {detail}",
            }
            break

    values_identical = value_divergence is None
    calls_identical = call_divergence is None
    return ComparisonResult(
        identical=values_identical and calls_identical,
        values_identical=values_identical,
        calls_identical=calls_identical,
        first_divergence=None if values_identical else value_divergence["text"],
        first_call_divergence=None if calls_identical else call_divergence["text"],
        worst_diff=worst,
        num_records=len(records_a) if values_identical else num_common,
        index=None if values_identical else value_divergence["index"],
        tag=None if values_identical else value_divergence["tag"],
        key=None if values_identical else value_divergence["key"],
        keys=() if values_identical else value_divergence["keys"],
        call_index=None if calls_identical else call_divergence["index"],
        call_tag=None if calls_identical else call_divergence["tag"],
    )


def format_result(result: ComparisonResult, atol: float = 0.0) -> str:
    if result.identical:
        return (
            f"IDENTICAL within atol={atol:g}: {result.num_records} records, "
            f"worst diff {result.worst_diff:g}"
        )
    if result.values_identical:
        return f"VALUES IDENTICAL, CALL ORDER CHANGED {result.first_call_divergence}"
    return f"FIRST DIVERGENCE {result.first_divergence}"
