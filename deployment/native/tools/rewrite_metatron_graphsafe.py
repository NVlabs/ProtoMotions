# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replace the pinned export's two inference-only masked square roots.

ProtoMotions' ``_sqrt_positive_part`` uses boolean indexing to make the
gradient zero at x=0. The resulting NonZero/GatherND/ScatterND is dynamically
shaped, so TensorRT cannot capture the whole engine in a CUDA graph. At
inference, this has the same value as sqrt(max(x, 0)). Keep the downloaded
model intact and write a distinct artifact; qualify it against the original.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(source: Path, destination: Path) -> None:
    model = onnx.load(str(source))
    graph = model.graph
    nodes = list(graph.node)
    producers = {output: node for node in nodes for output in node.output}
    replacements: dict[str, tuple[str, str]] = {}
    for mask_name in ("positive_mask", "positive_mask.3"):
        greater = producers.get(mask_name)
        if greater is None or greater.op_type != "Greater":
            raise ValueError(f"missing pinned sqrt-positive mask: {mask_name}")
        nonzero = [
            node
            for node in nodes
            if node.op_type == "NonZero" and node.input == [mask_name]
        ]
        if len(nonzero) != 1:
            raise ValueError(f"unexpected NonZero pattern for {mask_name}")
        # The final ScatterND is the only consumer of this masked-sqrt result.
        candidates = [
            node
            for node in nodes
            if node.op_type == "ScatterND"
            and node.name in ("ScatterND_3720", "ScatterND_8241")
        ]
        if len(candidates) != 2:
            raise ValueError("pinned masked-sqrt ScatterND nodes changed")
        final = candidates[0 if mask_name == "positive_mask" else 1]
        if final.output[0] not in {"onnx::Gather_896", "onnx::Gather_6583"}:
            raise ValueError("masked-sqrt output changed")
        replacements[final.name] = (greater.input[0], final.output[0])

    if sum(node.op_type == "NonZero" for node in nodes) != 2 or len(replacements) != 2:
        raise ValueError("only the two reviewed masked square roots may be rewritten")
    zero_name = "metatron_graphsafe_zero"
    if zero_name in producers or any(
        init.name == zero_name for init in graph.initializer
    ):
        raise ValueError("graph-safe constant name collision")
    graph.initializer.append(
        numpy_helper.from_array(np.array(0, np.float32), zero_name)
    )
    rewritten = []
    for node in nodes:
        if node.name not in replacements:
            rewritten.append(node)
            continue
        source_name, output_name = replacements[node.name]
        clipped = f"{node.name}_graphsafe_clipped"
        rewritten.append(
            helper.make_node(
                "Clip",
                [source_name, zero_name],
                [clipped],
                name=f"{node.name}_graphsafe_clip",
            )
        )
        rewritten.append(
            helper.make_node(
                "Sqrt", [clipped], [output_name], name=f"{node.name}_graphsafe_sqrt"
            )
        )

    # Remove the obsolete boolean-indexing subgraphs without touching any
    # unrelated output-producing node. Preserve the original topological order.
    required = {output.name for output in graph.output}
    retained = []
    for node in reversed(rewritten):
        if any(output in required for output in node.output):
            retained.append(node)
            required.update(node.input)
    graph.ClearField("node")
    graph.node.extend(reversed(retained))
    if any(node.op_type == "NonZero" for node in graph.node):
        raise ValueError("graph-safe rewrite left a data-dependent NonZero")
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    onnx.save(model, str(partial))
    partial.replace(destination)
    print(f"graph-safe model: {destination} nodes={len(graph.node)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("source and destination must differ")
    rewrite(args.source, args.destination)


if __name__ == "__main__":
    main()
