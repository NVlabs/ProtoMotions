# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the FK kernel a deployable policy binds as its own small ONNX graph.

A deployable recipe computes the robot's current body poses inside the policy
graph: a goal kernel takes the recipe's ``BakedAnchorFK`` as ``fk=`` and reads
the joint angles and the anchor pose from the sensor namespace
(``compute_fk_target_poses_only``). The policy graph keeps its four control
outputs. A host that needs the body poses outside the policy call (the native
Metatron runtime runs FK on the current measurement before every call) gets
them from ``fk.onnx``, exported here from the SAME kernel the policy traces,
rebuilt from the same config bundle beside the checkpoint, so the two agree by
construction. There is no second FK implementation: this module traces
``BakedAnchorFK.world_body_poses`` and nothing else.

The graph's inputs carry the names the policy graph gives the same context
paths (``deployment.export_to_onnx.host_input_names``: a Metatron export keeps
the sensor namespace, ``noisy.dof_pos`` -> ``noisy_dof_pos``; a plain-actor
tracker export first renames sensor paths to its ``current.*`` wire protocol),
so a host feeds one tensor under one name to both graphs. They are in the
kernel's argument order: joint angles ``[1, num_dofs]``, anchor position
``[1, 3]``, anchor rotation ``[1, 4]`` (xyzw). The batch axis is fixed to 1: the
FK plan serves one robot. The outputs are ``body_pos`` ``[1, num_bodies, 3]``
and ``body_rot`` ``[1, num_bodies, 4]`` (xyzw) in the robot's body order. The
sidecar ``fk.yaml`` beside the graph names the bodies, the joints, the anchor,
the inputs and the checkpoint; the native plan builder reads it and, given the
policy manifest, checks the body order, the joint order, the anchor and that
the FK inputs are inputs of the policy, so an FK graph is never paired with a
policy it does not belong to. The graph reaches its final name only after it
passed the static-op check and the ONNX Runtime comparison, so a refused export
leaves nothing on disk and never replaces an earlier pair.

Usage::

    python deployment/export_fk_to_onnx.py --checkpoint <dir>/inference_last.ckpt
    # writes <dir>/compiled_models/fk.onnx and fk.yaml

A bundle whose actor observations bind no FK kernel is refused: there is
nothing to export, since such a policy computes no body poses a host could share.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

# The kernel's argument order, which is also the record order of the native
# fixture and the binding order of the C++ FK engine.
FK_INPUT_ARGS = ("dof_pos", "anchor_pos", "anchor_rot")
FK_OUTPUTS = ("body_pos", "body_rot")
# A static TensorRT plan cannot hold data-dependent shapes or control flow.
DATA_DEPENDENT_OPS = frozenset({"NonZero", "If", "Loop", "Scan"})
SCHEMA = 1


def fk_binding(components):
    """The FK kernel the actor observations bind, and the paths it reads.

    A component qualifies when its static ``fk`` is a ``BakedAnchorFK`` and its
    bindings name ``dof_pos``, ``anchor_pos`` and ``anchor_rot``. Returns
    ``(fk, paths)`` with the paths in :data:`FK_INPUT_ARGS` order, or None when
    no component qualifies. Two qualifying components must agree in what the
    bundle saves of their kernels (class and ``export_init`` constants: anchor
    and body subset) and in their paths, since one FK graph is exported per
    policy; the bundle rebuilds a kernel per component, so identity is not the
    test.
    """
    from protomotions.utils.baked_fk import BakedAnchorFK

    found = None
    for name, component in components.items():
        fk = component.static_params.get("fk")
        bindings = component.get_bindings_dict()
        if not isinstance(fk, BakedAnchorFK):
            continue
        if not set(FK_INPUT_ARGS) <= bindings.keys():
            continue
        paths = tuple(bindings[arg] for arg in FK_INPUT_ARGS)
        signature = (type(fk), fk.export_init(), paths)
        if found is not None and found[1] != signature:
            raise ValueError(
                f"actor observations {found[2]!r} and {name!r} bind different FK "
                "kernels or state paths; one FK graph is exported per policy"
            )
        found = (fk, signature, name)
    return None if found is None else (found[0], found[1][2])


@dataclass(frozen=True)
class FkSidecar:
    """What ``fk.yaml`` records about an exported FK graph.

    ``inputs`` are ``(onnx_name, context_path, shape)`` in the kernel's argument
    order; ``outputs`` are ``(onnx_name, shape)`` in :data:`FK_OUTPUTS` order.
    Shapes carry the fixed batch of 1. ``checkpoint`` is the file name of the
    checkpoint whose bundle rebuilt the kernel.
    """

    body_names: Tuple[str, ...]
    joint_names: Tuple[str, ...]
    anchor_body_name: str
    anchor_body_index: int
    inputs: Tuple[Tuple[str, str, Tuple[int, ...]], ...]
    outputs: Tuple[Tuple[str, Tuple[int, ...]], ...]
    checkpoint: str

    def __post_init__(self):
        bodies, dofs = len(self.body_names), len(self.joint_names)
        if not 0 <= self.anchor_body_index < bodies:
            raise ValueError(
                f"anchor_body_index {self.anchor_body_index} outside {bodies} bodies"
            )
        if self.body_names[self.anchor_body_index] != self.anchor_body_name:
            raise ValueError(
                f"anchor_body_name {self.anchor_body_name!r} is not body "
                f"{self.anchor_body_index} ({self.body_names[self.anchor_body_index]!r})"
            )
        tails = tuple(path.rsplit(".", 1)[-1] for _, path, _ in self.inputs)
        if tails != FK_INPUT_ARGS:
            raise ValueError(
                f"FK inputs must read {list(FK_INPUT_ARGS)} in that order; got "
                f"{[path for _, path, _ in self.inputs]}"
            )
        expected_in = ((1, dofs), (1, 3), (1, 4))
        for (name, _path, shape), expected in zip(self.inputs, expected_in):
            if tuple(shape) != expected:
                raise ValueError(f"FK input {name!r} has shape {shape}, not {expected}")
        names = tuple(name for name, _ in self.outputs)
        if names != FK_OUTPUTS:
            raise ValueError(
                f"FK outputs must be {list(FK_OUTPUTS)}; got {list(names)}"
            )
        expected_out = ((1, bodies, 3), (1, bodies, 4))
        for (name, shape), expected in zip(self.outputs, expected_out):
            if tuple(shape) != expected:
                raise ValueError(
                    f"FK output {name!r} has shape {shape}, not {expected}"
                )

    @property
    def input_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _, _ in self.inputs)

    @property
    def input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {name: tuple(shape) for name, _, shape in self.inputs}

    @property
    def output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {name: tuple(shape) for name, shape in self.outputs}

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "type": "fk",
            "checkpoint": self.checkpoint,
            "anchor_body_name": self.anchor_body_name,
            "anchor_body_index": self.anchor_body_index,
            "body_names": list(self.body_names),
            "joint_names": list(self.joint_names),
            "inputs": [
                {"name": name, "source": path, "shape": list(shape)}
                for name, path, shape in self.inputs
            ],
            "outputs": [
                {"name": name, "shape": list(shape)} for name, shape in self.outputs
            ],
        }

    @classmethod
    def from_dict(cls, content: dict) -> "FkSidecar":
        if content.get("type") != "fk" or content.get("schema") != SCHEMA:
            raise ValueError(
                f"not an FK sidecar (type {content.get('type')!r}, schema "
                f"{content.get('schema')!r})"
            )
        return cls(
            body_names=tuple(content["body_names"]),
            joint_names=tuple(content["joint_names"]),
            anchor_body_name=content["anchor_body_name"],
            anchor_body_index=int(content["anchor_body_index"]),
            inputs=tuple(
                (entry["name"], entry["source"], tuple(entry["shape"]))
                for entry in content["inputs"]
            ),
            outputs=tuple(
                (entry["name"], tuple(entry["shape"])) for entry in content["outputs"]
            ),
            checkpoint=content["checkpoint"],
        )

    def to_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path) -> "FkSidecar":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def check_manifest(self, manifest_path) -> None:
        """Refuse pairing this FK graph with a policy it does not belong to.

        ``manifest_path`` is the native policy manifest: its ``robot_contract``
        names the policy's bodies, joints and anchor, and ``input_names`` the
        policy graph's inputs, which the FK inputs must be among (the host feeds
        one tensor under one name to both graphs).
        """
        manifest = json.loads(Path(manifest_path).read_text())
        robot = manifest["robot_contract"]
        if list(self.body_names) != list(robot["body_names"]):
            raise ValueError("the FK graph's body order differs from the manifest")
        if list(self.joint_names) != list(robot["joint_names"]):
            raise ValueError("the FK graph's joint order differs from the manifest")
        if self.anchor_body_index != robot["anchor_body_index"]:
            raise ValueError("the FK graph's anchor body differs from the manifest")
        policy_inputs = set(manifest["input_names"])
        missing = [name for name in self.input_names if name not in policy_inputs]
        if missing:
            raise ValueError(
                f"the FK graph's inputs {missing} are not inputs of the policy "
                "manifest; the host feeds one tensor under one name to both graphs"
            )


def _check_static(model, sidecar: FkSidecar) -> None:
    """The graph must be what a static TensorRT plan can hold, with the sidecar's I/O."""
    import onnx

    onnx.checker.check_model(model)
    bad = sorted({node.op_type for node in model.graph.node} & DATA_DEPENDENT_OPS)
    if bad:
        raise ValueError(
            f"the FK graph contains {bad}; a static TensorRT plan cannot hold "
            "data-dependent ops"
        )

    def shape_of(value):
        return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)

    graph_inputs = {value.name: shape_of(value) for value in model.graph.input}
    graph_outputs = {value.name: shape_of(value) for value in model.graph.output}
    if graph_inputs != sidecar.input_shapes or graph_outputs != sidecar.output_shapes:
        raise ValueError(
            f"FK graph I/O {graph_inputs} -> {graph_outputs} differs from the "
            f"sidecar {sidecar.input_shapes} -> {sidecar.output_shapes}"
        )


def _validate(onnx_path, fk, kinematic_info, sidecar: FkSidecar, samples: int) -> float:
    """ONNX Runtime against the kernel on random joints and anchor poses.

    Joints are drawn within the robot's limits, the anchor rotation is a random
    unit quaternion. Returns the largest absolute difference seen.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    lower = kinematic_info.dof_limits_lower.float()
    upper = kinematic_info.dof_limits_upper.float()
    generator = torch.Generator().manual_seed(59)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(samples):
        dof_pos = lower + (upper - lower) * torch.rand(
            1, len(lower), generator=generator
        )
        anchor_pos = torch.rand(1, 3, generator=generator) * 2 - 1
        anchor_rot = torch.randn(1, 4, generator=generator)
        anchor_rot = anchor_rot / anchor_rot.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            expected = fk.world_body_poses(dof_pos, anchor_pos, anchor_rot)
        feed = dict(
            zip(
                sidecar.input_names,
                (dof_pos.numpy(), anchor_pos.numpy(), anchor_rot.numpy()),
            )
        )
        got = session.run(list(FK_OUTPUTS), feed)
        for name, value, truth in zip(FK_OUTPUTS, got, expected):
            diff = float(np.abs(value - truth.numpy()).max())
            worst = max(worst, diff)
            if diff > 1e-5:
                raise ValueError(
                    f"ONNX Runtime {name} differs from the kernel by {diff:.3e} "
                    "(tolerance 1e-5)"
                )
    return worst


def write_fk_graph(
    fk,
    kinematic_info,
    sidecar: FkSidecar,
    output_path,
    validate: bool = True,
    samples: int = 8,
) -> Path:
    """Trace ``fk.world_body_poses`` to ``<output_path>/fk.onnx`` with ``fk.yaml``.

    The graph is traced to ``fk.onnx.partial``, checked (static ops, the
    sidecar's I/O and, with ``validate``, ONNX Runtime against the kernel on
    joints within ``kinematic_info``'s limits) and only then renamed to
    ``fk.onnx`` and joined by the sidecar; a refusal removes the partial and
    leaves any earlier pair untouched. Returns the path of the ONNX file.
    """
    import onnx
    import torch

    class WorldPoses(torch.nn.Module):
        def forward(self, dof_pos, anchor_pos, anchor_rot):
            return fk.world_body_poses(dof_pos, anchor_pos, anchor_rot)

    sample_inputs = (
        torch.zeros(1, len(sidecar.joint_names)),
        torch.zeros(1, 3),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / "fk.onnx"
    partial = output_path / "fk.onnx.partial"
    log.info(f"Exporting the FK kernel to {onnx_path}")
    try:
        torch.onnx.export(
            WorldPoses().eval(),
            sample_inputs,
            str(partial),
            input_names=list(sidecar.input_names),
            output_names=list(FK_OUTPUTS),
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        model = onnx.load(str(partial))
        _check_static(model, sidecar)
        if validate:
            worst = _validate(partial, fk, kinematic_info, sidecar, samples)
            log.info(f"ONNX Runtime matches the kernel: max abs diff {worst:.3e}")
        partial.replace(onnx_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    sidecar.to_yaml(onnx_path.with_suffix(".yaml"))
    log.info(
        f"FK graph ready {onnx_path} nodes={len(model.graph.node)} "
        f"inputs={list(sidecar.input_names)} outputs={list(FK_OUTPUTS)} "
        f"anchor={sidecar.anchor_body_name}"
    )
    return onnx_path


def export_fk(
    checkpoint: str,
    output_dir: Optional[str] = None,
    validate: bool = True,
    samples: int = 8,
) -> Path:
    """Write ``fk.onnx`` and ``fk.yaml`` for the policy saved at ``checkpoint``.

    The config bundle beside the checkpoint rebuilds the actor's observation
    components exactly as the policy exporter does; the FK kernel they bind is
    traced on its own (:func:`write_fk_graph`). Returns the path of the ONNX file.
    """
    from deployment.export_to_onnx import (
        deployable_observation_components,
        host_input_names,
    )
    from protomotions.utils.experiment_configs import (
        load_experiment_configs_from_checkpoint,
    )

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        # The sidecar names this file as the kernel's provenance, so it must be
        # the real checkpoint, not a bundle directory with a guessed file name.
        raise FileNotFoundError(f"checkpoint {checkpoint_path} does not exist")
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else checkpoint_path.parent / "compiled_models"
    )
    resolved = load_experiment_configs_from_checkpoint(checkpoint_path)
    components = deployable_observation_components(resolved)
    binding = fk_binding(components)
    if binding is None:
        raise ValueError(
            f"no actor observation of {checkpoint_path.name} binds an FK kernel "
            "(compute_fk_target_poses_only with fk=BakedAnchorFK reading dof_pos, "
            "anchor_pos and anchor_rot); nothing to export"
        )
    fk, paths = binding
    names = host_input_names(resolved, paths)
    kinematic_info = resolved.robot.kinematic_info
    body_names = tuple(kinematic_info.body_names)
    joint_names = tuple(kinematic_info.dof_names)
    shapes = ((1, len(joint_names)), (1, 3), (1, 4))
    sidecar = FkSidecar(
        body_names=body_names,
        joint_names=joint_names,
        anchor_body_name=body_names[fk.anchor_idx],
        anchor_body_index=int(fk.anchor_idx),
        inputs=tuple(zip(names, paths, shapes)),
        outputs=(
            (FK_OUTPUTS[0], (1, len(body_names), 3)),
            (FK_OUTPUTS[1], (1, len(body_names), 4)),
        ),
        checkpoint=checkpoint_path.name,
    )
    log.info(f"FK kernel of {checkpoint_path.name}: inputs {list(names)}")
    return write_fk_graph(fk, kinematic_info, sidecar, output_path, validate, samples)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="the policy checkpoint; its config bundle rebuilds the FK kernel",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="directory for fk.onnx and fk.yaml (default: <checkpoint_dir>/compiled_models)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the ONNX Runtime check against the kernel",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    export_fk(args.checkpoint, args.output, validate=not args.no_validate)


if __name__ == "__main__":
    main()
