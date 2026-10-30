# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a pipeline whose declared histories are LEAPP feedback state.

``export_pipeline_bundle`` keeps histories host-owned: every runtime replays
them with ``InputHistory``. This variant moves that update into the graph, so
every LEAPP runtime (ProtoMotions hosts, Isaac ROS, Isaac Sim) runs the same
one. The host no longer supplies the history inputs; it reports what was
applied on the previous tick instead:

* ``applied_<output>`` for each output a history records (e.g. the raw
  ``actions``), after any safety/controller edits;
* ``command_accepted``: false if that command was dropped. The history then
  does not advance, as when ``InputHistory.commit`` is not called.

Measurement histories record the current measurement inputs, which stay host
inputs. The first call after a reset (LEAPP initial values) repeats the current
measurements and uses each action history's reset fill, as
``InputHistory.reset``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch

from deployment.tracker_inputs import resolve_fill

if TYPE_CHECKING:
    from deployment.contract import DeploymentContract

NODE_NAME = "policy"
COMMAND_ACCEPTED = "command_accepted"
OUTPUT_NAMES = ("actions", "joint_pos_targets", "stiffness_targets", "damping_targets")


def _state_name(prefix: str, key: str) -> str:
    return f"{prefix}_" + re.sub(r"[^0-9A-Za-z_]", "_", key)


def applied_input_name(output_key: str) -> str:
    return f"applied_{output_key}"


def require_torchscript_disabled() -> None:
    """Fail unless TorchScript is off, as LEAPP tracing needs.

    LEAPP cannot see through ``@torch.jit.script`` helpers (e.g. the rotation
    utilities used by forward kinematics): their results are recorded as
    constants from the first traced tick, so the bundle is wrong afterwards.
    """
    if torch.jit._state._enabled:
        raise RuntimeError(
            "LEAPP tracing needs TorchScript disabled: set PYTORCH_JIT=0 "
            "before Python imports torch"
        )


@dataclass(frozen=True)
class _Buffer:
    """One history deque of InputHistory: newest first, ``depth`` entries."""

    key: str
    depth: int
    feature_shape: tuple[int, ...]
    reset_fill: np.ndarray | None  # None: repeat the current measurement

    @property
    def is_action(self) -> bool:
        return self.reset_fill is not None


class GraphHistory:
    """The contract's ``InputHistory`` update, written in traceable torch."""

    def __init__(self, contract: DeploymentContract) -> None:
        self.history_specs = [
            spec for spec in contract.inputs if spec.history or spec.observation_history
        ]
        if not self.history_specs:
            raise ValueError("the contract declares no history inputs")
        self.host_specs = [
            spec
            for spec in contract.inputs
            if not (spec.history or spec.observation_history)
        ]
        # A measurement history names the reading it records by its source
        # path (``noisy.dof_pos``). The tracker export renames current sensor
        # inputs to ``current.<leaf>`` (``tracker_input_keys``) and keeps the
        # leaf, so a host input's source path is its namespace plus that leaf.
        host_key_of = {spec.key: spec.key for spec in self.host_specs}
        for spec in self.host_specs:
            if "." in spec.key:
                leaf = spec.key.split(".", 1)[1]
                host_key_of.setdefault(f"{spec.source_namespace}.{leaf}", spec.key)
        buffers: dict[str, _Buffer] = {}
        self._buffer_keys: dict[str, str] = {}
        for spec in self.history_specs:
            if spec.history is not None:
                key = spec.history.output_key
                steps = spec.history.steps_back
                feature = (spec.shape[-1],)
                fill = np.asarray(
                    resolve_fill(spec.history.reset_fill, spec.shape[-1]), np.float32
                )
            else:
                current_key = spec.observation_history.current_key
                key = host_key_of.get(current_key)
                if key is None:
                    raise ValueError(
                        f"history input {spec.key!r} records {current_key!r} and "
                        f"needs {current_key} as a graph input; the host inputs "
                        f"are {sorted(host.key for host in self.host_specs)}"
                    )
                steps = spec.observation_history.steps_back
                feature = tuple(spec.shape[2:])
                fill = None
            previous = buffers.get(key)
            depth = max(max(steps), previous.depth if previous else 0)
            buffers[key] = _Buffer(key, depth, feature, fill)
            self._buffer_keys[spec.key] = key
        self.buffers = list(buffers.values())
        self.batch = int(contract.inputs[0].shape[0])

    @property
    def applied_keys(self) -> list[str]:
        return [buffer.key for buffer in self.buffers if buffer.is_action]

    def initial_state(self) -> dict[str, torch.Tensor]:
        """State right after a reset: the next call starts a new episode."""
        state = {"history_first": torch.ones(self.batch, 1)}
        for buffer in self.buffers:
            shape = (self.batch, buffer.depth, *buffer.feature_shape)
            state[_state_name("history", buffer.key)] = torch.zeros(shape)
            if not buffer.is_action:
                state[_state_name("previous", buffer.key)] = torch.zeros(
                    (self.batch, *buffer.feature_shape)
                )
        return state

    def advance(
        self,
        state: Mapping[str, torch.Tensor],
        current: Mapping[str, torch.Tensor],
        applied: Mapping[str, torch.Tensor],
        accepted: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Commit the previous tick, then return history inputs and next state.

        Mirrors ``InputHistory``: ``commit(previous measurements, applied)``
        happens only if the previous command was accepted and this is not the
        first tick after a reset.
        """
        first = state["history_first"] > 0.5
        push = accepted.reshape(self.batch, 1).to(torch.bool) & ~first
        values, next_state = (
            {},
            {"history_first": torch.zeros_like(state["history_first"])},
        )
        for buffer in self.buffers:
            name = _state_name("history", buffer.key)
            history = state[name]
            extra = (1,) * (history.dim() - 2)
            if buffer.is_action:
                newest = applied[buffer.key]
                reset = (
                    torch.as_tensor(buffer.reset_fill).to(history).expand_as(history)
                )
            else:
                newest = state[_state_name("previous", buffer.key)]
                reset = current[buffer.key].unsqueeze(1).expand_as(history)
                next_state[_state_name("previous", buffer.key)] = current[buffer.key]
            shifted = torch.cat([newest.unsqueeze(1), history[:, :-1]], dim=1)
            history = torch.where(push.view(self.batch, 1, *extra), shifted, history)
            history = torch.where(first.view(self.batch, 1, *extra), reset, history)
            values[buffer.key] = history
            next_state[name] = history
        inputs = {}
        for spec in self.history_specs:
            steps = (spec.history or spec.observation_history).steps_back
            history = values[self._buffer_keys[spec.key]]
            selected = history[:, [step - 1 for step in steps]]
            inputs[spec.key] = selected.reshape(spec.shape)
        return inputs, next_state


def _pipeline_arguments(history, current, input_keys):
    """The pipeline's positional arguments: a history input or a host input each."""
    arguments = []
    for key in input_keys:
        if key in history:
            arguments.append(history[key])
        elif key in current:
            arguments.append(current[key])
        else:
            raise KeyError(
                f"pipeline input {key!r} is neither a host input nor a history "
                "of the contract, so no tick supplies it"
            )
    return arguments


def _step(pipeline, graph_history, input_keys, state, tick):
    current = {spec.key: tick[spec.key] for spec in graph_history.host_specs}
    applied = {key: tick[applied_input_name(key)] for key in graph_history.applied_keys}
    history, next_state = graph_history.advance(
        state, current, applied, tick[COMMAND_ACCEPTED]
    )
    values = pipeline(*_pipeline_arguments(history, current, input_keys))
    return dict(zip(OUTPUT_NAMES, values)), next_state


def sample_ticks(
    graph_history: GraphHistory,
    sample: Mapping[str, torch.Tensor],
    count: int = 8,
    seed: int = 0,
) -> list[dict]:
    """Consecutive ticks around a sample: a reset, then accepted and dropped commands."""
    generator = torch.Generator().manual_seed(seed)
    ticks = []
    for index in range(count):
        tick = {}
        for spec in graph_history.host_specs:
            value = sample[spec.key]
            if value.dtype.is_floating_point and spec.provider not in ("masked_mimic",):
                value = value + 0.05 * torch.randn(value.shape, generator=generator)
            tick[spec.key] = value
        for key in graph_history.applied_keys:
            tick[applied_input_name(key)] = 0.3 * torch.randn(
                (
                    graph_history.batch,
                    *next(
                        b.feature_shape for b in graph_history.buffers if b.key == key
                    ),
                ),
                generator=generator,
            )
        tick[COMMAND_ACCEPTED] = torch.tensor([[index % 3 != 2]] * graph_history.batch)
        tick["reset"] = index == 0
        ticks.append(tick)
    return ticks


def export_graph_history_bundle(
    pipeline: torch.nn.Module,
    contract: DeploymentContract,
    input_keys: Sequence[str],
    ticks: Sequence[Mapping[str, torch.Tensor]],
    output: Path,
    opset: int = 17,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> Path:
    """Trace ``pipeline`` with LEAPP-owned histories over consecutive ``ticks``.

    ``input_keys`` are the pipeline's arguments in order, as contract keys.
    Each tick maps every host input key, ``applied_<output>`` and
    ``command_accepted`` to a batched tensor; the first tick must be a reset.
    LEAPP validates every recorded transition, including the next state.
    Returns the bundle YAML path.
    """
    import leapp
    from leapp import TensorSemantics, annotate
    from leapp.leapp_graph.datatypes import is_traced_type

    from deployment.native.tools.leapp_export import _pipeline_ports

    require_torchscript_disabled()
    graph_history = GraphHistory(contract)
    if not ticks or not ticks[0].get("reset", False):
        raise ValueError("the first tick must start from the initial state")
    ports = {
        entry["protomotions_key"]: entry for entry in _pipeline_ports(contract)["input"]
    }
    out_ports = {entry["name"]: entry for entry in _pipeline_ports(contract)["output"]}
    joint_names = next(
        (
            entry["element_names"][0]
            for entry in out_ports.values()
            if "element_names" in entry
        ),
        None,
    )

    def port(entry: dict, value: torch.Tensor, **semantics) -> TensorSemantics:
        extra = {"protomotions_key": entry["protomotions_key"]}
        return TensorSemantics(entry["name"], value, extra=extra, **semantics)

    reference_state = graph_history.initial_state()
    states = []
    with torch.no_grad():
        for tick in ticks:
            if tick.get("reset"):
                reference_state = graph_history.initial_state()
            states.append(reference_state)
            _, reference_state = _step(
                pipeline, graph_history, input_keys, reference_state, tick
            )

    output.mkdir(parents=True, exist_ok=False)
    traced_outputs = None
    leapp.start(name="policy", save_path=str(output), max_cached_io=len(ticks))
    try:
        for tick, state in zip(ticks, states):
            host_ports = []
            for spec in graph_history.host_specs:
                entry = ports[spec.key]
                semantics = {k: entry[k] for k in ("kind",) if k in entry}
                if "element_names" in entry and len(entry["element_names"]) == 1:
                    semantics["element_names"] = list(entry["element_names"][0])
                host_ports.append(port(entry, tick[spec.key], **semantics))
            for key in graph_history.applied_keys:
                semantics = {"element_names": list(joint_names)} if joint_names else {}
                host_ports.append(
                    TensorSemantics(
                        applied_input_name(key),
                        tick[applied_input_name(key)],
                        extra={
                            "protomotions_key": key,
                            "role": "applied on the previous tick",
                        },
                        **semantics,
                    )
                )
            host_ports.append(
                TensorSemantics(
                    COMMAND_ACCEPTED,
                    tick[COMMAND_ACCEPTED],
                    extra={"role": "false if the previous command was dropped"},
                )
            )
            names = [
                *(spec.key for spec in graph_history.host_specs),
                *(applied_input_name(k) for k in graph_history.applied_keys),
                COMMAND_ACCEPTED,
            ]
            traced = dict(zip(names, annotate.input_tensors(NODE_NAME, host_ports)))
            state_names = list(state)
            traced_state = dict(
                zip(
                    state_names,
                    annotate.state_tensors(
                        NODE_NAME, {name: state[name].clone() for name in state_names}
                    ),
                )
            )
            with torch.no_grad():
                outputs, next_state = _step(
                    pipeline, graph_history, input_keys, traced_state, traced
                )
            annotate.update_state(NODE_NAME, next_state)
            if traced_outputs is None:
                # Constant gains are not traced; LEAPP exports them as static outputs.
                traced_outputs = [n for n in OUTPUT_NAMES if is_traced_type(outputs[n])]
                if not {"actions", "joint_pos_targets"} <= set(traced_outputs):
                    raise ValueError("action outputs must be computed from the inputs")

            def out_port(name, value):
                entry = out_ports[name]
                semantics = {k: entry[k] for k in ("kind",) if k in entry}
                if "element_names" in entry:
                    semantics["element_names"] = list(entry["element_names"][0])
                return TensorSemantics(name, value, **semantics)

            annotate.output_tensors(
                NODE_NAME,
                [out_port(n, outputs[n]) for n in traced_outputs],
                static_outputs=[
                    out_port(n, outputs[n].detach().clone())
                    for n in OUTPUT_NAMES
                    if n not in traced_outputs
                ]
                or None,
                export_with="onnx-torchscript",
                backend_params={"opset_version": opset},
            )
    finally:
        leapp.stop()
    leapp.compile_graph(
        visualize=False, validate=True, rtol=rtol, atol=atol, strict=True
    )
    return output / "policy" / "policy.yaml"
