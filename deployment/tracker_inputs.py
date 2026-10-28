# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ONNX input protocol for ProtoMotions tracker deployment.

This module contains framework-agnostic NumPy logic used by standalone MuJoCo
deployment and hardware integrations such as RoboJuDo. Hardware/framework code
is still responsible for reading current robot sensors; this module owns the
frame conventions and semantic ONNX input assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import numpy as np

from deployment.state_utils import (
    apply_heading_offset_np,
    compute_heading_quat_inv_np,
    compute_yaw_offset_np,
    quat_rotate_np,
)
from protomotions.envs.context_sources import (
    CURRENT_NAMESPACE,
    SENSOR_SOURCES,
    current_namespace,
    source_of_path,
)
from protomotions.envs.sensors.config import require_root_identity_imu

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.sensors.metadata import SensorDeploymentMetadata
    from protomotions.robot_configs.base import RobotConfig


def tracker_sensor_metadata(
    env: EnvConfig, robot: RobotConfig, actor_keys: list[str]
) -> Optional[SensorDeploymentMetadata]:
    """Describe sensor-backed tracker inputs; privileged actors make no such claim.

    The current host protocol provides root-frame gyro readings directly.
    A mounted or non-root IMU needs a host estimator before this export path
    can consume it, so unsupported mounts are rejected rather than re-labelled.
    """
    sensor_bindings = [
        (key, path.path)
        for key in actor_keys
        for path in env.observation_components[key].dynamic_vars.values()
        if source_of_path(path.path) in SENSOR_SOURCES
    ]
    if not sensor_bindings:
        return None
    if env.sensors is None:
        key, path = sensor_bindings[0]
        raise ValueError(
            f"Actor observation {key!r} binds {path!r} without env.sensors"
        )
    mounts = env.sensors.mounts(robot)
    # Only the sensor channels the actor actually reads are exported as host
    # requirements: a configured sensor the policy never binds is not listed.
    # The channel of each bound sensor path is the one the name derives from.
    from protomotions.envs.sensors.measurement import derived_sensor_channel

    channels = {
        channel
        for _key, path in sensor_bindings
        for channel in (derived_sensor_channel(path.rsplit(".", 1)[-1]),)
        if channel is not None
    }
    metadata = env.sensors.deployment_metadata(robot, mounts, channels=channels)
    require_root_identity_imu(mounts)
    return metadata


def tracker_input_keys(source_keys: list[str]) -> list[str]:
    """Name current sensor inputs using the tracker's existing wire protocol.

    Source bindings and positional order stay unchanged. Only tracker ONNX
    names/sidecar keys change; MM/SAMP exports keep their noisy_* protocol.
    """
    sensor_prefixes = tuple(
        f"{current_namespace(source)}." for source in SENSOR_SOURCES
    )
    keys = [
        f"{CURRENT_NAMESPACE}." + key.split(".", 1)[1]
        if key.startswith(sensor_prefixes)
        else key
        for key in source_keys
    ]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "Tracker sensor input rename collides with a privileged input; resolve the observation sources"
        )
    return keys


# ---------------------------------------------------------------------------
# Action-history inputs
#
# ``BaseEnv.step`` processes the policy's action and steps physics.  Then it
# pushes the new state into the history, together with the action that
# produced it (slot 0), and only then builds observations.  So:
#
# * ``processed_action_history`` and ``raw_action_history`` read slots
#   ``0..N-1``: index ``j`` holds the action from ``j + 1`` policy calls ago.
# * ``historical.processed_actions`` reads slots ``1..N``: index ``j`` holds the
#   joint targets from ``j + 2`` calls ago.  Trackers trained before the two
#   keys above existed, the released ones included, read this.
#
# Until that many calls have run since a reset, a slot holds what the reset
# wrote.  The exporter writes all of this into each entry (``steps_back`` and
# ``fill_before_available``, alongside ``history``, ``first_action_lag`` and
# ``reset_value``), and a CPU test steps a real env to check training against
# these constants.
# ---------------------------------------------------------------------------

LEGACY_ACTION_HISTORY_KEY = "historical.processed_actions"

# Each action-history observation key: the graph output it feeds back, and how
# many calls back its first slot reaches.
ACTION_HISTORY_INPUTS = {
    "processed_action_history": ("joint_pos_targets", 1),
    "raw_action_history": ("actions", 1),
    LEGACY_ACTION_HISTORY_KEY: ("joint_pos_targets", 2),
    # The policy's own last raw action, fed back one call ago. A single-slot
    # feedback declared flat [1, D] rather than a [1, N, D] window.
    "current_action": ("actions", 1),
}

# What a reset leaves in the processed-action history: "zeros", or
# "pd_action_offset" for the default pose (the targets for a zero action).
# Raw actions always reset to zeros.
ACTION_HISTORY_RESET_FILL = "zeros"

# How to read an entry written before the exporter stated these facts: each
# key's convention when it was introduced, with a zeros reset.  These are
# frozen, so changing the constants above for new training doesn't change how
# old artifacts are read.  A tracker from code that reset to the default pose
# and was exported without the fields needs ``fill_before_available:
# pd_action_offset`` added to its YAML entry.
PRE_CONTRACT_FIRST_STEPS_BACK = {
    "processed_action_history": 1,
    "raw_action_history": 1,
    LEGACY_ACTION_HISTORY_KEY: 2,
    "current_action": 1,
}
PRE_CONTRACT_RESET_FILL = "zeros"


def _action_history_key(key: str) -> str:
    if key not in ACTION_HISTORY_INPUTS:
        raise ValueError(
            f"{key!r} is not an action-history input; expected one of "
            f"{sorted(ACTION_HISTORY_INPUTS)}"
        )
    return key


def action_history_output(key: str) -> str:
    """The graph output whose values the ``key`` input holds."""
    return ACTION_HISTORY_INPUTS[_action_history_key(key)][0]


def action_history_steps_back(key: str, num_slots: int) -> list[int]:
    """Calls back for each slot of a ``[1, num_slots, D]`` ``key`` input."""
    first = ACTION_HISTORY_INPUTS[_action_history_key(key)][1]
    return list(range(first, first + int(num_slots)))


def action_history_fill(
    key: str, pd_action_offset: Optional[Sequence[float]] = None
) -> str | list[float]:
    """The YAML ``fill_before_available`` value for this code's reset of ``key``.

    Written as an explicit list of joint targets when it is the default pose,
    so a consumer does not need to know where the offset lives.
    """
    if (
        action_history_output(key) != "joint_pos_targets"
        or ACTION_HISTORY_RESET_FILL == "zeros"
        or pd_action_offset is None
    ):
        return "zeros"
    return [float(value) for value in np.asarray(pd_action_offset).reshape(-1)]


def resolve_fill(
    spec: Any, num_dofs: int, pd_action_offset: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Turn a ``fill_before_available`` value into ``num_dofs`` values.

    ``spec`` is ``"zeros"``, ``"pd_action_offset"`` (which needs
    ``pd_action_offset``), or an explicit list of ``num_dofs`` values.
    """
    if spec is None:
        spec = "zeros"
    if isinstance(spec, str):
        if spec == "zeros":
            return np.zeros(num_dofs, dtype=np.float32)
        if spec != "pd_action_offset":
            raise ValueError(f"unknown fill_before_available value {spec!r}")
        if pd_action_offset is None:
            raise ValueError(
                "fill_before_available is pd_action_offset, but no offset was given"
            )
        spec = pd_action_offset
    fill = np.asarray(spec, dtype=np.float32)
    if fill.shape != (num_dofs,) or not np.isfinite(fill).all():
        raise ValueError(
            f"fill_before_available must hold {num_dofs} finite values, "
            f"got shape {fill.shape}"
        )
    return fill


def action_history_entries(meta: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """The YAML ``policy_inputs`` entries for action histories, by key."""
    return {
        entry["key"]: entry
        for entry in meta.get("policy_inputs", [])
        if entry.get("key") in ACTION_HISTORY_INPUTS
    }


class ActionFeedback:
    """Feed a tracker its own past actions the way training did.

    ``steps_back[i]`` is how many policy calls ago the values in input slot
    ``i`` were produced.  Slots with nothing recorded since the last reset
    hold ``fill``.  Record the graph's ``output_name`` output as returned,
    once per real policy call and before any deploy-side clamp or smoothing:
    that is what training stores.
    """

    def __init__(
        self,
        num_dofs: int,
        steps_back: Sequence[int],
        fill: Optional[np.ndarray] = None,
        output_name: str = "joint_pos_targets",
    ) -> None:
        steps = tuple(int(step) for step in steps_back)
        if not steps or min(steps) < 1:
            raise ValueError(
                f"steps_back must be positive call counts, got {steps_back!r}"
            )
        self.num_dofs = int(num_dofs)
        self.steps_back = steps
        self.output_name = output_name
        self.fill = resolve_fill(fill, self.num_dofs)
        # Row k holds the values from k + 1 calls ago.
        self._history = np.empty((max(steps), self.num_dofs), dtype=np.float32)
        self._rows = np.asarray([step - 1 for step in steps], dtype=np.intp)
        self._input = np.empty((len(steps), self.num_dofs), dtype=np.float32)
        self.reset()

    @classmethod
    def from_input_entry(
        cls,
        entry: Mapping[str, Any],
        num_dofs: int,
        pd_action_offset: Optional[Sequence[float]] = None,
    ) -> "ActionFeedback":
        """Build from a YAML action-history entry.

        The entry's ``key`` picks the graph output to feed back.  The lag and
        reset value come from ``steps_back`` and ``fill_before_available``,
        else from ``first_action_lag`` and ``reset_value``.  YAMLs with neither
        get the convention their key was trained with when it was introduced
        (``PRE_CONTRACT_*``), sized by the entry's ``[1, N, num_dofs]`` shape.
        ``pd_action_offset`` resolves a ``"pd_action_offset"`` fill.
        """
        key = _action_history_key(entry.get("key", ""))
        shape = [int(value) for value in entry.get("shape", [1, 1, num_dofs])]
        if len(shape) != 3 or shape[0] != 1 or shape[2] != int(num_dofs):
            raise ValueError(
                f"action-history input must have shape [1, N, {num_dofs}], got {shape}"
            )
        steps = entry.get("steps_back")
        if steps is None:
            first = entry.get("first_action_lag", PRE_CONTRACT_FIRST_STEPS_BACK[key])
            steps = list(range(int(first), int(first) + shape[1]))
        steps = [int(step) for step in steps]
        if len(steps) != shape[1]:
            raise ValueError(
                f"action-history input declares steps_back={steps} for "
                f"{shape[1]} history slots"
            )
        fill = entry.get("fill_before_available")
        if fill is None:
            reset_value = entry.get("reset_value")
            fill = (
                PRE_CONTRACT_RESET_FILL
                if reset_value is None
                else [float(reset_value)] * int(num_dofs)
            )
        fill = resolve_fill(fill, num_dofs, pd_action_offset=pd_action_offset)
        return cls(num_dofs, steps, fill, output_name=action_history_output(key))

    @classmethod
    def from_history_spec(
        cls,
        spec: Any,
        num_dofs: int,
        pd_action_offset: Optional[Sequence[float]] = None,
    ) -> "ActionFeedback":
        """Build from a contract ``InputSpec`` whose ``.history`` is set.

        The graph output to feed back, the per-slot call counts and the reset
        fill are read from the typed ``HistorySlotSpec`` the exporter wrote,
        not from the host-protocol constants.  The call counts are derived from
        the typed ``first_action_lag`` and ``slots`` (slot ``i`` holds the value
        from ``first_action_lag + i`` calls ago), not the ``steps_back`` field,
        which the spec's own docstring calls a legacy duplicate and which a spec
        carrying only the typed fields leaves empty.  ``pd_action_offset``
        resolves a default-pose fill carried as the literal ``"pd_action_offset"``.
        """
        slot = spec.history
        fill = resolve_fill(
            slot.reset_fill, num_dofs, pd_action_offset=pd_action_offset
        )
        steps = [slot.first_action_lag + i for i in range(slot.slots)]
        return cls(
            num_dofs,
            steps,
            fill,
            output_name=slot.output_key,
        )

    def reset(self) -> None:
        """Forget every recorded call, as a training reset does."""
        self._history[:] = self.fill
        self._input[:] = self.fill

    def record(self, output: np.ndarray) -> None:
        """Store the ``output_name`` values one real policy call returned."""
        values = np.asarray(output, dtype=np.float32).reshape(self.num_dofs)
        if len(self._history) > 1:
            self._history[1:] = self._history[:-1].copy()
        self._history[0] = values
        np.take(self._history, self._rows, axis=0, out=self._input)

    def as_input(self) -> np.ndarray:
        """The ``[len(steps_back), num_dofs]`` history for the next call.

        The same array is returned every time and ``record`` updates it in
        place; copy it to keep a call's input.
        """
        return self._input


@dataclass
class TrackerInputRequirements:
    """Semantic ONNX inputs requested by an exported tracker."""

    odom_from_start: bool
    odom_start_xy: bool
    future_pos: bool
    future_anchor_pos: bool
    start_heading: bool
    ref_rigid_body_pos: bool
    ref_anchor_pos: bool

    @property
    def needs_reference_positions(self) -> bool:
        """True if the policy consumes reference *positions* in any form.

        Both the full body array and the anchor-only slice come from the same
        source array, so either one means the deployment has to supply and
        align reference positions.
        """
        return self.future_pos or self.future_anchor_pos

    @property
    def needs_spatial_start(self) -> bool:
        return (
            self.odom_from_start
            or self.odom_start_xy
            or self.needs_reference_positions
            or self.start_heading
            or self.ref_rigid_body_pos
            or self.ref_anchor_pos
        )


@dataclass
class TrackerAlignment:
    """Per-motion-start spatial alignment state."""

    heading_offset: np.ndarray
    odom_start_xy: np.ndarray
    odom_start_heading_inv: np.ndarray
    robot_anchor_start_pos: np.ndarray
    motion_anchor_start_pos: np.ndarray


def get_tracker_input_requirements(contract) -> TrackerInputRequirements:
    """Which deploy quantities a host must supply, read from the contract alone.

    Dispatches on each input's declared ``key``, never on a name substring; a
    host reads this straight from the sidecar without the training config.
    """
    semantic_keys = {spec.key for spec in contract.inputs}
    return TrackerInputRequirements(
        odom_from_start="current.odom_disp_start" in semantic_keys,
        odom_start_xy="current.odom_start_xy" in semantic_keys,
        future_pos="mimic.future_pos" in semantic_keys,
        future_anchor_pos="mimic.future_anchor_pos" in semantic_keys,
        start_heading="current.odom_start_heading_inv" in semantic_keys,
        ref_rigid_body_pos="mimic.ref_state.rigid_body_pos" in semantic_keys,
        ref_anchor_pos="mimic.ref_anchor_pos" in semantic_keys,
    )


def init_tracker_alignment(
    robot_anchor_pos: np.ndarray,
    robot_anchor_rot: np.ndarray,
    motion_anchor_pos: np.ndarray,
    motion_anchor_rot: np.ndarray,
) -> TrackerAlignment:
    """Capture the shared spatial frame at motion start."""
    return TrackerAlignment(
        heading_offset=compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot),
        odom_start_xy=np.asarray(robot_anchor_pos[:2], dtype=np.float32).copy(),
        odom_start_heading_inv=compute_heading_quat_inv_np(robot_anchor_rot),
        robot_anchor_start_pos=np.asarray(robot_anchor_pos, dtype=np.float32).copy(),
        motion_anchor_start_pos=np.asarray(motion_anchor_pos, dtype=np.float32).copy(),
    )


def align_future_motion_refs(
    future_refs: dict,
    alignment: TrackerAlignment,
) -> dict:
    """Move the reference motion from its recorded frame into the robot's start frame.

    Every entry present in ``future_refs`` is aligned. Aligning positions is not
    optional: ``mimic.future_anchor_pos`` is sliced out of ``body_pos``, and the
    policy graph subtracts ``mimic.ref_anchor_pos`` from it to recover how far the
    reference itself travels. ``mimic.ref_anchor_pos`` is always expressed in the
    robot's frame, because the odometer offset differences it against the robot's
    own believed position. Leaving ``body_pos`` in the motion's recorded frame
    therefore makes that subtraction return the offset between the two frames --
    typically tens of metres, since clips live wherever they were captured --
    rather than the reference's travel.

    Keys the caller does not supply are left absent, so a deployment whose policy
    reads no reference positions does not have to provide them.
    """
    aligned = dict(future_refs)
    if "body_rot" in future_refs:
        aligned["body_rot"] = apply_heading_offset_np(
            alignment.heading_offset, future_refs["body_rot"]
        )
    if "body_pos" in future_refs:
        motion_disp = future_refs["body_pos"] - alignment.motion_anchor_start_pos
        aligned["body_pos"] = alignment.robot_anchor_start_pos + quat_rotate_np(
            alignment.heading_offset, motion_disp
        )
    return aligned


def compute_odom_disp_start(
    robot_anchor_xy: np.ndarray,
    alignment: TrackerAlignment,
    corruption_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Compute start-heading-frame odometer displacement from robot anchor XY."""
    robot_disp_world = (
        np.asarray(robot_anchor_xy, dtype=np.float32) - alignment.odom_start_xy
    )
    robot_disp_world_3d = np.array(
        [robot_disp_world[0], robot_disp_world[1], 0.0], dtype=np.float32
    )
    odom_disp_start = quat_rotate_np(
        alignment.odom_start_heading_inv, robot_disp_world_3d
    )[:2]
    if corruption_fn is not None:
        odom_disp_start = corruption_fn(odom_disp_start)
    return odom_disp_start.astype(np.float32)


def build_tracker_onnx_inputs(
    contract,
    *,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    anchor_rot: np.ndarray,
    root_local_ang_vel: np.ndarray,
    future_refs: dict,
    prev_actions: Optional[np.ndarray] = None,
    prev_raw_actions: Optional[np.ndarray] = None,
    action_histories: Optional[Mapping[str, np.ndarray]] = None,
    odom_start_xy: Optional[np.ndarray] = None,
    odom_disp_start: Optional[np.ndarray] = None,
    odom_start_heading_inv: Optional[np.ndarray] = None,
    ref_rigid_body_pos: Optional[np.ndarray] = None,
    ref_anchor_pos: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Assemble ONNX input arrays from the contract and the host's own tensors.

    The graph's inputs, their keys and the dimensions come from ``contract``
    alone -- the reference host reads only the sidecar. Each ``InputSpec`` is
    fed from the host's own ``key_to_array`` mapping by its declared ``key``,
    never a name substring, under its ``onnx_name``.

    ``prev_actions`` (joint targets) and ``prev_raw_actions`` (policy outputs)
    are action histories, ``[N, num_dofs]`` from ``ActionFeedback.as_input()``.
    A 1-D ``[num_dofs]`` array is accepted as a one-slot history; the caller is
    then responsible for it holding the values the input's lag calls for (see
    ``ACTION_HISTORY_INPUTS``).  ``action_histories`` maps action-history keys
    to their own windows and takes precedence, so a graph with more than one
    such input gets each at its own lag.
    """
    num_dofs = len(contract.robot.joint_names)
    anchor_body_index = contract.robot.anchor_body_index
    if prev_actions is None:
        prev_actions = np.zeros(num_dofs, dtype=np.float32)
    if prev_raw_actions is None:
        prev_raw_actions = np.zeros(num_dofs, dtype=np.float32)
    processed_window = np.asarray(prev_actions, dtype=np.float32).reshape(
        1, -1, num_dofs
    )

    key_to_array = {
        "current.dof_pos": dof_pos[None],
        "current.dof_vel": dof_vel[None],
        "current.anchor_rot": anchor_rot[None],
        "current.root_local_ang_vel": root_local_ang_vel[None],
        "processed_action_history": processed_window,
        "raw_action_history": np.asarray(prev_raw_actions, dtype=np.float32).reshape(
            1, -1, num_dofs
        ),
        LEGACY_ACTION_HISTORY_KEY: processed_window,
    }
    for key, window in (action_histories or {}).items():
        key_to_array[_action_history_key(key)] = np.asarray(
            window, dtype=np.float32
        ).reshape(1, -1, num_dofs)
    # Only source what the caller actually supplied. A deployment whose policy
    # reads no reference positions has no reason to carry the full body array.
    if "body_rot" in future_refs:
        key_to_array["mimic.future_rot"] = future_refs["body_rot"][None]
        key_to_array["mimic.future_anchor_rot"] = future_refs["body_rot"][
            :, anchor_body_index, :
        ][None]
    if "body_pos" in future_refs:
        key_to_array["mimic.future_pos"] = future_refs["body_pos"][None]
        key_to_array["mimic.future_anchor_pos"] = future_refs["body_pos"][
            :, anchor_body_index, :
        ][None]
    if "dof_pos" in future_refs:
        key_to_array["mimic.future_dof_pos"] = future_refs["dof_pos"][None]
    if "dof_vel" in future_refs:
        key_to_array["mimic.future_dof_vel"] = future_refs["dof_vel"][None]
    if odom_start_xy is not None:
        key_to_array["current.odom_start_xy"] = odom_start_xy[None]
    if odom_disp_start is not None:
        key_to_array["current.odom_disp_start"] = odom_disp_start[None]
    if odom_start_heading_inv is not None:
        key_to_array["current.odom_start_heading_inv"] = odom_start_heading_inv[None]
    if ref_anchor_pos is not None:
        # Reference anchor position at the current step. With
        # mimic.future_anchor_pos this tells the tracker how far the reference
        # itself moves, and it is the reference point for the odometer offset.
        key_to_array["mimic.ref_anchor_pos"] = ref_anchor_pos[None]
    if ref_rigid_body_pos is not None:
        # Full reference body array: only needed by observations that read every
        # reference body. The baked-FK tracker path does NOT -- it needs the
        # anchor alone -- so most deployments can leave this out.
        key_to_array["mimic.ref_state.rigid_body_pos"] = ref_rigid_body_pos[None]
        key_to_array.setdefault(
            "mimic.ref_anchor_pos", ref_rigid_body_pos[anchor_body_index][None]
        )

    onnx_inputs = {}
    unsourced = []
    for spec in contract.inputs:
        if spec.key not in key_to_array:
            unsourced.append(f"{spec.onnx_name} ({spec.key})")
            continue
        value = key_to_array[spec.key]
        onnx_inputs[spec.onnx_name] = (
            value.astype(np.bool_)
            if value.dtype == np.bool_
            else value.astype(np.float32)
        )
    if unsourced:
        # Skipping quietly here just defers the failure to onnxruntime, which
        # reports a missing feed without saying which semantic quantity is
        # absent or who was supposed to provide it.
        raise KeyError(
            "The exported policy requires inputs this assembler cannot source: "
            + ", ".join(unsourced)
            + ". Supply the missing reference data in `future_refs` or the "
            "optional arguments, or extend `build_tracker_onnx_inputs` to "
            "cover the semantic key."
        )
    return onnx_inputs
