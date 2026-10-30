# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed, self-validating deployment contract written into the ONNX sidecar.

The exporter builds a :class:`DeploymentContract` from a checkpoint and writes it
to ``unified_pipeline.yaml``; a deployment host reads the same file back through
:meth:`DeploymentContract.from_yaml`. The YAML keeps today's layout -- every
top-level key a host already reads is unchanged -- and gains two new top-level
keys: ``schema``, written first, so an old reader still finds what it expects
and a new reader can refuse a file whose schema it does not know; and ``host``,
written only for a simulation export (omitted means robot).

The contract validates itself at construction: a graph input with no deployment
kind or provider, a history input that feeds back an output the graph does not
produce, an anchor body that is not in the body list, control vectors the wrong
length, or a sensor input whose channel the sensors block does not carry each
raise here, at export time, instead of silently shipping a sidecar a robot
cannot be driven from. The contract also names the host it is for: an input
whose kind only a simulation host can feed (the ``simulated_*`` body state) is
refused in a robot contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from protomotions.envs.sensors.metadata import SensorDeploymentMetadata
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas

SCHEMA = "unified_pipeline/1"

# Providers a deployment host must map to one of its own subsystems.
SIMULATOR_TRUTH = "simulator_truth"
REFERENCE_MOTION = "reference_motion"
POLICY_HISTORY = "policy_history"
# Operator-supplied targets, style and velocity commands, not physical sensors
# or the tracker's reference-motion stream.
MASKED_MIMIC = "masked_mimic"
# The scene boxes a scene-conditioned policy sees (object poses, box geometry,
# visibility): a perception or scene-description stream the host supplies, not
# a physical sensor of the robot.
SCENE = "scene"

# The sensor channels a provider can name; these are the field names of
# SensorConfig. ``imu`` and ``joint_encoders`` are always present on a tracker;
# the other three are optional channels carried on SensorDeploymentMetadata.
SENSOR_KINDS = frozenset(
    {"joint_encoders", "imu", "localization", "base_velocity", "height"}
)
OPTIONAL_SENSOR_KINDS = frozenset({"localization", "base_velocity", "height"})

# The hosts an export can target. A robot host feeds raw sensor readings, its
# own past actions and the goal streams; a simulation host (a game engine, a
# browser MuJoCo, a sim-to-sim check) can also feed the simulator's own body
# state, the ``simulated_*`` kinds below. The robot set is every other kind in
# ELEMENT_LABELS; the simulation set is all of them.
ROBOT_HOST = "robot"
SIMULATION_HOST = "simulation"
HOSTS = (ROBOT_HOST, SIMULATION_HOST)
SIMULATION_ONLY_KINDS = frozenset(
    {
        "simulated_body_pos",
        "simulated_body_rot",
        "simulated_body_vel",
        "simulated_body_ang_vel",
        "simulated_ground_heights",
    }
)

# Context names whose values are the policy's own past actions, fed back from
# the graph's outputs. The leaves ``raw_actions``/``processed_actions`` are the
# command history under a historical namespace; the rest are top-level names.
_POLICY_HISTORY_PATHS = frozenset(
    {
        "raw_action_history",
        "processed_action_history",
        "previous_action",
        "previous_processed_action",
        # The policy's own last raw action, fed back one call ago.
        "current_action",
    }
)
_POLICY_HISTORY_LEAVES = frozenset({"raw_actions", "processed_actions"})


# How each host-facing deploy kind labels the elements of its last axes, for the
# host's columns. One of five shapes: ``joints`` (the per-joint axis is the joint
# names), ``bodies`` (the per-body axis is the body names, plus a coordinate
# axis), ``anchor`` (the single anchor body, plus a coordinate axis), ``coords``
# (a bare coordinate axis), ``none`` (no labels). The coordinate width is read
# off the tensor's own last axis, so one ``bodies`` row serves a vec3 and a quat.
# Every declared host kind must have a label rule; unknown kinds fail export.
ELEMENT_LABELS: Dict[str, str] = {
    "anchor_ang_vel": "coords",
    "anchor_index": "none",
    "anchor_pos": "coords",
    "anchor_rot": "coords",
    "anchor_vel": "coords",
    "body_ang_vel": "bodies",
    "body_vel": "bodies",
    "joint_pos": "joints",
    "joint_vel": "joints",
    "last_actions": "joints",
    "local_anchor_ang_vel": "none",
    "local_root_ang_vel": "none",
    "masked_mimic_anchor_rotation_mask": "none",
    "masked_mimic_anchor_translation_mask": "none",
    "masked_mimic_average_velocity": "coords",
    "masked_mimic_local_ref_pos": "bodies",
    "masked_mimic_local_ref_rot": "bodies",
    "masked_mimic_local_target_bodies_masks": "none",
    "masked_mimic_local_target_poses_masks": "none",
    "masked_mimic_local_time_offsets": "none",
    "masked_mimic_ref_pos": "bodies",
    "masked_mimic_ref_rot": "bodies",
    "masked_mimic_style_condition_masks": "none",
    "masked_mimic_style_frame_masks": "none",
    "masked_mimic_style_pose_samples": "none",
    "masked_mimic_style_velocity_samples": "none",
    "masked_mimic_target_bodies_masks": "none",
    "masked_mimic_target_poses_masks": "none",
    "masked_mimic_time_offsets": "none",
    "masked_mimic_velocity_command_mask": "none",
    "odom_disp_start": "coords",
    "odom_start_heading_inv": "coords",
    "odom_start_xy": "coords",
    "reference_motion_anchor_ang_vel": "anchor",
    "reference_motion_anchor_pos": "anchor",
    "reference_motion_anchor_pos_future": "anchor",
    "reference_motion_anchor_rot": "anchor",
    "reference_motion_anchor_vel": "anchor",
    "reference_motion_body_ang_vel": "bodies",
    "reference_motion_body_pos": "bodies",
    "reference_motion_body_pos_future": "bodies",
    "reference_motion_body_rot": "bodies",
    "reference_motion_body_vel": "bodies",
    "reference_motion_joint_pos": "joints",
    "reference_motion_joint_vel": "joints",
    "root_ang_vel": "coords",
    "root_height": "none",
    "root_pos": "coords",
    "root_rot": "coords",
    "root_vel": "coords",
    "scene_box_centers": "coords",
    "scene_box_extents": "coords",
    "scene_box_valid": "none",
    "scene_object_visible": "none",
    "scene_object_pos": "coords",
    "scene_object_rot": "coords",
    "scene_position_offset": "coords",
    "scene_yaw_offset": "none",
    "simulated_body_ang_vel": "bodies",
    "simulated_body_pos": "bodies",
    "simulated_body_rot": "bodies",
    "simulated_body_vel": "bodies",
    "simulated_ground_heights": "none",
}


def _check_host(host: str) -> None:
    if host not in HOSTS:
        raise ValueError(f"unknown deployment host {host!r}; expected one of {HOSTS}")


def host_can_feed(host: str, kind: str) -> bool:
    """Whether ``host`` can supply an input of ``kind``; an unknown host is refused.

    Only the simulator's own body state is host-specific: a simulation host
    feeds every kind, a robot host every kind but those.
    """
    _check_host(host)
    return host == SIMULATION_HOST or kind not in SIMULATION_ONLY_KINDS


def kinds_for_host(host: str) -> frozenset:
    """The host kinds an export for ``host`` may bind; an unknown host is refused."""
    _check_host(host)
    return frozenset(kind for kind in ELEMENT_LABELS if host_can_feed(host, kind))


def accepting_host(kind: str) -> str:
    """The least capable host that can feed ``kind``."""
    return SIMULATION_HOST if kind in SIMULATION_ONLY_KINDS else ROBOT_HOST

# The column labels of a coordinate axis of a given width.
_COORD_LABELS: Dict[int, Tuple[str, ...]] = {
    2: ("x", "y"),
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "w"),
}


def element_labels(
    kind: str,
    joint_names: Tuple[str, ...],
    body_names: Tuple[str, ...],
    anchor_name: str,
    coord_width: int,
) -> Tuple[Tuple[str, ...], ...]:
    """The element names along a deploy kind's last axes, for the host's columns.

    ``coord_width`` is the tensor's own last-axis size (3 for a vector, 4 for a
    quaternion, 2 for a planar quantity); it labels the coordinate axis of a
    ``bodies``, ``anchor`` or ``coords`` kind and is ignored for ``joints`` and
    ``none``. The result is the host's column labels, keyed only by the host
    kind and the coordinate width. An unknown kind is a refusal, listing the
    known kinds, so a misspelled ``deploy=`` string fails at export.
    """
    try:
        shape = ELEMENT_LABELS[kind]
    except KeyError:
        raise ValueError(
            f"deploy kind {kind!r} has no element-label rule; known kinds: "
            f"{sorted(ELEMENT_LABELS)}"
        ) from None
    if shape == "none":
        return ()
    if shape == "joints":
        return (tuple(joint_names),)
    # bodies / anchor / coords all label a coordinate axis of width coord_width.
    try:
        coords = _COORD_LABELS[coord_width]
    except KeyError:
        raise ValueError(
            f"deploy kind {kind!r} has coordinate width {coord_width}, which has "
            f"no label row; expected one of {sorted(_COORD_LABELS)}"
        ) from None
    if shape == "bodies":
        return (tuple(body_names), coords)
    if shape == "anchor":
        return ((anchor_name,), coords)
    return (coords,)


def provider_of(path: str, channel: Optional[str]) -> str:
    """Which host subsystem supplies the input at ``path``, from its namespace.

    ``clean``/``noisy`` and their ``*_historical`` twins are sensor readings and
    are supplied by ``channel``, the sensor channel the name derives from
    (``sensors.measurement.derived_sensor_channel``); a ``None`` channel on a
    sensor path is a refusal, since a sensor path with no channel cannot be fed.
    ``current``/``historical`` are privileged simulator truth -- a privileged
    policy exports with that stated, never renamed -- ``mimic.*`` is the
    reference motion and ``scene_boxes.*`` the scene stream; all ignore
    ``channel``. The action-history names are the
    policy's own past outputs. The caller refuses a path with no host kind before
    calling this; here a path whose namespace has no provider rule is the refusal.
    """
    # Past actions are the policy's own output fed back, in whatever namespace
    # the command history sits, so they are matched before the namespace rule.
    if (
        path in _POLICY_HISTORY_PATHS
        or path.rsplit(".", 1)[-1] in _POLICY_HISTORY_LEAVES
    ):
        return POLICY_HISTORY
    head = path.split(".", 1)[0]
    if head in ("clean", "noisy", "clean_historical", "noisy_historical"):
        if channel is None:
            raise ValueError(
                f"sensor path {path!r} has no sensor channel; the name derives "
                f"from no sensor-measured field, so no channel can read it"
            )
        return channel
    if head in ("current", "historical"):
        return SIMULATOR_TRUTH
    if head == "mimic":
        return REFERENCE_MOTION
    if head == MASKED_MIMIC:
        return MASKED_MIMIC
    if head == "scene_boxes":
        return SCENE
    raise ValueError(f"no provider rule for deployment path {path!r}")


def provider_for_path(path: str) -> str:
    """The provider of a deployment path, deriving the channel it needs.

    The single source of truth for ``provider_of``'s channel argument: a
    ``clean``/``noisy`` path is supplied by the channel the raw field its leaf
    derives from is measured on (``derived_sensor_channel``); a privileged,
    ``mimic`` or policy-history path has no channel. The exporter and the
    declarations test both call this, so the two cannot drift.
    """
    from protomotions.envs.context_sources import SENSOR_NAMESPACES
    from protomotions.envs.sensors.measurement import derived_sensor_channel

    leaf = path.rsplit(".", 1)[-1]
    channel = (
        derived_sensor_channel(leaf)
        if path.split(".", 1)[0] in SENSOR_NAMESPACES
        else None
    )
    return provider_of(path, channel)


@dataclass(frozen=True)
class HistorySlotSpec:
    """What a host feeds an action-history input, and how to fill it after reset.

    ``output_key`` is the graph output the history replays (``actions`` for the
    raw history, ``joint_pos_targets`` for the processed one); ``slots`` is how
    many past calls it holds; ``first_action_lag`` and ``reset_fill`` are the
    environment's own ``StateHistoryBuffer`` facts, so a host reproduces the
    exact post-reset behaviour the policy trained under.

    A host feeds each feedback in the input's declared shape: a ``slots``-length
    window declared ``[1, slots, D]`` keeps its slot axis, a single-slot feedback
    declared flat ``[1, D]`` (the policy's last action) is fed flat. The Python
    reference host does not do this yet -- ``build_tracker_onnx_inputs`` must
    source ``current_action`` and feed every feedback in its declared shape (the
    host-wiring follow-up).

    ``steps_back`` and ``reset_value`` are the same facts stated the way today's
    sidecar already states them, kept so the file stays a strict superset until
    the after-handover schema bump; ``from_yaml`` rebuilds from the four fields
    above and ignores these.
    """

    output_key: str
    slots: int
    first_action_lag: int
    reset_fill: str | List[float]
    steps_back: Tuple[int, ...] = ()
    reset_value: Optional[float] = None


@dataclass(frozen=True)
class ObservationHistorySpec:
    """Past measurements, newest first, sampled before applying the action.

    At inference t, step 1 is the measurement at t-1, not the current sample.
    Reset repeats the first measured state until enough policy calls have run.
    ``current_key`` need not also be a graph input: the host still measures it.
    """

    current_key: str
    steps_back: Tuple[int, ...]
    reset_fill: str = "initial_measurement"


@dataclass(frozen=True)
class InputSpec:
    """One ONNX graph input: the host key, what it is, and who supplies it."""

    onnx_name: str
    key: str
    source_namespace: str
    kind: str
    provider: str
    shape: Tuple[int, ...]
    dtype: str
    element_names: Tuple[Tuple[str, ...], ...]
    future_steps: Optional[int]
    history: Optional[HistorySlotSpec]
    source: Optional[str] = None
    observation_history: Optional[ObservationHistorySpec] = None


@dataclass(frozen=True)
class OutputSpec:
    """One ONNX graph output the robot is driven from."""

    onnx_name: str
    kind: str
    shape: Tuple[int, ...]
    joint_names: Optional[Tuple[str, ...]]


@dataclass(frozen=True)
class RobotBlock:
    """The robot the policy was trained for; indices are derived, never stored.

    ``conditionable_body_names`` is present only for a masked-mimic export: the
    host builds the body submasks and the style condition mask itself, so it must
    know, in order, which body each mask lane selects. The masked-mimic mask
    inputs carry per-field kinds, while this list identifies their body-axis
    ordering. It mirrors ``masked_mimic.conditionable_bodies`` and is ``None``
    for a plain tracker.
    """

    mjcf_path: str
    body_names: Tuple[str, ...]
    joint_names: Tuple[str, ...]
    anchor_body_name: str
    root_body_name: str
    conditionable_body_names: Optional[Tuple[str, ...]] = None

    @property
    def anchor_body_index(self) -> int:
        return self.body_names.index(self.anchor_body_name)

    @property
    def root_body_index(self) -> int:
        return self.body_names.index(self.root_body_name)


def build_robot_block(robot_config, conditionable_body_names=None) -> RobotBlock:
    """Read the robot block off a resolved robot config.

    ``conditionable_body_names`` is passed (in publisher order) only for a
    masked-mimic export; left ``None`` the block stays as the plain tracker's.
    """
    body_names = tuple(robot_config.kinematic_info.body_names)
    return RobotBlock(
        mjcf_path=robot_config.asset.asset_file_name,
        body_names=body_names,
        joint_names=tuple(robot_config.kinematic_info.dof_names),
        anchor_body_name=robot_config.resolved_anchor_body_name,
        root_body_name=body_names[0],
        conditionable_body_names=(
            None
            if conditionable_body_names is None
            else tuple(conditionable_body_names)
        ),
    )


@dataclass(frozen=True)
class ControlBlock:
    """The PD gains, limits and action->target conversion baked into the graph."""

    action_fn: str
    stiffness: Tuple[float, ...]
    damping: Tuple[float, ...]
    effort_limits: Optional[Tuple[float, ...]]
    action_scale: Optional[Tuple[float, ...]]
    pd_action_offset: Optional[Tuple[float, ...]]
    pd_target_max_accel: Optional[float]


def _as_float_tuple(value) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if hasattr(value, "flatten"):
        return tuple(float(x) for x in value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(float(x) for x in value)
    return (float(value),)


def build_control_block(
    action_config, robot_config, pd_target_max_accel: Optional[float]
) -> ControlBlock:
    """Read the control block off the action config and robot gains.

    Gains and the effort limit come through ``ControlInfo.require``, so a joint
    missing one fails loudly here instead of shipping a sidecar with a null the
    host silently replaces with its own number. ``action_scale`` and
    ``pd_action_offset`` are read from the action config's own parameters: the
    conversion the ONNX graph actually performs, recorded so two policies whose
    scales differ are not byte-identical from outside.
    """
    joint_names = list(robot_config.kinematic_info.dof_names)
    control_info = robot_config.control.control_info
    stiffness = tuple(float(control_info[j].require("stiffness")) for j in joint_names)
    damping = tuple(float(control_info[j].require("damping")) for j in joint_names)
    effort_limits = tuple(
        float(control_info[j].require("effort_limit")) for j in joint_names
    )
    params = action_config.params if action_config is not None else {}
    action_fn = action_config.fn.__name__ if action_config is not None else "unknown"
    return ControlBlock(
        action_fn=action_fn,
        stiffness=stiffness,
        damping=damping,
        effort_limits=effort_limits,
        action_scale=_as_float_tuple(params.get("action_scale")),
        pd_action_offset=_as_float_tuple(params.get("pd_action_offset")),
        pd_target_max_accel=(
            None if pd_target_max_accel is None else float(pd_target_max_accel)
        ),
    )


@dataclass(frozen=True)
class TimingBlock:
    """Control and physics rates; the integer decimation between them is derived."""

    control_dt: float
    physics_dt: float

    @property
    def decimation(self) -> int:
        return round(self.control_dt / self.physics_dt)


def build_timing_block(robot_config) -> TimingBlock:
    return TimingBlock(
        control_dt=float(robot_config.control_dt),
        physics_dt=1.0 / float(robot_config.simulation_params.mujoco.fps),
    )


@dataclass(frozen=True)
class MotionBlock:
    """The reference-motion horizon the policy reads ahead."""

    future_step_indices: Tuple[int, ...]

    def future_dt_seconds(self, control_dt: float) -> List[float]:
        return [round(s * control_dt, 6) for s in self.future_step_indices]


@dataclass(frozen=True)
class RuntimeBlock:
    """The ONNX graph's own names and the host keys behind them.

    ``passthrough_keys`` are observation inputs folded into constants during
    tracing, so a host does not feed them; ``obs_context_keys`` is every host
    key the observation stage reads. These keep the ``_runtime`` block the
    sidecar already carries.
    """

    onnx_in_names: Tuple[str, ...]
    onnx_out_names: Tuple[str, ...]
    onnx_name_to_in_key: Dict[str, str]
    passthrough_keys: Tuple[str, ...]
    obs_context_keys: Tuple[str, ...]


@dataclass(frozen=True)
class DeploymentContract:
    """The whole sidecar: inputs, outputs, robot, control, timing, motion, sensors.

    Construction validates the contract against itself; :meth:`to_yaml` writes
    today's key layout plus the one ``schema`` key; :meth:`from_yaml` reads the
    schema first and refuses a file it does not recognise before rebuilding the
    typed blocks.
    """

    schema: str
    inputs: Tuple[InputSpec, ...]
    outputs: Tuple[OutputSpec, ...]
    robot: RobotBlock
    control: ControlBlock
    timing: TimingBlock
    motion: MotionBlock
    runtime: RuntimeBlock
    sensors: Optional[SensorDeploymentMetadata]
    checkpoint: str
    control_type: str
    onnx_sha256: Optional[str] = None
    masked_mimic: Optional[GoalCanvas] = None
    # The host the export targets; every input kind must be one that host can
    # feed. Written to the sidecar only for a simulation host (omitted means
    # robot), so a robot sidecar is byte-identical to before the field existed.
    host: str = ROBOT_HOST

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"contract schema {self.schema!r} is not {SCHEMA!r}")
        _check_host(self.host)
        for spec in self.inputs:
            if not host_can_feed(self.host, spec.kind):
                raise ValueError(
                    f"input {spec.key!r} has host kind {spec.kind!r}, which only a "
                    f"{accepting_host(spec.kind)} host can feed; this contract is "
                    f"for a {self.host} host"
                )

        goal_bodies = (
            None
            if self.masked_mimic is None
            else self.masked_mimic.conditionable_bodies
        )
        if self.robot.conditionable_body_names != goal_bodies:
            raise ValueError(
                "robot.conditionable_body_names must match "
                "masked_mimic.conditionable_bodies in the same order; got "
                f"{self.robot.conditionable_body_names!r} and {goal_bodies!r}"
            )

        seen = set()
        for spec in self.inputs:
            if spec.onnx_name in seen:
                raise ValueError(f"duplicate ONNX input name {spec.onnx_name!r}")
            seen.add(spec.onnx_name)
        out_seen = set()
        for spec in self.outputs:
            if spec.onnx_name in out_seen:
                raise ValueError(f"duplicate ONNX output name {spec.onnx_name!r}")
            out_seen.add(spec.onnx_name)

        output_names = {spec.onnx_name for spec in self.outputs}
        for spec in self.inputs:
            if not spec.kind or not spec.provider:
                raise ValueError(
                    f"input {spec.key!r} has no deployment kind or provider; "
                    f"declare deploy= on its FieldPath"
                )
            if spec.history is not None and spec.history.output_key not in output_names:
                raise ValueError(
                    f"history input {spec.key!r} feeds back "
                    f"{spec.history.output_key!r}, which the graph does not "
                    f"produce; outputs: {sorted(output_names)}"
                )
            self._check_sensor_channel(spec)
            if spec.observation_history is not None:
                history = spec.observation_history
                if spec.history is not None:
                    raise ValueError(
                        f"{spec.key!r} cannot replay both actions and measurements"
                    )
                if history.reset_fill != "initial_measurement":
                    raise ValueError(
                        f"observation history for {spec.key!r} must reset_fill "
                        f"'initial_measurement', got {history.reset_fill!r}"
                    )
                if not history.steps_back:
                    raise ValueError(
                        f"observation history for {spec.key!r} has no steps_back"
                    )
                if min(history.steps_back) < 1:
                    raise ValueError(
                        f"observation history for {spec.key!r} has a steps_back "
                        f"below 1: {history.steps_back}"
                    )
                if len(spec.shape) < 2 or spec.shape[1] != len(history.steps_back):
                    raise ValueError(
                        f"observation history shape mismatch for {spec.key!r}: "
                        f"shape {spec.shape} but {len(history.steps_back)} "
                        f"steps_back ({history.steps_back})"
                    )
        if (
            any(spec.provider == MASKED_MIMIC for spec in self.inputs)
            and self.masked_mimic is None
        ):
            raise ValueError(
                "masked_mimic inputs require goal ordering in the masked_mimic block"
            )
        if self.masked_mimic is not None:
            self.masked_mimic.validate(
                frozenset(spec.key for spec in self.inputs), getup=False
            )

        if self.robot.anchor_body_name not in self.robot.body_names:
            raise ValueError(
                f"anchor body {self.robot.anchor_body_name!r} is not in the "
                f"robot's body list"
            )

        n = len(self.robot.joint_names)
        self._check_joint_vector("stiffness", self.control.stiffness, n)
        self._check_joint_vector("damping", self.control.damping, n)
        if self.control.effort_limits is None or any(
            v is None for v in self.control.effort_limits
        ):
            raise ValueError("control.effort_limits must be set for every joint")
        self._check_joint_vector("effort_limits", self.control.effort_limits, n)
        if self.control.action_scale is not None:
            self._check_joint_vector("action_scale", self.control.action_scale, n)
        if self.control.pd_action_offset is not None:
            self._check_joint_vector(
                "pd_action_offset", self.control.pd_action_offset, n
            )

    def _check_joint_vector(self, name: str, values, n: int) -> None:
        if len(values) != n:
            raise ValueError(
                f"control.{name} has {len(values)} entries for " f"{n} joints"
            )

    def _check_sensor_channel(self, spec: InputSpec) -> None:
        if spec.provider not in OPTIONAL_SENSOR_KINDS:
            return
        channel = getattr(self.sensors, spec.provider, None) if self.sensors else None
        if channel is None:
            raise ValueError(
                f"input {spec.key!r} is supplied by the {spec.provider!r} sensor, "
                f"which the sensors block does not carry; export it only when the "
                f"actor reads that channel"
            )

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        """Return the public sidecar representation, including host semantics."""
        robot, control, timing, motion = (
            self.robot,
            self.control,
            self.timing,
            self.motion,
        )
        return {
            "schema": self.schema,
            "type": "unified_pipeline",
            # Only for a simulation host; omitted (not "robot") otherwise so the
            # robot sidecar stays byte-identical.
            **({} if self.host == ROBOT_HOST else {"host": self.host}),
            "dt": timing.control_dt,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "default_joint_stiffness": list(control.stiffness),
            "default_joint_damping": list(control.damping),
            "policy_inputs": [_input_dict(spec) for spec in self.inputs],
            "policy_outputs": [_output_dict(spec) for spec in self.outputs],
            "_runtime": {
                "onnx_in_names": list(self.runtime.onnx_in_names),
                "onnx_out_names": list(self.runtime.onnx_out_names),
                "onnx_name_to_in_key": dict(self.runtime.onnx_name_to_in_key),
                "passthrough_keys": list(self.runtime.passthrough_keys),
                "obs_context_keys": list(self.runtime.obs_context_keys),
            },
            "metadata": {
                "checkpoint": self.checkpoint,
                "control_type": self.control_type,
                "onnx_sha256": self.onnx_sha256,
            },
            "sensors": asdict(self.sensors) if self.sensors is not None else None,
            **(
                {
                    "masked_mimic": {
                        **asdict(self.masked_mimic),
                        "future_steps": list(self.masked_mimic.future_steps),
                        "conditionable_bodies": list(
                            self.masked_mimic.conditionable_bodies
                        ),
                    }
                }
                if self.masked_mimic is not None
                else {}
            ),
            "robot": {
                "mjcf_path": robot.mjcf_path,
                "num_bodies": len(robot.body_names),
                "num_dofs": len(robot.joint_names),
                "anchor_body_name": robot.anchor_body_name,
                "anchor_body_index": robot.anchor_body_index,
                "root_body_name": robot.root_body_name,
                "root_body_index": robot.root_body_index,
                "body_names": list(robot.body_names),
                "joint_names": list(robot.joint_names),
                # Only for a masked-mimic export; omitted (not null) otherwise so
                # the plain tracker sidecar stays byte-identical. The host indexes
                # the body submask and style-condition mask lanes by this order.
                **(
                    {}
                    if robot.conditionable_body_names is None
                    else {
                        "conditionable_body_names": list(robot.conditionable_body_names)
                    }
                ),
            },
            "control": {
                "stiffness": list(control.stiffness),
                "damping": list(control.damping),
                "effort_limits": (
                    None
                    if control.effort_limits is None
                    else list(control.effort_limits)
                ),
                "action_scale": (
                    None if control.action_scale is None else list(control.action_scale)
                ),
                "pd_action_offset": (
                    None
                    if control.pd_action_offset is None
                    else list(control.pd_action_offset)
                ),
                "pd_target_max_accel": control.pd_target_max_accel,
                "action_fn": control.action_fn,
                # Kept at today's constant so the sidecar stays a strict superset
                # until an after-handover decision removes the host-side filter.
                "action_ema_alpha": 1.0,
            },
            "timing": {
                "control_dt": timing.control_dt,
                "physics_dt": timing.physics_dt,
                "decimation": timing.decimation,
            },
            "motion": {
                "future_step_indices": list(motion.future_step_indices),
                "future_dt_seconds": motion.future_dt_seconds(timing.control_dt),
            },
        }

    def to_yaml(self, path) -> Path:
        """Write the contract to ``path`` with ``schema`` first, today's layout."""
        path = Path(path)
        with open(path, "w") as handle:
            yaml.dump(self.to_dict(), handle, default_flow_style=None, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path) -> "DeploymentContract":
        """Rebuild the contract from a sidecar, refusing an unknown schema first."""
        path = Path(path)
        with open(path, "r") as handle:
            content = yaml.safe_load(handle)
        found = content.get("schema")
        if found != SCHEMA:
            raise ValueError(
                f"{path}: sidecar schema {found!r} is not {SCHEMA!r}; re-export "
                f"the checkpoint with deployment/export_to_onnx.py"
            )
        return cls(
            schema=found,
            inputs=tuple(_input_from_dict(d) for d in content["policy_inputs"]),
            outputs=tuple(_output_from_dict(d) for d in content["policy_outputs"]),
            robot=_robot_from_dict(content["robot"]),
            control=_control_from_dict(content["control"]),
            timing=_timing_from_dict(content["timing"]),
            motion=MotionBlock(
                future_step_indices=tuple(content["motion"]["future_step_indices"])
            ),
            runtime=_runtime_from_dict(content["_runtime"]),
            sensors=_sensors_from_dict(content["sensors"]),
            checkpoint=content["metadata"]["checkpoint"],
            control_type=content["metadata"]["control_type"],
            onnx_sha256=content["metadata"].get("onnx_sha256"),
            host=content.get("host", ROBOT_HOST),
            masked_mimic=(
                GoalCanvas(
                    **{
                        key: value
                        for key, value in content["masked_mimic"].items()
                        if key not in ("future_steps", "conditionable_bodies")
                    },
                    future_steps=tuple(content["masked_mimic"]["future_steps"]),
                    conditionable_bodies=tuple(
                        content["masked_mimic"]["conditionable_bodies"]
                    ),
                )
                if content.get("masked_mimic") is not None
                else None
            ),
        )


def _input_dict(spec: InputSpec) -> dict:
    entry: dict = {
        "name": spec.onnx_name,
        "key": spec.key,
        "shape": list(spec.shape),
        "kind": spec.kind,
        "provider": spec.provider,
        "source_namespace": spec.source_namespace,
        "dtype": spec.dtype,
    }
    if spec.history is not None:
        entry["output_key"] = spec.history.output_key
        entry["history"] = spec.history.slots
        entry["first_action_lag"] = spec.history.first_action_lag
        entry["steps_back"] = list(spec.history.steps_back)
        entry["fill_before_available"] = spec.history.reset_fill
        if spec.source is not None:
            entry["source"] = spec.source
        if spec.history.reset_value is not None:
            entry["reset_value"] = spec.history.reset_value
        return entry
    if spec.element_names:
        entry["element_names"] = [list(names) for names in spec.element_names]
    if spec.observation_history is not None:
        entry["observation_history"] = {
            "current_key": spec.observation_history.current_key,
            "steps_back": list(spec.observation_history.steps_back),
            "reset_fill": spec.observation_history.reset_fill,
        }
    if spec.future_steps is not None:
        entry["future_steps"] = spec.future_steps
    if spec.source is not None:
        entry["source"] = spec.source
    return entry


def _output_dict(spec: OutputSpec) -> dict:
    entry: dict = {
        "name": spec.onnx_name,
        "kind": spec.kind,
        "key": spec.onnx_name,
        "shape": list(spec.shape),
    }
    if spec.joint_names is not None:
        entry["joint_names"] = list(spec.joint_names)
    return entry


def _input_from_dict(entry: dict) -> InputSpec:
    history = None
    if "output_key" in entry:
        # from_yaml rebuilds from the four typed fields; the legacy steps_back
        # and reset_value are read when present but are not required.
        history = HistorySlotSpec(
            output_key=entry["output_key"],
            slots=entry["history"],
            first_action_lag=entry["first_action_lag"],
            reset_fill=entry["fill_before_available"],
            steps_back=tuple(entry.get("steps_back", ())),
            reset_value=entry.get("reset_value"),
        )
    elif "history" in entry:
        raise ValueError(
            f"input {entry['key']!r} has a history slot count but no "
            "output_key: this sidecar predates the windowed-input form and "
            "would otherwise load as a stateless input"
        )
    element_names = tuple(tuple(names) for names in entry.get("element_names", ()))
    return InputSpec(
        onnx_name=entry["name"],
        key=entry["key"],
        source_namespace=entry.get("source_namespace", entry["key"].split(".", 1)[0]),
        kind=entry["kind"],
        provider=entry.get("provider", ""),
        shape=tuple(entry["shape"]),
        dtype=entry.get("dtype", "float32"),
        element_names=element_names,
        future_steps=entry.get("future_steps"),
        history=history,
        source=entry.get("source"),
        observation_history=(
            ObservationHistorySpec(
                current_key=entry["observation_history"]["current_key"],
                steps_back=tuple(entry["observation_history"]["steps_back"]),
                reset_fill=entry["observation_history"]["reset_fill"],
            )
            if "observation_history" in entry
            else None
        ),
    )


def _output_from_dict(entry: dict) -> OutputSpec:
    joint_names = entry.get("joint_names")
    return OutputSpec(
        onnx_name=entry["name"],
        kind=entry["kind"],
        shape=tuple(entry["shape"]),
        joint_names=None if joint_names is None else tuple(joint_names),
    )


def _robot_from_dict(entry: dict) -> RobotBlock:
    conditionable = entry.get("conditionable_body_names")
    return RobotBlock(
        mjcf_path=entry["mjcf_path"],
        body_names=tuple(entry["body_names"]),
        joint_names=tuple(entry["joint_names"]),
        anchor_body_name=entry["anchor_body_name"],
        root_body_name=entry["root_body_name"],
        conditionable_body_names=(
            None if conditionable is None else tuple(conditionable)
        ),
    )


def _control_from_dict(entry: dict) -> ControlBlock:
    def _tuple(value):
        return None if value is None else tuple(value)

    return ControlBlock(
        action_fn=entry["action_fn"],
        stiffness=tuple(entry["stiffness"]),
        damping=tuple(entry["damping"]),
        effort_limits=_tuple(entry["effort_limits"]),
        action_scale=_tuple(entry["action_scale"]),
        pd_action_offset=_tuple(entry["pd_action_offset"]),
        pd_target_max_accel=entry["pd_target_max_accel"],
    )


def _timing_from_dict(entry: dict) -> TimingBlock:
    return TimingBlock(control_dt=entry["control_dt"], physics_dt=entry["physics_dt"])


def _runtime_from_dict(entry: dict) -> RuntimeBlock:
    return RuntimeBlock(
        onnx_in_names=tuple(entry["onnx_in_names"]),
        onnx_out_names=tuple(entry["onnx_out_names"]),
        onnx_name_to_in_key=dict(entry["onnx_name_to_in_key"]),
        passthrough_keys=tuple(entry["passthrough_keys"]),
        obs_context_keys=tuple(entry["obs_context_keys"]),
    )


def _sensors_from_dict(entry: Optional[dict]) -> Optional[SensorDeploymentMetadata]:
    if entry is None:
        return None
    from protomotions.envs.sensors.metadata import (
        BaseVelocityMetadata,
        HeightMetadata,
        IMUMetadata,
        LocalizationMetadata,
    )

    def _opt(cls, value):
        return None if value is None else cls(**value)

    return SensorDeploymentMetadata(
        imu=IMUMetadata(**entry["imu"]),
        anchor_body_name=entry["anchor_body_name"],
        localization=_opt(LocalizationMetadata, entry.get("localization")),
        base_velocity=_opt(BaseVelocityMetadata, entry.get("base_velocity")),
        height=_opt(HeightMetadata, entry.get("height")),
        schema=entry.get("schema", "physical-sensors/v1"),
        joint_coordinates=entry.get("joint_coordinates", "scalar_hinge"),
        w_last=entry.get("w_last", True),
        history_initialization=entry.get(
            "history_initialization", "repeat_first_reading"
        ),
    )


__all__ = [
    "SCHEMA",
    "ROBOT_HOST",
    "SIMULATION_HOST",
    "HOSTS",
    "SIMULATION_ONLY_KINDS",
    "host_can_feed",
    "kinds_for_host",
    "accepting_host",
    "SIMULATOR_TRUTH",
    "REFERENCE_MOTION",
    "POLICY_HISTORY",
    "MASKED_MIMIC",
    "SCENE",
    "SENSOR_KINDS",
    "OPTIONAL_SENSOR_KINDS",
    "ELEMENT_LABELS",
    "element_labels",
    "provider_of",
    "provider_for_path",
    "HistorySlotSpec",
    "ObservationHistorySpec",
    "InputSpec",
    "OutputSpec",
    "RobotBlock",
    "build_robot_block",
    "ControlBlock",
    "build_control_block",
    "TimingBlock",
    "build_timing_block",
    "MotionBlock",
    "RuntimeBlock",
    "DeploymentContract",
]
