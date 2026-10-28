# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The context vocabulary: every name the environment may publish, and nothing else.

``env.context`` is a read-only dictionary, built once per step, from path
string to value (``protomotions.envs.context.FrozenContext``). This module
declares the names in that dictionary, grouped by namespace, so that bindings
can spell them with IDE support (``EnvContext.current.rigid_body_pos`` is a
``FieldPath`` whose ``.path`` is ``"current.rigid_body_pos"``) and so that the
environment can check, when it is built, that every bound path is declared
and that what each publisher publishes matches its namespace's declaration.

Nothing here computes a value. The classes are never instantiated: a
publisher's ``publish(env_ids)`` returns the dict of names its namespace
declares (see ``base_env/publishers.py`` and the control components), and the
environment merges those dicts into the one context.

A name that not every configuration publishes says what it needs with
``requires=``. The tag is a statement about the robot, its sensors or the
experiment: ``FieldPath(requires="localization")`` is a name only a sensor
suite with a localization body produces; a name with no tag is published by
every configuration.

To make a name deployable -- so the exporter can feed it to a host -- give its
``FieldPath`` a host-kind string: ``FieldPath(deploy="joint_pos")``. The kind is
the host's name for the quantity and must have a row in ``ELEMENT_LABELS``
(``deployment/contract.py``); everything else (shape, dtype, the future and
history axes, the column labels, the sensor channel) is derived at export. Only
the names a shipped recipe or an in-repo test binds carry a string today; a bare
name a deployable observation binds is refused at export, naming the line to add.

The requirement names used below, and what each means:

- ``history``: a bound observation, reward or termination component reads a
  past state, so the history ring keeps at least one slot.
- ``contacts``: the robot declares contact bodies (``robot.contact_bodies``),
  so the simulator reports contacts.
- ``sensors``: the experiment declares a sensor suite (``config.sensors``).
- ``localization``: the sensor suite has a localization body, so it can place
  the robot in the world.
- ``base_velocity``: the sensor suite has the base-velocity sensor.
- ``height_sensor``: the sensor suite has the height sensor.
- ``terrain_heightfield``: the terrain has height points (every real terrain
  does; only flat test doubles do not).
- ``objects``: the scene library has scene objects (``scene_lib.num_scenes()``
  is greater than 0). Only the built scene library can tell, so the
  environment checks these names when it is built.
- ``pointclouds``: the scene library computed object pointclouds
  (``SceneLib.has_pointclouds``); checked when the environment is built, as
  above.
- ``motion_file``: the motion file carries the quantity. Only the loaded
  motion library can tell, so the environment checks these names when it is
  built, not from the configs.
- Any other tag is a switch in the config of the control component that
  publishes the namespace, read as ``config.<tag>`` being truthy (``style``,
  ``velocity_command``, ``reference_height_smoothing_seconds`` ...).

``context_requirements.py`` holds the table that turns a configuration into
these facts and the one rule that decides whether a path is published.

A ``NestedField`` tagged ``owner="control"`` is a control component's
namespace (declared with :func:`control_namespace`); it exists exactly when a
configured control component lists it in its ``namespaces``, and
``CONTROL_NAMESPACES`` is derived from that tag, not listed by hand.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from torch import Tensor

from protomotions.envs.context_paths import FieldPath, NestedField, declared_fields


def control_namespace(view_class: type) -> NestedField:
    """Declare a control component's namespace: ``NestedField(view_class,
    owner="control")``. It carries no ``requires``: the namespace exists when a
    configured control component declares it in its ``namespaces``."""
    return NestedField(view_class, owner="control")


class BodyStateView:
    """What every state source carries at every step: the body orientations
    and angular velocities, the joint state, and the root and anchor slices
    derived from them. Positions and linear velocities are not here because a
    sensor suite may not measure them."""

    rigid_body_rot: Tensor = FieldPath(deploy="body_rot")
    rigid_body_ang_vel: Tensor = FieldPath()
    dof_pos: Tensor = FieldPath(deploy="joint_pos")
    dof_vel: Tensor = FieldPath(deploy="joint_vel")
    anchor_idx: int = FieldPath()
    root_rot: Tensor = FieldPath()
    root_ang_vel: Tensor = FieldPath()
    root_local_ang_vel: Tensor = FieldPath(deploy="local_root_ang_vel")
    anchor_rot: Tensor = FieldPath(deploy="anchor_rot")
    anchor_ang_vel: Tensor = FieldPath()
    anchor_local_ang_vel: Tensor = FieldPath()


class SensorStateView(BodyStateView):
    """A sensor source at the current step (``clean``, ``noisy``): what the
    suite measures. Body positions need localization, body velocities the
    base-velocity sensor, the ground height under the root both localization
    and the height sensor. Sensors never report contacts or joint forces."""

    rigid_body_pos: Tensor = FieldPath(requires="localization", deploy="body_pos")
    root_pos: Tensor = FieldPath(requires="localization")
    anchor_pos: Tensor = FieldPath(requires="localization", deploy="anchor_pos")
    root_height: Tensor = FieldPath(requires="localization")
    # The episode's odometer: anchor XY at the episode start, the inverse of
    # the start heading, and the displacement since, in that heading frame.
    odom_start_xy: Tensor = FieldPath(requires="localization", deploy="odom_start_xy")
    odom_start_heading_inv: Tensor = FieldPath(
        requires="localization", deploy="odom_start_heading_inv"
    )
    odom_disp_start: Tensor = FieldPath(
        requires="localization", deploy="odom_disp_start"
    )
    rigid_body_vel: Tensor = FieldPath(requires="base_velocity")
    root_vel: Tensor = FieldPath(requires="base_velocity")
    anchor_vel: Tensor = FieldPath(requires="base_velocity")
    ground_heights: Tensor = FieldPath(requires=("localization", "height_sensor"))


class PrivilegedStateView(BodyStateView):
    """Simulator truth at the current step (``current``): every body quantity,
    the ground height under the anchor and the joint forces, always; the
    contact flags when the robot declares contact bodies."""

    rigid_body_pos: Tensor = FieldPath()
    rigid_body_vel: Tensor = FieldPath()
    ground_heights: Tensor = FieldPath()
    dof_forces: Tensor = FieldPath()
    root_pos: Tensor = FieldPath()
    root_vel: Tensor = FieldPath()
    root_height: Tensor = FieldPath()
    anchor_pos: Tensor = FieldPath(deploy="anchor_pos")
    anchor_vel: Tensor = FieldPath()
    rigid_body_contacts: Tensor = FieldPath(requires="contacts")


class CommandHistoryView:
    """The commands that led to each past state (slots 1 onward)."""

    raw_actions: Tensor = FieldPath(deploy="last_actions")
    processed_actions: Tensor = FieldPath(deploy="last_actions")


class PrivilegedHistoricalView(PrivilegedStateView, CommandHistoryView):
    """Past simulator truth (``historical``) with the command history.
    ``body_contacts`` is the past contact flags under the name the historical
    observation kernels bind."""

    body_contacts: Tensor = FieldPath(requires="contacts")


class SensorHistoricalView(SensorStateView, CommandHistoryView):
    """Past sensor estimates (``clean_historical``, ``noisy_historical``) with
    the command history; nothing more, since sensors carry no contacts."""


class RobotStateView:
    """The reference ``RobotState`` tensors under ``mimic.ref_state``: the six
    every motion file has, and two that only some files carry."""

    rigid_body_pos: Tensor = FieldPath(deploy="reference_motion_body_pos")
    rigid_body_rot: Tensor = FieldPath()
    rigid_body_vel: Tensor = FieldPath()
    rigid_body_ang_vel: Tensor = FieldPath()
    dof_pos: Tensor = FieldPath()
    dof_vel: Tensor = FieldPath()
    rigid_body_contacts: Tensor = FieldPath(requires="motion_file")
    local_rigid_body_rot: Tensor = FieldPath(requires="motion_file")


class MimicContext:
    """Reference motion: the current reference pose and the future targets."""

    ref_state: RobotStateView = NestedField(RobotStateView)
    future_pos: Tensor = FieldPath(deploy="reference_motion_body_pos_future")
    future_rot: Tensor = FieldPath(deploy="reference_motion_body_rot")
    future_vel: Tensor = FieldPath()
    future_ang_vel: Tensor = FieldPath()
    future_dof_pos: Tensor = FieldPath(deploy="reference_motion_joint_pos")
    future_dof_vel: Tensor = FieldPath(deploy="reference_motion_joint_vel")
    anchor_idx: int = FieldPath()
    ref_lr: Tensor = FieldPath()
    future_root_pos: Tensor = FieldPath()
    future_root_rot: Tensor = FieldPath()
    future_root_vel: Tensor = FieldPath()
    future_root_ang_vel: Tensor = FieldPath()
    ref_anchor_pos: Tensor = FieldPath(deploy="reference_motion_anchor_pos")
    ref_anchor_rot: Tensor = FieldPath()
    ref_anchor_vel: Tensor = FieldPath()
    ref_anchor_ang_vel: Tensor = FieldPath()
    future_anchor_pos: Tensor = FieldPath(deploy="reference_motion_anchor_pos_future")
    future_anchor_rot: Tensor = FieldPath(deploy="reference_motion_anchor_rot")
    future_anchor_vel: Tensor = FieldPath(deploy="reference_motion_anchor_vel")
    future_anchor_ang_vel: Tensor = FieldPath(deploy="reference_motion_anchor_ang_vel")
    # Only with MimicControlConfig.reference_height_smoothing_seconds > 0.
    smoothed_ref_anchor_height: Tensor = FieldPath(
        requires="reference_height_smoothing_seconds"
    )
    # Whether the clip ends within the next step (the clip-done check's input).
    clip_done: Tensor = FieldPath()


class MaskedMimicContext:
    """Sparse conditioning targets -- the operator's prompt for the MaskedMimic
    (Metatron) policy. Each modality beyond the global targets is a switch of
    ``MaskedMimicControlConfig``; a disabled modality is not published.

    The deployable prompt carries ``deploy=`` (provider ``masked_mimic``, the
    operator's prompt stream) in five groups: global targets, the anchor
    submasks, the local targets, the style conditioning and the velocity command.
    ``target_times`` and ``local_target_times`` are not graph inputs and stay bare
    (refused if an actor binds them). Semantics shared across the groups:
    positions are metres; time offsets are seconds remaining to the target, not
    timestamps; raw pose rotations are XYZW quaternions; a raw ``*_pos``/``*_rot``
    tensor carries the full published body axis (the conditionable subset lives in
    its mask, not the shape); the masks' conditionable-body order (and the style
    condition mask's) is ``robot.trackable_bodies_subset``
    (``GoalCanvas.conditionable_bodies``, control/masked_mimic_prompt.py), and a
    mask bit True means visible/active (the publisher defaults False;
    ``pack_goals`` sets True)."""

    # -- Global targets: requested targets in the host's localization frame (the
    # same frame as anchor_pos / root_pos; the graph makes them anchor-relative),
    # heading included. Raw body targets [env, future, bodies, 3] metres /
    # [.., 4] XYZW quaternion.
    ref_pos: Tensor = FieldPath(deploy="masked_mimic_ref_pos")
    ref_rot: Tensor = FieldPath(deploy="masked_mimic_ref_rot")
    target_times: Tensor = FieldPath()  # not a graph input; stays bare
    # Seconds remaining to each future target step.
    time_offsets: Tensor = FieldPath(deploy="masked_mimic_time_offsets")
    # Visibility masks (per step; per step/body/[translation, rotation]).
    target_poses_masks: Tensor = FieldPath(deploy="masked_mimic_target_poses_masks")
    target_bodies_masks: Tensor = FieldPath(deploy="masked_mimic_target_bodies_masks")
    # -- Anchor submasks: order [XY, Z] and [full rotation, heading].
    anchor_translation_mask: Tensor = FieldPath(
        requires="anchor_pose_submasking",
        deploy="masked_mimic_anchor_translation_mask",
    )
    anchor_rotation_mask: Tensor = FieldPath(
        requires="anchor_pose_submasking", deploy="masked_mimic_anchor_rotation_mask"
    )
    # -- Local targets: raw target poses the observation graph makes local (XY
    # relative to the target anchor, height to the target pose's lowest body,
    # rotation with the target-anchor heading removed); the host must NOT localise
    # them again. On when local_target_condition_probability > 0 (``local_stream``).
    local_ref_pos: Tensor = FieldPath(
        requires="local_stream", deploy="masked_mimic_local_ref_pos"
    )
    local_ref_rot: Tensor = FieldPath(
        requires="local_stream", deploy="masked_mimic_local_ref_rot"
    )
    local_target_times: Tensor = FieldPath(requires="local_stream")  # not a graph input
    local_time_offsets: Tensor = FieldPath(
        requires="local_stream", deploy="masked_mimic_local_time_offsets"
    )
    local_target_poses_masks: Tensor = FieldPath(
        requires="local_stream", deploy="masked_mimic_local_target_poses_masks"
    )
    local_target_bodies_masks: Tensor = FieldPath(
        requires="local_stream", deploy="masked_mimic_local_target_bodies_masks"
    )
    # -- Style: already-packed feature arrays (NOT raw poses/quaternions), so the
    # kind carries no labels. Layout: num_frames x conditionable_bodies x
    # STYLE_POSE/VELOCITY_FEATURES_PER_BODY (9 / 6, obs/masked_mimic.py), packed
    # flat by _query_style_samples in control/masked_mimic_control.py.
    style_pose_samples: Tensor = FieldPath(
        requires="style", deploy="masked_mimic_style_pose_samples"
    )
    style_velocity_samples: Tensor = FieldPath(
        requires="style", deploy="masked_mimic_style_velocity_samples"
    )
    style_condition_masks: Tensor = FieldPath(
        requires="style", deploy="masked_mimic_style_condition_masks"
    )
    style_frame_masks: Tensor = FieldPath(
        requires="style", deploy="masked_mimic_style_frame_masks"
    )
    # -- Velocity command: planar [vx, vy].
    average_velocity: Tensor = FieldPath(
        requires="velocity_command", deploy="masked_mimic_average_velocity"
    )
    velocity_command_mask: Tensor = FieldPath(
        requires="velocity_command", deploy="masked_mimic_velocity_command_mask"
    )


class SteeringContext:
    """Heading and speed targets of the steering task."""

    tar_dir: Tensor = FieldPath()
    tar_dir_theta: Tensor = FieldPath()
    tar_speed: Tensor = FieldPath()
    tar_face_dir: Tensor = FieldPath()
    prev_root_pos: Tensor = FieldPath()


class PathContext:
    """Waypoints of the path-following task."""

    tar_pos: Tensor = FieldPath()
    head_pos: Tensor = FieldPath()
    traj_samples: Tensor = FieldPath()
    height_conditioned: bool = FieldPath()
    head_body_id: int = FieldPath()


class TargetContext:
    """The target-reaching task's goal, and what its episode-ending checks read:
    the fall reference height (the ground under the anchor, or the last
    elevated support), the anchor height at the last elevated support and the
    oldest anchor position in the stuck window."""

    tar_pos: Tensor = FieldPath()
    tar_proximity_threshold: float = FieldPath()
    # Only with a TargetControl that has a failure check on (fall, gap or
    # stuck); ``TargetControl.provides`` answers the tag.
    fall_ref_height: Tensor = FieldPath(requires="failure_checks")
    last_support_root_height: Tensor = FieldPath(requires="failure_checks")
    oldest_root_pos: Tensor = FieldPath(requires="failure_checks")


class TerrainContext:
    """Terrain tensors used by component observations."""

    height_points: Tensor = FieldPath()
    height_samples: Tensor = FieldPath()
    # Metres per height-field cell (the terrain config's value), so a kernel
    # over the height field needs no terrain object.
    horizontal_scale: float = FieldPath()


class SceneSurfaceContext:
    """Scene-object surfaces. A configuration without scene objects publishes
    none of these (no zero-width stand-in); the pointclouds need a library
    that computed them."""

    object_pos: Tensor = FieldPath(requires="objects")
    object_rot: Tensor = FieldPath(requires="objects")
    neutral_pointclouds: Tensor = FieldPath(requires="pointclouds")
    object_valid_mask: Tensor = FieldPath(requires="objects")


class EnvContext:
    """Every name in the context map, by namespace; see the module docstring."""

    current: PrivilegedStateView = NestedField(PrivilegedStateView)
    clean: SensorStateView = NestedField(SensorStateView, requires="sensors")
    noisy: SensorStateView = NestedField(SensorStateView, requires="sensors")
    historical: PrivilegedHistoricalView = NestedField(
        PrivilegedHistoricalView, requires="history"
    )
    clean_historical: SensorHistoricalView = NestedField(
        SensorHistoricalView, requires=("sensors", "history")
    )
    noisy_historical: SensorHistoricalView = NestedField(
        SensorHistoricalView, requires=("sensors", "history")
    )

    # Policy commands start at slot 0; state history starts at slot 1. The two
    # action-history windows are the policy's own past actions fed back, the same
    # last_actions host kind the command history under the historical namespaces
    # carries, which the exporter reads as the tracker's history input.
    raw_action_history: Tensor = FieldPath(deploy="last_actions")
    processed_action_history: Tensor = FieldPath(deploy="last_actions")
    # The policy's own last raw action fed back one call ago (the recipe's
    # last_action observation reads this). The exporter treats it as a single-slot
    # action feedback from the "actions" output.
    current_action: Tensor = FieldPath(deploy="last_actions")
    current_processed_action: Tensor = FieldPath()
    previous_action: Tensor = FieldPath(requires="history")
    previous_processed_action: Tensor = FieldPath(requires="history")

    # The rows this context covers: None when it covers every env, the row
    # tensor when it was built for a reset's subset.
    env_ids: Optional[Tensor] = FieldPath()
    ground_heights: Tensor = FieldPath()  # the same object as current.ground_heights
    dt: float = FieldPath()
    progress_buf: Tensor = FieldPath()
    contact_body_ids: Tensor = FieldPath()
    non_termination_contact_body_ids: Tensor = FieldPath()
    terrain: TerrainContext = NestedField(
        TerrainContext, requires="terrain_heightfield"
    )
    scene: SceneSurfaceContext = NestedField(SceneSurfaceContext)
    # PRIVILEGED simulator truth: [env, selected bodies, selected partners, xyz].
    # World-frame normal force on each row body, in N; not friction or a wrench.
    # Never exposed through clean/noisy sensor estimates or hardware export.
    contact_forces: Tensor = FieldPath(requires="contacts")
    # Static name-to-axis maps, not observations or historical state.
    contact_body_index: Dict[str, int] = FieldPath(requires="contacts")
    contact_partner_index: Dict[str, int] = FieldPath(requires="contacts")
    prev_contact_force_magnitudes: Tensor = FieldPath(requires="contacts")

    # Published by the control component of the same task, through the
    # control manager; absent when the config has no such component.
    mimic: MimicContext = control_namespace(MimicContext)
    masked_mimic: MaskedMimicContext = control_namespace(MaskedMimicContext)
    steering: SteeringContext = control_namespace(SteeringContext)
    path: PathContext = control_namespace(PathContext)
    target: TargetContext = control_namespace(TargetContext)


# The top-level names (dt, progress_buf, current_action ...) form the namespace "".
NAMESPACES: Tuple[str, ...] = (
    "",
    *(name for name, attr in vars(EnvContext).items() if isinstance(attr, NestedField)),
)
# The namespaces tagged owner="control" above.
CONTROL_NAMESPACES: Tuple[str, ...] = tuple(
    name
    for name, attr in vars(EnvContext).items()
    if isinstance(attr, NestedField) and attr.owner == "control"
)


def _namespace_field(namespace: str) -> NestedField:
    attr = vars(EnvContext).get(namespace)
    if not isinstance(attr, NestedField):
        raise KeyError(f"'{namespace}' is not a declared context namespace")
    return attr


def declared_paths(namespace: str) -> Dict[str, FieldPath]:
    """The declared names of one namespace, by full path (``""`` = top level).

    Each FieldPath carries the name's own ``requires``; what the namespace as a
    whole needs is :func:`namespace_requires`.
    """
    if namespace == "":
        return {
            name: attr
            for name, attr in vars(EnvContext).items()
            if isinstance(attr, FieldPath)
        }
    _namespace_field(namespace)
    return declared_fields(getattr(EnvContext, namespace))


def namespace_requires(namespace: str) -> Tuple[str, ...]:
    """What a configuration needs for anything to be published under
    ``namespace`` (empty for the top level and for always-published
    namespaces)."""
    return () if namespace == "" else _namespace_field(namespace).requires


def deploy_for(path: str) -> Optional[str]:
    """The host kind a declared context path carries, or ``None``.

    Resolves the path to its declared ``FieldPath`` and returns its ``deploy``
    slot: the one string a researcher writes, the host's name for the quantity.
    Everything else the exporter needs (shape, dtype, the future and history
    axes, the column labels, the sensor channel) is derived at export, not
    declared. A path that is not declared, or carries no host kind, returns
    ``None``, and the exporter refuses to export it.
    """
    namespace = path.split(".", 1)[0] if "." in path else ""
    field = declared_paths(namespace).get(path)
    return None if field is None else field.deploy


def namespace_owner(namespace: str) -> Optional[str]:
    """``"control"`` for a control component's namespace, else None."""
    return None if namespace == "" else _namespace_field(namespace).owner


__all__ = [
    "BodyStateView",
    "CONTROL_NAMESPACES",
    "CommandHistoryView",
    "EnvContext",
    "MaskedMimicContext",
    "MimicContext",
    "NAMESPACES",
    "PathContext",
    "PrivilegedHistoricalView",
    "PrivilegedStateView",
    "RobotStateView",
    "SceneSurfaceContext",
    "SensorHistoricalView",
    "SensorStateView",
    "SteeringContext",
    "TargetContext",
    "TerrainContext",
    "control_namespace",
    "declared_paths",
    "deploy_for",
    "namespace_owner",
    "namespace_requires",
]
