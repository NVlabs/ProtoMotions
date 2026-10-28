# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline Metatron v1 task/style parity reference; not a runtime policy adapter."""

from __future__ import annotations

import math
import numpy as np

_ANCHOR_BODY = 16


_CONDITIONABLE_BODIES = np.asarray([16, 13, 7, 24, 32], dtype=np.int64)


_NUM_TARGET_STEPS = 5


_NUM_STYLE_FRAMES = 5


_NUM_BODIES = 33


_STEERING_HEADING_TARGET_TIMES_SECONDS = (0.10, 0.20, 0.30, 0.40, 0.50)


_RIGHT_FOOT_BODY = 13


_LEFT_FOOT_BODY = 7


_LEFT_HAND_BODY = 24


_RIGHT_HAND_BODY = 32


_STEERING_YAW_RATE_MAX_RAD_PER_SECOND = math.pi


_STEERING_FOOT_REORIENTATION_THRESHOLD_RAD = math.radians(30.0)


_STEERING_GAIT_CADENCE_BASE_HZ = 0.83


_STEERING_GAIT_CADENCE_PER_MPS_HZ = 0.93


_STEERING_GAIT_CADENCE_MAX_HZ = 2.0


_STEERING_STEP_APART_BASE_METERS = 0.13


_STEERING_STEP_APART_PER_MPS_METERS = 0.065


_DEPLOYED_STEERING_FORE_AFT_SWING_SCALE = 0.75


_STEERING_STANCE_HALF_WIDTH_BASE_METERS = 0.07


_DEPLOYED_STEERING_STANCE_HALF_WIDTH_STRAFE_METERS = 0.12


_STEERING_STEP_LIFT_METERS = 0.08


_STEERING_STATIONARY_SPEED_EPSILON_MPS = 0.05


def _normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise ValueError("encountered a zero-length quaternion")
    return q / norm


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float32).copy()
    out[..., :3] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize(np.asarray(q, dtype=np.float32))
    v = np.asarray(v, dtype=np.float32)
    qv = np.concatenate((v, np.zeros(v.shape[:-1] + (1,), dtype=np.float32)), axis=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., :3]


def _quat_to_tan_norm(q: np.ndarray) -> np.ndarray:
    q = _normalize(np.asarray(q, dtype=np.float32))
    tangent = _quat_rotate(
        q, np.broadcast_to(np.array([1.0, 0.0, 0.0], np.float32), q.shape[:-1] + (3,))
    )
    normal = _quat_rotate(
        q, np.broadcast_to(np.array([0.0, 0.0, 1.0], np.float32), q.shape[:-1] + (3,))
    )
    return np.concatenate((tangent, normal), axis=-1)


def _heading_quat(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32)


def _wrap_to_pi(angle: float) -> float:
    """Wrap a scalar angle to the same shortest-error range as training."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _steering_gait_cadence(speed: float) -> float:
    """Return the handrail cadence in cycles/second for a planar speed."""

    return min(
        _STEERING_GAIT_CADENCE_MAX_HZ,
        _STEERING_GAIT_CADENCE_BASE_HZ
        + _STEERING_GAIT_CADENCE_PER_MPS_HZ * max(float(speed), 0.0),
    )


def _support_heading_from_feet(
    support_feet: np.ndarray, fallback_heading: float
) -> float:
    """Infer forward from the right-to-left support line.

    The feet define a lateral axis, so rotating that axis -90 degrees gives
    the forward direction.  The line has no sign; choose the orientation
    nearest the current heading to avoid a pi flip when the feet are viewed in
    the opposite order or the robot is already turned.
    """

    support_feet = np.asarray(support_feet, dtype=np.float32)
    if support_feet.shape != (2, 3) or not np.all(np.isfinite(support_feet)):
        raise ValueError("support feet must be a finite (2, 3) array")
    lateral = support_feet[1, :2] - support_feet[0, :2]
    if float(np.linalg.norm(lateral)) <= 1.0e-5:
        return float(fallback_heading)
    support_heading = math.atan2(float(lateral[1]), float(lateral[0])) - 0.5 * math.pi
    if abs(_wrap_to_pi(support_heading - float(fallback_heading))) > 0.5 * math.pi:
        support_heading = _wrap_to_pi(support_heading + math.pi)
    return _wrap_to_pi(support_heading)


def _flatten_foot_yaw_rotations(rotations: np.ndarray) -> np.ndarray:
    """Preserve each foot's world yaw while removing roll and pitch."""

    rotations = np.asarray(rotations, dtype=np.float32)
    if rotations.shape != (2, 4) or not np.all(np.isfinite(rotations)):
        raise ValueError("foot rotations must be a finite (2, 4) array")
    if np.any(np.linalg.norm(rotations, axis=1) <= 1.0e-8):
        raise ValueError("foot rotations must be finite non-zero quaternions")
    rotations = _normalize(rotations)
    flattened = np.empty_like(rotations)
    for index, rotation in enumerate(rotations):
        x, y, z, w = (float(value) for value in rotation)
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        flattened[index] = _heading_quat(yaw)
    return flattened


def _apply_steering_targets(
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    current_anchor_pos: np.ndarray,
    current_heading: float,
    target_heading: float,
    command_velocity_world: np.ndarray,
    upright_anchor_z: float,
    default_body_position: np.ndarray,
    default_body_rotation: np.ndarray,
    gait_phase: float,
    *,
    conditionable_bodies: np.ndarray = _CONDITIONABLE_BODIES,
    stationary_foot_positions: np.ndarray | None = None,
    stationary_foot_rotations: np.ndarray | None = None,
    stationary_anchor_position: np.ndarray | None = None,
    stationary_heading: float | None = None,
    stationary_foot_correction: bool = False,
    hand_offset_override: np.ndarray | None = None,
    hand_rotation_override: np.ndarray | None = None,
) -> None:
    """Build the local steering handrail from sensor-visible state.

    Every slot contains a future torso position/heading and calibrated foot
    positions. While moving, only the final (0.5 s) slot contains a hand
    target, giving the arms freedom to move naturally through the earlier
    horizon. At zero speed, all five slots contain the hand hold so standing
    remains explicit. A Pico hand override makes all five hand targets
    explicit and anchors them to the planned foot midpoint. A stopped controller also
    exposes the planted foot orientations with roll/pitch removed and yaw
    preserved; keeping moving foot orientation hidden matches the trained
    local steering handrail and avoids over-constraining the ankle chain (which can
    show up as an unnatural hip twist). The future
    torso follows the commanded planar velocity; heading slews toward the
    requested heading at a bounded rate. Feet use a deterministic anti-phase
    gait around the calibrated standing offsets and point along the future
    torso heading. The lateral part of the swing fades toward a pure
    side-step so the legs do not drive one foot inward while the other moves
    outward. The hands stay at their calibrated rest offsets. No world map,
    reference motion, contacts, or privileged body state is used.
    At near-zero speed the swing amplitude is suppressed and the support frame
    is latched.  Both feet remain planted for heading changes up to 30 degrees;
    beyond that threshold the calibrated stance is reoriented around the same
    support midpoint so the torso can turn without leaving the feet permanently
    crossed behind it.
    """

    target_pos = np.asarray(target_pos, dtype=np.float32)
    target_rot = np.asarray(target_rot, dtype=np.float32)
    current_anchor_pos = np.asarray(current_anchor_pos, dtype=np.float32)
    command_velocity_world = np.asarray(command_velocity_world, dtype=np.float32)
    default_body_position = np.asarray(default_body_position, dtype=np.float32)
    default_body_rotation = np.asarray(default_body_rotation, dtype=np.float32)
    bodies = np.asarray(conditionable_bodies, dtype=np.int64)
    if target_pos.shape != (_NUM_TARGET_STEPS, _NUM_BODIES, 3):
        raise ValueError("steering target positions have the wrong shape")
    if target_rot.shape != (_NUM_TARGET_STEPS, _NUM_BODIES, 4):
        raise ValueError("steering target rotations have the wrong shape")
    if current_anchor_pos.shape != (3,) or command_velocity_world.shape != (2,):
        raise ValueError("steering anchor/velocity inputs have the wrong shape")
    if default_body_position.shape != (
        _NUM_BODIES,
        3,
    ) or default_body_rotation.shape != (
        _NUM_BODIES,
        4,
    ):
        raise ValueError("steering default pose has the wrong shape")
    if bodies.ndim != 1 or _ANCHOR_BODY not in bodies:
        raise ValueError("steering conditionable bodies must include the anchor")
    if stationary_foot_positions is not None:
        stationary_foot_positions = np.asarray(
            stationary_foot_positions, dtype=np.float32
        )
        if stationary_foot_positions.shape != (2, 3) or not np.all(
            np.isfinite(stationary_foot_positions)
        ):
            raise ValueError("stationary foot positions must be a finite (2, 3) array")
    if stationary_foot_rotations is not None:
        stationary_foot_rotations = np.asarray(
            stationary_foot_rotations, dtype=np.float32
        )
        if (
            stationary_foot_rotations.shape != (2, 4)
            or not np.all(np.isfinite(stationary_foot_rotations))
            or np.any(np.linalg.norm(stationary_foot_rotations, axis=1) <= 1.0e-8)
        ):
            raise ValueError(
                "stationary foot rotations must be finite non-zero (2, 4) quaternions"
            )
        # Keep only support yaw during a hold.  A transient pitch/roll
        # measurement should not become an ankle target while the feet are
        # supposed to remain planted on the floor.
        stationary_foot_rotations = _flatten_foot_yaw_rotations(
            stationary_foot_rotations
        )
    if stationary_anchor_position is not None:
        stationary_anchor_position = np.asarray(
            stationary_anchor_position, dtype=np.float32
        )
        if stationary_anchor_position.shape != (3,) or not np.all(
            np.isfinite(stationary_anchor_position)
        ):
            raise ValueError("stationary anchor position must be a finite 3-vector")
    if stationary_heading is not None and not math.isfinite(float(stationary_heading)):
        raise ValueError("stationary heading must be finite")
    if hand_offset_override is not None:
        hand_offset_override = np.asarray(hand_offset_override, dtype=np.float32)
        if hand_offset_override.shape != (2, 3) or not np.all(
            np.isfinite(hand_offset_override)
        ):
            raise ValueError(
                "steering hand offset override must be a finite (2, 3) array"
            )
    if hand_rotation_override is not None:
        hand_rotation_override = np.asarray(hand_rotation_override, dtype=np.float32)
        if (
            hand_rotation_override.shape != (2, 4)
            or not np.all(np.isfinite(hand_rotation_override))
            or np.any(np.linalg.norm(hand_rotation_override, axis=1) <= 1.0e-8)
        ):
            raise ValueError(
                "steering hand rotation override must be finite non-zero (2, 4) quaternions"
            )
        hand_rotation_override = _normalize(hand_rotation_override)
    values = np.concatenate(
        (
            current_anchor_pos,
            command_velocity_world,
            np.asarray([current_heading, target_heading, upright_anchor_z, gait_phase]),
        )
    )
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(default_body_position))
        or not np.all(np.isfinite(default_body_rotation))
    ):
        raise ValueError("steering emitter inputs must be finite")

    default_anchor_pos = default_body_position[_ANCHOR_BODY]
    default_anchor_rot = _normalize(default_body_rotation[_ANCHOR_BODY])
    relative_position = _quat_rotate(
        _quat_conj(default_anchor_rot), default_body_position - default_anchor_pos
    )
    relative_rotation = _quat_mul(
        np.broadcast_to(_quat_conj(default_anchor_rot), default_body_rotation.shape),
        default_body_rotation,
    )
    speed = float(np.linalg.norm(command_velocity_world))
    move_heading = (
        math.atan2(float(command_velocity_world[1]), float(command_velocity_world[0]))
        if speed > _STEERING_STATIONARY_SPEED_EPSILON_MPS
        else float(current_heading)
    )
    cadence = _steering_gait_cadence(speed)
    swing_amplitude = (
        0.0
        if speed <= _STEERING_STATIONARY_SPEED_EPSILON_MPS
        else _STEERING_STEP_APART_BASE_METERS
        + _STEERING_STEP_APART_PER_MPS_METERS * speed
    )
    relative_heading = _wrap_to_pi(float(target_heading) - float(current_heading))
    body_indices = {int(body): index for index, body in enumerate(bodies)}
    # A latched support frame is supplied once the steering-to-idle brake has
    # reached zero. Treat it as stationary at that point; waiting for the
    # epsilon before latching lets the gait continue moving the support frame.
    stationary = stationary_foot_positions is not None
    stationary_foot_heading = (
        float(stationary_heading)
        if stationary and stationary_heading is not None
        else float(current_heading)
    )
    support_heading = (
        _support_heading_from_feet(stationary_foot_positions, stationary_foot_heading)
        if stationary
        else float(current_heading)
    )

    def apply_hand_rotation(slot: int, heading_rotation: np.ndarray) -> None:
        if hand_rotation_override is None:
            return
        for hand_index, body in enumerate((_LEFT_HAND_BODY, _RIGHT_HAND_BODY)):
            if body not in body_indices:
                continue
            base_rotation = _normalize(
                _quat_mul(heading_rotation, relative_rotation[body])
            )
            target_rot[slot, body] = _normalize(
                _quat_mul(base_rotation, hand_rotation_override[hand_index])
            )

    default_foot_center_relative = 0.5 * (
        relative_position[_RIGHT_FOOT_BODY] + relative_position[_LEFT_FOOT_BODY]
    )
    default_half_width = 0.5 * abs(
        float(
            relative_position[_LEFT_FOOT_BODY][1]
            - relative_position[_RIGHT_FOOT_BODY][1]
        )
    )
    calibrated_half_width = max(
        _STEERING_STANCE_HALF_WIDTH_BASE_METERS, default_half_width
    )
    for slot, time_offset in enumerate(_STEERING_HEADING_TARGET_TIMES_SECONDS):
        yaw = float(current_heading) + float(
            np.clip(
                relative_heading,
                -_STEERING_YAW_RATE_MAX_RAD_PER_SECOND * time_offset,
                _STEERING_YAW_RATE_MAX_RAD_PER_SECOND * time_offset,
            )
        )
        heading_rotation = _heading_quat(yaw)
        # Deployed steering keeps a modest upright bias using only the current
        # lowest FK point as its local floor proxy. The clean-profile emitter is
        # separate and retains its original measured-height behavior.
        if stationary:
            # In a standstill, the feet are the support frame. Rebuilding the
            # torso from the live anchor would turn IMU sway into a whole-body
            # target translation and keeps the robot moving indefinitely.
            if stationary_anchor_position is not None:
                torso = stationary_anchor_position.copy()
            else:
                support_center = stationary_foot_positions.mean(axis=0)
                torso = support_center - _quat_rotate(
                    heading_rotation, default_foot_center_relative
                )
        else:
            torso = np.asarray(
                [
                    current_anchor_pos[0] + command_velocity_world[0] * time_offset,
                    current_anchor_pos[1] + command_velocity_world[1] * time_offset,
                    float(upright_anchor_z),
                ],
                dtype=np.float32,
            )
        target_pos[slot, _ANCHOR_BODY] = torso
        target_rot[slot, _ANCHOR_BODY] = heading_rotation
        # During locomotion the trained handrail conditions foot position only;
        # exposing ankle orientation there over-constrains the leg chain and
        # can make the hip twist while stepping. During a standstill, preserve
        # the measured world-frame foot yaw captured with the support
        # positions, with roll/pitch flattened. This is a true planted-foot
        # hold, not a snap to the default-pose orientation.
        reorient_feet = stationary and (
            stationary_foot_correction
            or abs(_wrap_to_pi(float(yaw) - support_heading))
            > _STEERING_FOOT_REORIENTATION_THRESHOLD_RAD
        )
        if stationary and stationary_foot_rotations is not None and not reorient_feet:
            target_rot[slot, _RIGHT_FOOT_BODY] = stationary_foot_rotations[0]
            target_rot[slot, _LEFT_FOOT_BODY] = stationary_foot_rotations[1]
        else:
            foot_heading = (
                float(yaw)
                if reorient_feet
                else (stationary_foot_heading if stationary else float(yaw))
            )
            foot_heading_rotation = _heading_quat(foot_heading)
            for body in (_RIGHT_FOOT_BODY, _LEFT_FOOT_BODY):
                if body in body_indices:
                    target_rot[slot, body] = _normalize(
                        _quat_mul(foot_heading_rotation, relative_rotation[body])
                    )
        # Keep both arms at the calibrated rest pose. While moving, the target
        # is rebuilt below from the planned foot support after the gait targets
        # exist. At zero speed the measured feet already define that support.
        if hand_offset_override is None and stationary:
            for body in (_LEFT_HAND_BODY, _RIGHT_HAND_BODY):
                if body in body_indices:
                    hand_relative_to_support = (
                        relative_position[body] - default_foot_center_relative
                    )
                    target_pos[slot, body] = stationary_foot_positions.mean(
                        axis=0
                    ) + _quat_rotate(heading_rotation, hand_relative_to_support)
                    # A stopped steering hold exposes the calibrated hand
                    # orientation at every horizon.
                    target_rot[slot, body] = _normalize(
                        _quat_mul(heading_rotation, relative_rotation[body])
                    )
        if stationary:
            if reorient_feet:
                support_center = stationary_foot_positions.mean(axis=0)
                for body in (_RIGHT_FOOT_BODY, _LEFT_FOOT_BODY):
                    if body not in body_indices:
                        continue
                    foot_offset = relative_position[body] - default_foot_center_relative
                    target_pos[slot, body] = support_center + _quat_rotate(
                        heading_rotation, foot_offset
                    )
                    target_rot[slot, body] = _normalize(
                        _quat_mul(heading_rotation, relative_rotation[body])
                    )
            else:
                target_pos[slot, _RIGHT_FOOT_BODY] = stationary_foot_positions[0]
                target_pos[slot, _LEFT_FOOT_BODY] = stationary_foot_positions[1]
            if hand_offset_override is not None:
                hand_support = stationary_foot_positions.mean(axis=0)
                hand_support[2] = float(np.min(stationary_foot_positions[:, 2]))
                for hand_index, body in enumerate((_LEFT_HAND_BODY, _RIGHT_HAND_BODY)):
                    if body not in body_indices:
                        continue
                    hand_relative_to_feet = (
                        relative_position[body] - default_foot_center_relative
                    )
                    target_pos[slot, body] = hand_support + _quat_rotate(
                        heading_rotation,
                        hand_relative_to_feet + hand_offset_override[hand_index],
                    )
                    target_rot[slot, body] = _normalize(
                        _quat_mul(heading_rotation, relative_rotation[body])
                    )
            apply_hand_rotation(slot, heading_rotation)
            continue
        for body, sign in ((_RIGHT_FOOT_BODY, -1.0), (_LEFT_FOOT_BODY, 1.0)):
            if body not in body_indices:
                continue
            # Anti-phase swing is useful along travel, but a full lateral
            # swing moves one foot toward the other during a side-step. Fade
            # that component as travel becomes perpendicular to the torso
            # heading. The wider side stance remains the only separation
            # bias. This only changes requested targets and does not
            # infer velocity or displacement from the robot.
            lateral_swing_scale = (
                min(
                    1.0,
                    _DEPLOYED_STEERING_STANCE_HALF_WIDTH_STRAFE_METERS
                    / swing_amplitude,
                )
                * abs(math.cos(_wrap_to_pi(move_heading - yaw)))
                if swing_amplitude > 0.0
                else 0.0
            )
            offset = relative_position[body].copy()
            offset[1] = sign * (
                calibrated_half_width
                + _DEPLOYED_STEERING_STANCE_HALF_WIDTH_STRAFE_METERS
                * abs(math.sin(_wrap_to_pi(move_heading - yaw)))
            )
            phase = float(gait_phase) + 2.0 * math.pi * cadence * time_offset
            if body == _LEFT_FOOT_BODY:
                phase += math.pi
            swing_axis = np.asarray(
                [
                    _DEPLOYED_STEERING_FORE_AFT_SWING_SCALE
                    * math.cos(move_heading - yaw),
                    lateral_swing_scale * math.sin(move_heading - yaw),
                ],
                dtype=np.float32,
            )
            offset[:2] += swing_amplitude * math.cos(phase) * swing_axis
            if swing_amplitude > 0.0:
                offset[2] += _STEERING_STEP_LIFT_METERS * max(0.0, -math.sin(phase))
            target_pos[slot, body] = torso + _quat_rotate(heading_rotation, offset)

        # Ordinary Xbox steering and Pico use the same planned-foot support.
        # Xbox exposes these calibrated down-hand positions only at the final
        # 0.5 s mask; Pico adds controller offsets and exposes all horizons.
        # Neither target follows live or projected torso sway.
        if _RIGHT_FOOT_BODY in body_indices and _LEFT_FOOT_BODY in body_indices:
            hand_support = 0.5 * (
                target_pos[slot, _RIGHT_FOOT_BODY] + target_pos[slot, _LEFT_FOOT_BODY]
            )
            hand_support[2] = float(
                min(
                    target_pos[slot, _RIGHT_FOOT_BODY, 2],
                    target_pos[slot, _LEFT_FOOT_BODY, 2],
                )
            )
        else:
            hand_support = torso + _quat_rotate(
                heading_rotation, default_foot_center_relative
            )
        for hand_index, body in enumerate((_LEFT_HAND_BODY, _RIGHT_HAND_BODY)):
            if body not in body_indices:
                continue
            hand_relative_to_feet = (
                relative_position[body] - default_foot_center_relative
            )
            hand_delta = (
                hand_offset_override[hand_index]
                if hand_offset_override is not None
                else np.zeros(3, dtype=np.float32)
            )
            target_pos[slot, body] = hand_support + _quat_rotate(
                heading_rotation,
                hand_relative_to_feet + hand_delta,
            )
        apply_hand_rotation(slot, heading_rotation)


def _steering_target_masks(
    *,
    conditionable_bodies: np.ndarray = _CONDITIONABLE_BODIES,
    stationary: bool = False,
    feet_active: bool = True,
    feet_rotation_active: bool = True,
    torso_height: bool = False,
    hand_hold_all_horizons: bool = False,
    hand_rotation_hold_all_horizons: bool = False,
    hand_condition_active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the five-slot anchor + foot-position/hand steering contract.

    Moving steering keeps foot orientation hidden, matching the trained
    position-only handrail. Hand position is visible only at the terminal 0.5 s
    slot while moving, or at all slots when ``hand_hold_all_horizons`` is set.
    Stationary steering can additionally expose planted foot orientations and
    the anchor's Z submask because its target is reconstructed from the
    grounded foot support frame. PICO passes ``feet_rotation_active=False``:
    it fixes support position while allowing the policy to choose a natural
    ankle orientation. Stationary hand position/orientation is visible at all
    slots. When ``hand_condition_active`` is supplied it is authoritative:
    only actively clutched Pico hands expose pose goals.
    """

    bodies = np.asarray(conditionable_bodies, dtype=np.int64)
    masks = np.zeros((_NUM_TARGET_STEPS, len(bodies), 2), dtype=np.float32)
    pose_masks = np.zeros(_NUM_TARGET_STEPS, dtype=np.float32)
    anchor_translation = np.zeros((_NUM_TARGET_STEPS, 2), dtype=np.float32)
    anchor_rotation = np.zeros((_NUM_TARGET_STEPS, 2), dtype=np.float32)
    if hand_condition_active is not None:
        hand_condition_active = np.asarray(hand_condition_active, dtype=bool)
        if hand_condition_active.shape != (2,):
            raise ValueError("hand_condition_active must have shape (2,)")

    anchor_index = int(np.flatnonzero(bodies == _ANCHOR_BODY)[0])
    pose_masks[:] = 1.0
    masks[:, anchor_index, :] = 1.0  # projected torso translation + heading
    for body in (_RIGHT_FOOT_BODY, _LEFT_FOOT_BODY):
        body_matches = np.flatnonzero(bodies == body)
        if body_matches.size and feet_active:
            index = int(body_matches[0])
            masks[:, index, 0] = 1.0  # moving/corrective steering is position-only
            if stationary and feet_rotation_active:
                masks[:, index, 1] = 1.0  # local corrective stance orientation
    for side, body in enumerate((_LEFT_HAND_BODY, _RIGHT_HAND_BODY)):
        body_matches = np.flatnonzero(bodies == body)
        if body_matches.size:
            if hand_condition_active is not None and not hand_condition_active[side]:
                continue
            hand_index = int(body_matches[0])
            hand_slots = (
                slice(None)
                if hand_condition_active is not None
                or stationary
                or hand_hold_all_horizons
                else slice(-1, None)
            )
            masks[hand_slots, hand_index, 0] = 1.0
            if hand_rotation_hold_all_horizons:
                # Pico controller rotations are entry-relative and valid at
                # every hand target horizon. The ordinary moving handrail
                # and stationary idle handrail remain position-only unless
                # this explicit override is set.
                masks[hand_slots, hand_index, 1] = 1.0
    # Match the local steering contract: anchor translation means planar XY;
    # the torso height is carried in the hidden target tensor but is not an
    # active steering objective. Height is reserved for the explicit idle or
    # recovery contracts.
    if not stationary:
        anchor_translation[:, 0] = 1.0
    if stationary or torso_height:
        # Deployed steering keeps the torso upright relative to the lowest
        # deployable FK point. The clean profile leaves torso_height false,
        # preserving its original target contract.
        anchor_translation[:, 1] = 1.0
    anchor_rotation[:, 1] = 1.0  # [full rotation, heading]
    return masks, pose_masks, anchor_translation, anchor_rotation


def _multimodal_style_features(
    style_states: list[dict[str, np.ndarray]],
    conditionable_bodies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode v5 style pose and velocity streams in the target's local frame."""

    if len(style_states) != _NUM_STYLE_FRAMES:
        raise ValueError(f"style requires exactly {_NUM_STYLE_FRAMES} frames")
    bodies = np.asarray(conditionable_bodies, dtype=np.int64)
    pose_frames: list[np.ndarray] = []
    velocity_frames: list[np.ndarray] = []
    for state in style_states:
        if "body_vel" not in state or "body_ang_vel" not in state:
            raise ValueError(
                "the multimodal style contract requires gvs and gavs in the motion file"
            )
        body_pos = np.asarray(state["body_pos"], dtype=np.float32)
        body_rot = _normalize(np.asarray(state["body_rot"], dtype=np.float32))
        body_vel = np.asarray(state["body_vel"], dtype=np.float32)
        body_ang_vel = np.asarray(state["body_ang_vel"], dtype=np.float32)
        anchor_pos = body_pos[_ANCHOR_BODY]
        anchor_rot = body_rot[_ANCHOR_BODY]
        forward = _quat_rotate(
            anchor_rot, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        )
        heading_inv = _quat_conj(
            _heading_quat(math.atan2(float(forward[1]), float(forward[0])))
        )

        selected_pos = body_pos[bodies]
        relative_xy = selected_pos[:, :2] - anchor_pos[:2]
        relative_xy_3d = np.concatenate(
            (relative_xy, np.zeros((len(bodies), 1), dtype=np.float32)), axis=-1
        )
        local_xy = _quat_rotate(heading_inv, relative_xy_3d)[:, :2]
        local_z = selected_pos[:, 2:3] - float(np.min(body_pos[:, 2]))
        local_pos = np.concatenate((local_xy, local_z), axis=-1)
        local_rot = _quat_mul(
            np.broadcast_to(heading_inv, (len(bodies), 4)), body_rot[bodies]
        )
        pose_frames.append(
            np.concatenate((local_pos, _quat_to_tan_norm(local_rot)), axis=-1)
        )

        heading_for_bodies = np.broadcast_to(heading_inv, (len(bodies), 4))
        local_vel = _quat_rotate(heading_for_bodies, body_vel[bodies])
        local_ang_vel = _quat_rotate(heading_for_bodies, body_ang_vel[bodies])
        velocity_frames.append(np.concatenate((local_vel, local_ang_vel), axis=-1))

    return (
        np.stack(pose_frames).reshape(-1).astype(np.float32),
        np.stack(velocity_frames).reshape(-1).astype(np.float32),
    )
