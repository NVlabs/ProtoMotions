# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PICO/XRoboToolkit input reader shared by the MuJoCo and Unitree launchers.

XRoboToolkit exposes controller/headset poses and, when body tracking is
available, a 24-joint body stream in Unity coordinates. This module converts
only the two hand positions into the deployment robot frame. It prefers the
body stream and falls back to left/right controller poses relative to the
headset, so a headset without ankle trackers can still provide teleoperation.

The SDK is optional at import time.  Selecting ``--pico`` gives a clear setup
error when the external XRoboToolkit Python binding is not installed.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation


PICO_A = 1 << 0
PICO_B = 1 << 1
PICO_X = 1 << 2
PICO_Y = 1 << 3
_PICO_CLIENT_LOCK_PREFIX = "protomotions_pico_xrt"


def pico_button_names(buttons: int) -> tuple[str, ...]:
    """Return stable human-readable names for a Pico button bit mask."""

    return tuple(
        name
        for bit, name in (
            (PICO_A, "A"),
            (PICO_B, "B"),
            (PICO_X, "X"),
            (PICO_Y, "Y"),
        )
        if int(buttons) & bit
    )


_NUM_BODY_JOINTS = 24
_ROOT_JOINT = 0
_NECK_JOINT = 12
_LEFT_WRIST_JOINT = 22
_RIGHT_WRIST_JOINT = 23

# Unity: X-right, Y-up, Z-forward (left-handed)
# Robot:  X-forward, Y-left, Z-up (right-handed)
_UNITY_TO_ROBOT = np.asarray(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)

# This is the calibrated headset/controller basis used by the Pico
# reference stream:
# Unity [x, y, z] -> robot [-z, -x, y].  Keep it separate from the body
# tracking basis above because the body path also applies its calibrated root
# correction, while controller teleop is defined by headset yaw only.
_UNITY_TO_HEADSET_ROBOT = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)

# This is the calibrated root correction for the headset/controller frame.
# It is needed
# for the root-local wrist coordinates to use the same robot-facing convention
# as the reference Pico implementation.
_ROOT_OFFSET = SciRotation.from_euler("xyz", [0.0, 0.0, -90.0], degrees=True)


@dataclass(frozen=True)
class PicoSnapshot:
    """Latest valid headset sample in the deployment coordinate contract."""

    timestamp_ns: int
    buttons: int
    axes: tuple[float, float, float, float]
    body_poses: np.ndarray | None
    hand_positions_local: np.ndarray
    hand_positions_world: np.ndarray
    hand_rotations_local: np.ndarray
    neck_position_local: np.ndarray
    tracking_source: str
    headset_heading: float = 0.0
    headset_heading_valid: bool = False
    hand_positions_valid: bool = True
    hand_world_valid: bool = True
    hand_rotations_valid: bool = True
    # XRoboToolkit exposes the non-thumb finger triggers separately from the
    # face buttons. Keep them in the same immutable sample as the poses so a
    # hand command can never be paired with a newer/older input packet.
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0

    @property
    def teleop_pose_valid(self) -> bool:
        """Whether this packet can safely initialize headset teleoperation."""

        return bool(
            self.headset_heading_valid
            and self.hand_positions_valid
            and self.hand_world_valid
            and self.hand_rotations_valid
        )


def _default_client_lock_path() -> Path:
    return Path("/tmp") / f"{_PICO_CLIENT_LOCK_PREFIX}_{os.getuid()}.lock"


def _acquire_client_lock(path: Path) -> int:
    """Hold one cross-process lease for the vendor SDK's global client."""

    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            owner = os.read(descriptor, 64).decode("utf-8", errors="replace").strip()
        finally:
            os.close(descriptor)
        owner_detail = f" (owner pid {owner})" if owner else ""
        raise RuntimeError(
            "another ProtoMotions Pico client is already running"
            f"{owner_detail}; stop it before starting a second deployment"
        ) from exc
    os.ftruncate(descriptor, 0)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def normalize_body_poses(body_poses: Any) -> np.ndarray:
    """Validate and copy the raw XRT ``(24, 7)`` pose array."""

    poses = np.asarray(body_poses, dtype=np.float32)
    if poses.shape != (_NUM_BODY_JOINTS, 7):
        raise ValueError(f"Pico body pose must have shape (24, 7), got {poses.shape}")
    if not np.all(np.isfinite(poses)):
        raise ValueError("Pico body pose contains non-finite values")
    required = poses[[_ROOT_JOINT, _LEFT_WRIST_JOINT, _RIGHT_WRIST_JOINT]]
    # XRT can return an all-zero row while a joint is not tracked.  Root may
    # legitimately be at the origin, so use its quaternion as the validity
    # signal and require a non-zero wrist position or quaternion.
    if np.linalg.norm(required[0, 3:]) <= 1.0e-5:
        raise ValueError("Pico root pose is unavailable")
    for name, row in (
        ("left wrist", required[1]),
        ("right wrist", required[2]),
    ):
        if np.linalg.norm(row[:3]) <= 1.0e-5 and np.linalg.norm(row[3:]) <= 1.0e-5:
            raise ValueError(f"Pico {name} pose is unavailable")
    return poses.copy()


def body_poses_to_local_points(
    body_poses: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(left/right wrist positions, neck position)`` root-local.

    The conversion follows the reference three-point position path:
    Unity positions are first mapped with ``[-x, z, y]`` and then expressed in
    the corrected root frame.  No foot or ankle data is read.
    """

    poses = normalize_body_poses(body_poses)
    robot_positions = poses[:, :3] @ _UNITY_TO_ROBOT.T
    root_rotation_unity = SciRotation.from_quat(poses[_ROOT_JOINT, 3:])
    root_rotation_robot = (
        SciRotation.from_matrix(
            _UNITY_TO_ROBOT @ root_rotation_unity.as_matrix() @ _UNITY_TO_ROBOT.T
        )
        * _ROOT_OFFSET
    )
    root_position = robot_positions[_ROOT_JOINT]
    root_inverse = root_rotation_robot.inv()
    hands = np.stack(
        (
            root_inverse.apply(robot_positions[_LEFT_WRIST_JOINT] - root_position),
            root_inverse.apply(robot_positions[_RIGHT_WRIST_JOINT] - root_position),
        ),
        axis=0,
    ).astype(np.float32)
    neck = root_inverse.apply(robot_positions[_NECK_JOINT] - root_position).astype(
        np.float32
    )
    return hands, neck


def headset_pose_to_robot_yaw(headset_pose: Any) -> float:
    """Return headset yaw after converting Unity orientation to robot axes."""

    value = np.asarray(headset_pose, dtype=np.float32).reshape(-1)
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("Pico headset pose is unavailable")
    if np.linalg.norm(value[3:]) <= 1.0e-5:
        raise ValueError("Pico headset orientation is unavailable")
    headset_rotation_unity = SciRotation.from_quat(value[3:])
    headset_rotation_robot = SciRotation.from_matrix(
        _UNITY_TO_HEADSET_ROBOT
        @ headset_rotation_unity.as_matrix()
        @ _UNITY_TO_HEADSET_ROBOT.T
    )
    return float(headset_rotation_robot.as_euler("xyz")[2])


def controller_poses_to_frame_data(
    left_pose: Any, right_pose: Any, headset_pose: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return headset-local XY/Z and floor-frame controller positions.

    The returned arrays are ordered explicitly as ``left, right``.  The
    controller positions are converted to robot axes, then their XY
    displacement from the headset is rotated by the headset yaw only.  Pitch
    and roll of the headset must not tilt the hand target frame. Controller
    rotations use that same yaw-only headset frame. The second array remains
    in the converted world/floor frame so its Z component can be used as an
    absolute floor-relative height offset.
    """
    poses = []
    for name, pose in (
        ("left controller", left_pose),
        ("right controller", right_pose),
        ("headset", headset_pose),
    ):
        value = np.asarray(pose, dtype=np.float32).reshape(-1)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Pico {name} pose is unavailable")
        if np.linalg.norm(value[3:]) <= 1.0e-5:
            raise ValueError(f"Pico {name} orientation is unavailable")
        poses.append(value)
    left, right, headset = poses
    robot_positions = (
        np.stack((left[:3], right[:3], headset[:3])) @ _UNITY_TO_HEADSET_ROBOT.T
    )
    headset_yaw = headset_pose_to_robot_yaw(headset)
    inverse_yaw = SciRotation.from_euler("z", -headset_yaw)
    headset_position = robot_positions[2]
    controller_rotations_robot = np.stack(
        tuple(
            SciRotation.from_matrix(
                _UNITY_TO_HEADSET_ROBOT
                @ SciRotation.from_quat(value[3:]).as_matrix()
                @ _UNITY_TO_HEADSET_ROBOT.T
            ).as_matrix()
            for value in (left, right)
        ),
        axis=0,
    )
    hand_rotations_local = np.stack(
        tuple(
            SciRotation.from_matrix(
                inverse_yaw.as_matrix() @ controller_rotation
            ).as_quat()
            for controller_rotation in controller_rotations_robot
        ),
        axis=0,
    ).astype(np.float32)
    hands = np.stack(
        (
            inverse_yaw.apply(robot_positions[0] - headset_position),
            inverse_yaw.apply(robot_positions[1] - headset_position),
        ),
        axis=0,
    ).astype(np.float32)
    return (
        hands,
        robot_positions[:2].astype(np.float32),
        hand_rotations_local,
        headset_yaw,
    )


def controller_poses_to_frame_points(
    left_pose: Any, right_pose: Any, headset_pose: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return controller frame positions without the headset heading."""

    hands, hands_world, _, _ = controller_poses_to_frame_data(
        left_pose, right_pose, headset_pose
    )
    return hands, hands_world


def controller_poses_to_local_points(
    left_pose: Any, right_pose: Any, headset_pose: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning ordered headset-local hand positions."""

    hands, _ = controller_poses_to_frame_points(left_pose, right_pose, headset_pose)
    return hands, np.zeros(3, dtype=np.float32)


def _read_pose(xrt: Any, name: str) -> Any:
    getter = getattr(xrt, f"get_{name}_pose", None)
    if getter is None:
        raise ValueError(f"XRoboToolkit has no {name} pose API")
    return getter()


def _read_button(xrt: Any, name: str) -> bool:
    getter = getattr(xrt, f"get_{name}_button", None)
    if getter is None:
        return False
    try:
        return bool(getter())
    except Exception:
        return False


def _read_axis(xrt: Any, name: str) -> tuple[float, float]:
    getter = getattr(xrt, f"get_{name}_axis", None)
    if getter is None:
        return 0.0, 0.0
    try:
        values = np.asarray(getter(), dtype=np.float32).reshape(-1)
        if values.size < 2 or not np.all(np.isfinite(values[:2])):
            return 0.0, 0.0
        return float(np.clip(values[0], -1.0, 1.0)), float(
            np.clip(values[1], -1.0, 1.0)
        )
    except Exception:
        return 0.0, 0.0


def _read_analog(xrt: Any, name: str) -> float:
    """Read one XRoboToolkit analog control."""

    getter = getattr(xrt, f"get_{name}", None)
    if getter is None:
        return 0.0
    try:
        value = float(getter())
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _read_finger_controls(xrt: Any) -> tuple[float, float, float, float]:
    """Return left/right trigger and grip values from one XRT sample."""

    values = tuple(
        _read_analog(xrt, name)
        for name in (
            "left_trigger",
            "right_trigger",
            "left_grip",
            "right_grip",
        )
    )
    return values  # type: ignore[return-value]


def _read_buttons_and_axes(xrt: Any) -> tuple[int, tuple[float, float, float, float]]:
    buttons = 0
    for bit, name in (
        (PICO_A, "A"),
        (PICO_B, "B"),
        (PICO_X, "X"),
        (PICO_Y, "Y"),
    ):
        if _read_button(xrt, name):
            buttons |= bit
    left_x, left_y = _read_axis(xrt, "left")
    right_x, right_y = _read_axis(xrt, "right")
    return buttons, (left_x, left_y, right_x, right_y)


class PicoInput:
    """Background XRT reader with latest-sample semantics.

    XRoboToolkit access is isolated to this reader thread.  The deployment
    control thread only copies the latest validated snapshot, so headset SDK
    polling is not part of the policy inference hot path.
    """

    def __init__(
        self,
        stale_timeout_seconds: float = 0.25,
        *,
        xrt_module: Any | None = None,
        client_lock_path: Path | None = None,
    ) -> None:
        if xrt_module is None:
            try:
                import xrobotoolkit_sdk as xrt
            except ImportError as exc:
                raise RuntimeError(
                    "--pico requires the external XRoboToolkit Python binding; "
                    "install XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 and run "
                    "its PC service first"
                ) from exc
        else:
            xrt = xrt_module
        self._xrt = xrt
        missing_finger_apis = [
            name
            for name in (
                "get_left_trigger",
                "get_right_trigger",
                "get_left_grip",
                "get_right_grip",
            )
            if not callable(getattr(self._xrt, name, None))
        ]
        if missing_finger_apis:
            raise RuntimeError(
                "--pico requires the XRoboToolkit finger-control APIs; "
                f"missing {', '.join(missing_finger_apis)}"
            )
        self._stale_timeout_seconds = float(stale_timeout_seconds)
        if self._stale_timeout_seconds <= 0.0:
            raise ValueError("Pico stale timeout must be positive")
        self._client_lock_path = (
            _default_client_lock_path()
            if client_lock_path is None
            else Path(client_lock_path)
        )
        self._client_lock_fd = _acquire_client_lock(self._client_lock_path)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: PicoSnapshot | None = None
        self._latest_buttons = 0
        self._latest_axes = (0.0, 0.0, 0.0, 0.0)
        self._last_packet_time: float | None = None
        self._last_pose_time: float | None = None
        self._last_stamp_ns: int | None = None
        # Newer XRoboToolkit bindings expose a callback packet counter. A
        # device timestamp is not guaranteed to advance for every state
        # packet, so prefer the counter and retain the timestamp fallback for
        # older installations.
        self._state_count_getter = getattr(self._xrt, "get_state_update_count", None)
        self._last_state_count: int | None = None
        self._previous_buttons = 0
        self._last_error: str | None = None
        self._closed = False
        try:
            self._xrt.init()
        except Exception:
            self._release_client_lock()
            raise
        self._thread = threading.Thread(
            target=self._run, name="pico-input-reader", daemon=True
        )
        self._thread.start()

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def status_message(self) -> str:
        """Return a concise startup/runtime diagnostic without calling the SDK."""

        with self._lock:
            snapshot = self._latest
            last_packet_time = self._last_packet_time
            last_pose_time = self._last_pose_time
            last_error = self._last_error
            state_count = self._last_state_count
        now = time.monotonic()
        if last_packet_time is None:
            message = (
                "local SDK initialized, but no headset state packet has arrived; "
                "check the USB bridge, select PC Service IP 127.0.0.1 in the "
                "headset, and enable Head + Controller + Send"
            )
        else:
            packet_age = max(0.0, now - last_packet_time)
            pose_age = (
                "never"
                if last_pose_time is None
                else f"{max(0.0, now - last_pose_time):.2f}s"
            )
            source = "unknown" if snapshot is None else snapshot.tracking_source
            message = (
                f"packets={state_count or 'timestamp-based'} "
                f"last_packet_age={packet_age:.2f}s source={source} "
                f"teleop_pose_age={pose_age}"
            )
            if snapshot is not None and not snapshot.teleop_pose_valid:
                message += (
                    "; headset/controller pose unavailable—keep both controllers "
                    "awake and enable Head + Controller tracking"
                )
        if last_error:
            message += f"; reader_error={last_error}"
        return message

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                buttons, axes = _read_buttons_and_axes(self._xrt)
                (
                    left_trigger,
                    right_trigger,
                    left_grip,
                    right_grip,
                ) = _read_finger_controls(self._xrt)
                with self._lock:
                    self._latest_buttons = buttons
                    self._latest_axes = axes
                stamp_ns = int(self._xrt.get_time_stamp_ns())
                state_count: int | None = None
                if self._state_count_getter is not None:
                    state_count = int(self._state_count_getter())
                    if state_count <= 0 or state_count == self._last_state_count:
                        time.sleep(0.001)
                        continue
                elif (
                    self._last_stamp_ns is not None and stamp_ns == self._last_stamp_ns
                ):
                    time.sleep(0.001)
                    continue
                body_poses = None
                tracking_source = "controllers"
                hand_positions_valid = True
                hand_world_valid = True
                hand_rotations_valid = True
                headset_heading = 0.0
                headset_heading_valid = False
                headset_pose = None
                try:
                    headset_pose = _read_pose(self._xrt, "headset")
                    headset_heading = headset_pose_to_robot_yaw(headset_pose)
                    headset_heading_valid = True
                except (TypeError, ValueError):
                    pass
                try:
                    # Controller teleop is intentionally preferred over the
                    # optional SMPL/body stream. It gives the exact headset
                    # torso frame requested by the operator and preserves the
                    # explicit left/right controller ordering.
                    if headset_pose is None:
                        headset_pose = _read_pose(self._xrt, "headset")
                    (
                        hands,
                        hands_world,
                        hand_rotations_local,
                        headset_heading,
                    ) = controller_poses_to_frame_data(
                        _read_pose(self._xrt, "left_controller"),
                        _read_pose(self._xrt, "right_controller"),
                        headset_pose,
                    )
                    headset_heading_valid = True
                    neck = np.zeros(3, dtype=np.float32)
                except (TypeError, ValueError):
                    # Body data is retained for diagnostics, but it is not a
                    # substitute for controller-frame teleop: it is pelvis
                    # relative and cannot provide the requested headset XY
                    # reference on a headset without foot trackers.
                    body_poses = None
                    hand_positions_valid = False
                    hand_world_valid = False
                    hand_rotations_valid = False
                    if self._xrt.is_body_data_available():
                        try:
                            body_poses = normalize_body_poses(
                                self._xrt.get_body_joints_pose()
                            )
                            hands, neck = body_poses_to_local_points(body_poses)
                            hands_world = np.zeros((2, 3), dtype=np.float32)
                            hand_rotations_local = (
                                np.broadcast_to([0.0, 0.0, 0.0, 1.0], (2, 4))
                                .copy()
                                .astype(np.float32)
                            )
                            tracking_source = "body"
                        except (TypeError, ValueError):
                            body_poses = None
                    if body_poses is None:
                        # Button fields can remain valid while one controller
                        # pose is temporarily unavailable. Preserve the
                        # safety/mode button stream, but never expose zero
                        # hand positions as teleoperation data.
                        hands = np.zeros((2, 3), dtype=np.float32)
                        hands_world = np.zeros((2, 3), dtype=np.float32)
                        hand_rotations_local = (
                            np.broadcast_to([0.0, 0.0, 0.0, 1.0], (2, 4))
                            .copy()
                            .astype(np.float32)
                        )
                        neck = np.zeros(3, dtype=np.float32)
                        tracking_source = "buttons_only"
                snapshot = PicoSnapshot(
                    timestamp_ns=stamp_ns,
                    buttons=buttons,
                    axes=axes,
                    body_poses=body_poses,
                    hand_positions_local=hands,
                    hand_positions_world=hands_world,
                    hand_rotations_local=hand_rotations_local,
                    neck_position_local=neck,
                    tracking_source=tracking_source,
                    headset_heading=headset_heading,
                    headset_heading_valid=headset_heading_valid,
                    hand_positions_valid=hand_positions_valid,
                    hand_world_valid=hand_world_valid,
                    hand_rotations_valid=hand_rotations_valid,
                    left_trigger=left_trigger,
                    right_trigger=right_trigger,
                    left_grip=left_grip,
                    right_grip=right_grip,
                )
                with self._lock:
                    self._latest = snapshot
                    now = time.monotonic()
                    self._last_packet_time = now
                    if snapshot.teleop_pose_valid:
                        self._last_pose_time = now
                    self._last_stamp_ns = stamp_ns
                    if state_count is not None:
                        self._last_state_count = state_count
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.001)

    def poll(self) -> PicoSnapshot | None:
        """Return the newest sample, or ``None`` until one is valid."""

        with self._lock:
            snapshot = self._latest
            buttons = self._latest_buttons
            axes = self._latest_axes
        if snapshot is None:
            return None
        # All SDK calls stay in the reader thread. Body poses can arrive more
        # slowly than button samples, so expose the newest controls alongside
        # the newest valid body sample without touching XRT from the control
        # thread.
        if buttons == snapshot.buttons and axes == snapshot.axes:
            return snapshot
        return PicoSnapshot(
            timestamp_ns=snapshot.timestamp_ns,
            buttons=buttons,
            axes=axes,
            body_poses=snapshot.body_poses,
            hand_positions_local=snapshot.hand_positions_local,
            hand_positions_world=snapshot.hand_positions_world,
            hand_rotations_local=snapshot.hand_rotations_local,
            neck_position_local=snapshot.neck_position_local,
            tracking_source=snapshot.tracking_source,
            headset_heading=snapshot.headset_heading,
            headset_heading_valid=snapshot.headset_heading_valid,
            hand_positions_valid=snapshot.hand_positions_valid,
            hand_world_valid=snapshot.hand_world_valid,
            hand_rotations_valid=snapshot.hand_rotations_valid,
            left_trigger=snapshot.left_trigger,
            right_trigger=snapshot.right_trigger,
            left_grip=snapshot.left_grip,
            right_grip=snapshot.right_grip,
        )

    def wait_for_sample(
        self,
        timeout_seconds: float | None = None,
        *,
        require_teleop_pose: bool = True,
        progress_interval_seconds: float = 2.0,
        progress_callback: Any | None = None,
    ) -> PicoSnapshot | None:
        """Wait for one fresh headset/controller sample after startup.

        XRoboToolkit connects asynchronously. Starting the MuJoCo viewer
        before the first packet makes the process look permanently stuck in
        damping, even though the input service may connect later. A None
        timeout waits until the operator finishes connecting the headset or
        interrupts the process. Progress callbacks explain which layer is
        still missing instead of reporting the local service as a live device.
        """

        if timeout_seconds is None:
            deadline = None
        else:
            timeout = float(timeout_seconds)
            if not np.isfinite(timeout) or timeout <= 0.0:
                raise ValueError("Pico sample timeout must be finite and positive")
            deadline = time.monotonic() + timeout
        progress_interval = float(progress_interval_seconds)
        if not np.isfinite(progress_interval) or progress_interval <= 0.0:
            raise ValueError("Pico progress interval must be finite and positive")
        next_progress = time.monotonic()
        while deadline is None or time.monotonic() < deadline:
            sample = self.poll()
            ready = sample is not None and not self.timed_out()
            if ready and (not require_teleop_pose or sample.teleop_pose_valid):
                return sample
            now = time.monotonic()
            if progress_callback is not None and now >= next_progress:
                progress_callback(self.status_message())
                next_progress = now + progress_interval
            time.sleep(0.01)
        return None

    def rising_buttons(self, buttons: int) -> int:
        rising = int(buttons) & ~self._previous_buttons
        self._previous_buttons = int(buttons)
        return rising

    def timed_out(self) -> bool:
        with self._lock:
            last_packet_time = self._last_packet_time
            last_pose_time = self._last_pose_time
        # Buttons can continue arriving after a controller falls asleep. Treat
        # pose freshness—not merely packet freshness—as the teleop safety
        # contract so stale hands cannot remain armed indefinitely.
        freshest_required_time = last_pose_time
        return (
            last_packet_time is None
            or freshest_required_time is None
            or time.monotonic() - freshest_required_time > self._stale_timeout_seconds
        )

    def _release_client_lock(self) -> None:
        descriptor = getattr(self, "_client_lock_fd", -1)
        if descriptor < 0:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        self._client_lock_fd = -1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        close = getattr(self._xrt, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._release_client_lock()
