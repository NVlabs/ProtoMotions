# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete controller for idle, location, reaching and steering inference tasks.

Owns task commands, deadlines, keyboard input and optional get-up recovery.
The shared goal packer converts its targets into the policy's observation
format. The controller latches the robot's body positions and rotations from
the state the env hands ``reset_rows`` and ``advance`` (``body_pos``,
``body_rot``): the script setters called between steps, the get-up latch and
the prompt packer read the latched measurement, which is the robot's current
pose then (nothing moves the robot between the env's read and the next
step), so ``publish`` stays a pure read. It owns no episode-ending check: the
parent's clip-done rule is not an inference task's.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple

import torch

from protomotions.envs.control.base import bound_paths
from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.masked_mimic_tasks.goals import BodyGoal, pack_goals
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    MarkerState,
    VisualizationMarkerConfig,
)
from protomotions.utils import rotations
from protomotions.utils.baked_fk import BakedAnchorFK
from protomotions.utils.robot_heading import (
    heading_to_quat_for_forward_axis,
    semantic_forward_xy,
)


@dataclass
class GetupRecoveryConfig:
    """Fallen-robot torso-height goals for every mode; seconds and meters.

    Recovery starts below ``enter_height_fraction`` of the default standing
    torso height and ends above ``exit_height_fraction``. While active, only
    repeated local torso-height goals are shown, ``time_gap_seconds`` apart.
    The first deadline grows with the height deficit at ``height_gap /
    time_gap_seconds`` m/s, clamped to ``[time_gap_seconds, max_time_seconds]``.
    """

    height_gap: float = 0.2
    time_gap_seconds: float = 0.25
    max_time_seconds: float = 1.25
    hold_max_time_seconds: float = 2.0
    enter_height_fraction: float = 0.65
    exit_height_fraction: float = 0.9

    def validate(self):
        if self.height_gap <= 0 or self.time_gap_seconds <= 0:
            raise ValueError("Get-up height and time gaps must be positive")
        if not (
            self.time_gap_seconds <= self.max_time_seconds <= self.hold_max_time_seconds
        ):
            raise ValueError(
                "Get-up times must satisfy time gap <= max time <= hold max time"
            )
        if not 0 < self.enter_height_fraction < self.exit_height_fraction <= 1:
            raise ValueError(
                "Get-up height fractions must satisfy 0 < enter < exit <= 1"
            )


@dataclass
class MaskedMimicTaskControlConfig(MimicControlConfig):
    """Inference-only goals. Script setters use world coordinates, seconds, m/s.

    The prompt's layout is ``canvas``, derived by ``configure_masked_mimic_task``
    from the trained checkpoint's MaskedMimicControlConfig and the robot; the
    training sampler's probabilities have no meaning here and are not carried.
    """

    _target_: str = (
        "protomotions.envs.control.masked_mimic_tasks.controller.MaskedMimicTaskControl"
    )
    canvas: Optional[GoalCanvas] = None  # required; see validate
    task: str = "idle"
    command_source: str = "script"
    horizon_seconds: float = 1.0
    speed: float = 1.0
    heading: float = 0.0
    # Steering only: None faces the movement heading; times are in seconds.
    facing: Optional[float] = None
    turn_rate_degrees_per_second: float = 90.0
    min_target_time_seconds: float = 0.1
    height_target_times_seconds: Tuple[float, ...] = (0.5, 1.0)
    getup: Optional[GetupRecoveryConfig] = None
    location_offset: Tuple[float, float] = (2.0, 0.0)
    location_speed: float = 1.5  # Keyboard arrival timing: distance / speed (m/s).
    reach_body_name: Optional[str] = None
    reach_offset: Tuple[float, float, float] = (0.3, 0.0, 0.1)
    # Simulation seconds between diagnostics; zero disables printing.
    status_print_interval_seconds: float = 0.5

    def validate(self):
        if self.canvas is None:
            raise ValueError(
                "MaskedMimicTaskControlConfig needs its GoalCanvas: "
                "configure_masked_mimic_task derives it from the trained "
                "MaskedMimicControlConfig and the robot"
            )
        if self.task not in MaskedMimicTaskControl.MODES or self.command_source not in (
            "script",
            "keyboard",
        ):
            raise ValueError("Unsupported native task or command source")
        if not math.isfinite(self.horizon_seconds) or self.horizon_seconds <= 0:
            raise ValueError("Goal horizon must be finite and positive")
        if not math.isfinite(self.location_speed) or self.location_speed <= 0:
            raise ValueError("Location speed must be finite and positive")
        if self.task == "steering" or (
            self.command_source == "keyboard" and self.canvas.has_velocity_command
        ):
            self.validate_steering()
        if self.getup is not None:
            self.getup.validate()
        if (
            not math.isfinite(self.speed)
            or self.speed < 0
            or not math.isfinite(self.heading)
        ):
            raise ValueError("Speed must be finite/nonnegative and heading finite")
        if (
            len(self.location_offset) != 2
            or len(self.reach_offset) != 3
            or not all(
                math.isfinite(x) for x in (*self.location_offset, *self.reach_offset)
            )
        ):
            raise ValueError("Goal offsets must be finite XY / XYZ vectors")

    def validate_steering(self):
        """Require a velocity input and non-overlapping steering/point-goal slots."""
        if not self.canvas.has_velocity_command:
            raise ValueError("Steering requires a model trained with velocity commands")
        times = self.height_target_times_seconds
        if (
            self.canvas.num_masked_future_steps <= 1 + len(times)
            or list(times) != sorted(times)
            or not all(math.isfinite(t) and t > 0 for t in times)
            or not all(
                math.isfinite(x) and x > 0
                for x in (
                    self.turn_rate_degrees_per_second,
                    self.min_target_time_seconds,
                )
            )
            or (self.facing is not None and not math.isfinite(self.facing))
        ):
            raise ValueError(
                "Steering needs positive turn rate and times, sorted finite height "
                "times, a finite facing, and a free last frame for point goals"
            )


class MaskedMimicTaskControl(MimicControl):
    """Task commands share one prompt packer; no training sampler."""

    # The parent's ``mimic.*`` names plus the packed ``masked_mimic.*`` goals.
    namespaces = ("mimic", "masked_mimic")
    # ``PackedGoals`` publishes every masked-mimic modality as a tensor (empty
    # masks for the ones not in use), so a binding to any of them is always
    # satisfied here, whatever the canvas carries: none of these switches is
    # a field of this config. (Absence instead of zero-filled modalities is
    # after handover; it changes the trained width.)
    _ALWAYS_PACKED = (
        "anchor_pose_submasking",
        "local_stream",
        "style",
        "velocity_command",
    )
    MODES = {"idle": 0, "reaching": 1, "location": 2, "steering": 3}
    MAX_SPEED = 5.0  # m/s, shared by keyboard and scripted commands.

    @classmethod
    def provides(cls, config, tag):
        """Every masked-mimic modality is published (see ``_ALWAYS_PACKED``);
        any other switch is read off the config as usual."""
        if tag in cls._ALWAYS_PACKED:
            return True
        return super().provides(config, tag)

    @classmethod
    def check_bindings(cls, config, robot_config, bound_paths):
        """The interactive controller serves exactly the trained layout."""
        config.validate()
        config.canvas.validate(bound_paths, getup=config.getup is not None)

    @classmethod
    def terminations(cls, config, robot_config):
        """No check: an inference task runs until the script or the viewer says so."""
        return {}

    def __init__(self, config, inputs):
        super().__init__(config, inputs)
        config.validate()
        canvas = config.canvas
        robot = inputs.robot
        names = robot.kinematic_info.body_names
        self.anchor = names.index(canvas.anchor_body)
        self.body_ids = [names.index(name) for name in canvas.conditionable_bodies]
        if self.anchor not in self.body_ids:
            raise ValueError(
                "Native goals require the anchor in trackable_bodies_subset"
            )
        reach_name = config.reach_body_name
        self.reach_body = None if reach_name is None else names.index(reach_name)
        if self.reach_body is not None and (
            self.reach_body not in self.body_ids or self.reach_body == self.anchor
        ):
            raise ValueError("Reaching requires a conditionable non-anchor body")
        if config.task == "reaching" and self.reach_body is None:
            raise ValueError("Reaching requires reach_body_name")
        add = self.state.add
        self.mode = add("mode", (), torch.long, 0)
        self.position = add("position", (3,), torch.float, 0.0)
        self.heading = add("heading", (), torch.float, 0.0)
        self.speed = add("speed", (), torch.float, 0.0)
        self.location_speed = add(
            "location_speed",
            (),
            torch.float,
            min(config.location_speed, self.MAX_SPEED),
        )
        self.reach_position = add("reach_position", (3,), torch.float, 0.0)
        self.deadline = add("deadline", (), torch.float, 0.0)
        self.facing = add("facing", (), torch.float, 0.0)
        self.recovering = add("recovering", (), torch.bool, False)
        # The latched measurement (see the module docstring).
        num_bodies = robot.kinematic_info.num_bodies
        self.body_pos = add("body_pos", (num_bodies, 3), torch.float, 0.0)
        identity = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(num_bodies, 4)
        self.body_rot = add("body_rot", (num_bodies, 4), torch.float, identity)
        self.forward_axis = robot.semantic_forward_axis_xy
        if config.getup is not None:
            self.upright_height = self._default_upright_height()
        self._bindings = None
        self._keys = {}
        self._status_steps_until_print = 0
        if config.command_source == "keyboard":
            if inputs.headless:
                raise ValueError("Keyboard goals require a visible simulator")
            self._bindings = inputs.ui.scope("native_goals")
            self._keyboard_modes = ["idle", "location"]
            if canvas.has_velocity_command:
                self._keyboard_modes.append("steering")
            if self.reach_body is not None:
                self._keyboard_modes.append("reaching")
            for key, label in (
                ("X", "Cycle control mode"),
                ("C", "Increase steering / location speed"),
                ("Z", "Decrease steering / location speed"),
                ("W", "Turn movement counter-clockwise / move target +X"),
                ("S", "Turn movement clockwise / move target -X"),
                ("A", "Turn facing counter-clockwise / move target +Y"),
                ("D", "Turn facing clockwise / move target -Y"),
                ("U", "Raise reaching target"),
                ("I", "Lower reaching target"),
            ):
                self._keys[key] = self._bindings.register(key, key, label)

    def _default_upright_height(self):
        """Torso height above the lowest body in the robot's default FK pose."""
        robot = self.inputs.robot
        if robot.default_dof_pos is None:
            raise ValueError("Get-up recovery requires robot default_dof_pos")
        fk = BakedAnchorFK(
            robot.kinematic_info,
            self.anchor,
            torch.arange(robot.kinematic_info.num_bodies),
        )
        dof_pos = torch.as_tensor(robot.default_dof_pos, dtype=torch.float32)[None]
        pos, rotation, _, _ = fk.body_kinematics(dof_pos, torch.zeros_like(dof_pos))
        # The upright reference fixes the anchor orientation, not the root's.
        pos = (
            rotation[:, self.anchor].transpose(-1, -2)[:, None]
            @ (pos - pos[:, self.anchor, None])[..., None]
        ).squeeze(-1)
        return float(pos[0, self.anchor, 2] - pos[0, :, 2].amin())

    def _clock(self):
        return self.inputs.clock.progress_buf.to(self.position) * self.inputs.dt

    def _above_ground(self, world):
        """Goal heights follow the terrain under the goal, not under the robot."""
        above = world.clone()
        above[:, 2] -= self.inputs.ground_height(world)
        return above

    def _world(self, above_ground):
        world = above_ground.clone()
        world[:, 2] += self.inputs.ground_height(above_ground)
        return world

    def _latch(self, env_ids, state):
        """Keep the measured body poses of ``env_ids`` (None: every env, a
        full state; otherwise ``state`` holds those rows)."""
        rows = slice(None) if env_ids is None else env_ids
        self.body_pos[rows] = state.rigid_body_pos
        self.body_rot[rows] = state.rigid_body_rot

    def _capture(self, env_ids):
        self.position[env_ids] = self._above_ground(self.body_pos[env_ids, self.anchor])
        self.heading[env_ids] = rotations.calc_heading(
            self.body_rot[env_ids, self.anchor], True
        )
        if self.reach_body is not None:
            self.reach_position[env_ids] = self._above_ground(
                self.body_pos[env_ids, self.reach_body]
            )
        self.deadline[env_ids] = self._clock()[env_ids] + self.config.horizon_seconds

    def reset_rows(self, env_ids, state):
        super().reset_rows(env_ids, state)
        self._latch(env_ids, state)
        self._capture(env_ids)
        # The env has already cleared these rows' progress (the reset's clear
        # phase runs before its controls phase), so the clock reads 0 here and
        # a deadline is simply its horizon.
        self.deadline[env_ids] = self.config.horizon_seconds
        self.mode[env_ids] = self.MODES[self.config.task]
        self.speed[env_ids] = 0
        self.location_speed[env_ids] = min(self.config.location_speed, self.MAX_SPEED)
        self.recovering[env_ids] = False
        if self.config.task == "location":
            self.position[env_ids, :2] += torch.tensor(
                self.config.location_offset, device=self.inputs.device
            )
            self.heading[env_ids] = math.atan2(
                self.config.location_offset[1], self.config.location_offset[0]
            )
            if self.config.command_source == "keyboard":
                self.deadline[env_ids] = self._location_travel_time(
                    env_ids, self.position[env_ids, :2]
                )
            else:
                distance = math.hypot(*self.config.location_offset)
                seconds = max(self.config.horizon_seconds, distance / self.MAX_SPEED)
                self.deadline[env_ids] = seconds
                self.location_speed[env_ids] = distance / seconds
        elif self.config.task == "reaching":
            self.reach_position[env_ids] += torch.tensor(
                self.config.reach_offset, device=self.inputs.device
            )
        elif self.config.task == "steering":
            self._start_steering(
                env_ids,
                self.config.speed,
                self.config.heading,
                self.config.heading
                if self.config.facing is None
                else self.config.facing,
                torch.zeros(len(env_ids), device=self.inputs.device),
            )
        # A robot reset onto the ground starts in recovery: latch the reset
        # rows from their fresh pose here, so that publishing the context
        # afterwards is a pure read.
        self._update_getup(env_ids)

    def hold(self, env_ids):
        self._capture(env_ids)
        self.mode[env_ids] = self.MODES["idle"]
        self.speed[env_ids] = 0

    def set_location(self, env_ids, position_xy, seconds):
        self._set_timed_target(env_ids, position_xy, seconds, reaching=False)

    def set_reaching(self, env_ids, position_xyz, seconds):
        if self.reach_body is None:
            raise ValueError(
                "Configure reach_body_name before sending reaching commands"
            )
        self._set_timed_target(env_ids, position_xyz, seconds, reaching=True)

    def _set_timed_target(self, env_ids, value, seconds, reaching):
        value = torch.as_tensor(
            value, device=self.inputs.device, dtype=self.position.dtype
        )
        if (
            value.shape != (len(env_ids), 3 if reaching else 2)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Target must be a finite world-space batch")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Target duration must be finite and positive")
        self._capture(env_ids)
        if reaching:
            self.reach_position[env_ids] = self._above_ground(value)
        else:
            delta = value - self.position[env_ids, :2]
            # A distant point must not imply travel faster than the speed cap.
            seconds = torch.maximum(
                delta.norm(dim=-1) / self.MAX_SPEED,
                self.deadline.new_full((len(env_ids),), seconds),
            )
            self.location_speed[env_ids] = delta.norm(dim=-1) / seconds
            moving = delta.norm(dim=-1) > 1e-6
            self.heading[env_ids] = torch.where(
                moving, torch.atan2(delta[:, 1], delta[:, 0]), self.heading[env_ids]
            )
            self.position[env_ids, :2] = value
        self.mode[env_ids] = self.MODES["reaching" if reaching else "location"]
        self.speed[env_ids] = 0
        self.deadline[env_ids] = self._clock()[env_ids] + seconds

    def set_steering(self, env_ids, speed, heading, facing=None):
        """Move along world ``heading`` at ``speed`` m/s, facing ``facing`` (default heading)."""
        # Scripted mode switches must satisfy the same canvas contract as startup.
        self.config.validate_steering()
        self._start_steering(
            env_ids,
            speed,
            heading,
            heading if facing is None else facing,
            self._clock()[env_ids],
        )

    def _start_steering(self, env_ids, speed, heading, facing, now):
        speed, heading, facing = [
            torch.as_tensor(
                x, device=self.inputs.device, dtype=self.position.dtype
            ).expand(len(env_ids))
            for x in (speed, heading, facing)
        ]
        if (
            not torch.isfinite(speed).all()
            or not torch.isfinite(heading).all()
            or not torch.isfinite(facing).all()
            or (speed < 0).any()
        ):
            raise ValueError(
                "Steering requires finite nonnegative speed and finite world headings"
            )
        # The facing deadline is fixed when the command changes: turn time at
        # the configured rate, so the policy sees a shrinking, reachable horizon.
        forward = semantic_forward_xy(
            self.body_rot[env_ids, self.anchor], self.forward_axis, w_last=True
        )
        error = facing - torch.atan2(forward[:, 1], forward[:, 0])
        error = (error + math.pi) % (2 * math.pi) - math.pi
        turn = error.abs() / math.radians(self.config.turn_rate_degrees_per_second)
        self.mode[env_ids] = self.MODES["steering"]
        self.speed[env_ids], self.heading[env_ids] = (
            speed.clamp_max(self.MAX_SPEED),
            heading,
        )
        self.facing[env_ids] = facing
        self.deadline[env_ids] = now + turn.clamp_min(
            self.config.min_target_time_seconds
        )

    def _cycle_keyboard_mode(self, env_ids):
        """Start each mode at the current pose without resetting the robot."""
        modes = [self.MODES[name] for name in self._keyboard_modes]
        current = modes.index(int(self.mode[env_ids[0]]))
        name = self._keyboard_modes[(current + 1) % len(modes)]
        self.hold(env_ids)
        self.mode[env_ids] = self.MODES[name]
        if name == "steering":
            forward = semantic_forward_xy(
                self.body_rot[env_ids, self.anchor], self.forward_axis, w_last=True
            )
            self.set_steering(env_ids, 0.0, torch.atan2(forward[:, 1], forward[:, 0]))
        logging.getLogger(__name__).info("Native control mode: %s", name)

    def _location_travel_time(self, env_ids, target_xy):
        current = self.body_pos[env_ids, self.anchor, :2]
        distance = (target_xy - current).norm(dim=-1)
        return (distance / self.location_speed[env_ids].clamp_min(0.1)).clamp_min(
            self.config.min_target_time_seconds
        )

    def _set_keyboard_location(self, env_ids, target_xy):
        speed = self.location_speed[env_ids].clone()
        travel_time = self._location_travel_time(env_ids, target_xy)
        self.set_location(env_ids, target_xy, self.config.horizon_seconds)
        # Keep the selected keyboard pace even when the target is at our feet.
        self.location_speed[env_ids] = speed
        self.deadline[env_ids] = self._clock()[env_ids] + travel_time

    def _adjust_keyboard_speed(self, env_ids, delta):
        mode = int(self.mode[env_ids[0]])
        if mode == self.MODES["steering"]:
            self.set_steering(
                env_ids,
                (self.speed[env_ids] + delta).clamp_min(0),
                self.heading[env_ids],
                self.facing[env_ids],
            )
            speed = self.speed[env_ids[0]]
        elif mode == self.MODES["location"]:
            self.location_speed[env_ids] = (self.location_speed[env_ids] + delta).clamp(
                0.1, self.MAX_SPEED
            )
            self._set_keyboard_location(env_ids, self.position[env_ids, :2])
            speed = self.location_speed[env_ids[0]]
        else:
            return
        logging.getLogger(__name__).info("Target speed: %.1f m/s", float(speed))

    def advance(self, state):
        logging = super().advance(state)
        self._latch(None, state)
        self._update_getup(None)
        self._maybe_print_status(state)
        if self._bindings is None:
            return logging
        for key, handle in self._keys.items():
            if not handle.consume():
                continue
            index = int(self.inputs.ui.active_env_id)
            if not 0 <= index < self.inputs.num_envs:
                raise IndexError("Active environment is out of range")
            ids = torch.tensor([index], device=self.inputs.device)
            if key == "X":
                self._cycle_keyboard_mode(ids)
            elif key in ("C", "Z"):
                self._adjust_keyboard_speed(ids, 0.1 if key == "C" else -0.1)
            elif self.mode[index] == self.MODES["idle"]:
                continue
            elif self.mode[index] == self.MODES["steering"] and key in "WASD":
                heading_delta = math.radians({"W": 10, "S": -10}.get(key, 0))
                facing_delta = math.radians({"A": 10, "D": -10}.get(key, 0))
                self.set_steering(
                    ids,
                    self.speed[ids],
                    self.heading[ids] + heading_delta,
                    self.facing[ids] + facing_delta,
                )
            else:
                delta = self.position.new_tensor(
                    {
                        "W": (0.2, 0, 0),
                        "S": (-0.2, 0, 0),
                        "A": (0, 0.2, 0),
                        "D": (0, -0.2, 0),
                        "U": (0, 0, 0.1),
                        "I": (0, 0, -0.1),
                    }[key]
                )
                if self.mode[index] == self.MODES["reaching"]:
                    self.set_reaching(
                        ids,
                        self._world(self.reach_position[ids]) + delta,
                        self.config.horizon_seconds,
                    )
                elif self.mode[index] == self.MODES["location"] and key in "WASD":
                    self._set_keyboard_location(
                        ids,
                        self.position[ids, :2] + delta[:2],
                    )
        return logging

    def _maybe_print_status(self, state):
        """Report the active robot's task error, not averages over other robots."""
        interval = self.config.status_print_interval_seconds
        if interval <= 0:
            return
        if self._status_steps_until_print > 0:
            self._status_steps_until_print -= 1
            return
        self._status_steps_until_print = max(1, round(interval / self.inputs.dt)) - 1
        index = int(self.inputs.ui.active_env_id) if self._bindings is not None else 0
        ids = torch.tensor([index], device=self.inputs.device)
        position = state.rigid_body_pos[index, self.anchor]
        if self.recovering[index]:
            height = position[2] - state.rigid_body_pos[index, :, 2].amin()
            message = (
                f"recovery; torso height={height:.3f} m, "
                f"target={self.upright_height:.3f} m, "
                f"error={self.upright_height - height:+.3f} m; task goals masked"
            )
        else:
            mode = int(self.mode[index])
            velocity = state.rigid_body_vel[index, self.anchor, :2]
            speed = velocity.norm()
            if mode == self.MODES["steering"]:
                target = self.speed[index] * torch.stack(
                    (self.heading[index].cos(), self.heading[index].sin())
                )
                forward = semantic_forward_xy(
                    state.rigid_body_rot[ids, self.anchor],
                    self.forward_axis,
                    w_last=True,
                )[0]
                heading_error = self.facing[index] - torch.atan2(forward[1], forward[0])
                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
                message = (
                    f"steering; velocity error={(target - velocity).norm():.3f} m/s; "
                    f"speed actual={speed:.3f}, target={self.speed[index]:.3f} m/s; "
                    f"heading error={math.degrees(float(heading_error)):+.1f} deg"
                )
            else:
                distance = (self.position[index, :2] - position[:2]).norm()
                target_speed = (
                    self.location_speed[index]
                    if mode == self.MODES["location"]
                    else 0.0
                )
                name = next(name for name, value in self.MODES.items() if value == mode)
                message = (
                    f"{name}; location error={distance:.3f} m; "
                    f"speed actual={speed:.3f}, target={target_speed:.3f} m/s, "
                    f"error={speed - target_speed:+.3f} m/s"
                )
                if mode == self.MODES["reaching"]:
                    target = self._world(self.reach_position[ids])[0]
                    error = (
                        target - state.rigid_body_pos[index, self.reach_body]
                    ).norm()
                    message += f"; reaching error={error:.3f} m"
        logging.getLogger(__name__).info("[native control env=%d] %s", index, message)

    def _update_getup(self, env_ids):
        """Latch fallen state with hysteresis from the latched body positions;
        idempotent for an unchanged state."""
        getup = self.config.getup
        if getup is None:
            return
        selected = slice(None) if env_ids is None else env_ids
        pos = self.body_pos[selected]
        height = pos[:, self.anchor, 2] - pos[..., 2].amin(dim=-1)
        recovering = self.recovering[selected]
        recovering |= height < getup.enter_height_fraction * self.upright_height
        recovering &= height < getup.exit_height_fraction * self.upright_height
        self.recovering[selected] = recovering

    def getup_time(self, height_deficit):
        """First recovery deadline for a torso-height deficit in meters."""
        getup = self.config.getup
        speed = getup.height_gap / getup.time_gap_seconds
        return (height_deficit / speed).clamp(
            getup.time_gap_seconds, getup.max_time_seconds
        )

    def _pack_prompt(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        clock = self._clock()[selected]
        mode, heading = self.mode[selected], self.heading[selected]
        position = self.position[selected].clone()
        ground = self.inputs.ground_height(position)
        position[:, 2] += ground
        steering = mode == self.MODES["steering"]
        recovering = self.recovering[selected]
        axis = torch.zeros_like(position)
        axis[:, 2] = 1
        rotation = rotations.quat_from_angle_axis(heading, axis, True)
        deadline = self.deadline[selected]
        canvas = self.config.canvas
        slot = canvas.num_masked_future_steps - 1
        goals = [
            BodyGoal(
                self.anchor,
                slot,
                deadline,
                ~steering & ~recovering,
                position=position,
                rotation=rotation,
                translation=(True, False),
                orientation=(False, True),
            )
        ]
        if self.reach_body is not None:
            goals.append(
                BodyGoal(
                    self.reach_body,
                    slot,
                    deadline,
                    (mode == self.MODES["reaching"]) & ~recovering,
                    position=self._world(self.reach_position[selected]),
                )
            )
        # Steering is a facing goal, torso-height goals and a velocity command.
        # Fallen robots see only the local get-up goals added below.
        walking = steering & ~recovering
        body_pos, body_rot = self.body_pos[selected], self.body_rot[selected]
        current = body_pos[:, self.anchor]
        current_ground = self.inputs.ground_height(current)
        ground = torch.where(steering, current_ground, ground)
        # Non-steering policies may have fewer target frames. Do not allocate
        # invisible steering slots in their canvas.
        if steering.any():
            facing = heading_to_quat_for_forward_axis(
                self.facing[selected], self.forward_axis, w_last=True
            )
            goals.append(
                BodyGoal(
                    self.anchor,
                    0,
                    torch.maximum(
                        deadline, clock + self.config.min_target_time_seconds
                    ),
                    walking,
                    rotation=facing,
                    orientation=(False, True),
                )
            )
            stand = current.clone()
            stand[:, 2] = current_ground + self.inputs.robot.default_root_height
            for frame, seconds in enumerate(self.config.height_target_times_seconds, 1):
                goals.append(
                    BodyGoal(
                        self.anchor,
                        frame,
                        clock + seconds,
                        walking,
                        position=stand,
                        translation=(False, True),
                    )
                )
        velocity = (
            torch.stack((heading.cos(), heading.sin()), -1) * self.speed[selected, None]
        )
        packed = pack_goals(
            goals,
            clock=clock,
            body_count=len(self.inputs.robot.kinematic_info.body_names),
            conditionable_ids=self.body_ids,
            anchor_id=self.anchor,
            frames=canvas.num_masked_future_steps,
            style_frames=canvas.style_frames or 0,
            ground_height=ground,
            velocity=velocity,
            velocity_mask=walking[:, None],
        )
        if recovering.any():
            self._pack_getup(packed, recovering, clock, body_pos, body_rot)
        return packed

    def _pack_getup(self, packed, fallen, clock, pos, rot):
        """Repeat one local torso-height goal, spaced by the get-up time gap,
        from the latched body poses ``pos`` [E, B, 3] and ``rot`` [E, B, 4]."""
        getup = self.config.getup
        frames = self.config.canvas.num_masked_future_steps
        floor = pos[..., 2].amin(dim=-1)
        target = pos.clone()
        target[:, self.anchor, 2] = floor + self.upright_height
        deficit = (self.upright_height - (pos[:, self.anchor, 2] - floor)).clamp_min(0)
        offsets = (
            self.getup_time(deficit)[:, None]
            + torch.arange(frames, device=clock.device) * getup.time_gap_seconds
        ).clamp_max(getup.hold_max_time_seconds)
        packed.local_ref_pos = packed.local_ref_pos.clone()
        packed.local_ref_pos[fallen] = target[fallen, None].expand(-1, frames, -1, -1)
        packed.local_ref_rot = packed.local_ref_rot.clone()
        packed.local_ref_rot[fallen] = rot[fallen, None].expand(-1, frames, -1, -1)
        packed.local_target_poses_masks[fallen] = True
        packed.local_target_bodies_masks.view(
            len(clock), frames, len(self.body_ids), 2
        )[fallen, :, self.body_ids.index(self.anchor), 0] = True
        packed.local_time_offsets[fallen] = offsets[fallen]
        packed.local_target_times = packed.local_target_times.clone()
        packed.local_target_times[fallen] = clock[fallen, None] + offsets[fallen]

    def publish(self, env_ids):
        """The parent's ``mimic.*`` names plus the packed ``masked_mimic.*`` goals.

        A pure read: the body poses and the get-up latch it packs were updated
        by ``advance`` (every env, after physics) or by ``reset_rows`` (the
        rows just reset), never here.
        """
        published = super().publish(env_ids)
        published.update(self._pack_prompt(env_ids).names())
        return published

    def close(self):
        if self._bindings is not None:
            self._bindings.unregister_all()

    def create_visualization_markers(self, headless):
        if headless:
            return {}
        return {
            "native_goals": VisualizationMarkerConfig(
                type="sphere",
                color=(0.1, 0.7, 0.2),
                markers=[MarkerConfig(size="regular") for _ in self.body_ids],
            ),
            "native_movement": VisualizationMarkerConfig(
                type="arrow",
                color=(0.8, 0.0, 0.0),
                markers=[MarkerConfig(size="regular")],
            ),
            "native_facing": VisualizationMarkerConfig(
                type="arrow",
                color=(0.0, 0.0, 0.8),
                markers=[MarkerConfig(size="regular")],
            ),
        }

    def get_markers_state(self, state):
        if self.inputs.headless:
            return {}
        goal = self._pack_prompt()
        masks = goal.target_bodies_masks.view(
            self.inputs.num_envs,
            self.config.canvas.num_masked_future_steps,
            len(self.body_ids),
            2,
        )
        position = goal.ref_pos[:, -1, self.body_ids].clone()
        position[~masks[:, -1, :, 0], 2] = -1000
        markers = {
            "native_goals": MarkerState(
                translation=position, orientation=goal.ref_rot[:, -1, self.body_ids]
            )
        }
        root = state.rigid_body_pos[:, 0]
        for name, heading in (
            ("native_movement", self.heading),
            ("native_facing", self.facing),
        ):
            position = root.clone()
            position[:, :2] += torch.stack((heading.cos(), heading.sin()), dim=-1)
            # Hide only inactive visualization markers, never physical geometry.
            # Both arrows share the root-height origin and overlap when aligned.
            position[self.mode != self.MODES["steering"], 2] = -1000
            markers[name] = MarkerState(
                translation=position[:, None],
                orientation=rotations.heading_to_quat(heading, w_last=True)[:, None],
            )
        return markers


def configure_masked_mimic_task(env_config, robot_config, task, getup=None):
    """Prepare interactive inference while preserving the trained goal dimensions.

    Derive the goal canvas from the one trained MaskedMimicControlConfig and
    the robot, check it against what the observations bind (and against a
    get-up preset, which needs a local stream), then replace the masked
    reference controller and disable training terminations so interactive
    goals are not cut short by tracking failures. The episode limit is the
    inference default (``EnvConfig.apply_inference_defaults`` runs before the
    hooks). Other control components and observation bindings are unchanged.
    """
    controls = [
        (name, config)
        for name, config in env_config.control_components.items()
        if isinstance(config, MaskedMimicControlConfig)
    ]
    if len(controls) != 1:
        raise ValueError("Native goals require exactly one MaskedMimicControlConfig")
    name, previous = controls[0]
    if not previous.anchor_pose_submasking:
        raise ValueError(
            "Checkpoint must use anchor submode observations for native goals"
        )
    canvas = GoalCanvas.derive(previous, robot_config)
    canvas.validate(bound_paths(env_config.observation_components), getup is not None)
    env_config.control_components[name] = MaskedMimicTaskControlConfig(
        task=task,
        future_steps=canvas.future_steps,
        # Only low-pose terminations read the smoothed height; they are removed below.
        reference_height_smoothing_seconds=0.0,
        canvas=canvas,
        getup=getup,
    )
    env_config.termination_components = {}
