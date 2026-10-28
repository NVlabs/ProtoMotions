# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One simulator-free deployment context, built by the real publishers.

The exporter needs, for every context name an actor observation binds, the
tensor's own shape and dtype, whether it carries a future-step axis and whether
it carries a history-slot axis. All four are facts the environment already
produces: this module builds a single context the way ``build_env`` does -- a
synthetic ``RobotState`` through the real :class:`SensorRuntime`, a
:class:`StateHistoryBuffer` sized by the actor's own observations, a
:class:`StatePublisher`, and the configured reference control over a tiny reference
stand-in -- and returns the merged ``{path: value}`` map the exporter reads
shapes off, with no simulator, motion library or GPU.

Future windows and history cannot be identified from tensor rank alone. Probe
the publishers with one additional future step or history slot:

- the mimic control published at horizon ``K`` and ``K + 1`` names the
  **future** windows (``mimic.future_*``);
- a MaskedMimic control is also probed at masked horizon ``M + 1``; its masks
  flatten the time/body/property axes, so compare full shapes;
- the state history published at depth ``D`` and ``D + 1`` names the **slot**
  windows (the action rings and the historical namespaces).

``build_sample_context`` returns ``(context, future_names, slot_names)``; the
exporter reads a bound path's tensor from ``context`` and tests membership of
the two name sets to decide the future and history blocks. The one thing it does
not derive here is the host kind: that is the single ``deploy=`` string the
declaration still carries.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple

import torch
from torch import Tensor

from protomotions.envs.base_env.publishers import StatePublisher
from protomotions.envs.context_sources import PRIVILEGED, SENSOR_SOURCES
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.obs.state_history_buffer import (
    HistorySample,
    StateHistoryBuffer,
    history_depth,
)
from protomotions.envs.state import BodyState
from protomotions.envs.steps import normalize_steps
from protomotions.envs.base_env.reset.placed_reference import ReferenceWindow
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.robot_configs.base import RobotConfig


def _identity_quat(num: int, *leading: int, device: torch.device) -> Tensor:
    """A unit xyzw quaternion field of shape ``(*leading, num, 4)``."""
    quat = torch.zeros(*leading, num, 4, device=device)
    quat[..., 3] = 1.0
    return quat


class _SampleReference:
    """Synthetic reference queries used by Mimic and MaskedMimic at export.

    It answers ``current_and_window`` and ``done`` with synthetic tensors sized
    from the horizon the control asks for (``K = time_offsets.shape[0]``), the
    way the motion library would. The current reference state leaves
    ``dof_forces``/``rigid_body_contacts``/``local_rigid_body_rot`` ``None``, as
    the library does, so the mimic namespace publishes exactly the reference
    names the vocabulary declares and no undeclared ones.
    """

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        num_dofs: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self.num_envs = num_envs
        self._bodies = num_bodies
        self._dofs = num_dofs
        self._device = device
        self.generator = generator
        self.ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.times = torch.zeros(num_envs, device=device)

    def _rows(self, env_ids: Optional[Tensor]) -> int:
        return self.num_envs if env_ids is None else int(len(env_ids))

    def _randn(self, *shape: int) -> Tensor:
        return torch.randn(*shape, generator=self.generator, device=self._device)

    def lengths(self, motion_ids: Tensor) -> Tensor:
        return torch.full_like(motion_ids, 10.0, dtype=torch.float32)

    def raw(self, motion_ids: Tensor, times: Tensor) -> RobotState:
        rows = len(motion_ids)
        bodies, dofs, device = self._bodies, self._dofs, self._device
        return RobotState(
            state_conversion=StateConversion.COMMON,
            rigid_body_pos=self._randn(rows, bodies, 3),
            rigid_body_rot=_identity_quat(bodies, rows, device=device),
            rigid_body_vel=torch.zeros(rows, bodies, 3, device=device),
            rigid_body_ang_vel=torch.zeros(rows, bodies, 3, device=device),
            dof_pos=torch.zeros(rows, dofs, device=device),
            dof_vel=torch.zeros(rows, dofs, device=device),
        )

    def raw_positions(self, motion_ids: Tensor, times: Tensor) -> Tensor:
        return self.raw(motion_ids, times).rigid_body_pos

    def placement_offset(self, positions: Tensor, env_ids: Tensor) -> Tensor:
        return torch.zeros(len(env_ids), 1, 3, device=self._device)

    def window(self, env_ids: Optional[Tensor], times: Tensor) -> ReferenceWindow:
        rows, slots = times.shape
        ids = self.ids if env_ids is None else self.ids[env_ids]
        state = self.raw(ids[:, None].expand(-1, slots).reshape(-1), times.reshape(-1))

        def windowed(value: Tensor) -> Tensor:
            return value.reshape(rows, slots, *value.shape[1:])

        return ReferenceWindow(
            times=times,
            pos=windowed(state.rigid_body_pos),
            rot=windowed(state.rigid_body_rot),
            vel=windowed(state.rigid_body_vel),
            ang_vel=windowed(state.rigid_body_ang_vel),
            dof_pos=windowed(state.dof_pos),
            dof_vel=windowed(state.dof_vel),
        )

    def current_and_window(
        self, env_ids: Optional[Tensor], time_offsets: Tensor
    ) -> Tuple[RobotState, ReferenceWindow]:
        selected = slice(None) if env_ids is None else env_ids
        times = self.times[selected]
        return (
            self.raw(self.ids[selected], times),
            self.window(env_ids, times[:, None] + time_offsets),
        )

    def done(self, env_ids: Optional[Tensor] = None) -> Tensor:
        return torch.zeros(self._rows(env_ids), dtype=torch.bool, device=self._device)


def _mimic_inputs(
    robot_config: RobotConfig,
    device: torch.device,
    num_envs: int,
    reference: _SampleReference,
) -> TaskInputs:
    """Inputs used by reference controls; no simulator, UI or terrain service."""
    return TaskInputs(
        num_envs=num_envs,
        device=device,
        dt=float(robot_config.control_dt),
        robot=robot_config,
        clock=None,
        reference=reference,
        ground_height=None,
        generator=reference.generator,
        headless=True,
        ui=None,
        viewer_reset=None,
        restart_episode=None,
        teleport=None,
    )


def _grown_on_axis_one(base: Dict[str, Any], probe: Dict[str, Any]) -> Set[str]:
    """Names whose tensor is one longer on axis 1 in ``probe`` than in ``base``.

    Every other axis (the batch and the feature tail) is identical; only the
    future or slot axis moved. Non-tensor and rank-<2 values never qualify.
    """
    grown: Set[str] = set()
    for name, value in base.items():
        other = probe.get(name)
        if not torch.is_tensor(value) or not torch.is_tensor(other):
            continue
        if value.ndim < 2 or other.ndim < 2:
            continue
        if (
            value.shape[0] == other.shape[0]
            and other.shape[1] == value.shape[1] + 1
            and value.shape[2:] == other.shape[2:]
        ):
            grown.add(name)
    return grown


def _synthetic_robot_state(
    num_envs: int,
    num_bodies: int,
    num_dofs: int,
    device: torch.device,
    generator: torch.Generator,
) -> RobotState:
    """A finite ``RobotState`` the sensor runtime can measure and reconstruct.

    Orientations are identity (valid unit rotations), positions are finite and
    velocities zero; the optional ``dof_forces``/``rigid_body_contacts``/
    ``local_rigid_body_rot`` are left ``None``, as the simulator getters do, so
    nothing fabricates a channel the suite does not carry.
    """
    return RobotState(
        state_conversion=StateConversion.COMMON,
        rigid_body_pos=torch.randn(
            num_envs, num_bodies, 3, generator=generator, device=device
        ),
        rigid_body_rot=_identity_quat(num_bodies, num_envs, device=device),
        rigid_body_vel=torch.zeros(num_envs, num_bodies, 3, device=device),
        rigid_body_ang_vel=torch.zeros(num_envs, num_bodies, 3, device=device),
        dof_pos=torch.zeros(num_envs, num_dofs, device=device),
        dof_vel=torch.zeros(num_envs, num_dofs, device=device),
    )


def _publish_state(
    initial: HistorySample, depth: int, anchor_idx: int
) -> Dict[str, Any]:
    """One ``StatePublisher`` publish over a ring of ``depth`` past slots."""
    ring = StateHistoryBuffer(depth, initial)
    return StatePublisher(ring, anchor_idx).publish(None)


def _publish_mimic(
    robot_config: RobotConfig,
    device: torch.device,
    state: RobotState,
    reference: _SampleReference,
    config: MimicControlConfig,
) -> Dict[str, Any]:
    """Publish the configured reference control, including enabled modalities."""
    num_envs = state.dof_pos.shape[0]
    control_class = (
        MaskedMimicControl
        if isinstance(config, MaskedMimicControlConfig)
        else MimicControl
    )
    control = control_class(
        config,
        _mimic_inputs(robot_config, device, num_envs, reference),
    )
    control.reset_rows(torch.arange(num_envs, device=device), state)
    return control.publish(None)


def reference_control_config(env_config: EnvConfig) -> MimicControlConfig:
    """The single reference publisher used by the exported policy."""
    matches = {
        name: config
        for name, config in env_config.control_components.items()
        if isinstance(config, MimicControlConfig)
    }
    if len(matches) != 1:
        raise ValueError(
            "export requires exactly one MimicControl or MaskedMimicControl "
            f"component, found {len(matches)}: {sorted(matches)}"
        )
    return next(iter(matches.values()))


def build_sample_context(
    robot_config: RobotConfig,
    env_config: EnvConfig,
    actor_obs_configs: Dict[str, MdpComponent],
    device: str | torch.device = "cpu",
) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    """Build one deployment context and the future/slot name sets.

    ``actor_obs_configs`` are the actor's bound observation components (already
    body-selection-resolved); they size the history ring exactly as the
    environment's own ring is sized. ``env_config`` supplies the sensor suite
    (``for_inference`` so the clean and noisy estimates coincide) and the mimic
    control's future horizon. The returned ``context`` merges the state and
    mimic namespaces; ``future_names`` and ``slot_names`` are the names found to
    carry a future or a history axis by the horizon and depth probes.
    """
    device = torch.device(device)
    num_envs = 1
    num_bodies = len(robot_config.kinematic_info.body_names)
    num_dofs = robot_config.kinematic_info.num_dofs
    num_actions = robot_config.number_of_actions
    anchor_idx = robot_config.anchor_body_index

    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    synthetic = _synthetic_robot_state(
        num_envs, num_bodies, num_dofs, device, generator
    )

    # Sensor estimates come from the real runtime, read exactly, so the clean
    # and noisy sources carry only the channels the declared suite measures.
    estimates = None
    if env_config.sensors is not None:
        from protomotions.envs.sensors.runtime import SensorRuntime

        ground_query = None
        if env_config.sensors.height is not None:
            ground_query = lambda points: torch.zeros(  # noqa: E731
                points.shape[0], 1, device=points.device
            )
        runtime = SensorRuntime(
            env_config.sensors.for_inference(),
            robot_config,
            num_envs,
            device,
            ground_query=ground_query,
        )
        estimates = runtime.reset(torch.arange(num_envs, device=device), synthetic)

    privileged = BodyState(
        rigid_body_rot=synthetic.rigid_body_rot,
        rigid_body_ang_vel=synthetic.rigid_body_ang_vel,
        dof_pos=synthetic.dof_pos,
        dof_vel=synthetic.dof_vel,
        rigid_body_pos=synthetic.rigid_body_pos,
        rigid_body_vel=synthetic.rigid_body_vel,
        ground_heights=torch.zeros(num_envs, device=device),
    )
    states: Dict[str, BodyState] = {PRIVILEGED: privileged}
    if estimates is not None:
        for source in SENSOR_SOURCES:
            states[source] = getattr(estimates, source)
    zero_actions = torch.zeros(num_envs, num_actions, device=device)
    initial = HistorySample(states, zero_actions, zero_actions.clone())

    depth = history_depth(actor_obs_configs)
    published_state = _publish_state(initial, depth, anchor_idx)
    slot_names = _grown_on_axis_one(
        published_state, _publish_state(initial, depth + 1, anchor_idx)
    )

    reference = _SampleReference(num_envs, num_bodies, num_dofs, device, generator)
    config = reference_control_config(env_config)
    if isinstance(config, MaskedMimicControlConfig):
        paths = frozenset(
            path
            for component in actor_obs_configs.values()
            for path in component.get_bindings_dict().values()
        )
        GoalCanvas.derive(config, robot_config).validate(paths, getup=False)
    horizon = len(normalize_steps(config.future_steps))
    published_mimic = _publish_mimic(robot_config, device, synthetic, reference, config)
    future_names = _grown_on_axis_one(
        published_mimic,
        _publish_mimic(
            robot_config,
            device,
            synthetic,
            reference,
            replace(config, future_steps=horizon + 1),
        ),
    )
    if isinstance(config, MaskedMimicControlConfig):
        probe = _publish_mimic(
            robot_config,
            device,
            synthetic,
            reference,
            replace(config, num_masked_future_steps=config.num_masked_future_steps + 1),
        )
        # Body masks flatten the time/body/property axes, so compare the full
        # shapes rather than assuming every future dimension grows by one.
        future_names.update(
            name
            for name, value in published_mimic.items()
            if torch.is_tensor(value) and value.shape != probe[name].shape
        )

    # History kernels use the control timestep to age their tokens. As a scalar
    # it is folded into the exported graph, not requested from the host.
    context: Dict[str, Any] = {
        **published_state,
        **published_mimic,
        "dt": float(robot_config.control_dt),
    }
    return context, future_names, slot_names


__all__ = ["build_sample_context", "reference_control_config"]
