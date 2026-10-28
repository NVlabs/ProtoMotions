# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publishers of the context walk for objects without a ``publish`` of their own.

``BaseEnv._publish(env_ids)`` walks ``_publish_owners`` and merges what each
``publish(env_ids)`` returns, keyed by the owner's ``name``, into one flat
``FrozenContext`` (see ``protomotions/envs/context.py``; the context adds its
own ``dt`` and ``env_ids``). The owners with tensors of their own
(``EpisodeBuffers``, ``ContactMemory``) publish from ``owners.py``; the
classes here adapt the state history, the terrain, the scene library, the
simulator's contact matrix and the env's contact body ids.

A subset context (the rows being reset) slices every per-env tensor to
``env_ids`` and clones nothing beyond what the owners clone themselves
(``progress_buf``, ``prev_contact_force_magnitudes``); the history snapshot is
already owned by ``capture``, and everything else is the same view the
adapted object holds.

The names of one state source (root and anchor slices, ``root_height``, the
local angular velocities) are derived by ``state_names.body_state_names``,
which every state namespace -- current or historical, privileged or sensor --
and the ONNX exporters share, so the derivations cannot drift.

:class:`HistorySampler` is the write side of the same ring: it turns one
simulator read (and the sensors' estimates of it) into the ``HistorySample``
the step's record phase pushes and the reset's history phase seeds with.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from protomotions.envs.base_env.owners import ActionStage, select_rows
from protomotions.envs.contact_features import contact_flags
from protomotions.envs.context_requirements import has_heightfield
from protomotions.envs.context_sources import (
    PRIVILEGED,
    SENSOR_SOURCES,
    current_namespace,
    historical_namespace,
)
from protomotions.envs.obs.state_history_buffer import HistorySample
from protomotions.envs.sensors.runtime import SensorEstimates
from protomotions.envs.state import BodyState
from protomotions.envs.state_names import body_state_names
from protomotions.simulator.base_simulator.simulator_state import RobotState


class HistorySampler:
    """One ``HistorySample`` from a simulator read and the sensors' estimates.

    The privileged source is the read itself plus the terrain's ground height
    under the observation anchor (``robot_config.anchor_body_index``); the
    height sensor separately samples under the root, so the two differ on
    uneven terrain. Contact flags are derived from the simulator's contact
    matrix while it reports contacts. The terrain's query, the anchor index
    and the simulator's flag are read at every sample, as the env always did.
    Each sensor source is the estimate under its own name. The commands are
    the action stage's current raw and processed actions, sliced to
    ``env_ids`` like everything else.
    """

    def __init__(self, terrain, robot_config, simulator, action_stage: ActionStage):
        self.terrain = terrain
        self.robot_config = robot_config
        self.simulator = simulator
        self.action_stage = action_stage

    def sample(
        self,
        state: RobotState,
        estimates: Optional[SensorEstimates],
        env_ids: Optional[Tensor] = None,
    ) -> HistorySample:
        privileged = BodyState(
            rigid_body_rot=state.rigid_body_rot,
            rigid_body_ang_vel=state.rigid_body_ang_vel,
            dof_pos=state.dof_pos,
            dof_vel=state.dof_vel,
            rigid_body_pos=state.rigid_body_pos,
            rigid_body_vel=state.rigid_body_vel,
            dof_forces=state.dof_forces,
            ground_heights=self.terrain.get_ground_heights(
                state.rigid_body_pos[:, self.robot_config.anchor_body_index]
            ).squeeze(-1),
        )
        if self.simulator.contacts_enabled:
            privileged.rigid_body_contacts = contact_flags(
                self.simulator.get_contact_forces(env_ids)
            )
        states = {PRIVILEGED: privileged}
        if estimates is not None:
            states.update(
                {source: getattr(estimates, source) for source in SENSOR_SOURCES}
            )
        selector = slice(None) if env_ids is None else env_ids
        return HistorySample(
            states,
            self.action_stage.raw[selector],
            self.action_stage.processed[selector],
        )


def prefix_names(namespace: str, names: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"<namespace>.<name>": value}`` for every name."""
    return {f"{namespace}.{name}": value for name, value in names.items()}


class StatePublisher:
    """Current and past states of every source, and the command history.

    One ``capture`` of the history ring per publish; every name is a slice of
    that owned snapshot. Slot 0 is the current state and the command that
    produced it (``current_action``); slots 1 onward are the past, published
    under the historical namespaces when the ring keeps any past slot.
    ``ground_heights`` at the top level is the privileged
    ``current.ground_heights`` object.
    """

    name = "state"

    def __init__(self, state_history, anchor_body_index: int):
        self.state_history = state_history
        self.anchor_idx = anchor_body_index
        self.num_history_steps = state_history.num_history_steps

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        captured = self.state_history.capture(env_ids)
        published: Dict[str, Any] = {}
        for source, state in captured.states.items():
            current = state.map(lambda value: value[:, 0])
            published.update(
                prefix_names(
                    current_namespace(source),
                    body_state_names(current, self.anchor_idx),
                )
            )
            if self.num_history_steps:
                past = state.map(lambda value: value[:, 1:])
                namespace = historical_namespace(source)
                published.update(
                    prefix_names(namespace, body_state_names(past, self.anchor_idx))
                )
                published[f"{namespace}.raw_actions"] = captured.raw_actions[:, 1:]
                published[f"{namespace}.processed_actions"] = (
                    captured.processed_actions[:, 1:]
                )
                if past.rigid_body_contacts is not None:
                    published[f"{namespace}.body_contacts"] = past.rigid_body_contacts
        published["raw_action_history"] = captured.raw_actions
        published["processed_action_history"] = captured.processed_actions
        published["current_action"] = captured.raw_actions[:, 0]
        published["current_processed_action"] = captured.processed_actions[:, 0]
        if self.num_history_steps:
            published["previous_action"] = captured.raw_actions[:, 1]
            published["previous_processed_action"] = captured.processed_actions[:, 1]
        # The env always samples privileged ground heights; a bare test ring
        # may not, and then the vocabulary check reports the missing name.
        if "current.ground_heights" in published:
            published["ground_heights"] = published["current.ground_heights"]
        return published


class TerrainPublisher:
    """The terrain's height field, when it has one (flat test terrains do not).

    Whether it has one is ``context_requirements.has_heightfield``, the same
    predicate the build check uses to decide whether to expect ``terrain.*``.
    """

    name = "terrain"

    def __init__(self, terrain):
        self.terrain = terrain

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        terrain = self.terrain
        if not has_heightfield(terrain):
            return {}
        return {
            "terrain.height_points": select_rows(terrain.height_points, env_ids),
            "terrain.height_samples": terrain.height_samples,
            "terrain.horizontal_scale": terrain.config.horizontal_scale,
        }


class ScenePublisher:
    """Scene-object surfaces for the nearest-surface observations.

    Publishes the three object names only when the scene library has scene
    objects (the ``objects`` requirement) and the neutral pointclouds only
    when it computed them (``pointclouds``); a configuration without them
    publishes nothing under ``scene`` instead of a zero-width stand-in, and a
    binding to a missing name is refused by its tag.
    """

    name = "scene"

    def __init__(self, scene_lib, simulator, num_envs: int, device: torch.device):
        self.scene_lib = scene_lib
        self.simulator = simulator
        self.num_envs = num_envs
        self.device = device

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        scene_lib = self.scene_lib
        if scene_lib.num_scenes() <= 0:
            return {}
        if env_ids is not None:
            env_ids = env_ids.to(self.device)
        scene_env_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else env_ids
        )
        object_state = self.simulator.get_object_root_state(env_ids)
        published = {
            "scene.object_pos": object_state.root_pos,
            "scene.object_rot": object_state.root_rot,
            "scene.object_valid_mask": scene_lib.get_per_object_valid_mask(
                scene_env_ids
            ),
        }
        if scene_lib.has_pointclouds:
            published["scene.neutral_pointclouds"] = (
                scene_lib.get_scene_neutral_pointcloud(scene_env_ids)
            )
        return published


class ContactPublisher:
    """The simulator's contact matrix and its name-to-axis maps.

    On the env's publish list only when ``simulator.contacts_enabled``; the
    matrix is read once per context, for the rows asked for.
    """

    name = "contacts"

    def __init__(self, simulator):
        self.simulator = simulator

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        layout = self.simulator.contact_layout
        return {
            "contact_forces": self.simulator.get_contact_forces(env_ids),
            "contact_body_index": layout.body_index,
            "contact_partner_index": layout.partner_index,
        }


class StaticPublisher:
    """The env's two constant body-id tensors: the selected contact rows and
    the bodies whose contact does not terminate. Constructed with the values,
    so it holds no reference to the env."""

    name = "static"

    def __init__(
        self, contact_body_ids: Tensor, non_termination_contact_body_ids: Tensor
    ):
        self.contact_body_ids = contact_body_ids
        self.non_termination_contact_body_ids = non_termination_contact_body_ids

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        return {
            "contact_body_ids": self.contact_body_ids,
            "non_termination_contact_body_ids": self.non_termination_contact_body_ids,
        }


__all__ = [
    "ContactPublisher",
    "HistorySampler",
    "ScenePublisher",
    "StatePublisher",
    "StaticPublisher",
    "TerrainPublisher",
    "prefix_names",
]
