# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owners of ``BaseEnv``'s per-environment state.

Glossary, one sentence each:

- An **owner** is a small object that holds some of the environment's
  per-env state (or adapts an object that does) and implements the walks
  below that concern it.
- A **row owner** has per-env rows that ``reset`` clears or re-seeds
  (:class:`RowOwner`, ``reset_rows(env_ids)``).
- A **snapshot owner** can hand out a copy of its state and copy it back in
  place, for the evaluator's save/restore around an evaluation rollout
  (:class:`SnapshotOwner`, ``save_state()`` / ``load_state(snapshot)``).
- A **publisher** contributes names to the context: ``publish(env_ids)``
  returns a ``{path: value}`` dict whose paths are the vocabulary of
  ``context_views.py`` (:class:`Publisher`).
- An **advancer** moves its state on by one env step after physics, between
  the history push and the publish (:class:`Advancer`, ``advance()``).
- A **forwarder** is a property on the env that returns an owner's tensor
  under the name callers always used (the *Forwarders* section of ``env.py``).

``BaseEnv`` used to hold every per-env buffer itself and to spell out, inside
``reset``, ``save_state`` and ``restore_state``, what each one needs. Here the
buffers live in owners, and the env keeps one declared list per walk
(``build.declare_owners`` builds the lists and ``build.py`` documents the
reasons for their order).

An owner implements whichever walks concern it. The split between the
modules of this package is by walk and by subject: this module holds the
env's own per-env buffers and the adapters on the snapshot walk;
``publishers.py`` holds the stateless adapters that are only on the context
walk; the ``reset`` package holds the reset chain's owners (the spawn placer,
on the snapshot and the step walk; the reference motion, on the step walk)
and helpers.

Three owners here hold tensors of their own: :class:`EpisodeBuffers`,
:class:`ContactMemory` and :class:`ActionStage` (these are also the only row
owners). The rest are adapters over objects that already have a save/restore
pair (the simulator, the state history, the sensor runtime, the terrain and
scene observation caches, the control manager), so that the snapshot walk
covers them with the same two calls. Adapters do not define ``reset_rows``:
the reset chain writes those objects itself, in its own named phases.
:class:`ControlsState` is on both the snapshot and the step walk: the
control components are saved, restored and advanced through their manager.

A snapshot is ``{owner.name: owner.save_state()}``. An owner with tensors of
its own returns a dict of clones under the keys the env's flat snapshot used
before this split; an adapter returns the adapted object's own snapshot (a
``HistorySample``, a ``SensorSnapshot``, the simulator's dict ...) and hands
it straight back on ``load_state``. Snapshots live in memory only (the
evaluator saves before its rollout and restores after), so there is no
compatibility to keep with the flat layout.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol

import torch
from torch import Tensor

from protomotions.envs.action.config import ActionConfig
from protomotions.envs.base_env.episode_tensors import EpisodeTensors
from protomotions.envs.contact_features import contact_magnitudes
from protomotions.simulator.base_simulator.simulator_state import ResetState


class RowOwner(Protocol):
    """What the reset-row walk needs: per-env rows that reset clears or re-seeds."""

    name: str

    def reset_rows(self, env_ids: Tensor) -> None:
        """Clear or re-seed the rows of ``env_ids`` (called inside ``reset``)."""


class SnapshotOwner(Protocol):
    """What the save/restore walk needs: a named, owned snapshot and its inverse."""

    name: str

    def save_state(self) -> Any:
        """An owned snapshot (cloned tensors, or the adapted object's own snapshot)."""

    def load_state(self, snapshot: Any) -> None:
        """Copy a ``save_state`` result back, in place."""


class Publisher(Protocol):
    """What the context walk needs: the names and values this owner publishes."""

    # Unique on the env's publish list; the producer a duplicate-key error names.
    name: str

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        """``{path: value}`` for ``env_ids`` (None: every env), sliced to those rows.

        Values are the owner's own tensors or views of them; an owner clones
        only where the context must not share a buffer the step overwrites.

        Must be side-effect free, with no exceptions: the env builds a context
        whenever one is asked for (``env.context`` after a drop, a reset's
        subset, the build check), so a publisher that changed state or drew
        a random number would make reading the context an action. State that
        depends on the simulator is updated in ``step`` (the advance phase)
        or in ``reset``, never while publishing.
        ``tests/test_publish_is_pure.py`` publishes twice on the real env and
        checks that no owner's saved state and no random stream moved.
        """


class Advancer(Protocol):
    """What the step walk needs: state that moves on by one step after physics."""

    # Names the owner in the wiring report and in the step-order tests.
    name: str

    def advance(self, state: Any) -> Optional[Mapping[str, Tensor]]:
        """Advance by one env step, given the robot state the env just read.

        Called by ``BaseEnv.step`` after the new state is in the history and
        before the context is published, in the order of ``_advance_owners``
        (``build.py`` declares the list and the reasons for its order).
        Returns logging for ``extras`` (the control components'), or None.
        """


def select_rows(tensor: Tensor, env_ids: Optional[Tensor]) -> Tensor:
    """The rows of ``env_ids``, or the whole tensor for a context over every env."""
    return tensor if env_ids is None else tensor[env_ids]


###############################################################
# Owners with tensors of their own
###############################################################
class EpisodeBuffers:
    """Episode counters, the per-step done/reward outputs and the reset request.

    ``progress_buf`` counts the steps of the current episode; ``reset_buf`` and
    ``terminate_buf`` are the outputs of the last check; ``rew_buf`` the last
    reward. ``max_episode_length`` lives here because the timeout check reads
    it with ``progress_buf`` (the agent's curriculum assigns it).
    ``reset_request`` is the mask of envs asked to reset at the next check (the
    viewer's reset key): a request, not a poisoned clock.
    """

    name = "episode"

    def __init__(self, num_envs: int, device: torch.device, max_episode_length: int):
        self.max_episode_length = max_episode_length
        # The per-env slots live in a registry (the same one the control
        # components use): it resets their rows and holds the snapshot.
        self.state = EpisodeTensors(num_envs, device)
        self.progress_buf = self.state.add("progress_buf", (), torch.long, 0)
        self.reset_buf = self.state.add("reset_buf", (), torch.bool, False)
        self.terminate_buf = self.state.add("terminate_buf", (), torch.bool, False)
        self.reset_request = self.state.add("reset_request", (), torch.bool, False)
        # Until the first reset every env needs one (the agent resets them all
        # at start); a reset row then reads False, the slot's reset value.
        self.reset_buf.fill_(True)
        self.terminate_buf.fill_(True)
        # A step output: the score phase overwrites it before anything reads
        # it, so it is not a slot and never part of the snapshot.
        self.rew_buf = torch.zeros(num_envs, device=device, dtype=torch.float)

    def reset_rows(self, env_ids: Tensor) -> None:
        self.state.reset_rows(env_ids)

    def tick(self) -> None:
        """One more step of every episode (the clock phase of ``BaseEnv.step``)."""
        self.progress_buf += 1

    def restart(self, env_ids: Tensor) -> None:
        """Start the episodes of ``env_ids`` over without a reset (the animation
        players, which teleport the robot and keep stepping)."""
        self.progress_buf[env_ids] = 0

    def request_reset(self, env_ids: Tensor) -> None:
        """Ask for ``env_ids`` to reset at the next done check."""
        self.reset_request[env_ids] = True

    def take_reset_request(self) -> Tensor:
        """The pending requests, cleared (the score phase of ``BaseEnv.step``)."""
        requested = self.reset_request.clone()
        self.reset_request[:] = False
        return requested

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        # A copy: the counter advances before the context is rebuilt.
        return {"progress_buf": select_rows(self.progress_buf, env_ids).clone()}

    def save_state(self) -> dict:
        # The four slots as owned clones (rew_buf is a step output, see __init__).
        return self.state.save_state()

    def load_state(self, snapshot: dict) -> None:
        self.state.load_state(snapshot)


class EpisodeClock:
    """The read-only view of the episode counters a control component gets.

    ``progress_buf`` is the owner's tensor (read only, by contract and by the
    static test over ``envs/control``); ``elapsed_seconds`` is a fresh tensor,
    the same arithmetic the controls did (``progress_buf * dt``).
    """

    def __init__(self, episode: EpisodeBuffers, dt: float):
        self._episode = episode
        self._dt = dt

    @property
    def progress_buf(self) -> Tensor:
        return self._episode.progress_buf

    @property
    def elapsed_seconds(self) -> Tensor:
        return self._episode.progress_buf * self._dt

    @property
    def max_episode_length(self) -> int:
        return self._episode.max_episode_length


class ContactMemory:
    """Last step's per-body contact strength, for the impact-change penalty.

    One row per selected contact body (the simulator's contact layout); zero
    rows when contacts are off.
    """

    name = "contact_memory"

    def __init__(self, num_envs: int, num_rows: int, device: torch.device):
        self.state = EpisodeTensors(num_envs, device)
        self.prev_contact_force_magnitudes = self.state.add(
            "prev_contact_force_magnitudes", (num_rows,), torch.float, 0.0
        )

    def reset_rows(self, env_ids: Tensor) -> None:
        self.state.reset_rows(env_ids)

    def update(self, contact_forces: Tensor) -> None:
        """Keep only the derived per-body strength for the next penalty."""
        self.prev_contact_force_magnitudes.copy_(contact_magnitudes(contact_forces))

    def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
        # A copy: ``update`` overwrites the buffer at the end of the step. On
        # the env's publish list only when the simulator reports contacts.
        return {
            "prev_contact_force_magnitudes": select_rows(
                self.prev_contact_force_magnitudes, env_ids
            ).clone()
        }

    def save_state(self) -> dict:
        return self.state.save_state()

    def load_state(self, snapshot: dict) -> None:
        self.state.load_state(snapshot)


class ActionStage:
    """The current step's policy action, raw and as the simulator command.

    ``action_config`` is the env config's :class:`ActionConfig` (the action
    function bound to its parameters) or None for a pass-through. It is bound
    once, here, at construction: to change how actions are processed, swap the
    stage (``env._action_stage``), not ``env.config.action_config``, which the
    stage no longer reads. Only the current step is kept here; previous actions
    come from the state history.
    """

    name = "action_stage"

    def __init__(
        self,
        action_config: Optional[ActionConfig],
        num_envs: int,
        num_actions: int,
        device: torch.device,
    ):
        self.action_config = action_config
        self.raw = torch.zeros(num_envs, num_actions, dtype=torch.float, device=device)
        self.processed = torch.zeros(
            num_envs, num_actions, dtype=torch.float, device=device
        )
        # The config's tensors move to the action's device once, on first use.
        self.device_ready = False

    def act(self, action: Tensor) -> Tensor:
        """Store the raw action, process it, store and return the result.

        The env is outside the learner's graph: an action that carries one is
        refused, not detached. ``buffer[:] = x`` from a graph-carrying tensor
        would turn the persistent buffer into a graph node and pin the whole
        action graph for its lifetime (the interactive-eval GPU leak the old
        silent ``.detach()`` papered over); the agent's rollouts, the
        evaluators and the inference entry points all step under
        ``torch.no_grad()``, so a caller that reaches this has forgotten to.
        The processed action is held to the same rule: an action function
        whose parameter requires grad would hand back a graph-carrying result
        from a grad-free action.
        Without an action config the action itself is returned (not a copy):
        a simulator may keep the caller's tensor.
        """
        if action.requires_grad:
            raise ValueError(
                "the action handed to env.step carries an autograd graph "
                "(requires_grad=True); the environment is outside the learner's "
                "graph: step under torch.no_grad() or pass action.detach()"
            )
        processed = self._process(action)
        if processed.requires_grad:
            raise ValueError(
                "the action function's output carries an autograd graph "
                "(requires_grad=True) while its input did not: one of its "
                "parameters requires grad; pass that parameter as a plain "
                "tensor (requires_grad=False), the environment is outside the "
                "learner's graph"
            )
        self.raw[:] = action
        self.processed[:] = processed
        return processed

    def _process(self, action: Tensor) -> Tensor:
        if self.action_config is None:
            return action
        if not self.device_ready:
            # The config's tensors move to the action's device once (a moved
            # copy; the env config's own stays where it was).
            self.action_config = self.action_config.to(action.device)
            self.device_ready = True
        # The function reads only the action and the config's parameters; it
        # does not get an env context.
        return self.action_config.apply(action)["processed_action"]

    def reset_rows(self, env_ids: Tensor) -> None:
        self.raw[env_ids] = 0
        self.processed[env_ids] = 0

    def save_state(self) -> dict:
        return {
            "raw_action": self.raw.clone(),
            "processed_action": self.processed.clone(),
        }

    def load_state(self, snapshot: dict) -> None:
        self.raw.copy_(snapshot["raw_action"])
        self.processed.copy_(snapshot["processed_action"])


###############################################################
# Adapters over objects with their own save/restore pair
###############################################################
class SimulatorPhysicalState:
    """Robot (and object) poses plus the backend's solver state.

    Restoring writes the poses back with ``reset_envs`` and only then puts the
    backend state back, because the reset clears it.
    """

    name = "sim_physical"

    def __init__(self, simulator, scene_lib, num_envs: int, device: torch.device):
        self.simulator = simulator
        self.scene_lib = scene_lib
        # Every restore writes every env back; the index vector is a constant.
        self._all_env_ids = torch.arange(num_envs, device=device)

    def save_state(self) -> dict:
        snapshot = {
            "robot_state": self.simulator.get_robot_state().clone(),
            "simulator_backend_state": self.simulator.save_backend_state(),
        }
        if self.scene_lib.num_objects_per_scene > 0:
            snapshot["object_state"] = self.simulator.get_object_root_state()
        return snapshot

    def load_state(self, snapshot: dict) -> None:
        # Every env is about to be teleported back. Clear the simulator's
        # memory of the contacts at the current poses first: otherwise a robot
        # that was standing on the ground here can be launched on the first
        # physics substep after the write-back.
        flush_contact_cache = getattr(self.simulator, "flush_contact_cache", None)
        if flush_contact_cache is not None:
            flush_contact_cache()
        self.simulator.reset_envs(
            ResetState.from_robot_state(snapshot["robot_state"]),
            snapshot.get("object_state"),
            self._all_env_ids,
        )
        # Reset clears solver history; restore it only after the physical write.
        self.simulator.restore_backend_state(snapshot["simulator_backend_state"])


class SimulatorRuntimeState:
    """The simulator's stepping state: action history, latency, reset age ...

    Separate from :class:`SimulatorPhysicalState` because ``reset_envs`` zeroes
    this state: the one real constraint is that it goes back AFTER that write.
    It sits after the observation caches in the snapshot list only to keep the
    restore's historical call order; nothing in between reads it.
    """

    name = "sim_runtime"

    def __init__(self, simulator):
        self.simulator = simulator

    def save_state(self) -> Any:
        return self.simulator.save_state()

    def load_state(self, snapshot: Any) -> None:
        self.simulator.restore_state(snapshot)


class HistoryState:
    """The state history ring; its snapshot is the ring's own ``HistorySample``.

    No ``reset_rows``: re-seeding the ring needs the state read back after the
    simulator write, so it stays a named phase of ``reset``.
    """

    name = "history"

    def __init__(self, state_history):
        self.state_history = state_history

    def save_state(self) -> Any:
        return self.state_history.save_state()

    def load_state(self, snapshot: Any) -> None:
        self.state_history.load_state(snapshot)


class SensorsState:
    """The sensor runtime's generator state and episode-start values (its own
    ``SensorSnapshot``); only in the walk when the experiment declares a
    sensor suite."""

    name = "sensors"

    def __init__(self, sensors):
        self.sensors = sensors

    def save_state(self) -> Any:
        return self.sensors.save_state()

    def load_state(self, snapshot: Any) -> None:
        self.sensors.load_state(snapshot)


class ObservationCache:
    """The env's cached dynamic observations, by name.

    ``buffer`` is the observation manager's dict (``env._observations.buffer``);
    the cache holds that dict, not the env. The snapshot is a dict of clones;
    restoring refills the same dict in place (clear, then cloned entries), so
    observations an evaluation added disappear and the restored tensors are
    copies the snapshot does not share.
    """

    name = "obs_cache"

    def __init__(self, buffer: Dict[str, Tensor]):
        self.buffer = buffer

    def save_state(self) -> Dict[str, Tensor]:
        return {name: tensor.clone() for name, tensor in self.buffer.items()}

    def load_state(self, snapshot: Dict[str, Tensor]) -> None:
        self.buffer.clear()
        self.buffer.update({name: tensor.clone() for name, tensor in snapshot.items()})


class ControlsState:
    """The control components, through their manager: their registries for the
    snapshot walk, their ``advance(state)`` for the step walk, and the
    teleports the animation players queued, applied to the simulator once
    every control advanced."""

    name = "controls"

    def __init__(self, control_manager, simulator):
        self.control_manager = control_manager
        self.simulator = simulator

    def advance(self, state: Any) -> Mapping[str, Tensor]:
        """Advance every control component on the state the env read; apply
        the queued teleports; return the components' logging."""
        logging = self.control_manager.advance(state)
        self.control_manager.teleports.apply(self.simulator)
        return logging

    def save_state(self) -> Any:
        return self.control_manager.save_state()

    def load_state(self, snapshot: Any) -> None:
        self.control_manager.load_state(snapshot)
