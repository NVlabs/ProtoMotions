# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The environment: the simulator and what surrounds it, run as named phases.

The environment holds the simulator and the objects around it (the motion
library, the history buffer, the sensors, the control components), runs
``step()`` and ``reset()`` as fixed sequences of named phases, and keeps its
per-env state in small holder objects that it loops over for reset,
save/restore and context building. Those holders are the "owners" of
``owners.py``, whose docstring defines the terms used below (owner, row
owner, snapshot owner, publisher, advancer, forwarder). Tasks are composed
through ``EnvConfig``'s component dicts (observations, rewards, terminations,
control components), not by subclassing.

What the environment holds, declared per instance by ``build.build_env`` (the
objects, in construction order) and ``build.declare_owners`` (the four
lists); ``build.py`` documents both orders with their reasons:

- the constructor arguments ``config``, ``robot_config``, ``device``,
  ``terrain``, ``scene_lib``, ``simulator``, with ``num_envs`` and ``dt``
  read off the simulator;
- the owners of the per-env state (``owners.py``; the reset chain's in the
  ``reset`` package): ``_episode`` (the progress, reset, terminate and
  reward buffers), ``_contact_memory``, ``_action_stage``, ``_motion`` (the
  library, its manager and the clock) and ``_spawn`` (the respawn offsets);
- the reset helpers, which hold no per-env state of their own:
  ``_recovery`` (the fall-pose cache), ``_pose`` (the reset pose) and
  ``_perturbation`` (the reset noise);
- the other objects: ``sensors`` (None without a suite), ``_state_history``
  with its ``_history_sampler`` (``publishers.py``), ``control_manager`` with
  ``_controls`` (the ``ControlsState`` owner that puts it on the snapshot
  and advance walks), the viewer's ``_markers`` (``markers.py``) and
  ``_key_bindings``; ``sensors`` and ``control_manager`` are public because
  deployment and the evaluators read them, the rest is the env's own;
- the managers: one ``_component_manager`` and the three MDP managers
  ``_observations``, ``_rewards``, ``_terminations`` (``mdp_managers.py``);
- the four lists ``_advance_owners``, ``_row_owners``, ``_snapshot_owners``,
  ``_publish_owners``;
- the cached all-env context ``_current_context``, the flag ``_advancing``
  (True while the advance owners run), the perturbation switch
  ``_perturbations_enabled``, and ``extras``, which ``step`` rebuilds.

The env's public attributes and methods are ``step``, ``reset``, ``observe``,
``close``, the snapshot pair ``save_state`` / ``restore_state``, the
checkpoint trio ``get_state_dict`` / ``load_state_dict`` / ``get_task_id``,
``get_obs``, ``context``, the perturbation switch, the two body-id tensors,
a few getters, and the forwarders to the owners (the *Forwarders* section of
the class, each with its in-tree callers; the tensor forwarders are read-only
names over the owners' buffers, written in place, and the env's own code reads
the owners directly); ``tests/test_base_env_surface.py`` pins that list, so nothing
becomes public by accident. The context vocabulary is ``context_views.py``.

## The step, as a list of phases

``step()`` is one list of one-line phase calls. ``STEP_PHASES`` names the
phases in order and ``step()``'s one-word comments mark each one; those are
the only two copies of the list (a test records the phases and compares them
with ``STEP_PHASES``). ``step()`` never resets an env: it reports the done
flags and the agent calls ``reset``. Why this order:

- The clock ticks right after physics, so the timeout kernel and the grace
  mask see the step count AFTER this step. The timeout kernel fires at
  ``progress_buf >= max_episode_length - 1``, one step early; it is kept as
  it is for parity (changing it is a later behaviour PR).
- The history records the physics result BEFORE the advance owners run, so
  the state the context publishes is the measurement. Control components
  must not read ``env.context`` while they advance: it would show the
  reference before the re-alignment, so the ``context`` property raises
  while ``_advancing`` is set. They read the motion manager, the simulator
  or the history directly.
- There are two robot-state reads. The first (``read``) is the measurement
  the history and the context carry, and the state the advance owners
  receive. The advance owners run after it and can move the robot (the
  animation players queue poses through ``TaskInputs.teleport``, applied once
  every control advanced), so the ``raw/*`` extras are read again in
  ``report`` and may legitimately differ from the observations' source.
- Observations are computed before rewards and terminations from the one
  published context, so all three see the same state; rewards then
  terminations is the order the logging keys of ``extras`` always had.
- The keyboard flag is consumed AFTER the checks: it becomes a reset request
  on the episode buffers (``EpisodeBuffers.request_reset``, a mask, never a
  write into the clock) that the NEXT step's check takes, which is when the
  viewer's reset always took effect. A control component may consume the
  flag earlier, in its own ``advance`` (the kinematic replay does, through
  ``TaskInputs.viewer_reset``), and then the env's phase sees nothing.

## The reset, as a list of phases

``reset()`` is the same shape: ``RESET_PHASES`` names the phases, ``reset()``'s
comments mark them. ``reset(None)`` turns into an explicit all-env tensor at
the top and an empty list returns early. The first four phases (assign,
pose, recover, perturb) are the reset package's work and the ``reset``
package docstring says why they run in that order. The remaining phases run
in the order they always had: one simulator write, the row owners' clears,
the read-back of the physical post-reset measurement that seeds the history
for every kind of reset (reference resets included; no synthetic pre-episode
poses), the sensor reset, the history fill, the control components' row
reset (``reset_rows``, on the fresh motion ids and times and that
measurement), then ``observe(env_ids)``.
``observe`` is the one place a subset context is built; it always passes the
explicit row tensor so the component manager never compiles for a reset (see
its docstring).

## The four lists

``_advance_owners`` is the step's advance phase (the motion clock, the
control components, the spawn re-alignment; static, reading whatever motion
manager is installed); ``_row_owners`` the rows ``reset`` clears;
``_snapshot_owners`` what ``save_state`` collects and ``restore_state``
writes back, in this order; ``_publish_owners`` what ``_publish`` merges into
the context, keyed by each owner's unique ``name`` (each control component's
dict joins under its own producer name, ``control/<component name>``; ``dt``
and ``env_ids`` are the context's own). ``build.wiring_report`` prints all
four.

## The two checks

Before anything is built, ``train_agent`` and ``inference_agent`` run the
config-only check (``context_check.run_config_binding_check``): every bound
path of every observation, reward, termination and evaluation component
against the names the configs imply. It raises on a binding this
configuration can never satisfy and logs the ones that depend on the motion
file. On the first context, ``build.validate_bindings`` runs the build-time
check that the published and the declared names agree (the ``check`` phase
of ``build_env``): each control component published the namespaces its
class declares, every published name is declared, every declared name
without a ``requires=`` tag is present, no tagged name is published by a
configuration that does not satisfy its tag, every binding is declared and
present (an absent name says which fact is missing), and the component
manager's own checks pass. The live motion-shard switch
(``install_motion_lib``) re-runs it.

## How to add state

The common case is task state: put it in your control component, where it
is saved, restored, advanced and published for you (see "How to add a
control component"). The recipe below is for the environment's own buffers.
Put the buffer in an owner: an existing one, or a new class in ``owners.py``
implementing the walks it needs (``reset_rows`` for rows a reset clears,
``save_state`` / ``load_state`` for the snapshot, ``publish`` for the
context, ``advance`` for the step; ``owners.py`` defines the terms).
Construct it in ``build_env`` at the phase its dependencies allow and put it
on the lists it belongs to in ``declare_owners``.
``test_env_snapshot_completeness.py`` fails for a per-env tensor that does
not come back after ``restore_state`` and for an owner built but left off
the snapshot list.

## How to add a context field

1. Declare the name in ``context_views.py``: a ``FieldPath()`` under its
   namespace, with ``requires=`` naming what a configuration needs for the
   name to exist, if a configuration may lack it (the requirement names are
   listed in that module's docstring).
2. Return it from the owning publisher's or control component's ``publish``
   under that path. The build fails if the two disagree: a published name
   nobody declared, a declared untagged name nobody published, or a tagged
   name published by a configuration that does not satisfy the tag.
3. Bind it: kernels spell it as ``EnvContext.<namespace>.<name>`` and read it
   as ``context["<namespace>.<name>"]``.

## How to add a context namespace

For a control component's namespace: write a class of ``FieldPath``
declarations in ``context_views.py``, hang it on ``EnvContext`` with one
``control_namespace(...)`` line, and list the namespace in the component's
``namespaces``. A binding to it without that component is refused with "no
control component publishes ...". For a namespace the environment itself
publishes, see ``build.py``'s docstring, where the publish list lives.
Nothing else is registered: ``NAMESPACES`` is read off ``EnvContext``.

## How to add a control component

Subclass ``ControlComponent`` (``control/base.py``): declare ``namespaces``
(the control namespaces it publishes; empty if none), implement ``publish``,
``step`` and ``reset``, give it a config dataclass with a ``_target_``, and
list it in ``config.control_components``. ``_publish`` merges the
component's dict under its own producer name (``control/<name>``), and the
build check compares the declaration with what that component published; the
config-only check reads the declaration off the class, so a binding to the
namespace is refused before a build when the component is not configured.
If one of your context fields is tagged with a switch of your config
(``requires="style"``), the default ``provides`` reads ``config.style``;
override ``provides`` only when the switch is not a plain field.
"""

from functools import cached_property
from typing import Dict, Optional, TYPE_CHECKING, Tuple

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.envs.context import FrozenContext, control_producer
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.sensors.runtime import SensorRuntime
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.control.manager import ControlManager
from protomotions.envs.base_env.build import build_env, validate_bindings
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.base_env.markers import Markers
from protomotions.envs.base_env.mdp_managers import (
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from protomotions.envs.base_env.owners import (
    ActionStage,
    Advancer,
    ContactMemory,
    ControlsState,
    EpisodeBuffers,
    Publisher,
    RowOwner,
    SnapshotOwner,
)
from protomotions.envs.base_env.publishers import HistorySampler
from protomotions.envs.randomness import Randomness
from protomotions.envs.base_env.reset import (
    RecoveryBank,
    PlacedReference,
    ReferenceMotion,
    ResetPerturbation,
    ResetPose,
    SpawnPlacer,
)
from protomotions.simulator.base_simulator.user_interface import KeyBindingScope
from protomotions.components.pose_lib import build_body_ids_tensor

from protomotions.robot_configs.base import RobotConfig

# The phases of ``step``, in order (the module docstring says why).
STEP_PHASES = (
    "act",
    "physics",
    "clock",
    "read",
    "sense",
    "record",
    "advance",
    "publish",
    "observe",
    "score",
    "report",
    "keyboard",
)

# The phases of ``reset``, in order (the module docstring and the ``reset``
# package say why).
RESET_PHASES = (
    "assign",
    "pose",
    "recover",
    "perturb",
    "write",
    "clear",
    "read",
    "sense",
    "history",
    "controls",
    "observe",
)

if TYPE_CHECKING:
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.motion_lib import MotionLib


class BaseEnv:
    """The environment: the simulator, the terrain and scene, the reference
    motion, and the MDP components of ``EnvConfig`` run over the context the
    owners publish. See the module docstring for what it holds and why the
    step and the reset run in the order they do.

    Example:
        >>> env = BaseEnv(config, robot_config, device, terrain, simulator,
        ...               scene_lib, motion_lib)
        >>> obs, _ = env.reset()
        >>> obs, rewards, dones, terminated, extras = env.step(actions)
    """

    def __init__(
        self,
        config: EnvConfig,
        robot_config: RobotConfig,
        device: torch.device,
        terrain: "Terrain",
        simulator: Simulator,
        scene_lib: "SceneLib",
        motion_lib: "MotionLib",
        *args,
        **kwargs,
    ):
        """Store the arguments and build everything else through
        ``build.build_env`` (which documents the construction order).

        ``terrain``, ``scene_lib`` and ``motion_lib`` are always provided (an
        empty scene library / motion library stands for none); ``simulator``
        is the pre-created shell the build initialises.
        """
        self.config = config
        self.robot_config = robot_config
        self.device = device
        self.terrain = terrain
        self.scene_lib = scene_lib
        self.simulator = simulator
        # Pushes and reset noise; switched off by the evaluator, see
        # set_perturbations_enabled.
        self._perturbations_enabled = True
        self.num_envs = simulator.num_envs
        self.dt = self.simulator.dt
        # The all-env context: built once per step (the publish phase) and
        # reused by observations, rewards and terminations until the next
        # step or reset drops it.
        self._current_context: Optional[FrozenContext] = None
        # True while the advance owners run (the motion clock, the control
        # components, the re-alignment): reading ``context`` then would show
        # the state before the re-alignment, so the property refuses.
        self._advancing = False
        build_env(self, motion_lib)

    # What the env holds, declared per instance by build.build_env (the
    # objects, in construction order) and build.declare_owners (the four
    # lists); see the module docstring. Beyond these: the constructor
    # arguments, its three flags (_perturbations_enabled, _current_context,
    # _advancing) and ``extras``, which step() rebuilds.
    randomness: Randomness  # the per-stage random generators (env.randomness)
    _motion: ReferenceMotion  # the motion library, its manager, the clock
    _episode: EpisodeBuffers  # progress / reset / terminate / reward buffers
    _contact_memory: ContactMemory  # last step's contact strengths
    _action_stage: ActionStage  # the raw and processed action of this step
    _spawn: SpawnPlacer  # respawn offsets; the per-step re-alignment
    reference: PlacedReference  # the tasks' query: clips placed at the spawn
    _recovery: RecoveryBank  # the fall-pose cache and the recovery swap
    _pose: ResetPose  # default / reference / mixed reset pose
    _perturbation: ResetPerturbation  # the reset noise, under the switch
    sensors: Optional[SensorRuntime]  # only with a declared sensor suite
    _key_bindings: Optional[KeyBindingScope]  # the viewer's keys; None once closed
    control_manager: ControlManager  # the control components
    _markers: Markers  # the viewer's markers: terrain height map + controls'
    _component_manager: ComponentManager  # one kernel cache for the three groups
    _observations: ObservationManager  # observation kernels and the obs buffer
    _rewards: RewardManager  # reward kernels; the reward and its logging
    _terminations: TerminationManager  # timeout, reset requests, the merged rules
    _history_sampler: HistorySampler  # a state read -> the ring's sample
    _state_history: StateHistoryBuffer  # the history ring the context reads
    _controls: ControlsState  # the control manager on the snapshot/advance walks
    _row_owners: Tuple[RowOwner, ...]
    _snapshot_owners: Tuple[SnapshotOwner, ...]
    _publish_owners: Tuple[Publisher, ...]
    _advance_owners: Tuple[Advancer, ...]

    ###############################################################
    # Forwarders to the state owners
    ###############################################################
    # The per-env buffers live in the owners (``owners.py``, the ``reset``
    # package); the agents, the evaluators and the control components keep
    # reading them off the env under the names they always had. One property
    # per owner field, each with a docstring naming its in-tree callers. Every
    # tensor getter returns the owner's tensor itself, so
    # ``env.progress_buf[ids] = 0`` (a control component) writes the owner's
    # buffer; the tensor forwarders have no setter, because the buffer IS the
    # owner's and rebinding it would silently split the aliases. The one
    # setter that remains (``max_episode_length``) assigns a scalar the
    # curriculum replaces; the reference motion is reached as its owner
    # (``reference_motion``), which the shard switcher clears and re-installs
    # through ``install_motion_lib``. The env's own code reads
    # the owners directly (``self._episode.progress_buf`` ...);
    # ``tests/test_base_env_surface.py`` pins these names, so a forwarder added
    # here is a deliberate addition to that list.

    # --- EpisodeBuffers ------------------------------------------------------
    @property
    def progress_buf(self) -> Tensor:
        """Episode step counter. Read and written in place by the control
        components (steering, target, path follower, kinematic replay,
        external kinematic, masked-mimic tasks)."""
        return self._episode.progress_buf

    @property
    def reset_buf(self) -> Tensor:
        """Which envs need a reset: ``step``'s third return, read by the
        steering, target, kinematic replay and external kinematic controls."""
        return self._episode.reset_buf

    @property
    def terminate_buf(self) -> Tensor:
        """Which envs terminated: ``step``'s fourth return and
        ``extras["terminate"]``; the same four controls read it."""
        return self._episode.terminate_buf

    @property
    def rew_buf(self) -> Tensor:
        """Last step's reward, ``step``'s second return (the agent keeps the
        returned object; the parity harness checks it is this buffer)."""
        return self._episode.rew_buf

    @property
    def max_episode_length(self) -> int:
        """Timeout length. The agent's curriculum assigns it between epochs
        (``agents/base_agent/agent.py``); the CLI overrides and
        ``utils/config_utils.py`` set it through the config."""
        return self._episode.max_episode_length

    @max_episode_length.setter
    def max_episode_length(self, value: int) -> None:
        self._episode.max_episode_length = value

    # --- ReferenceMotion -----------------------------------------------------
    @property
    def reference_motion(self) -> ReferenceMotion:
        """The reference-motion owner: ``motion_lib`` (the installed library)
        and ``motion_manager`` (the current manager, or None without motions).
        Read by the agents (``agent.py``, the shard switcher, ADD), the mimic
        evaluator and the examples; the controls read the reference through
        their inputs, not here. A live shard switch clears it
        (``ReferenceMotion.clear``) and re-installs through
        ``install_motion_lib``; nothing else assigns to it."""
        return self._motion

    def install_motion_lib(self, motion_lib: "MotionLib") -> None:
        """The live motion-shard switch (``agents/base_agent/motion_shards.py``):
        install a prebuilt library in place of the current one and re-run the
        build-time check that the published and the declared names agree.

        ``ReferenceMotion.install`` validates the library, smooths its
        contacts and builds the new manager (the construction-time install
        goes through it too, from ``build_env``). The advance list is static,
        so nothing is rebuilt; but the switch can change which of the
        motion-file-dependent ``mimic.ref_state.*`` names the reference
        carries, so the build-time check that published and declared names
        agree runs again. The check builds one context (not a logged call,
        no random draws) and the cache is emptied again, as before.
        """
        self._motion.install(motion_lib)
        self._current_context = None
        validate_bindings(self)
        self._current_context = None

    ###############################################################
    # Getters
    ###############################################################
    def is_simulation_running(self):
        """Whether the physics simulation is running (the examples' loops)."""
        return self.simulator.is_simulation_running()

    def get_obs(self):
        """Every cached observation (terrain, scene, dynamic components), as
        copies; see ``ObservationManager.get_obs``."""
        return self._observations.get_obs()

    def consume_reset_request(self) -> bool:
        """Return and consume a user-interface reset request."""
        return self._key_bindings.reset.consume()

    def _all_env_ids(self) -> Tensor:
        """A fresh index vector over every env (the explicit "all rows")."""
        return torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    @cached_property
    def contact_body_ids(self) -> torch.Tensor:
        """Common robot IDs corresponding to the selected contact-matrix rows."""
        ids = (
            self.simulator.contact_layout.common_body_ids
            if self.simulator.contacts_enabled
            else ()
        )
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @cached_property
    def non_termination_contact_body_ids(self) -> torch.Tensor:
        """Body indices that don't trigger termination on contact."""
        body_names = self.robot_config.kinematic_info.body_names
        if self.robot_config.non_termination_contact_bodies == "all":
            return build_body_ids_tensor(body_names, body_names, self.device)
        else:
            return build_body_ids_tensor(
                body_names,
                self.robot_config.non_termination_contact_bodies,
                self.device,
            )

    ###############################################################
    # Perturbation switch
    ###############################################################
    @property
    def perturbations_enabled(self) -> bool:
        """Whether pushes and reset noise are applied (see set_perturbations_enabled)."""
        return self._perturbations_enabled

    def set_perturbations_enabled(self, enabled: bool) -> None:
        """Switch the physical perturbations on or off.

        Perturbations are the simulator's push randomization and the reset
        noise from ``robot_config.reset_noise``; the evaluator switches them
        off so success metrics measure tracking, not robustness. Sensor noise
        and action noise are not affected. Switching only writes flags: it
        draws no random number and leaves the pending push schedule as it is,
        so re-enabling resumes the pushes that were pending. Pushes are only
        re-enabled when the simulator config has push randomization.
        """
        self._perturbations_enabled = bool(enabled)
        self.simulator.set_push_enabled(
            self._perturbations_enabled and self.simulator.has_push_randomization
        )

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, action: Tensor):
        """Step the environment forward one timestep.

        One phase per line, in the order ``STEP_PHASES`` declares; the module
        docstring explains the order. Nothing is reset here: the done flags
        are returned and the agent calls ``reset``.

        Args:
            action: Raw action tensor from the policy [num_envs, num_actions]

        Returns:
            obs, rewards, dones, terminated, extras
        """
        self.extras = {}
        # act: store the raw action, process it and store the simulator command
        processed = self._action_stage.act(action)
        # physics
        self.simulator.step(processed, markers_callback=self._markers.state)
        # clock
        self._episode.tick()
        # read: the measurement the history and the context carry
        state = self.simulator.get_robot_state()
        # sense
        estimates = self.sensors.step(state) if self.sensors else None
        # record: the new state is in the history; the cached context is stale
        self._state_history.push(self._history_sampler.sample(state, estimates))
        self._current_context = None
        # advance: motion clock, control components (their logging joins the
        # extras), re-alignment; the context is not readable until this is over
        self._advancing = True
        try:
            for owner in self._advance_owners:
                logging = owner.advance(state)
                if logging:
                    self.extras.update(logging)
        finally:
            self._advancing = False
        # publish: the one context observations, rewards and terminations read
        ctx = self._current_context = self._publish()
        # observe
        self._observations.compute(ctx, self._all_env_ids())
        # score: rewards, then the done checks (the timeout, the pending reset
        # requests and every termination entry, the controls' included); both
        # managers hand back their logging
        episode = self._episode
        reward, reward_logging = self._rewards.compute(ctx, episode.progress_buf)
        episode.rew_buf[:] = reward
        self.extras.update(reward_logging)
        reset, terminated, term_logging = self._terminations.compute(
            ctx,
            episode.progress_buf,
            episode.max_episode_length,
            episode.take_reset_request(),
        )
        episode.reset_buf[:], episode.terminate_buf[:] = reset, terminated
        self.extras.update(term_logging)
        # report: the done flag, the raw state (a second read) and this step's
        # contact strengths for the next impact penalty
        self.extras["terminate"] = episode.terminate_buf
        self.extras.update(self._raw_state_extras(ctx))
        if self.simulator.contacts_enabled:
            self._contact_memory.update(ctx["contact_forces"])
        # keyboard: the viewer's reset key asks every env to reset at the next check
        if self.consume_reset_request():
            episode.request_reset(self._all_env_ids())
        obs = self.get_obs()
        return (
            obs,
            episode.rew_buf,
            episode.reset_buf,
            episode.terminate_buf,
            self.extras,
        )

    def on_epoch_end(self, current_epoch: int):
        """Hook the agent calls at the end of each training epoch; a no-op here."""
        pass

    def observe(self, env_ids: Tensor) -> None:
        """Compute the observations of ``env_ids`` from a context over those rows.

        The one place a subset context is built: ``reset``'s last phase (the
        constructor computes the dynamic components from the cached all-env
        context instead, see ``ObservationManager.compute_components``).
        ``env_ids`` is always the explicit row tensor,
        never None for "every env": the component manager compiles only for a
        context whose ``env_ids`` is None (the all-env context ``step``
        publishes), and post-reset observations run uncompiled; the parity
        harness's ``compile_path`` key sees a slip. The cached all-env context
        is dropped first; the subset context is not cached.
        """
        self._current_context = None
        self._observations.compute(self._publish(env_ids), env_ids)

    def _raw_state_extras(self, ctx: FrozenContext) -> Dict[str, Tensor]:
        """The ``raw/<field>`` extras: the robot state read again after the
        advance owners ran (they can move the robot, see the module
        docstring), flattened per body for logging and evaluation metrics.
        With complete contact labels the contact flags are the context's."""
        rbs: RobotState = self.simulator.get_robot_state()
        if self.simulator.has_complete_contact_labels:
            rbs.rigid_body_contacts = ctx["current.rigid_body_contacts"]
        extras: Dict[str, Tensor] = {}
        for k, _ in rbs.get_shape_mapping(flattened=True).items():
            value = rbs.flatten_bodies(k)
            if value is not None:
                extras[f"raw/{k}"] = value
        return extras

    ###############################################################
    # The published context
    ###############################################################
    @property
    def context(self) -> FrozenContext:
        """The context over every env: built once after the step, cached until
        the next step or reset invalidates it (reset rows get their own subset
        context from ``_publish(env_ids)``, which is never cached). Not valid
        while the control components advance: a read there raises."""
        if self._advancing:
            raise RuntimeError(
                "env.context is not valid while the control components advance: "
                "read the motion manager, the simulator or the history directly"
            )
        if self._current_context is None:
            self._current_context = self._publish()
        return self._current_context

    def _publish(self, env_ids: Optional[Tensor] = None) -> FrozenContext:
        """The context for ``env_ids`` (None: every env): what the publish
        owners and the control components publish, merged into one read-only
        map keyed by producer name (each control component under its own,
        ``control/<name>``, so the check can tell who published what). See
        the module docstring for the list and the vocabulary."""
        parts = {owner.name: owner.publish(env_ids) for owner in self._publish_owners}
        for name, component in self.control_manager.components.items():
            parts[control_producer(name)] = component.publish(env_ids)
        return FrozenContext.merge(parts, env_ids=env_ids, dt=self.dt)

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset(
        self,
        env_ids=None,
        sample_flat=False,
        force_default_mask=None,
        disable_motion_resample=False,
    ):
        """Reset environments and return observations.

        One phase per line, in the order ``RESET_PHASES`` declares; the module
        docstring and the ``reset`` package explain the order.

        Args:
            env_ids: Environment IDs to reset, or None to reset all
            sample_flat: If True, spawn on flat terrain (useful for evaluation)
            force_default_mask: Optional boolean mask [len(env_ids)]: these
                envs reset from the default pose instead of a reference clip
                (a mixed reset). No in-tree caller passes it; kept on purpose
                as the public mixed-reset hook, see
                ``ReferenceMotion.ref_reset_envs`` and ``ResetPose.compute``.
            disable_motion_resample: If True, skip resampling motions (use
                existing motion_ids/times); the evaluators replay specific
                motions this way.

        Returns:
            obs: Dictionary of observation tensors
            info: Dictionary containing reset metadata (currently empty)
        """
        if env_ids is None:
            env_ids = self._all_env_ids()

        if len(env_ids) == 0:
            return self.get_obs(), {}

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        # assign: which envs reset from a reference clip, and sample their clips
        ref_env_ids, motion_ids, motion_times = self._motion.ref_reset_envs(
            env_ids, force_default_mask, disable_motion_resample
        )
        # pose: default, reference or mixed, placed at a freshly sampled spawn
        new_states, new_object_states = self._pose.compute(
            env_ids, ref_env_ids, motion_ids, motion_times, sample_flat
        )
        # recover: a Bernoulli subset starts from a cached fall pose instead
        new_states, _ = self._recovery.swap_in(
            env_ids, new_states, self._spawn.respawn_root_offset
        )
        # perturb: the reset noise, under the perturbation switch
        self._perturbation.apply(new_states, self._perturbations_enabled)
        # write
        self.simulator.reset_envs(new_states, new_object_states, env_ids)
        # clear: the row owners
        for owner in self._row_owners:
            owner.reset_rows(env_ids)
        # read: the physical post-reset measurement seeds everything below
        state = self.simulator.get_robot_state(env_ids)
        # sense
        estimates = self.sensors.reset(env_ids, state) if self.sensors else None
        # history: every slot of the reset rows holds the measurement
        self._state_history.reset(
            env_ids, self._history_sampler.sample(state, estimates, env_ids)
        )
        self._current_context = None
        # controls: the control components see the fresh motion ids and times
        # and the post-reset robot state
        self.control_manager.reset_rows(env_ids, state)
        # observe: the reset rows only, from a subset context
        self.observe(env_ids)

        return self.get_obs(), {}

    ###############################################################
    # Checkpoint, snapshot, close
    ###############################################################
    def get_state_dict(self):
        """Environment state for checkpointing: the motion manager's, if any."""
        motion_manager = self._motion.motion_manager
        if motion_manager is not None:
            return {"motion_manager": motion_manager.get_state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        """Load a ``get_state_dict`` checkpoint back into the motion manager."""
        motion_manager = self._motion.motion_manager
        if motion_manager is not None:
            motion_manager.load_state_dict(state_dict["motion_manager"])

    def get_task_id(self):
        """Task identifier for logging and checkpointing: the motion file name,
        or ``"null"`` without motions."""
        if self._motion.motion_manager is not None:
            return self._motion.motion_lib.motion_file.split("/")[-1]
        return "null"

    def save_state(self) -> dict:
        """Snapshot physical, sensor, task and observation state for evaluation.

        Training resumes with the cached observations after evaluation. Preserve
        those caches as well as history, rather than recomputing a new sample.
        One entry per snapshot owner, collected in restore order.
        """
        return {owner.name: owner.save_state() for owner in self._snapshot_owners}

    def restore_state(self, snapshot: dict) -> None:
        """Write a ``save_state`` snapshot back, owner by owner, in list order.

        The order is the one ``build.declare_owners`` documents: the
        simulator's contacts are flushed before its poses are written back,
        and the backend state, the action history and the reset age go back
        after that write (which clears them). Nothing here advances physics:
        that would put simulator truth one tick ahead of the restored sensor
        ring and history, and could apply pushes or actuator noise during the
        restore. Restoring the sensor history neither draws noise nor
        reconstructs body state.
        """
        for owner in self._snapshot_owners:
            owner.load_state(snapshot[owner.name])
        self._current_context = None

    def close(self) -> None:
        """Release control-component and env-owned UI handles, then close
        the simulator. Safe to call multiple times."""
        control_manager = getattr(self, "control_manager", None)
        if control_manager is not None:
            for component in control_manager.components.values():
                component.close()

        ui = getattr(self, "_key_bindings", None)
        if ui is not None:
            ui.unregister_all()
            self._key_bindings = None

        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()
