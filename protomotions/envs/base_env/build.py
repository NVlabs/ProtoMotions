# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Construction of ``BaseEnv`` in dependency order.

``BaseEnv.__init__`` stores its arguments, initialises its three flags
(``_perturbations_enabled``, the ``_current_context`` cache and
``_advancing``) and calls :func:`build_env`, which creates every object the
env holds and declares the env's four owner lists in the order
``BUILD_PHASES`` names; the comment at each phase of ``build_env`` is the
reason for its place. In short: the random streams come first because every
later object is handed its stage's generator; the motion library is
installed next because the control components read its manager; the
control components are built from one ``TaskInputs`` (no control holds the
env or the simulator, ``ControlManager.check_holds_no_handles`` refuses one
that does) and come before the markers because they contribute markers, and
the markers
before the simulator allocates its GPU memory from the marker set
(``_initialize_with_markers``), so nothing reads simulator state before that
allocation; the history buffer needs the first state read (and the sensors'
first estimate of it); the owner lists need every object built above them,
and the fall-pose generation saves and restores through the snapshot walk
they declare; the binding check needs the first context, which the declared
publishers build; the fall-pose cache is generated after the check and
before the first observations, at the same point relative to every other
random draw as always (it steps the real simulator and draws from the global
and the terrain's generators); the first observations come last, from the
cached all-env context. None of the reset helpers holds the env: the
recovery bank, the reset pose and the spawn placer hold the simulator, the
others hold values.

No reset happens at construction: it would consume random draws that the
agent's first ``reset`` is entitled to.

## The four declared lists

:func:`declare_owners` builds them, with a comment on every entry saying
what it is; this is why each order is what it is.

Advance owners (the step's advance phase, after the history push): the order
the inline statements always had. The clock advances reference time before
the control components read it; the re-alignment runs after them and before
the publish, so observations and rewards compare against the re-anchored
reference. The list is static: the two motion entries read whatever manager
is installed (the live motion-shard switcher replaces it through
``install_motion_lib``) and do nothing without one, or without
``realign_motion_with_humanoid_on_each_step``.

Row owners (the rows ``reset()`` clears): the order the clears always had
(counters, then contact memory, then the action buffers).

Snapshot owners (``save_state()`` collects, and ``restore_state()`` writes
back, in this order): the order the restore always had. The simulator's
contacts must be flushed before the poses are written, and the backend's
solver state can only go back after that write (``reset_envs`` clears it).
``reset_envs`` also zeroes the simulator's action history and reset age, so
``sim_runtime`` is restored late, after the observation buffer and before
the control manager; nothing in between reads it. ``save_state`` uses the
same list, so the pieces are collected in restore order. The recovery bank's
fall-pose cache is on no list: generated once at construction and only ever
read.

Publish owners (what ``BaseEnv._publish(env_ids)`` merges, with each control
component's dict under ``control/<name>``, into one flat read-only
``FrozenContext``, ``protomotions/envs/context.py``): every owner has a
``name``, unique on the list; ``_publish`` keys the merge by it, so a key
published twice is reported with both producers. ``dt`` and ``env_ids`` are
the context's own and are added by ``FrozenContext.merge``. The two contact
entries are on the list only when the simulator reports contacts. The order
carries no dependency (every publisher reads state the step already wrote; a
name published twice is an error): it is the order the old
``_build_context`` filled the tree in.

To add a namespace the environment itself publishes (one that is not a
control component's): declare its class in ``context_views.py`` and hang it
on ``EnvContext`` as a ``NestedField``, with ``requires=`` naming what a
configuration needs for it to exist; write a publisher with a unique
``name`` (``publishers.py``) and put it on the publish list in
:func:`declare_owners`. A new requirement name needs a row in
``context_requirements.REQUIREMENTS`` (the predicate on the configuration
and the sentence a refusal shows); ``tests/test_context_check.py`` fails for
a tag without one.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, TYPE_CHECKING

import torch

from protomotions.envs.base_env.mdp_managers import (
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from protomotions.envs.base_env.owners import (
    ActionStage,
    ContactMemory,
    ControlsState,
    EpisodeBuffers,
    EpisodeClock,
    HistoryState,
    ObservationCache,
    SensorsState,
    SimulatorPhysicalState,
    SimulatorRuntimeState,
)
from protomotions.envs.base_env.markers import Markers
from protomotions.envs.base_env.publishers import (
    ContactPublisher,
    HistorySampler,
    ScenePublisher,
    StatePublisher,
    StaticPublisher,
    TerrainPublisher,
)
from protomotions.envs.base_env.reset import (
    RecoveryBank,
    PlacedReference,
    ReferenceMotion,
    ResetPerturbation,
    ResetPose,
    SpawnPlacer,
)
from protomotions.envs.body_selection import (
    resolve_body_selections,
    resolve_rule_selections,
)
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import (
    binding_errors,
    control_namespace_errors,
    control_producer,
    published_vocabulary_errors,
    requirement_errors,
)
from protomotions.envs.context_requirements import Facts, bound_components
from protomotions.envs.control.base import bound_paths
from protomotions.envs.control.manager import ControlManager, TeleportQueue
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer, history_depth
from protomotions.envs.randomness import Randomness
from protomotions.envs.sensors.runtime import SensorRuntime
from protomotions.envs.terminations.rule import TerminationRule, components_of

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv

log = logging.getLogger(__name__)

# The phases of ``build_env``, in order (the module docstring says why).
BUILD_PHASES = (
    "randomness",
    "motion",
    "robot_model",
    "owners",
    "sensors",
    "controls",
    "markers",
    "managers",
    "history",
    "lists",
    "check",
    "recovery",
    "observe",
)


def build_env(env: "BaseEnv", motion_lib) -> None:
    """Create the objects ``env`` holds and declare its owner lists, in order.

    ``env`` has its constructor arguments stored (``config``, ``robot_config``,
    ``device``, ``terrain``, ``scene_lib``, ``simulator``, ``num_envs``, ``dt``)
    and its three flags initialised by the constructor
    (``_perturbations_enabled`` is True, ``_current_context`` is None,
    ``_advancing`` is False), and nothing else; ``motion_lib`` is the prebuilt
    motion library to install.
    """
    config, robot_config, device = env.config, env.robot_config, env.device
    simulator, terrain, scene_lib = env.simulator, env.terrain, env.scene_lib
    num_envs = env.num_envs

    # randomness: the per-stage generators every draw site below is handed
    # (all None, the global stream, until C6 makes the seeds real); seeded
    # from the process seed training set.
    env.randomness = Randomness(torch.initial_seed(), device)

    # motion: the library, its manager and the clock. First, because the
    # control components built below read the manager.
    env._motion = ReferenceMotion(
        config.motion_manager,
        robot_config,
        scene_lib,
        num_envs,
        env.dt,
        device,
        config.ref_contact_smooth_window,
        env.randomness.generator("motion_sampling"),
    )
    env._motion.install(motion_lib)

    # robot_model: the kinematic tensors move to the device once, IN PLACE.
    # IsaacGym's reset path runs forward kinematics on this same object, so
    # dropping this passes every CPU test and fails on IsaacGym at the first
    # reset (not verified on a GPU here).
    kinematic_info = robot_config.kinematic_info
    kinematic_info.to(device)

    # owners: the per-env buffers and the reset chain's helpers. None holds
    # the env; the spawn placer, the recovery bank and the reset pose hold
    # the simulator. The contact memory keeps only the selected rows'
    # previous strength for the impact cost (the simulator's contact layout,
    # known before it allocates).
    env._episode = EpisodeBuffers(num_envs, device, config.max_episode_length)
    num_contact_rows = (
        len(simulator.contact_layout.body_index) if simulator.contacts_enabled else 0
    )
    env._contact_memory = ContactMemory(num_envs, num_contact_rows, device)
    # Current step only; previous actions come from the state history.
    env._action_stage = ActionStage(
        config.action_config, num_envs, robot_config.number_of_actions, device
    )
    env._spawn = SpawnPlacer(
        terrain,
        scene_lib,
        env._motion,
        simulator,
        config.ref_respawn_offset,
        config.skip_correct_terrain_height_on_flat,
        num_envs,
        device,
    )
    # reference: the one query the tasks read the reference motion through
    # (clips placed where each env spawned, the clip assignment's writers);
    # public, composed over the two owners above and reading them live.
    env.reference = PlacedReference(
        env._motion,
        env._spawn,
        scene_lib,
        config.ref_object_respawn_offset,
        num_envs,
        device,
    )
    env._recovery = RecoveryBank(
        getattr(config, "recovery_reset", None),
        simulator,
        terrain,
        kinematic_info.dof_limits_lower,
        kinematic_info.dof_limits_upper,
        robot_config.number_of_actions,
        num_envs,
        device,
        env.randomness.generator("recovery"),
    )
    env._pose = ResetPose(
        simulator,
        scene_lib,
        env._motion,
        env._spawn,
        config.ref_object_respawn_offset,
        device,
    )
    env._perturbation = ResetPerturbation(
        robot_config,
        kinematic_info.dof_limits_lower,
        kinematic_info.dof_limits_upper,
        env.randomness.generator("reset_noise"),
    )

    # sensors: only with a declared suite; its private generator is seeded
    # from the config.
    env.sensors = None
    if config.sensors is not None:
        env.sensors = SensorRuntime(
            config.sensors,
            robot_config,
            num_envs,
            device,
            ground_query=terrain.get_ground_heights,
        )

    # controls: the viewer's reset key, then the control components from one
    # TaskInputs (everything a control reads: the batch, the robot, the
    # read-only clock, the placed reference, the ground query, the control
    # stage's generator, the UI exceptions, and the animation players' two
    # declared writes). They contribute markers and so must exist before them.
    env._key_bindings = simulator.user_interface.scope("env")
    env._key_bindings.register("R", "reset", "Reset all environments")
    teleports = TeleportQueue()
    inputs = TaskInputs(
        num_envs=num_envs,
        device=device,
        dt=env.dt,
        robot=robot_config,
        clock=EpisodeClock(env._episode, env.dt),
        reference=env.reference,
        ground_height=ground_height_query(terrain),
        generator=env.randomness.generator("control"),
        headless=simulator.headless,
        ui=simulator.user_interface,
        viewer_reset=env._key_bindings.reset,
        restart_episode=env._episode.restart,
        teleport=teleports.request,
    )
    env.control_manager = ControlManager(config.control_components, inputs, teleports)
    env.control_manager.check_holds_no_handles()

    # markers: the simulator allocates its GPU memory from the marker set
    # (the terrain height-map markers and the control components'; the
    # step's physics phase hands ``_markers.state`` to the simulator, which
    # reads the robot state once per frame for every control's markers).
    env._markers = Markers(
        env.control_manager,
        terrain,
        simulator.get_root_state,
        simulator.get_robot_state,
        show_terrain_markers=config.show_terrain_markers,
        headless=simulator.headless,
        num_envs=num_envs,
        device=device,
    )
    simulator._initialize_with_markers(env._markers.configs())

    # managers: one component manager, three MDP managers over it.
    env._component_manager = ComponentManager(device)
    bind_managers(env)

    # history: the sampler turns a state read into the ring's sample (ground
    # under the anchor, contact flags when the simulator reports contacts,
    # the current commands); the first read seeds the ring (and the sensors'
    # first estimate of it); the sensor reset draws from the sensor generator.
    env._history_sampler = HistorySampler(
        terrain, robot_config, simulator, env._action_stage
    )
    initial_state = simulator.get_robot_state()
    all_env_ids = torch.arange(num_envs, device=device)
    estimates = env.sensors.reset(all_env_ids, initial_state) if env.sensors else None
    # The ring keeps as many past slots as the bound observation, reward and
    # termination components read (zero when nothing reads the past); the
    # config-only check computes the same depth from the configs.
    env._state_history = StateHistoryBuffer(
        history_depth(bound_components(config, robot_config)),
        env._history_sampler.sample(initial_state, estimates),
    )

    # lists: everything the lists refer to exists now; the fall-pose
    # generation below saves and restores through the snapshot walk declared
    # here.
    declare_owners(env)

    # check: published and declared names agree, every binding is published;
    # then the wiring report.
    validate_bindings(env)
    log.info("%s", wiring_report(env))

    # recovery: the fall-pose cache, at the same point relative to every other
    # random draw as always.
    if env._recovery.enabled:
        env._recovery.generate(
            env._pose.default_reset_state,
            env._pose.default_object_state,
            save_state=env.save_state,
            restore_state=env.restore_state,
        )

    # observe: every observation is a kernel; from the cached all-env context
    # (the compiled path, as the step's), which stays cached.
    env._observations.compute_components(env.context, env._all_env_ids())


def bind_managers(env: "BaseEnv") -> None:
    """Build the three MDP managers over the one shared component manager.

    One ``ComponentManager`` for all three groups: its compiled-kernel cache
    is keyed by component name, and splitting it would change behaviour for
    a config that names a reward and an observation alike. The observation
    manager owns the dynamic observation buffer; the ``ObservationCache``
    snapshot owner (``declare_owners``) holds that same dict.
    """
    config = env.config
    # Body selections are names in the saved configuration; each table is
    # resolved to index tensors here, once, against this robot. The configs
    # are not mutated.
    robot_config = env.robot_config
    env._observations = ObservationManager(
        env._component_manager,
        resolve_body_selections(config.observation_components, robot_config),
        env.num_envs,
        env.device,
    )
    env._rewards = RewardManager(
        env._component_manager,
        resolve_body_selections(config.reward_components, robot_config),
        env.num_envs,
        env.device,
        config.reset_grace_period,
    )
    env._terminations = TerminationManager(
        env._component_manager,
        resolve_rule_selections(termination_table(env), robot_config),
        env.num_envs,
        env.device,
    )


def termination_table(env: "BaseEnv") -> Dict[str, TerminationRule]:
    """The recipe's termination entries and the control components' checks
    (``ControlManager.terminations``, keyed ``<control>/<check>``) as one
    table of ``TerminationRule``, refusing a name on both sides."""
    recipe = dict(env.config.termination_components or {})
    controls = env.control_manager.terminations()
    shared = sorted(set(recipe) & set(controls))
    if shared:
        raise ValueError(
            f"termination names used by both the recipe and a control: {shared}; "
            "rename the recipe's entry (a control's checks are keyed <control>/<check>)"
        )
    return {**recipe, **controls}


def ground_height_query(terrain) -> Callable[[torch.Tensor], torch.Tensor]:
    """``TaskInputs.ground_height``: the terrain height under points ``[n, 2]``
    or ``[n, 3]`` as ``[n]`` (the terrain's query returns a column)."""

    def ground_height(points: torch.Tensor) -> torch.Tensor:
        return terrain.get_ground_heights(points).squeeze(-1)

    return ground_height


def declare_owners(env: "BaseEnv") -> None:
    """Declare the four owner lists: reset rows, snapshot, publish, advance.

    The module docstring says why each order is what it is; the comment on
    each entry here says what it is. The snapshot list IS the restore order;
    the publish list is keyed by each owner's ``name`` (unique, asserted
    here) and carries the contact matrix and the contact memory only when
    the simulator reports contacts (the control components publish through
    the control manager, not this list); the advance list is static.
    ``controls`` is the same owner on the snapshot and the advance list.
    """
    simulator = env.simulator
    # The order reset() clears rows in.
    env._row_owners = (
        env._episode,  # progress / reset / terminate
        env._contact_memory,  # last step's contact strengths
        env._action_stage,  # the raw and processed action buffers
    )
    sensors = (SensorsState(env.sensors),) if env.sensors is not None else ()
    env._controls = ControlsState(env.control_manager, simulator)
    # save_state() collects, and restore_state() writes back, in this order.
    env._snapshot_owners = (
        # flush contacts, write poses back, THEN backend state
        SimulatorPhysicalState(simulator, env.scene_lib, env.num_envs, env.device),
        HistoryState(env._state_history),  # the history ring, copied in place
        *sensors,  # only with a sensor suite: generator + episode start
        env._episode,  # progress / reset / terminate
        env._spawn,  # respawn_root_offset
        env._motion,  # the clip assignment; nothing in the walk reads it
        env._action_stage,  # raw and processed action of the current step
        env._contact_memory,  # last step's contact strengths
        # cached dynamic observations (dict refilled in place)
        ObservationCache(env._observations.buffer),
        SimulatorRuntimeState(simulator),  # action history, latency, reset age ...
        env._controls,  # control components' task tensors
        env.randomness,  # the per-stage generators (nothing until C6)
    )
    contacts = simulator.contacts_enabled
    # What _publish merges into the context, keyed by each owner's name.
    env._publish_owners = (
        # current/past states of every source
        StatePublisher(env._state_history, env.robot_config.anchor_body_index),
        TerrainPublisher(env.terrain),  # height points and samples
        # object surfaces (empty without objects)
        ScenePublisher(env.scene_lib, simulator, env.num_envs, env.device),
        # the contact matrix and its axis maps
        *((ContactPublisher(simulator),) if contacts else ()),
        env._episode,  # progress_buf
        # prev_contact_force_magnitudes
        *((env._contact_memory,) if contacts else ()),
        # the two contact body-id tensors
        StaticPublisher(env.contact_body_ids, env.non_termination_contact_body_ids),
    )
    names = [owner.name for owner in env._publish_owners] + [
        control_producer(name) for name in env.control_manager.components
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"publish owners share a name: {duplicates}"
    # The order step() moves state on in, after the history push.
    env._advance_owners = (
        env._motion,  # ReferenceMotion.advance: the current manager's clock
        env._controls,  # ControlsState.advance: the controls, then the teleports
        env._spawn,  # SpawnPlacer.advance: the per-step re-alignment, if asked
    )


def validate_bindings(env: "BaseEnv") -> None:
    """The build-time check that the published and the declared names agree,
    and that every binding reads a published name. Reads the cached all-env
    context (``env.context`` builds it if it was dropped) and runs before any
    kernel executes: once here, at the ``check`` phase of ``build_env``, and
    again after a live motion-library switch
    (``BaseEnv.install_motion_lib``).

    1. Each control component published exactly the namespaces its class
       declares (``ControlComponent.namespaces``, which the config-only check
       trusts), read off its own producer name in the context. This runs
       first so that a namespace a component publishes without declaring it
       is reported once, naming the component, instead of name by name by
       the two checks below.
    2. What the publishers published matches ``context_views.py``: in every
       published namespace, nothing undeclared is published and every
       declared name without a ``requires=`` tag is present; and no tagged
       name is published by a configuration that does not satisfy its tag
       (``context.requirement_errors``, with the facts read off the built
       env). Both name the producer of an offending name.
    3. Every bound path of every observation, reward and termination
       component (the control components' checks included) is a declared
       name and is present; a bound name this configuration does not publish
       says which fact is missing (``context_requirements.status``).
    4. The component manager's own availability and required-key checks.
    5. Each configured control component's ``check_bindings`` over the union
       of the three tables' bound paths: a component whose published layout
       depends on its config (the masked mimic goal canvas) refuses a binding
       it cannot serve, with the reason.
    """
    context = env.context
    facts = Facts.from_env(env)
    components = env.control_manager.components
    errors = control_namespace_errors(
        context,
        {name: component.namespaces for name, component in components.items()},
    )
    if errors:
        raise ValueError(
            "the control components' published namespaces do not match "
            "their declarations:\n  " + "\n  ".join(errors)
        )
    errors = published_vocabulary_errors(context) + requirement_errors(context, facts)
    if errors:
        raise ValueError(
            "the published context does not match the declared vocabulary "
            "(context_views.py):\n  " + "\n  ".join(errors)
        )
    # The managers' tables, not the config's: ``bind_managers`` resolved the
    # body selections into these copies (the config keeps its names), and the
    # component manager refuses an unresolved selection.
    tables = {
        "observation": env._observations.components,
        "reward": env._rewards.components,
        "termination": components_of(env._terminations.components),
    }
    for group, bound in tables.items():
        errors = binding_errors(bound, context, group=group, facts=facts)
        if errors:
            raise ValueError("\n".join(errors))
        env._component_manager.validate_bindings(bound, context, group=group)
    bound = bound_paths(*tables.values())
    for component in components.values():
        type(component).check_bindings(component.config, env.robot_config, bound)


def wiring_report(env: "BaseEnv") -> str:
    """One block naming the four declared lists, for the log and the tests:
    the advance owners in step order, the publish owners and then each
    control component (``control/<name>``) with what it publishes (count and
    first names), the row owners in reset order and the snapshot owners in
    restore order.

    Reads the cached all-env context and groups its keys by the producer
    the merge recorded (``FrozenContext.producers``); it publishes nothing
    itself, so a publisher with a side effect is not run again.
    """
    lines = [
        "BaseEnv wiring",
        "  advance owners (step order): "
        + ", ".join(owner.name for owner in env._advance_owners),
        "  publish owners (context producers):",
    ]
    by_producer: Dict[str, list] = {}
    for key, producer in env.context.producers.items():
        by_producer.setdefault(producer, []).append(key)
    producers = [owner.name for owner in env._publish_owners] + [
        control_producer(name) for name in env.control_manager.components
    ]
    for name in producers:
        keys = by_producer.get(name, [])
        shown = ", ".join(keys[:4]) + (", ..." if len(keys) > 4 else "")
        detail = f" ({shown})" if keys else ""
        lines.append(f"    {name}: {len(keys)} names{detail}")
    lines.append(
        "  row owners (reset clears): "
        + ", ".join(owner.name for owner in env._row_owners)
    )
    lines.append(
        "  snapshot owners (restore order): "
        + ", ".join(owner.name for owner in env._snapshot_owners)
    )
    return "\n".join(lines)


__all__ = [
    "BUILD_PHASES",
    "bind_managers",
    "build_env",
    "declare_owners",
    "ground_height_query",
    "termination_table",
    "validate_bindings",
    "wiring_report",
]
