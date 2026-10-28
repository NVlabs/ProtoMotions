# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliberate refactor mistakes, patched into memory only.

Each mutation rewrites one method's source text and rebinds it on its class,
replaces one method of a state owner (``base_env/owners.py`` or the
``base_env/reset`` package) with a deliberately wrong version, or wraps one
method so that something runs around it (the phase-order mutation reorders
the env's declared advance list after ``BaseEnv.__init__``; the fall-cache
mutation drops the restore from ``RecoveryBank.generate``), so the real env
built afterwards runs the mistake. They exist to prove the harness notices: a
recording with a mutation applied must differ from a clean one or, for a
mistake the build check rejects, the mutated env must fail to build.
``_patch`` asserts every pattern it edits is present, so a mutation that no
longer matches today's source fails loudly instead of silently testing
nothing; the owner replacements and the wrappers track structure rather than
wording, so moving a statement inside an owner does not break them.

The owner modules are imported only when a mutation is applied: the parity
script runs this catalogue (unapplied) against base trees that predate them.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import textwrap
import types
from contextlib import contextmanager
from typing import Callable, Dict, List, Tuple, Union


class Around:
    """A wrapper mutation: ``around(original, self, *args, **kwargs)`` runs in
    place of ``self.method(*args, **kwargs)`` and decides when (and whether)
    to call the original."""

    def __init__(self, around: Callable):
        self.around = around


# (old, new): ``old`` is the source text to replace, or a tuple of alternative
# spellings of it (a base tree and this tree may spell one line differently;
# the first alternative found is replaced).
Edit = Tuple[Union[str, Tuple[str, ...]], str]
SourceTarget = Tuple[str, str, str, List[Edit]]  # module, class, method, edits
OwnerTarget = Tuple[str, str, str, Callable]  # module, class, method, replacement
AroundTarget = Tuple[str, str, str, Around]  # module, class, method, wrapper
Target = Union[SourceTarget, OwnerTarget, AroundTarget]

_ENV = ("protomotions.envs.base_env.env", "BaseEnv")
_EVAL = ("protomotions.agents.evaluators.base_evaluator", "BaseEvaluator")
_STATE_PUBLISHER = ("protomotions.envs.base_env.publishers", "StatePublisher")
_OWNERS = "protomotions.envs.base_env.owners"
_SPAWN = "protomotions.envs.base_env.reset.spawn"
_RECOVERY_BANK = "protomotions.envs.base_env.reset.recovery"

# Two lines of BaseEnv.reset: the recovery phase and the simulator write.
_WRITE = "self.simulator.reset_envs(new_states, new_object_states, env_ids)\n"
_RECOVERY = (
    "new_states, _ = self._recovery.swap_in(\n"
    "        env_ids, new_states, self._spawn.respawn_root_offset\n"
    "    )\n"
)
# The one subset-context observation call: BaseEnv.observe's last line.
_SUBSET_OBS = "self._observations.compute(self._publish(env_ids), env_ids)"


def _env(method: str, edits: List[Edit]) -> SourceTarget:
    return (*_ENV, method, edits)


def _env_around(method: str, around: Callable) -> AroundTarget:
    return (*_ENV, method, Around(around))


def _evaluator(method: str, edits: List[Edit]) -> SourceTarget:
    return (*_EVAL, method, edits)


def _state_publisher(method: str, edits: List[Edit]) -> SourceTarget:
    return (*_STATE_PUBLISHER, method, edits)


def _owner(
    class_name: str, method: str, replacement: Callable, module: str = _OWNERS
) -> OwnerTarget:
    return (module, class_name, method, replacement)


def _around(module: str, class_name: str, method: str, around: Callable):
    return (module, class_name, method, Around(around))


def _skipped(self, *args, **kwargs) -> None:
    """Replacement for an owner method the mutation leaves out entirely."""


def _progress_reset_all_rows(self, env_ids) -> None:
    """``EpisodeBuffers.reset_rows`` that clears EVERY env's progress, not only
    the rows being reset (the done flags are still cleared per row)."""
    self.progress_buf[:] = 0
    self.reset_buf[env_ids] = False
    self.terminate_buf[env_ids] = False


def _controls_before_motion_clock(init, self, *args, **kwargs) -> None:
    """``BaseEnv.__init__`` followed by a reordering of the declared advance
    list: the control components advance before the motion clock (the other
    entries keep their relative order)."""
    init(self, *args, **kwargs)
    owners = list(self._advance_owners)
    owners.sort(key=lambda owner: owner.name != "controls")
    self._advance_owners = tuple(owners)


def _objects_only_move_to_spawn(self, env_ids, new_states, new_object_states):
    """``SpawnPlacer.move_to_spawn`` that moves the scene objects to the spawn
    but forgets the robot."""
    if self.scene_lib.num_scenes() > 0:
        new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)
    return new_states, new_object_states


def _no_restore(snapshot) -> None:
    """A ``restore_state`` that puts nothing back."""


def _generate_without_restore(generate, self, *args, **kwargs):
    """``RecoveryBank.generate`` whose restore of the pre-generation snapshot
    is skipped (``restore_state`` is a keyword argument of the call)."""
    kwargs["restore_state"] = _no_restore
    return generate(self, *args, **kwargs)


MUTATIONS: Dict[str, List[Target]] = {
    # --- reset chain ---------------------------------------------------------
    "actions_not_zeroed_on_reset": [_owner("ActionStage", "reset_rows", _skipped)],
    "contact_forces_not_cleared_on_reset": [
        _owner("ContactMemory", "reset_rows", _skipped)
    ],
    "progress_reset_all_rows": [
        _owner("EpisodeBuffers", "reset_rows", _progress_reset_all_rows)
    ],
    "reset_noise_before_recovery": [
        _env(
            "reset",
            [(_RECOVERY, "pass\n"), (_WRITE, _RECOVERY + "    " + _WRITE)],
        )
    ],
    # Neither the reset tail nor observe drops the cached all-env context, so
    # env.context keeps describing the state before the reset.
    "stale_context_after_reset": [
        _env("reset", [("self._current_context = None\n", "pass\n")]),
        _env("observe", [("self._current_context = None\n", "pass\n")]),
    ],
    "reset_recomputes_all_obs": [
        _env(
            "observe",
            [
                (
                    _SUBSET_OBS,
                    "self._observations.compute(self._publish(), self._all_env_ids())",
                )
            ],
        )
    ],
    # History is seeded from the measurement BEFORE the simulator write, so
    # the first post-reset observations describe the previous episode's pose.
    "history_seeded_before_sim_write": [
        _env(
            "reset",
            [
                (_WRITE, "pass\n"),
                (
                    "self.control_manager.reset_rows(env_ids, state)\n",
                    _WRITE + "    self.control_manager.reset_rows(env_ids, state)\n",
                ),
            ],
        )
    ],
    "respawn_offset_not_applied": [
        _owner("SpawnPlacer", "move_to_spawn", _objects_only_move_to_spawn, _SPAWN)
    ],
    # The fall-pose cache is generated by stepping the real simulator at
    # construction; forgetting to restore the snapshot leaves the simulator in
    # the fallen poses with its solver history and action buffers advanced.
    "fall_cache_leaks_into_training": [
        _around(_RECOVERY_BANK, "RecoveryBank", "generate", _generate_without_restore)
    ],
    # Same values, different path: an all-env reset builds the global context,
    # so the post-reset observations run on the compiled path. Only the call
    # log (``execute_all[all]`` instead of ``execute_all[subset:n]``) sees it.
    "full_reset_uses_global_ctx": [
        _env(
            "observe",
            [
                (
                    _SUBSET_OBS,
                    "self._observations.compute(self._publish("
                    "None if len(env_ids) == self.num_envs else env_ids), env_ids)",
                )
            ],
        )
    ],
    # --- context publish -----------------------------------------------------
    # The state publisher forgets a derived name. No rollout can see this: the
    # env's build check rejects it before the first step (a test asserts that,
    # not a parity comparison).
    "publish_drops_derived_names": [
        _state_publisher(
            "publish",
            [
                (
                    'published["ground_heights"] = published["current.ground_heights"]\n',
                    'published["ground_heights"] = published["current.ground_heights"]\n'
                    '        del published["current.root_local_ang_vel"]\n',
                )
            ],
        )
    ],
    # --- step phase order ----------------------------------------------------
    # A reordered copy of the declared advance list: controls before the clock.
    "control_before_motion_clock": [
        _env_around("__init__", _controls_before_motion_clock)
    ],
    # The clock phase moves from before the checks to after them.
    "progress_tick_after_checks": [
        _env(
            "step",
            [
                ("self._episode.tick()\n", "pass\n"),
                (
                    'self.extras["terminate"] = episode.terminate_buf\n',
                    'self.extras["terminate"] = episode.terminate_buf\n'
                    "    self._episode.tick()\n",
                ),
            ],
        )
    ],
    "raw_extras_dropped": [
        _env("_raw_state_extras", [('extras[f"raw/{k}"] = value\n', "pass\n")])
    ],
    # The viewer's R key is consumed but never acted on (the request mask
    # here, the progress sentinel on a base tree that predates it).
    "user_reset_ignored": [
        _env(
            "step",
            [
                (
                    (
                        "episode.request_reset(self._all_env_ids())\n",
                        "episode.force_timeout()\n",
                    ),
                    "pass\n",
                )
            ],
        )
    ],
    # --- evaluator write-back ------------------------------------------------
    "restore_skips_obs_cache": [_owner("ObservationCache", "load_state", _skipped)],
    "restore_skips_history": [_owner("HistoryState", "load_state", _skipped)],
    "restore_skips_sensor_state": [_owner("SensorsState", "load_state", _skipped)],
    # The evaluator drives the shared env with zero actions instead of the
    # policy's: only the recorded ``eval/step`` records can see it, because the
    # write-back restores everything training reads afterwards.
    "evaluator_actions_zeroed": [
        _evaluator(
            "_policy_action",
            [
                (
                    "model_outs = self.agent.inference_step(obs_td)\n",
                    "model_outs = {\n"
                    "            key: torch.zeros_like(value)\n"
                    "            for key, value in self.agent.inference_step(obs_td).items()\n"
                    "        }\n",
                )
            ],
        )
    ],
    # Re-enable pushes and reset noise BEFORE the env is written back. The
    # restore's reset_envs then schedules pushes, which draws from the global
    # torch generator and shifts every random number training sees afterwards.
    "perturbations_reenabled_before_restore": [
        _evaluator(
            "evaluate",
            [
                ("self.cleanup_after_evaluation()\n", "self.__swap__()\n"),
                (
                    "self._restore_perturbations()\n",
                    "self.cleanup_after_evaluation()\n",
                ),
                ("self.__swap__()\n", "self._restore_perturbations()\n"),
            ],
        )
    ],
}


def _patch(module_name: str, class_name: str, method: str, edits: List[Edit]):
    module = importlib.import_module(module_name)
    owner = getattr(module, class_name)
    original = owner.__dict__[method]
    source = textwrap.dedent(inspect.getsource(original))
    for old, new in edits:
        candidates = old if isinstance(old, tuple) else (old,)
        found = next((text for text in candidates if text in source), None)
        assert found is not None, f"{class_name}.{method}: pattern not found: {old!r}"
        source = source.replace(found, new, 1)
    namespace = dict(vars(module))
    exec(compile(source, f"<mutated {class_name}.{method}>", "exec"), namespace)
    compiled = namespace[method]
    if getattr(compiled, "__closure__", None) is None:
        # Rebind to the module's LIVE globals (not the copy the exec ran in)
        # so a module-level name the method reads can still be monkeypatched
        # after the mutation is applied, the way the spies rebind module
        # attributes. (``apply_reset_noise`` was the case while ``reset``
        # called it from ``env.py``; it is now spied in
        # ``reset/perturbation.py``, which no source mutation rewrites.)
        mutated = types.FunctionType(
            compiled.__code__, module.__dict__, method, compiled.__defaults__, None
        )
        mutated.__kwdefaults__ = compiled.__kwdefaults__
    else:
        # A decorated method (e.g. ``@torch.no_grad()``) is a closure; keep the
        # decorated object the way the module built it.
        mutated = compiled
    setattr(owner, method, mutated)

    def undo():
        setattr(owner, method, original)

    return undo


def _replace(module_name: str, class_name: str, method: str, replacement: Callable):
    """Rebind ``class_name.method`` to ``replacement`` (an owner mutation)."""
    module = importlib.import_module(module_name)
    owner = getattr(module, class_name)
    original = owner.__dict__[method]
    setattr(owner, method, replacement)

    def undo():
        setattr(owner, method, original)

    return undo


def _wrap(module_name: str, class_name: str, method: str, around: Callable):
    """Rebind ``class_name.method`` so ``around(original, self, ...)`` runs."""
    module = importlib.import_module(module_name)
    owner = getattr(module, class_name)
    original = owner.__dict__[method]

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        return around(original, self, *args, **kwargs)

    setattr(owner, method, wrapped)

    def undo():
        setattr(owner, method, original)

    return undo


def _apply_target(target: Target) -> Callable[[], None]:
    module_name, class_name, method, change = target
    if isinstance(change, Around):
        return _wrap(module_name, class_name, method, change.around)
    if callable(change):
        return _replace(module_name, class_name, method, change)
    return _patch(module_name, class_name, method, change)


def apply(name: str) -> Callable[[], None]:
    """Apply one mutation; returns a callable that reverts it."""
    undos = [_apply_target(target) for target in MUTATIONS[name]]

    def undo():
        for revert in reversed(undos):
            revert()

    return undo


@contextmanager
def applied(name: str):
    undo = apply(name)
    try:
        yield
    finally:
        undo()
