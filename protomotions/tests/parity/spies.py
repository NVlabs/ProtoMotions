# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass-through instrumentation that must not change behaviour.

Every spy calls the original with the same arguments and returns the same
object; it only appends a label to a :class:`CallLog` (or, for the compile
path, to the :class:`CompilePathSpy`'s own list). Everything installed is
reversible through ``CallLog.uninstall()`` / ``CompilePathSpy.uninstall()``,
because the tests build several envs in one process.

The call spies cover only the calls whose ORDER the refactor plan promises to
keep: the env's own entry points, the simulator writes, the history/sensor/
control collaborators' step/reset/save/load, motion sampling, terrain spawn
sampling and reset noise. Pure read helpers (``get_robot_state``, ``capture``,
``_publish``, the owners' ``publish`` ...) are deliberately not spied: the
refactor is free to call them differently as long as the values agree.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import random
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

import protomotions.envs.base_env.env as env_module
from protomotions.envs.randomness import STAGES
from protomotions.envs.component_manager import ComponentManager


def _count_suffix(signature, count_arg: str, args, kwargs) -> str:
    """``[n=<len>]`` of the named argument, ``[n=all]`` when it is absent/None."""
    value = kwargs.get(count_arg)
    if value is None and signature is not None:
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None:
            value = bound.arguments.get(count_arg)
    if value is None:
        return "[n=all]"
    try:
        return f"[n={len(value)}]"
    except TypeError:
        return "[n=?]"


class CallLog(list):
    """Ordered labels of instrumented calls, plus the undo list for the spies.

    ``take()`` drains the entries recorded since the previous record. ``paused()``
    suspends logging while the recorder reads state through spied methods, so
    the harness's own reads do not show up as env activity.
    """

    def __init__(self):
        super().__init__()
        self._paused = 0
        self._undos: List[Callable[[], None]] = []

    @contextmanager
    def paused(self):
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1

    @property
    def is_paused(self) -> bool:
        return self._paused > 0

    def note(self, label: str) -> None:
        if not self._paused:
            self.append(label)

    def take(self) -> List[str]:
        entries = list(self)
        self.clear()
        return entries

    def wrap(
        self, obj, name: str, label: str, *, count_arg: Optional[str] = None
    ) -> bool:
        """Replace ``obj.name`` with a logging pass-through.

        Returns False (and installs nothing) when the attribute does not exist,
        so ``hasattr`` / ``getattr(..., None)`` checks in the env keep their
        meaning. Instance attributes shadow the class method and are deleted
        again on uninstall; module attributes are restored in place.
        """
        if not hasattr(obj, name):
            return False
        original = getattr(obj, name)
        in_own_dict = name in getattr(obj, "__dict__", {})
        try:
            signature = inspect.signature(original)
        except (TypeError, ValueError):
            signature = None

        @functools.wraps(original)
        def spy(*args, **kwargs):
            suffix = ""
            if count_arg is not None:
                suffix = _count_suffix(signature, count_arg, args, kwargs)
            self.note(label + suffix)
            return original(*args, **kwargs)

        setattr(obj, name, spy)

        def undo():
            if in_own_dict:
                setattr(obj, name, original)
            else:
                delattr(obj, name)

        self._undos.append(undo)
        return True

    def uninstall(self) -> None:
        while self._undos:
            self._undos.pop()()


_ENV_METHODS = {
    "step": None,
    "reset": "env_ids",
    "save_state": None,
    "restore_state": None,
}
_SIM_METHODS = {
    "step": None,
    "reset_envs": "env_ids",
    "save_state": None,
    "restore_state": None,
    "save_backend_state": None,
    "restore_backend_state": None,
    "flush_contact_cache": None,
}
_SENSOR_METHODS = {
    "step": None,
    "reset": "env_ids",
    "save_state": None,
    "load_state": None,
}
_HISTORY_METHODS = {
    "push": None,
    "reset": "env_ids",
    "save_state": None,
    "load_state": None,
}
# The control manager's calls carry one label per lifecycle event, shared by
# the two spellings of the control contract (``advance`` / ``reset_rows`` /
# ``load_state`` here, ``step`` / ``reset`` / ``restore_state`` on a base tree
# that predates it), so both sides log the same call list.
_CONTROL_METHODS = {
    "advance": ("step", None),
    "step": ("step", None),
    "reset_rows": ("reset", "env_ids"),
    "reset": ("reset", "env_ids"),
    "save_state": ("save_state", None),
    "load_state": ("restore_state", None),
    "restore_state": ("restore_state", None),
}
_MOTION_METHODS = {"sample_motions": "env_ids", "post_physics_step": None}


def reset_noise_module():
    """The module whose ``apply_reset_noise`` the reset chain calls.

    The reset perturbation helper (``base_env/reset/perturbation.py``) where
    the tree has one; ``env.py`` on a base tree that predates the ``reset``
    package. Both sides of a comparison log the same ``apply_reset_noise``
    label, so moving the call is not a call-order change.
    """
    try:
        return importlib.import_module("protomotions.envs.base_env.reset.perturbation")
    except ImportError:
        return env_module


def reference_motion(env):
    """The reference-motion owner: ``env.reference_motion`` on this tree. On a
    tree from before the forwarders went the env itself still carries
    ``motion_lib`` and ``motion_manager``, and the harness records the merge
    base with this code too (the same shape as the ``_state_history`` /
    ``state_history`` read below)."""
    return getattr(env, "reference_motion", env)


def install_call_spies(env, log: CallLog) -> CallLog:
    """Wrap the env, its collaborators and module-level ``apply_reset_noise``."""
    targets = [
        (env, "env", _ENV_METHODS),
        (env.simulator, "sim", _SIM_METHODS),
        (getattr(env, "sensors", None), "sensors", _SENSOR_METHODS),
        # ``_state_history`` where the tree has made it private, the public
        # name on a base tree that predates that.
        (
            getattr(env, "_state_history", getattr(env, "state_history", None)),
            "history",
            _HISTORY_METHODS,
        ),
        (getattr(env, "control_manager", None), "control", _CONTROL_METHODS),
        (
            getattr(reference_motion(env), "motion_manager", None),
            "motion",
            _MOTION_METHODS,
        ),
        (getattr(env, "terrain", None), "terrain", {"sample_valid_locations": None}),
    ]
    for obj, prefix, methods in targets:
        if obj is None:
            continue
        for name, spec in methods.items():
            label, count_arg = spec if isinstance(spec, tuple) else (name, spec)
            log.wrap(obj, name, f"{prefix}.{label}", count_arg=count_arg)
    log.wrap(reset_noise_module(), "apply_reset_noise", "apply_reset_noise")
    return log


class CompilePathSpy:
    """Record whether each ``ComponentManager.execute_all`` took the compiled path.

    ``execute_all`` compiles only when ``ctx.env_ids`` is None (the stable
    all-env cache). An all-env reset that builds a global context instead of a
    subset context produces identical values but moves post-reset observations
    onto that path; this spy is what makes the harness see it.

    The markers (``all`` / ``subset:n``) go to this object's own ``entries``
    list, drained by ``take()``, so a record's ``compile_path`` is separate
    from its ``calls``. Logging pauses together with the given ``CallLog``.
    """

    def __init__(self, log: CallLog):
        self.log = log
        self.entries: List[str] = []
        self._original = None

    def take(self) -> List[str]:
        entries = list(self.entries)
        self.entries.clear()
        return entries

    def install(self) -> "CompilePathSpy":
        if self._original is not None:
            return self
        original = ComponentManager.__dict__["execute_all"]
        spy = self

        @functools.wraps(original)
        def execute_all(manager, components, ctx):
            if not spy.log.is_paused:
                env_ids = getattr(ctx, "env_ids", None)
                spy.entries.append(
                    "all" if env_ids is None else f"subset:{len(env_ids)}"
                )
            return original(manager, components, ctx)

        ComponentManager.execute_all = execute_all
        self._original = original
        return self

    def uninstall(self) -> None:
        if self._original is not None:
            ComponentManager.execute_all = self._original
            self._original = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()


def rng_states(env) -> Dict[str, Tensor]:
    """Every random generator the rollout can touch, as tensors.

    The process-global torch/numpy/python generators, CUDA when the env is
    on it, the env's per-stage generators (``rng/randomness/<stage>``, the
    ``Randomness`` owner), the sensor runtime's private generator and the
    terrain's private spawn-sampling generator
    (``Terrain._spawn_sample_generator``). Those two are the only private
    generators in production (``sensors/runtime.py`` and
    ``terrains/terrain.py``); the fake terrain has none.
    """
    states = {"torch": torch.get_rng_state().clone()}
    device = getattr(env, "device", None)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state(device).clone()
    numpy_state = np.random.get_state()
    states["numpy"] = torch.as_tensor(np.asarray(numpy_state[1]).astype(np.int64))
    states["numpy_pos"] = torch.tensor([int(numpy_state[2])], dtype=torch.int64)
    version, internal, gauss_next = random.getstate()
    states["python"] = torch.tensor(
        [version, *internal, 0 if gauss_next is None else 1], dtype=torch.int64
    )
    randomness = getattr(env, "randomness", None)
    if randomness is not None:
        for stage in STAGES:
            # A base export built before the seeds were real answers None for
            # a stage (its draws were on the global stream); the key is absent.
            generator = randomness.generator(stage)
            if generator is not None:
                states[f"randomness/{stage}"] = generator.get_state().clone()
    sensors = getattr(env, "sensors", None)
    if sensors is not None:
        states["sensors"] = sensors.generator.get_state().clone()
    spawn = getattr(getattr(env, "terrain", None), "_spawn_sample_generator", None)
    if isinstance(spawn, torch.Generator):
        states["terrain_spawn"] = spawn.get_state().clone()
    return states
