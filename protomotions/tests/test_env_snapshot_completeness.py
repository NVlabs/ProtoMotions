# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every per-env tensor the env holds must come back after ``restore_state``.

The evaluator saves the env before its rollout and restores it after; training
then continues the interrupted episodes. A per-env tensor added without a
snapshot entry keeps the evaluation's values through that restore, and nothing
else notices. This test walks everything reachable from ``vars(env)`` -- the
simulator, the sensors, the history ring, the control manager and its
components, the observation callbacks, the state owners, and whatever a future
change adds -- recursing through attributes, dicts, dataclasses, lists and
tuples, and finds every tensor whose first dimension is the env count. It
perturbs all of them in place, restores the snapshot and requires each one to
be back to its original, except the paths on an allowlist that names its
reason (those are constants and are left alone). An allowlist entry matches
by attribute-chain suffix, because the walk labels a shared object with the
first path that reaches it and the reset helpers hold back-references to the
env's own collaborators (``env._spawn._motion`` is ``env._motion``): the
entry holds whichever of those paths the walk took first. An allowlist entry
that matches nothing fails too, so the list cannot go stale, and a container
the depth limit stopped at while it still held such a tensor fails instead
of being skipped quietly.

The same walk checks the owner list: every object it reaches that is an owner
(it has ``save_state``, ``load_state`` and a ``name``) must be an element of
``env._snapshot_owners``, so an owner built but left off the list fails here.

Fast: the 3-env fake scenario with sensors, a mimic control component, a
motion library and the recovery-reset cache (``fake3_mimic``). Slow, behind
the ``parity_slow`` marker: the MuJoCo ``smpl_mimic`` scenario.
"""

import dataclasses
import os
import types

import pytest
import torch

pytest.importorskip("mujoco")

from protomotions.envs.mdp_component import MdpComponent  # noqa: E402
from protomotions.tests.parity import (  # noqa: E402
    SCENARIOS,
    build_env,
    eager_mdp_kernels,
)

# Attribute names the walk does not enter, on any object, and why.
SKIPPED_ATTRIBUTES = {
    "config": "configuration, not state (the env's, the simulator's, a component's)",
    "robot_config": "configuration, not state",
    "action_config": (
        "the action function's static parameters, bound once by ActionStage"
    ),
    "_current_context": (
        "a per-step derived cache of views over state walked elsewhere: "
        "restore_state drops it (None) and the next step rebuilds it"
    ),
}
# Attribute / dict / dataclass / sequence nesting below ``vars(env)``. Deep
# enough for the history (ring -> sample -> states dict -> body state ->
# tensor) and a control component's nested caches. A container the limit
# stops at while it still holds a per-env tensor fails the test.
MAX_DEPTH = 6
# Objects with a ``__dict__`` the walk must not treat as containers. An
# MdpComponent is configuration wherever it appears (a kernel bound to context
# paths plus its static parameters); the MDP managers hold the env config's
# component dicts, which the walk reaches through them.
_NOT_CONTAINERS = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    MdpComponent,
)

# Paths (or prefixes: a match continues with "." or "[") that need not come
# back, and why; each is matched as a suffix of the walk's path (see
# ``_allowed``). Everything else that fails to come back is a missing snapshot
# entry.
COMMON_ALLOWLIST = {
    "env.extras": (
        "step outputs: step() replaces the dict and every entry is rebuilt by "
        "the next step before anything reads it"
    ),
    "env._episode.rew_buf": (
        "a step output: the step's score phase overwrites it before anything "
        "reads it, so it was never part of the snapshot"
    ),
    "env._snapshot_owners[0]._all_env_ids": (
        "SimulatorPhysicalState's constant index vector (arange over the envs) "
        "for the full write-back"
    ),
    "env._motion.motion_manager.init_start_probs": (
        "the motion manager's constant: every env's probability of starting a "
        "clip at t=0, written once at construction"
    ),
    "env._motion.motion_manager._env_has_fixed_motion": (
        "the motion manager's constant: which envs are pinned to their scene's "
        "clip, written once at construction"
    ),
}
FAKE_ALLOWLIST = {
    **COMMON_ALLOWLIST,
    "env._recovery.fall_states": (
        "the recovery bank's fall-pose cache: generated once at construction "
        "and only ever read by the recovery reset"
    ),
    "env._pose.default_reset_state": (
        "a cached_property of the reset pose (so it lands in its vars once "
        "computed): the simulator's default reset pose, a constant"
    ),
    "env.simulator.default": (
        "the fake simulator's default reset pose: a constant, never written "
        "after construction"
    ),
    "env.simulator.state.dof_forces": (
        "the kinematic fake never writes joint forces (constant zeros); its "
        "reset refreshes only the kinematic fields"
    ),
    "env.simulator.state.rigid_body_contacts": (
        "the kinematic fake never writes contact flags (constant zeros); the "
        "env derives contacts from get_contact_forces instead"
    ),
    "env.simulator.reset_calls": (
        "the fake's bookkeeping: the env-id vector of every reset_envs call "
        "(a full reset's has the env count as its length); instrumentation, "
        "not state"
    ),
    "env.sensors.env_ids": "a constant index vector (arange over the envs), not state",
}
MUJOCO_ALLOWLIST = {
    **COMMON_ALLOWLIST,
    "env.control_manager.components['mimic'].future_time_offsets": (
        "a registered constant of the mimic control (dt times the configured "
        "future-step indices, one entry per future step) whose length matches "
        "the env count only because the scenario has one env and one future "
        "step; the registry neither resets nor snapshots constants"
    ),
    "env.control_manager.components['mimic'].state._constants['future_time_offsets']": (
        "the same constant, reached again through the registry's constants table"
    ),
    "env._motion.motion_lib": (
        "the reference motion library's per-clip tables (lengths, frame counts, "
        "dt, start offsets, sampling weights), whose length matches the env "
        "count only because the scenario has one clip and one env; library "
        "data, not per-env state"
    ),
    "env._motion.motion_manager.motion_weights": (
        "the per-clip sampling weights (one clip, one env here): curriculum "
        "state the evaluator's write-back changes AFTER the restore on purpose, "
        "not episode state"
    ),
    "env._motion.motion_manager._eligible": (
        "the per-clip eligibility mask (one clip, one env here): a constant "
        "written once at construction"
    ),
    "env.terrain.height_points": (
        "the height-map sample grid, one identical copy per env: a constant "
        "written once at construction"
    ),
    "env.terrain._spawn_sample_index_buffer": (
        "scratch space for the spawn sampler: random_ overwrites it on every "
        "sampling call before its values are read, so nothing persists in it"
    ),
}

slow = pytest.mark.parity_slow
skip_unless_slow = pytest.mark.skipif(
    os.environ.get("PROTOMOTIONS_PARITY_SLOW") != "1",
    reason="MuJoCo scenario build; set PROTOMOTIONS_PARITY_SLOW=1",
)


def _children(value):
    """The labelled parts of a container, or None for a leaf."""
    if isinstance(value, dict):
        return [(f"[{key!r}]", item) for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [(f"[{index}]", item) for index, item in enumerate(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            (f".{field.name}", getattr(value, field.name, None))
            for field in dataclasses.fields(value)
        ]
    attributes = getattr(value, "__dict__", None)
    if attributes is None or isinstance(value, _NOT_CONTAINERS):
        return None
    return [
        (f".{name}", item)
        for name, item in attributes.items()
        if name not in SKIPPED_ATTRIBUTES
    ]


class _Walk:
    """Everything reachable from ``vars(env)``, once per object.

    ``tensors`` maps the path of every tensor whose first dimension is the env
    count to that tensor; ``objects`` maps every container entered to its first
    path; ``unreachable`` lists the containers the depth limit stopped at while
    they still held a per-env tensor nothing else reached.
    """

    def __init__(self, env, max_depth=MAX_DEPTH):
        self.num_envs = env.num_envs
        self.tensors = {}
        self.objects = {}
        self._visited = {id(env)}
        self._stopped = []
        for label, child in _children(env):
            self._visit(child, "env" + label, max_depth)
        self.unreachable = [
            path for path, oid in self._stopped if oid not in self._visited
        ]

    def _per_env(self, tensor):
        return tensor.dim() >= 1 and tensor.shape[0] == self.num_envs

    def _visit(self, value, path, depth):
        if torch.is_tensor(value):
            if self._per_env(value):
                self.tensors[path] = value
            return
        children = _children(value)
        if children is None or id(value) in self._visited:
            return
        if depth == 0:
            # Not entered here; a later, shallower path may still reach it.
            if self._holds_per_env_tensor(value, set(self._visited)):
                self._stopped.append((path, id(value)))
            return
        self._visited.add(id(value))
        self.objects[id(value)] = (path, value)
        for label, child in children:
            self._visit(child, path + label, depth - 1)

    def _holds_per_env_tensor(self, value, seen):
        if torch.is_tensor(value):
            return self._per_env(value)
        children = _children(value)
        if children is None or id(value) in seen:
            return False
        seen.add(id(value))
        return any(self._holds_per_env_tensor(child, seen) for _, child in children)


def _is_owner(obj):
    return (
        callable(getattr(obj, "save_state", None))
        and callable(getattr(obj, "load_state", None))
        and isinstance(getattr(obj, "name", None), str)
    )


def _assert_every_owner_is_declared(env, walk):
    declared = {id(owner) for owner in env._snapshot_owners}
    undeclared = [
        path
        for oid, (path, obj) in walk.objects.items()
        if _is_owner(obj) and oid not in declared
    ]
    assert not undeclared, (
        "state owners reachable from the env but not in env._snapshot_owners "
        f"(their state would never be saved or restored): {undeclared}"
    )


def _allowed(path, allowlist):
    """Whether ``path`` is an allowlisted tensor or lies inside an allowlisted
    container. An entry ``env.<chain>`` matches a path that ends with
    ``.<chain>`` or continues past it with "." or "[", wherever that chain
    starts: ``env._motion.motion_lib`` covers ``env._spawn._motion.motion_lib``
    too, so the test does not depend on which attribute the walk reaches a
    shared object through first. The leading "." anchors a segment boundary."""
    for entry in allowlist:
        chain = entry[len("env") :] if entry.startswith("env.") else entry
        if path.endswith(chain) or chain + "." in path or chain + "[" in path:
            return True
    return False


def _view_key(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _restored(current, original):
    """Back to the original, up to the physical write-back's rounding.

    The simulator poses go back through ``reset_envs`` (float32 tensors into
    float64 MuJoCo coordinates, quaternion order and angular-velocity frame
    conversions, and back), which is exact only to ~1e-7. The perturbation is
    +1, so a tensor that was NOT restored is six orders of magnitude outside
    the tolerance. Integer and bool tensors must match exactly.
    """
    if current.shape != original.shape or current.dtype != original.dtype:
        return False
    if current.is_floating_point():
        return torch.allclose(current, original, rtol=0.0, atol=1e-6)
    return torch.equal(current, original)


@torch.no_grad()
def _perturb(tensors):
    """Change every distinct tensor view once, in place (aliases count once)."""
    seen = set()
    for tensor in tensors.values():
        key = _view_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        if tensor.dtype == torch.bool:
            tensor.logical_not_()
        else:
            tensor.add_(1)


def _assert_round_trip_complete(env, allowlist):
    generator = torch.Generator().manual_seed(5)
    num_actions = env.robot_config.number_of_actions
    env.reset()
    for _ in range(3):
        action = 0.1 * torch.randn(env.num_envs, num_actions, generator=generator)
        env.step(action.to(env.device))

    snapshot = env.save_state()
    walk = _Walk(env)
    assert not walk.unreachable, (
        f"MAX_DEPTH={MAX_DEPTH} stopped at containers that still hold per-env "
        f"tensors; raise it or skip them by name with a reason: {walk.unreachable}"
    )
    _assert_every_owner_is_declared(env, walk)
    before = walk.tensors
    assert before, "the walk found no per-env tensor"
    unused = [
        prefix for prefix in allowlist if not any(_allowed(p, [prefix]) for p in before)
    ]
    assert not unused, f"allowlist entries that match nothing (stale): {unused}"
    # Allowlisted tensors are constants: left alone, so the env stays usable.
    checked = {p: t for p, t in before.items() if not _allowed(p, allowlist)}
    originals = {path: tensor.detach().clone() for path, tensor in checked.items()}
    _perturb(checked)

    env.restore_state(snapshot)

    after = _Walk(env).tensors
    not_restored = []
    for path, original in originals.items():
        current = after.get(path)
        if current is None or not _restored(current, original):
            not_restored.append(path)
    assert not not_restored, (
        "per-env tensors not restored by restore_state (missing snapshot "
        f"entries, or an allowlist entry with a reason): {not_restored}"
    )


def test_fake_mimic_env_snapshot_restores_every_per_env_tensor():
    with eager_mdp_kernels():
        env = build_env(SCENARIOS["fake3_mimic"]).env
        try:
            _assert_round_trip_complete(env, FAKE_ALLOWLIST)
        finally:
            env.close()


@slow
@skip_unless_slow
def test_mujoco_smpl_mimic_env_snapshot_restores_every_per_env_tensor():
    with eager_mdp_kernels():
        env = build_env(SCENARIOS["smpl_mimic"]).env
        try:
            _assert_round_trip_complete(env, MUJOCO_ALLOWLIST)
        finally:
            env.close()
