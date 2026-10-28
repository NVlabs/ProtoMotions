# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseEnv parity harness: record a training-shaped rollout, compare two of them.

What it is
----------
A behaviour-preserving refactor of ``protomotions/envs/base_env/env.py`` must
show that training numbers are bit-for-bit unchanged on CPU. This package
records a fixed-seed rollout of the REAL ``BaseEnv`` -- built the way
``train_agent`` builds it and driven the way ``BaseAgent.fit`` and the real
evaluators drive it -- and compares the recording at the merge base (the
commit the PR branched from) with the recording at the head.

Recording and comparing
-----------------------
From a checkout (MuJoCo environment, CPU)::

    python -m protomotions.tests.parity list
    python -m protomotions.tests.parity record --scenario smpl_mimic --out a.pt
    python -m protomotions.tests.parity compare a.pt b.pt [--atol 0] [--ignore calls]

``compare`` exits 0 when the traces are identical, 1 on a value divergence,
2 when only the call order changed, 3 when the comparison itself failed.

``scripts/env_parity.sh <base-ref> [scenario ...]`` does the whole dance:
exports the base code (no LFS payloads), copies this harness and the test
doubles it runs on into it, records the base twice (``BASE NOT
DETERMINISTIC`` if those differ in values or calls), records the head, and
prints one IDENTICAL / CALL ORDER CHANGED / DIVERGENCE line per scenario,
keeping the traces of any non-identical scenario.

When to run it
--------------
It is the standard check for a PR that touches the environment path: anything
under ``protomotions/envs/``, ``components/motion_lib.py``, the kernel packages
(``envs/obs``, ``envs/rewards``, ``envs/terminations``), a simulator, the
shared test doubles under ``protomotions/tests/support/`` or this package. The
expected outcome is every scenario IDENTICAL, or every changed value named in
the PR body (and the handover register, where the project keeps one).
``PARITY_BASE_OWN_SCENARIOS=1`` and ``PARITY_BASE_OWN_SUPPORT=1`` keep the
base's own scenario table and doubles when the PR changes them (a scenario the
base does not know is then recorded on the head only and reported HEAD-ONLY,
one only the base knows is reported BASE-ONLY, and a run that compared nothing
fails). It is not in CI; the fast self-tests
in ``tests/test_parity_harness.py`` are, and the slow ones run with
``PROTOMOTIONS_PARITY_SLOW=1``.

``python -m protomotions.tests.parity record --mutation NAME`` records with a
deliberate refactor mistake patched in (see ``mutations.MUTATIONS``); a clean
trace must differ from it (or, for a mistake the build check rejects, the
mutated env must fail to build), which is how the harness's sensitivity is
tested.

What a record holds
-------------------
One flat dict per ``env.reset`` / ``env.step`` call, per ``env.step`` the
evaluator makes (``eval/step``) and one for the evaluator write-back
(``eval/restored``): every observation and extras tensor, reward/done/
terminated, ``progress_buf``, ``respawn_root_offset``, ``max_episode_length``,
motion ids/times, the simulator's robot state, current actions and runtime
snapshot (``sim/<name>``: action history, latency, reset age ...), the full
action history (``history/raw_actions``, ``history/processed_actions``), the
values evaluators read through ``env.context`` (``ctx/<path>``), the state of
every random generator (torch, numpy, python, the sensor generator and the
terrain's spawn generator), the fall-pose cache on the initial reset, the
evaluation result and ``motion_weights`` on ``eval/restored``, the aliasing
booleans (``rew is env.rew_buf`` etc.) the agent relies on, and two lists:
``compile_path`` (``all`` / ``subset:n`` for every ``execute_all``, i.e. which
observation calls took the compiled path) and ``calls`` (the order of the
instrumented calls since the previous record).

How two traces are compared
---------------------------
Values first, call order second. Every key except ``calls`` is a value key
(``compile_path`` included: moving post-reset observations onto the compiled
path is a real divergence). ``calls`` only lists the calls whose ORDER the
refactor plan promises to keep (``env.step``/``reset``/``save_state``/
``restore_state``, simulator writes, history/sensor/control step-reset-save-
load, motion sampling, terrain spawn sampling, reset noise); the spies that
record the order of the calls wrap only those, so a pure read helper may be
called differently by the refactor.

The two blind spots this closes
-------------------------------
1. The prototype replayed a HAND-WRITTEN copy of the evaluator's steps between
   ``save_state`` and ``restore_state``. Here the real evaluator runs
   (``stub_agent.make_evaluator`` gives it the agent/model/fabric surface it
   needs) and every step it takes is recorded, so a refactor that changes what
   the evaluator does to the shared env is seen.
2. A value-only comparison cannot see which observation calls take the
   compiled path, the order of internal calls, or a shifted random stream whose
   effect only surfaces later. ``compile_path``, ``calls`` and ``rng/*`` record
   those directly.

How context values are read
---------------------------
The recorder reads every context value through one accessor,
``protomotions.envs.context_paths.resolve_path(ctx, path)``, with the path
string as the record key (``ctx/mimic.ref_state.dof_pos``). ``resolve_path``
accepts both the flat map and attribute objects, so one recorder can compare
a base and a head that differ in context layout.

Known gaps and limitations
--------------------------
- Eager kernels only, on CPU: ``component_manager.TORCH_COMPILE_AVAILABLE`` is
  forced off for the whole recording. Compiled kernels are not compared.
- The G1 scenarios run ``mjcf/g1_bm_no_mesh_box_feet.xml``, a DIFFERENT G1
  model from production's ``g1_holo_compat.xml`` (box feet, joint
  frictionloss 0.03 instead of 0.1, no solimplimit), chosen because the
  production meshes are LFS payloads.
- ``g1_five_body`` pins the env side of the G1 replication recipe (bindings,
  kernels, the mimic control, the termination and evaluator rules) plus
  pushes and action noise, not its physical randomizations: every backend
  draws the friction-bucket and centre-of-mass samples at build (the base
  simulator), but MuJoCo applies neither (only IsaacGym, IsaacLab and Newton
  do), and the body-mass randomization is cleared because MuJoCo refuses it.
  Its build-time random stream therefore also differs from a GPU build of the
  same recipe (no body-mass draw).
- The mutation catalogue mixes source-text edits of today's ``env.py`` (and
  ``BaseEvaluator``) with replacements of state-owner methods
  (``base_env/owners.py``, ``base_env/reset``) and wrappers around one
  method (the phase-order mutation reorders the declared advance list after
  ``BaseEnv.__init__``; the fall-cache mutation drops the restore from
  ``RecoveryBank.generate``). Every PR that moves a patched statement must
  re-port its pattern (``_patch`` fails loudly when one no longer matches);
  the owner replacements and the wrappers follow the structure and survive
  rewording.
- Scenes: no scenario has scene objects. The MuJoCo backend asserts
  ``scene_lib.num_scenes() == 0`` and the armchair mesh is an LFS payload, so
  the scene publisher's non-empty branch, the ``object_state`` snapshot key
  and the object respawn code have no coverage. A scene scenario (box
  primitive on a backend that supports objects, or an object model in the
  fake simulator) is still owed.
- GPU mode (``--simulator``/``--num-envs``) is wired through but untested.
"""

from __future__ import annotations

import importlib

from protomotions.tests.parity.compare import (
    ComparisonResult,
    compare,
    format_result,
    load_trace,
    save_trace,
)
from protomotions.tests.parity.mutations import MUTATIONS, apply, applied

# Everything that pulls in the env, the simulators or the evaluators is
# imported on first use, so ``python -m protomotions.tests.parity compare``
# and ``list`` stay cheap (the script runs compare three times per scenario).
_LAZY = {
    "record": "protomotions.tests.parity.recorder",
    "summarize": "protomotions.tests.parity.recorder",
    "SCENARIOS": "protomotions.tests.parity.scenarios",
    "BuiltEnv": "protomotions.tests.parity.scenarios",
    "Scenario": "protomotions.tests.parity.scenarios",
    "build_env": "protomotions.tests.parity.scenarios",
    "eager_mdp_kernels": "protomotions.tests.parity.scenarios",
    "CallLog": "protomotions.tests.parity.spies",
    "CompilePathSpy": "protomotions.tests.parity.spies",
    "install_call_spies": "protomotions.tests.parity.spies",
    "rng_states": "protomotions.tests.parity.spies",
    "StubAgent": "protomotions.tests.parity.stub_agent",
    "StubFabric": "protomotions.tests.parity.stub_agent",
    "StubModel": "protomotions.tests.parity.stub_agent",
    "make_evaluator": "protomotions.tests.parity.stub_agent",
}

__all__ = [
    "MUTATIONS",
    "ComparisonResult",
    "apply",
    "applied",
    "compare",
    "format_result",
    "load_trace",
    "save_trace",
    *sorted(_LAZY),
]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
