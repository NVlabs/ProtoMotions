# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record a fixed-seed rollout of the real ``BaseEnv`` driven like training.

The drive mirrors ``BaseAgent.fit``: reset the done envs at the top of every
step, step with a policy action, collect the next done ids. Half way through
comes the epoch boundary in fit's order (``agent.py``, the end of the epoch
loop): the REAL evaluator excursion first (``evaluator.evaluate()`` on
stubbed agent/model/fabric, then ``update_training_state``), THEN the
curriculum's ``max_episode_length`` change, THEN ``env.on_epoch_end``. The
harness then delivers the viewer's R key inside the next simulator step, and
training continues with the done ids it had before evaluation (fit continues
the restored episodes).
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from protomotions.envs.context_paths import resolve_path
from protomotions.tests.parity import mutations
from protomotions.tests.parity.scenarios import (
    REPO_ROOT,
    BuiltEnv,
    Scenario,
    build_env,
    eager_mdp_kernels,
)
from protomotions.tests.parity.spies import (
    CallLog,
    CompilePathSpy,
    install_call_spies,
    reference_motion,
    rng_states,
)
from protomotions.tests.parity.stub_agent import StubAgent, StubModel, make_evaluator

Trace = Dict[str, Any]

ACTION_SEED = 1234
ACTION_SCALE = 0.4
STATE_KEYS = ("rigid_body_pos", "rigid_body_rot", "dof_pos", "dof_vel")
# Context paths read through ``resolve_path`` (each recorded only when present).
CONTEXT_PATHS = (
    "progress_buf",
    "current_action",
    "previous_action",
    "ground_heights",
    "prev_contact_force_magnitudes",
    "contact_forces",
    "current.root_pos",
    "noisy.dof_pos",
    "mimic.ref_state.rigid_body_pos",
    "mimic.ref_state.dof_pos",
    "mimic.future_pos",
    # The task controls' own state, so a schedule or double-buffer change is
    # attributed to the control and not only seen through an observation.
    "steering.tar_dir",
    "steering.tar_speed",
    "steering.prev_root_pos",
    "path.tar_pos",
    "path.head_pos",
    "target.tar_pos",
)


def _git_head() -> str:
    """Commit of the code the harness imported (best effort)."""
    try:
        top = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if top.returncode != 0 or os.path.realpath(top.stdout.strip()) != str(
            REPO_ROOT
        ):
            return "unknown"
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return head.stdout.strip() if head.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _module_version(name: str) -> Optional[str]:
    module = sys.modules.get(name)
    return getattr(module, "__version__", None) if module is not None else None


def _scheduled_reset_ids(
    scenario: Scenario, step: int, done_ids: torch.Tensor
) -> torch.Tensor:
    """Union of the env's dones with the scenario's scheduled partial resets."""
    if not scenario.scheduled_resets:
        return done_ids
    extra = [
        env for env, period in scenario.scheduled_resets if step % period == period - 1
    ]
    if not extra:
        return done_ids
    extra_ids = torch.tensor(extra, dtype=torch.long, device=done_ids.device)
    return torch.unique(torch.cat([done_ids, extra_ids]))


def _context_tensor(ctx, path: str) -> Optional[torch.Tensor]:
    """``resolve_path`` that treats a missing path as absent.

    This is the one accessor the harness uses for context values, so it keeps
    working on both sides of the context refactor as long as ``resolve_path``
    stays the lookup (see the package docstring).
    """
    try:
        value = resolve_path(ctx, path)
    except AttributeError:
        return None
    return value if torch.is_tensor(value) else None


def _as_array_tensor(value) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value.copy())
    return None


def _state_history(env):
    """The env's history ring: ``env._state_history`` where the tree has made
    it private, ``env.state_history`` on a base tree that predates that."""
    history = getattr(env, "_state_history", None)
    return history if history is not None else env.state_history


def _fall_cache(env):
    """The recovery reset's fall-pose cache: the recovery bank's where the
    tree has one (``env._recovery.fall_states``), the env's own attribute on
    a base tree that predates the ``reset`` package; None when not generated."""
    recovery = getattr(env, "_recovery", None)
    if recovery is not None:
        return recovery.fall_states
    return getattr(env, "_fall_reset_states", None)


def _as_scalar(value) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value) and value.numel() == 1:
        return float(value.item())
    if isinstance(value, np.generic):
        return float(value)
    return None


class _Recorder:
    def __init__(self, built: BuiltEnv, log: CallLog, compile_spy: CompilePathSpy):
        self.built = built
        self.env = built.env
        self.log = log
        self.compile_spy = compile_spy
        self.records: List[Dict[str, Any]] = []
        self.eval_steps = 0

    def snap(
        self,
        tag: str,
        obs=None,
        rew=None,
        done=None,
        term=None,
        extras=None,
        *,
        fall_cache: bool = False,
        **fields_,
    ) -> None:
        env = self.env
        record: Dict[str, Any] = {"tag": tag}
        with self.log.paused():
            for source, values in (("obs", obs), ("extras", extras)):
                for key, value in (values or {}).items():
                    if torch.is_tensor(value):
                        record[f"{source}/{key}"] = value.detach().clone()
            for key, value in (("rew", rew), ("done", done), ("term", term)):
                if value is not None:
                    record[key] = value.detach().clone()
            record["progress_buf"] = env.progress_buf.clone()
            record["respawn_root_offset"] = env._spawn.respawn_root_offset.clone()
            record["max_episode_length"] = torch.tensor(
                int(env.max_episode_length), dtype=torch.long
            )
            motion_manager = getattr(reference_motion(env), "motion_manager", None)
            if motion_manager is not None:
                record["motion_ids"] = motion_manager.motion_ids.clone()
                record["motion_times"] = motion_manager.motion_times.clone()
            state = env.simulator.get_robot_state()
            for key in STATE_KEYS:
                record[f"state/{key}"] = getattr(state, key).clone()
            record["sim_actions"] = env.simulator.get_current_actions().clone()
            # The simulator-side runtime state the env snapshots for evaluation
            # (action history, latency, reset age ... or the fake's actions and
            # ticks): what the historical restore bugs missed.
            for name, value in env.simulator.save_state().items():
                tensor = _as_array_tensor(value)
                if tensor is not None:
                    record[f"sim/{name}"] = tensor
            captured = _state_history(env).capture()
            record["history/raw_actions"] = captured.raw_actions.clone()
            record["history/processed_actions"] = captured.processed_actions.clone()
            # What evaluators and evaluation components read through env.context.
            ctx = env.context
            for path in CONTEXT_PATHS:
                value = _context_tensor(ctx, path)
                if value is not None:
                    record[f"ctx/{path}"] = value.detach().clone()
            for name, value in rng_states(env).items():
                record[f"rng/{name}"] = value
            if rew is not None:
                record["identity/rew_is_rew_buf"] = rew is env.rew_buf
                record["identity/done_is_reset_buf"] = done is env.reset_buf
                record["identity/term_is_terminate_buf"] = term is env.terminate_buf
                record["identity/extras_terminate_is_terminate_buf"] = (
                    extras.get("terminate") is env.terminate_buf
                )
            fall_states = _fall_cache(env) if fall_cache else None
            if fall_states is not None:
                for field in fields(fall_states):
                    value = getattr(fall_states, field.name)
                    if torch.is_tensor(value):
                        record[f"fall_cache/{field.name}"] = value.detach().clone()
            for key, value in fields_.items():
                record[key] = (
                    value.detach().clone() if torch.is_tensor(value) else value
                )
        record["compile_path"] = self.compile_spy.take()
        record["calls"] = self.log.take()
        self.records.append(record)

    @contextmanager
    def _recording_evaluator_steps(self):
        """Snap an ``eval/step`` record after every ``env.step`` the evaluator makes.

        A recorder-owned one-shot wrapper around the env's ``step`` (on top of
        the spy when one is installed), removed again on exit.
        """
        env = self.env
        original = env.step
        in_own_dict = "step" in env.__dict__

        def step(action):
            result = original(action)
            obs, rew, done, term, extras = result
            self.eval_steps += 1
            self.snap(
                "eval/step", obs=obs, rew=rew, done=done, term=term, extras=extras
            )
            return result

        env.step = step
        try:
            yield
        finally:
            if in_own_dict:
                env.step = original
            else:
                del env.step

    def _arm_keyboard_reset(self) -> None:
        """Deliver the viewer's R key inside the NEXT simulator step, once.

        ``Simulator.step`` calls ``user_interface.begin_step()`` before physics,
        which clears key presses not consumed during the previous step, so a
        press made between env steps never reaches ``consume_reset_request()``
        on a real simulator (only the fake, whose ``step`` has no
        ``begin_step``, would honour it). A viewer delivers presses during the
        step, after the physics callback; so does this wrapper, then unwraps.
        """
        simulator = self.env.simulator
        original = simulator.step
        in_own_dict = "step" in simulator.__dict__

        def restore():
            if in_own_dict:
                simulator.step = original
            else:
                del simulator.step

        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            restore()
            simulator.user_interface.handle_key_event("R", pressed=True)
            return result

        simulator.step = step

    def _evaluation_fields(self, result) -> Dict[str, Any]:
        env = self.env
        score = float("nan") if result.score is None else float(result.score)
        out: Dict[str, Any] = {
            "eval/num_items": torch.tensor(int(result.num_items), dtype=torch.long),
            "eval/score": torch.tensor(score, dtype=torch.float),
        }
        for name in ("failed", "evaluated"):
            value = getattr(result, name, None)
            if torch.is_tensor(value):
                out[f"eval/{name}"] = value.detach().clone()
        for key, value in (result.logs or {}).items():
            scalar = _as_scalar(value)
            if scalar is not None:
                out[f"eval/logs/{key}"] = torch.tensor(scalar, dtype=torch.float)
        motion_manager = getattr(reference_motion(env), "motion_manager", None)
        if motion_manager is not None:
            out["motion_weights"] = motion_manager.motion_weights.clone()
        return out

    def run(self) -> None:
        env, scenario = self.env, self.built.scenario
        generator = torch.Generator().manual_seed(ACTION_SEED)
        # The policy's action width is the robot config's; the simulator has
        # no action count of its own.
        num_actions = env.robot_config.number_of_actions

        def act() -> torch.Tensor:
            action = ACTION_SCALE * torch.randn(
                env.num_envs, num_actions, generator=generator
            )
            return action.to(env.device)

        def rollout(phase: str, start: int, count: int, done_ids: torch.Tensor):
            for step in range(start, start + count):
                reset_ids = _scheduled_reset_ids(scenario, step, done_ids)
                obs, _ = env.reset(reset_ids)  # BaseAgent.fit: reset done envs first
                self.snap(f"{phase}/reset", obs=obs, reset_ids=reset_ids)
                action = act()
                obs, rew, done, term, extras = env.step(action.clone())
                self.snap(
                    f"{phase}/step",
                    obs=obs,
                    rew=rew,
                    done=done,
                    term=term,
                    extras=extras,
                )
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            return done_ids

        obs, _ = env.reset()
        self.snap("init/reset", obs=obs, fall_cache=True)
        done_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        half = scenario.steps // 2
        done_ids = rollout("train0", 0, half, done_ids)

        # Epoch boundary in BaseAgent.fit's order: the evaluator excursion first.
        with tempfile.TemporaryDirectory(prefix="parity-agent-") as root_dir:
            model = StubModel(env.num_envs, num_actions, device=env.device)
            agent = StubAgent(env, model, root_dir)
            evaluator = make_evaluator(self.built, agent)
            if evaluator is None:
                self.snap("eval/skipped")
            else:
                with self._recording_evaluator_steps():
                    result = evaluator.evaluate()
                evaluator.update_training_state(result)
                self.snap(
                    "eval/restored",
                    obs=env.get_obs(),
                    eval_env_steps=torch.tensor(self.eval_steps, dtype=torch.long),
                    **self._evaluation_fields(result),
                )
        # ... then the curriculum's episode-length change and the epoch hook.
        env.max_episode_length = env.max_episode_length + 5
        env.on_epoch_end(1)

        # Keyboard reset path (R key -> forced timeout -> done at the next
        # check), pressed during the first simulator step of the second half.
        self._arm_keyboard_reset()
        # fit continues the restored episodes with the done ids it had before.
        rollout("train1", half, scenario.steps - half, done_ids)


def record(scenario: Scenario, *, mutation: Optional[str] = None) -> Trace:
    """Record one scenario; ``mutation`` names an entry in ``mutations.MUTATIONS``.

    Spies, the mutation, the compile flag and the process-global random
    generators (torch, numpy, python) are all put back before returning, so
    several recordings can run in one process without leaking into each other.
    """
    started = time.time()
    global_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    undo_mutation = mutations.apply(mutation) if mutation else None
    try:
        with eager_mdp_kernels():
            built = build_env(scenario)
            built_at = time.time()
            log = CallLog()
            with CompilePathSpy(log) as compile_spy:
                try:
                    install_call_spies(built.env, log)
                    recorder = _Recorder(built, log, compile_spy)
                    recorder.run()
                finally:
                    log.uninstall()
                    built.env.close()
    finally:
        if undo_mutation is not None:
            undo_mutation()
        torch.set_rng_state(global_rng[0])
        np.random.set_state(global_rng[1])
        random.setstate(global_rng[2])
    finished = time.time()
    meta = {
        "scenario": asdict(scenario),
        "mutation": mutation,
        "torch": torch.__version__,
        "mujoco": _module_version("mujoco"),
        "python": platform.python_version(),
        "git_head": _git_head(),
        "build_s": built_at - started,
        "rollout_s": finished - built_at,
        "wall_time_s": finished - started,
        "num_records": len(recorder.records),
    }
    return {"meta": meta, "records": recorder.records}


def summarize(trace: Trace) -> Dict[str, Any]:
    """One-line summary used by the CLI."""
    records = trace["records"]
    dones = sum(int(record["done"].sum()) for record in records if "done" in record)
    meta = trace["meta"]
    return {
        "scenario": meta["scenario"]["name"],
        "mutation": meta["mutation"],
        "records": len(records),
        "eval_steps": sum(1 for record in records if record["tag"] == "eval/step"),
        "dones": dones,
        "keys_per_step": len(records[-1]) if records else 0,
        "build_s": round(meta["build_s"], 2),
        "rollout_s": round(meta["rollout_s"], 2),
    }
