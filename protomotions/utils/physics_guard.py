"""Recover individual IsaacLab slots before divergent states reach the learner.

Recoverable events use recovered_*.pt, never the fatal trace filename watched
by the launcher. CUDA/PhysX backend errors are intentionally not swallowed.
"""
import json
import os
from pathlib import Path
import time
import weakref

import torch


class PhysicsGuard:
    def __init__(self, sim, env=None, folder=None, joint_speed_limit=1000., max_consecutive=3):
        self.sim = sim
        self.env_ref = weakref.ref(env) if env is not None else lambda: None
        self.limit = float(joint_speed_limit)
        self.max_consecutive = int(max_consecutive)
        if self.limit <= 0 or self.max_consecutive < 1:
            raise ValueError("Invalid physics guard limits")
        self.folder = Path(folder or "output/physics_guard") / ("rank" + os.environ.get("RANK", "0"))
        self.folder.mkdir(parents=True, exist_ok=True)
        self.retired = torch.zeros(sim.num_envs, dtype=torch.bool, device=sim.device)
        self.invalid = self.retired.clone()
        self.consecutive = torch.zeros(sim.num_envs, dtype=torch.long, device=sim.device)
        self.control_step = 0
        self.event_count = 0
        self.total_recovered = 0

    def raw(self):
        import warp as wp
        from protomotions.utils.physics_trace import FIELDS
        result = {}
        for name in FIELDS:
            value = getattr(self.sim._robot.data, name)
            value = value if isinstance(value, torch.Tensor) else wp.to_torch(value)
            result[name] = value.detach().clone()
        result["contact_forces"] = self.sim._get_simulator_bodies_contact_buf(None).rigid_body_contact_forces.detach().clone()
        return result

    def bad_mask(self, state):
        bad = torch.zeros_like(self.retired)
        for value in state.values():
            bad |= ~torch.isfinite(value).reshape(self.sim.num_envs, -1).all(1)
        bad |= state["joint_vel"].abs().amax(1) > self.limit
        bad |= state["body_lin_vel_w"].abs().flatten(1).amax(1) > self.limit
        return bad

    def park(self, ids):
        """Write a known finite pose, independent of any corrupted state."""
        if ids.numel() == 0:
            return
        sim = self.sim
        root = torch.zeros((len(ids), 13), device=sim.device)
        root[:, 0] = (ids % 32).float() * 2
        root[:, 1] = (ids // 32).float() * 2
        root[:, 2] = -50.
        root[:, 6] = 1.  # IsaacLab native xyzw.
        q = sim.robot_config.default_dof_pos.to(sim.device)[sim.data_conversion.dof_convert_to_sim]
        q = q.unsqueeze(0).repeat(len(ids), 1)
        sim._scene.reset(ids)
        sim._robot.write_root_state_to_sim(root, ids)
        sim._robot.write_joint_state_to_sim(q, torch.zeros_like(q), None, ids)
        sim._robot.set_joint_position_target(q, env_ids=ids)
        sim._robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=ids)
        sim._robot.set_joint_effort_target(torch.zeros_like(q), env_ids=ids)
        sim._common_actions[ids] = 0
        sim._previous_actions[ids] = 0
        sim._prev_prev_actions[ids] = 0

    def record(self, ids, state, substep):
        env = self.env_ref()
        record = {"event": "physics_recovered", "time_ns": time.time_ns(),
                  "control_step": self.control_step, "substep": substep,
                  "env_ids": ids.cpu().tolist(), "joint_speed_limit": self.limit}
        if env is not None:
            record["motion_ids"] = env.motion_manager.motion_ids[ids].cpu().tolist()
            record["motion_times"] = env.motion_manager.motion_times[ids].cpu().tolist()
            record["episode_progress"] = env.progress_buf[ids].cpu().tolist()
            provider = getattr(env, "failure_diagnostics_context", None)
            if provider is not None:
                record["training"] = provider()
        self.event_count += 1
        self.total_recovered += len(ids)
        # Bound detailed snapshots; retain all small event records.
        if self.event_count <= 20:
            path = self.folder / f"recovered_{record['time_ns']}.pt"
            detail = {"record": record, "before_control_step": {k: v[ids].cpu() for k,v in self.before.items()},
                      "bad_state": {k: v[ids].cpu() for k,v in state.items()}}
            trace = getattr(self.sim, "_physics_trace", None)
            if trace is not None:
                detail["latest_reset"] = trace.select(trace.latest_reset, ids)
            torch.save(detail, path)
            record["snapshot"] = str(path)
        with (self.folder / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("PHYSICS_GUARD_RECOVERED " + json.dumps(record), flush=True)

    def inspect_and_recover(self, state, substep):
        bad = self.bad_mask(state)
        fresh = bad & ~self.invalid
        if fresh.any():
            ids = fresh.nonzero().flatten()
            self.record(ids, state, substep)
            self.consecutive[ids] += 1
            self.retired[ids] |= self.consecutive[ids] >= self.max_consecutive
            self.invalid |= fresh
        # Also sanitize already quarantined slots; never integrate their corrupt state.
        ids = (bad | self.invalid).nonzero().flatten()
        if ids.numel():
            self.park(ids)
            repaired = self.raw()
            if self.bad_mask(repaired).any():
                raise RuntimeError("Physics guard could not restore finite simulator state; backend recovery required")

    def step(self):
        self.control_step += 1
        self.invalid.copy_(self.retired)
        self.before = self.raw()
        self.inspect_and_recover(self.before, -1)
        sim = self.sim
        for substep in range(sim.decimation):
            sim._apply_control()
            # Override every control write for quarantined slots.
            self.park(self.invalid.nonzero().flatten())
            sim._scene.write_data_to_sim()
            sim._sim.step(render=False)
            if substep + 1 == sim.decimation and not sim.headless:
                sim._sim.render()
            sim._scene.update(dt=sim._sim.get_physics_dt())
            self.inspect_and_recover(self.raw(), substep)
        self.consecutive[~self.invalid] = 0
        if self.control_step == 1 or self.control_step % 100 == 0 or self.invalid.any():
            status = {"control_steps": self.control_step, "recovered_transitions": self.total_recovered,
                      "retired_env_ids": self.retired.nonzero().flatten().cpu().tolist(),
                      "invalid_this_step": int(self.invalid.sum()), "pid": os.getpid()}
            tmp = self.folder / "status.json.tmp"
            tmp.write_text(json.dumps(status)); tmp.replace(self.folder / "status.json")

    def finish_env_step(self, env):
        bad = self.invalid
        env.rew_buf[bad] = 0
        env.reset_buf[bad] = True
        env.terminate_buf[bad] = True
        env.prev_contact_force_magnitudes[bad] = 0
        env.extras["raw/physics_valid"] = ~bad.clone()
        env.extras["physics_guard/invalid_count"] = bad.sum()
        env.extras["physics_guard/retired_count"] = self.retired.sum()
