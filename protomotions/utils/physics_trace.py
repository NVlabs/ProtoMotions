"""Opt-in bounded reset/substep recorder. Disabled unless TRACE_DIR is set.

Native IsaacLab ordering is retained. Reads/clones and finite checks synchronize
GPU work, so a passing diagnostic does not rule out a timing-dependent defect.
"""
import json
import os
from pathlib import Path
import time
import weakref
import torch

FIELDS = ('root_pos_w', 'root_quat_w', 'root_lin_vel_w', 'root_ang_vel_w',
          'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w',
          'joint_pos', 'joint_vel', 'applied_torque', 'joint_pos_target')
RESET_FIELDS = ('root_pos', 'root_rot', 'root_vel', 'root_ang_vel', 'dof_pos', 'dof_vel')


def bind_env(env):
    folder = os.environ.get('PROTOMOTIONS_PHYSICS_TRACE_DIR')
    if folder:
        env.simulator._physics_trace = PhysicsTrace(env, folder)


class PhysicsTrace:
    def __init__(self, env, folder):
        self.env_ref = weakref.ref(env)
        self.sim = env.simulator
        self.folder = Path(folder) / ('rank' + os.environ.get('RANK', '0'))
        self.folder.mkdir(parents=True, exist_ok=True)
        self.step_id = 0
        self.reset_count = 0
        self.latest_reset = {}
        self.reset_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.frames = []
        self.t0 = time.monotonic()
        self.max_velocity = 0.0

    def raw(self):
        import warp as wp
        d = self.sim._robot.data
        result = {}
        for name in FIELDS:
            value = getattr(d, name)
            if not isinstance(value, torch.Tensor):
                value = wp.to_torch(value)
            result[name] = value.detach().clone()
        result['contact_forces'] = self.sim._get_simulator_bodies_contact_buf(None).rigid_body_contact_forces.detach().clone()
        return result

    def context(self):
        env = self.env_ref()
        result = {}
        for label, owner, attr in (
            ('motion_ids', env.motion_manager, 'motion_ids'),
            ('motion_times', env.motion_manager, 'motion_times'),
            ('episode_progress', env, 'progress_buf'),
            ('reset_buf', env, 'reset_buf'),
            ('terminate_buf', env, 'terminate_buf'),
            ('raw_action', env, '_current_raw_action'),
            ('processed_action', env, '_current_processed_action')):
            value = getattr(owner, attr, None)
            if isinstance(value, torch.Tensor):
                result[label] = value.detach().clone()
        return result

    def update_reset(self, prefix, values, ids, subset=False):
        for key, value in values.items():
            label = prefix + key
            if label not in self.latest_reset:
                self.latest_reset[label] = torch.zeros((self.sim.num_envs,) + value.shape[1:], dtype=value.dtype, device=value.device)
            self.latest_reset[label][ids] = value if subset else value[ids]

    def before_reset(self, ids):
        if ids.numel() == 0:
            return
        self.update_reset('before/', self.raw(), ids)
        self.update_reset('old_context/', self.context(), ids)
        self.reset_valid[ids] = True
        self.reset_step[ids] = self.step_id
        self.reset_count += ids.numel()

    def before_state_write(self, state, ids):
        if ids is None:
            ids = torch.arange(self.sim.num_envs, device=self.sim.device)
        self.update_reset('target_native/', {k: getattr(state, k).detach().clone() for k in RESET_FIELDS}, ids, subset=True)
        self.update_reset('new_context/', self.context(), ids)

    @staticmethod
    def select(group, ids):
        return {k: v[ids].detach().cpu().clone() for k, v in group.items()}

    def dump(self, name, ids, reason):
        env = self.env_ref()
        provider = getattr(env, 'failure_diagnostics_context', None)
        context = provider() if provider else {}
        payload = {
            'reason': reason, 'env_ids': ids.detach().cpu(), 'control_step': self.step_id,
            'training_context': context, 'reset_valid': self.reset_valid[ids].cpu(),
            'reset_step': self.reset_step[ids].cpu(),
            'latest_reset': self.select(self.latest_reset, ids),
            'step_context': self.select(self.step_context, ids),
            'frames': [{'substep': i, 'state': self.select(s, ids)} for i, s in self.frames],
            'joint_names': self.sim._robot.joint_names, 'body_names': self.sim._robot.body_names,
            'physics_dt': self.sim._sim.get_physics_dt(), 'decimation': self.sim.decimation,
            'note': 'Native simulator body/joint order and quaternion convention; substep -1 is before physics. GPU reads synchronize execution.',
        }
        tmp = self.folder / (name + '.pt.tmp')
        torch.save(payload, tmp)
        tmp.replace(self.folder / (name + '.pt'))

    def step(self):
        self.step_id += 1
        self.step_context = self.context()
        self.frames = [(-1, self.raw())]
        for idx in range(self.sim.decimation):
            self.sim._apply_control()
            self.sim._scene.write_data_to_sim()
            self.sim._sim.step(render=False)
            if (idx + 1) % self.sim.decimation == 0 and not self.sim.headless:
                self.sim._sim.render()
            self.sim._scene.update(dt=self.sim._sim.get_physics_dt())
            state = self.raw()
            self.frames.append((idx, state))
            bad = torch.zeros(self.sim.num_envs, dtype=torch.bool, device=state['joint_vel'].device)
            for value in state.values():
                bad |= ~torch.isfinite(value).reshape(self.sim.num_envs, -1).all(dim=1)
            bad |= (state['joint_vel'].abs() > 1e5).any(dim=1)
            if bad.any():
                ids = bad.nonzero().flatten()[:32]
                reason = f'nonfinite state or joint speed > 100000 rad/s at control_step={self.step_id} substep={idx}'
                self.dump('failure_' + str(time.time_ns()), ids, reason)
                print('PHYSICS_TRACE_FAILURE ' + reason, flush=True)
                raise RuntimeError(reason + '; saved in ' + str(self.folder))
        self.max_velocity = max(self.max_velocity, float(state['joint_vel'].abs().max()))
        if self.step_id == 1 or self.step_id % 100 == 0:
            ids = torch.tensor(sorted(set([0, min(101, self.sim.num_envs - 1)])), device=state['joint_vel'].device)
            self.dump('latest_sample', ids, 'periodic healthy sample')
            status = {'control_steps': self.step_id, 'env_steps': self.step_id * self.sim.num_envs,
                      'reset_count': self.reset_count, 'max_final_substep_joint_velocity': self.max_velocity,
                      'elapsed_seconds': time.monotonic() - self.t0, 'pid': os.getpid(),
                      'motion_ids': self.step_context['motion_ids'].unique().cpu().tolist(),
                      'time_min': float(self.step_context['motion_times'].min()),
                      'time_max': float(self.step_context['motion_times'].max())}
            tmp = self.folder / 'status.json.tmp'
            tmp.write_text(json.dumps(status, indent=2)); tmp.replace(self.folder / 'status.json')
            print('PHYSICS_TRACE_STATUS ' + json.dumps(status), flush=True)
