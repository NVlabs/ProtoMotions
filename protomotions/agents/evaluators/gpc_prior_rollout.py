"""Repeated-pose, no-reset closed-loop evaluation and rank-safe aggregation."""
import csv
import logging
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from protomotions.agents.common.latent import LATENT_KEY
from protomotions.agents.evaluators.gpc_prior_offline import evaluation_rng
from protomotions.agents.evaluators.gpc_prior_trajectory import (
    trajectory_metrics, apd_metrics, fall_recovery_metrics, cluster_statistics, json_safe,
)


def select_rollout_ids(count, total, seed, rank, world, shared=True):
    """Global budget independent of world size; zero-work ranks are allowed."""
    budget = len(range(rank,total,world))
    required = total if shared else budget
    if count < required:
        raise ValueError(f'Need {required} distinct starting motions, library has {count}')
    perm = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return perm[:total][rank::world] if shared else perm[:budget]


def _body_ids(robot, configured, feet=False):
    names = robot.kinematic_info.body_names
    if configured is None and feet:
        mapping = robot.common_naming_to_robot_body_names
        configured = tuple(mapping.get('all_left_foot_bodies',[])+mapping.get('all_right_foot_bodies',[]))
    if configured is None:
        configured = tuple(names)
    if len(set(configured)) != len(configured):
        raise ValueError('Evaluation body names must be unique')
    return [names.index(name) for name in configured]


def _standing(state, n, ground, anchor, cfg):
    q = state.rigid_body_rot[:n,anchor]
    up = 1-2*(q[:,0].square()+q[:,1].square())
    return ((state.rigid_body_pos[:n,anchor,2]-ground >= cfg.recovery_height)
            & (up >= cfg.recovery_up_z)
            & (state.root_vel[:n].norm(dim=-1) <= cfg.recovery_max_linear_speed)
            & (state.root_ang_vel[:n].norm(dim=-1) <= cfg.recovery_max_angular_speed))


def _sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def _all_records(fabric, records):
    if fabric.world_size == 1: return records
    if not dist.is_initialized(): raise RuntimeError('Distributed evaluation requires an initialized process group')
    collected = [None]*fabric.world_size
    dist.all_gather_object(collected,records)
    return [record for part in collected for record in part]


def collect_batch(evaluator, ids, speed, batch_seed, directions):
    """Sample complete episodes. A commanded velocity delta is measured before physics."""
    cfg, env, device = evaluator.config, evaluator.env, evaluator.device
    n, anchor = len(ids), env.robot_config.anchor_body_index
    with evaluation_rng(batch_seed,device):
        obs = evaluator._reset_cohort(ids)
        state, bad = evaluator._physical_state(n)
        origin = state.root_pos[:n].clone()
        origin[:,2] = env.terrain.get_ground_heights(state.root_pos[:n]).reshape(-1)
        positions = [(state.rigid_body_pos[:n]-origin[:,None]).detach().cpu()]
        roots = [(state.root_pos[:n,:2]-origin[:,:2]).detach().cpu()]
        initial_dof = state.dof_pos[:n].detach().cpu() if state.dof_pos is not None else None
        initial_root_rot = state.root_rot[:n].detach().cpu()
        ground = env.terrain.get_ground_heights(state.rigid_body_pos[:n,anchor]).reshape(-1)
        bads = [bad.cpu()]
        stables = [_standing(state,n,ground,anchor,cfg).cpu()]
        invalid = torch.zeros(n,dtype=torch.bool,device=device)
        tokens, latency, deltas, actions = [], [], None, []
        for step in range(1,cfg.max_eval_steps+1):
            measured = step>cfg.latency_warmup_steps and (step-cfg.latency_warmup_steps-1)%cfg.latency_stride==0
            if measured: _sync(device); started=time.perf_counter()
            action, extra = evaluator.agent.model.generate(evaluator._obs(obs,n))
            if measured:
                _sync(device)
                latency.append(dict(ms=(time.perf_counter()-started)*1000,batch_size=n))
            invalid |= ~torch.isfinite(action).all(-1)
            # Finite actions are never suppressed after falls.
            action = torch.nan_to_num(action)
            actions.append(action.detach().cpu())
            if LATENT_KEY in extra: tokens.append(extra[LATENT_KEY].detach().cpu())
            if speed>0 and step==cfg.perturbation_step:
                env_ids = torch.arange(n,device=device)
                delta = torch.zeros(n,3,device=device)
                delta[:,:2] = directions.to(device)*speed
                before = env.simulator.get_root_state(env_ids).root_vel.clone()
                env.simulator._apply_root_velocity_impulse(delta,torch.zeros_like(delta),env_ids)
                after = env.simulator.get_root_state(env_ids).root_vel
                deltas = (after-before).detach().cpu()
                if not torch.allclose(after-before,delta,atol=1e-4,rtol=1e-4):
                    raise RuntimeError('Simulator did not apply the requested root velocity increment')
            obs,_,_,_,_ = evaluator._step(action,n)  # reference done is deliberately ignored
            state,bad = evaluator._physical_state(n)
            finite = torch.isfinite(state.rigid_body_pos[:n]).flatten(1).all(-1)
            invalid |= ~finite
            ground = env.terrain.get_ground_heights(state.rigid_body_pos[:n,anchor]).reshape(-1)
            positions.append((state.rigid_body_pos[:n]-origin[:,None]).detach().cpu())
            roots.append((state.root_pos[:n,:2]-origin[:,:2]).detach().cpu())
            bads.append((bad | invalid).cpu())
            stables.append((_standing(state,n,ground,anchor,cfg) & ~invalid).cpu())
    return dict(position=torch.stack(positions,1),root_xy=torch.stack(roots,1),
                bad=torch.stack(bads,1),stable=torch.stack(stables,1),invalid=invalid.cpu(),
                initial_dof=initial_dof,initial_root_rot=initial_root_rot,
                tokens=torch.stack(tokens,1) if tokens else None,
                actions=torch.stack(actions,1),latency=latency,applied_delta_v=deltas)


def _rarity(tokens):
    if tokens is None: return None
    result = torch.zeros(tokens.shape[:2],dtype=torch.float64)
    for j in range(tokens.shape[-1]):
        _, inverse, counts = torch.unique(tokens[:,:,j],return_inverse=True,return_counts=True)
        result -= (counts[inverse].double()/inverse.numel()).log()
    return (result/tokens.shape[-1]).mean(1)


@torch.no_grad()
def run_prior_rollouts(evaluator, *, label='rollout'):
    cfg, env = evaluator.config, evaluator.env
    rank, world = evaluator.fabric.global_rank, evaluator.fabric.world_size
    shared = not env.motion_lib.different_motion_files_across_ranks
    ids = select_rollout_ids(env.motion_lib.num_motions(),cfg.rollout_num_poses,cfg.seed,
                            rank,world,shared).to(evaluator.device)
    repeated_ids = ids.repeat_interleave(cfg.rollouts_per_pose)
    bodies = _body_ids(env.robot_config,cfg.body_names)
    feet = _body_ids(env.robot_config,cfg.foot_names,feet=True)
    if cfg.jerk_window_seconds < 3*env.dt:
        raise ValueError('Jerk window must include at least 3 control intervals')
    output = Path(cfg.output_dir) if cfg.output_dir else Path(evaluator.root_dir)/'gpc_eval'
    output.mkdir(parents=True,exist_ok=True)
    tag = f'epoch_{evaluator.agent.current_epoch}_eval_{evaluator.eval_count+1}_{label}'
    all_metrics, conditions = {}, {}
    names = env.robot_config.kinematic_info.body_names
    for condition_index,speed in enumerate(cfg.perturbation_speeds):
        condition = 'no_push' if speed==0 else 'push_'+str(speed).replace('.','p')+'mps'
        logging.getLogger(__name__).info('GPC rollout %s/%s: rank=%d poses=%d repeats=%d steps=%d', label,condition,rank,len(ids),cfg.rollouts_per_pose,cfg.max_eval_steps)
        parts, latencies = [], []
        # Directions are independent of policy RNG and identical across conditions/weights.
        generator = torch.Generator().manual_seed(cfg.seed+rank*100003)
        angles = torch.rand(len(repeated_ids),generator=generator)*2*math.pi
        directions = torch.stack([angles.cos(),angles.sin()],-1)
        for start in range(0,len(repeated_ids),evaluator.num_envs):
            stop = min(start+evaluator.num_envs,len(repeated_ids))
            part = collect_batch(evaluator,repeated_ids[start:stop],speed,
                                 cfg.seed+rank*100003+start,directions[start:stop])
            latencies.extend(part.pop('latency'))
            parts.append(part)
        records, episodes, videos = [], [], []
        if parts:
            data = {key: torch.cat([part[key] for part in parts]) if parts[0][key] is not None else None
                    for key in parts[0]}
            position, root = data['position'], data['root_xy']
            raw = trajectory_metrics(position,env.dt,bodies,feet,cfg.jerk_window_seconds,cfg.jerk_min_path_m)
            for values in raw.values(): values[data['invalid']]=float('nan')
            pushes = [cfg.perturbation_step if speed>0 else None]*len(position)
            episodes = fall_recovery_metrics(data['bad'],data['stable'],env.dt,cfg.failure_persistence_steps,
                cfg.recovery_hold_seconds,cfg.recovery_timeout_seconds,pushes,cfg.recovery_trigger=='fall_or_push')
            rarity = _rarity(data['tokens'])
            for e, record in enumerate(episodes):
                record.update(rank=rank,motion_id=int(repeated_ids[e]),repeat=e%cfg.rollouts_per_pose,
                              invalid=bool(data['invalid'][e]),condition=condition,
                              initial_upright=bool(~data['bad'][e,0]))
                record.update({key:float(values[e]) for key,values in raw.items()})
                record['token_rarity_nats'] = float(rarity[e]) if rarity is not None else None
                record['applied_delta_v'] = data['applied_delta_v'][e].tolist() if speed>0 else [0.,0.,0.]
            for pose_index,motion in enumerate(ids.tolist()):
                sl=slice(pose_index*cfg.rollouts_per_pose,(pose_index+1)*cfg.rollouts_per_pose)
                pos = position[sl]; rr=root[sl]; these=episodes[sl]
                mismatch = (pos[:,0]-pos[0,0]).abs().max()
                if mismatch>2e-3:
                    raise ValueError(f'Repeated pose {motion} initial positions differ by {float(mismatch)} m')
                # Config disables reset noise; ensure joint states and root orientation also agree.
                if data['initial_dof'] is not None and (data['initial_dof'][sl]-data['initial_dof'][sl][0]).abs().max()>1e-4:
                    raise ValueError('Repeated poses have different initial joint states')
                rotations=data['initial_root_rot'][sl]
                if (1-(rotations*rotations[0]).sum(-1).abs()).abs().max()>1e-4:
                    raise ValueError('Repeated poses have different initial root orientation')
                valid = ~data['invalid'][sl]
                metrics = apd_metrics(pos[valid],rr[valid],bodies)
                for key in ['survival_rate','never_fell_rate','fall_count','mean_capped_survival_seconds']+list(raw):
                    vals=np.array([r[key] for r in these],dtype=float)
                    metrics[key]=float(vals[np.isfinite(vals)].mean()) if np.isfinite(vals).any() else float('nan')
                eligible=[r for r in these if r['recovery_eligible']]
                metrics['recovery_success_rate__numerator']=sum(r['recovery_success'] for r in eligible)
                metrics['recovery_success_rate__denominator']=len(eligible)
                metrics['_motion_id']=motion
                metrics['_rank']=rank
                records.append(metrics)
            if cfg.save_trajectories:
                torch.save(dict(**data,motion_ids=repeated_ids.cpu(),dt=env.dt,body_names=names,
                                parent_indices=env.robot_config.kinematic_info.parent_indices,
                                selected_body_ids=bodies,foot_body_ids=feet,
                                condition=condition,config=cfg.__dict__,episodes=json_safe(episodes)),
                           output/f'{tag}_{condition}_rank_{rank}.trajectories.pt')
            if cfg.video_cases and rank==0:
                from protomotions.agents.evaluators.gpc_prior_video import render_cases
                videos=render_cases(position,episodes,env.robot_config.kinematic_info.parent_indices,
                                    env.dt,output/f'{tag}_{condition}',cfg.video_cases,cfg.video_fps)
        global_records=_all_records(evaluator.fabric,records)
        global_episodes=_all_records(evaluator.fabric,episodes)
        global_latency=_all_records(evaluator.fabric,latencies)
        metrics=cluster_statistics(global_records,seed=cfg.seed,bootstrap_samples=cfg.bootstrap_samples)
        metrics.update(episode_count=len(global_episodes),initial_pose_count=len(global_records),
                       recovery_trigger_count=sum(e['recovery_triggered'] for e in global_episodes),
                       recovery_eligible_count=sum(e['recovery_eligible'] for e in global_episodes),
                       recovery_censored_count=sum(e['recovery_censored'] for e in global_episodes),
                       recovery_success_count=sum(e['recovery_success'] is True for e in global_episodes),
                       invalid_episode_count=sum(e['invalid'] for e in global_episodes),
                       horizon_seconds=cfg.max_eval_steps*env.dt,
                       foot_body_count=len(feet),latency_sample_count=len(global_latency))
        if global_latency:
            timing=np.array([v['ms'] for v in global_latency])
            metrics.update(latency_ms=float(timing.mean()),latency_p95_ms=float(np.quantile(timing,.95)),
                           latency_std_ms=float(timing.std()),
                           latency_batch_size_min=min(v['batch_size'] for v in global_latency),
                           latency_batch_size_max=max(v['batch_size'] for v in global_latency))
        else: metrics.update(latency_ms=float('nan'),latency_p95_ms=float('nan'))
        all_metrics.update({f'{condition}/{key}':value for key,value in metrics.items()})
        if speed==0: all_metrics.update(metrics)
        conditions[condition]=dict(metrics=metrics,poses=global_records,episodes=global_episodes,videos=videos,
                                   latency_samples=global_latency,delta_v_mps=speed)
        if rank==0:
            with (output/f'{tag}_{condition}.episodes.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(global_episodes[0]) if global_episodes else [])
                writer.writeheader();writer.writerows(json_safe(global_episodes))
    protocol=dict(root_coordinate_frame='world XY relative to each episode initial root XY',
                  pose_coordinate_frame='world XYZ with initial root XY and terrain height removed',
                  body_names=[names[i] for i in bodies],foot_names=[names[i] for i in feet],
                  apd_pose_definition='mean over pairs and times of L2 of flattened body XYZ vector; metres; custom',
                  normalized_jerk_definition='D^5 integral squared third forward difference / path_length^2; custom discrete Hogan-Sternad',
                  jerk_window_seconds_actual=round(cfg.jerk_window_seconds/env.dt)*env.dt,
                  jerk_min_path_m=cfg.jerk_min_path_m,
                  latency_definition='synchronized wall time: state TensorDict preparation + all AR groups + frozen decoder; per batch, not per robot',
                  perturbation_timing='after action computation, before control step physics; readback verified',
                  confidence_interval='95% percentile bootstrap over initial-pose clusters; undefined for one pose',
                  sampling_interval_seconds=env.dt)
    return all_metrics,dict(protocol=protocol,conditions=conditions,motion_ids=ids.cpu().tolist())
