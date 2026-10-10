"""Explicit P1/P2 trajectory metrics, independent of simulator and GPU.

NJ = D**5 * integral(||d3x/dt3||**2 dt) / path_length**2.
This is the dimensionless squared-jerk family of Hogan & Sternad (2009),
with arc length as amplitude. This discrete convention is custom, not an
assertion of bitwise equivalence to GPC Table 9's unpublished implementation.
"""
import math
import numpy as np
import torch


def _window_sum(x, length):
    sums = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(1)], 1)
    return sums[:, length:] - sums[:, :-length]


def trajectory_metrics(position, dt, body_ids=None, foot_ids=(), window_sec=.4,
                       min_path=1e-6):
    """[episode,time,body,xyz] -> per-episode metrics; no synthetic padding.

    Zero-distance windows are excluded (undefined normalization), with coverage
    reported. A perfectly stationary trajectory has zero acceleration/raw jerk
    and NA normalized jerk, never a fake normalized score of zero.
    """
    if dt <= 0 or window_sec < 3*dt or min_path <= 0:
        raise ValueError('Need dt > 0, at least three intervals/window, positive min_path')
    if position.ndim != 4 or position.shape[-1] != 3 or position.shape[1] < 4:
        raise ValueError('Expected [E,T>=4,B,3] positions')
    p = position.double()
    body_ids = list(range(p.shape[2])) if body_ids is None else list(body_ids)
    if not body_ids or len(set(body_ids)) != len(body_ids):
        raise ValueError('Body set must be nonempty and unique')
    vel = torch.diff(p, dim=1) / dt
    acceleration = torch.diff(vel, dim=1) / dt
    jerk = torch.diff(acceleration, dim=1) / dt
    result = {'mean_acceleration': acceleration[:,:,body_ids].norm(dim=-1).mean((1,2)),
              'jerk_mps3': jerk[:,:,body_ids].norm(dim=-1).mean((1,2))}
    if foot_ids:
        result['foot_jerk'] = jerk[:,:,list(foot_ids)].norm(dim=-1).mean((1,2))
    # 0.4 s at 50 Hz is 20 intervals, hence 21 position samples.
    intervals = int(round(window_sec/dt))
    frames = intervals+1
    for name, indices in [('normalized_jerk', body_ids), ('normalized_foot_jerk', list(foot_ids))]:
        if not indices or p.shape[1] < frames:
            result[name] = torch.full((len(p),), float('nan'), dtype=torch.float64)
            result[name+'_valid_window_fraction'] = torch.zeros(len(p), dtype=torch.float64)
            continue
        distance = _window_sum((vel[:,:,indices]*dt).norm(dim=-1), intervals)
        # Third forward differences live at central positions. Rectangle
        # quadrature covers the available (frames-3) derivative samples.
        energy = _window_sum(jerk[:,:,indices].square().sum(-1)*dt, frames-3)
        valid = (distance > min_path) & torch.isfinite(distance) & torch.isfinite(energy)
        nj = (intervals*dt)**5 * energy / distance.square().clamp_min(min_path**2)
        nj = nj.masked_fill(~valid, float('nan'))
        result[name] = nj.flatten(1).nanmean(1)
        result[name+'_valid_window_fraction'] = valid.double().mean((1,2))
    return result


def apd_metrics(position, root_xy, body_ids=None):
    """One pose's [R,T,B,3] origin-aligned world positions and [R,T,2].

    Root: mean per-frame XY Euclidean distance. Pose: per-frame L2 of the
    flattened body-coordinate vector (metres); additionally return mean-body L2.
    Pairs then frames are averaged, and the caller weights initial poses equally.
    """
    if position.shape[0] < 2:
        return dict(apd_root=float('nan'), apd_pose=float('nan'), apd_pose_mean_body=float('nan'), apd_pair_count=0)
    indices = list(range(position.shape[2])) if body_ids is None else list(body_ids)
    roots, poses, bodies = [], [], []
    for i in range(len(position)):
        for j in range(i+1, len(position)):
            delta = position[i,:,indices] - position[j,:,indices]
            roots.append((root_xy[i]-root_xy[j]).norm(dim=-1).mean())
            poses.append(delta.flatten(1).norm(dim=-1).mean())
            bodies.append(delta.norm(dim=-1).mean())
    return dict(apd_root=float(torch.stack(roots).mean()), apd_pose=float(torch.stack(poses).mean()),
                apd_pose_mean_body=float(torch.stack(bodies).mean()), apd_pair_count=len(roots))


def fall_recovery_metrics(bad, stable, dt, persistence, hold_seconds, timeout_seconds,
                          push_steps=None, include_push=True):
    """[E,T] booleans include initial state at t=0. One first trigger/episode.

    A fall event requires `persistence` bad samples. Initial non-upright counts
    as a fall/trigger at t=0. A new fall is counted only after stable hold.
    Recovery requires continuous stricter standing stability after the trigger.
    Late triggers without enough observation time are censored, not failures.
    """
    bad, stable = bad.bool().cpu(), stable.bool().cpu()
    hold = max(1, math.ceil(hold_seconds/dt))
    timeout = max(hold, math.ceil(timeout_seconds/dt))
    push_steps = [None]*len(bad) if push_steps is None else push_steps
    result = []
    for e in range(len(bad)):
        fallen = bool(bad[e,0]); falls = int(fallen)
        trigger = 0 if fallen else None
        trigger_kind = 'initial_fall' if fallen else None
        streak, stable_streak = 0, 0
        first_fall = 0 if fallen else None
        success_at = None
        for t in range(1, bad.shape[1]):
            streak = streak+1 if bad[e,t] else 0
            stable_streak = stable_streak+1 if stable[e,t] else 0
            if not fallen and streak >= persistence:
                fallen = True; falls += 1
                if first_fall is None: first_fall = t
                if trigger is None: trigger, trigger_kind = t, 'fall'
            if include_push and push_steps[e] == t and trigger is None:
                trigger, trigger_kind = t, 'applied_push'
                # Pre-trigger stability cannot satisfy the post-trigger hold.
                stable_streak = 0
            if fallen and stable_streak >= hold:
                fallen = False
            if trigger is not None and success_at is None and t > trigger:
                # Only post-trigger samples count; success can occur after a fall.
                if t-trigger >= hold and stable_streak >= hold and t-trigger <= timeout:
                    success_at = t
        eligible = trigger is not None and (success_at is not None or bad.shape[1]-1-trigger >= timeout)
        result.append(dict(fall_count=falls, never_fell_rate=float(falls==0),
                           survival_rate=float(not bad[e,-1]),
                           mean_capped_survival_seconds=(bad.shape[1]-1 if first_fall is None else first_fall)*dt,
                           recovery_triggered=trigger is not None, recovery_eligible=eligible,
                           recovery_censored=trigger is not None and not eligible,
                           recovery_success=(success_at is not None) if eligible else None,
                           recovery_seconds=(success_at-trigger)*dt if success_at is not None else None,
                           recovery_trigger_step=trigger, recovery_trigger_kind=trigger_kind))
    return result


def cluster_statistics(records, *, seed=0, bootstrap_samples=500):
    """Equal-pose means + between-pose SD and clustered bootstrap 95% CI.

    Each record is one initial pose, already averaged over its repeated episodes.
    Ratios with varying denominators are supplied via <metric>__numerator and
    <metric>__denominator, bootstrapped together rather than averaging rates.
    """
    if not records:
        return {}
    keys = sorted({k for r in records for k in r if not k.startswith('_') and '__' not in k})
    ratios = sorted({k.split('__')[0] for r in records for k in r if k.endswith('__numerator')})
    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(len(records), size=(bootstrap_samples,len(records)))
    output = {}
    for key in keys+ratios:
        if key in ratios:
            a = np.array([r.get(key+'__numerator',0.) for r in records], dtype=float)
            b = np.array([r.get(key+'__denominator',0.) for r in records], dtype=float)
            valid = b > 0
            mean = float(a.sum()/b.sum()) if b.sum() else float('nan')
            values = np.divide(a, b, out=np.full_like(a,np.nan), where=valid)
            aa, bb = a[sample_ids].sum(1), b[sample_ids].sum(1)
            boot = np.divide(aa,bb,out=np.full_like(aa,np.nan),where=bb>0)
        else:
            values = np.array([r.get(key,float('nan')) for r in records],dtype=float)
            valid = np.isfinite(values)
            mean = float(values[valid].mean()) if valid.any() else float('nan')
            draws = values[sample_ids]
            counts = np.isfinite(draws).sum(1)
            boot = np.divide(np.nansum(draws,1),counts,out=np.full(len(draws),np.nan),where=counts>0)
        output[key] = mean
        output[key+'_valid_pose_count'] = int(valid.sum())
        output[key+'_std'] = float(np.std(values[valid],ddof=1)) if valid.sum()>1 else float('nan')
        good = boot[np.isfinite(boot)]
        ci = np.quantile(good,[.025,.975]) if len(good) and valid.sum()>1 else [float('nan')]*2
        output[key+'_ci95_low'], output[key+'_ci95_high'] = map(float,ci)
    return output


def json_safe(value):
    """JSON null is NA; W&B/TensorBoard can receive NaN separately."""
    if isinstance(value,dict): return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value
