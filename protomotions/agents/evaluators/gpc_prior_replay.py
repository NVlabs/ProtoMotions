"""Recompute P1 trajectory metrics and recovery offline from saved physics traces."""
import argparse
import csv
import glob
import json
from pathlib import Path
import numpy as np
import torch
from protomotions.agents.evaluators.gpc_prior_trajectory import (
    trajectory_metrics,apd_metrics,fall_recovery_metrics,cluster_statistics,json_safe,
)


def analyze(paths, window=None):
    records,episodes=[],[]
    signature=None
    for path in paths:
        data=torch.load(path,map_location='cpu',weights_only=True)
        cfg=data['config']; dt=data['dt']
        current=(data['condition'],dt,cfg['seed'],cfg['rollouts_per_pose'],tuple(data['body_names']))
        if signature is None: signature=current
        if current!=signature: raise ValueError('Only matching conditions/protocols may be combined')
        bodies=data['selected_body_ids']; feet=data['foot_body_ids']
        p=data['position']; r=cfg['rollouts_per_pose']
        values=trajectory_metrics(p,dt,bodies,feet,window or cfg['jerk_window_seconds'],cfg['jerk_min_path_m'])
        for val in values.values(): val[data['invalid']]=float('nan')
        pushes=[cfg['perturbation_step'] if data['applied_delta_v'] is not None else None]*len(p)
        events=fall_recovery_metrics(data['bad'],data['stable'],dt,cfg['failure_persistence_steps'],
                                    cfg['recovery_hold_seconds'],cfg['recovery_timeout_seconds'],
                                    pushes,cfg['recovery_trigger']=='fall_or_push')
        for i,event in enumerate(events): event.update({k:float(v[i]) for k,v in values.items()})
        episodes.extend(events)
        if len(p)%r: raise ValueError('Incomplete repeat group in trajectory file')
        for first in range(0,len(p),r):
            sl=slice(first,first+r); valid=~data['invalid'][sl]
            record=apd_metrics(p[sl][valid],data['root_xy'][sl][valid],bodies)
            for key in ['survival_rate','never_fell_rate','fall_count','mean_capped_survival_seconds']+list(values):
                a=np.array([e[key] for e in events[sl]])
                record[key]=float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else float('nan')
            eligible=[e for e in events[sl] if e['recovery_eligible']]
            record['recovery_success_rate__numerator']=sum(e['recovery_success'] for e in eligible)
            record['recovery_success_rate__denominator']=len(eligible)
            records.append(record)
    if not records: raise ValueError('No saved trajectories')
    metrics=cluster_statistics(records,seed=cfg['seed'],bootstrap_samples=cfg['bootstrap_samples'])
    metrics.update(episode_count=len(episodes),initial_pose_count=len(records),
                   recovery_trigger_count=sum(e['recovery_triggered'] for e in episodes),
                   recovery_eligible_count=sum(e['recovery_eligible'] for e in episodes),
                   recovery_censored_count=sum(e['recovery_censored'] for e in episodes))
    return dict(condition=signature[0],metrics=metrics,inputs=list(map(str,paths)),
                note='Stored physical trajectory replay; latency is not recomputed; standing flags retain original thresholds')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectories',required=True,nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--jerk-window',type=float,default=None)
    args=parser.parse_args()
    paths=sorted({p for pattern in args.trajectories for p in glob.glob(pattern)})
    report=analyze(paths,args.jerk_window)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(json_safe(report),indent=2)+'\n')
    with output.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['metric','value']);writer.writerows(json_safe(report['metrics']).items())
    print(json.dumps(json_safe(report['metrics']),indent=2))

if __name__=='__main__': main()
