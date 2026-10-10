import math
import numpy as np
import pytest
import torch
from protomotions.agents.evaluators.gpc_prior_trajectory import (
    trajectory_metrics, apd_metrics, fall_recovery_metrics, cluster_statistics, json_safe,
)
from protomotions.agents.evaluators.gpc_prior_rollout import select_rollout_ids
from protomotions.agents.evaluators.gpc_prior_config import GPCPriorEvaluatorConfig


def make_curve(t, power):
    p=torch.zeros(1,len(t),1,3,dtype=torch.float64)
    p[0,:,0,0]=t**power
    return p


def test_acceleration_analytic_and_jerk_zero():
    t=torch.linspace(0,1,101,dtype=torch.float64)
    r=trajectory_metrics(make_curve(t,2),.01,foot_ids=[0])
    assert r['mean_acceleration'].item()==pytest.approx(2.)
    assert r['normalized_jerk'].item()<1e-15
    assert r['foot_jerk'].item()<1e-7


def test_dimensionless_jerk_formula_scaling_and_window_duration():
    t=torch.linspace(0,1,101,dtype=torch.float64)
    p=make_curve(t,3)
    a=trajectory_metrics(p,.01,window_sec=1.)
    # x=t^3, jerk=6; 98 third-difference samples; L=1, D=1.
    assert a['normalized_jerk'].item()==pytest.approx(36*.98,rel=1e-7)
    b=trajectory_metrics(p*7,.03,window_sec=3.)
    assert b['normalized_jerk'].item()==pytest.approx(a['normalized_jerk'].item(),rel=1e-7)
    assert b['mean_acceleration'].item()==pytest.approx(a['mean_acceleration'].item()*7/9)


def test_stationary_or_missing_foot_is_na_with_coverage():
    r=trajectory_metrics(torch.zeros(2,30,3,3),.02)
    assert torch.isnan(r['normalized_jerk']).all()
    assert torch.isnan(r['normalized_foot_jerk']).all()
    assert r['normalized_jerk_valid_window_fraction'].tolist()==[0.,0.]
    assert r['mean_acceleration'].tolist()==[0.,0.]
    assert json_safe({'x':float('nan')})=={'x':None}


def test_apd_known_trajectories_and_repeat_count():
    p=torch.zeros(2,3,2,3)
    p[1,:,:,0]=torch.tensor([0.,1.,2.])[:,None]
    r=apd_metrics(p,p[:,:,0,:2])
    assert r['apd_root']==1.
    assert r['apd_pose']==pytest.approx(math.sqrt(2))
    assert r['apd_pose_mean_body']==1.
    assert r['apd_pair_count']==1
    assert math.isnan(apd_metrics(p[:1],p[:1,:,0,:2])['apd_root'])


def test_recovery_requires_trigger_hold_and_enough_observation():
    bad=torch.tensor([[0,0,0,0,0,0,0,0,0], [0,1,1,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,1,1], [0,1,1,1,1,1,1,1,1]],dtype=torch.bool)
    records=fall_recovery_metrics(bad,~bad,.1,2,.2,.4)
    assert records[0]['recovery_success'] is None and not records[0]['recovery_triggered']
    assert records[1]['recovery_success'] is True and records[1]['fall_count']==1
    assert records[1]['recovery_seconds']==pytest.approx(.2)
    assert records[2]['recovery_censored'] and records[2]['recovery_success'] is None
    assert records[3]['recovery_success'] is False
    push=fall_recovery_metrics(bad[:1],~bad[:1],.1,2,.2,.4,push_steps=[2])
    assert push[0]['recovery_trigger_kind']=='applied_push' and push[0]['recovery_success'] is True
    no_push=fall_recovery_metrics(bad[:1],~bad[:1],.1,2,.2,.4,push_steps=[2],include_push=False)
    assert not no_push[0]['recovery_triggered']


def test_cluster_ratio_uses_counts_not_rank_means():
    records=[dict(x=2.,recovery_success_rate__numerator=1,recovery_success_rate__denominator=1),
             dict(x=4.,recovery_success_rate__numerator=0,recovery_success_rate__denominator=9)]
    a=cluster_statistics(records,seed=1)
    assert a['x']==3. and a['x_std']==pytest.approx(math.sqrt(2))
    assert a['recovery_success_rate']==.1
    assert a==cluster_statistics(records,seed=1)
    missing=cluster_statistics([dict(recovery_success_rate__numerator=0,recovery_success_rate__denominator=0)])
    assert math.isnan(missing['recovery_success_rate'])


def test_global_pose_budget_empty_ranks_and_disjointness():
    parts=[select_rollout_ids(100,2,9,r,8) for r in range(8)]
    assert [len(p) for p in parts]==[1,1,0,0,0,0,0,0]
    assert len(set(torch.cat(parts).tolist()))==2
    combined=torch.cat([select_rollout_ids(100,15,9,r,8) for r in range(8)])
    assert set(combined.tolist())==set(select_rollout_ids(100,15,9,0,1).tolist())


def test_profiles_and_explicit_override():
    import argparse
    from protomotions.agents.evaluators.gpc_prior_cli import add_prior_eval_arguments, explicit_prior_eval_options
    p=argparse.ArgumentParser(); add_prior_eval_arguments(p)
    options=explicit_prior_eval_options(p.parse_args(['--prior-eval-profile','full','--prior-eval-repeats','4']))
    assert options['rollout_num_poses']==15 and options['rollouts_per_pose']==4
    assert options['max_eval_steps']==1000 and options['perturbation_speeds']==(0.,.5,1.5)
    GPCPriorEvaluatorConfig(**options).validate()
    with pytest.raises(ValueError): GPCPriorEvaluatorConfig(rollouts_per_pose=1).validate()
    with pytest.raises(ValueError): GPCPriorEvaluatorConfig(perturbation_speeds=(0.,1.),max_eval_steps=10).validate()


def test_report_media_is_scoped_to_current_evaluation(tmp_path):
    from protomotions.agents.evaluators.gpc_prior_reporting import report_media, log_training_media
    from types import SimpleNamespace
    (tmp_path/'epoch_0_eval_1_raw_train.png').write_bytes(b'')
    (tmp_path/'epoch_0_eval_2_raw_train.png').write_bytes(b'')
    media=report_media({'agent_epoch':0,'eval_count':2},tmp_path)
    assert len(media)==1 and 'eval_2' in next(iter(media))
    # Nonzero ranks must not even access/create a logger experiment.
    class Forbidden:
        @property
        def experiment(self): raise AssertionError('nonzero rank opened W&B')
    log_training_media(SimpleNamespace(global_rank=1,loggers=[Forbidden()]),{},tmp_path,0)
