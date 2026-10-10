"""Real Gloo collectives with empty ranks and variable recovery denominators."""
import math
from types import SimpleNamespace
import torch.distributed as dist
from protomotions.agents.evaluators.gpc_prior_rollout import _all_records, select_rollout_ids
from protomotions.agents.evaluators.gpc_prior_trajectory import cluster_statistics


def main():
    dist.init_process_group('gloo')
    rank,world=dist.get_rank(),dist.get_world_size()
    fabric=SimpleNamespace(world_size=world)
    ids=select_rollout_ids(100,2,11,rank,world)
    records=[dict(x=float(rank),recovery_success_rate__numerator=1 if rank==0 else 0,
                  recovery_success_rate__denominator=1 if rank==0 else 3) for _ in ids]
    combined=_all_records(fabric,records)
    assert len(combined)==2
    result=cluster_statistics(combined,seed=7)
    assert result['x']==.5 and result['recovery_success_rate']==.25
    summaries=[None]*world
    dist.all_gather_object(summaries,result)
    assert all(s==result for s in summaries)
    if rank==0: print(f'GPC_FULL_{world}_RANK_GLOO_PASS: empty ranks + clustered CIs + recovery ratio')
    dist.destroy_process_group()

if __name__=='__main__': main()
