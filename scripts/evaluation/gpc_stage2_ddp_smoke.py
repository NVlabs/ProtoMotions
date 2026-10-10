"""Run using torchrun --standalone --nproc_per_node=8, CPU/Gloo only.

Checks real 8-rank reduction with unequal sample counts against an analytical
uniform-logit NLL and a single-process reference histogram. Does not use GPUs.
"""
import math
import torch
import torch.distributed as dist
from tensordict import TensorDict
from protomotions.agents.common.latent import TARGET_LATENT_KEY
from protomotions.agents.evaluators.gpc_prior_offline import evaluate_prior_offline
from protomotions.tests.test_gpc_prior_p0 import FakeModel


def main():
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    class Fabric:
        world_size = world
        def all_reduce(self, value, reduce_op):
            assert reduce_op == 'sum'
            dist.all_reduce(value)
            return value
    n = rank + 1
    batches = [TensorDict({'state': torch.ones(n, 3), TARGET_LATENT_KEY: torch.zeros(n, 2, dtype=torch.long)}, [n])]
    result, gt, sampled = evaluate_prior_offline(FakeModel(), batches, device=torch.device('cpu'),
                                               seed=87+rank, fabric=Fabric())
    assert abs(result['val_nll']-math.log(7)) < 1e-6
    assert result['gt_token_count'] == world*(world+1)
    reference = torch.zeros(2, 7, dtype=torch.long)
    for r in range(world):
        rng = torch.Generator().manual_seed(87+r)
        values = torch.randint(7, (r+1, 2), generator=rng)
        for j in range(2):
            reference[j] += torch.bincount(values[:,j], minlength=7)
    assert torch.equal(sampled, reference)
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    assert all(item == result for item in gathered)
    if rank == 0:
        print(f'GPC_P0_{world}_RANK_GLOO_PASS: {result}')
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
