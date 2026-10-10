"""Teacher-forced evaluation on frozen state/label batches; no simulator required."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from protomotions.agents.common.latent import TARGET_LATENT_KEY
from protomotions.agents.evaluators.gpc_prior_metrics import (
    token_sufficient_statistics, summarize_token_statistics,
    token_histogram, evaluate_token_distribution,
)


@contextmanager
def evaluation_rng(seed, device):
    py, np_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            # Only seed the device used by this rank; do not alter other GPU RNGs.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


@torch.no_grad()
def evaluate_prior_offline(model, batches, *, device, seed=0, top_p=.9,
                           temperature=1., fabric=None):
    """Same states, GT prefixes for NLL; no GT prefixes for sampled histograms.

    batches contain TARGET_LATENT_KEY and optional 'valid_token_mask'. Frozen
    labels are computed by Stage 1 once, never regenerated from the prior.
    """
    spec = model.latent_tokenization
    groups, vocab = spec.num_prior_tokens, spec.prior_token_vocab_size
    stats = torch.zeros(groups, 4, device=device, dtype=torch.float64)
    gt = torch.zeros(groups, vocab, device=device, dtype=torch.long)
    sampled = torch.zeros_like(gt)
    training = model.training
    try:
        with evaluation_rng(seed, device):
            model.eval()
            for batch in batches:
                td = batch.to(device).clone()
                targets = td[TARGET_LATENT_KEY]
                valid = td.get('valid_token_mask', None)
                prefixes = targets if valid is None else targets.masked_fill(~valid.bool(), 0)
                forced = model.prior(model._prior_tensordict(td, prefixes))
                stats += token_sufficient_statistics(forced[model.prior.logits_key], targets, valid)
                del forced
                # _prior_tensordict without targets contains ONLY state context.
                generated = model.prior.generate(model._prior_tensordict(td), num_tokens=groups,
                                                  top_p=top_p, temperature=temperature)
                gt += token_histogram(targets, vocab, valid)
                sampled += token_histogram(generated[model.prior.generated_tokens_key], vocab, valid)
            if fabric is not None and fabric.world_size > 1:
                stats = fabric.all_reduce(stats, reduce_op='sum')
                gt = fabric.all_reduce(gt, reduce_op='sum')
                sampled = fabric.all_reduce(sampled, reduce_op='sum')
            metrics = summarize_token_statistics(stats)
            metrics.update(evaluate_token_distribution(gt, sampled))
            metrics['group_size'] = spec.fsq_scalars_per_prior_token
            return metrics, gt.cpu(), sampled.cpu()
    finally:
        model.train(training)


def frozen_encoder_digest(model):
    """Stable provenance of the actual frozen encoder weights, not a mutable path."""
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(model.target_latent_encoder.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    """Evaluate an exported tensor-only cohort without starting Isaac Lab."""
    import argparse
    import csv
    import hashlib
    import json
    from pathlib import Path
    from tensordict import TensorDict
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.agents.utils.normalization import materialize_lazy_running_stats_from_state_dict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True, nargs='+', help='offline_validation_rank_N.pt exported by evaluator')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--top-p', type=float, default=.9)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--weights', choices=['raw', 'ema'], default='raw')
    args = parser.parse_args()
    if not 0 < args.top_p <= 1 or args.temperature <= 0:
        parser.error('Require 0 < top-p <= 1 and temperature > 0')
    import os
    import torch.distributed as dist
    checkpoint = Path(args.checkpoint)
    cache_paths = [Path(path) for path in args.cache]
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    configs = checkpoint.parent / 'resolved_configs_inference.pt'
    if not configs.exists():
        configs = checkpoint.parent / 'resolved_configs.pt'
    # Repository-native resolved dataclasses: same trusted config path as inference_agent.
    config = torch.load(configs, map_location='cpu', weights_only=False)['agent'].model
    device = torch.device(args.device)
    fabric = None
    if world > 1:
        if device.type == 'cuda':
            device = torch.device('cuda', local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
        class Reduction:
            world_size = world
            def all_reduce(self, value, reduce_op):
                dist.all_reduce(value)
                return value
        fabric = Reduction()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    caches = [torch.load(path, map_location='cpu', weights_only=True) for path in cache_paths[rank::world]]
    with evaluation_rng(args.seed, device):
        model = get_class(config._target_)(config)
        materialize_lazy_running_stats_from_state_dict(model, state['model'])
        model.load_state_dict(state['model'], strict=True)
        actual = state.get('inference_weights', 'raw')
        if args.weights != actual:
            if args.weights == 'ema' and 'ema' in state:
                model.prior.load_state_dict(state['ema']['shadow'], strict=True)
            else:
                raise ValueError(f'Checkpoint contains {actual} weights, not requested {args.weights}; use a full training checkpoint for both')
        model.to(device)
    encoder_sha = frozen_encoder_digest(model)
    batches = []
    for cache in caches:
        if cache.get('encoder_sha256') != encoder_sha:
            raise ValueError('Cached labels were produced by a different frozen Stage 1 encoder')
        if cache['group_size'] != model.latent_tokenization.fsq_scalars_per_prior_token:
            raise ValueError('Cached label grouping does not match checkpoint')
        batches.extend(TensorDict(batch, batch_size=[len(batch[TARGET_LATENT_KEY])]) for batch in cache['batches'])
    metrics, gt, sampled = evaluate_prior_offline(model, batches, device=device,
        seed=args.seed+rank, top_p=args.top_p, temperature=args.temperature, fabric=fabric)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    report = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                  caches=[str(path.resolve()) for path in cache_paths],
                  cache_sha256=[digest(path) for path in cache_paths], world_size=world,
                  config=vars(args), metrics={f'eval/{k}': v for k,v in metrics.items()})
    output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['metric', 'value']); writer.writerows(report['metrics'].items())
    torch.save({'gt_counts': gt, 'sample_counts': sampled}, output.with_suffix('.counts.pt'))
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
