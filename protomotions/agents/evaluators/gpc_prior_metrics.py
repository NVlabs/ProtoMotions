"""CPU-testable Stage 2 metrics. Natural logs; no label smoothing or dense one-hot."""
import math
import torch
import torch.nn.functional as F


def token_sufficient_statistics(logits, targets, valid=None):
    """Return [group, (NLL sum, top1 sum, top5 sum, valid count)]."""
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected logits [B,G,K] and aligned targets [B,G]")
    valid = torch.ones_like(targets, dtype=torch.bool) if valid is None else valid.bool()
    if valid.shape != targets.shape:
        raise ValueError("valid mask must match targets")
    labels = targets.masked_fill(~valid, 0)
    # Exclude masked NaN logits before CE, not after multiplication by zero.
    values = logits.float().masked_fill(~valid[..., None], 0)
    nll = F.cross_entropy(values.flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
    top1 = values.argmax(-1).eq(labels)
    top5 = values.topk(min(5, values.shape[-1]), -1).indices.eq(labels[..., None]).any(-1)
    return torch.stack([nll.masked_fill(~valid, 0).double().sum(0),
                        (top1 & valid).double().sum(0), (top5 & valid).double().sum(0),
                        valid.double().sum(0)], -1)


def summarize_token_statistics(stats):
    total = stats[:, 3].sum()
    if total <= 0:
        raise ValueError("No valid tokens in evaluation cohort")
    metrics = dict(val_nll=float(stats[:, 0].sum() / total),
                   token_top1_acc=float(stats[:, 1].sum() / total),
                   token_top5_acc=float(stats[:, 2].sum() / total))
    for j in range(len(stats)):
        if stats[j, 3] > 0:
            metrics[f"token_acc_pos_{j}"] = float(stats[j, 1] / stats[j, 3])
    return metrics


def token_histogram(tokens, vocab_size, valid=None):
    if tokens.ndim != 2 or vocab_size < 2:
        raise ValueError("Expected [B,G] tokens and vocabulary >= 2")
    valid = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid.bool()
    if valid.shape != tokens.shape:
        raise ValueError("valid mask must match tokens")
    return torch.stack([torch.bincount(tokens[:, j][valid[:, j]], minlength=vocab_size)
                        for j in range(tokens.shape[1])])


def evaluate_token_distribution(gt_counts, sample_counts):
    """Counts must be globally SUM-reduced before nonlinear entropy/JS metrics."""
    if gt_counts.shape != sample_counts.shape or gt_counts.ndim != 2:
        raise ValueError("Expected matching [group,vocabulary] histograms")
    if (gt_counts < 0).any() or (sample_counts < 0).any():
        raise ValueError("Histogram counts cannot be negative")
    p, q = gt_counts.double(), sample_counts.double()
    if (p.sum(-1) == 0).any() or (q.sum(-1) == 0).any():
        raise ValueError("Every group needs at least one valid sample")
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    hp = -torch.special.xlogy(p, p).sum(-1)
    hq = -torch.special.xlogy(q, q).sum(-1)
    m = (p + q) / 2
    js = .5 * (torch.special.xlogy(p, p / m.clamp_min(1e-300)).sum(-1)
               + torch.special.xlogy(q, q / m.clamp_min(1e-300)).sum(-1))
    k = p.shape[-1]
    return dict(sample_entropy_norm=float((hq / math.log(k)).mean()),
                gt_entropy_norm=float((hp / math.log(k)).mean()),
                token_js_divergence=float(js.mean()),
                token_utilization=float((q > 0).double().mean()),
                gt_token_utilization=float((p > 0).double().mean()),
                effective_vocab=float(hq.exp().mean()), gt_effective_vocab=float(hp.exp().mean()),
                sample_token_count=int(sample_counts.sum()), gt_token_count=int(gt_counts.sum()),
                vocab_size=k, num_groups=p.shape[0])
