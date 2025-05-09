import torch
from torch import Tensor
from typing import Optional


def discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values, gamma, tau):
    lastgaelam = 0
    mb_advs = torch.zeros_like(mb_rewards)
    num_steps = mb_rewards.shape[0]

    for t in reversed(range(num_steps)):
        not_done = 1.0 - mb_fdones[t]
        not_done = not_done

        delta = mb_rewards[t] + gamma * mb_next_values[t] - mb_values[t]
        lastgaelam = delta + gamma * tau * not_done * lastgaelam
        mb_advs[t] = lastgaelam

    return mb_advs


def bounds_loss(mu: Tensor) -> Tensor:
    soft_bound = 1.0
    mu_loss_high = (
        torch.maximum(mu - soft_bound, torch.tensor(0, device=mu.device)) ** 2
    )
    mu_loss_low = (
        torch.minimum(mu + soft_bound, torch.tensor(0, device=mu.device)) ** 2
    )
    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
    return b_loss


def normalization_with_masks(values: Tensor, masks: Optional[Tensor]):
    if masks is None:
        return (values - values.mean()) / (values.std() + 1e-8)

    values_mean, values_var = get_mean_var_with_masks(values, masks)
    values_std = torch.sqrt(values_var)
    normalized_values = (values - values_mean) / (values_std + 1e-8)

    return normalized_values


def get_mean_var_with_masks(values: Tensor, masks: Tensor):
    sum_mask = masks.sum()
    values_mask = values * masks
    values_mean = values_mask.sum() / sum_mask
    min_sqr = (((values_mask) ** 2) / sum_mask).sum() - (
        (values_mask / sum_mask).sum()
    ) ** 2
    values_var = min_sqr * sum_mask / (sum_mask - 1)
    return values_mean, values_var
