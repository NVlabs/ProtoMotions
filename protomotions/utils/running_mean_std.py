from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RunningMeanStd(nn.Module):
    def __init__(
        self,
        epsilon: int = 1e-5,
        shape: Tuple[int, ...] = (),
        device="cuda:0",
        clamp_value: Optional[float] = None,
    ):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "mean", torch.zeros(shape, dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "var", torch.ones(shape, dtype=torch.float64, device=device)
        )
        # self.count = epsilon
        self.register_buffer("count", torch.ones((), dtype=torch.long, device=device))
        self.clamp_value = clamp_value

    @torch.no_grad()
    def update(self, arr: torch.tensor) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(
        self, batch_mean: torch.tensor, batch_var: torch.tensor, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        new_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / new_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        self.mean[:] = new_mean
        self.var[:] = new_var
        self.count.fill_(new_count)

    def maybe_clamp(self, x: Tensor):
        if self.clamp_value is None:
            return x
        else:
            return torch.clamp(x, -self.clamp_value, self.clamp_value)

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        if not un_norm:
            result = (arr - self.mean.float()) / torch.sqrt(
                self.var.float() + self.epsilon
            )
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = (
                arr * torch.sqrt(self.var.float() + self.epsilon) + self.mean.float()
            )

        return result
