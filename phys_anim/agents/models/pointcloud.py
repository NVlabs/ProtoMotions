# Implementation adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
# Paper: https://arxiv.org/abs/1801.07829

import torch
from torch import nn
import torch.nn.functional as F

from phys_anim.agents.models.common import NormObsBase


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(NormObsBase):
    # This implementation is adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    # It is slightly modified. Removed batch normalization and dropout.
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in)
        self.non_point_inputs = 8 * 3 + 6 + config.num_contact_bodies * 3
        self.num_out = num_out

        if not hasattr(self, "linear_input_dim"):
            self.linear_input_dim = (
                self.config.emb_dims * 2
                + 8 * 3
                + 6
                + self.config.num_contact_bodies * 3
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=1, bias=False), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16 * 2, 16, kernel_size=1, bias=False), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16 * 2, 32, kernel_size=1, bias=False), nn.ReLU()
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(16 + 16 + 32, self.config.emb_dims, kernel_size=1, bias=False),
            nn.ReLU(),
        )

        # We combine the point cloud features with the object bounding box and orientation
        self.linear1 = nn.Linear(self.linear_input_dim, 128, bias=False)
        self.linear2 = nn.Linear(128, self.num_out)

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        points, identity = (
            input_dict["obs"][:, : -self.non_point_inputs],
            input_dict["obs"][:, -self.non_point_inputs :],
        )
        if return_norm_obs:
            x, obs = self.apply_conv(
                {"obs": points}, already_normalized, return_norm_obs
            )
            obs = torch.cat((obs, identity), dim=-1)
        else:
            x = self.apply_conv({"obs": points}, already_normalized, return_norm_obs)

        x = torch.cat((x, identity), dim=-1)
        outs = self.apply_linear(x)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs

    def apply_conv(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )
        batch_size = obs.size(0)

        obs = obs.reshape(batch_size, -1, 3)
        obs = obs.permute(0, 2, 1)

        x = get_graph_feature(obs, k=self.config.num_neighbours)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.config.num_neighbours)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.config.num_neighbours)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv1d(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        if return_norm_obs:
            return x, obs
        else:
            return x

    def apply_linear(self, x):
        x = F.relu(self.linear1(x))
        outs = self.linear2(x)

        return outs


class PointsWithIdentity(DGCNN):
    def __init__(self, config, num_in: int, num_out: int):
        self.linear_input_dim = (
            config.emb_dims * 2
            + 8 * 3
            + 6
            + config.num_contact_bodies * 3
            + config.max_num_objects
        )
        super().__init__(config, num_in, num_out)
        self.non_point_inputs = (
            8 * 3 + 6 + config.num_contact_bodies * 3 + config.max_num_objects
        )
