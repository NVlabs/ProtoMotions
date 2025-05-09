import numpy as np
import torch
import torch.nn as nn


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super(AverageMeter, self).__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class TensorAverageMeter:
    def __init__(self):
        self.tensors = []

    def add(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self):
        if len(self.tensors) == 0:
            return 0
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return 0
        else:
            return cat.mean()

    def clear(self):
        self.tensors = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self):
        mean_dict = {k: v.mean() for k, v in self.data.items()}
        return mean_dict

    def clear(self):
        self.data = {}

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean
