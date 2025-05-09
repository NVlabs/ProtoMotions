import torch
from torch import nn

from protomotions.utils.device_dtype_mixin import DeviceDtypeModuleMixin


class ReplayBuffer(DeviceDtypeModuleMixin, nn.Module):
    def __init__(self, buffer_size):
        super().__init__()
        self._head = 0
        self._is_full = False
        self._buffer_size = buffer_size
        self._buffer_keys = []

    def reset(self):
        self._head = 0
        self._is_full = False

    def get_buffer_size(self):
        return self._buffer_size

    def __len__(self) -> int:
        return self._buffer_size if self._is_full else self._head

    def store(self, data_dict):
        self._maybe_init_data_buf(data_dict)

        n = next(iter(data_dict.values())).shape[0]
        buffer_size = self.get_buffer_size()
        assert n <= buffer_size

        for key in self._buffer_keys:
            curr_buf = getattr(self, key)
            curr_n = data_dict[key].shape[0]
            assert n == curr_n

            end = self._head + n
            if end >= self._buffer_size:
                diff = self._buffer_size - self._head
                curr_buf[self._head :] = data_dict[key][:diff].clone()
                curr_buf[: n - diff] = data_dict[key][diff:].clone()
                self._is_full = True
            else:
                curr_buf[self._head : end] = data_dict[key].clone()

        self._head = (self._head + n) % buffer_size

    def sample(self, n):
        indices = torch.randint(0, len(self), (n,), device=self.device)

        samples = dict()
        for k in self._buffer_keys:
            v = getattr(self, k)
            samples[k] = v[indices].clone()

        return samples

    def _maybe_init_data_buf(self, data_dict):
        buffer_size = self.get_buffer_size()

        for k, v in data_dict.items():
            if not hasattr(self, k):
                v_shape = v.shape[1:]
                self.register_buffer(
                    k,
                    torch.zeros(
                        (buffer_size,) + v_shape, dtype=v.dtype, device=self.device
                    ),
                    persistent=False,
                )
                self._buffer_keys.append(k)
