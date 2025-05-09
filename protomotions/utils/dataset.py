import numpy as np
from torch.utils.data import Dataset


class GeneralizedDataset(Dataset):
    def __init__(self, batch_size, *tensors, shuffle=False):
        assert len(tensors) > 0
        lengths = {len(t) for t in tensors}
        assert len(lengths) == 1

        self.num_tensors = next(iter(lengths))
        self.batch_size = batch_size
        assert (
            self.num_tensors % self.batch_size == 0
        ), f"{self.num_tensors} {self.batch_size}"
        self.tensors = tensors
        self.do_shuffle = shuffle

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.random.permutation(self.tensors[0].shape[0])
        self.tensors = [t[index] for t in self.tensors]

    def __len__(self):
        return self.num_tensors // self.batch_size

    def __getitem__(self, index):
        assert index < len(self), f"{index} {len(self)}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_tensors)
        return [t[start_idx:end_idx] for t in self.tensors]
