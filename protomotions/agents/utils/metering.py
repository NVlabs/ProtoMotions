# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Performance timing and metric averaging utilities for profiling training.

This module provides utilities for measuring and reporting time spent in different
parts of the training loop, as well as accumulating and averaging tensor metrics.

Key Classes:
    - Timer: Individual timer for a named operation
    - TimeReport: Manager for multiple timers with reporting
    - TensorAverageMeter: Accumulates and averages tensor values
    - TensorAverageMeterDict: Dictionary of TensorAverageMeters
"""

import time
import torch
from operator import itemgetter


class Timer:
    """Individual timer for measuring elapsed time.

    Tracks total time and number of activations for a named operation.

    Args:
        name: Identifier for this timer.

    Attributes:
        time_total: Cumulative time in seconds.
        num_ons: Number of times timer was activated.
    """

    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.time_total = 0.0
        self.num_ons = 0

    def on(self):
        assert self.start_time is None, "Timer {} is already turned on!".format(
            self.name
        )
        self.num_ons += 1
        self.start_time = time.time()

    def off(self):
        assert self.start_time is not None, "Timer {} not started yet!".format(
            self.name
        )
        self.time_total += time.time() - self.start_time
        self.start_time = None

    def report(self):
        if self.num_ons > 0:
            print(
                "Time report [{}]: {:.2f} {:.4f} seconds".format(
                    self.name, self.time_total, self.time_total / self.num_ons
                )
            )

    def clear(self):
        self.start_time = None
        self.time_total = 0.0


class TimeReport:
    """Manager for multiple timers with reporting capabilities.

    Maintains a collection of named timers and provides methods for
    starting, stopping, and reporting timing statistics.

    Example:
        >>> time_report = TimeReport()
        >>> time_report.add_timer("data_collection")
        >>> time_report.start_timer("data_collection")
        >>> # ... do work ...
        >>> time_report.end_timer("data_collection")
        >>> time_report.report()
    """

    def __init__(self):
        self.timers = {}

    def add_timer(self, name):
        assert name not in self.timers, "Timer {} already exists!".format(name)
        self.timers[name] = Timer(name=name)

    def start_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].on()

    def end_timer(self, name):
        assert name in self.timers, "Timer {} does not exist!".format(name)
        self.timers[name].off()

    def report(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
        else:
            print("------------Time Report------------")

            timer_with_times = []
            for timer_name in self.timers.keys():
                timer_with_times.append(
                    (self.timers[timer_name].time_total, self.timers[timer_name])
                )
            timer_with_times.sort(key=itemgetter(0))

            for _, timer in timer_with_times:
                timer.report()
            print("-----------------------------------")

    def clear_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].clear()
        else:
            for timer_name in self.timers.keys():
                self.timers[timer_name].clear()

    def pop_timer(self, name=None):
        if name is not None:
            assert name in self.timers, "Timer {} does not exist!".format(name)
            self.timers[name].report()
            del self.timers[name]
        else:
            self.report()
            self.timers = {}


class TensorAverageMeter:
    """Accumulates and averages tensor values.

    Collects tensor values and computes their mean. Supports memory optimization
    by storing tensors in lower precision or on CPU.

    Args:
        dtype: Data type for storing tensors (default: torch.float16 to save memory).
        device: Device to store tensors on (default: None keeps original device,
                use 'cpu' to save GPU memory).

    Example:
        >>> meter = TensorAverageMeter()
        >>> meter.add(torch.tensor([1.0, 2.0, 3.0]))
        >>> meter.add(torch.tensor([4.0, 5.0, 6.0]))
        >>> print(meter.mean())  # 3.5
    """

    def __init__(self, dtype=torch.float16, device=None):
        """
        Args:
            dtype: Data type for storing tensors. Use torch.float16 to save memory
                   since these tensors don't require gradients.
            device: Device to store tensors on. If None, keeps on original device.
                   Use 'cpu' to save GPU memory by moving tensors to CPU.
        """
        self.tensors = []
        self.dtype = dtype
        self.device = device

    def add(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        # Detach from computation graph and convert to specified dtype to save memory
        x_optimized = x.detach().to(dtype=self.dtype)

        # Optionally move to specified device (e.g., CPU to save GPU memory)
        if self.device is not None:
            x_optimized = x_optimized.to(device=self.device)

        self.tensors.append(x_optimized)

    def mean(self):
        if len(self.tensors) == 0:
            return 0

        # Move tensors back to GPU if they were stored on CPU
        if self.device == "cpu" and len(self.tensors) > 0:
            # Assume we want result on the same device as the first tensor would have been
            # Use cuda if available, otherwise cpu
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            tensors_on_device = [t.to(target_device) for t in self.tensors]
            cat = torch.cat(tensors_on_device, dim=0)
        else:
            cat = torch.cat(self.tensors, dim=0)

        if cat.numel() == 0:
            return 0
        else:
            # Convert back to float32 for computation to maintain numerical precision
            return cat.float().mean()

    def clear(self):
        self.tensors = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    """Dictionary of TensorAverageMeters for managing multiple metrics.

    Maintains a collection of TensorAverageMeters, automatically creating new
    meters as needed when keys are accessed. Useful for tracking multiple
    training metrics (losses, rewards, etc.).

    Args:
        dtype: Data type for storing tensors in child TensorAverageMeters.
        device: Device to store tensors on.

    Example:
        >>> meter_dict = TensorAverageMeterDict()
        >>> meter_dict.add({"loss": torch.tensor(0.5), "reward": torch.tensor(10.0)})
        >>> meter_dict.add({"loss": torch.tensor(0.3), "reward": torch.tensor(15.0)})
        >>> print(meter_dict.mean())  # {"loss": 0.4, "reward": 12.5}
    """

    def __init__(self, dtype=torch.float16, device=None):
        """
        Args:
            dtype: Data type for storing tensors in child TensorAverageMeters.
                   Use torch.float16 to save memory since these tensors don't require gradients.
            device: Device to store tensors on. If None, keeps on original device.
                   Use 'cpu' to save GPU memory by moving tensors to CPU.
        """
        self.data = {}
        self.dtype = dtype
        self.device = device

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter(dtype=self.dtype, device=self.device)
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
