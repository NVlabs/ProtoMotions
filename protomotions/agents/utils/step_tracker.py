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
import torch
from torch import Tensor, nn


class StepTracker(nn.Module):
    steps: Tensor

    def __init__(
        self, num_envs: int, min_steps: int, max_steps: int, device: torch.device
    ):
        super().__init__()

        self.register_buffer(
            "steps", torch.zeros(num_envs, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "cur_max_steps", torch.zeros(num_envs, dtype=torch.long), persistent=False
        )

        self.num_envs = num_envs
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.to(device)

    def advance(self):
        self.steps += 1

    def done_indices(self):
        return torch.nonzero(
            torch.greater_equal(self.steps, self.cur_max_steps), as_tuple=False
        ).squeeze(-1)

    def reset_steps(self, env_ids: Tensor = None):
        if env_ids is None:
            env_ids = torch.arange(
                0, self.num_envs, device=self.device, dtype=torch.long
            )

        n = len(env_ids)
        self.steps[env_ids] = 0
        self.cur_max_steps[env_ids] = torch.randint(
            self.min_steps,
            self.max_steps,
            size=[n],
            dtype=torch.long,
            device=self.device,
        )

    def shift_counter(self, env_ids: Tensor, shift: Tensor):
        self.steps[env_ids] -= shift
        self.cur_max_steps[env_ids] -= shift

    @property
    def device(self) -> torch.device:
        """Get device from registered buffers."""
        return self.steps.device
