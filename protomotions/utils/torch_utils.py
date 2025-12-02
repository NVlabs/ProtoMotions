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
"""PyTorch utility functions.

Includes helper functions for gradient computation, tensor conversion, and seeding.
"""

import os
import random
import numpy as np
import torch


def grad_norm(params):
    """Compute L2 norm of gradients across all parameters.

    Args:
        params: List of parameters with gradients.

    Returns:
        Scalar tensor with gradient norm.
    """
    grad_norm = 0.0
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad**2)
    return torch.sqrt(grad_norm)


def to_torch(
    x, dtype=torch.float, device="cuda:0", requires_grad=False
) -> torch.Tensor:
    """Convert data to PyTorch tensor with specified dtype and device.

    Args:
        x: Data to convert.
        dtype: PyTorch data type.
        device: Target device.
        requires_grad: Whether tensor requires gradients.

    Returns:
        PyTorch tensor.
    """
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def seeding(seed=0, torch_deterministic=False):
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
        torch_deterministic: If True, configure PyTorch for deterministic execution.

    Returns:
        The seed used.
    """
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed
