# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry-point bodies for the train and inference shims.

Importing this package pulls nothing heavy; the modules under it import torch,
wandb and the simulator backends only when their ``run`` is called.
"""
