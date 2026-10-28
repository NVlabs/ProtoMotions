# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive MaskedMimic goals over a saved student policy.

Select the task and its parameters with ordinary env config overrides.
This hook changes task controls only; it does not rebuild the trained model.
"""

from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    configure_masked_mimic_task,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


def apply_inference_overrides(configs: ExperimentConfigs, args) -> None:
    if args.full_eval:
        raise ValueError("Interactive goals cannot run motion-tracking full evaluation")
    # This preset targets students trained with local get-up goals. For other
    # students, explicitly disable recovery with the getup=None CLI override.
    configure_masked_mimic_task(
        configs.env, configs.robot, "idle", getup=GetupRecoveryConfig()
    )
