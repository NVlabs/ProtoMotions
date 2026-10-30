# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive MaskedMimic goals over a saved student policy.

The task and its parameters are chosen here, in the hook. CLI ``--overrides``
are applied to the saved training config before this hook runs, and that config
has no task fields, so they cannot select a task: to change it, copy this file
and edit the ``configure_masked_mimic_task(...)`` call. The hook changes task
controls only; it does not rebuild the trained model.
"""

from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    configure_masked_mimic_task,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


def apply_inference_overrides(configs: ExperimentConfigs, args) -> None:
    if args.full_eval:
        raise ValueError("Interactive goals cannot run motion-tracking full evaluation")
    # This preset targets students trained with local get-up goals. For another
    # student, copy this file and pass getup=None here; the getup field cannot
    # be set from the command line.
    configure_masked_mimic_task(
        configs.env, configs.robot, "idle", getup=GetupRecoveryConfig()
    )
