# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test trained agents and visualize their behavior.

This script loads trained checkpoints and runs agents in the simulation
environment for inference, visualization, and analysis. It supports interactive
controls, video recording, and motion playback.

Inference derives its config view from the training bundle
(``resolved_configs.json``) beside the checkpoint at load, then applies
inference-specific overrides.

Override Priority (highest wins; the view is built in reverse of this order):

1. CLI overrides (``--overrides``) - highest; a later hook that undoes one is
   refused by name
2. The ``--inference-config`` file's ``apply_inference_overrides()`` hook (if passed)
3. The recipe's own ``apply_inference_overrides()``, loaded from the
   ``experiment_config.py`` copy beside the checkpoint
4. Framework inference defaults derived from ``resolved_configs.json`` at load

Note: ``configure_robot_and_simulator()`` is NOT called during inference
(its result is already in the training bundle).

Keyboard Controls
-----------------

During inference, these controls are available:

- **J**: Apply random forces to test robustness
- **R**: Reset all environments
- **O**: Toggle camera view
- **L**: Start/stop video recording
- **Q**: Quit
- **W/A/S/D**: Move target when running with
  ``--inference-config examples/inference/target_keyboard.py``

This file is an import-safe shim: importing it parses nothing and imports no
heavy dependency. ``main()`` picks the simulator backend and hands off to
``protomotions.launch.inference.run``.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Import the simulator backend before torch, then run the inference body."""
    from protomotions.launch.startup import simulator_from_argv
    from protomotions.utils.simulator_imports import import_simulator_before_torch

    argv = list(sys.argv[1:] if argv is None else argv)
    app_launcher = import_simulator_before_torch(simulator_from_argv(argv))
    from protomotions.launch.inference import run

    run(argv, app_launcher)


if __name__ == "__main__":
    from protomotions.launch.startup import run_reporting_failures

    run_reporting_failures(main)
