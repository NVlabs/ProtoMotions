# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a policy from a Python experiment configuration.

Quick start
-----------
From the repository root, with IsaacLab installed and a G1 motion file prepared::

    python protomotions/train_agent.py \\
        --robot-name g1 --simulator isaaclab \\
        --experiment-path examples/experiments/mimic/mlp.py \\
        --experiment-name my_tracker \\
        --motion-file /path/to/g1_motions.pt \\
        --num-envs 2048 --batch-size 4096 \\
        --headless true \\
        --overrides agent.num_mini_epochs=2

The experiment builds robot/simulator, terrain, scene, motion, environment, and
agent configs. CLI overrides modify these configs before training. Paths start
at ``robot``, ``simulator``, ``terrain``, ``scene_lib``, ``motion_lib``, ``env``, or
``agent``; use ``agent.num_mini_epochs``, not ``agent.config.num_mini_epochs``.

Fresh training, resume, and warm start
-------------------------------------
* Fresh: use a new experiment name without ``--checkpoint``.
* Resume: rerun with the same experiment name. If its output directory contains
  ``last.ckpt``, this takes precedence over ``--checkpoint``. Saved configs and
  training state are restored; the experiment is not rebuilt and config
  overrides are ignored.
* Warm start: use a new experiment name and add
  ``--checkpoint results/my_tracker/last.ckpt``. Configs are built from the
  requested experiment and overrides; weights are loaded without restoring the
  old training state. The model must be compatible with those weights.

Saved files
-----------
The default output directory is ``results/<experiment-name>/``:

* ``resolved_configs.json``: the resolved config bundle, a tagged JSON tree
  rebuilt through the real constructors on resume; the one saved bundle, from
  which inference derives its view at load.
* ``experiment_config.py``: a copy of the experiment source for reference.
* ``config.yaml``: saved launch arguments.
* ``last.ckpt``: checkpoint written during training.

To inspect configs without training, add ``--create-config-only`` and use a new
experiment name. This rebuilds configs even when a checkpoint was supplied; it
does not convert weights or verify compatibility with an old model.

This file is an import-safe shim: importing it parses nothing and imports no
heavy dependency. ``main()`` picks the simulator backend and hands off to
``protomotions.launch.training.run``. See ``examples/experiments/format.py`` for
experiment hooks and ``docs/source/user_guide/configuration.rst`` for the full
configuration guide.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Import the simulator backend before torch, then run the training body."""
    from protomotions.launch.startup import simulator_from_argv
    from protomotions.utils.simulator_imports import import_simulator_before_torch

    argv = list(sys.argv[1:] if argv is None else argv)
    app_launcher = import_simulator_before_torch(simulator_from_argv(argv))
    from protomotions.launch.training import run

    run(argv, app_launcher)


if __name__ == "__main__":
    from protomotions.launch.startup import run_reporting_failures

    run_reporting_failures(main)
