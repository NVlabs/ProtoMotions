# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the ProtoMotions cluster launchers.

`train_slurm.py` (SLURM) and `train_lepton.py` (DGX Cloud Lepton) submit to very
different systems -- sbatch + srun + a job array with a walltime, versus
`lep job create` with no walltime and an in-job restart loop. The inner
`protomotions/train_agent.py` command line is no longer rebuilt here: each
launcher declares only its own cluster flags, parses with `parse_known_args`,
and forwards every remaining token to `train_agent.py` verbatim (SLURM prepends
`--use-slurm`; Lepton has no walltime and relies on filesystem-based resume, so
it omits it). A launcher reads the few values it needs to shape the submission
off that tail through `launch.training_cli.read_launch_facts`. What stays shared
is the code snapshot both launchers rsync to the cluster, so the two cannot
drift on what they upload.
"""

# The code snapshot both launchers rsync to the cluster. One list, because the
# two used to keep their own and drifted: train_lepton's copy was missing
# `.worktrees` and `.claude`, which on a working tree that keeps its git
# worktrees in-tree is tens of gigabytes of duplicate checkouts pushed over a
# Teleport tunnel on every submit.
#
# Everything here is either not code (caches, editor state, session data), a
# duplicate checkout, or a large blob that is pre-staged on the cluster and
# passed by absolute path instead of re-uploaded.
RSYNC_EXCLUDES = [
    ".git",
    ".idea",
    ".claude",
    ".cursor",
    ".vscode",
    ".ruff_cache",
    "**/__pycache__",
    "**/*.egg-info",
    # Local worktrees and in-tree virtualenvs are full duplicate checkouts /
    # gigabytes -- never ship them.
    ".worktrees",
    ".venv**",
    "outputs",
    "output/*",
    "results",
    "exps",
    "tmp",
    "**/*tmp*",
    "wandb",
    # Root-level scratch caches (e.g. token/plan caches).
    "**/_*_cache.pt",
    # Large data blobs, pre-staged on the cluster rather than re-uploaded.
    "data/smpl",
    "data/amass",
    "data/pretrained_models",
    "docs",
    "docs/*",
    # Media that never belongs in a code snapshot.
    "**/*.mp4",
    "**/*.avi",
    "**/*.gif",
    "**/*.jpg",
    "**/*.png",
    "**/*.onnx",
]
