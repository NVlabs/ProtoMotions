# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
SLURM Training Launcher for ProtoMotions
=========================================

This script provides a template for launching ProtoMotions training jobs on SLURM clusters.
It handles code synchronization, container setup, and job submission.

BEFORE USING:
-------------
1. Write a site YAML describing your cluster, rather than editing this file.
   Keys are listed in SITE_DEFAULTS below; any you omit keep their default.

       login_node: login.mycluster.edu
       base_dir: /scratch/{account}/experiments
       account: my_allocation
       partition: gpu
       container_mounts: /scratch:/scratch:rw
       container_images:
         isaacgym: /containers/isaacgym.sqsh

   Point the script at it with --site-config PATH or $PROTOMOTIONS_SLURM_SITE.
   A file at slurm_site.yaml in the repository root is picked up
   automatically.

   To reach more than one cluster on the same allocation, list them under
   `clusters:` and pick one with --cluster NAME. Keys at the top level are
   shared by every cluster -- put `account` there, since that is the axis that
   does NOT change when you switch cluster -- and each cluster overrides only
   what differs (its login_node, partition, container images):

       account: my_allocation          # shared: same on every cluster
       default_cluster: a100            # used when --cluster is omitted
       clusters:
         a100:
           login_node: a100-login.mycluster.edu
           partition: gpu
           container_images: {isaaclab: /containers/isaaclab.sqsh}
         l40:
           login_node: l40-login.mycluster.edu
           partition: batch
           container_images: {isaaclab: /containers/isaaclab.sqsh}

   Editing this file directly works, but every `git pull` becomes a merge
   conflict and your cluster paths end up in version control.

2. Check what resolved, before submitting anything (add --cluster NAME to see
   a specific one):

       python protomotions/train_slurm.py --print-site

3. Ensure you have SSH access to your cluster login node
4. Prepare your container images (see CONTAINER SETUP below)

USAGE EXAMPLE:
--------------
python protomotions/train_slurm.py \\
    --robot-name=g1 \\
    --simulator=isaacgym \\
    --num-envs=4096 \\
    --batch-size=32768 \\
    --motion-file=data/motions/my_motion.pt \\
    --experiment-path=examples/experiments/mimic/mimic_mlp.py \\
    --experiment-name=my_experiment \\
    --user=myusername

CONTAINER SETUP:
----------------
You'll need to prepare container images with the required dependencies:
- For IsaacGym: PyTorch + IsaacGym + ProtoMotions dependencies
- For IsaacLab: Isaac Lab + ProtoMotions dependencies
- For Newton: PyTorch + Newton + Warp + ProtoMotions dependencies

Convert Docker images to Singularity/Enroot format as required by your cluster.
"""

import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

# Make `import protomotions...` resolve when this is run as
# `python protomotions/train_slurm.py` -- then the script's own directory, not the
# repo root, is on sys.path -- without requiring PYTHONPATH/direnv or an editable
# install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protomotions._train_launch_common import (  # noqa: E402
    RSYNC_EXCLUDES,
    build_train_agent_args,
)


# =============================================================================
# CLUSTER CONFIGURATION - EDIT THIS SECTION FOR YOUR CLUSTER
# =============================================================================

# These are the shipped defaults. They are deliberately not usable: every value
# below names a placeholder, so a mis-configured submission fails loudly instead
# of quietly sending a job to the wrong place.
#
# Point the script at your cluster with a site file rather than by editing this
# block. Editing here means every `git pull` is a merge conflict, and a private
# cluster path is one `git add -p` away from being published. See SITE_CONFIG
# below for the search order, and `--print-site` to see what was loaded.
SITE_DEFAULTS = {
    # Login node hostname, e.g. "login.mycluster.edu".
    "login_node": "YOUR_CLUSTER_LOGIN_NODE",
    # Base directory for experiments on the cluster filesystem. May contain
    # "{account}", substituted with --account, which is how a site with
    # per-allocation directories keeps one entry here instead of one per team.
    "base_dir": "/path/to/your/experiments/directory",
    # Container images, Singularity .sif or Enroot .sqsh, keyed by simulator.
    "container_images": {
        "isaacgym": "/path/to/containers/isaacgym.sqsh",
        "isaaclab": "/path/to/containers/isaaclab.sqsh",
        "newton": "/path/to/containers/newton.sqsh",
    },
    # Python executable inside each container.
    "python_executables": {
        "isaacgym": "python",
        "isaaclab": "/workspace/isaaclab/isaaclab.sh -p",  # Isaac Lab wrapper
        "newton": "python",
    },
    # Default SLURM account (your allocation/project).
    "account": "your_account",
    # Default SLURM partitions, comma-separated.
    "partition": "gpu",
    # Filesystem mounts for the container.
    "container_mounts": "/scratch:/scratch:rw",
    # Job array size for auto-resume: N tasks run sequentially (one at a time),
    # each up to --slurm-time, so total walltime is N x --slurm-time. A site with
    # long jobs sets this once here; --array-size / --srun-repeat override it.
    "array_size": 5,
    # Extra rsync exclude patterns appended to the built-in list, for
    # workflow/checkout-specific junk (personal side-project dirs, pre-staged data
    # directories, etc.) that should not ship to the cluster on every submit.
    "rsync_excludes": [],
    # Opt-in startup retry (see the --startup-* flags). Defaults to today's
    # behaviour: no retry loop. When on, the batch layer wraps srun in the shared
    # run_with_retry (scripts/cluster/retry_lib.sh), which re-runs a scene build
    # that aborts before training and fast-fails a repeated Python exception. A
    # hung build is caught by the always-on stall watchdog inside srun (lower
    # --stall-timeout-min for known-flaky builds), so there is no separate startup
    # watchdog. These live here so a team can set a cluster-wide default in the site
    # file -- the loader rejects unknown keys, so a site file could not set them
    # without these entries.
    "startup_max_attempts": 1,
    "startup_repeat_limit": 3,
    "startup_retry_backoff_sec": 15,
}

# Anything still equal to its shipped default is unset, and submitting with one
# is refused. Matching on the literal rather than a separate "is it configured"
# flag means a half-filled site file is caught too.
_PLACEHOLDER_MARKERS = ("YOUR_", "/path/to/", "your_account")

# Where a site file is looked for, first match wins:
#
#   1. --site-config PATH
#   2. $PROTOMOTIONS_SLURM_SITE
#   3. slurm_site.yaml, beside the repository this script lives in
#
# Entry 3 is what lets a fork keep cluster settings in the tree, shared through
# git, without them reaching anyone downstream: the file is excluded from the
# release projection, so it is version-controlled but never published.
SITE_CONFIG_ENV = "PROTOMOTIONS_SLURM_SITE"
SITE_CONFIG_DEFAULT_RELPATH = Path("slurm_site.yaml")

# A site file may describe several clusters that share one allocation. These two
# top-level keys structure that: `clusters` maps a name to the settings that
# differ on that cluster, and `default_cluster` names the one used when --cluster
# is omitted. Everything else at the top level (notably `account`) is shared by
# every cluster, which is what keeps the allocation a separate axis from the
# cluster. They are recognised in addition to the per-cluster keys in
# SITE_DEFAULTS.
CLUSTERS_KEY = "clusters"
DEFAULT_CLUSTER_KEY = "default_cluster"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _overlay(site, incoming, *, replace_dicts=False):
    """Overlay `incoming` onto `site`. Dict values merge key-by-key unless
    replace_dicts is set, in which case they replace the whole dict.

    A cluster's overrides replace dicts: a cluster that supports only some
    simulators lists only those container images, and must not inherit the other
    simulators' placeholder paths from SITE_DEFAULTS (which would then be flagged
    as unconfigured and refuse a submission that never needed them)."""
    for key, value in incoming.items():
        if (
            isinstance(value, dict)
            and isinstance(site.get(key), dict)
            and not replace_dicts
        ):
            site[key].update(value)
        else:
            site[key] = dict(value) if isinstance(value, dict) else value


def load_site_config(explicit_path=None, cluster=None):
    """Return SITE_DEFAULTS overlaid with the first site file found.

    A site file is either flat (one cluster, keys at the top level) or defines a
    `clusters:` mapping of named clusters that each override the shared top-level
    settings. `cluster` (from --cluster), else `default_cluster:`, selects one;
    with a single cluster and neither given, that one is used. `account` stays a
    shared top-level key so the same allocation is used whichever cluster is
    picked. The resolved cluster name is recorded under "_cluster"."""
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get(SITE_CONFIG_ENV):
        candidates.append(Path(os.environ[SITE_CONFIG_ENV]))
    candidates.append(_repo_root() / SITE_CONFIG_DEFAULT_RELPATH)

    allowed = set(SITE_DEFAULTS)
    site = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in SITE_DEFAULTS.items()
    }
    for candidate in candidates:
        if not candidate.is_file():
            continue
        import yaml

        loaded = yaml.safe_load(candidate.read_text()) or {}
        clusters = loaded.pop(CLUSTERS_KEY, None) or {}
        default_cluster = loaded.pop(DEFAULT_CLUSTER_KEY, None)

        unknown = set(loaded) - allowed
        if unknown:
            raise ValueError(
                f"{candidate}: unknown key(s) {sorted(unknown)}; expected any of "
                f"{sorted(allowed | {CLUSTERS_KEY, DEFAULT_CLUSTER_KEY})}"
            )
        # Shared, top-level settings (e.g. account); a cluster may override any.
        _overlay(site, loaded)

        chosen = None
        if clusters:
            if not isinstance(clusters, dict):
                raise ValueError(
                    f"{candidate}: '{CLUSTERS_KEY}' must be a mapping of name -> settings"
                )
            chosen = cluster or default_cluster
            if chosen is None:
                if len(clusters) == 1:
                    chosen = next(iter(clusters))
                else:
                    raise ValueError(
                        f"{candidate}: defines clusters {sorted(clusters)} but none was "
                        f"selected; pass --cluster NAME or set '{DEFAULT_CLUSTER_KEY}'"
                    )
            if chosen not in clusters:
                raise ValueError(
                    f"{candidate}: unknown cluster {chosen!r}; available: {sorted(clusters)}"
                )
            overrides = clusters[chosen] or {}
            unknown_c = set(overrides) - allowed
            if unknown_c:
                raise ValueError(
                    f"{candidate}: cluster {chosen!r} has unknown key(s) {sorted(unknown_c)}; "
                    f"expected any of {sorted(allowed)}"
                )
            _overlay(site, overrides, replace_dicts=True)
        elif cluster is not None:
            raise ValueError(
                f"{candidate}: --cluster {cluster!r} was given but this site file defines "
                f"no '{CLUSTERS_KEY}'"
            )
        elif default_cluster is not None:
            # A default_cluster with nothing to select from is a mis-structured
            # file, not a silent no-op.
            raise ValueError(
                f"{candidate}: '{DEFAULT_CLUSTER_KEY}' is set to {default_cluster!r} but no "
                f"'{CLUSTERS_KEY}' mapping is defined"
            )

        site["_source"] = str(candidate)
        site["_cluster"] = chosen
        site["_clusters_available"] = sorted(clusters)
        return site

    site["_source"] = None
    site["_cluster"] = None
    site["_clusters_available"] = []
    return site


def unconfigured_values(site):
    """Names of settings still carrying a shipped placeholder."""
    unset = []
    for key, value in site.items():
        if key.startswith("_"):
            continue
        items = value.items() if isinstance(value, dict) else [(None, value)]
        for sub, val in items:
            if isinstance(val, str) and any(m in val for m in _PLACEHOLDER_MARKERS):
                unset.append(key if sub is None else f"{key}.{sub}")
    return unset


# =============================================================================
# END CLUSTER CONFIGURATION
# =============================================================================

# The names used throughout the rest of this file. Seeded from the shipped
# defaults so importing the module is side-effect free; main() replaces them
# with the loaded site before anything touches the cluster.
CLUSTER_LOGIN_NODE = SITE_DEFAULTS["login_node"]
CLUSTER_BASE_DIR = SITE_DEFAULTS["base_dir"]
CONTAINER_IMAGES = dict(SITE_DEFAULTS["container_images"])
PYTHON_EXECUTABLES = dict(SITE_DEFAULTS["python_executables"])
CONTAINER_MOUNTS = SITE_DEFAULTS["container_mounts"]
DEFAULT_SLURM_ACCOUNT = SITE_DEFAULTS["account"]
DEFAULT_SLURM_PARTITION = SITE_DEFAULTS["partition"]


def apply_site(site):
    """Publish a loaded site into the module-level names used below."""
    globals().update(
        CLUSTER_LOGIN_NODE=site["login_node"],
        CLUSTER_BASE_DIR=site["base_dir"],
        CONTAINER_IMAGES=site["container_images"],
        PYTHON_EXECUTABLES=site["python_executables"],
        CONTAINER_MOUNTS=site["container_mounts"],
    )


def subprocess_run(cmd, ignore_err=False, **kwargs):
    """Run subprocess command and raise on error unless ignored."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0 and not ignore_err:
        raise Exception(f"Command failed: {cmd}")
    return result


def check_wandb_credentials(user):
    """
    Check if wandb credentials are available on the remote machine.
    Returns the API key if it needs to be passed explicitly, or None if already configured.
    """
    # Check for ~/.netrc with wandb.ai entry
    check_cmd = f"ssh {user}@{CLUSTER_LOGIN_NODE} \"grep -q 'machine api.wandb.ai' ~/.netrc 2>/dev/null && echo 'found' || echo 'not_found'\""
    result = subprocess_run(
        check_cmd, shell=True, capture_output=True, text=True, ignore_err=True
    )

    if result.stdout.strip() == "found":
        print("WANDB credentials found in ~/.netrc on remote. No API key needed.")
        return None

    # Check environment variable
    check_env_cmd = f'ssh {user}@{CLUSTER_LOGIN_NODE} "bash -l -c \'test -n \\"\\$WANDB_API_KEY\\" && echo found || echo not_found\'"'
    result = subprocess_run(
        check_env_cmd, shell=True, capture_output=True, text=True, ignore_err=True
    )

    if result.stdout.strip() == "found":
        print("WANDB_API_KEY found in remote environment.")
        return None

    # Try local environment
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb_api_key:
        print("Using WANDB_API_KEY from local environment.")
        return wandb_api_key

    # Prompt user
    print("WANDB credentials not found. Options:")
    print("  1. Run 'wandb login' on the cluster")
    print("  2. Set WANDB_API_KEY in your cluster ~/.bashrc")
    print("  3. Enter API key now")
    wandb_api_key = input("Enter WANDB API key (or press Enter to skip): ").strip()
    return wandb_api_key if wandb_api_key else None


def create_parser():
    """Create argument parser with all training options."""
    parser = argparse.ArgumentParser(
        description="Launch ProtoMotions training on SLURM cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--robot-name", type=str, required=True, help="Robot name (e.g., 'g1', 'smpl')"
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator (isaacgym/isaaclab/newton)",
    )
    parser.add_argument(
        "--num-envs", type=int, required=True, help="Number of parallel environments"
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Training batch size"
    )
    parser.add_argument(
        "--motion-file", type=str, required=True, help="Path to motion data file"
    )
    parser.add_argument(
        "--experiment-path", type=str, required=True, help="Path to experiment config"
    )
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Experiment name for logging"
    )
    parser.add_argument("--user", type=str, required=True, help="Cluster username")

    # Optional arguments
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    # --headless is intentionally NOT declared here. train_agent.py owns it
    # (nargs="?", parse_bool), and parse_known_args forwards it through unchanged.
    # Declared here as a bare flag it silently swallowed the value -- the same
    # class of bug as --seed below.
    training_limit_group = parser.add_mutually_exclusive_group()
    training_limit_group.add_argument(
        "--training-max-steps",
        type=int,
        default=10000000000,
        help="Max training steps",
    )
    training_limit_group.add_argument(
        "--training-max-iterations",
        type=int,
        default=None,
        help="Max complete training iterations",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="physical_animation",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--use-slurm", action="store_true", default=True, help="Enable SLURM autoresume"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--torch-deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic PyTorch ops (forwarded to train_agent.py)",
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[], help="Config overrides (key=value)"
    )

    # SLURM arguments
    parser.add_argument("-t", "--slurm-time", default="4:00:00", help="Job time limit")
    parser.add_argument(
        "--stall-timeout-min",
        type=int,
        default=30,
        help="Kill training if the SLURM log has not grown for this many minutes -- "
        "a wedged rank makes the whole job go silent at the distributed barrier -- so "
        "the array's next element restarts (on a possibly different node) instead of "
        "the job sitting until the walltime. 0 disables; raise it for very long epochs.",
    )
    # Left as None so the site file can supply them; a value given here still
    # wins, because the site is only consulted for what the caller omitted.
    parser.add_argument("-a", "--account", default=None, help="SLURM account")
    parser.add_argument("-p", "--partition", default=None, help="SLURM partition")
    parser.add_argument(
        "--site-config",
        default=None,
        help="Cluster site YAML; overrides $PROTOMOTIONS_SLURM_SITE and slurm_site.yaml",
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="Which cluster in the site file's 'clusters:' to submit to "
        "(default: the site's default_cluster). --print-site lists the choices.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository to rsync to the cluster (default: the one this script lives in)",
    )
    parser.add_argument(
        "--print-site",
        action="store_true",
        help="Print the resolved cluster settings and exit, without submitting",
    )
    parser.add_argument(
        "--array-size",
        "--srun-repeat",
        dest="array_size",
        type=int,
        default=None,
        help="Job array size for auto-resume: N tasks run one at a time, each up "
        "to --slurm-time, so total walltime is N x --slurm-time. --srun-repeat is "
        "a backward-compatible alias. Default: the site's array_size (else 5).",
    )
    # Opt-in startup retry. All default to None so the site file can supply a
    # team-wide value (main() fills them like array_size); an explicit flag still
    # wins. With the resolved default (1 attempt) the emitted script is
    # byte-identical to the no-retry path.
    parser.add_argument(
        "--startup-max-attempts",
        type=int,
        default=None,
        help="In-place srun attempts per array element, for scene-build failures "
        "that abort BEFORE epoch 0 (no training progress at stake). 1 (the default) "
        "emits no retry loop at all. Retries reuse the same allocation instead of "
        "costing a fresh array-element scheduling round trip. A hung (not aborted) "
        "scene build is killed by the always-on stall watchdog inside srun -- lower "
        "--stall-timeout-min for known-flaky builds -- which then feeds this loop.",
    )
    parser.add_argument(
        "--startup-repeat-limit",
        type=int,
        default=None,
        help="Stop retrying after this many attempts fail with the SAME Python "
        "exception -- a deterministic bug, not a flaky abort. 0 disables the check. "
        "Inert while --startup-max-attempts is 1. Default 3.",
    )
    parser.add_argument(
        "--startup-retry-backoff-sec",
        type=int,
        default=None,
        help="Seconds to sleep between startup attempts (default 15).",
    )
    parser.add_argument(
        "--only-upload-code", action="store_true", help="Only sync code, don't submit"
    )

    return parser


def sync_code_to_cluster(user, exp_folder, local_repo, extra_excludes=None):
    """Sync local code to cluster, excluding unnecessary files.

    `extra_excludes` (from the site file's rsync_excludes) is appended to the
    built-in list, for workflow-specific junk and pre-staged data directories.
    """
    exclude_patterns = list(RSYNC_EXCLUDES)
    if extra_excludes:
        exclude_patterns += list(extra_excludes)
    exclude_str = " ".join([f'--exclude="{f}"' for f in exclude_patterns])

    # Create remote directory
    subprocess_run(
        f"ssh {user}@{CLUSTER_LOGIN_NODE} 'mkdir -p {exp_folder}'", shell=True
    )

    # Sync code
    rsync_cmd = f"rsync -az --partial -m --chmod=775 {exclude_str} {local_repo}/ {user}@{CLUSTER_LOGIN_NODE}:{exp_folder}/"
    print(f"Syncing code: {rsync_cmd}")
    subprocess_run(rsync_cmd, shell=True)


def _retry_lib_source(exp_folder) -> str:
    """bash sourcing the shared retry/watchdog primitives (kill_tree,
    run_with_stall_watchdog, run_with_retry) from the staged snapshot. Both the
    batch layer and the in-srun command source the SAME file, so their retry and
    watchdog logic cannot drift from lepton_train_loop.sh's. Absolute path: the
    batch layer never cds, and inside srun the snapshot is mounted at exp_folder
    too. Unit-tested by sourcing the file directly (test_cluster_retry_lib.py)."""
    return f"source {exp_folder}/scripts/cluster/retry_lib.sh"


def build_job_command(args, exp_folder, python_path, passthrough_args=None):
    """Build the training command to run inside the container.

    `passthrough_args` are CLI flags this launcher does not recognise (typically
    experiment-contributed, e.g. gpc/prior.py's --tracker-checkpoint); they are
    forwarded verbatim to train_agent.py, which does know them.
    """
    # Simulator-specific install. The IsaacLab container's PATH points at a venv
    # whose bare `python`/`pip` are not runnable, so pip must go through
    # `isaaclab.sh -p -m pip`; a plain `pip install` there fails with exit 127.
    if args.simulator == "isaaclab":
        ilab_pip = "/workspace/isaaclab/isaaclab.sh -p -m pip"
        # NB: no `pip install --upgrade wandb` here. The image's wandb already
        # satisfies protomotions' pin (<0.24); an unpinned upgrade pulls whatever
        # shipped that day (wandb 0.29 violates the pin and drags click to 8.5,
        # which conflicts with isaacsim-kernel's click==8.1.7). Bake a specific
        # wandb into the image if a newer one is genuinely needed.
        install_cmds = (
            f"{ilab_pip} uninstall -y protomotions || true; "
            f"{ilab_pip} install -e . --no-deps; "
        )
    else:
        install_cmds = (
            "pip uninstall -y protomotions 2>/dev/null; "
            "pip install -e . --no-dependencies; "
        )

    # IsaacLab lazily bootstraps a shared pip env when AppLauncher first starts;
    # warm it once on local rank 0 so the node's ranks do not race on it.
    warmup_cmd = ""
    if args.simulator == "isaaclab":
        warmup_cmd = (
            'ISAAC_WARMUP_DIR="/tmp/isaaclab_warmup_${SLURM_JOB_ID}_${SLURMD_NODENAME}"; '
            'ISAAC_WARMUP_READY="${ISAAC_WARMUP_DIR}/ready"; '
            'ISAAC_WARMUP_FAIL="${ISAAC_WARMUP_DIR}/failed"; '
            'if [ "${SLURM_LOCALID:-0}" = "0" ]; then '
            'mkdir -p "${ISAAC_WARMUP_DIR}"; rm -f "${ISAAC_WARMUP_READY}" "${ISAAC_WARMUP_FAIL}"; '
            f'if {python_path} -c "from isaaclab.app import AppLauncher; app = AppLauncher({{\\"headless\\": True}}).app; app.close()"; then '
            'touch "${ISAAC_WARMUP_READY}"; '
            'else rc=$?; echo "${rc}" > "${ISAAC_WARMUP_FAIL}"; exit "${rc}"; fi; '
            "else waited=0; "
            'while [ ! -f "${ISAAC_WARMUP_READY}" ] && [ ! -f "${ISAAC_WARMUP_FAIL}" ]; do '
            "sleep 2; waited=$((waited + 2)); "
            'if [ "${waited}" -ge 300 ]; then echo "Timed out waiting for IsaacLab warmup"; exit 1; fi; done; '
            'if [ -f "${ISAAC_WARMUP_FAIL}" ]; then echo "IsaacLab warmup failed on local rank 0"; exit 1; fi; '
            "fi; "
        )

    # Only local rank 0 installs; other ranks wait on a marker so they do not race
    # on shared container packages (relevant when --ngpu > 1 puts several ranks per node).
    #
    # Clearing the marker belongs to rank 0, inside the branch. Done before it,
    # by every rank, a rank that starts late -- a slow image pull is enough --
    # deletes the marker after rank 0 has already installed and touched it, and
    # then waits for a file nobody will write again. The waiters also get a
    # timeout, so a failed install on rank 0 ends the job instead of hanging it
    # until the wall clock does. The warmup block below already works this way.
    job_cmd = (
        f"cd {exp_folder}; "
        'if [ "${SLURM_LOCALID:-0}" = "0" ]; then '
        f"rm -f /tmp/_pip_done_${{SLURM_JOB_ID}}; {install_cmds} "
        f"touch /tmp/_pip_done_${{SLURM_JOB_ID}}; "
        "else waited=0; while [ ! -f /tmp/_pip_done_${SLURM_JOB_ID} ]; do "
        "sleep 2; waited=$((waited + 2)); "
        'if [ "${waited}" -ge 900 ]; then echo "Timed out waiting for the rank-0 install"; exit 1; fi; '
        "done; fi; "
        f"{warmup_cmd}"
    )

    # Add WANDB API key if needed, as its own `export` statement. A bare
    # `WANDB_API_KEY=<key> ` prefix would attach to the next `export
    # PYTHONUNBUFFERED=1` builtin (below) rather than the training process, so the
    # key would never reach train_agent.py.
    if args.use_wandb:
        wandb_key = check_wandb_credentials(args.user)
        if wandb_key:
            job_cmd += f"export WANDB_API_KEY={wandb_key}; "

    # Assemble the train_agent.py command, then run it under the stall watchdog
    # below. The shared tail (robot/sim/motion/.../overrides, incl. experiment
    # passthrough) comes from build_train_agent_args so this stays in lockstep with
    # train_lepton.py and the two launchers cannot drift. Appended after it are the
    # SLURM-only flags: --use-slurm (the walltime autoresume callback that
    # stops-and-requeues at ~3.5h, which Lepton omits) and the reproducibility
    # flags this parser owns -- --seed / --torch-deterministic are consumed by
    # parse_known_args, so they must be forwarded explicitly or --seed 3 is
    # silently dropped and every job runs train_agent's default seed 0. All three
    # start with -- so the trailing --overrides (nargs="*") does not swallow them.
    # (--seed / --torch-deterministic are appended here rather than in the shared
    # builder because train_lepton.py does not yet expose them; unifying that is a
    # follow-up.)
    train_cmd = (
        f"{python_path} -u protomotions/train_agent.py "
        f"{build_train_agent_args(args, passthrough_args)} "
        f"--use-slurm --seed={args.seed} "
    )
    if args.torch_deterministic:
        train_cmd += "--torch-deterministic "

    # Stall watchdog. It catches ONE failure mode: a WHOLE-JOB wedge, where a rank
    # gets stuck (e.g. never finishes setup) and every other rank then blocks at
    # the next collective waiting for it -- so the whole job goes silent and the
    # SLURM log stops growing, with no exit code, until the walltime reclaims it.
    # Because a blocked collective silences the healthy ranks too, "the log stopped
    # growing" is a reliable signal for THAT case. It is deliberately NOT a
    # per-rank health check: a single rank hung while the others keep doing useful,
    # log-producing work would not be caught, and any rank's output resets the
    # observed silence. That case is rarer (collectives usually couple the ranks)
    # and out of scope here. Kill after STALL_TIMEOUT_MIN minutes of no growth; the
    # failed array element then restarts on the next (possibly different) node. 0
    # disables. kill_tree + run_with_stall_watchdog are sourced from the shared
    # retry_lib.sh (after the `cd {exp_folder}` above put the snapshot on the path)
    # so this and lepton_train_loop.sh run the same watchdog. `source PATH` adds no
    # single quotes, so it is safe inside the single-quoted `bash -c '...'` srun
    # wraps job_cmd in.
    log_file = f"{exp_folder}/slurm_output.log"
    job_cmd += (
        f"export PYTHONUNBUFFERED=1; LOGFILE={log_file}; "
        f"STALL_TIMEOUT_MIN={args.stall_timeout_min}; {_retry_lib_source(exp_folder)}; "
        f"run_with_stall_watchdog {train_cmd}"
    )

    return job_cmd


def generate_slurm_script(args, exp_folder, job_cmd, container_image, job_name):
    """Generate SLURM batch script content."""
    log_file = f"{exp_folder}/slurm_output.log"

    # The IsaacLab container ships a uv-managed interpreter under /root/.local, but
    # pyxis mounts the host $HOME over the container's /root and hides it, which breaks
    # `isaaclab.sh -p` (its venv python then dangles). --no-container-mount-home keeps
    # the image's interpreter intact; mount ~/.netrc explicitly so W&B still
    # authenticates, since disabling the home mount also hides it.
    container_mounts = CONTAINER_MOUNTS
    if args.use_wandb:
        container_mounts += ",${HOME}/.netrc:/root/.netrc:ro"

    # Container run command (adjust for your cluster's container runtime).
    # stdbuf -oL line-buffers srun's output so slurm_output.log grows promptly --
    # the stall watchdog in job_cmd keys off that file's growth.
    # Any single quote inside job_cmd (e.g. a shlex-quoted passthrough value with a
    # space) would otherwise close this outer single-quoted string early and word-
    # split the rest. Escape them with the POSIX '\'' idiom so the wrapping is safe.
    job_cmd_quoted = job_cmd.replace("'", "'\\''")
    srun_cmd = (
        f"stdbuf -oL -eL srun "
        f"--container-image={container_image} "
        f"--no-container-mount-home "
        f"--container-mounts={container_mounts} "
        f"/bin/bash -c '{job_cmd_quoted}'"
    )

    # Opt-in startup retry. When off (the default), the body is the bare srun_cmd.
    # When on, source the shared retry_lib.sh at the batch layer and wrap srun in
    # run_with_retry, which re-runs a scene build that failed before training and
    # fast-fails a repeated Python exception. RETRY_ON_PROGRESS=break: the moment a
    # checkpoint is written (training reached), stop retrying in place -- the SLURM
    # array's next task resumes from it. No GAVE_UP sentinel here: a deterministic
    # scene-build bug fast-fails per array task (each is quick), and the array is the
    # coarse bound. srun_cmd is passed as ARGUMENTS (not re-quoted): it already
    # contains a single-quoted bash -c '...' and must not be escaped again.
    retry_on = args.startup_max_attempts > 1
    if retry_on:
        ckpt = f"{exp_folder}/results/{args.experiment_name}/last.ckpt"
        body = (
            f"{_retry_lib_source(exp_folder)}\n"
            f"export LOGFILE={log_file} "
            f"CKPT={ckpt} "
            f"RETRY_MAX_ATTEMPTS={args.startup_max_attempts} "
            f"RETRY_REPEAT_LIMIT={args.startup_repeat_limit} "
            f"RETRY_BACKOFF_SEC={args.startup_retry_backoff_sec} "
            f"RETRY_ON_PROGRESS=break\n"
            f"run_with_retry {srun_cmd}"
        )
    else:
        body = srun_cmd

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={args.account}
#SBATCH --partition={args.partition}
#SBATCH --nodes={args.nodes}
#SBATCH --gpus-per-node={args.ngpu}
#SBATCH --ntasks-per-node={args.ngpu}
#SBATCH --time={args.slurm_time}
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --array=0-{args.array_size}%1

# Job array enables automatic resume: if job times out, next array task continues

{body}
"""
    return script, log_file


def print_site(site):
    print(f"site file: {site['_source'] or '(none found -- using shipped defaults)'}")
    available = site.get("_clusters_available") or []
    if available:
        print(f"cluster: {site.get('_cluster')}  (available: {', '.join(available)})")
    for key in sorted(k for k in site if not k.startswith("_")):
        print(f"  {key}: {site[key]}")
    missing = unconfigured_values(site)
    print(f"  unset: {', '.join(missing) if missing else '(none)'}")


def main():
    # --print-site is a diagnostic and must not require a full training command
    # line, so it is handled before the real parser demands --robot-name and the
    # rest. This is the first thing to run when a submission goes somewhere
    # unexpected.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--site-config", default=None)
    pre.add_argument("--cluster", default=None)
    pre.add_argument("--print-site", action="store_true")
    pre_args, _ = pre.parse_known_args()
    if pre_args.print_site:
        print_site(load_site_config(pre_args.site_config, pre_args.cluster))
        return

    parser = create_parser()
    # parse_known_args, not parse_args: experiment files contribute their own CLI
    # flags via additional_experiment_arguments (e.g. gpc/prior.py's required
    # --tracker-checkpoint), which train_agent.py parses but this launcher does
    # not. Rejecting unknown flags made every such run unsubmittable; instead
    # forward the unknowns to train_agent.py (see build_job_command).
    args, passthrough_args = parser.parse_known_args()

    # Resolve the cluster before anything else, so a misconfiguration is caught
    # before code is synced rather than after.
    site = load_site_config(args.site_config, args.cluster)
    apply_site(site)
    args.account = args.account or site["account"]
    args.partition = args.partition or site["partition"]
    # A value given on the CLI is authoritative, so reflect it back before the
    # placeholder check: a site file that leaves the (now shared) account unset
    # must not be refused when --account supplied a real one.
    site["account"] = args.account
    site["partition"] = args.partition
    # array_size default comes from the site file (a team with long jobs sets it
    # once) when neither --array-size nor its --srun-repeat alias was given.
    if args.array_size is None:
        args.array_size = site.get("array_size", 5)
    # The startup-retry knobs resolve the same way: a flag left None takes the site
    # value (which defaults to the off value in SITE_DEFAULTS), and an explicit flag
    # still wins.
    for _key, _default in (
        ("startup_max_attempts", 1),
        ("startup_repeat_limit", 3),
        ("startup_retry_backoff_sec", 15),
    ):
        if getattr(args, _key) is None:
            setattr(args, _key, site.get(_key, _default))

    if missing := unconfigured_values(site):
        cluster_line = ""
        if site.get("_clusters_available"):
            cluster_line = (
                f"Cluster:  {site.get('_cluster')} "
                f"(available: {', '.join(site['_clusters_available'])})\n"
            )
        raise SystemExit(
            "Refusing to submit: these cluster settings are still the shipped "
            f"placeholders -- {', '.join(missing)}.\n\n"
            f"Searched: --site-config, ${SITE_CONFIG_ENV}, "
            f"{_repo_root() / SITE_CONFIG_DEFAULT_RELPATH}\n"
            f"Loaded:   {site['_source'] or '(nothing)'}\n"
            f"{cluster_line}\n"
            "Write a site YAML with the keys you need and pass it with "
            "--site-config, or place it at the path above. `--print-site` shows "
            "what was resolved."
        )

    # Startup retry multiplies with the job array. Print the worst-case execution
    # count before anything is submitted.
    if args.startup_max_attempts > 1:
        worst = args.startup_max_attempts * (args.array_size + 1)
        print(
            f"[startup-retry] up to {args.startup_max_attempts} attempts x "
            f"{args.array_size + 1} array elements = {worst} executions worst case"
        )

    # Setup paths
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # --repo-root so this can sync a checkout other than its own; without it the
    # only way to launch a different tree was to copy the script and edit it.
    local_repo = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    # base_dir may be templated on the account, which varies per allocation.
    exp_folder = os.path.join(
        CLUSTER_BASE_DIR.format(account=args.account), args.user, f"exp-{timestamp}"
    )

    # Resolve the container up front. A cluster may stage only some simulators
    # (its container_images replace rather than merge), so an unsupported
    # --simulator must fail here, before the ssh/rsync below runs.
    container_image = CONTAINER_IMAGES.get(args.simulator)
    python_path = PYTHON_EXECUTABLES.get(args.simulator, "python")
    if not container_image:
        raise ValueError(
            f"No container configured for simulator {args.simulator!r} on cluster "
            f"{site.get('_cluster') or '(default)'}; configured: {sorted(CONTAINER_IMAGES)}"
        )

    print(f"Local repository: {local_repo}")
    print(f"Remote experiment folder: {exp_folder}")

    # Sync code
    sync_code_to_cluster(args.user, exp_folder, local_repo, site.get("rsync_excludes"))

    if args.only_upload_code:
        print("Code uploaded. Exiting (--only-upload-code).")
        return

    # Build job command
    job_cmd = build_job_command(args, exp_folder, python_path, passthrough_args)

    # SLURM job name carries the submit TIME so a squeue row maps to its exp-<ts>
    # directory and its output log -- otherwise squeue tells you nothing about
    # where a job's output lives.
    #
    # Time only, not the full timestamp: squeue's NAME column is 8 characters, so
    # a leading date would fill it entirely and every job submitted on the same
    # day would render identically. The date is recoverable from the job's submit
    # date, and the exp-<slurm_job_id> symlink created after submission below
    # gives the exact directory anyway.
    job_name = f"{timestamp.split('_')[1]}_{args.experiment_name}"
    slurm_script, log_file = generate_slurm_script(
        args, exp_folder, job_cmd, container_image, job_name
    )

    print("\n" + "=" * 60)
    print("SLURM SCRIPT:")
    print("=" * 60)
    print(slurm_script)

    # Write the sbatch script to a temp file, not the cwd: a relative tmp/ pollutes
    # whatever directory you launched from and fails outright when the cwd is not
    # writable (the normal case in an agent sandbox).
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=f"slurm_{timestamp}_", suffix=".sh", delete=False
    ) as handle:
        handle.write(slurm_script)
        local_script = handle.name

    remote_script = f"{exp_folder}/submit.sh"
    subprocess_run(
        f"scp {local_script} {args.user}@{CLUSTER_LOGIN_NODE}:{remote_script}",
        shell=True,
    )

    # Submit and capture the job id from "Submitted batch job N".
    submit_cmd = f"ssh {args.user}@{CLUSTER_LOGIN_NODE} 'chmod +x {remote_script}; sbatch {remote_script}'"
    print(f"\nSubmitting: {submit_cmd}")
    result = subprocess_run(submit_cmd, shell=True, capture_output=True, text=True)
    sbatch_out = (result.stdout or "").strip()
    if sbatch_out:
        print(sbatch_out)
    slurm_job_id = "unknown"
    for line in sbatch_out.splitlines():
        if "Submitted batch job" in line:
            slurm_job_id = line.split()[-1].strip()
            break

    # Make the exp directory addressable by SLURM job id, so squeue's JOBID column
    # leads straight to a path. For an array job sbatch returns the base id, which
    # is the part squeue prints before the "_<task>" suffix, so exp-<base id> is
    # what you can actually copy off a squeue row.
    #
    # A symlink rather than a rename: the sbatch script, --output and --error all
    # have the exp-<ts> path baked in already, and the job may be pending or
    # running by now. Failure here is not worth aborting a submitted job over, so
    # it is best-effort and only reported.
    exp_link = None
    if slurm_job_id != "unknown":
        candidate = os.path.join(os.path.dirname(exp_folder), f"exp-{slurm_job_id}")
        link_result = subprocess_run(
            f"ssh {args.user}@{CLUSTER_LOGIN_NODE} 'ln -sfn {exp_folder} {candidate}'",
            shell=True,
            capture_output=True,
            text=True,
            ignore_err=True,
        )
        if link_result.returncode == 0:
            exp_link = candidate
        else:
            print(
                f"warning: could not link exp-{slurm_job_id} -> {exp_folder} "
                f"({(link_result.stderr or '').strip()})"
            )

    # Copy-back hint: auto-discover the helper wherever it lives (it has moved
    # between directories). rglob avoids naming any specific path this file must
    # not reference; first non-worktree match wins.
    copy_back = next(
        (
            p
            for p in sorted(local_repo.rglob("copy_back_from_slurm*.py"))
            if ".worktrees" not in p.parts and ".git" not in p.parts
        ),
        None,
    )
    copy_back_cmd = ""
    if copy_back is not None:
        rel = copy_back.relative_to(local_repo)
        copy_back_cmd = (
            f"python3 {rel} --exp-id {timestamp} --account {args.account} "
            f"--user {args.user} --download-last"
        )

    print("\n" + "=" * 60)
    print("JOB SUBMITTED!")
    print("=" * 60)
    print(f"Monitor logs:  ssh {args.user}@{CLUSTER_LOGIN_NODE} 'tail -f {log_file}'")
    print(
        f"Check status:  ssh {args.user}@{CLUSTER_LOGIN_NODE} 'squeue -u {args.user}'"
    )
    print(
        f"Cancel job:    ssh {args.user}@{CLUSTER_LOGIN_NODE} 'scancel {slurm_job_id}'"
    )
    if exp_link:
        print(f"Exp by job id: {exp_link}")
    if copy_back_cmd:
        print(f"Copy back:     {copy_back_cmd}")
    print("=" * 60)

    # Machine-readable block for automated parsing (agents, scripts parse this).
    print("")
    print("=== SUBMISSION SUMMARY ===")
    print(f"slurm_job_id={slurm_job_id}")
    print(f"exp_id={timestamp}")
    print(f"exp_dir={exp_folder}")
    if exp_link:
        print(f"exp_link={exp_link}")
    print(f"output_log={log_file}")
    if copy_back_cmd:
        print(f"copy_back={copy_back_cmd}")
    print("=== END SUMMARY ===")


if __name__ == "__main__":
    main()
