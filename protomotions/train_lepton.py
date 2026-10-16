# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lepton (DGX Cloud) Training Launcher for ProtoMotions
=====================================================

Submits ProtoMotions training to NVIDIA DGX Cloud Lepton. This is the Lepton
counterpart to `train_slurm.py`; the two share only the inner train_agent.py
command line (see `_train_launch_common.py`) because the systems differ deeply:

  SLURM (train_slurm.py)          Lepton (this file)
  ---------------------          ------------------
  sbatch + srun                  `lep job create`
  job array + 4h walltime        no walltime; one job runs to completion
  --use-slurm autoresume @3.5h   no autoresume; filesystem-based resume only
  enroot .sqsh via srun          registry image pulled by the platform
  rsync to a login node          rsync to a dev pod over managed Teleport SSH

HOW IT WORKS
------------
1. rsync the repo to an immutable snapshot on AMLFS, through the dev pod's
   Teleport SSH host (`root@<workspace-id>-<pod>.<teleport-cluster>`). Native
   ssh/rsync tunnel through Teleport once `tsh login` + `tsh config` are done.
2. `lep job create` a single job that mounts AMLFS at /lustre and runs
   `scripts/cluster/lepton_train_loop.sh`, which installs the snapshot, warms
   IsaacLab, and runs train_agent.py in a crash-restart loop. Resume needs no
   flags: train_agent.py finds results/<exp>/last.ckpt in the snapshot and
   continues (it saves last.ckpt every K epochs regardless of SLURM).

BEFORE USING
------------
Write a `lepton_site.yaml` describing your workspace (keys in SITE_DEFAULTS
below), or pass --site-config PATH. Confirm it with --print-site. Prerequisites
(one-time): `lep login`, a Teleport-enabled dev pod reachable as the host above
(`tsh login` + `tsh config`), a registry-auth secret for the private image, and
(for W&B) a WANDB_API_KEY workspace secret.

    python protomotions/train_lepton.py \\
        --robot-name=g1 --simulator=isaaclab \\
        --num-envs=4096 --batch-size=32768 \\
        --motion-file=data/motion_for_trackers/g1_bones_seed_mini.pt \\
        --experiment-path=examples/experiments/mimic/mlp_bm_l2c2.py \\
        --experiment-name=my_run
        # dry-run by default; add --execute to actually stage + submit

KNOWN LIMITATIONS vs the SLURM path (train_slurm.py)
----------------------------------------------------
- The shipped example config has an IsaacLab image placeholder. Other simulators
  require a compatible registry image and Python command in `lepton_site.yaml`.
- Driving this launcher (rsync + lep) needs a live managed-Teleport session. It
  expires roughly every 12h and re-login (`tsh login`) requires an interactive
  browser SSO, so long unattended automation must refresh it.
- Multi-node (--nodes > 1) is refused until rendezvous, shared-snapshot resume,
  and watchdog behavior are qualified together. Use single-node multi-GPU.
"""

import argparse
import datetime
import getpass
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

# Make `import protomotions...` resolve when this is run as
# `python protomotions/train_lepton.py` -- then the script's own directory, not the
# repo root, is on sys.path -- without requiring PYTHONPATH/direnv or an editable
# install. Same standalone invocation train_slurm.py supports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protomotions._train_launch_common import (  # noqa: E402
    RSYNC_EXCLUDES,
    build_train_agent_argv,
)


# =============================================================================
# WORKSPACE CONFIGURATION
# =============================================================================
# Shipped defaults are deliberately unusable placeholders, so a mis-configured
# submission fails loudly instead of going somewhere unexpected. Point the script
# at your workspace with a `lepton_site.yaml` rather than editing this block --
# editing here makes every `git pull` a merge conflict, and the file is excluded
# from the release projection so it never ships.
SITE_DEFAULTS = {
    # Lepton workspace id (short code, e.g. from `lep workspace id`). Used to
    # build the Teleport SSH host <workspace_id>-<pod_name>.<teleport_cluster>.
    "workspace_id": "YOUR_WORKSPACE_ID",
    # Lepton workspace name -- the AMLFS mount source is /<workspace_name>.
    "workspace_name": "YOUR_WORKSPACE_NAME",
    # Node group that carries the GPU shapes (from `lep node resource-shape`).
    "node_group": "YOUR_NODE_GROUP",
    # node-nfs storage name backing AMLFS (from `lep node storage`).
    "storage_name": "YOUR_STORAGE_NAME",
    # Per-user snapshot base on the mounted cluster filesystem; "{user}" is
    # substituted with --user. Snapshots become <base_dir>/exp-<timestamp>.
    # Lives here (not hardcoded) so the shipped file carries no cluster path.
    "base_dir": "/YOUR_MOUNT/users/{user}/exp",
    # Managed Teleport cluster/proxy the dev pod is reachable through.
    "teleport_cluster": "YOUR_TELEPORT_CLUSTER",
    # Dev pod name used purely as the rsync gateway to AMLFS.
    "pod_name": "YOUR_POD_NAME",
    # Registry-auth object name for pulling the (private) container image;
    # passed to `lep job create --image-pull-secrets`. Not the NGC key itself.
    "registry_secret": "YOUR_REGISTRY_SECRET",
    # Container images keyed by simulator (registry refs, not .sqsh files).
    "container_images": {
        "isaaclab": "YOUR_ISAACLAB_IMAGE",
    },
    # Python invocation inside each container.
    "python_executables": {
        "isaacgym": "python",
        "isaaclab": "/workspace/isaaclab/isaaclab.sh -p",
        "newton": "python",
    },
}

# Anything still equal to a shipped placeholder is unconfigured; submitting with
# one is refused. Matching the literal (rather than a separate "configured?" flag)
# also catches a half-filled site file.
_PLACEHOLDER_MARKERS = ("YOUR_",)

SITE_CONFIG_ENV = "PROTOMOTIONS_LEPTON_SITE"
SITE_CONFIG_DEFAULT_RELPATH = Path("lepton_site.yaml")

# gpu.<N>xa100-80gb resource shape per GPUs-on-a-node. Full-node 8x preferred for
# multi-node. Override with --resource-shape for other accelerators.
_A100_SHAPES = {
    1: "gpu.a100-80gb",
    2: "gpu.2xa100-80gb",
    4: "gpu.4xa100-80gb",
    8: "gpu.8xa100-80gb",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_site_config(explicit_path=None):
    """Return SITE_DEFAULTS overlaid with the first site file found.

    Search order (first match wins): --site-config, $PROTOMOTIONS_LEPTON_SITE,
    then lepton_site.yaml beside the repository. The file is flat (no multi-cluster
    layer -- a Lepton workspace is a single target).
    """
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
        unknown = set(loaded) - allowed
        if unknown:
            raise ValueError(
                f"{candidate}: unknown key(s) {sorted(unknown)}; expected any of "
                f"{sorted(allowed)}"
            )
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(site.get(key), dict):
                site[key].update(value)
            else:
                site[key] = value
        site["_source"] = str(candidate)
        return site

    site["_source"] = None
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


def create_parser():
    parser = argparse.ArgumentParser(
        description="Launch ProtoMotions training on DGX Cloud Lepton",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- training arguments (same names as train_slurm.py) ---
    parser.add_argument(
        "--robot-name", type=str, required=True, help="Robot name (e.g. 'g1')"
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
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    training_limit_group = parser.add_mutually_exclusive_group()
    training_limit_group.add_argument(
        "--training-max-steps", type=int, default=10000000000, help="Max training steps"
    )
    training_limit_group.add_argument(
        "--training-max-iterations",
        type=int,
        default=None,
        help="Max complete training iterations",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Warm-start from checkpoint"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="physical_animation",
        help="W&B project name",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="GPUs per node")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--overrides", nargs="*", default=[], help="Config overrides (key=value)"
    )

    # --- Lepton / workspace arguments (CLI wins over the site file) ---
    parser.add_argument(
        "--user",
        type=str,
        default=getpass.getuser(),
        help="Cluster username; names your per-user cluster dirs AND the W&B secret owner "
        "(WANDB_API_KEY.<user>). Defaults to your local login -- pass it explicitly "
        "if your Lepton username differs.",
    )
    parser.add_argument(
        "--site-config",
        default=None,
        help="Lepton site YAML; overrides $PROTOMOTIONS_LEPTON_SITE and lepton_site.yaml",
    )
    parser.add_argument(
        "--workspace-id", default=None, help="Override site workspace_id"
    )
    parser.add_argument(
        "--workspace-name", default=None, help="Override site workspace_name"
    )
    parser.add_argument("--node-group", default=None, help="Override site node_group")
    parser.add_argument(
        "--storage-name", default=None, help="Override site storage_name"
    )
    parser.add_argument(
        "--pod-name", default=None, help="Override site pod_name (rsync gateway)"
    )
    parser.add_argument(
        "--registry-secret", default=None, help="Override site registry_secret"
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Override the site container image for this simulator",
    )
    parser.add_argument(
        "--resource-shape",
        default=None,
        help="Lepton resource shape (default: derived from --ngpu)",
    )
    parser.add_argument(
        "--max-runs", type=int, default=75, help="Crash-restart cap for the in-job loop"
    )
    parser.add_argument(
        "--no-progress-limit",
        type=int,
        default=3,
        help="Give up after this many consecutive in-job restarts that do not "
        "advance the checkpoint (a deterministic failure). Raise it for a flaky "
        "scene build that aborts several times before epoch 0; the default 3 kills "
        "a Lepton job after three ordinary aborts.",
    )
    parser.add_argument(
        "--repeat-limit",
        type=int,
        default=3,
        help="Give up sooner if the SAME Python exception repeats this many "
        "restarts in a row -- a deterministic bug, not a flaky abort. 0 disables "
        "the check (fall back to --no-progress-limit alone). Default 3.",
    )
    parser.add_argument(
        "--stall-timeout-min",
        type=int,
        default=30,
        help="Kill training if its log has not grown for this many minutes -- a "
        "wedged rank makes the whole job go silent at the distributed barrier -- so "
        "the loop restarts instead of burning the walltime. 0 disables; raise it for "
        "jobs with very long epochs.",
    )
    parser.add_argument(
        "--preemptible",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the job to be evicted (burst capacity beyond guaranteed quota). "
        "An evicted job is relaunched by Lepton up to --max-job-retries, and the "
        "relaunch resumes from last.ckpt. Default off (guaranteed, not evicted).",
    )
    parser.add_argument(
        "--max-job-retries",
        type=int,
        default=10,
        help="With --preemptible, how many times Lepton relaunches the whole job "
        "after eviction/node failure (each relaunch resumes from last.ckpt).",
    )
    parser.add_argument(
        "--shared-memory-mib", type=int, default=65536, help="Job /dev/shm size (MiB)"
    )
    parser.add_argument(
        "--queue-priority", type=int, default=4, help="Lepton queue priority (1-9)"
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=259200,
        help="Keep the finished job this long",
    )
    parser.add_argument(
        "--visibility",
        choices=["public", "private"],
        default="private",
        help="Job visibility",
    )

    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository to stage (default: the one this script lives in)",
    )
    parser.add_argument(
        "--exp-folder",
        default=None,
        help="Reuse an existing AMLFS snapshot path and skip staging -- e.g. to "
        "resume a run from its last.ckpt, or relaunch without re-rsyncing. Default: "
        "a fresh exp-<timestamp> snapshot under the site's base_dir.",
    )
    parser.add_argument(
        "--print-site",
        action="store_true",
        help="Print the resolved workspace settings and exit",
    )
    parser.add_argument(
        "--only-upload-code", action="store_true", help="Stage code but do not submit"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually stage + submit (default is a dry run)",
    )

    return parser


def teleport_host(site, user):
    """root@<workspace-id>-<pod>.<teleport-cluster>: the dev pod over managed Teleport."""
    return f"root@{site['workspace_id']}-{site['pod_name']}.{site['teleport_cluster']}"




def stage_code(host, exp_folder, local_repo, execute):
    """rsync the repo to an immutable snapshot on AMLFS through the Teleport host."""
    # Create the snapshot dir and ensure rsync exists on the pod: the dev pod image
    # ships without it (and its container fs is ephemeral, so a prior install may be
    # gone), which otherwise fails as "rsync: command not found" / protocol mismatch.
    # Pods run as root, so apt-get works.
    remote_prep = (
        f"mkdir -p {exp_folder}; "
        "command -v rsync >/dev/null 2>&1 || "
        "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y -qq rsync >/dev/null)"
    )
    prep_cmd = ["ssh", host, remote_prep]
    rsync_cmd = (
        ["rsync", "-az", "--partial", "-m", "--chmod=775"]
        + [f"--exclude={p}" for p in RSYNC_EXCLUDES]
        + [f"{local_repo}/", f"{host}:{exp_folder}/"]
    )
    print("Staging code:")
    print("  " + " ".join(prep_cmd))
    print("  " + " ".join(rsync_cmd))
    if not execute:
        return
    subprocess.run(prep_cmd, check=True)
    subprocess.run(rsync_cmd, check=True)


def build_container_command(python_path, args, exp_folder, passthrough=None):
    """The bash the job runs: cd into the snapshot and hand off to the resume loop.

    Every positional handed to lepton_train_loop.sh is shlex-quoted individually,
    so the container shell splits this command back into exactly these arguments --
    no value is word-split on a space it happens to contain, and no quoting is lost
    to a second re-split (the loop forwards the train_agent.py tail with `"$@"`).
    The interpreter positional may itself be a command+flag (e.g. "isaaclab.sh -p")
    -- it stays ONE positional here, which the loop word-splits deliberately at use.
    """
    positionals = [
        python_path,
        args.simulator,
        args.experiment_name,
        exp_folder,
        str(args.max_runs),
        *build_train_agent_argv(args, passthrough),
    ]
    quoted = " ".join(shlex.quote(p) for p in positionals)
    return (
        f"set -euo pipefail; cd {shlex.quote(exp_folder)}; "
        f"bash scripts/cluster/lepton_train_loop.sh {quoted}"
    )


def _lepton_job_name(experiment_name):
    """Map an experiment name to a valid Lepton job name.

    Lepton job names must be RFC-1123 labels: lowercase alphanumeric and '-',
    starting and ending alphanumeric. An experiment name may contain '_' or other
    characters (e.g. 'lepton_smoke'), so map runs of invalid characters to '-'.
    Lepton appends its own short suffix, so cap the length. The experiment name
    itself is used unchanged for --experiment-name / results/<exp>.
    """
    name = re.sub(r"[^a-z0-9-]+", "-", experiment_name.lower()).strip("-")
    name = name[:36].strip("-")
    return name or "job"


def build_lep_job_create(args, site, exp_folder, container_command):
    image = args.image or site["container_images"].get(args.simulator)
    if not image:
        raise SystemExit(
            f"No container image configured for simulator {args.simulator!r}; "
            f"configured: {sorted(site['container_images'])}. Pass --image or add it "
            f"to the site file."
        )
    resource_shape = args.resource_shape or _A100_SHAPES.get(args.ngpu)
    if not resource_shape:
        raise SystemExit(
            f"No default resource shape for --ngpu {args.ngpu}; pass --resource-shape."
        )
    mount_spec = f"/{site['workspace_name']}:/lustre:node-nfs:{site['storage_name']}"

    cmd = [
        "lep",
        "job",
        "create",
        "--name",
        _lepton_job_name(args.experiment_name),
        "--resource-shape",
        resource_shape,
        "--node-group",
        args.node_group or site["node_group"],
        "--num-workers",
        str(args.nodes),
        "--container-image",
        image,
        "--image-pull-secrets",
        args.registry_secret or site["registry_secret"],
        "--mount",
        mount_spec,
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        f"STALL_TIMEOUT_MIN={args.stall_timeout_min}",
        "--env",
        f"RETRY_MAX_ATTEMPTS={args.no_progress_limit}",
        "--env",
        f"RETRY_REPEAT_LIMIT={args.repeat_limit}",
        "--shared-memory-size",
        str(args.shared_memory_mib),
        # The in-job loop handles crashes within a running job. Job-level retry
        # relaunches the WHOLE job after eviction/node failure; keep it at 0 unless
        # the job is preemptible (then Lepton can evict it and it must come back,
        # resuming from last.ckpt). The loop's give-up sentinel stops a
        # deterministic failure from churning the whole retry budget.
        "--max-failure-retry",
        "0",
        "--max-job-failure-retry",
        str(args.max_job_retries if args.preemptible else 0),
        "--ttl-seconds-after-finished",
        str(args.ttl_seconds),
        "--log-collection",
        "true",
        "--queue-priority",
        str(args.queue_priority),
        "--visibility",
        args.visibility,
    ]
    if args.preemptible:
        cmd += ["--can-be-preempted"]
    if args.use_wandb:
        # Containers have no home mount, so W&B reads the key from an env var
        # sourced from the workspace secret. Lepton suffixes private secrets with
        # the owner name (WANDB_API_KEY.<user>); expose it as WANDB_API_KEY.
        cmd += ["--secret", f"WANDB_API_KEY=WANDB_API_KEY.{args.user}"]
    cmd += ["--command", container_command]
    return cmd


def reject_unqualified_multinode(nodes):
    """Refuse --nodes > 1 (rather than warn) until multi-node is qualified.

    Multi-node rendezvous and the restart/watchdog behavior for multiple workers
    sharing one snapshot and durable log have not been qualified together.
    Single-node multi-GPU is the supported path. Kept as a function so a test can
    assert the refusal without driving the whole submit.
    """
    if nodes > 1:
        raise SystemExit(
            f"Refusing to submit a {nodes}-node job: multi-node on Lepton is not "
            "qualified. Rendezvous and restart/watchdog behavior must be validated "
            "for multiple workers sharing one snapshot and log. Use single-node "
            "multi-GPU instead."
        )


def print_site(site):
    print(f"site file: {site['_source'] or '(none found -- using shipped defaults)'}")
    for key in sorted(k for k in site if not k.startswith("_")):
        print(f"  {key}: {site[key]}")
    missing = unconfigured_values(site)
    print(f"  unset: {', '.join(missing) if missing else '(none)'}")


def main():
    # --print-site is a diagnostic and must not require a full training command.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--site-config", default=None)
    pre.add_argument("--print-site", action="store_true")
    pre_args, _ = pre.parse_known_args()
    if pre_args.print_site:
        print_site(load_site_config(pre_args.site_config))
        return

    # parse_known_args, not parse_args: experiment files contribute their own CLI
    # flags via additional_experiment_arguments (e.g. gpc/prior.py's required
    # --tracker-checkpoint), which train_agent.py parses but this launcher does
    # not. Forward the unknowns to train_agent.py rather than rejecting them.
    args, passthrough_args = create_parser().parse_known_args()
    site = load_site_config(args.site_config)

    # CLI overrides win over the site file; reflect them back before the placeholder
    # check so a value supplied on the command line satisfies it.
    for cli_key, site_key in [
        ("workspace_id", "workspace_id"),
        ("workspace_name", "workspace_name"),
        ("node_group", "node_group"),
        ("storage_name", "storage_name"),
        ("pod_name", "pod_name"),
        ("registry_secret", "registry_secret"),
    ]:
        val = getattr(args, cli_key)
        if val is not None:
            site[site_key] = val
    # The image lives under container_images[simulator]; reflect an --image
    # override there too, so it satisfies the placeholder check like the flat keys.
    if args.image is not None and isinstance(site.get("container_images"), dict):
        site["container_images"][args.simulator] = args.image
    # Only the active simulator's image matters; drop the rest so an unused
    # simulator's shipped placeholder (e.g. isaaclab's) does not trip the
    # placeholder check for a run targeting a different, fully-configured
    # simulator. A missing active image then fails cleanly in build_lep_job_create.
    if isinstance(site.get("container_images"), dict):
        active_image = site["container_images"].get(args.simulator)
        site["container_images"] = (
            {args.simulator: active_image} if active_image is not None else {}
        )

    if missing := unconfigured_values(site):
        raise SystemExit(
            "Refusing to submit: these workspace settings are still the shipped "
            f"placeholders -- {', '.join(missing)}.\n\n"
            f"Searched: --site-config, ${SITE_CONFIG_ENV}, "
            f"{_repo_root() / SITE_CONFIG_DEFAULT_RELPATH}\n"
            f"Loaded:   {site['_source'] or '(nothing)'}\n\n"
            "Write a lepton_site.yaml with the keys you need, or pass them on the "
            "command line. `--print-site` shows what was resolved."
        )

    reject_unqualified_multinode(args.nodes)

    # A relative --checkpoint resolves against the job's snapshot cwd, and files
    # under rsync-excluded dirs (e.g. data/pretrained_models) are not staged -- so a
    # relative warm-start path silently 404s in the job. Point at a pre-staged
    # absolute /lustre path instead.
    if args.checkpoint and not os.path.isabs(args.checkpoint):
        print(
            f"WARNING: --checkpoint {args.checkpoint!r} is a relative path. Warm-start "
            "checkpoints must be pre-staged on AMLFS and passed as an absolute /lustre "
            "path; a relative path under an rsync-excluded dir will not exist in the job."
        )

    python_path = site["python_executables"].get(args.simulator, "python")
    local_repo = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    host = teleport_host(site, args.user)

    # --exp-folder reuses an existing AMLFS snapshot (skip staging) -- e.g. to resume
    # a run from its results/<exp>/last.ckpt. Otherwise stage the repo into a fresh
    # immutable snapshot.
    reuse = args.exp_folder is not None
    if reuse:
        exp_folder = args.exp_folder
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # .replace, not .format: base_dir may legitimately contain other braces;
        # only {user} is substituted, and an unknown field must not raise KeyError.
        exp_folder = f"{site['base_dir'].replace('{user}', args.user)}/exp-{timestamp}"

    print(f"Local repository: {local_repo}")
    print(f"Teleport host:    {host}")
    print(
        f"Snapshot (AMLFS): {exp_folder}{'  (reused, staging skipped)' if reuse else ''}"
    )

    if reuse:
        print("Reusing existing snapshot; skipping code staging.")
        # A give-up sentinel from a prior attempt now makes the in-job loop
        # fast-FAIL (see lepton_train_loop.sh), so on this deliberate resume we
        # must clear it. If we cannot reach the pod to do so, refuse to submit
        # rather than continue: a stale sentinel would make the resumed job exit
        # nonzero immediately without training, which reads as a mysterious
        # instant failure. `rm -f` still exits 0 when the sentinel is absent, so a
        # nonzero here is a real connection failure worth stopping on.
        if args.execute:
            clear = subprocess.run(
                ["ssh", host, f"rm -f {shlex.quote(exp_folder)}/.lepton_loop_gave_up"]
            )
            if clear.returncode != 0:
                raise SystemExit(
                    f"Could not clear the give-up sentinel on {host} (ssh exit "
                    f"{clear.returncode}). A stale .lepton_loop_gave_up would make the "
                    "resumed job fail immediately without training. Fix the connection "
                    "(tsh login may have expired) and retry."
                )
    else:
        stage_code(host, exp_folder, local_repo, args.execute)
    if args.only_upload_code:
        print("Code staged. Exiting (--only-upload-code).")
        return

    container_command = build_container_command(
        python_path, args, exp_folder, passthrough_args
    )
    lep_cmd = build_lep_job_create(args, site, exp_folder, container_command)

    print("\n" + "=" * 60)
    print("lep job create (argv passed directly; shown quoted for copy/paste):")
    print("=" * 60)
    print(shlex.join(lep_cmd))
    print("=" * 60)

    if not args.execute:
        print("\nDry run. Re-run with --execute to stage the code and submit.")
        return

    result = subprocess.run(lep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"lep job create failed (exit {result.returncode})")

    print("\n" + "=" * 60)
    print("JOB SUBMITTED!")
    print("=" * 60)
    print(f"Status:  lep job get -n {_lepton_job_name(args.experiment_name)}")
    print("Events:  lep job get -i <id> ; lep job events -i <id>")
    print("Logs:    lep log get -j <id> --limit 5000")
    print(f"File log: ssh {host} 'tail -f {exp_folder}/lepton_train.log'")
    print("=" * 60)


if __name__ == "__main__":
    main()
