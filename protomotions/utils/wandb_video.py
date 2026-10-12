"""Optional, rank-zero background rendering; no simulator imports in the trainer."""

import argparse
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)
DEFAULTS = {
    "wandb_video": False,
    "wandb_video_every": 200,
    "wandb_video_duration": 10.0,
    "wandb_video_fps": 30,
    "wandb_video_width": 1280,
    "wandb_video_height": 720,
    "wandb_video_motion_id": 0,
    "wandb_video_gpu": None,
    "wandb_video_timeout": 300.0,
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def add_video_arguments(parser):
    # SUPPRESS distinguishes an explicit CLI override from a saved resume value.
    parser.add_argument(
        "--wandb-video",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Upload best-policy videos (default: disabled)",
    )
    for key, default in DEFAULTS.items():
        if key == "wandb_video":
            continue
        kind = (
            positive_float
            if key.endswith(("duration", "timeout"))
            else nonnegative_int
            if key.endswith(("motion_id", "gpu"))
            else positive_int
        )
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=kind,
            default=argparse.SUPPRESS,
            help=f"Best-policy video setting (default: {default})",
        )


def explicit_video_options(args):
    return {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key)}


def resolve_video_options(args, explicit):
    for key, default in DEFAULTS.items():
        setattr(args, key, explicit.get(key, getattr(args, key, default)))
    if args.wandb_video:
        if not args.use_wandb:
            raise ValueError("--wandb-video requires --use-wandb")
        if args.simulator != "isaaclab":
            raise ValueError("--wandb-video currently supports only isaaclab")
        for key in (
            "wandb_video_every",
            "wandb_video_fps",
            "wandb_video_width",
            "wandb_video_height",
        ):
            positive_int(getattr(args, key))
        for key in ("wandb_video_duration", "wandb_video_timeout"):
            positive_float(getattr(args, key))
        nonnegative_int(args.wandb_video_motion_id)
        if args.wandb_video_gpu is not None:
            nonnegative_int(args.wandb_video_gpu)
        if args.wandb_video_width % 2 or args.wandb_video_height % 2:
            raise ValueError("Video width and height must be even for H.264")


def video_cli_options(args):
    result = []
    for key, value in explicit_video_options(args).items():
        if key == "wandb_video":
            result.append("--wandb-video" if value else "--no-wandb-video")
        elif value is not None:
            result.append(f"--{key.replace('_', '-')}={value}")
    return result


def worker_environment(environ, gpu):
    # A recorder must never inherit Fabric/SLURM/torchrun's distributed identity.
    exact = {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NODE_RANK",
        "LIGHTNING_NODE_RANK",
        "LT_CLI_USED",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
    }
    prefixes = (
        "SLURM_",
        "TORCHELASTIC_",
        "PMI_",
        "PMIX_",
        "OMPI_",
        "MV2_",
        "WANDB_",
        "NCCL_",
        "PROTOMOTIONS_PHYSICS_TRACE_",
    )
    env = {
        k: v
        for k, v in environ.items()
        if k not in exact and not k.startswith(prefixes)
    }
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    if devices is not None and gpu >= len(devices):
        raise ValueError(f"Video GPU {gpu} is outside CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = devices[gpu].strip() if devices else str(gpu)
    env.update(WANDB_MODE="disabled", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    return env


def select_video_motions(count, fixed, rng=None):
    """Sample two distinct extra motions without consuming training RNG state."""
    if not 0 <= fixed < count:
        raise ValueError(f"Video motion {fixed} is outside library of {count} motions")
    rng = rng or random.SystemRandom()
    # Map samples around the fixed ID without allocating a full library list.
    extra = rng.sample(range(count - 1), min(2, count - 1))
    return [fixed] + [i if i < fixed else i + 1 for i in extra]


class BestPolicyVideoRecorder:
    """One subprocess at a time. All failures stay outside training collectives."""

    def __init__(self, args, agent):
        self.options = {
            key: getattr(args, key, default) for key, default in DEFAULTS.items()
        }
        self.agent = agent
        self.root = Path(agent.root_dir).resolve()
        self.job = None
        self.last_trigger = None

    def _checkpoint_for_video(self):
        # SFT has no evaluation components, so it cannot produce score_based.ckpt.
        if getattr(self.agent.config, "_target_", "").endswith(".DiscretePriorPEFTSFTAgent"):
            return self.root / "inference_last.ckpt"
        return self.root / "score_based.ckpt"

    def tick(self, agent):
        if agent.fabric.global_rank != 0 or not self.options["wandb_video"]:
            return
        try:
            self._poll(agent.current_epoch)
            epoch = agent.current_epoch
            if (
                epoch <= 0
                or epoch % self.options["wandb_video_every"]
                or epoch == self.last_trigger
            ):
                return
            self.last_trigger = epoch
            if self.job is not None:
                log.warning(
                    "Best-policy video at epoch %s skipped: recorder busy", epoch
                )
                return
            if not self._checkpoint_for_video().is_file():
                log.info(
                    "Best-policy video at epoch %s skipped: no evaluated best checkpoint",
                    epoch,
                )
                return
            self._start(agent)
        except Exception:
            log.warning("Best-policy video failed; training continues", exc_info=True)
            self._discard_job()

    def _start(self, agent):
        import tempfile

        folder = self.root / "videos" / f"epoch_{agent.current_epoch:08d}"
        if folder.exists():
            folder = folder.with_name(f"{folder.name}_{time.time_ns()}")
        folder.mkdir(parents=True, exist_ok=False)
        scratch = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=folder))
        self.job = {
            "folder": folder,
            "scratch": scratch,
            "process": None,
            "started": time.monotonic(),
        }
        # Copy, never hard-link: training overwrites score_based.ckpt in place.
        checkpoint = self._checkpoint_for_video()
        shutil.copyfile(checkpoint, scratch / "score_based.ckpt")
        shutil.copyfile(
            self.root / "resolved_configs_inference.pt",
            scratch / "resolved_configs_inference.pt",
        )
        if checkpoint.name == "inference_last.ckpt":
            # Render SFT with the same future-reference target source as training.
            # Its default inference override expects keyboard input, unavailable headless.
            import copy
            import torch
            snapshot = scratch / "resolved_configs_inference.pt"
            configs = torch.load(snapshot, map_location="cpu", weights_only=False)
            configs["env"].control_components["target"] = copy.deepcopy(
                agent.env.config.control_components["target"]
            )
            torch.save(configs, snapshot)
            log.info("SFT video uses latest inference checkpoint and future-reference targets")
        motion_file = agent.env.motion_lib.motion_file
        if not motion_file or "slurmrank" in Path(motion_file).name:
            raise ValueError("Recorder requires rank 0's resolved motion file")
        scene_file = getattr(agent.env.scene_lib.config, "scene_file", None)
        request = {
            **self.options,
            "trigger_epoch": agent.current_epoch,
            "motion_file": str(Path(motion_file).resolve()),
            "scene_file": str(Path(scene_file).resolve()) if scene_file else None,
            "output": str(folder / "best_policy.mp4"),
        }
        motions = select_video_motions(
            agent.env.motion_lib.num_motions(), self.options["wandb_video_motion_id"]
        )
        if len(motions) < 3:
            log.warning("Only %s distinct video motions available", len(motions))
        self.job.update(
            batch_folder=folder,
            request=request,
            motions=motions,
            index=0,
        )
        self._launch_motion()

    def _launch_motion(self):
        job = self.job
        scratch = job["scratch"]
        index = job["index"]
        motion = job["motions"][index]
        folder = job["batch_folder"]
        if index:
            folder = folder / f"random_{index}_motion_{motion}"
            folder.mkdir()
        job["folder"] = folder
        job["started"] = time.monotonic()
        request = {
            **job["request"],
            "wandb_video_motion_id": motion,
            "output": str(folder / "best_policy.mp4"),
        }
        request_file = scratch / "request.json"
        request_file.write_text(json.dumps(request))
        gpu = self.options["wandb_video_gpu"]
        if gpu is None:
            gpu = self.agent.fabric.device.index or 0
        env = worker_environment(os.environ, gpu)
        command = [
            sys.executable,
            "-m",
            "protomotions.inference_agent",
            "--simulator",
            "isaaclab",
            "--checkpoint",
            str(scratch / "score_based.ckpt"),
            "--headless",
            "--num-envs",
            "1",
            "--video-request",
            str(request_file),
        ]
        with (folder / "render.log").open("w") as output:
            self.job["process"] = subprocess.Popen(
                command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        process = self.job["process"]

        def expire():
            if process.poll() is None:
                log.warning(
                    "Best-policy video exceeded timeout; terminating %s", folder
                )
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        timer = threading.Timer(self.options["wandb_video_timeout"], expire)
        timer.daemon = True
        self.job["timer"] = timer
        timer.start()
        log.info(
            "Rendering best-policy video for epoch %s in background",
            request["trigger_epoch"],
        )

    def _poll(self, current_epoch, advance=True):
        if self.job is None or self.job["process"] is None:
            return
        process = self.job["process"]
        status = process.poll()
        if status is None:
            if (
                time.monotonic() - self.job["started"]
                <= self.options["wandb_video_timeout"]
            ):
                return
            log.warning("Best-policy video timed out: %s", self.job["folder"])
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            status = -9
        self.job["timer"].cancel()
        self.job["timer"] = None
        self.job["process"] = None
        folder = self.job["folder"]
        try:
            if status != 0:
                raise RuntimeError(
                    f"Renderer exited {status}; see {folder / 'render.log'}"
                )
            self._upload(current_epoch)
        except Exception:
            log.warning(
                "Best-policy motion video failed; continuing batch", exc_info=True
            )
        finally:
            (folder / "best_policy.partial.mp4").unlink(missing_ok=True)
            if advance and self.job["index"] + 1 < len(self.job["motions"]):
                self.job["index"] += 1
                self.job["process"] = None
                self._launch_motion()
            else:
                self._discard_job()

    def _upload(self, current_epoch):
        folder = self.job["folder"]
        index = self.job["index"]
        key = (
            "videos/best_policy" if index == 0 else f"videos/best_policy_random_{index}"
        )
        metadata = json.loads((folder / "best_policy.json").read_text())
        video = folder / "best_policy.mp4"
        if not video.is_file() or not video.stat().st_size:
            raise RuntimeError("Renderer produced no video")
        import wandb
        from lightning.pytorch.loggers import WandbLogger

        logger = next(
            x for x in self.agent.fabric.loggers if isinstance(x, WandbLogger)
        )
        caption = (
            f"trigger epoch={metadata['trigger_epoch']}, best epoch={metadata['best_epoch']}, "
            f"score={metadata['best_score']}, motion={metadata['motion_id']}"
        )
        # Lightning uses trainer/global_step; avoid setting W&B's internal history step.
        logger.experiment.log(
            {
                key: wandb.Video(str(video), format="mp4", caption=caption),
                "trainer/global_step": current_epoch,
                **{
                    (f"videos/{k}" if index == 0 else f"{key}/{k}"): metadata[k]
                    for k in (
                        "trigger_epoch",
                        "best_epoch",
                        "best_score",
                        "motion_id",
                    )
                },
            }
        )
        log.info("Uploaded best-policy video: %s", video)

    def _discard_job(self):
        job, self.job = self.job, None
        if job is None:
            return
        if job.get("timer") is not None:
            job["timer"].cancel()
        process = job["process"]
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        shutil.rmtree(job["scratch"], ignore_errors=True)
        (job["folder"] / "best_policy.partial.mp4").unlink(missing_ok=True)

    def close(self):
        try:
            self._poll(self.agent.current_epoch, advance=False)
        except Exception:
            log.warning("Could not upload final best-policy video", exc_info=True)
        finally:
            self._discard_job()
