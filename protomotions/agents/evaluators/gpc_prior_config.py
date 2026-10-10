"""Bounded, reproducible GPC stage-two evaluation configuration."""

from dataclasses import dataclass
from typing import Optional, Tuple

from protomotions.agents.evaluators.config import EvaluatorConfig


@dataclass
class GPCPriorEvaluatorConfig(EvaluatorConfig):
    _target_: str = "protomotions.agents.evaluators.gpc_prior_evaluator.GPCPriorEvaluator"
    # Primary weights determine the score/best checkpoint; comparison is optional.
    weights: str = "ema"
    compare_raw_ema: bool = False
    max_eval_steps: int = 1000
    num_eval_envs: int = 32
    teacher_steps: int = 64
    teacher_metric_stride: int = 4
    seed: int = 20260929
    validation_motion_file: Optional[str] = None
    output_dir: Optional[str] = None
    enabled: bool = True
    mode: str = "both"  # offline, rollout, both; training cadence is epochs
    temperature: float = 1.0
    top_p: float = 0.9
    histogram_top_k: int = 20
    # Repeated-pose cohort is GLOBAL; every pose and its repeats stay on one rank.
    rollout_num_poses: int = 8
    rollouts_per_pose: int = 8
    perturbation_speeds: Tuple[float, ...] = (0.0,)
    perturbation_step: int = 200
    body_names: Optional[Tuple[str, ...]] = None  # all rigid bodies by default
    foot_names: Optional[Tuple[str, ...]] = None  # robot's common foot naming
    jerk_window_seconds: float = 0.4
    jerk_min_path_m: float = 1e-6
    recovery_height: float = 0.45
    recovery_up_z: float = 0.8
    recovery_max_linear_speed: float = 0.5
    recovery_max_angular_speed: float = 1.0
    recovery_hold_seconds: float = 0.5
    recovery_timeout_seconds: float = 3.0
    recovery_trigger: str = "fall_or_push"  # fall-only is also available
    latency_warmup_steps: int = 10
    latency_stride: int = 10
    bootstrap_samples: int = 500
    save_trajectories: bool = True
    video_cases: int = 0  # optional skeleton replays of actual simulated traces
    video_fps: int = 25
    # Upright-locomotion diagnostic, not a general acrobatics success criterion.
    min_anchor_height: float = 0.30
    min_up_z: float = 0.25
    failure_persistence_steps: int = 5

    def validate(self):
        if self.rollout_num_poses < 1 or self.rollouts_per_pose < 2:
            raise ValueError("Need at least one pose and two repeats for APD")
        if not self.perturbation_speeds or self.perturbation_speeds[0] != 0 or any(v < 0 for v in self.perturbation_speeds):
            raise ValueError("Perturbation speeds must start at zero and be nonnegative")
        if len(set(self.perturbation_speeds)) != len(self.perturbation_speeds):
            raise ValueError("Perturbation conditions must be unique")
        if len(self.perturbation_speeds)>1 and not 1 <= self.perturbation_step < self.max_eval_steps:
            raise ValueError("Perturbation step must be inside the rollout horizon")
        if self.recovery_trigger not in ("fall", "fall_or_push"):
            raise ValueError("Invalid recovery trigger")
        if self.recovery_timeout_seconds < self.recovery_hold_seconds or self.recovery_hold_seconds <= 0:
            raise ValueError("Recovery timeout must cover a positive stable hold")
        if self.recovery_height < self.min_anchor_height or not self.min_up_z <= self.recovery_up_z <= 1:
            raise ValueError("Recovery thresholds must be at least as strict as survival")
        if min(self.jerk_window_seconds,self.jerk_min_path_m,self.recovery_max_linear_speed,self.recovery_max_angular_speed) <= 0:
            raise ValueError("Trajectory and recovery scales must be positive")
        if self.latency_warmup_steps < 0 or self.video_cases < 0:
            raise ValueError("Warmup and video counts cannot be negative")
        if self.mode not in ("both", "offline", "rollout"):
            raise ValueError("mode must be both, offline, or rollout")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature > 0 and 0 < top_p <= 1 required")
        if self.eval_metrics_every is not None and self.eval_metrics_every < 1:
            raise ValueError("eval_metrics_every must be positive or None")
        if self.weights not in ("ema", "raw"):
            raise ValueError("GPC evaluator weights must be 'ema' or 'raw'")
        for name in ("max_eval_steps", "num_eval_envs", "teacher_steps",
                     "teacher_metric_stride", "failure_persistence_steps", "histogram_top_k",
                     "latency_stride", "bootstrap_samples", "video_fps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_anchor_height < 0 or not -1 <= self.min_up_z <= 1:
            raise ValueError("Invalid upright-survival thresholds")
        if self.validation_motion_file and "slurmrank" in self.validation_motion_file:
            raise ValueError("Prior validation requires one concrete MotionLib .pt file")
