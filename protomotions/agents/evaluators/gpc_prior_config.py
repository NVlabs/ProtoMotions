"""Bounded, reproducible GPC stage-two evaluation configuration."""

from dataclasses import dataclass
from typing import Optional

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
    # Upright-locomotion diagnostic, not a general acrobatics success criterion.
    min_anchor_height: float = 0.30
    min_up_z: float = 0.25
    failure_persistence_steps: int = 5

    def validate(self):
        if self.mode not in ("both", "offline", "rollout"):
            raise ValueError("mode must be both, offline, or rollout")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature > 0 and 0 < top_p <= 1 required")
        if self.eval_metrics_every is not None and self.eval_metrics_every < 1:
            raise ValueError("eval_metrics_every must be positive or None")
        if self.weights not in ("ema", "raw"):
            raise ValueError("GPC evaluator weights must be 'ema' or 'raw'")
        for name in ("max_eval_steps", "num_eval_envs", "teacher_steps",
                     "teacher_metric_stride", "failure_persistence_steps", "histogram_top_k"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_anchor_height < 0 or not -1 <= self.min_up_z <= 1:
            raise ValueError("Invalid upright-survival thresholds")
        if self.validation_motion_file and "slurmrank" in self.validation_motion_file:
            raise ValueError("Prior validation requires one concrete MotionLib .pt file")
