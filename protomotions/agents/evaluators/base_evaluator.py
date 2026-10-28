# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base evaluator for agent evaluation and metrics computation.

This module provides the base evaluation infrastructure for computing performance
metrics during training and evaluation. Evaluators run periodic assessments of
agent performance and compute task-specific metrics.

Key Classes:
    - BaseEvaluator: Base class for all evaluators with hook-based customization

Key Features:
    - Periodic evaluation during training
    - One rollout loop with hooks for motion coverage and action selection
    - MdpComponent-based evaluation with threshold failure detection
    - Aggregate metrics via plugin system (see aggregate_metrics.py)
    - Episode statistics aggregation
    - Distributed evaluation support

Note:
    Aggregate metric plugins (SmoothnessAggregateMetric, ActionSmoothnessAggregateMetric)
    are defined in aggregate_metrics.py and compute post-hoc statistics over
    accumulated MotionMetrics trajectories.
"""

import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any
from lightning.fabric import Fabric

from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.evaluators.manager import EvaluationManager, EvaluationResult
from protomotions.agents.evaluators.config import EvaluatorConfig
from protomotions.agents.evaluators.aggregate_metrics import (
    SmoothnessAggregateMetric,
    ActionSmoothnessAggregateMetric,
)

class BaseEvaluator:
    """Base class for agent evaluation and metrics computation.

    Runs periodic evaluations during training to assess agent performance.
    Collects episode statistics, computes task-specific metrics, and provides
    feedback for checkpoint selection (best model saving).

    Args:
        agent: The agent being evaluated.
        fabric: Lightning Fabric instance for distributed evaluation.
        config: Evaluator configuration specifying eval frequency and length.

    Example:
        >>> evaluator = BaseEvaluator(agent, fabric, config)
        >>> result = evaluator.evaluate()
    """

    def __init__(self, agent: Any, fabric: Fabric, config: EvaluatorConfig):
        """
        Initialize the evaluator.

        Args:
            agent: The agent to evaluate
            fabric: Lightning Fabric instance for distributed training
        """
        self.agent = agent
        self.fabric = fabric
        self.config = config
        self.config.validate()


        self.metric_plugins = []
        self._register_plugins()
        self.eval_count = 0

        self._evaluation: Optional[EvaluationManager] = None

        # Instance state for metrics collection during evaluation
        self._metrics: Optional[Dict] = None

    @property
    def device(self) -> torch.device:
        """Device for computations (from fabric)."""
        return self.fabric.device

    @property
    def env(self) -> BaseEnv:
        """Environment instance (from agent)."""
        return self.agent.env

    @property
    def root_dir(self):
        """Root directory for saving outputs (from agent)."""
        return self.agent.root_dir

    @torch.no_grad()
    def evaluate(self) -> EvaluationResult:
        """
        Evaluate the agent and calculate metrics.
        This is the main entry point that orchestrates the evaluation process.

        Action selection belongs to the evaluator config. This method only
        measures; the training caller decides whether to apply its results.

        Returns:
            Logs, optional checkpoint score, evaluated count and trial outcomes.
        """
        if not (self.config.metric_components or self.config.failure_components):
            return EvaluationResult({}, None, 0)

        modes = [(module, module.training) for module in self.agent.model.modules()]
        try:
            self.agent.eval()
            self._disable_perturbations()
            self._metrics = self.initialize_eval()
            if self._metrics is None:
                return EvaluationResult({}, None, 0)
            self.run_evaluation()
            result = EvaluationResult(
                *self.process_eval_results(),
                failed=self._evaluation.failed.clone(),
                evaluated=self._evaluation.evaluated,
            )
            self.eval_count += 1
            return result
        finally:
            try:
                self.cleanup_after_evaluation()
            finally:
                self._restore_perturbations()
                # Preserve mixed modes (e.g. frozen expert eval, student train).
                for module, training in modes:
                    module.training = training

    @property
    def num_envs(self) -> int:
        """Number of environments (from agent)."""
        return self.agent.num_envs

    @property
    def max_eval_steps(self) -> int:
        """Maximum steps per evaluation episode."""
        return self.config.max_eval_steps

    def initialize_eval(self) -> Dict:
        """Initialize evaluation tracking."""
        self._save_training_state()
        self._init_eval_component_buffers(self.num_envs)
        return {}

    def _save_training_state(self):
        self._env_snapshot = self.env.save_state()
        self._model_context_snapshot = self.agent.model.save_rollout_context()

    def run_evaluation(self) -> None:
        """Run the evaluation process."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.evaluate_episode(env_ids, self.max_eval_steps)

    def evaluate_episode(self, env_ids: Tensor, max_steps: int) -> None:
        """Run a single episode batch.

        Subclasses customize scheduling and collection through hooks:
        - _on_episode_start: pre-reset setup
        - _get_reset_kwargs: customize env.reset() call
        - _scored_trials: map active environments to trial IDs
        - _on_episode_step: per-step data collection

        Failed trials stop collecting immediately. Their environments restart as
        unscored background environments while the remaining trials finish.

        Args:
            env_ids: Environment IDs to evaluate [num_envs]
            max_steps: Maximum steps for this episode
        """
        self._on_episode_start(env_ids)

        obs = self._reset_episode(env_ids)
        self._unscored_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._unscored_envs[env_ids] = False
        self.agent.pre_collect_step(0)
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        active_envs, trial_ids = self._scored_trials(env_ids, 0)
        for step_idx in range(max_steps):
            if active_envs.numel() == 0:
                break
            actions = self._policy_action(obs_td)

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            # Measure before any background reset invalidates the step context.
            self._check_evaluation_failures(active_envs, trial_ids)
            self._on_episode_step(active_envs, trial_ids, extras, actions)
            # Failure is latched in the manager, not the environment. A reset
            # cannot re-add a trial: this set only shrinks within a batch.
            # The manager enforces exclusion; filtering here also avoids needless rollout.
            active_envs, trial_ids = self._scored_trials(env_ids, step_idx + 1)
            # No more actions are needed when this batch has finished. The
            # outer scheduler still visits the remaining motion batches.
            if active_envs.numel() == 0 or step_idx + 1 == max_steps:
                break
            next_unscored = torch.ones_like(self._unscored_envs)
            next_unscored[active_envs] = False
            just_finished = next_unscored & ~self._unscored_envs
            self._unscored_envs = next_unscored
            restart_ids = (just_finished | (dones.bool() & next_unscored)).nonzero().flatten()
            if restart_ids.numel() > 0:
                obs = self._reset_envs(restart_ids)
            self.agent.pre_collect_step(step_idx + 1)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

    def _reset_episode(self, env_ids: Tensor):
        return self._reset_envs(env_ids)

    def _reset_envs(self, env_ids: Tensor):
        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        self.agent.model.reset_rollout_context(env_ids=env_ids)
        return obs

    def _on_episode_start(self, env_ids: Tensor) -> None:
        """Hook called before episode reset. Override in subclasses for pre-reset setup.

        Args:
            env_ids: Environment IDs about to be reset
        """
        pass

    def _get_reset_kwargs(self) -> dict:
        """Hook to provide extra kwargs for env.reset(). Override in subclasses.

        Returns:
            Dictionary of kwargs passed to env.reset()
        """
        return {}

    def _scored_trials(self, env_ids: Tensor, step_idx: int):
        """Base trials use environment IDs; failed trials never re-enter."""
        active = ~self._evaluation.failed[env_ids]
        return env_ids[active], env_ids[active]

    def _on_episode_step(self, env_ids: Tensor, trial_ids: Tensor, extras: Dict, actions: Tensor) -> None:
        """Hook called after each step. Override in subclasses to collect metrics.

        Args:
            env_ids: Environment IDs active this step
            trial_ids: Corresponding trial IDs for collection
            extras: Extra data from env.step()
            actions: Actions taken this step
        """
        pass

    def _policy_action(self, obs_td) -> Tensor:
        """Run the policy and select the action it drives the robot from."""
        model_outs = self.agent.inference_step(obs_td)
        return model_outs[self.agent.model.deployable_action_key]

    def process_eval_results(self) -> Tuple[Dict, Optional[float], int]:
        """Process collected metrics and prepare for logging.

        Returns:
            Tuple containing:
                - Dict of evaluation metrics for logging
                - Optional score value for determining best model
                - Number of items evaluated (for weighted cross-rank aggregation)
        """
        return self._evaluation.results() if self._evaluation is not None else ({}, None, 0)

    def get_state_dict(self) -> Dict:
        return {"eval_count": self.eval_count}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.eval_count = state_dict["eval_count"]

    def update_training_state(self, result: EvaluationResult):
        """Apply results when explicitly requested by the training caller."""

    def cleanup_after_evaluation(self) -> None:
        """Clean up after evaluation."""
        # Initialization can fail between snapshots. Restore each captured part.
        env_snapshot = self.__dict__.pop("_env_snapshot", None)
        model_snapshot = self.__dict__.pop("_model_context_snapshot", None)
        try:
            if env_snapshot is not None:
                self.env.restore_state(env_snapshot)
        finally:
            if model_snapshot is not None:
                self.agent.model.load_rollout_context(model_snapshot)
        self._metrics = None
        self._evaluation = None

    def _disable_perturbations(self) -> None:
        """Temporarily disable perturbations for clean evaluation metrics.

        During training, perturbations (push randomization and reset noise) are
        active in the shared environment. These must be disabled during evaluation
        so that the success metric reflects pure tracking ability, not robustness
        to disturbances. Two things act on these results: when ``score_key`` is
        set, the best success rate so far decides which checkpoint is saved as
        ``score_based.ckpt``, and ``MimicEvaluator.update_training_state`` uses
        each motion's pass or fail to re-weight how often training samples it.
        The env owns the switch (``set_perturbations_enabled``); this only
        remembers the state to put back.
        """
        self._saved_perturbations_enabled = self.env.perturbations_enabled
        self.env.set_perturbations_enabled(False)

    def _restore_perturbations(self) -> None:
        """Restore perturbation state after evaluation."""
        if "_saved_perturbations_enabled" in self.__dict__:
            self.env.set_perturbations_enabled(
                self.__dict__.pop("_saved_perturbations_enabled")
            )

    def _init_eval_component_buffers(self, num_eval_ids: int) -> None:
        self._evaluation = EvaluationManager(
            self.config, self.env.robot_config, num_eval_ids, self.num_envs, self.device
        )
        self._evaluation.validate_bindings(
            self._evaluation.components, self.env.context, group="evaluation"
        )

    def _check_evaluation_failures(self, active_env_ids: Tensor, active_motion_ids: Tensor) -> None:
        if self._evaluation is not None:
            self._evaluation.update(self.env.context, active_env_ids, active_motion_ids)

    def _add_robot_state_metrics(
        self,
        metrics: Dict[str, MotionMetrics],
        num_motions: int,
        motion_num_frames: torch.Tensor,
        max_eval_steps: int,
    ) -> None:
        """
        Add metrics for raw robot state (dof_pos, rigid_body_pos, etc.).
        This is needed for derived metrics like smoothness.

        Args:
            metrics: Existing metrics dict to add to
            num_motions: Number of motions to evaluate
            motion_num_frames: Number of frames per motion
            max_eval_steps: Maximum evaluation steps
        """
        state = self.env.simulator.get_robot_state()
        for key, shape in state.get_shape_mapping(flattened=True).items():
            if (
                key == "rigid_body_contacts"
                and not self.env.simulator.has_complete_contact_labels
            ):
                continue
            metrics[key] = MotionMetrics(
                num_motions, motion_num_frames, max_eval_steps,
                num_sub_features=shape[0], device=self.device,
            )

    def _register_plugins(self) -> None:
        """Register metric computation plugins. Override in subclasses."""
        pass

    def _register_smoothness_plugin(
        self, window_sec: float = 0.4, high_jerk_threshold: float = 6500.0
    ) -> bool:
        """
        Convenience method to register smoothness aggregate metric.

        Args:
            window_sec: Window size in seconds for smoothness computation
            high_jerk_threshold: Threshold for classifying high jerk frames

        Returns:
            True when registered. Invalid settings raise at construction.
        """
        self.metric_plugins.append(
            SmoothnessAggregateMetric(self, window_sec, high_jerk_threshold)
        )
        return True

    def _register_action_smoothness_plugin(self) -> bool:
        """
        Convenience method to register action smoothness aggregate metric.

        Measures how much actions change between consecutive timesteps.

        Returns:
            True when registered. Invalid settings raise at construction.
        """
        self.metric_plugins.append(ActionSmoothnessAggregateMetric(self))
        return True

    def _compute_additional_metrics(
        self, metrics: Dict[str, MotionMetrics]
    ) -> Dict[str, float]:
        """
        Run all registered metric plugins to compute additional metrics.

        Args:
            metrics: Dictionary of collected MotionMetrics

        Returns:
            Dictionary of additional computed metrics
        """
        additional_metrics = {}
        for plugin in self.metric_plugins:
            additional_metrics.update(plugin.compute(metrics))
        return additional_metrics

    def _save_list_to_file(
        self, items: list, filename: str, subdirectory: Optional[str] = None
    ) -> None:
        """
        Save a list of items to a text file (one per line).

        Args:
            items: List of items to save
            filename: Name of output file
            subdirectory: Optional subdirectory within root_dir
        """
        if subdirectory:
            output_dir = self.root_dir / subdirectory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
        else:
            output_path = self.root_dir / filename

        print(f"Saving to: {output_path}")
        with open(output_path, "w") as f:
            for item in items:
                f.write(f"{item}\n")

    @torch.no_grad()
    def simple_test_policy(self) -> None:
        """
        Simple evaluation loop for interactive testing.

        Runs the policy indefinitely, resetting the envs that finish; press
        Ctrl+C to stop.
        """
        self.agent.eval()
        done_indices = None
        step = 0

        print("Evaluating policy... (Ctrl+C to stop)")
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                self.agent.pre_collect_step(step)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)

                action = self._policy_action(obs_td)

                _, _, dones, _, _ = self.env.step(action)

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                step += 1
        except KeyboardInterrupt:
            print(f"\nStopped after {step} steps.")
