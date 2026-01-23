# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""Base agent implementation for reinforcement learning.

This module provides the core agent class that all RL algorithms extend. It handles
the complete training lifecycle including rollout collection, experience buffering,
optimization, checkpointing, evaluation, and distributed training coordination.

Key Classes:
    - BaseAgent: Abstract base class for all RL agents

Key Features:
    - Distributed training with Lightning Fabric
    - Experience buffer management
    - Automatic checkpoint saving/loading
    - Periodic evaluation during training
    - Reward normalization
    - Episode statistics tracking
"""

import torch
from torch import Tensor
import torch.distributed as dist
from abc import abstractmethod
import logging
import torch.nn as nn

import time
import math
from pathlib import Path
from typing import Optional, Dict

from lightning.fabric import Fabric
from tensordict import TensorDict

from protomotions.utils.hydra_replacement import get_class

from protomotions.agents.utils.metering import TimeReport, TensorAverageMeterDict
from protomotions.agents.utils.data import DictDataset, ExperienceBuffer
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.utils.normalization import RewardRunningMeanStd, RunningMeanStd
from rich.progress import track
from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.base_agent.config import BaseAgentConfig
from protomotions.agents.utils.training import aggregate_scalar_metrics

log = logging.getLogger(__name__)


class BaseAgent:
    """Base class for reinforcement learning agents.

    Provides the core training infrastructure that all algorithm implementations extend.
    Handles experience collection, optimization loop, checkpointing, and evaluation.
    Subclasses must implement model creation and algorithm-specific training logic.

    Args:
        fabric: Lightning Fabric instance for distributed training.
        env: Environment instance for interaction.
        config: Agent configuration with hyperparameters.
        root_dir: Directory for saving checkpoints and logs (optional, uses logger dir if available).

    Attributes:
        model: Neural network model (created by subclass).
        optimizer: Optimizer for model parameters.
        experience_buffer: Buffer for storing rollout data.
        evaluator: Evaluator for computing performance metrics.
    """

    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(
        self,
        fabric: Fabric,
        env: BaseEnv,
        config: BaseAgentConfig,
        root_dir: Optional[Path] = None,
    ):
        """Initialize the base agent.

        Sets up distributed training infrastructure, initializes tracking metrics,
        and creates the evaluator. Subclasses should call super().__init__() first.

        Args:
            fabric: Lightning Fabric for distributed training and device management.
            env: Environment instance for agent-environment interaction.
            config: Configuration containing hyperparameters and training settings.
            root_dir: Optional directory for saving outputs (uses logger dir if None).
        """
        self.fabric = fabric
        self.device: torch.device = self.fabric.device
        self.env = env
        self.motion_lib = self.env.motion_lib
        self.motion_manager = self.env.motion_manager
        self.config = config

        self.num_envs: int = self.env.num_envs
        self.num_steps: int = self.config.num_steps
        self.num_mini_epochs: int = self.config.num_mini_epochs
        self.gamma: float = self.config.gamma
        self._should_stop: bool = False
        self.max_epochs: int = (
            self.config.training_max_steps
            // self.fabric.world_size
            // self.num_envs
            // self.num_steps
        )

        # timer
        self.time_report = TimeReport()

        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.current_rewards = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        if self.config.normalize_rewards:
            self.running_reward_norm = RewardRunningMeanStd(
                fabric=self.fabric,
                shape=(1,),
                gamma=self.gamma,
                device=self.device,
                clamp_value=self.config.normalized_reward_clamp_value,
            )
        else:
            self.running_reward_norm = None

        self.episode_reward_meter = TensorAverageMeterDict(device=self.device)
        self.episode_length_meter = TensorAverageMeterDict(device=self.device)
        self.episode_env_tensors = TensorAverageMeterDict(device=self.device)
        self.step_count = 0
        self.current_epoch = 0
        self.fit_start_time = None
        self.best_evaluated_score = None

        # Hacky flag to skip policy update right after eval to avoid training spikes
        self._skip_next_policy_update = False

        # Set root_dir: use logger's root_dir if available, otherwise use passed parameter
        if self.fabric.loggers:
            self.root_dir = Path(self.fabric.loggers[0].root_dir)
        elif root_dir is not None:
            self.root_dir = Path(root_dir)
        else:
            raise ValueError("No root_dir provided and no logger available")

        EvaluatorClass = get_class(self.config.evaluator._target_)
        self.evaluator: BaseEvaluator = EvaluatorClass(
            agent=self, fabric=self.fabric, config=self.config.evaluator
        )

        self.just_loaded_checkpoint_should_evaluate = False

    @property
    def should_stop(self):
        return self.fabric.broadcast(self._should_stop)

    def setup(self):
        self.fabric.call("on_model_init_start")
        model = self.create_model()

        # Move model to device BEFORE materializing lazy modules
        model = model.to(self.device)
        self.model = model

        # Once model is created, we pass fabric to the RunningMeanStd modules.
        # This allows the modules to internally handle distributed aggregation of normalization moments.
        def pass_fabric_to_running_mean_std(module):
            if isinstance(module, RunningMeanStd):
                module.fabric = self.fabric

        self.model.apply(pass_fabric_to_running_mean_std)

        # Materialize lazy modules (LazyLinear, RunningMeanStd)
        # by running a dummy forward pass before wrapping with DDP
        log.info("Materializing lazy modules...")
        with torch.no_grad():
            dummy_obs = self.env.get_obs()
            dummy_obs = self.add_agent_info_to_obs(dummy_obs)
            dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
            _ = self.model(dummy_obs_td)

        self.fabric.call("on_model_init_end")

        self.fabric.call("on_optimizer_init_start")
        self.create_optimizers(model)
        self.fabric.call("on_optimizer_init_end")

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_optimizers(self, model: nn.Module):
        pass

    def load(self, checkpoint: Path, load_env: bool = True):
        if checkpoint is not None:
            self.fabric.call("on_load_checkpoint_start")
            path_before_resolve = Path(checkpoint)
            checkpoint = path_before_resolve.resolve()
            print(f"Loading model from checkpoint: {checkpoint}")

            state_dict = torch.load(
                checkpoint, map_location=self.device, weights_only=False
            )
            self.load_parameters(state_dict)

            self.just_loaded_checkpoint_should_evaluate = True

            if load_env:
                # Load env state from the same directory as the checkpoint.
                task_id = self.env.get_task_id()
                env_checkpoint = self.root_dir / f"env_{task_id}.ckpt"
                if env_checkpoint.exists():
                    print(f"Loading env checkpoint: {env_checkpoint}")
                    env_state_dict = torch.load(
                        env_checkpoint, map_location=self.device, weights_only=False
                    )
                    self.env.load_state_dict(env_state_dict)

            self.fabric.call("on_load_checkpoint_end")

    def load_parameters(self, state_dict):
        """Load agent parameters from state dictionary.

        Restores training state including epoch counter, step count, timing info,
        best scores, normalization statistics, and model weights.

        Args:
            state_dict: Dictionary containing saved agent state from checkpoint.
                       Expected keys: epoch, step_count, run_start_time, best_evaluated_score,
                       running_reward_norm (if normalization enabled), model.
        """
        self.current_epoch = state_dict["epoch"]

        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]

        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        if self.config.normalize_rewards:
            self.running_reward_norm.load_state_dict(state_dict["running_reward_norm"])

        self.model.load_state_dict(state_dict["model"])

    # -----------------------------
    # Model Saving and State Dict
    # -----------------------------
    def get_state_dict(self, state_dict):
        """Get complete state dictionary for checkpointing.

        Collects all agent state including model weights, training progress,
        and normalization statistics into a single dictionary for saving.

        Args:
            state_dict: Existing state dict to update (typically empty dict).

        Returns:
            Updated state dictionary containing all agent state.
        """
        extra_state_dict = {
            "model": self.model.state_dict(),
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "best_evaluated_score": self.best_evaluated_score,
        }

        if self.config.normalize_rewards:
            extra_state_dict["running_reward_norm"] = (
                self.running_reward_norm.state_dict()
            )

        state_dict.update(extra_state_dict)
        return state_dict

    def save(self, checkpoint_name: str = "last.ckpt", new_high_score: bool = False):
        """
        Save model checkpoint and environment state.

        Args:
            checkpoint_name: Name of checkpoint file (e.g., "last.ckpt" or "epoch_100.ckpt")
            new_high_score: Whether this is a new high score (will also save as score_based.ckpt)
        """
        self.fabric.call("on_save_checkpoint_start", self)

        save_dir = self.root_dir
        state_dict = self.get_state_dict({})

        # Save the main checkpoint
        self.fabric.save(save_dir / checkpoint_name, state_dict)
        log.info(f"Saved checkpoint: {save_dir / checkpoint_name}")

        # Save environment checkpoint for unique task IDs
        task_id = self.env.get_task_id()
        per_rank_task_id = [None for _ in range(self.fabric.world_size)]
        dist.all_gather_object(per_rank_task_id, task_id)

        # Only ranks with unique task IDs save the env checkpoint
        rank_to_task_id = {}
        seen_task_ids = set()
        for rank, tid in enumerate(per_rank_task_id):
            if tid not in seen_task_ids:
                seen_task_ids.add(tid)
                rank_to_task_id[rank] = tid

        if self.fabric.global_rank in rank_to_task_id:
            env_checkpoint = save_dir / f"env_{task_id}.ckpt"
            env_state_dict = self.env.get_state_dict()
            torch.save(env_state_dict, env_checkpoint)
            log.info(
                f"Saved env checkpoint: {env_checkpoint}, rank {self.fabric.global_rank}"
            )
        self.fabric.barrier()

        # Check if new high score flag is consistent across devices
        gathered_high_score = self.fabric.all_gather(new_high_score)
        assert all(
            [x == gathered_high_score[0] for x in gathered_high_score]
        ), "New high score flag should be the same across all ranks."

        if new_high_score:
            self.fabric.save(save_dir / "score_based.ckpt", state_dict)
            log.info(
                f"New best performing controller found with score {self.best_evaluated_score}. "
                f"Model saved to {save_dir / 'score_based.ckpt'}"
            )

        self.fabric.call("on_save_checkpoint_end", self)

    # -----------------------------
    # Experience Buffer and Training Loop
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        """Register algorithm-specific keys in the experience buffer.

        Subclasses override this to add custom keys to the experience buffer
        (e.g., AMP adds discriminator observations, ASE adds latent codes).
        """
        pass

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect experience data during rollout at current timestep.

        Called once per timestep during the data collection (rollout) phase.
        Subclasses implement this to:
        1. Query the policy to select actions from observations
        2. Store intermediate training data in experience buffer (e.g., values, log probs)
        3. Return actions to apply to the environment

        Args:
            obs: Dictionary of observations from environment
            step: Current timestep index in the rollout [0, num_steps)

        Returns:
            Actions tensor to apply to environment [num_envs, action_dim]
        """
        # Single forward pass through model
        output_td = self.model(obs_td)

        # Store all model outputs in experience buffer
        for key in self.model_output_keys:
            if key in output_td:
                assert torch.all(
                    torch.isfinite(output_td[key])
                ), f"NaN or Inf in {key}: {output_td[key]}"
                self.experience_buffer.update_data(key, step, output_td[key])

        # Return action for environment
        return output_td["action"]

    def fit(self):
        """Main training loop for the agent.

        Executes the complete training process including:
        1. Experience buffer setup (auto-registers keys from model outputs)
        2. Environment rollouts (data collection)
        3. Model optimization
        4. Periodic evaluation
        5. Checkpoint saving
        6. Logging and metrics

        The loop runs for max_epochs epochs, where each epoch collects num_steps
        of experience from num_envs parallel environments, then performs multiple
        optimization steps on the collected data.

        Note:
            This is the main entry point for training. Call setup() before fit().
        """
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(
            self.num_envs, self.num_steps, device=self.device
        )

        # Get initial observations from environment
        obs, _ = self.env.reset()
        obs = self.add_agent_info_to_obs(obs)
        obs_td = self.obs_dict_to_tensordict(obs)

        # Register environment observation keys
        for key, env_tensor in obs_td.items():
            shape = env_tensor.shape
            dtype = env_tensor.dtype
            self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Auto-register model output keys by running one forward pass
        with torch.no_grad():
            output_td = self.model(obs_td)

            # Track which keys are model outputs (not from environment)
            self.model_output_keys = self.model.out_keys
            for key in self.model_output_keys:
                value = output_td[key]
                if isinstance(value, torch.Tensor):
                    # Handle both scalar and tensor outputs
                    if value.ndim == 1:  # Scalar per env
                        self.experience_buffer.register_key(key)
                    else:  # Tensor per env
                        self.experience_buffer.register_key(
                            key, shape=value.shape[1:], dtype=value.dtype
                        )

            log.info(f"Auto-registered model output keys: {self.model_output_keys}")

        # Basic keys always needed
        self.experience_buffer.register_key("rewards")
        if self.config.normalize_rewards:
            self.experience_buffer.register_key("unnormalized_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.register_algorithm_experience_buffer_keys()

        # Force reset on fit start
        done_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.max_epochs:
            self.epoch_start_time = time.time()

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    # Reset returns observations directly
                    obs, _ = self.env.reset(done_indices)
                    obs = self.add_agent_info_to_obs(obs)
                    obs_td = self.obs_dict_to_tensordict(obs)

                    # Store observations in the experience buffer
                    for key, env_tensor in obs_td.items():
                        self.experience_buffer.update_data(key, step, env_tensor)

                    # Collect: Get actions and store algorithm-specific data
                    action = self.collect_rollout_step(obs_td, step)
                    self.check_obs_for_nans(obs_td, action)

                    # Step the environment
                    next_obs, rewards, dones, terminated, extras = self.env.step(action)
                    assert torch.all(
                        torch.isfinite(rewards)
                    ), f"NaN or Inf in rewards: {rewards}"

                    next_obs = self.add_agent_info_to_obs(next_obs)
                    next_obs_td = self.obs_dict_to_tensordict(next_obs)

                    # Allow subclasses to modify dones/terminated (e.g., AMP discriminator termination)
                    dones, terminated, extras = self.post_env_step_modifications(
                        dones, terminated, extras
                    )
                    done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

                    # Record metrics and store data from this rollout step
                    self.record_rollout_step(
                        next_obs_td,
                        action,
                        rewards,
                        dones,
                        terminated,
                        done_indices,
                        extras,
                        step,
                    )

                    self.step_count += self.get_step_count_increment()

                self.normalize_rewards_in_buffer()

            # Skip policy update right after eval to avoid training spikes (hacky fix)
            if self._skip_next_policy_update:
                training_log_dict = {"skipped_policy_update": 1.0}
                self._skip_next_policy_update = False
                # Still need to preprocess dataset (compute advantages/returns) before clearing
                self.pre_process_dataset()
                # Clear the experience buffer to reset for next epoch
                _ = self.experience_buffer.make_dict()
            else:
                training_log_dict = self.optimize_model()

            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save epoch-based checkpoint (epoch_xxx.ckpt)
            if (
                self.config.save_epoch_checkpoint_every is not None
                and self.current_epoch % self.config.save_epoch_checkpoint_every == 0
            ):
                epoch_checkpoint_name = f"epoch_{self.current_epoch}.ckpt"
                self.save(checkpoint_name=epoch_checkpoint_name)

            # Save last.ckpt at specified intervals
            if self.current_epoch % self.config.save_last_checkpoint_every == 0:
                self.save(checkpoint_name="last.ckpt")

            if (
                self.evaluator is not None
                and self.evaluator.config.eval_metrics_every is not None
                and (
                    self.current_epoch > 0
                    and self.current_epoch % self.evaluator.config.eval_metrics_every
                    == 0
                )
                or self.just_loaded_checkpoint_should_evaluate
            ):
                self.fabric.call("on_eval_start", self)

                eval_log_dict, evaluated_score = self.evaluator.evaluate()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                self.fabric.call("on_eval_end", self)

                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(checkpoint_name="last.ckpt", new_high_score=True)
                training_log_dict.update(eval_log_dict)
                self.just_loaded_checkpoint_should_evaluate = False

                # Skip next policy update to avoid training spikes after eval (hacky fix)
                self._skip_next_policy_update = True

            self.post_epoch_logging(training_log_dict)
            if self.config.max_episode_length_manager is not None:
                max_episode_length = (
                    self.config.max_episode_length_manager.current_max_episode_length(
                        self.current_epoch
                    )
                )
                self.env.max_episode_length = max_episode_length
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.fabric.call("on_training_stop", self)
                self.save(checkpoint_name="last.ckpt")
                return

        self.time_report.report()
        self.save(checkpoint_name="last.ckpt")
        self.fabric.call("on_fit_end", self)

    # -----------------------------
    # Environment Interaction Helpers
    # -----------------------------
    def add_agent_info_to_obs(self, obs: Dict) -> Dict:
        """Add agent-specific observations to the environment observations.

        This can be used to augment observations from both reset() and step()
        with agent-specific information (e.g., latent codes, discriminator obs).
        """
        return obs

    def obs_dict_to_tensordict(self, obs_dict: Dict) -> TensorDict:
        """Convert observation dict to TensorDict.

        Args:
            obs_dict: Dictionary of observation tensors from environment.

        Returns:
            TensorDict with same keys and values.
        """
        batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
        return TensorDict(obs_dict, batch_size=batch_size, device=self.device)

    def post_env_step_modifications(self, dones, terminated, extras):
        """Allow subclasses to modify dones/terminated after env.step().

        This hook allows algorithm-specific modifications (e.g., AMP discriminator termination).

        Args:
            dones: Reset flags from environment
            terminated: Termination flags from environment
            extras: Info dictionary from environment

        Returns:
            Modified (dones, terminated, extras) tuple
        """
        return dones, terminated, extras

    def check_obs_for_nans(self, obs, action):
        # Check for NaNs in observations and actions
        for key in obs.keys():
            assert not torch.isnan(obs[key]).any(), f"NaN in {key}: {obs[key]}"
            assert not torch.isinf(obs[key]).any(), f"Inf in {key}: {obs[key]}"
        assert not torch.isnan(action).any(), f"NaN in action: {action}"
        assert not torch.isinf(action).any(), f"Inf in action: {action}"

    def record_rollout_step(
        self,
        next_obs_td: TensorDict,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        terminated: Tensor,
        done_indices: Tensor,
        extras: Dict,
        step: int,
    ):
        """Record metrics and store data after environment step during rollout.

        Called once per timestep during data collection phase, after the environment
        has been stepped. Handles:
        1. Episode statistics tracking (rewards, lengths)
        2. Environment extras logging
        3. Experience buffer updates (actions, rewards, dones)
        4. Reward normalization if enabled

        Args:
            next_obs: Observations after environment step
            actions: Actions that were applied
            rewards: Rewards from environment step
            dones: Reset flags indicating episode completion
            terminated: Termination flags indicating early termination
            done_indices: Indices of environments that finished episodes
            extras: Additional info dictionary from environment
            step: Current timestep index in the rollout [0, num_steps)
        """

        self.current_rewards += rewards
        self.current_lengths += 1

        if len(done_indices) > 0:
            self.episode_reward_meter.add(
                {"episode_reward": self.current_rewards[done_indices]}
            )
            self.episode_length_meter.add(
                {"episode_length": self.current_lengths[done_indices]}
            )

        not_dones = 1.0 - dones.float()
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * not_dones

        extras_mean_std_dict = {}
        for key in extras:
            if "raw/" in key:
                continue
            if isinstance(extras[key], torch.Tensor):
                extra_val = extras[key].float()
                if extras[key].numel() == 1:
                    extras_mean_std_dict[key] = extra_val.item()
                else:
                    extras_mean_std_dict[f"{key}_mean"] = extra_val.mean()
                    # extras_mean_std_dict[f"{key}_std"] = extra_val.std()
        self.episode_env_tensors.add(extras_mean_std_dict)

        self.experience_buffer.update_data("dones", step, dones)

        if self.config.normalize_rewards:
            self.running_reward_norm.record_reward(rewards, terminated)
        self.experience_buffer.update_data("rewards", step, rewards)

    @torch.no_grad()
    def normalize_rewards_in_buffer(self):
        """Normalize rewards after data collection."""
        if not self.config.normalize_rewards:
            return

        rewards = self.experience_buffer.rewards
        self.experience_buffer.batch_update_data("unnormalized_rewards", rewards.clone())
        self.experience_buffer.batch_update_data(
            "rewards", self.running_reward_norm.normalize(rewards)
        )

    # -----------------------------
    # Optimization
    # -----------------------------
    @torch.no_grad()
    def pre_process_dataset(self):
        # Allows for preprocessing of the dataset before it is converted to the DictDataset.
        pass

    def optimize_model(self) -> Dict:
        self.pre_process_dataset()
        dataset = self.process_dataset(self.experience_buffer.make_dict())
        self.train()
        training_log_dict = {}

        for batch_idx in track(
            range(self.max_num_batches()),
            description=f"Epoch {self.current_epoch}, training...",
        ):
            iter_log_dict = {}
            dataset_idx = batch_idx % len(dataset)

            # Reshuffle dataset at the beginning of each mini epoch if configured.
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # Check for NaNs in the batch.
            for key in batch_dict.keys():
                if torch.isnan(batch_dict[key]).any():
                    print(f"NaN in {key}: {batch_dict[key]}")
                    raise ValueError("NaN in training")

            iter_log_dict = self.perform_optimization_step(batch_dict, batch_idx)

            # Memory optimization: Detach intermediate tensors to prevent gradient retention
            for k, v in iter_log_dict.items():
                if isinstance(v, torch.Tensor):
                    iter_log_dict[k] = v.detach()
                if k in training_log_dict:
                    training_log_dict[k][0] += iter_log_dict[k]
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [iter_log_dict[k], 1]

            # Memory optimization: Clear batch_dict to free memory early
            del batch_dict

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()
        return training_log_dict

    @abstractmethod
    def perform_optimization_step(self, batch_dict) -> Dict:
        # Perform a single optimization step and return the log dictionary
        pass

    def post_epoch_logging(self, training_log_dict: Dict):
        end_time = time.time()

        # Get mean episode statistics and clear meters
        episode_reward_dict = self.episode_reward_meter.mean_and_clear()
        episode_length_dict = self.episode_length_meter.mean_and_clear()

        log_dict = {
            "info/episode_length": episode_length_dict.get("episode_length", 0),
            "info/episode_reward": episode_reward_dict.get("episode_reward", 0),
            "info/frames": torch.tensor(self.step_count),
            "info/gframes": torch.tensor(self.step_count / (10**9)),
            "times/fps_last_epoch": (self.num_steps * self.get_step_count_increment())
            / (end_time - self.epoch_start_time),
            "times/fps_total": self.step_count / (end_time - self.fit_start_time),
            "times/training_hours": (end_time - self.fit_start_time) / 3600,
            "times/training_minutes": (end_time - self.fit_start_time) / 60,
            "times/last_epoch_seconds": (end_time - self.epoch_start_time),
            "rewards/task_rewards": self.experience_buffer.rewards.mean().item(),
        }
        if self.config.normalize_rewards:
            log_dict["rewards/unnormalized_task_rewards"] = (
                self.experience_buffer.unnormalized_rewards.mean().item()
            )

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}
        if len(env_log_dict) > 0:
            log_dict.update(env_log_dict)
        log_dict.update(training_log_dict)

        # Aggregate metrics across all devices before logging
        # This ensures wandb reports representative metrics from all ranks, not just rank 0
        aggregated_log_dict = aggregate_scalar_metrics(log_dict, self.fabric)

        # wandb logger does this: assert rank_zero_only.rank == 0
        self.fabric.log_dict(aggregated_log_dict)

    # -----------------------------
    # Helper Functions
    # -----------------------------
    def eval(self):
        """Set the model to evaluation mode.

        Disables training-specific behaviors like dropout and batch normalization updates.
        Typically called before collecting experience or during evaluation.
        """
        self.model.eval()

    def train(self):
        self.model.train()

    def max_num_batches(self):
        """Calculate maximum number of minibatches per epoch.

        Returns:
            Integer number of minibatches needed to process all collected experience.
        """
        return math.ceil(
            self.num_envs
            * self.num_steps
            * self.num_mini_epochs
            / self.config.batch_size
        )

    def get_step_count_increment(self):
        """Calculate step count increment for distributed training.

        Accounts for multiple GPUs and nodes in step counting.

        Returns:
            Number of environment steps per training iteration across all processes.
        """
        return (
            self.num_envs * self.fabric.world_size
        )  # fabric.world_size = num gpu * num nodes

    def terminate_early(self):
        """Request early termination of training.

        Sets a flag that will cause the training loop to exit gracefully
        after the current epoch completes.
        """
        self._should_stop = True

    @torch.no_grad()
    def process_dataset(self, dataset):
        """Process experience buffer into minibatch dataset.

        Converts the collected experience into a dataset that yields minibatches
        for optimization. Shuffles data for better training dynamics.

        Args:
            dataset: Dictionary of experience tensors from experience buffer.

        Returns:
            DictDataset that yields shuffled minibatches of specified batch_size.
        """
        dataset = DictDataset(self.config.batch_size, dataset, shuffle=True)
        return dataset
