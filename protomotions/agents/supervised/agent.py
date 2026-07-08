# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised rollout imitation agent.

This agent collects rollouts with the student policy, labels those states with
an expert policy, and optimizes a configured supervision loss. Algorithms such
as MaskedMimic are experiment/model configurations of this generic loop.
"""

import torch
from torch import Tensor
from tensordict import TensorDict
import logging

from protomotions.utils.config_utils import load_resolved_configs_from_checkpoint
from protomotions.utils.hydra_replacement import get_class
from typing import Tuple, Dict
from pathlib import Path

from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.common.common import weight_init_trainable
from protomotions.agents.common.supervision import compute_supervision_loss
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.supervised.config import RolloutActor
from protomotions.agents.supervised.expert_utils import get_expert_actor_in_keys
from protomotions.agents.utils.normalization import RunningMeanStd

log = logging.getLogger(__name__)


class SupervisedAgent(BaseAgent):
    """Student/expert rollout agent for supervised distillation.

    The agent collects TensorDict rollouts, writes configured model outputs into
    the experience buffer, and optimizes ``SupervisionLossConfig``. Models and
    experiment files define which keys are predictions and labels.
    """

    model: BaseModel

    def create_model(self):
        model_cls = get_class(self.config.model._target_)
        model: BaseModel = model_cls(config=self.config.model)
        if not getattr(model, "skip_default_weight_init", False):
            model.apply(weight_init_trainable)

        # Optionally load a pre-trained expert model if provided.
        # Note: Expert observation components are loaded in the experiment file
        # and prefixed with "expert_" for use during distillation training.
        expert_model_path = self.config.expert_model_path
        if expert_model_path is not None:
            log.info(f"Loading expert model from: {expert_model_path}")

            checkpoint_path = Path(expert_model_path)
            assert (
                checkpoint_path.exists()
            ), f"Could not find expert model at {checkpoint_path}"

            resolved_configs = load_resolved_configs_from_checkpoint(checkpoint_path)

            self.expert_env_config = resolved_configs["env"]
            expert_agent_config: PPOAgentConfig = resolved_configs["agent"]

            # Create the expert model
            ExpertModelConfig = get_class(expert_agent_config.model._target_)
            expert_model: BaseModel = ExpertModelConfig(
                config=expert_agent_config.model
            )

            # Move model to device BEFORE materializing lazy modules
            expert_model = expert_model.to(self.device)
            expert_model.reset_rollout_context(
                num_envs=self.num_envs,
                device=self.device,
            )

            # Once model is created, we pass fabric to the RunningMeanStd modules.
            # This allows the modules to internally handle distributed aggregation of normalization moments.
            def pass_fabric_to_running_mean_std(module):
                if isinstance(module, RunningMeanStd):
                    module.fabric = self.fabric

            expert_model.apply(pass_fabric_to_running_mean_std)

            expert_actor = self._external_expert_module_from(expert_model)
            expert_actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
            if not expert_actor_in_keys:
                expert_actor_in_keys = list(getattr(expert_actor, "in_keys", []))

            log.info("Materializing expert actor lazy modules...")
            # External experts are frozen inference modules. Only the actor is
            # needed to label actions; materializing the full actor-critic model
            # can require critic-only observations that the distillation env does
            # not provide.
            expert_model.eval()
            with torch.no_grad():
                dummy_obs = self.env.get_obs()
                # Build expert obs tensordict (strips "expert_" prefix from keys)
                dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
                dummy_expert_obs_td = self._build_expert_obs_td(
                    dummy_obs_td, expert_actor_in_keys
                )
                self._materialize_frozen_external_expert(
                    expert_actor,
                    dummy_expert_obs_td,
                )

            # Load weights before any distributed wrapper changes module keys.
            pre_trained_expert = torch.load(
                str(checkpoint_path),
                map_location=self.device,
                weights_only=False,
            )
            self._load_external_expert_state(
                expert_model,
                pre_trained_expert["model"],
            )
            for param in expert_model.parameters():
                param.requires_grad = False

            # Keep the external expert as a plain frozen module. The trainable
            # student is wrapped by create_optimizers(); the expert only labels
            # rollouts and does not need gradient synchronization.
            self.expert_model = expert_model
            self.expert_actor = expert_actor
            self.expert_actor_in_keys = expert_actor_in_keys
            self.expert_model.eval()
        else:
            self.expert_model = None
            self.expert_actor = None
            self.expert_actor_in_keys = []

        return model

    @staticmethod
    def _external_expert_module_from(expert_model):
        wrapped_module = getattr(expert_model, "module", None)
        if wrapped_module is not None and (
            callable(wrapped_module)
            or hasattr(wrapped_module, "_actor")
            or hasattr(wrapped_module, "actor")
        ):
            expert_model = wrapped_module
        return getattr(
            expert_model,
            "_actor",
            getattr(expert_model, "actor", expert_model),
        )

    def _external_expert_module(self):
        expert_actor = getattr(self, "expert_actor", None)
        if expert_actor is not None:
            return expert_actor
        return self._external_expert_module_from(self.expert_model)

    def _materialize_frozen_external_expert(self, expert_actor, dummy_obs_td):
        """Materialize a frozen expert without updating distributed normalizers."""
        freeze_states = []
        for module in expert_actor.modules():
            if hasattr(module, "_freeze_running"):
                freeze_states.append((module, module._freeze_running))
                module._freeze_running = True
        try:
            _ = expert_actor(dummy_obs_td)
        finally:
            for module, freeze_running in freeze_states:
                module._freeze_running = freeze_running

    def _load_external_expert_state(self, expert_model, model_state_dict):
        expert_actor = self._external_expert_module_from(expert_model)
        for prefix in ("_actor.", "actor."):
            actor_state_dict = {
                key[len(prefix) :]: value
                for key, value in model_state_dict.items()
                if key.startswith(prefix)
            }
            if actor_state_dict:
                expert_actor.load_state_dict(actor_state_dict)
                return

        expert_model.load_state_dict(model_state_dict)

    def _build_expert_obs_td(
        self, obs_td: TensorDict, expert_in_keys: list
    ) -> TensorDict:
        """Build expert observation TensorDict by stripping 'expert_' prefix from keys.

        The experiment file adds expert observation components with "expert_" prefix
        (e.g., "expert_max_coords_obs"). This method maps those back to the keys
        the expert model expects (e.g., "max_coords_obs").

        Args:
            obs_td: Full observation TensorDict with both student and expert_* keys
            expert_in_keys: List of keys the expert model expects

        Returns:
            TensorDict with keys matching expert model's in_keys
        """
        expert_obs = {}
        for key in expert_in_keys:
            expert_key = f"expert_{key}"
            if expert_key in obs_td.keys():
                # Prefer prefixed expert observation
                expert_obs[key] = obs_td[expert_key]
            else:
                raise KeyError(
                    f"Expert model requires observation '{expert_key}' for "
                    f"expert input '{key}'. Available keys: {list(obs_td.keys())}"
                )
        return TensorDict(expert_obs, batch_size=obs_td.batch_size, device=self.device)

    def create_optimizers(self, model: BaseModel):
        optimizer = instantiate_optimizer(
            self.config.model.optimizer,
            model.optimization_module(),
        )
        self.training_model, self.supervised_optimizer = self._setup_model_optimizer(
            model,
            optimizer,
        )

    # -----------------------------
    # Training Loop and Dataset Processing
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        if self.expert_model is not None:
            self.experience_buffer.register_key(
                "expert_actions",
                shape=(self.env.robot_config.number_of_actions,),
            )

    def register_algorithm_experience_buffer_keys_from_obs(self, obs_td: TensorDict):
        target_key = self.config.loss.target_key
        if hasattr(self.experience_buffer, target_key):
            return

        if target_key in obs_td.keys():
            value = obs_td[target_key]
        else:
            with self._eval_model_for_buffer_registration(), torch.no_grad():
                output_td = self._collect_rollout_output(obs_td.clone())
            if target_key not in output_td.keys():
                raise KeyError(
                    f"Supervised loss target_key '{target_key}' was not produced by "
                    f"the rollout output. Available keys: {list(output_td.keys())}"
                )
            value = output_td[target_key]

        self.experience_buffer.register_key(
            target_key,
            shape=value.shape[1:],
            dtype=value.dtype,
        )

    def _collect_external_expert_action(self, obs_td: TensorDict) -> torch.Tensor:
        expert_actor = self._external_expert_module()
        expert_in_keys = getattr(self, "expert_actor_in_keys", None)
        if not expert_in_keys:
            expert_in_keys = list(getattr(expert_actor, "in_keys", []))
        expert_obs_td = self._build_expert_obs_td(
            obs_td,
            expert_in_keys,
        )
        expert_output_td = expert_actor(expert_obs_td)
        if "mean_action" in expert_output_td.keys():
            return expert_output_td["mean_action"]
        if "action" in expert_output_td.keys():
            return expert_output_td["action"]
        raise KeyError(
            "External expert actor must produce either 'mean_action' or 'action'. "
            f"Available keys: {list(expert_output_td.keys())}"
        )

    def _collect_rollout_output(self, obs_td: TensorDict) -> TensorDict:
        rollout_actor = self.config.rollout_actor
        if rollout_actor not in (RolloutActor.STUDENT, RolloutActor.EXPERT):
            raise ValueError(f"Unsupported supervised rollout_actor: {rollout_actor}")

        has_external_expert = self.expert_model is not None
        if rollout_actor == RolloutActor.EXPERT and not has_external_expert:
            model_expert_rollout = getattr(
                self.model,
                "collect_expert_rollout",
                None,
            )
            if model_expert_rollout is None:
                raise ValueError(
                    "rollout_actor=EXPERT needs an expert source: set "
                    "expert_model_path for an external expert, or use a model "
                    "that defines collect_expert_rollout."
                )
            output_td = model_expert_rollout(obs_td)
        else:
            output_td = self.model(obs_td)

        if has_external_expert:
            expert_action = self._collect_external_expert_action(obs_td)
            output_td["expert_actions"] = expert_action
            if rollout_actor == RolloutActor.EXPERT:
                output_td["action"] = expert_action
                output_td["mean_action"] = expert_action

        return output_td

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect student action and expert label for the current state."""
        output_td = self._collect_rollout_output(obs_td)

        if self.config.rollout_actor == RolloutActor.EXPERT:
            action = output_td["action"]
        elif "privileged_action" in output_td:
            action = output_td[
                "privileged_action"
            ]  # During training, we use the privileged action
        else:
            action = output_td["action"]  # During evaluation, we use the action

        # Store model outputs
        output_keys = list(
            dict.fromkeys(list(self.model_output_keys) + [self.config.loss.target_key])
        )
        for key in output_keys:
            if key in output_td:
                self.experience_buffer.update_data(key, step, output_td[key])
            elif key not in obs_td.keys():
                raise KeyError(
                    f"Supervised rollout output did not contain required key '{key}'. "
                    f"Available keys: {list(output_td.keys())}"
                )

        if self.expert_model is not None and "expert_actions" not in output_keys:
            self.experience_buffer.update_data(
                "expert_actions", step, output_td["expert_actions"]
            )

        output_td["action"] = action
        return output_td

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        # Update model
        iter_log_dict = {}
        loss, loss_dict = self.supervised_step(batch_dict)
        iter_log_dict.update(loss_dict)
        grad_clip_dict = self._step_optimizer(
            loss=loss,
            model=self.training_model,
            optimizer=self.supervised_optimizer,
            model_name="model",
        )
        iter_log_dict.update(grad_clip_dict)

        return iter_log_dict

    # -----------------------------
    # Model Forward Pass and Loss Computation
    # -----------------------------
    def supervised_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Compute supervised imitation loss from a rollout batch."""
        # Convert to TensorDict and run model forward
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.training_model(batch_td)

        supervised_loss, supervised_log_dict = compute_supervision_loss(
            batch_td,
            self.config.loss,
        )
        actions = (
            batch_td["privileged_action"]
            if "privileged_action" in batch_td.keys()
            else batch_td["action"]
        )

        extra_loss, extra_log_dict = self.calculate_extra_loss(batch_td, actions)

        model_loss, model_log_dict = self.model.compute_model_loss(
            batch_td,
            current_epoch=self.current_epoch,
            zero_loss=supervised_loss,
            log_prefix="model",
        )

        loss = supervised_loss + extra_loss + model_loss

        log_dict = {
            "supervised/loss": supervised_loss.detach(),
            "supervised/extra_loss": extra_loss.detach(),
            "supervised/model_loss": model_loss.detach(),
            "losses/supervised_loss": loss.detach(),
        }
        log_dict.update(supervised_log_dict)
        log_dict.update(model_log_dict)
        log_dict.update(extra_log_dict)

        return loss, log_dict

    def calculate_extra_loss(self, batch_dict, actions) -> Tuple[Tensor, Dict]:
        l2c2_weight = self.config.l2c2_weight
        if l2c2_weight <= 0:
            return torch.tensor(0.0, device=self.device), {}

        l2c2_loss = self._calculate_l2c2_loss(batch_dict)
        return l2c2_weight * l2c2_loss, {
            "supervised/l2c2_loss": l2c2_loss.detach(),
        }

    def _calculate_l2c2_loss(self, batch_td: TensorDict) -> Tensor:
        """L2C2 Lipschitz-ratio regularizer ported from the PPO actor path."""
        obs_pairs = self.config.l2c2_obs_pairs
        if not obs_pairs:
            raise ValueError(
                "l2c2_weight > 0 requires at least one l2c2_obs_pairs entry."
            )

        prediction_key = self.config.loss.prediction_key
        if prediction_key not in batch_td.keys():
            raise KeyError(
                f"L2C2 prediction key '{prediction_key}' is missing. "
                f"Available keys: {list(batch_td.keys())}"
            )

        clean_td = batch_td.clone()
        prediction = batch_td[prediction_key]
        input_ss = prediction.new_zeros(())
        input_n = 0

        for noisy_key, clean_key in obs_pairs.items():
            if noisy_key not in batch_td.keys():
                raise KeyError(
                    f"L2C2 noisy observation key '{noisy_key}' is missing. "
                    f"Available keys: {list(batch_td.keys())}"
                )
            if clean_key not in batch_td.keys():
                raise KeyError(
                    f"L2C2 clean observation key '{clean_key}' is missing. "
                    f"Available keys: {list(batch_td.keys())}"
                )

            noisy_obs = batch_td[noisy_key]
            clean_obs = batch_td[clean_key]
            if noisy_obs.shape != clean_obs.shape:
                raise ValueError(
                    f"L2C2 observation pair '{noisy_key}'/'{clean_key}' has "
                    f"mismatched shapes: {tuple(noisy_obs.shape)} vs "
                    f"{tuple(clean_obs.shape)}"
                )

            clean_td[noisy_key] = clean_obs
            diff = noisy_obs - clean_obs
            input_ss = input_ss + diff.pow(2).sum()
            input_n += diff.numel()

        if input_n == 0:
            raise ValueError("l2c2_obs_pairs must reference non-empty tensors.")

        input_dist = (input_ss / input_n).detach()
        clean_td = self.training_model(clean_td)
        if prediction_key not in clean_td.keys():
            raise KeyError(
                f"L2C2 clean forward did not produce prediction key '{prediction_key}'. "
                f"Available keys: {list(clean_td.keys())}"
            )

        output_dist = (prediction - clean_td[prediction_key]).pow(2).mean()
        return output_dist / (input_dist + 1e-8)

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["supervised_optimizer"] = self.supervised_optimizer.state_dict()
        return state_dict

    def _load_training_state(self, state_dict):
        super()._load_training_state(state_dict)
        optimizer_state = state_dict.get(
            "supervised_optimizer",
            state_dict.get("maskedmimic_optimizer"),
        )
        if optimizer_state is None:
            raise KeyError("supervised_optimizer")
        self.supervised_optimizer.load_state_dict(optimizer_state)
