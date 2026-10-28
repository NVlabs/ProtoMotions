# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised rollout imitation agent.

This agent collects simulator rollouts and optimizes a configured supervision
loss. Labels can come from observations, model-owned target encoders, or an
optional frozen external expert. It is not an offline dataset training loop.
"""

import torch
from torch import Tensor
from tensordict import TensorDict
import logging

from protomotions.utils.experiment_configs import load_experiment_configs_from_checkpoint
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
from protomotions.agents.supervised.expert_utils import (
    get_expert_actor_in_keys,
    validate_expert_sensor_contract,
    validate_expert_action_contract,
)

log = logging.getLogger(__name__)


def _require_deployable_expert(expert_actor) -> None:
    """Refuse an expert that does not speak the deployable-model contract.

    The distillation loop labels rollouts through the expert's deployable graph
    (``forward_deployable``) and reads the labelled action under the key the
    expert names (``deployable_action_key``). A plain ``nn.Module`` that lacks
    either name cannot be an expert directly; wrap it in a caller-written adapter
    that supplies both before handing it to this agent.
    """
    missing = [
        name
        for name in ("forward_deployable", "deployable_action_key")
        if not hasattr(expert_actor, name)
    ]
    if missing:
        raise TypeError(
            f"External expert {type(expert_actor).__name__} is missing "
            f"{', '.join(missing)}: the deployable-model contract requires both "
            "forward_deployable and deployable_action_key. Wrap a plain module in "
            "a caller-written adapter that provides them."
        )


class SupervisedAgent(BaseAgent):
    """Student/expert rollout agent for supervised distillation.

    The agent collects TensorDict rollouts, writes configured model outputs into
    the experience buffer, and optimizes ``SupervisionLossConfig``. Models and
    experiment files define which keys are predictions and labels.
    """

    model: BaseModel

    def _before_create_model(self) -> None:
        super()._before_create_model()
        self.expert_model = None
        if self.config.expert_model_path is not None:
            self._expert_episode_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._resample_episode_actor(
                torch.arange(self.num_envs, device=self.device)
            )

    def create_model(self):
        model_cls = get_class(self.config.model._target_)
        model: BaseModel = model_cls(config=self.config.model)
        if not getattr(model, "skip_default_weight_init", False):
            model.apply(weight_init_trainable)
        if self.config.expert_model_path is not None:
            self._load_external_expert(Path(self.config.expert_model_path))
        return model

    def _load_external_expert(self, checkpoint_path: Path) -> None:
        # Expert observation components are added by the experiment file with
        # an "expert_" prefix; see _build_expert_obs_td.
        log.info(f"Loading expert model from: {checkpoint_path}")
        resolved_configs = load_experiment_configs_from_checkpoint(checkpoint_path)
        validate_expert_action_contract(
            resolved_configs.robot,
            resolved_configs.env,
            self.env.robot_config,
            self.env.config,
        )
        expert_agent_config: PPOAgentConfig = resolved_configs.agent
        validate_expert_sensor_contract(
            resolved_configs.env, expert_agent_config, self.env.config.sensors
        )
        expert_model: BaseModel = get_class(expert_agent_config.model._target_)(
            config=expert_agent_config.model
        ).to(self.device)
        expert_model.reset_rollout_context(num_envs=self.num_envs, device=self.device)
        self._attach_fabric_to_running_mean_std(expert_model)

        # Actor-critic experts expose the actor as ``_actor``; a plain model is
        # its own actor. Only the actor labels actions, and materializing the
        # full model could require critic observations the distillation env
        # does not provide.
        expert_actor = getattr(expert_model, "_actor", expert_model)
        expert_actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
        _require_deployable_expert(expert_actor)
        expert_model.eval()
        with torch.no_grad():
            dummy_obs_td = self.obs_dict_to_tensordict(self.env.get_obs())
            self._forward_expert(
                expert_actor, self._build_expert_obs_td(dummy_obs_td, expert_actor_in_keys)
            )

        state_dict = torch.load(
            str(checkpoint_path), map_location=self.device, weights_only=False
        )["model"]
        if expert_actor is not expert_model:
            prefix = "_actor."
            state_dict = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        expert_actor.load_state_dict(state_dict)
        for param in expert_model.parameters():
            param.requires_grad = False

        # The expert stays a plain frozen module outside the DDP wrapper; it
        # only labels rollouts and needs no gradient synchronization.
        self.expert_model = expert_model
        self.expert_actor = expert_actor
        self.expert_actor_in_keys = expert_actor_in_keys

    @staticmethod
    def _forward_expert(actor, observations):
        """Label through the expert's deployable graph; no isinstance on the class."""
        return actor.forward_deployable(observations)

    def _build_expert_obs_td(
        self, obs_td: TensorDict, expert_in_keys: list
    ) -> TensorDict:
        """Map ``expert_<key>`` observations back to the keys the expert expects."""
        expert_obs = {}
        for key in expert_in_keys:
            expert_key = f"expert_{key}"
            if expert_key not in obs_td.keys():
                raise KeyError(
                    f"Expert model requires observation '{expert_key}' for "
                    f"expert input '{key}'. Available keys: {list(obs_td.keys())}"
                )
            expert_obs[key] = obs_td[expert_key]
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
        # Observation keys were registered by the base loop; only a model-produced
        # label still needs a buffer.
        target_key = self.config.loss.target_key
        if hasattr(self.experience_buffer, target_key):
            return
        if target_key == "expert_actions" and self.expert_model is None:
            # The one label no model produces: the external expert's. Say what
            # is missing instead of listing every rollout key below.
            raise ValueError(
                "the supervised loss targets 'expert_actions', the external "
                "expert's labels, but no expert is loaded: pass "
                "--expert-model-path (the agent config's expert_model_path) with "
                "the expert's checkpoint"
            )
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
        expert_output_td = self._forward_expert(
            self.expert_actor,
            self._build_expert_obs_td(obs_td, self.expert_actor_in_keys),
        )
        return expert_output_td[self.expert_actor.deployable_action_key]

    def _collect_rollout_output(self, obs_td: TensorDict) -> TensorDict:
        if self.config.rollout_actor is RolloutActor.EXPERT:
            output_td = self.model.collect_expert_rollout(obs_td)
        else:
            output_td = self.model(obs_td)
        if self.expert_model is not None:
            output_td["expert_actions"] = self._collect_external_expert_action(obs_td)
        return output_td

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect the executed action, model outputs and labels for this state."""
        output_td = self._collect_rollout_output(obs_td)

        if self.config.rollout_actor is RolloutActor.EXPERT:
            action = output_td["action"]
        else:
            # Student training executes the posterior prediction when available.
            action = output_td.get("privileged_action", output_td["action"])

        output_keys = list(self.model_output_keys) + [self.config.loss.target_key]
        if self.expert_model is not None:
            # Teacher-driven episodes execute the expert action; the stored
            # prediction keys stay the student's own outputs.
            action = torch.where(
                self._expert_episode_mask[:, None], output_td["expert_actions"], action
            )
            output_keys.append("expert_actions")
        output_td["action"] = action

        for key in dict.fromkeys(output_keys):
            if key in output_td:
                self.experience_buffer.update_data(key, step, output_td[key])
            elif key not in obs_td.keys():
                raise KeyError(
                    f"Supervised rollout output did not contain required key '{key}'. "
                    f"Available keys: {list(output_td.keys())}"
                )

        return output_td

    def _resample_episode_actor(self, env_ids):
        """Keep one rollout policy per episode, including across rollout chunks."""
        self._expert_episode_mask[env_ids] = (
            torch.rand(env_ids.numel(), device=self.device)
            < self.config.expert_rollout_probability
        )

    def post_env_step_modifications(self, dones, terminated, extras):
        dones, terminated, extras = super().post_env_step_modifications(
            dones, terminated, extras
        )
        if self.expert_model is not None:
            env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            self._resample_episode_actor(env_ids)
            self.expert_model.reset_rollout_context(env_ids=env_ids)
        return dones, terminated, extras

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        loss, iter_log_dict = self.supervised_step(batch_dict)
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
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.training_model(batch_td)

        supervised_loss, supervised_log_dict = compute_supervision_loss(
            batch_td,
            self.config.loss,
        )
        model_loss, model_log_dict = self.model.compute_model_loss(
            batch_td,
            current_epoch=self.current_epoch,
            zero_loss=supervised_loss,
            log_prefix="model",
        )
        loss = supervised_loss + model_loss

        log_dict = {
            "supervised/loss": supervised_loss.detach(),
            "supervised/model_loss": model_loss.detach(),
            "losses/supervised_loss": loss.detach(),
        }
        log_dict.update(supervised_log_dict)
        log_dict.update(model_log_dict)
        return loss, log_dict

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["supervised_optimizer"] = self.supervised_optimizer.state_dict()
        return state_dict

    def _load_training_state(self, state_dict):
        super()._load_training_state(state_dict)
        self.supervised_optimizer.load_state_dict(state_dict["supervised_optimizer"])
