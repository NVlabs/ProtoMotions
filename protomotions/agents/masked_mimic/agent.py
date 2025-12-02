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
"""MaskedMimic agent implementation for versatile motion control.

This module implements the MaskedMimic algorithm which learns to reconstruct expert
tracker actions from partial observations. It trains on data from a full-body motion
tracker while randomly masking observations, enabling the agent to handle diverse
control tasks from incomplete information.

Training Process:
    1. Phase 1: Train expert full-body tracker (separate)
    2. Phase 2: Train MaskedMimic to imitate expert with masked observations

Key Classes:
    - MaskedMimic: Main MaskedMimic agent class extending BaseAgent

References:
    Tessler et al. "MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting" (2024)
"""

import torch
from torch import Tensor
from tensordict import TensorDict
import logging

from lightning.fabric import Fabric
from protomotions.utils.hydra_replacement import get_class, instantiate
from typing import Tuple, Dict, Optional
from pathlib import Path

from protomotions.envs.mimic.env import Mimic as MimicEnv
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.masked_mimic.model import MaskedMimicModel
from protomotions.agents.masked_mimic.config import MaskedMimicAgentConfig, VaeNoiseType
from protomotions.agents.common.common import weight_init
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.utils.training import handle_model_grad_clipping
from protomotions.agents.utils.normalization import RunningMeanStd

log = logging.getLogger(__name__)


class MaskedMimic(BaseAgent):
    """MaskedMimic agent for versatile motion control.

    Learns to reconstruct expert tracker actions from partial observations by
    training on data from a full-body motion tracker. The agent uses masked
    observations (randomly occluded body parts or features) and learns to infer
    the complete action from incomplete information. This enables versatile control
    where the agent can respond to various types of motion commands.

    Training process:
    1. **Phase 1**: Train expert full-body tracker (separate training)
    2. **Phase 2**: Train MaskedMimic to imitate expert with masked observations

    Key features:
    - **Masked observations**: Randomly masks input features during training
    - **Action reconstruction**: Learns to predict expert tracker actions
    - **Optional VAE**: Can use VAE latent codes for additional control
    - **Versatile control**: Single policy handles diverse motion tasks

    Args:
        fabric: Lightning Fabric instance for distributed training.
        env: Mimic environment for motion tracking.
        config: MaskedMimic configuration including expert model path and masking parameters.
        root_dir: Optional root directory for saving outputs.

    Attributes:
        expert_model: Pre-trained full-body tracker model (loaded from config).
        vae_noise: VAE latent codes for each environment (if using VAE).

    Example:
        >>> fabric = Fabric(devices=4)
        >>> env = Mimic(config, robot_config, simulator_config, device)
        >>> config.expert_model_path = "results/expert_tracker/"
        >>> agent = MaskedMimic(fabric, env, config)
        >>> agent.setup()
        >>> agent.train()

    Note:
        Requires pre-trained expert tracker model specified in config.expert_model_path.
    """

    env: MimicEnv
    model: MaskedMimicModel
    config: MaskedMimicAgentConfig

    def __init__(
        self,
        fabric: Fabric,
        env: MimicEnv,
        config: MaskedMimicAgentConfig,
        root_dir: Optional[Path] = None,
    ):
        super().__init__(fabric, env, config, root_dir=root_dir)

    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    config: MaskedMimicAgentConfig

    def setup(self):
        # Initialize VAE noise for each environment.
        # Create vae_noise tensor before super().setup() to ensure it can be used to initialize the lazy linear layers in the model.
        if self.config.model.vae is not None:
            self.vae_noise = torch.zeros(
                self.num_envs,
                self.config.model.vae.vae_latent_dim,
                dtype=torch.float,
                device=self.device,
            )
        super().setup()

    def create_model(self):
        MaskedMimicModelConfig = get_class(self.config.model._target_)
        model: MaskedMimicModel = MaskedMimicModelConfig(config=self.config.model)
        model.apply(weight_init)

        # Optionally load a pre-trained expert model if provided.
        if self.config.expert_model_path is not None:
            print(
                "Loading pre-trained full-body tracker from:",
                self.config.expert_model_path,
            )
            # "score_based.ckpt" is the name of the file that is saved when a new high score is achieved
            checkpoint_path = Path(self.config.expert_model_path)
            assert Path(
                checkpoint_path
            ).exists(), f"Could not find expert model at {checkpoint_path}"

            # Load frozen configs from resolved_configs.pt
            expert_model_dir = checkpoint_path.parent
            resolved_configs_path = expert_model_dir / "resolved_configs.pt"
            assert (
                resolved_configs_path.exists()
            ), f"Could not find resolved configs at {resolved_configs_path}"

            log.info(f"Loading expert configs from {resolved_configs_path}")
            resolved_configs = torch.load(
                resolved_configs_path, map_location="cpu", weights_only=False
            )

            self.expert_env_config: MimicEnvConfig = resolved_configs["env"]
            expert_agent_config: PPOAgentConfig = resolved_configs["agent"]

            # Verify the expert was trained with a compatible environment
            assert (
                self.env.mimic_obs_cb.config.mimic_target_pose.num_future_steps
                == self.expert_env_config.mimic_obs.mimic_target_pose.num_future_steps
            )
            assert (
                self.env.mimic_obs_cb.config.mimic_target_pose.type
                == self.expert_env_config.mimic_obs.mimic_target_pose.type
            )
            assert (
                self.env.mimic_obs_cb.config.mimic_target_pose.with_time
                == self.expert_env_config.mimic_obs.mimic_target_pose.with_time
            )

            # Create the expert model
            ExpertModelConfig = get_class(expert_agent_config.model._target_)
            expert_model: BaseModel = ExpertModelConfig(
                config=expert_agent_config.model
            )

            # Move model to device BEFORE materializing lazy modules
            expert_model = expert_model.to(self.device)

            # Once model is created, we pass fabric to the RunningMeanStd modules.
            # This allows the modules to internally handle distributed aggregation of normalization moments.
            def pass_fabric_to_running_mean_std(module):
                if isinstance(module, RunningMeanStd):
                    module.fabric = self.fabric

            expert_model.apply(pass_fabric_to_running_mean_std)

            log.info("Materializing expert model lazy modules...")
            with torch.no_grad():
                dummy_obs = self.env.get_obs()
                dummy_obs = self.add_agent_info_to_obs(dummy_obs)
                dummy_obs_td = self.obs_dict_to_tensordict(dummy_obs)
                _ = expert_model(dummy_obs_td)

            self.expert_model = self.fabric.setup(expert_model)

            # loading should be done after fabric.setup to ensure the model is on the correct fabric.device
            pre_trained_expert = torch.load(
                str(checkpoint_path),
                map_location=self.fabric.device,
                weights_only=False,
            )
            self.expert_model.load_state_dict(pre_trained_expert["model"])
            for param in self.expert_model.parameters():
                param.requires_grad = False
            self.expert_model.eval()  # Just incase
        else:
            self.expert_model = None

        return model

    def create_optimizers(self, model: MaskedMimicModel):
        optimizer = instantiate(
            self.config.model.optimizer,
            params=list(model.parameters()),
        )
        self.model, self.maskedmimic_optimizer = self.fabric.setup(model, optimizer)

    # -----------------------------
    # VAE Noise Management
    # -----------------------------
    def reset_vae_noise(self, env_ids):
        """Reset the VAE noise tensor based on the selected noise type."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if type(env_ids) is list:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        noise_type = self.config.model.vae.vae_noise_type
        if noise_type == VaeNoiseType.NORMAL:
            epsilon = torch.randn(
                env_ids.shape[0],
                self.model.config.vae.vae_latent_dim,
                device=self.device,
            )  # sampling epsilon
        elif noise_type == VaeNoiseType.UNIFORM:
            epsilon = torch.rand(
                env_ids.shape[0],
                self.model.config.vae.vae_latent_dim,
                device=self.device,
            )  # sampling epsilon
        elif noise_type == VaeNoiseType.ZEROS:
            epsilon = torch.zeros(
                env_ids.shape[0],
                self.model.config.vae.vae_latent_dim,
                device=self.device,
            )  # no noise
        else:
            raise NotImplementedError
        self.vae_noise[env_ids] = epsilon

    # -----------------------------
    # Environment Step and Reset Handling
    # -----------------------------
    def post_env_step_modifications(self, dones, terminated, extras):
        dones, terminated, extras = super().post_env_step_modifications(
            dones, terminated, extras
        )
        if self.model.config.vae is not None:
            self.reset_vae_noise(dones.nonzero(as_tuple=False).squeeze(-1))
        return dones, terminated, extras

    def add_agent_info_to_obs(self, obs):
        """Add agent-specific observations to the environment observations."""
        if self.config.model.vae is not None:
            obs["vae_noise"] = self.vae_noise.clone()
        return obs

    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        self.maskedmimic_optimizer.load_state_dict(state_dict["maskedmimic_optimizer"])

    # -----------------------------
    # Training Loop and Dataset Processing
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        # MaskedMimic-specific keys (action, mean_action, prior_mu, etc. auto-registered from model)
        self.experience_buffer.register_key(
            "expert_actions", shape=(self.env.robot_config.number_of_actions,)
        )

    def collect_rollout_step(self, obs_td: TensorDict, step):
        """Collect MaskedMimic-specific data: policy actions and expert actions."""
        # Note: vae_noise already added to obs by add_agent_info_to_obs

        # Convert to TensorDict and run student model (with encoder)
        output_td = self.model(obs_td)

        # Get student action
        if "privileged_action" in output_td:
            action = output_td[
                "privileged_action"
            ]  # During training, we use the privileged action
        else:
            action = output_td["action"]  # During evaluation, we use the action

        # Run expert model to get target action
        expert_output_td = self.expert_model(obs_td)
        if "mean_action" in expert_output_td:
            expert_action = expert_output_td[
                "mean_action"
            ]  # Use deterministic expert action
        else:
            expert_action = expert_output_td["action"]

        # Store model outputs
        for key in self.model_output_keys:
            if key in output_td:
                self.experience_buffer.update_data(key, step, output_td[key])

        # Store expert action
        self.experience_buffer.update_data("expert_actions", step, expert_action)

        return action

    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        # Update model
        iter_log_dict = {}
        loss, loss_dict = self.masked_mimic_step(batch_dict)
        iter_log_dict.update(loss_dict)
        self.maskedmimic_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(loss)
        grad_clip_dict = handle_model_grad_clipping(
            config=self.config,
            fabric=self.fabric,
            model=self.model,
            optimizer=self.maskedmimic_optimizer,
            model_name="model",
        )
        iter_log_dict.update(grad_clip_dict)
        self.maskedmimic_optimizer.step()

        return iter_log_dict

    # -----------------------------
    # Model Forward Pass and Loss Computation
    # -----------------------------
    def masked_mimic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Compute MaskedMimic loss from batch."""
        # Convert to TensorDict and run model forward
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.model(batch_td)

        # Extract outputs
        actions = batch_td["privileged_action"]
        expert_actions = batch_dict["expert_actions"]

        # Behavioral cloning loss
        bc_loss = torch.square(actions - expert_actions).mean()

        extra_loss, extra_log_dict = self.calculate_extra_loss(batch_dict, actions)

        # KL divergence loss (if using VAE)
        if hasattr(self.config.model, "vae"):
            vae_kld_schedule = self.config.model.vae.kld_schedule

            if vae_kld_schedule is not None:
                vae_kld_loss = self.model.kl_loss(batch_td)
                vae_kld_loss = torch.mean(torch.sum(vae_kld_loss, dim=-1))

                kld_coeff = vae_kld_schedule.init_kld_coeff + min(
                    max(0, self.current_epoch - vae_kld_schedule.start_epoch)
                    / (vae_kld_schedule.end_epoch - vae_kld_schedule.start_epoch),
                    1,
                ) * (vae_kld_schedule.end_kld_coeff - vae_kld_schedule.init_kld_coeff)

                vae_kld_loss = vae_kld_loss * kld_coeff
            else:
                vae_kld_loss = 0.0
        else:
            vae_kld_loss = 0.0

        loss = bc_loss + extra_loss + vae_kld_loss

        log_dict = {
            "masked_mimic/bc_loss": bc_loss.detach(),
            "masked_mimic/extra_loss": extra_loss.detach(),
            "losses/masked_mimic_loss": loss.detach(),
        }
        if hasattr(self.config.model, "vae"):
            log_dict["masked_mimic/vae_kld_loss"] = (
                vae_kld_loss.detach()
                if isinstance(vae_kld_loss, torch.Tensor)
                else torch.tensor(vae_kld_loss)
            )
            if vae_kld_schedule is not None:
                log_dict["masked_mimic/kld_coeff"] = kld_coeff

        log_dict.update(extra_log_dict)

        return loss, log_dict

    def calculate_extra_loss(self, batch_dict, actions) -> Tuple[Tensor, Dict]:
        return torch.tensor(0.0, device=self.device), {}

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        extra_state_dict = {
            "maskedmimic_optimizer": self.maskedmimic_optimizer.state_dict(),
        }
        state_dict.update(extra_state_dict)
        return state_dict
