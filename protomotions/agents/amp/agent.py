# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import logging

from torch import Tensor
import math

from lightning.fabric import Fabric
from hydra.utils import instantiate

from protomotions.agents.utils.data_utils import swap_and_flatten01
from protomotions.utils.replay_buffer import ReplayBuffer
from protomotions.agents.amp.model import AMPModel
from protomotions.agents.common.common import weight_init
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.ppo.agent import PPO

log = logging.getLogger(__name__)


class AMP(PPO):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        super().__init__(fabric, env, config)
        self.amp_replay_buffer = ReplayBuffer(self.config.discriminator_replay_size).to(
            self.device
        )

    def setup(self):
        model: AMPModel = instantiate(self.config.model)
        model.apply(weight_init)
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )
        discriminator_optimizer = instantiate(
            self.config.model.config.discriminator_optimizer,
            params=list(model._discriminator.parameters()),
        )

        (
            self.model,
            self.actor_optimizer,
            self.critic_optimizer,
            self.discriminator_optimizer,
        ) = self.fabric.setup(
            model, actor_optimizer, critic_optimizer, discriminator_optimizer
        )
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        super().register_extra_experience_buffer_keys()
        self.experience_buffer.register_key("amp_rewards")

    def update_disc_replay_buffer(self, data_dict):
        buf_size = self.amp_replay_buffer.get_buffer_size()
        buf_total_count = len(self.amp_replay_buffer)

        values = list(data_dict.values())
        numel = values[0].shape[0]

        for i in range(1, len(values)):
            assert numel == values[i].shape[0]

        if buf_total_count > buf_size:
            keep_probs = (
                torch.ones(numel, device=self.device)
                * self.config.discriminator_replay_keep_prob
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            for k, v in data_dict.items():
                data_dict[k] = v[keep_mask]

        if numel > buf_size:
            rand_idx = torch.randperm(numel)
            rand_idx = rand_idx[:buf_size]
            for k, v in data_dict.items():
                data_dict[k] = v[rand_idx]

        self.amp_replay_buffer.store(data_dict)

    @torch.no_grad()
    def process_dataset(self, dataset):
        historical_self_obs = swap_and_flatten01(
            self.experience_buffer.historical_self_obs
        )

        num_samples = historical_self_obs.shape[0]

        if len(self.amp_replay_buffer) == 0:
            replay_historical_self_obs = historical_self_obs
        else:
            replay_dict = self.amp_replay_buffer.sample(num_samples)
            replay_historical_self_obs = replay_dict["historical_self_obs"]

        expert_historical_self_obs = self.get_expert_historical_self_obs(num_samples)

        discriminator_training_data_dict = {
            "agent_historical_self_obs": historical_self_obs,
            "replay_historical_self_obs": replay_historical_self_obs,
            "expert_historical_self_obs": expert_historical_self_obs,
        }

        dataset.update(discriminator_training_data_dict)

        self.update_disc_replay_buffer({"historical_self_obs": historical_self_obs})

        return super().process_dataset(dataset)

    def get_expert_historical_self_obs(self, num_samples: int):
        motion_ids = self.motion_lib.sample_motions(num_samples)
        num_steps = self.env.self_obs_cb.config.num_historical_steps

        dt = self.env.dt
        truncate_time = dt * (num_steps - 1)

        # Since negative times are added to these values in build_historical_self_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip].
        motion_times0 = self.motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 = motion_times0 + truncate_time

        obs = self.env.self_obs_cb.build_self_obs_demo(
            motion_ids, motion_times0, num_steps
        ).clone()

        return obs.view(num_samples, -1)

    # -----------------------------
    # Reward Calculation
    # -----------------------------
    @torch.no_grad()
    def calculate_extra_reward(self):
        rew = super().calculate_extra_reward()

        historical_self_obs = self.experience_buffer.historical_self_obs
        amp_r = self.model._discriminator.compute_reward(
            {
                "historical_self_obs": historical_self_obs.view(
                    self.num_envs * self.num_steps, -1
                )
            }
        ).view(self.num_steps, self.num_envs)

        self.experience_buffer.batch_update_data("amp_rewards", amp_r)

        extra_reward = amp_r * self.config.discriminator_reward_w + rew
        return extra_reward

    # -----------------------------
    # Optimization
    # -----------------------------
    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        extra_opt_steps_dict = super().extra_optimization_steps(batch_dict, batch_idx)
        if batch_idx == 0:
            self.discriminator_optimizer.zero_grad()

        if batch_idx < self.discriminator_max_num_batches():
            discriminator_loss, discriminator_loss_dict = self.discriminator_step(
                batch_dict
            )
            extra_opt_steps_dict.update(discriminator_loss_dict)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(discriminator_loss)
            discriminator_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._discriminator,
                self.discriminator_optimizer,
                "discriminator",
            )
            extra_opt_steps_dict.update(discriminator_grad_clip_dict)
            self.discriminator_optimizer.step()

        return extra_opt_steps_dict

    def discriminator_step(self, batch_dict):
        agent_obs = batch_dict["agent_historical_self_obs"][
            : self.config.discriminator_batch_size
        ]
        replay_obs = batch_dict["replay_historical_self_obs"][
            : self.config.discriminator_batch_size
        ]
        expert_obs = batch_dict["expert_historical_self_obs"][
            : self.config.discriminator_batch_size
        ]
        combined_obs = torch.cat([agent_obs, expert_obs], dim=0)
        combined_obs.requires_grad_(True)

        combined_dict = self.model._discriminator.compute_logits(
            {"historical_self_obs": combined_obs}, return_norm_obs=True
        )
        combined_logits = combined_dict["outs"]
        combined_norm_obs = combined_dict["norm_historical_self_obs"]

        replay_logits = self.model._discriminator.compute_logits(
            {"historical_self_obs": replay_obs}
        )

        agent_logits = combined_logits[: self.config.discriminator_batch_size]
        expert_logits = combined_logits[self.config.discriminator_batch_size :]

        expert_loss = -torch.nn.functional.logsigmoid(expert_logits).mean()
        unlabeled_loss = torch.nn.functional.softplus(agent_logits).mean()
        replay_loss = torch.nn.functional.softplus(replay_logits).mean()

        neg_loss = 0.5 * (unlabeled_loss + replay_loss)
        class_loss = 0.5 * (expert_loss + neg_loss)

        # Gradient penalty
        disc_grad = torch.autograd.grad(
            combined_logits,
            combined_norm_obs,
            grad_outputs=torch.ones_like(combined_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_grad_norm = torch.sum(torch.square(disc_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_grad_norm)
        grad_loss: Tensor = self.config.discriminator_grad_penalty * disc_grad_penalty

        if self.config.discriminator_weight_decay > 0:
            all_weight_params = self.model._discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = total * self.config.discriminator_weight_decay
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.discriminator_logit_weight_decay > 0:
            logit_params = self.model._discriminator.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])
            logit_weight_decay_loss: Tensor = (
                logit_total * self.config.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        with torch.no_grad():
            pos_acc = self.compute_pos_acc(expert_logits)
            agent_acc = self.compute_neg_acc(agent_logits)
            replay_acc = self.compute_neg_acc(replay_logits)
            neg_acc = 0.5 * (agent_acc + replay_acc)

            log_dict = {
                "losses/discriminator_loss": loss.detach(),
                "discriminator/pos_acc": pos_acc.detach(),
                "discriminator/agent_acc": agent_acc.detach(),
                "discriminator/replay_acc": replay_acc.detach(),
                "discriminator/neg_acc": neg_acc.detach(),
                "discriminator/grad_penalty": disc_grad_penalty.detach(),
                "discriminator/grad_loss": grad_loss.detach(),
                "discriminator/class_loss": class_loss.detach(),
                "discriminator/l2_logit_total": logit_total.detach(),
                "discriminator/l2_logit_loss": logit_weight_decay_loss.detach(),
                "discriminator/l2_total": total.detach(),
                "discriminator/l2_loss": weight_decay_loss.detach(),
                "discriminator/expert_logit_mean": expert_logits.detach().mean(),
                "discriminator/agent_logit_mean": agent_logits.detach().mean(),
                "discriminator/replay_logit_mean": replay_logits.detach().mean(),
            }
            log_dict["discriminator/negative_logit_mean"] = 0.5 * (
                log_dict["discriminator/agent_logit_mean"]
                + log_dict["discriminator/replay_logit_mean"]
            )

        return loss, log_dict

    # -----------------------------
    # Discriminator Metrics and Utility
    # -----------------------------
    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()

    def discriminator_max_num_batches(self):
        return math.ceil(
            self.num_envs
            * self.num_steps
            * self.config.num_discriminator_mini_epochs
            / self.config.batch_size
        )

    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def terminate_early(self):
        self._should_stop = True

    def post_epoch_logging(self, training_log_dict):
        training_log_dict["rewards/amp_rewards"] = self.experience_buffer.amp_rewards.mean()
        super().post_epoch_logging(training_log_dict)
