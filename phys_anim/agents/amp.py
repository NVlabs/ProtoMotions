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

from torch import Tensor
import math

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from phys_anim.agents.utils.data_utils import swap_and_flatten01
from phys_anim.agents.ppo import PPO, get_params
from phys_anim.utils.replay_buffer import ReplayBuffer
from phys_anim.utils.dataset import GeneralizedDataset
from phys_anim.agents.models.discriminator import JointDiscMLP
from phys_anim.envs.amp.common import DiscHumanoid


class AMP(PPO):
    env: DiscHumanoid

    def __init__(self, fabric: Fabric, env: DiscHumanoid, config):
        super().__init__(fabric, env, config)

        self.disable_discriminator = self.env.config.disable_discriminator
        if not self.disable_discriminator:
            self.discriminator_obs_size_per_step = (
                self.env.config.discriminator_obs_size_per_step
            )

            self.experience_buffer.register_key("discriminator_rewards")

            self.discriminator_replay_buffer = ReplayBuffer(
                self.config.discriminator_replay_size
            )

            self.discriminator_grad_norm_before_clip = 0.0
            self.discriminator_grad_norm_after_clip = 0.0

            self.discriminator_batch_size: int = self.config.discriminator_batch_size
            self.discriminator_obs_historical_steps: int = (
                self.env.config.discriminator_obs_historical_steps
            )

            self.experience_buffer.register_key(
                "discriminator_obs",
                shape=(
                    self.discriminator_obs_size_per_step
                    * self.discriminator_obs_historical_steps,
                ),
            )

    def setup(self):
        super().setup()

        if self.disable_discriminator:
            return

        discriminator: JointDiscMLP = instantiate(
            self.config.discriminator,
            num_in=self.discriminator_obs_size_per_step
            * self.env.config.discriminator_obs_historical_steps,
        )
        discriminator_optimizer = instantiate(
            self.config.discriminator_optimizer,
            params=discriminator.parameters(),
        )

        self.discriminator, self.discriminator_optimizer = self.fabric.setup(
            discriminator, discriminator_optimizer
        )

        if self.config.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler = instantiate(
                self.config.discriminator_lr_scheduler,
                optimizer=self.discriminator_optimizer,
            )
            self.lr_schedulers.append(self.discriminator_lr_scheduler)

    def eval(self):
        super().eval()
        if not self.disable_discriminator:
            self.discriminator.eval()

    def train(self):
        super().train()
        if not self.disable_discriminator:
            self.discriminator.train()

    def discriminator_forward(self, obs: Tensor, return_norm_obs=False) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_norm_obs=return_norm_obs)

    def max_num_batches(self):
        return max(super().max_num_batches(), self.discriminator_max_num_batches())

    def discriminator_max_num_batches(self):
        if self.disable_discriminator:
            return 0

        return math.ceil(
            self.num_envs
            * self.num_steps
            * self.config.num_discriminator_mini_epochs
            / self.discriminator_batch_size
        )

    def extra_optimization_steps(self, batch_idx: int):
        extra_opt_steps_dict = super().extra_optimization_steps(batch_idx)
        if not self.disable_discriminator:
            if batch_idx == 0:
                self.discriminator_optimizer.zero_grad()

            is_accumulating = (
                ((batch_idx + 1) % self.config.gradient_accumulation_steps != 0)
                or self.config.gradient_accumulation_steps <= 0
            ) and ((batch_idx + 1) < self.discriminator_max_num_batches())
            num_accumulation_steps = min(
                self.config.gradient_accumulation_steps,
                self.discriminator_max_num_batches(),
            )
            if num_accumulation_steps <= 0:
                num_accumulation_steps = self.discriminator_max_num_batches()

            if batch_idx < self.discriminator_max_num_batches():
                with self.fabric.no_backward_sync(
                    self.discriminator, enabled=is_accumulating
                ):
                    discriminator_loss, discriminator_loss_dict = (
                        self.discriminator_step(batch_idx)
                    )
                    extra_opt_steps_dict.update(discriminator_loss_dict)
                    scaled_discriminator_loss = (
                        discriminator_loss / num_accumulation_steps
                    )
                    self.fabric.backward(scaled_discriminator_loss)

                if not is_accumulating:
                    discriminator_grad_clip_dict = self.handle_discriminator_grad_clipping()
                    extra_opt_steps_dict.update(discriminator_grad_clip_dict)
                    self.discriminator_optimizer.step()
                    self.discriminator_optimizer.zero_grad()

        return extra_opt_steps_dict

    def handle_discriminator_grad_clipping(self):
        discriminator_params = get_params(list(self.discriminator.parameters()))
        discriminator_grad_norm_before_clip = torch_utils.grad_norm(discriminator_params)

        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(discriminator_grad_norm_before_clip)
                or discriminator_grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(discriminator_grad_norm_before_clip)

        # sanity check
        discriminator_bad_grads_count = 0
        if bad_grads:

            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in discriminator_params if p.grad is not None],
                    dim=0,
                )
                raise ValueError(
                    f"NaN gradient"
                    + f" {all_params.isfinite().logical_not().float().mean().item()}"
                    + f" {all_params.abs().min().item()}"
                    + f" {all_params.abs().max().item()}"
                    + f" {discriminator_grad_norm_before_clip.item()}"
                )
            else:
                discriminator_bad_grads_count = 1
                for p in discriminator_params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            self.fabric.clip_gradients(
                self.discriminator,
                self.discriminator_optimizer,
                max_norm=self.config.gradient_clip_val,
                error_if_nonfinite=False,
            )
        discriminator_grad_norm_after_clip = torch_utils.grad_norm(discriminator_params)

        clip_dict = {
            "jd/grad_norm_before_clip": discriminator_grad_norm_before_clip.detach(),
            "jd/grad_norm_after_clip": discriminator_grad_norm_after_clip.detach(),
            "jd/bad_grads_count": discriminator_bad_grads_count,
        }

        return clip_dict

    def handle_reset(self, actor_state):
        actor_state = super().handle_reset(actor_state)
        if not self.disable_discriminator:
            actor_state["discriminator_obs"] = self.env.make_disc_obs().view(
                self.num_envs, -1
            )

        return actor_state

    def post_env_step(self, actor_state):
        actor_state = super().post_env_step(actor_state)

        if not self.disable_discriminator:

            actor_state["discriminator_obs"] = actor_state["extras"]["disc_obs"]
            self.experience_buffer.update_data(
                "discriminator_obs",
                actor_state["step"],
                actor_state["discriminator_obs"],
            )

        return actor_state

    def calculate_discriminator_reward(self, discriminator_obs: Tensor) -> Tensor:
        disc_logits = self.discriminator_forward(discriminator_obs)

        prob = 1 / (1 + torch.exp(-disc_logits))
        disc_r = (
            -torch.log(
                torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device))
            )
            * self.config.discriminator_reward_w
        )
        return disc_r

    def calculate_extra_reward(self):
        rew = super().calculate_extra_reward()

        if self.disable_discriminator:
            return rew

        discriminator_obs = self.experience_buffer.discriminator_obs
        disc_r = self.calculate_discriminator_reward(
            discriminator_obs.view(self.num_envs * self.num_steps, -1)
        ).view(self.num_steps, self.num_envs)

        self.experience_buffer.batch_update_data("discriminator_rewards", disc_r)

        extra_reward = disc_r + rew
        return extra_reward

    def create_actor_state(self):
        state = super().create_actor_state()
        if not self.disable_discriminator:
            state["discriminator_obs"] = self.env.make_disc_obs().view(
                self.num_envs, -1
            )

        return state

    @torch.no_grad()
    def generate_datasets(self):
        super().generate_datasets()

        if self.disable_discriminator:
            return

        agent_obs = swap_and_flatten01(self.experience_buffer.discriminator_obs)

        agent_dict = {"discriminator_obs": agent_obs}

        num_agent = agent_obs.shape[0]

        if self.discriminator_replay_buffer.get_total_count() == 0:
            replay_obs = agent_obs
        else:
            replay_dict = self.discriminator_replay_buffer.sample(num_agent)
            replay_obs = replay_dict["discriminator_obs"]

        motion_ids = self.motion_lib.sample_motions(num_agent)
        demo_obs = self.make_joint_disc_obs(motion_ids)

        # Saves memory
        if hasattr(self, "discriminator_dataset"):
            del self.discriminator_dataset

        self.discriminator_dataset = GeneralizedDataset(
            self.discriminator_batch_size,
            agent_obs,
            replay_obs,
            demo_obs,
            shuffle=True,
        )

        self.update_replay_buffer(agent_dict)

    def make_joint_disc_obs(self, motion_ids: Tensor):
        num_ids = motion_ids.shape[0]
        dt = self.env.dt

        truncate_time = dt * (self.discriminator_obs_historical_steps - 1)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        motion_times0 = self.motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 = motion_times0 + truncate_time
        obs = self.env.build_disc_obs_demo(motion_ids, motion_times0)

        return obs.view(num_ids, -1)

    def update_replay_buffer(self, data_dict):
        buf_size = self.discriminator_replay_buffer.get_buffer_size()
        buf_total_count = self.discriminator_replay_buffer.get_total_count()

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

        self.discriminator_replay_buffer.store(data_dict)

    def compute_discriminator_loss(self, dataset_idx: int):
        (
            agent_obs,
            replay_obs,
            demo_obs,
        ) = self.discriminator_dataset[dataset_idx]

        demo_obs.requires_grad_(True)

        agent_logits = self.discriminator_forward(obs=agent_obs)
        replay_logits = self.discriminator_forward(obs=replay_obs)

        demo_dict = self.discriminator_forward(obs=demo_obs, return_norm_obs=True)
        demo_logits = demo_dict["outs"]
        demo_norm_obs = demo_dict["norm_obs"]

        pos_loss = self.disc_loss_pos(demo_logits)
        agent_loss = self.disc_loss_neg(agent_logits)
        replay_loss = self.disc_loss_neg(replay_logits)

        neg_loss = 0.5 * (agent_loss + replay_loss)

        class_loss = 0.5 * (pos_loss + neg_loss)

        pos_acc = self.compute_pos_acc(demo_logits)
        agent_acc = self.compute_neg_acc(agent_logits)
        replay_acc = self.compute_neg_acc(replay_logits)

        neg_acc = 0.5 * (agent_acc + replay_acc)

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            demo_logits,
            demo_norm_obs,
            grad_outputs=torch.ones_like(demo_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_demo_grad_norm = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad_norm)
        grad_loss: Tensor = self.config.discriminator_grad_penalty * disc_grad_penalty

        if self.config.discriminator_weight_decay > 0:
            all_weight_params = self.discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = total * self.config.discriminator_weight_decay
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.discriminator_logit_weight_decay > 0:
            logit_params = self.discriminator.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])

            logit_weight_decay_loss: Tensor = (
                logit_total * self.config.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        log_dict = {
            "loss": loss.detach(),
            "pos_acc": pos_acc.detach(),
            "agent_acc": agent_acc.detach(),
            "replay_acc": replay_acc.detach(),
            "neg_acc": neg_acc.detach(),
            "grad_penalty": disc_grad_penalty.detach(),
            "grad_loss": grad_loss.detach(),
            "class_loss": class_loss.detach(),
            "l2_logit_total": logit_total.detach(),
            "l2_logit_loss": logit_weight_decay_loss.detach(),
            "l2_total": total.detach(),
            "l2_loss": weight_decay_loss.detach(),
            "demo_logit_mean": demo_logits.detach().mean(),
            "agent_logit_mean": agent_logits.detach().mean(),
            "replay_logit_mean": replay_logits.detach().mean(),
        }

        log_dict["negative_logit_mean"] = 0.5 * (
            log_dict["agent_logit_mean"] + log_dict["replay_logit_mean"]
        )

        return loss, log_dict

    @staticmethod
    def disc_loss_neg(disc_logits) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc_logits, torch.zeros_like(disc_logits)
        )
        return loss

    @staticmethod
    def disc_loss_pos(disc_logits) -> Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            disc_logits, torch.ones_like(disc_logits)
        )
        return loss

    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()

    def discriminator_step(self, batch_idx: int):
        dataset_idx = batch_idx % len(self.discriminator_dataset)
        # Reshuffle at the beginning of every mini epoch
        if (
            dataset_idx == 0
            and batch_idx != 0
            and self.discriminator_dataset.do_shuffle
        ):
            self.discriminator_dataset.shuffle()

        discriminator_loss, discriminator_log_dict = self.compute_discriminator_loss(
            dataset_idx
        )

        discriminator_log_dict = {
            f"jd/{k}": v for k, v in discriminator_log_dict.items()
        }

        return discriminator_loss, discriminator_log_dict

    def post_epoch_logging(self, training_log_dict):
        if not self.disable_discriminator:
            training_log_dict["jd/grad_norm_before_clip"] = (
                self.discriminator_grad_norm_before_clip
            )
            training_log_dict["jd/grad_norm_after_clip"] = (
                self.discriminator_grad_norm_after_clip
            )
            training_log_dict["rewards/discriminator_rewards"] = (
                self.experience_buffer.discriminator_rewards.mean()
            )

        training_log_dict["actor/logstd_min"] = self.actor.logstd.min().detach()
        training_log_dict["actor/logstd_max"] = self.actor.logstd.max().detach()
        training_log_dict["actor/logstd_mean"] = self.actor.logstd.mean().detach()

        super().post_epoch_logging(training_log_dict)

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        if not self.disable_discriminator:
            state_dict["discriminator"] = self.discriminator.state_dict()
            state_dict["discriminator_optimizer"] = (
                self.discriminator_optimizer.state_dict()
            )
            if self.config.discriminator_lr_scheduler is not None:
                state_dict["discriminator_lr_scheduler"] = (
                    self.discriminator_lr_scheduler.state_dict()
                )

        return state_dict

    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        if not self.disable_discriminator:
            self.discriminator.load_state_dict(state_dict["discriminator"])
            self.discriminator_optimizer.load_state_dict(
                state_dict["discriminator_optimizer"]
            )
            if self.config.discriminator_lr_scheduler is not None:
                self.discriminator_lr_scheduler.load_state_dict(
                    state_dict["discriminator_lr_scheduler"]
                )
