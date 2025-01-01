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

from phys_anim.agents.mimic_vae import MimicVAE
from phys_anim.agents.models.actor import PPO_Actor
from phys_anim.agents.utils.data_utils import DictDataset
from phys_anim.envs.mimic.common import MimicHumanoid

from omegaconf import OmegaConf
from typing import Tuple, Dict

import torch
from torch import Tensor
from lightning.fabric import Fabric

from hydra.utils import instantiate
from pathlib import Path


class MimicVAEDagger(MimicVAE):
    def __init__(self, fabric: Fabric, env: MimicHumanoid, config):
        super().__init__(fabric, env, config)

        if self.config.dagger.gt_actor_path is not None:
            print(
                "Loading pre-trained full-body tracker from:",
                self.config.dagger.gt_actor_path,
            )

            # "score_based.ckpt" is the name of the file that is saved when a new high score is achieved
            checkpoint_path = self.config.dagger.gt_actor_path + "/score_based.ckpt"
            if not Path(checkpoint_path).exists():
                checkpoint_path = self.config.dagger.gt_actor_path + "/last.ckpt"
            pre_trained_expert = torch.load(checkpoint_path)

            self.gt_actor_config = OmegaConf.load(
                Path(self.config.dagger.gt_actor_path + "/last.ckpt").resolve().parent
                / "config.yaml"
            )
            self.gt_actor: PPO_Actor = instantiate(
                self.gt_actor_config.algo.config.actor,
                num_in=self.num_obs,
                num_act=self.num_act,
            ).to(self.device)

            self.gt_actor.load_state_dict(pre_trained_expert["actor"])
            for param in self.gt_actor.parameters():
                param.requires_grad = False
            self.gt_actor.eval()  # Just incase

            self.experience_buffer.register_key("gt_actions", shape=(self.num_act,))

    def pre_env_step(self, obs, step) -> Tensor:
        actor_outs = super().pre_env_step(obs, step)

        if self.config.dagger.gt_actor_path is not None:
            # collect ground truth actions from expert
            gt_actor_inputs = {}
            for key, value in obs.items():
                if key != "mimic_target_poses":
                    gt_actor_inputs[key] = value

            gt_actor_inputs["mimic_target_poses"] = (
                self.env.mimic_obs_cb.build_target_poses(
                    num_future_steps=self.gt_actor_config.env.config.mimic_target_pose.num_future_steps,
                    target_pose_type=self.gt_actor_config.env.config.mimic_target_pose.type,
                    with_time=self.gt_actor_config.env.config.mimic_target_pose.with_time,
                    env_ids=torch.arange(self.env.num_envs, device=self.device),
                )
            )

            gt_actor_outs = self.gt_actor.eval_forward(gt_actor_inputs)

            gt_actor_mus = gt_actor_outs["mus"]
            gt_actor_sampled_actions = gt_actor_outs["actions"]

            self.experience_buffer.update_data("gt_actions", step, gt_actor_mus)

            if self.config.dagger.collect_data_with_expert:
                actor_outs["actions"] = gt_actor_sampled_actions
            else:
                actor_outs["actions"] = actor_outs["mus"]

        return actor_outs

    def calculate_extra_actor_loss(self, batch_idx, batch_dict) -> Tuple[Tensor, Dict]:
        extra_loss, extra_actor_log_dict = super().calculate_extra_actor_loss(
            batch_idx, batch_dict
        )

        # add BC loss
        bc_loss = self.bc_loss(batch_dict)

        extra_actor_log_dict["actor/bc_loss"] = bc_loss.detach()

        return extra_loss + bc_loss * self.config.dagger.bc_coeff, extra_actor_log_dict

    def bc_loss(self, batch_dict):
        new_mu = self.actor.training_forward(batch_dict)["mus"]
        gt_actions = batch_dict["gt_actions"]
        bc_loss = torch.square(new_mu - gt_actions).mean()

        return bc_loss

    def training_step(self, batch_idx: int) -> Dict:
        if not self.config.dagger.only_bc:
            return super().training_step(batch_idx)

        iter_log_dict = {}

        if batch_idx == 0:
            self.actor_optimizer.zero_grad()

        is_accumulating = (
            ((batch_idx + 1) % self.config.gradient_accumulation_steps != 0)
            or self.config.gradient_accumulation_steps <= 0
        ) and ((batch_idx + 1) < self.ac_max_num_batches())
        num_accumulation_steps = min(
            self.config.gradient_accumulation_steps, self.ac_max_num_batches()
        )
        if num_accumulation_steps <= 0:
            num_accumulation_steps = self.ac_max_num_batches()

        if batch_idx < self.ac_max_num_batches():
            with self.fabric.no_backward_sync(self.actor, enabled=is_accumulating):
                actor_loss, actor_loss_dict = self.actor_step(batch_idx)
                scaled_actor_loss = actor_loss / num_accumulation_steps
                self.fabric.backward(scaled_actor_loss)

            if not is_accumulating:
                actor_grad_clip_dict = self.handle_actor_grad_clipping()
                iter_log_dict.update(actor_grad_clip_dict)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()
                self.actor.logstd_tick(self.current_epoch)

            iter_log_dict.update(actor_loss_dict)

        extra_opt_steps_dict = self.extra_optimization_steps(batch_idx)

        iter_log_dict.update(extra_opt_steps_dict)

        if batch_idx == (self.max_num_batches() - 1):
            for lr in self.lr_schedulers:
                lr.step()

        return iter_log_dict

    def generate_datasets(self):
        if not self.config.dagger.only_bc:
            return super().generate_datasets()

        actor_critic_data_dict = self.experience_buffer.make_dict()

        # Saves memory
        if hasattr(self, "actor_critic_dataset"):
            del self.actor_critic_dataset

        self.actor_critic_dataset = DictDataset(
            self.config.batch_size, actor_critic_data_dict, shuffle=True
        )

    def actor_step(self, batch_idx) -> Tuple[Tensor, Dict]:
        if not self.config.dagger.only_bc:
            return super().actor_step(batch_idx)

        dataset_idx = batch_idx % len(self.actor_critic_dataset)
        # Reshuffling the data at the beginning of each mini epoch.
        # Only doing this in the actor and not the critic to
        # avoid extra reshuffles.
        if dataset_idx == 0 and batch_idx != 0 and self.actor_critic_dataset.do_shuffle:
            self.actor_critic_dataset.shuffle()
        batch_dict = self.actor_critic_dataset[dataset_idx]

        return self.calculate_extra_actor_loss(batch_idx, batch_dict)
