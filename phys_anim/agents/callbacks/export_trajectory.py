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

from phys_anim.agents.callbacks.base_callback import RL_EvalCallback
from phys_anim.agents.ppo import PPO
from phys_anim.envs.mimic.isaacgym import MimicHumanoid
from phys_anim.agents.models.actor import ActorFixedSigma

import torch
from torch import Tensor
from pathlib import Path

import yaml


def parse_motions(mfile: str):
    if mfile.endswith(".yaml"):
        with open(mfile) as file:
            motions = yaml.load(file, yaml.CLoader)["motions"]
    elif mfile.endswith(".npy"):
        motions = [{"name": Path(mfile).name.split(".")[0]}]
    else:
        raise ValueError(f"Can't handle motion file '{mfile}'")

    return motions


class ExportTrajectory(RL_EvalCallback):
    training_loop: PPO
    env: MimicHumanoid
    actor: ActorFixedSigma

    def __init__(self, config, training_loop: PPO):
        super().__init__(config, training_loop)
        self.num_export_envs = self.config.num_export_envs
        self.trajectories = []

        mfile: str = self.config.motion_file
        self.motions = parse_motions(mfile)

        self.did_reset = torch.zeros(
            self.num_export_envs, dtype=torch.long, device=self.device
        )

        self.reset_buffers()

    def reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.num_export_envs)]
        self.act_buf = [[] for _ in range(self.num_export_envs)]
        self.motion_ids_buf = []
        self.did_reset[:] = 0

    def on_pre_evaluate_policy(self):
        # Doing this in two lines because of type annotation issues.
        env: MimicHumanoid = self.training_loop.env
        self.env = env

        actor: ActorFixedSigma = self.training_loop.actor
        self.actor = actor

        # self.env.disable_reset = True
        # self.env.disable_reset_track = True

        self.reset_all_envs()

        if self.config.override_logstd is not None:
            self.actor.logstd[:] = self.config.override_logstd

    def reset_all_envs(self):
        all_ids = torch.arange(
            0, self.training_loop.num_envs, device=self.device, dtype=torch.long
        )

        if self.config.balance_motion_ids:
            motion_ids = torch.fmod(all_ids, self.env.motion_lib.num_sub_motions())
        else:
            motion_ids = None

        self.env.reset_track(all_ids, new_motion_ids=motion_ids)
        self.env.reset(all_ids)

        self.motion_ids_buf = self.env.motion_ids[: self.num_export_envs].cpu().tolist()

    def on_pre_eval_env_step(self, actor_state):
        obs: Tensor = actor_state["obs"].cpu()
        target_actions: Tensor = actor_state["mus"]

        if self.env.config.residual_control:
            target_actions = self.env.residual_actions_to_actual(
                target_actions,
                target_ids=self.env.motion_ids,
                target_times=self.env.motion_times,
            )

        target_actions = target_actions.cpu()

        for i in range(self.num_export_envs):
            self.obs_buf[i].append(obs[i])
            self.act_buf[i].append(target_actions[i])

        # Override default of using deterministic rollout.
        actor_state["actions"] = actor_state["sampled_actions"]
        return actor_state

    def package_trajectories(self):
        for i in range(self.num_export_envs):
            if self.did_reset[i]:
                continue

            obs_stack = torch.stack(self.obs_buf[i], dim=0)
            act_stack = torch.stack(self.act_buf[i], dim=0)
            motion_id = self.motion_ids_buf[i]

            # if self.config.override_motion_id is not None:
            #     motion_id = self.config.override_motion_id

            trajectory = {
                "obs": obs_stack,
                "act": act_stack,
                "motion_name": self.motions[motion_id]["name"],
            }
            self.trajectories.append(trajectory)

    def on_post_eval_env_step(self, actor_state):
        done_indices = actor_state["done_indices"]
        self.did_reset[done_indices] = 1

        step = actor_state["step"]
        if (step + 1) % self.config.trajectory_length == 0:
            print("Resetting envs")
            self.package_trajectories()

            if len(self.trajectories) >= self.config.num_trajectories:
                # This is a bit hacky, should make it more robust
                self.on_post_evaluate_policy()
                exit(0)
            else:
                print(
                    f"Not enough trajectories (currently at {len(self.trajectories)}/"
                    f"{self.config.num_trajectories}), resetting envs"
                )
                self.reset_buffers()
                self.reset_all_envs()

        return actor_state

    def on_post_evaluate_policy(self):
        export_dir = Path(self.config.export_dir)
        export_dir.mkdir(exist_ok=True, parents=True)

        assert (
            len(self.trajectories) >= self.config.num_trajectories
        ), f"Not enough trajectories (only {len(self.trajectories)}/{self.config.num_trajectories})"

        to_export = self.trajectories[: self.config.num_trajectories]

        width = 4
        for i, traj in enumerate(to_export):
            num_string = str(i).zfill(width)
            torch.save(traj, export_dir / f"{num_string}.pt")
