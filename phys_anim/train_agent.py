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

import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

has_robot_arg = False
backbone = None
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if "robot" in arg:
        has_robot_arg = True
    if "backbone" in arg:
        if not has_robot_arg:
            raise ValueError("+robot argument should be provided before +backbone")
        if "isaacgym" in arg.split("=")[-1]:
            import isaacgym  # noqa: F401

            backbone = "isaacgym"
        elif "isaacsim" in arg.split("=")[-1]:
            from isaacsim import SimulationApp
            from phys_anim.envs.base_interface.isaacsim_utils.experiences import (
                get_experience,
            )

            backbone = "isaacsim"

import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from utils.config_utils import *  # noqa: E402, F403
from utils.common import seeding

from phys_anim.agents.callbacks.slurm_autoresume import (
    SlurmAutoResume,
)  # noqa: E402

from phys_anim.agents.ppo import PPO  # noqa: E402


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    autoresume = SlurmAutoResume()
    id = autoresume.details.get("id")
    if (
        id is not None
        and "wandb" in config
        and OmegaConf.select(config, "wandb.wandb_id", default=None) is None
    ):
        config = OmegaConf.merge(config, OmegaConf.create({"wandb": {"wandb_id": id}}))

    torch.set_float32_matmul_precision("medium")

    fabric: Fabric = instantiate(config.fabric)
    fabric.launch()

    if config.seed is not None:
        rank = fabric.global_rank
        if rank is None:
            rank = 0
        fabric.seed_everything(config.seed + rank)
        seeding(config.seed + rank, torch_deterministic=config.torch_deterministic)

    if backbone == "isaacsim":
        experience = get_experience(config.headless, False)
        simulation_app = SimulationApp(
            {"headless": config.headless}, experience=experience
        )

    env = instantiate(config.env, device=fabric.device)

    algo: PPO = instantiate(config.algo, env=env, fabric=fabric)
    algo.setup()
    if config.auto_load_latest:
        latest_checkpoint = Path(fabric.loggers[0].root_dir) / "last.ckpt"
        if latest_checkpoint.exists():
            config.checkpoint = latest_checkpoint
    algo.load(config.checkpoint)

    save_dir = Path(fabric.loggers[0].log_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving config file to {save_dir}")
    with open(save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)

    algo.fit()


if __name__ == "__main__":
    main()
