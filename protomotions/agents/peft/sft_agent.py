# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT agent for PEFT adapters on a frozen discrete-token GPC prior."""

from pathlib import Path
from typing import Optional

from lightning.fabric import Fabric

from protomotions.agents.fine_tuning.pretrained_modules import PretrainedModulesMixin
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.supervised.agent import SupervisedAgent
from protomotions.agents.peft.prior_setup import DiscretePriorPEFTSetupMixin


# SFT uses SupervisedAgent directly, so it opts into the shared frozen-module
# lifecycle here. RLFT receives the same mixin through FineTuningAgent.
class DiscretePriorPEFTSFTAgent(
    DiscretePriorPEFTSetupMixin,
    PretrainedModulesMixin,
    SupervisedAgent,
):
    """Train a discrete-prior PEFT adapter with target-token supervision.

    The model owns the expert labeling path through the frozen target encoder;
    the generic supervised loop stores those labels and applies the configured
    supervised loss during optimization.
    """

    def __init__(self, fabric: Fabric, env, config, root_dir: Optional[Path] = None):
        if getattr(config.model, "critic", None) is not None:
            raise ValueError("DiscretePriorPEFTSFTAgent does not use a critic.")
        super().__init__(fabric, env, config, root_dir=root_dir)
        if getattr(config, "physics_guard_enabled", True):
            if env.simulator.__class__.__module__ == "protomotions.simulator.isaaclab.simulator":
                from protomotions.utils.physics_guard import PhysicsGuard
                env.simulator._physics_guard = PhysicsGuard(
                    env.simulator, env=env, folder=self.root_dir / "physics_guard",
                    joint_speed_limit=getattr(config, "physics_guard_joint_speed_limit", 1000.),
                    max_consecutive=getattr(config, "physics_guard_max_consecutive", 3))

    @property
    def has_critic(self):
        return False

    def create_model(self):
        # SupervisedAgent's external-expert slot is unused here. PEFT SFT gets
        # labels from the frozen target encoder inside DiscretePriorPEFTSFTModel.
        self.expert_model = None
        return super().create_model()

    def _should_build_target_encoder(self, mimic_target_poses_dim: int) -> bool:
        if mimic_target_poses_dim <= 0:
            raise ValueError(
                "DiscretePriorPEFTSFTAgent requires environment observations to include "
                "mimic_target_poses so the frozen target encoder can build "
                "supervision labels."
            )
        return True

    def create_optimizers(self, model):
        optimizer = instantiate_optimizer(
            self.config.model.actor_optimizer,
            model,
            params=self._actor_optimizer_params(model),
        )
        self.training_model, self.supervised_optimizer = self._setup_model_optimizer(
            model,
            optimizer,
        )

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        # RLFT warm-start reads the actor optimizer state from SFT checkpoints.
        state_dict["actor_optimizer"] = self.supervised_optimizer.state_dict()
        return state_dict

    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        import torch
        self.experience_buffer.register_key("physics_valid", dtype=torch.bool)

    def record_rollout_step(self, next_obs, actions, rewards, dones, terminated,
                            done_indices, extras, step):
        import torch
        valid = extras.get("raw/physics_valid", torch.ones_like(dones, dtype=torch.bool))
        self.experience_buffer.update_data("physics_valid", step, valid)
        # Exclude failed episodes from episode meters while still resetting their counters.
        super().record_rollout_step(next_obs, actions, rewards, dones, terminated,
                                   done_indices[valid[done_indices]], extras, step)
        # Invalid rows are never forwarded to the trainable model. Also sanitize
        # stored rows so generic buffer NaN checks cannot abort before filtering.
        bad = ~valid
        if bad.any():
            for key in self.experience_buffer.store_dict:
                if key not in ("physics_valid", "dones"):
                    getattr(self.experience_buffer, key)[step, bad] = 0

    def perform_optimization_step(self, batch_dict, batch_idx):
        if "physics_valid" not in batch_dict or getattr(self.env.simulator, "_physics_guard", None) is None:
            return super().perform_optimization_step(batch_dict, batch_idx)
        from protomotions.utils.guarded_sft import guarded_optimization_step
        return guarded_optimization_step(self, batch_dict)
