# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The minimum agent, model and fabric the real evaluators need.

``BaseEvaluator``, ``MimicEvaluator`` and ``MaskedMimicEvaluator`` reach into
``self.agent.*`` and ``self.fabric.*``; these stubs provide exactly that
surface so the harness can call ``evaluator.evaluate()`` unchanged instead of
replaying a hand-written copy of its steps.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict

from protomotions.tests.parity.scenarios import BuiltEnv
from protomotions.tests.parity.spies import reference_motion
from protomotions.utils.hydra_replacement import get_class


class StubModel(torch.nn.Module):
    """Deterministic policy drawing from its own generator, never the global one.

    ``forward`` returns the three action keys the evaluators may read
    (``action``, ``mean_action``, ``privileged_action``), all the same tensor.
    """

    deployable_action_key = "mean_action"

    def __init__(
        self,
        num_envs: int,
        num_actions: int,
        *,
        seed: int = 4321,
        scale: float = 0.4,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.scale = scale
        self.output_device = torch.device("cpu") if device is None else device
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, obs_td) -> Dict[str, torch.Tensor]:
        action = self.scale * torch.randn(
            self.num_envs, self.num_actions, generator=self.generator
        )
        action = action.to(self.output_device)
        return {"action": action, "mean_action": action, "privileged_action": action}

    forward_deployable = forward

    def save_rollout_context(self) -> dict:
        return {}

    def load_rollout_context(self, state) -> None:
        pass

    def reset_rollout_context(self, env_ids=None, num_envs=None, device=None) -> None:
        pass


class StubFabric:
    """What the evaluators read from Lightning Fabric."""

    def __init__(self, device: torch.device):
        self.device = device
        self.global_rank = 0
        self.world_size = 1


class StubAgent:
    """What the evaluators read from ``BaseAgent``.

    ``root_dir`` must exist: ``MimicEvaluator`` writes failed-motion lists and
    the predicted motion library under it.
    """

    def __init__(self, env, model: StubModel, root_dir: Path, device=None):
        self.env = env
        self.model = model
        self.num_envs = env.num_envs
        self.device = env.device if device is None else device
        self.motion_lib = reference_motion(env).motion_lib
        self.root_dir = Path(root_dir)
        self.current_epoch = 0

    def eval(self) -> None:
        self.model.eval()

    def pre_collect_step(self, step: int) -> None:
        pass

    def add_agent_info_to_obs(self, obs: Dict) -> Dict:
        return obs

    def obs_dict_to_tensordict(self, obs_dict: Dict) -> TensorDict:
        # Same as BaseAgent.obs_dict_to_tensordict.
        batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
        return TensorDict(obs_dict, batch_size=batch_size, device=self.device)

    def inference_step(self, obs_td):
        return self.model(obs_td)


def make_evaluator(
    built: BuiltEnv, stub_agent: StubAgent, stub_fabric: Optional[StubFabric] = None
) -> Optional[Any]:
    """Instantiate the experiment's evaluator on the stubs, or ``None``.

    Returns ``None`` when the scenario has no agent config or its evaluator has
    neither metric nor failure components (``evaluate()`` would be a no-op).
    ``max_eval_steps`` is overridden on a copy of the config so the excursion
    stays short.
    """
    agent_cfg = built.configs.get("agent")
    evaluator_cfg = getattr(agent_cfg, "evaluator", None)
    if evaluator_cfg is None:
        return None
    if not (evaluator_cfg.metric_components or evaluator_cfg.failure_components):
        return None
    config = replace(evaluator_cfg, max_eval_steps=built.scenario.eval_max_steps)
    fabric = StubFabric(stub_agent.device) if stub_fabric is None else stub_fabric
    return get_class(config._target_)(stub_agent, fabric, config)
