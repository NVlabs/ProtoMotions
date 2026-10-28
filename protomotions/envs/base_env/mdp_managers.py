# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three MDP managers: observations, rewards, terminations.

Each one runs its group of ``MdpComponent`` kernels over a published context
and owns what the env used to do around them: the observation manager writes
the dynamic observation buffer (row checks, buffer growth, subset against
all-env writes) and runs the terrain and scene observation callbacks; the
reward manager applies the reset grace period and combines the raw rewards;
the termination manager combines the timeout, the pending reset requests and
the termination table (the recipe's kernels and the control components'
checks). The reward and termination managers are pure functions of what the
env hands them (the context, the counters, the reset requests) and RETURN
their logging dicts instead of writing ``env.extras``: ``BaseEnv.step``
writes the episode buffers and merges the dicts, so the orchestrator stays
the one place that touches shared state.

All three run their kernels through ONE ``ComponentManager``, the env's. Its
compiled-kernel cache is keyed by component name, so one cache serves the
three groups exactly as before this split; a separate manager per group
would change behaviour for a config that names a reward and an observation
alike. The compile rule is the component manager's and is untouched: a
context with ``env_ids is None`` (the all-env context ``step`` publishes)
takes the compiled path, a subset context (``reset``) runs eagerly.

The reward combining rule lives with the rewards
(``protomotions/envs/rewards/combine.py``, :func:`combine_rewards`: the
``RewardMeta`` weights, multiplicative terms, clamping, the reset grace period)
and the termination folding rule with the terminations
(``protomotions/envs/terminations/rule.py``: one ``TerminationRule`` per entry,
a threshold cutting a scalar kernel, ``truncation`` keeping a flag out of
``terminated``); both are imported here. Look there when a reward or a
termination does not behave as its metadata says.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor

from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import FrozenContext
from protomotions.envs.rewards.combine import combine_rewards
from protomotions.envs.terminations.base import check_max_length_term
from protomotions.envs.terminations.rule import (
    TerminationRule,
    evaluate_terminations,
    or_terminations,
)


class ObservationManager:
    """The observation buffer: every observation is a kernel of the table.

    ``buffer`` is the manager's own dict, filled here by name; the env's
    ``ObservationCache`` snapshot owner holds that same dict and saves and
    restores it. The terrain height map and the scene surfaces are kernels a
    recipe binds like any other (``envs/obs/terrain.py``, ``envs/obs/scene.py``);
    nothing is published that nobody bound.
    """

    def __init__(
        self,
        component_manager: ComponentManager,
        components: Mapping[str, Any],
        num_envs: int,
        device: torch.device,
    ):
        self.component_manager = component_manager
        self.components = components
        self.buffer: Dict[str, Tensor] = {}
        self.num_envs = num_envs
        self.device = device

    def compute(self, ctx: FrozenContext, env_ids: Tensor) -> None:
        """Compute the observations of ``env_ids`` and write them into the buffer.

        ``ctx`` is either a context over those rows (``ctx.env_ids`` equals
        ``env_ids``: a reset) or over every env (``ctx.env_ids`` is None: the
        step, where ``env_ids`` lists every row).
        """
        self.compute_components(ctx, env_ids)

    def compute_components(self, ctx: FrozenContext, env_ids: Tensor) -> None:
        """Run the observation kernels on ``ctx`` and write the rows ``env_ids``.

        ``BaseEnv``'s constructor calls this with the cached all-env context so
        the first observations take the compiled path.
        Checks that the kernels returned the rows the context describes,
        grows the buffer on first sight of a name, then writes in place.
        """
        raw_obs = self.component_manager.execute_all(
            components=self.components, ctx=ctx
        )

        context_env_ids = getattr(ctx, "env_ids", None)
        if context_env_ids is not None and not torch.equal(context_env_ids, env_ids):
            raise ValueError(
                "Observation context env_ids must match the destination env_ids"
            )

        for name, obs_value in raw_obs.items():
            expected_rows = (
                len(env_ids) if context_env_ids is not None else self.num_envs
            )
            if obs_value.dim() == 0 or obs_value.shape[0] != expected_rows:
                raise ValueError(
                    f"Observation component '{name}' must return {expected_rows} rows "
                    f"for a {'subset' if context_env_ids is not None else 'global'} "
                    f"context; got {obs_value.shape[0] if obs_value.dim() else 0}"
                )
            if obs_value.shape[-1] == 0:
                raise ValueError(
                    f"observation '{name}' has a zero last dimension; a module that "
                    "reads it would be a no-op"
                )

            if name not in self.buffer:
                self.buffer[name] = torch.zeros(
                    self.num_envs,
                    *obs_value.shape[1:],
                    dtype=obs_value.dtype,
                    device=self.device,
                )
            rows = obs_value if context_env_ids is not None else obs_value[env_ids]
            self.buffer[name][env_ids] = rows

    def get_obs(self) -> Dict[str, Tensor]:
        """Every cached observation, as copies, by name."""
        return {name: tensor.clone() for name, tensor in self.buffer.items()}


class RewardManager:
    """The reward kernels, the reset grace period and the combining rule."""

    def __init__(
        self,
        component_manager: ComponentManager,
        components: Mapping[str, Any],
        num_envs: int,
        device: torch.device,
        reset_grace_period: int,
    ):
        self.component_manager = component_manager
        self.components = components
        self.num_envs = num_envs
        self.device = device
        self.reset_grace_period = reset_grace_period

    def grace_mask(self, progress_buf: Tensor) -> Optional[Tensor]:
        """Which envs are within ``reset_grace_period`` steps of their reset
        (rewards that are unreliable right after a reset, such as power or
        contact changes, are zeroed there); None when there is no period."""
        if self.reset_grace_period <= 0:
            return None
        return progress_buf <= self.reset_grace_period

    def compute(
        self, ctx: FrozenContext, progress_buf: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """``(combined reward, logging)``; the logging holds the raw and scaled
        per-component rewards and, last, ``total_env_reward``."""
        grace_mask = self.grace_mask(progress_buf)
        raw_rewards = self.component_manager.execute_all(
            components=self.components, ctx=ctx
        )
        combined, logging = combine_rewards(
            raw_rewards=raw_rewards,
            components=self.components,
            grace_mask=grace_mask,
            num_envs=self.num_envs,
            device=self.device,
        )
        logging["total_env_reward"] = combined
        return combined, logging


class TerminationManager:
    """The timeout, the pending reset requests and the termination table.

    Only the maximum episode length is checked here directly; every other
    termination (falls, height, tracking error, a clip's end, a path
    deviation ...) is a ``TerminationRule`` of the table: the recipe's
    entries and the control components' checks, merged at build
    (``build.termination_table``) and run from the context alike
    (``evaluate_terminations`` cuts the scalar kernels, ``or_terminations``
    folds the flags, a truncating entry staying out of ``terminated``). The
    reset requests (the viewer's key) are handed in by the env, so this
    manager, like the reward manager, is a pure function of its arguments.
    """

    def __init__(
        self,
        component_manager: ComponentManager,
        components: Mapping[str, TerminationRule],
        num_envs: int,
        device: torch.device,
    ):
        self.component_manager = component_manager
        self.components = components
        self.num_envs = num_envs
        self.device = device

    def compute(
        self,
        ctx: FrozenContext,
        progress_buf: Tensor,
        max_episode_length: int,
        reset_request: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """``(reset, terminated, logging)``.

        ``reset_request`` is the mask of envs asked to reset (the viewer's
        key). ``reset`` is the timeout OR the requests OR the table's flags;
        ``terminated`` (the no-bootstrap flag) leaves the timeout, the
        requests and the truncating entries out. The logging holds one
        ``termination/<name>`` flag per entry.
        """
        reset = check_max_length_term(progress_buf, max_episode_length) | reset_request
        flags = evaluate_terminations(self.component_manager, self.components, ctx)
        entry_reset, terminated, logging = or_terminations(
            flags, self.components, self.num_envs, self.device
        )
        return reset | entry_reset, terminated, logging


__all__ = [
    "ObservationManager",
    "RewardManager",
    "TerminationManager",
    "combine_rewards",
]
