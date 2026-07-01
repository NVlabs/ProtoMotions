# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base model interface for agent neural networks.

This module defines the abstract base class that all agent models must implement.
It provides a TensorDictModule interface for clean, compilable models.

Key Classes:
    - BaseModel: Abstract base class for all agent models (TensorDictModule)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.base_agent.config import BaseModelConfig
from abc import abstractmethod


@dataclass(frozen=True)
class RolloutStateSpec:
    """Declared model-owned per-env rollout context.

    A model declares the tensor shape, dtype, and initial-value distribution for
    per-env context that must be carried during rollout and replayed during
    optimization. The framework owns when context is initialized and which rows
    are reset after episode boundaries.
    """

    shape: tuple[int, ...] = ()
    init: str | Callable = "zeros"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        shape = self.shape
        if isinstance(shape, int):
            shape = (shape,)
        object.__setattr__(self, "shape", tuple(shape))

    def make_initial_value(self, num_envs: int, device) -> torch.Tensor:
        shape = (num_envs, *self.shape)
        init = getattr(self.init, "value", self.init)
        if callable(init):
            value = init(num_envs, device)
            return value.to(device=device, dtype=self.dtype)
        if init in ("zeros", "zero"):
            return torch.zeros(shape, dtype=self.dtype, device=device)
        if init in ("normal", "randn"):
            return torch.randn(shape, dtype=self.dtype, device=device)
        if init in ("uniform", "rand"):
            return torch.rand(shape, dtype=self.dtype, device=device)
        raise NotImplementedError(f"Unsupported rollout state init: {self.init}")


class ProtoMotionsTensorDictModule(TensorDictModuleBase):
    """Common contract for ProtoMotions-owned TensorDict modules.

    Use this for modules that read and write named TensorDict keys and may own
    rollout state that must be saved in the experience buffer. Examples include
    policy/critic modules, normalizing observation processors, container
    modules, and models with rollout-local latent state.
    """

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass for ProtoMotions TensorDict modules.

        ``log_internals`` is part of the common contract. Modules that produce
        diagnostics can honor it; stateless modules can ignore it.
        """
        raise NotImplementedError

    def _proto_children(self) -> list["ProtoMotionsTensorDictModule"]:
        """Nearest child modules that participate in the ProtoMotions contract."""
        children = []
        seen = set()

        def collect(module):
            for child in module.children():
                if isinstance(child, ProtoMotionsTensorDictModule):
                    child_id = id(child)
                    if child_id not in seen:
                        seen.add(child_id)
                        children.append(child)
                else:
                    collect(child)

        collect(self)
        return children

    def rollout_state_specs(self) -> dict[str, RolloutStateSpec]:
        """Local model-owned per-env rollout state declarations."""
        return {}

    def _rollout_state_specs_recursive(self) -> dict[str, RolloutStateSpec]:
        """Collect local and child rollout-state declarations.

        Containers get child declarations automatically. If two modules claim
        the same TensorDict key, the declarations must be identical so reset
        and experience-buffer registration cannot disagree silently.
        """
        specs = dict(self.rollout_state_specs())
        for child in self._proto_children():
            for key, child_spec in child._rollout_state_specs_recursive().items():
                if key in specs and specs[key] != child_spec:
                    raise ValueError(
                        f"Conflicting rollout state spec for '{key}': "
                        f"{specs[key]} vs {child_spec}"
                    )
                specs[key] = child_spec
        return specs

    def reset_rollout_context(
        self, env_ids=None, num_envs: int = None, device=None
    ) -> None:
        """Initialize or reseed model-owned context carried across rollout steps.

        ``num_envs`` and ``device`` are supplied once during setup, after the
        simulator is built. Later calls pass only ``env_ids`` to reseed rows for
        environments that just ended. This method intentionally does not infer
        shape/device changes mid-run; rebuilding the simulator should call setup
        again and reinitialize explicitly.
        """
        if num_envs is not None or device is not None:
            if num_envs is None or device is None:
                raise ValueError(
                    "reset_rollout_context requires both num_envs and device "
                    "when initializing rollout context."
                )
            self._init_own_rollout_state(num_envs=num_envs, device=device)

        if env_ids is not None:
            self._reset_own_rollout_state(env_ids=env_ids)

        for child in self._proto_children():
            child.reset_rollout_context(
                env_ids=env_ids,
                num_envs=num_envs,
                device=device,
            )

    def init_rollout_state(self, num_envs: int, device) -> None:
        """Allocate declared rollout-context buffers for this module tree."""
        self._init_own_rollout_state(num_envs=num_envs, device=device)
        for child in self._proto_children():
            child.init_rollout_state(num_envs=num_envs, device=device)

    def _init_own_rollout_state(self, num_envs: int, device) -> None:
        """Allocate this module's own declared rollout-context buffers."""
        for key, spec in self.rollout_state_specs().items():
            state = spec.make_initial_value(num_envs, torch.device(device))
            if hasattr(self, key):
                setattr(self, key, state)
            else:
                self.register_buffer(key, state, persistent=False)

    def reset_rollout_state(self, env_ids=None) -> None:
        """Reseed selected rows for this module tree's rollout context."""
        self._reset_own_rollout_state(env_ids=env_ids)
        for child in self._proto_children():
            child.reset_rollout_state(env_ids=env_ids)

    def _reset_own_rollout_state(self, env_ids=None) -> None:
        """Reseed selected rows of this module's initialized rollout context."""
        for key, spec in self.rollout_state_specs().items():
            state = getattr(self, key, None)
            if not torch.is_tensor(state):
                raise RuntimeError(self._uninitialized_rollout_state_message(key))

            rows = self._normalize_rollout_env_ids(env_ids, state)
            if rows.numel() == 0:
                continue
            state[rows] = spec.make_initial_value(rows.shape[0], state.device)

    def read_rollout_state(self, tensordict: TensorDict) -> TensorDict:
        """Inject carried rollout context into ``tensordict`` when absent.

        Rollout forwards read from model-owned buffers; optimization forwards
        read replayed values already present in the batch. This helper makes the
        distinction explicit and avoids each model hand-writing buffer fallbacks.
        """
        batch_size = tensordict.batch_size[0]
        for key in self.rollout_state_specs():
            if key in tensordict.keys():
                continue

            state = getattr(self, key, None)
            if not torch.is_tensor(state):
                raise RuntimeError(self._uninitialized_rollout_state_message(key))
            if batch_size > state.shape[0]:
                raise RuntimeError(
                    f"Rollout state '{key}' has {state.shape[0]} envs, but "
                    f"TensorDict batch size is {batch_size}."
                )
            tensordict[key] = state[:batch_size].clone()

        for child in self._proto_children():
            child.read_rollout_state(tensordict)
        return tensordict

    @staticmethod
    def _normalize_rollout_env_ids(env_ids, state: torch.Tensor) -> torch.Tensor:
        """Return row indices on the rollout-state device for reset writes."""
        if env_ids is None:
            return torch.arange(state.shape[0], device=state.device, dtype=torch.long)
        if isinstance(env_ids, list):
            return torch.tensor(env_ids, device=state.device, dtype=torch.long)
        return env_ids.to(device=state.device, dtype=torch.long)

    @staticmethod
    def _uninitialized_rollout_state_message(key: str) -> str:
        return (
            f"Rollout state '{key}' is not initialized. Call "
            "reset_rollout_context(num_envs=..., device=...) first."
        )

    def rollout_context_keys(self) -> list:
        """Module-owned state that must be stored from rollout and replayed."""
        keys = list(self.rollout_state_specs().keys())
        for child in self._proto_children():
            keys.extend(child.rollout_context_keys())
        return list(dict.fromkeys(keys))

    def experience_buffer_keys(self) -> list:
        """Keys produced during rollout that the experience buffer must store."""
        return list(dict.fromkeys(self.out_keys + self.rollout_context_keys()))

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss,
        log_prefix: str = "model",
    ):
        """Optional module-owned auxiliary loss for agent optimization loops.

        Most modules do not own an auxiliary loss. Models that do, such as a
        VAE-backed policy head, override this and return ``(loss, log_dict)``.
        """
        loss = zero_loss * 0.0
        log_dict = {}
        for child in self._proto_children():
            child_loss, child_log_dict = child.compute_model_loss(
                tensordict,
                current_epoch=current_epoch,
                zero_loss=zero_loss,
                log_prefix=log_prefix,
            )
            loss = loss + child_loss
            log_dict.update(child_log_dict)
        return loss, log_dict


class BaseModel(ProtoMotionsTensorDictModule):
    """Base class for all agent models.

    All models are TensorDictModules with a single forward method that
    processes observations and returns all model outputs in a TensorDict.

    Args:
        config: Model configuration with architecture parameters.

    Attributes:
        config: Stored configuration for the model.
        in_keys: Input keys for TensorDict (set by subclasses).
        out_keys: Output keys for TensorDict (default: ["action"]).
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config

        # Default output keys (subclasses can override)
        self.out_keys = ["action"]
        # in_keys will be set by subclasses based on their architecture
        self.in_keys = []

    def optimization_module(self):
        """Module whose parameters should be optimized by the owning agent."""
        return self

    def materialize_from_state_dict(self, state_dict: dict) -> None:
        """Create lazily-owned modules needed for strict state-dict loading."""
        pass

    def materialize(self, tensordict: TensorDict) -> TensorDict:
        """Run the setup-time pass used to create lazy module parameters."""
        return self(tensordict)

    @abstractmethod
    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass through the model.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with model outputs added.
        """
        pass
