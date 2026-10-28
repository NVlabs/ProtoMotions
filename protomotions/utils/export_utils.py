# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracing modules for exporting the deployable pipeline to ONNX.

These ``torch.nn.Module`` wrappers turn the training-time compute into a graph
``torch.onnx.export`` can trace. ``deployment/export_to_onnx.py`` drives
them and writes the typed ``DeploymentContract`` sidecar.

Modules:
    - ObservationExportModule: raw context tensors -> observation tensors, by
      running the configured observation kernels.
    - ActionExportModule: raw policy action -> processed PD targets, by running
      the action config's function with its tensors baked in as buffers.
    - UnifiedPipelineModule: context -> observations -> policy -> action, the
      three stages chained into one traceable graph.
"""

from __future__ import annotations

import torch
from typing import Optional, Dict, Any, TYPE_CHECKING

from protomotions.envs.context_paths import resolve_path

if TYPE_CHECKING:
    from protomotions.envs.action.config import ActionConfig


class ActionExportModule(torch.nn.Module):
    """Module that wraps an ActionConfig's function for ONNX export.

    Takes raw actions from the policy and produces processed actions with
    stiffness/damping targets by running the config's action function. The
    config's tensor parameters become buffers and its constants are baked in,
    so the traced graph carries them.
    """

    def __init__(
        self,
        action_config: Optional[ActionConfig],
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self._constants = {}
        self._output_keys = ["processed_action", "stiffness_targets", "damping_targets"]

        if action_config is None:
            self._action_function = None
            return

        self._action_function = action_config.fn
        for key, value in action_config.tensors().items():
            self.register_buffer(key, value.to(device))
        self._constants = dict(action_config.constants())

    def get_output_keys(self) -> list:
        return self._output_keys

    def forward(self, action: torch.Tensor) -> tuple:
        func_kwargs = {"action": action}
        for name in self._buffers:
            func_kwargs[name] = getattr(self, name)
        func_kwargs.update(self._constants)

        result = self._action_function(**func_kwargs)
        return (
            result["processed_action"],
            result["stiffness_targets"],
            result["damping_targets"],
        )


class UnifiedPipelineModule(torch.nn.Module):
    """Unified module that combines observations + policy + action processing.

    Pipeline: Context -> Observations -> Policy -> Action Processing

    Outputs:
        - actions: Raw actions from the policy
        - processed_action: PD targets (clamped and transformed)
        - stiffness_targets: Per-DOF stiffness values
        - damping_targets: Per-DOF damping values
    """

    def __init__(
        self,
        observation_module: "ObservationExportModule",
        policy_module: torch.nn.Module,
        action_module: "ActionExportModule",
        policy_in_keys: list,
        passthrough_keys: list = None,
    ):
        super().__init__()
        self.observation_module = observation_module
        if not hasattr(policy_module, "forward_deployable") or not hasattr(
            policy_module, "deployable_action_key"
        ):
            # The export boundary traces the deployable path only; a module
            # without the contract has an unknown forward (a training forward
            # that samples would be exported silently as the deployable graph),
            # so refuse it here rather than ship a sampling policy. This is the
            # same construction-time refusal D4 applies to external experts.
            raise TypeError(
                f"{type(policy_module).__name__} has no deployable contract: the "
                "exporter traces forward_deployable only. Declare "
                "forward_deployable(td) and deployable_action_key on the policy "
                "(the D4 model contract)."
            )
        if not policy_in_keys:
            raise ValueError(
                "A deployable policy must declare at least one batched input"
            )
        self.policy_module = policy_module
        self.action_module = action_module
        self.policy_in_keys = policy_in_keys
        self.passthrough_keys = passthrough_keys or []

        self.obs_output_keys = observation_module.get_output_keys()
        self.obs_input_keys = observation_module.get_input_keys()
        self.num_obs_inputs = len(self.obs_input_keys)

    def get_all_input_keys(self) -> list:
        return self.obs_input_keys + self.passthrough_keys

    def forward(self, *all_tensors) -> tuple:
        obs_context_tensors = all_tensors[: self.num_obs_inputs]
        passthrough_tensors = all_tensors[self.num_obs_inputs :]

        obs_outputs = self.observation_module(*obs_context_tensors)
        obs_dict = {key: obs_outputs[i] for i, key in enumerate(self.obs_output_keys)}

        for key, tensor in zip(self.passthrough_keys, passthrough_tensors):
            obs_dict[key] = tensor

        from tensordict import TensorDict

        policy_input = TensorDict(
            {key: obs_dict[key] for key in self.policy_in_keys},
            # Computed policy observations are per-env, unlike context constants.
            # Keep this dynamic so eager inference and ONNX agree at any batch.
            batch_size=[obs_dict[self.policy_in_keys[0]].shape[0]],
        )

        # The constructor refuses a policy without the deployable contract, so
        # the export traces the deterministic forward_deployable, never a
        # training forward that might sample, and reads the action under the
        # key the policy itself declares (never a hardcoded "mean_action").
        policy_output = self.policy_module.forward_deployable(policy_input)
        actions = policy_output[self.policy_module.deployable_action_key]

        processed_action, stiffness_targets, damping_targets = self.action_module(
            actions
        )

        return actions, processed_action, stiffness_targets, damping_targets


class ObservationExportModule(torch.nn.Module):
    """Module that wraps observation functions for ONNX export.

    This module takes raw context tensors as inputs and computes observations
    by calling the configured observation functions. It's designed to be
    exported to ONNX for deployment.

    Works with MdpComponent-based observation config format::

        observation_components = {
            "max_coords_obs": MdpComponent(
                compute_func=compute_humanoid_max_coords_observations,
                dynamic_vars={"body_pos": EnvContext.current.rigid_body_pos, ...},
                static_params={"local_obs": True},
            ),
        }

    Args:
        observation_configs: Dict of MdpComponent instances, one per observation.
        sample_context: Sample context dict to determine input shapes and resolve mappings.
        device: Device for tensor operations.

    Example:
        >>> from protomotions.envs.mdp_component import MdpComponent
        >>> from protomotions.envs.context_views import EnvContext
        >>> configs = {
        ...     "max_coords_obs": MdpComponent(
        ...         compute_func=compute_max_coords,
        ...         dynamic_vars={"body_pos": EnvContext.current.rigid_body_pos},
        ...     )
        ... }
        >>> context = env.context
        >>> module = ObservationExportModule(configs, context, device)
    """

    def __init__(
        self,
        observation_configs: Dict[str, Any],
        sample_context: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        # Import here to avoid circular dependency
        from protomotions.envs.mdp_component import is_mdp_component

        # Store observation functions and their input mappings
        self._obs_functions = {}
        self._obs_input_mappings = {}  # obs_name -> {arg_name: context_key}
        self._obs_constants = {}  # obs_name -> {arg_name: value}
        self._input_keys = set()  # All unique context keys needed
        self._output_keys = []  # Ordered list of output observation names

        for obs_name, cfg in observation_configs.items():
            assert is_mdp_component(cfg), "Observation config must be a MdpComponent"
            router = cfg
            compute_func = router.get_compute_func()
            bindings_dict = router.get_bindings_dict()  # {param_name: path_string}
            params = router.get_params()

            self._obs_functions[obs_name] = compute_func
            self._output_keys.append(obs_name)

            input_mapping = {}
            for arg_name, context_path in bindings_dict.items():
                input_mapping[arg_name] = context_path
                self._input_keys.add(context_path)

            self._obs_input_mappings[obs_name] = input_mapping
            self._obs_constants[obs_name] = params

        # Convert to ordered list for consistent ONNX input ordering
        self._input_keys = sorted(list(self._input_keys))

        # A binding that resolves to a non-tensor is a folded constant (a
        # published scalar such as the anchor body index), not a graph input;
        # move it to constants and keep only tensors as ONNX inputs. resolve_path
        # raises AttributeError ("context has no ...") for a name the sample
        # context lacks, and _build_samples has already refused any such binding,
        # so a failure here is a real defect to surface, not a case to swallow.
        for obs_name, mapping in self._obs_input_mappings.items():
            for arg_name, var_expr in list(mapping.items()):
                value = resolve_path(sample_context, var_expr)
                if not isinstance(value, torch.Tensor):
                    self._obs_constants[obs_name][arg_name] = value
                    del self._obs_input_mappings[obs_name][arg_name]

        # Rebuild input keys after removing non-tensors
        self._input_keys = set()
        for obs_name, mapping in self._obs_input_mappings.items():
            for var_expr in mapping.values():
                self._input_keys.add(var_expr)
        self._input_keys = sorted(list(self._input_keys))

    def get_input_keys(self) -> list:
        """Get ordered list of input context keys needed."""
        return self._input_keys

    def get_output_keys(self) -> list:
        """Get ordered list of output observation names."""
        return self._output_keys

    def forward(self, *args) -> tuple:
        """Compute all observations from input tensors.

        Args:
            *args: Input tensors in the order of get_input_keys().

        Returns:
            Tuple of observation tensors in the order of get_output_keys().
        """
        # Build context dict from positional args
        context = {key: tensor for key, tensor in zip(self._input_keys, args)}

        outputs = []
        for obs_name in self._output_keys:
            func = self._obs_functions[obs_name]
            input_mapping = self._obs_input_mappings[obs_name]
            constants = self._obs_constants[obs_name]

            # Build kwargs for the function
            func_kwargs = {}

            # Add tensor inputs from context
            for arg_name, var_expr in input_mapping.items():
                func_kwargs[arg_name] = context[var_expr]

            # Add constants
            func_kwargs.update(constants)

            # Call the observation function
            obs_value = func(**func_kwargs)
            outputs.append(obs_value)

        return tuple(outputs)
