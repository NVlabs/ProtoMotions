# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utilities for exporting trained models to ONNX format.

This module provides functions to export TensorDict-based models to ONNX format
using torch.onnx.export. The exported models can be used for deployment
and inference in production environments.

Key Functions:
    - export_onnx: Export a TensorDictModule to ONNX format
    - export_ppo_model: Export a trained PPO model to ONNX
    - export_with_action_processing: Export model with baked-in action processing
"""

import torch
import json
from pathlib import Path
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from typing import Optional, Dict, Any, Union

from protomotions.robot_configs.base import ControlType


class ONNXExportWrapper(torch.nn.Module):
    """Wrapper for TensorDictModule that accepts positional args for ONNX export.

    TensorDictModules expect a TensorDict argument, but torch.onnx.export
    uses positional tensor inputs. This wrapper bridges the gap.
    """

    def __init__(self, module: TensorDictModuleBase, in_keys: list, batch_size: int):
        super().__init__()
        self.module = module
        self.in_keys = in_keys
        self._batch_size = batch_size

    def forward(self, *args):
        """Forward that reconstructs TensorDict from positional args."""
        # Reconstruct TensorDict from positional args
        # Use stored batch_size since args[0].shape[0] doesn't work during JIT tracing
        td = TensorDict(
            {key: tensor for key, tensor in zip(self.in_keys, args)},
            batch_size=[self._batch_size],
        )

        output_td = self.module(td)

        return tuple(output_td[key] for key in self.module.out_keys)


class ActionProcessingModule(torch.nn.Module):
    """Standalone action processing module for ONNX export.

    This module takes raw actions and applies clamping and PD/torque transformations.
    It's designed to be exported separately from the policy model.

    For PD control modes (BUILT_IN_PD, PROPORTIONAL):
        output = pd_action_offset + pd_action_scale * clamp(action)

    For TORQUE control mode:
        output = clamp(action) * torque_limits

    Args:
        control_type: Control type enum (BUILT_IN_PD, PROPORTIONAL, or TORQUE).
        clamp_value: Value to clamp actions to [-clamp_value, clamp_value].
        pd_action_offset: Offset for PD target computation (for PD modes).
        pd_action_scale: Scale for PD target computation (for PD modes).
        torque_limits: Torque limits for TORQUE mode (required for TORQUE mode).
        stiffness: Per-DOF stiffness values for PD control (optional).
        damping: Per-DOF damping values for PD control (optional).
    """

    def __init__(
        self,
        control_type: ControlType,
        clamp_value: float,
        pd_action_offset: Optional[torch.Tensor] = None,
        pd_action_scale: Optional[torch.Tensor] = None,
        torque_limits: Optional[torch.Tensor] = None,
        stiffness: Optional[torch.Tensor] = None,
        damping: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Store control mode as boolean (resolved at construction time).
        self._is_pd_control = control_type in (
            ControlType.BUILT_IN_PD,
            ControlType.PROPORTIONAL,
        )

        # Determine dtype from provided tensors.
        if pd_action_offset is not None:
            dtype = pd_action_offset.dtype
        elif torque_limits is not None:
            dtype = torque_limits.dtype
        else:
            dtype = torch.float32

        self.register_buffer("clamp_value", torch.tensor(clamp_value, dtype=dtype))

        if self._is_pd_control:
            if pd_action_offset is None or pd_action_scale is None:
                raise ValueError(
                    "pd_action_offset and pd_action_scale required for PD control modes"
                )
            self.register_buffer("pd_action_offset", pd_action_offset.clone())
            self.register_buffer("pd_action_scale", pd_action_scale.clone())
            self.output_type = "pd_targets"
        elif control_type == ControlType.TORQUE:
            if torque_limits is None:
                raise ValueError("torque_limits required for TORQUE control mode")
            self.register_buffer("torque_limits", torque_limits.clone())
            self.output_type = "torque_targets"
        else:
            raise ValueError(f"Unsupported control type: {control_type}")

        # Register stiffness and damping buffers for ONNX export.
        if stiffness is not None:
            self.register_buffer("stiffness", stiffness.clone())
        else:
            self.stiffness = None

        if damping is not None:
            self.register_buffer("damping", damping.clone())
        else:
            self.damping = None

    def forward(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process raw action into PD targets or torques.

        Args:
            action: Raw action tensor [batch_size, num_actions]

        Returns:
            Tuple of (actions, targets, stiffness_targets, damping_targets).
            Each tensor has shape [batch_size, num_actions].
        """
        # Clamp actions.
        clamped_action = torch.clamp(action, -self.clamp_value, self.clamp_value)

        # Apply transformation.
        if self._is_pd_control:
            targets = self.pd_action_offset + self.pd_action_scale * clamped_action
        else:
            targets = clamped_action * self.torque_limits

        # Broadcast stiffness and damping to batch size.
        batch_size = action.shape[0]
        if self.stiffness is not None:
            stiffness_targets = self.stiffness.unsqueeze(0).expand(batch_size, -1)
        else:
            stiffness_targets = torch.zeros_like(action)

        if self.damping is not None:
            damping_targets = self.damping.unsqueeze(0).expand(batch_size, -1)
        else:
            damping_targets = torch.zeros_like(action)

        return clamped_action, targets, stiffness_targets, damping_targets


class ActionProcessingONNXWrapper(torch.nn.Module):
    """Wrapper that includes action post-processing in the ONNX export.

    This wrapper combines the policy model with action clamping and
    PD/torque transformations, producing final targets that can be
    directly applied by the simulator without further processing.

    For PD control modes (BUILT_IN_PD, PROPORTIONAL):
        output = pd_action_offset + pd_action_scale * clamp(action)

    For TORQUE control mode:
        output = clamp(action) * torque_limits

    Args:
        module: TensorDictModule (policy) to wrap.
        in_keys: List of input keys for the module.
        batch_size: Batch size for TensorDict reconstruction.
        action_key: Key in module output containing the action (default: "action").
        control_type: Control type enum (BUILT_IN_PD, PROPORTIONAL, or TORQUE).
        clamp_value: Value to clamp actions to [-clamp_value, clamp_value].
        pd_action_offset: Offset for PD target computation (for PD modes).
        pd_action_scale: Scale for PD target computation (for PD modes).
        torque_limits: Torque limits for TORQUE mode scaling.

    Attributes:
        out_keys: Output keys for the wrapper (always ["pd_targets"] or ["torque_targets"]).
    """

    def __init__(
        self,
        module: TensorDictModuleBase,
        in_keys: list,
        batch_size: int,
        action_key: str,
        control_type: ControlType,
        clamp_value: float,
        pd_action_offset: Optional[torch.Tensor] = None,
        pd_action_scale: Optional[torch.Tensor] = None,
        torque_limits: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.module = module
        self.in_keys = in_keys
        self._batch_size = batch_size
        self.action_key = action_key

        # Use the standalone ActionProcessingModule
        self.action_processor = ActionProcessingModule(
            control_type=control_type,
            clamp_value=clamp_value,
            pd_action_offset=pd_action_offset,
            pd_action_scale=pd_action_scale,
            torque_limits=torque_limits,
        )
        self.output_type = self.action_processor.output_type
        self.out_keys = [self.output_type]

    def forward(self, *args):
        """Forward pass: model -> action_processor."""
        # Reconstruct TensorDict from positional args
        td = TensorDict(
            {key: tensor for key, tensor in zip(self.in_keys, args)},
            batch_size=[self._batch_size],
        )

        # Run the model
        output_td = self.module(td)

        # Get raw action from model output
        action = output_td[self.action_key]

        # Process action
        targets = self.action_processor(action)

        return (targets,)


def export_action_processing(
    control_type: ControlType,
    clamp_value: float,
    pd_action_offset: torch.Tensor,
    pd_action_scale: torch.Tensor,
    torque_limits: torch.Tensor,
    num_actions: int,
    path: str,
    batch_size: int = 1,
    meta: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    opset_version: int = 17,
):
    """Export action processing module to ONNX.

    Creates a standalone ONNX model that takes raw actions and outputs
    PD targets or torques.

    Args:
        control_type: Control type enum (BUILT_IN_PD, PROPORTIONAL, or TORQUE).
        clamp_value: Value to clamp actions to [-clamp_value, clamp_value].
        pd_action_offset: Offset for PD target computation.
        pd_action_scale: Scale for PD target computation.
        torque_limits: Torque limits for TORQUE mode.
        num_actions: Number of action dimensions.
        path: Path to save the ONNX model (must end with .onnx).
        batch_size: Batch size for sample input (default: 1).
        meta: Optional additional metadata to save.
        validate: If True, validates the exported model with onnxruntime.
        opset_version: ONNX opset version to use (default: 17).
    """
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    # Move tensors to CPU
    pd_action_offset = pd_action_offset.cpu()
    pd_action_scale = pd_action_scale.cpu()
    torque_limits = torque_limits.cpu()

    # Create the action processing module
    action_processor = ActionProcessingModule(
        control_type=control_type,
        clamp_value=clamp_value,
        pd_action_offset=pd_action_offset,
        pd_action_scale=pd_action_scale,
        torque_limits=torque_limits,
    )
    action_processor.eval()
    action_processor.cpu()

    # Create sample input
    sample_action = torch.randn(batch_size, num_actions, dtype=pd_action_offset.dtype)

    print(f"Exporting action processing to ONNX...")
    print(f"  Control type: {control_type.name}")
    print(f"  Clamp value: {clamp_value}")
    print(f"  Num actions: {num_actions}")
    print(f"  Output type: {action_processor.output_type}")

    torch.onnx.export(
        action_processor,
        (sample_action,),
        path,
        input_names=["raw_action"],
        output_names=[action_processor.output_type],
        dynamic_axes={
            "raw_action": {0: "batch_size"},
            action_processor.output_type: {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"✓ Exported action processing to {path}")

    # Save metadata
    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["model_type"] = "action_processing"
    meta["control_type"] = control_type.name
    meta["clamp_value"] = clamp_value
    meta["num_actions"] = num_actions
    meta["output_type"] = action_processor.output_type
    meta["input_name"] = "raw_action"
    meta["output_name"] = action_processor.output_type

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"✓ Exported metadata to {meta_path}")

    # Validate
    if validate:
        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            # Run ONNX
            onnx_input = {"raw_action": sample_action.numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]

            # Run PyTorch
            with torch.no_grad():
                pytorch_output = action_processor(sample_action).numpy()

            # Compare
            max_diff = abs(onnx_output - pytorch_output).max()
            print(f"✓ Validation: max diff = {max_diff:.2e}")

        except ImportError:
            print("⚠ Warning: onnxruntime not installed, skipping validation.")
        except Exception as e:
            print(f"⚠ Warning: Validation failed: {e}")

    return action_processor


def export_onnx(
    module: TensorDictModuleBase,
    td: TensorDict,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    opset_version: int = 17,
):
    """Export a TensorDictModule to ONNX format.

    Uses torch.onnx.export to export the module. Creates a wrapper that
    converts between TensorDict and positional tensor inputs for ONNX compatibility.

    Args:
        module: TensorDictModule to export.
        td: Sample TensorDict input (used for tracing).
        path: Path to save the ONNX model (must end with .onnx).
        meta: Optional additional metadata to save.
        validate: If True, validates the exported model with onnxruntime.
        opset_version: ONNX opset version to use (default: 17).

    Raises:
        ValueError: If path doesn't end with .onnx.

    Example:
        >>> from protomotions.agents.ppo.model import PPOModel
        >>> from tensordict import TensorDict
        >>> model = PPOModel(config)
        >>> sample_input = TensorDict({"obs": torch.randn(1, 128)}, batch_size=1)
        >>> export_onnx(model, sample_input, "policy.onnx")
    """
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    # Move to CPU and select only required input keys
    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    module.eval()

    in_keys = list(module.in_keys)
    out_keys = list(module.out_keys)

    print(f"Exporting model to ONNX (PyTorch {torch.__version__})...")
    print(f"  Input keys: {in_keys}")
    print(f"  Output keys: {out_keys}")

    # Create wrapper that accepts positional args instead of TensorDict
    batch_size = td.batch_size[0] if td.batch_size else 1
    wrapper = ONNXExportWrapper(module, in_keys, batch_size)
    wrapper.eval()

    # Prepare input tuple for torch.onnx.export
    input_tensors = tuple(td[key] for key in in_keys)

    # Create input/output names for ONNX
    input_names = [f"input_{i}" for i in range(len(in_keys))]
    output_names = [f"output_{i}" for i in range(len(out_keys))]

    torch.onnx.export(
        wrapper,
        input_tensors,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in input_names},
            **{name: {0: "batch_size"} for name in output_names},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"✓ Exported ONNX model to {path}")

    # Save metadata
    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = in_keys
    meta["out_keys"] = out_keys
    meta["in_shapes"] = [list(td[k].shape) for k in in_keys]
    meta["onnx_input_names"] = input_names
    meta["onnx_output_names"] = output_names
    meta["input_mapping"] = {
        onnx_name: semantic_name
        for onnx_name, semantic_name in zip(input_names, in_keys)
    }
    meta["output_mapping"] = {
        onnx_name: semantic_name
        for onnx_name, semantic_name in zip(output_names, out_keys)
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"✓ Exported metadata to {meta_path}")

    # Validate with onnxruntime
    if validate:
        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            def to_numpy(tensor):
                return (
                    tensor.detach().cpu().numpy()
                    if tensor.requires_grad
                    else tensor.cpu().numpy()
                )

            onnxruntime_input = {
                name: to_numpy(tensor)
                for name, tensor in zip(input_names, input_tensors)
            }

            ort_output = ort_session.run(None, onnxruntime_input)
            assert len(ort_output) == len(out_keys), f"Output length mismatch: {len(ort_output)} vs {len(out_keys)}"

            print("✓ ONNX model validation successful!")

        except ImportError:
            print("⚠ Warning: onnxruntime not installed, skipping validation.")
        except Exception as e:
            print(f"⚠ Warning: ONNX validation failed: {e}")


def export_with_action_processing(
    module: TensorDictModuleBase,
    td: TensorDict,
    path: str,
    action_key: str,
    control_type: ControlType,
    clamp_value: float,
    pd_action_offset: Optional[torch.Tensor] = None,
    pd_action_scale: Optional[torch.Tensor] = None,
    torque_limits: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    opset_version: int = 17,
):
    """Export a TensorDictModule to ONNX with baked-in action processing.

    This function wraps the model with action clamping and PD/torque transformations,
    producing an ONNX model that outputs final targets (PD targets or torques) ready
    to be applied directly by the simulator.

    For PD control modes:
        output = pd_action_offset + pd_action_scale * clamp(action, -clamp_value, clamp_value)

    For TORQUE control mode:
        output = clamp(action, -clamp_value, clamp_value) * torque_limits

    Args:
        module: TensorDictModule (policy) to export.
        td: Sample TensorDict input (used for tracing).
        path: Path to save the ONNX model (must end with .onnx).
        action_key: Key in module output containing the action (e.g., "action", "mean_action").
        control_type: Control type enum (BUILT_IN_PD, PROPORTIONAL, or TORQUE).
        clamp_value: Value to clamp actions to [-clamp_value, clamp_value].
        pd_action_offset: Offset tensor for PD target computation (required for PD modes).
        pd_action_scale: Scale tensor for PD target computation (required for PD modes).
        torque_limits: Torque limits tensor for TORQUE mode (required for TORQUE mode).
        meta: Optional additional metadata to save.
        validate: If True, validates the exported model with onnxruntime.
        opset_version: ONNX opset version to use (default: 17).

    Raises:
        ValueError: If path doesn't end with .onnx or required parameters are missing.

    Example:
        >>> from protomotions.agents.ppo.model import PPOModel
        >>> model = PPOModel(config)
        >>> sample_input = TensorDict({"obs": torch.randn(1, 128)}, batch_size=1)
        >>> export_with_action_processing(
        ...     model._actor,
        ...     sample_input,
        ...     "policy_with_actions.onnx",
        ...     action_key="mean_action",
        ...     control_type=ControlType.BUILT_IN_PD,
        ...     clamp_value=1.0,
        ...     pd_action_offset=torch.zeros(29),
        ...     pd_action_scale=torch.ones(29),
        ... )
    """
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    # Move to CPU and select only required input keys
    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    module.eval()

    # Move action processing tensors to CPU
    if pd_action_offset is not None:
        pd_action_offset = pd_action_offset.cpu()
    if pd_action_scale is not None:
        pd_action_scale = pd_action_scale.cpu()
    if torque_limits is not None:
        torque_limits = torque_limits.cpu()

    in_keys = list(module.in_keys)

    # Determine output type based on control type
    if control_type in (ControlType.BUILT_IN_PD, ControlType.PROPORTIONAL):
        output_type = "pd_targets"
    else:
        output_type = "torque_targets"

    out_keys = [output_type]

    print(f"Exporting model with action processing to ONNX (PyTorch {torch.__version__})...")
    print(f"  Input keys: {in_keys}")
    print(f"  Output keys: {out_keys}")
    print(f"  Control type: {control_type.name}")
    print(f"  Clamp value: {clamp_value}")
    print(f"  Action key: {action_key}")

    # Create wrapper that includes action processing
    batch_size = td.batch_size[0] if td.batch_size else 1
    wrapper = ActionProcessingONNXWrapper(
        module=module,
        in_keys=in_keys,
        batch_size=batch_size,
        action_key=action_key,
        control_type=control_type,
        clamp_value=clamp_value,
        pd_action_offset=pd_action_offset,
        pd_action_scale=pd_action_scale,
        torque_limits=torque_limits,
    )
    wrapper.eval()

    # Prepare input tuple for torch.onnx.export
    input_tensors = tuple(td[key] for key in in_keys)

    # Create input/output names for ONNX
    input_names = [f"input_{i}" for i in range(len(in_keys))]
    output_names = [f"output_{i}" for i in range(len(out_keys))]

    torch.onnx.export(
        wrapper,
        input_tensors,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in input_names},
            **{name: {0: "batch_size"} for name in output_names},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"✓ Exported ONNX model to {path}")

    # Save metadata
    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = in_keys
    meta["out_keys"] = out_keys
    meta["in_shapes"] = [list(td[k].shape) for k in in_keys]
    meta["onnx_input_names"] = input_names
    meta["onnx_output_names"] = output_names
    meta["input_mapping"] = {
        onnx_name: semantic_name
        for onnx_name, semantic_name in zip(input_names, in_keys)
    }
    meta["output_mapping"] = {
        onnx_name: semantic_name
        for onnx_name, semantic_name in zip(output_names, out_keys)
    }
    # Action processing metadata
    meta["action_processing_baked"] = True
    meta["control_type"] = control_type.name
    meta["clamp_value"] = clamp_value
    meta["output_type"] = output_type
    meta["action_key"] = action_key

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"✓ Exported metadata to {meta_path}")

    # Validate with onnxruntime
    if validate:
        try:
            import onnxruntime as ort

            ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            def to_numpy(tensor):
                return (
                    tensor.detach().cpu().numpy()
                    if tensor.requires_grad
                    else tensor.cpu().numpy()
                )

            onnxruntime_input = {
                name: to_numpy(tensor)
                for name, tensor in zip(input_names, input_tensors)
            }

            ort_output = ort_session.run(None, onnxruntime_input)
            assert len(ort_output) == len(out_keys), (
                f"Output length mismatch: {len(ort_output)} vs {len(out_keys)}"
            )

            print("✓ ONNX model validation successful!")

        except ImportError:
            print("⚠ Warning: onnxruntime not installed, skipping validation.")
        except Exception as e:
            print(f"⚠ Warning: ONNX validation failed: {e}")


def export_ppo_actor(
    actor, sample_obs: Dict[str, torch.Tensor], path: str, validate: bool = True
):
    """Export a PPO actor's mu network to ONNX.

    Exports the mean network (mu) of a PPO actor, which is the core policy
    network without the distribution layer. Uses real observations from the
    environment to ensure proper tracing.

    Args:
        actor: PPOActor instance to export.
        sample_obs: Sample observation dict from environment (via agent.get_obs()).
        path: Path to save the ONNX model.
        validate: If True, validates the exported model.

    Example:
        >>> # Get real observations from environment
        >>> env.reset()
        >>> sample_obs = agent.get_obs()
        >>> export_ppo_actor(agent.model._actor, sample_obs, "ppo_actor.onnx")
    """
    # Create TensorDict from sample observations
    batch_size = sample_obs[list(sample_obs.keys())[0]].shape[0]
    td = TensorDict(sample_obs, batch_size=batch_size)

    meta = {
        "model_type": "PPOActor",
        "observation_keys": list(sample_obs.keys()),
        "observation_shapes": {k: list(v.shape) for k, v in sample_obs.items()},
    }

    export_onnx(actor, td, path, meta=meta, validate=validate)


def export_ppo_critic(
    critic, sample_obs: Dict[str, torch.Tensor], path: str, validate: bool = True
):
    """Export a PPO critic network to ONNX.

    Uses real observations from the environment to ensure proper tracing.

    Args:
        critic: PPO critic (MultiHeadedMLP) instance to export.
        sample_obs: Sample observation dict from environment (via agent.get_obs()).
        path: Path to save the ONNX model.
        validate: If True, validates the exported model.

    Example:
        >>> # Get real observations from environment
        >>> env.reset()
        >>> sample_obs = agent.get_obs()
        >>> export_ppo_critic(agent.model._critic, sample_obs, "ppo_critic.onnx")
    """
    # Create TensorDict from sample observations
    batch_size = sample_obs[list(sample_obs.keys())[0]].shape[0]
    td = TensorDict(sample_obs, batch_size=batch_size)

    meta = {
        "model_type": "PPOCritic",
        "num_out": critic.config.num_out,
        "observation_keys": list(sample_obs.keys()),
        "observation_shapes": {k: list(v.shape) for k, v in sample_obs.items()},
    }

    export_onnx(critic, td, path, meta=meta, validate=validate)


def export_ppo_model(
    model, sample_obs: Dict[str, torch.Tensor], output_dir: str, validate: bool = True
):
    """Export a complete PPO model (actor and critic) to ONNX.

    Exports both the actor and critic networks to separate ONNX files
    in the specified directory.

    Args:
        model: PPOModel instance to export.
        sample_obs: Sample observation dict for tracing.
        output_dir: Directory to save the ONNX models.
        validate: If True, validates the exported models.

    Returns:
        Dict with paths to exported files.

    Example:
        >>> model = trained_agent.model
        >>> sample_obs = {"obs": torch.randn(1, 128)}
        >>> paths = export_ppo_model(model, sample_obs, "exported_models/")
        >>> print(paths)
        {'actor': 'exported_models/actor.onnx', 'critic': 'exported_models/critic.onnx'}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_path = str(output_dir / "actor.onnx")
    critic_path = str(output_dir / "critic.onnx")

    print("Exporting PPO Actor...")
    export_ppo_actor(model._actor, sample_obs, actor_path, validate=validate)

    print("\nExporting PPO Critic...")
    export_ppo_critic(model._critic, sample_obs, critic_path, validate=validate)

    print(f"\nExport complete! Models saved to {output_dir}")

    return {
        "actor": actor_path,
        "critic": critic_path,
        "metadata": {
            "actor_meta": str(output_dir / "actor.json"),
            "critic_meta": str(output_dir / "critic.json"),
        },
    }


###############################################################################
# Unified Pipeline Export (Context -> Actions)
###############################################################################


class UnifiedPipelineModule(torch.nn.Module):
    """Unified module that combines observations + policy + action processing.

    This module takes raw context tensors as inputs and returns raw actions,
    post-processed actions (PD targets or torques), and stiffness/damping targets.

    Pipeline: Context -> Observations -> Policy -> Action Processing

    Inputs are split into two groups:
    1. Context tensors for observation computation (from observation_module.get_input_keys())
    2. Additional tensors passed directly to policy (e.g., historical observations)

    Outputs:
        - actions: Raw actions from the policy (used by env.step())
        - joint_pos_targets: Actions after clamping and PD/torque transform
        - stiffness_targets: Per-DOF stiffness values (constant, broadcast to batch)
        - damping_targets: Per-DOF damping values (constant, broadcast to batch)

    Args:
        observation_module: ObservationExportModule for computing observations
        policy_module: Policy model (actor) that takes observations and returns actions
        action_processing_module: ActionProcessingModule for post-processing
        policy_in_keys: List of observation keys the policy expects as input
        policy_action_key: Key for the action output from policy (e.g., 'mean_action')
        passthrough_keys: Additional keys passed directly to policy (not from observations)
    """
    
    def __init__(
        self,
        observation_module: "ObservationExportModule",
        policy_module: torch.nn.Module,
        action_processing_module: "ActionProcessingModule",
        policy_in_keys: list,
        policy_action_key: str = "mean_action",
        passthrough_keys: list = None,
    ):
        super().__init__()
        self.observation_module = observation_module
        self.policy_module = policy_module
        self.action_processing_module = action_processing_module
        self.policy_in_keys = policy_in_keys
        self.policy_action_key = policy_action_key
        self.passthrough_keys = passthrough_keys or []
        
        # Get observation output keys for mapping to policy inputs
        self.obs_output_keys = observation_module.get_output_keys()
        
        # Context inputs for observation computation
        self.obs_input_keys = observation_module.get_input_keys()
        self.num_obs_inputs = len(self.obs_input_keys)
    
    def get_all_input_keys(self) -> list:
        """Get all input keys: observation context + passthrough."""
        return self.obs_input_keys + self.passthrough_keys
    
    def forward(self, *all_tensors) -> tuple:
        """Run the full pipeline from context to actions.

        Args:
            *all_tensors: Input tensors in order of:
                1. observation_module.get_input_keys()
                2. passthrough_keys

        Returns:
            Tuple of (actions, joint_pos_targets, stiffness_targets, damping_targets).
        """
        # Split inputs: observation context vs passthrough.
        obs_context_tensors = all_tensors[: self.num_obs_inputs]
        passthrough_tensors = all_tensors[self.num_obs_inputs :]

        # Step 1: Compute observations from context.
        obs_outputs = self.observation_module(*obs_context_tensors)

        # Build observation dict from computed observations.
        obs_dict = {
            key: obs_outputs[i] for i, key in enumerate(self.obs_output_keys)
        }

        # Add passthrough tensors directly to obs_dict.
        for key, tensor in zip(self.passthrough_keys, passthrough_tensors):
            obs_dict[key] = tensor

        # Step 2: Run policy with all required inputs.
        from tensordict import TensorDict

        # Infer batch size from passthrough tensors (which always have batch dim)
        # or from a policy input tensor, avoiding constant tensors like body_ids
        batch_size = None
        if passthrough_tensors:
            batch_size = passthrough_tensors[0].shape[0]
        else:
            # Find a tensor with "current_state_" prefix which should have batch dim
            for key in self.policy_in_keys:
                if key in obs_dict and key.startswith("current_state_"):
                    batch_size = obs_dict[key].shape[0]
                    break
        if batch_size is None:
            # Fallback: use first policy input that's 2D or more
            for key in self.policy_in_keys:
                if key in obs_dict and obs_dict[key].dim() >= 2:
                    batch_size = obs_dict[key].shape[0]
                    break
        if batch_size is None:
            batch_size = 1

        policy_input = TensorDict(
            {key: obs_dict[key] for key in self.policy_in_keys},
            batch_size=[batch_size],
        )

        policy_output = self.policy_module(policy_input)

        # Extract raw actions.
        actions = policy_output[self.policy_action_key]

        # Step 3: Apply action processing (returns 4 tensors).
        actions, joint_pos_targets, stiffness_targets, damping_targets = (
            self.action_processing_module(actions)
        )

        return actions, joint_pos_targets, stiffness_targets, damping_targets


###############################################################################
# YAML Configuration Generation
###############################################################################

# Mapping from ONNX input names to (name, kind) tuples for isaac-deploy YAML.
ONNX_INPUT_MAPPING = {
    # Current state
    "current_state_dof_pos": ("joint_pos", "joint_pos"),
    "current_state_dof_vel": ("joint_vel", "joint_vel"),
    "current_state_root_ang_vel": ("root_body_ang_vel", "root_body_ang_vel"),
    "current_state_root_local_ang_vel": ("root_body_ang_vel", "root_body_ang_vel"),
    "current_state_root_rot": ("root_body_rot", "root_body_rot"),
    "current_state_anchor_rot": ("anchor_rot", "anchor_body_rot"),
    "current_state_rigid_body_pos": ("body_pos", "body_pos"),
    "current_state_rigid_body_rot": ("body_rot", "body_rot"),
    # Historical state
    "historical_actions": ("last_actions", "last_actions"),
    "historical_dof_pos": ("joint_pos_history", "joint_pos"),
    "historical_dof_vel": ("joint_vel_history", "joint_vel"),
    "historical_root_ang_vel": ("root_body_ang_vel_history", "root_body_ang_vel"),
    "historical_root_local_ang_vel": ("root_body_ang_vel_history", "root_body_ang_vel"),
    "historical_root_rot": ("root_body_rot_history", "root_body_rot"),
    "historical_anchor_rot": ("anchor_rot_history", "anchor_body_rot"),
    "historical_rigid_body_pos": ("body_pos_history", "body_pos"),
    "historical_rigid_body_rot": ("body_rot_history", "body_rot"),
    "historical_rigid_body_vel": ("body_vel_history", "body_vel"),
    "historical_rigid_body_ang_vel": ("body_ang_vel_history", "body_ang_vel"),
    "historical_ground_heights": ("ground_heights_history", "ground_heights"),
    "previous_actions": ("previous_actions", "last_actions"),
    # Reference motion (mimic)
    "mimic_ref_ang_vel": ("reference_motion_body_ang_vel", "reference_motion_body_ang_vel"),
    "mimic_ref_dof_pos": ("reference_motion_joint_pos", "reference_motion_joint_pos"),
    "mimic_ref_dof_vel": ("reference_motion_joint_vel", "reference_motion_joint_vel"),
    "mimic_ref_rot": ("reference_motion_body_rot", "reference_motion_body_rot"),
    "mimic_ref_anchor_rot": ("reference_motion_anchor_rot", "reference_motion_body_rot"),
    # MaskedMimic sparse conditioning
    "masked_mimic_ref_pos": ("masked_mimic_ref_pos", "masked_mimic_body_pos"),
    "masked_mimic_ref_rot": ("masked_mimic_ref_rot", "masked_mimic_body_rot"),
    "masked_mimic_target_bodies_masks": ("masked_mimic_body_masks", "masked_mimic_masks"),
    "masked_mimic_target_poses_masks": ("masked_mimic_pose_masks", "masked_mimic_masks"),
    "masked_mimic_time_offsets": ("masked_mimic_time_offsets", "masked_mimic_time"),
    # VAE
    "vae_noise": ("vae_noise", "vae_noise"),
}

# Mapping from ONNX output names to (name, kind) tuples for isaac-deploy YAML.
ONNX_OUTPUT_MAPPING = {
    "actions": ("actions", "actions"),
    "joint_pos_targets": ("joint_pos_targets", "joint_pos_targets"),
    "stiffness_targets": ("stiffness_targets", "stiffness_targets"),
    "damping_targets": ("damping_targets", "damping_targets"),
}


def _build_policy_input(
    onnx_name: str,
    input_shapes: Dict[str, list],
    joint_names: list,
    body_names: list,
    anchor_body: str = "pelvis",
) -> Optional[Dict[str, Any]]:
    """Build a policy input entry for YAML from ONNX name."""
    if onnx_name not in ONNX_INPUT_MAPPING:
        return None

    name, kind = ONNX_INPUT_MAPPING[onnx_name]
    shape = input_shapes.get(onnx_name, [1, 1])

    # Normalize shape to have batch size 1.
    shape = [1] + list(shape)[1:]

    # Reference motion inputs don't have history fields.
    is_reference_motion = kind.startswith("reference_motion_")

    # Infer history from shape (dimension 1 if > 2D).
    history = shape[1] if len(shape) >= 3 and shape[1] > 1 else 0

    # Determine include_current_value_in_history:
    # - True for simulator values (joint_pos, joint_vel, root_body_rot, root_body_ang_vel,
    #   anchor_body_rot) and last_actions.
    # - False for historical observations (history != 0 and not last_actions).
    simulator_kinds = {
        "joint_pos",
        "joint_vel",
        "root_body_rot",
        "root_body_ang_vel",
        "anchor_body_rot",
    }
    if kind == "last_actions":
        include_current = True
    elif kind in simulator_kinds and history == 0:
        include_current = True
    else:
        include_current = False

    entry = {
        "name": name,
        "kind": kind,
        "shape": shape,
        "key": onnx_name,
    }

    if kind.endswith("_rot"):
        entry["quaternion_order"] = "xyzw"

    if not is_reference_motion:
        entry["history"] = history
        entry["include_current_value_in_history"] = include_current

    if "joint" in kind or kind == "last_actions":
        entry["joint_names"] = joint_names

    # Handle body_names for body-related observations.
    if "body" in kind and kind != "anchor_body_rot":
        # For reference_motion_body_rot with anchor, use just the anchor body.
        if onnx_name == "mimic_ref_anchor_rot":
            entry["body_names"] = [anchor_body]
        elif len(body_names) in shape:
            entry["body_names"] = body_names

    # Handle anchor_body_rot observations.
    if kind == "anchor_body_rot":
        entry["anchor_body"] = anchor_body

    if kind == "last_actions":
        entry["output_key"] = "actions"

    return entry


def _build_policy_output(
    onnx_name: str,
    output_shapes: Dict[str, list],
    joint_names: list,
    stiffness: list,
    damping: list,
    use_onnx_for_gains: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build a policy output entry for YAML from ONNX name.

    Args:
        onnx_name: Name of the ONNX output.
        output_shapes: Dictionary mapping ONNX output names to shapes.
        joint_names: List of joint names.
        stiffness: List of stiffness values (used only if use_onnx_for_gains=False).
        damping: List of damping values (used only if use_onnx_for_gains=False).
        use_onnx_for_gains: If True, read stiffness/damping from ONNX outputs.
                           If False, use constant values from YAML.
    """
    if onnx_name not in ONNX_OUTPUT_MAPPING:
        return None

    name, kind = ONNX_OUTPUT_MAPPING[onnx_name]
    shape = output_shapes.get(onnx_name, [1, len(joint_names)])

    # Normalize shape to have batch size 1.
    shape = [1] + list(shape)[1:]

    entry = {
        "name": name,
        "kind": kind,
        "key": onnx_name,  # Always use ONNX output name as key.
        "shape": shape,
    }

    # Add joint_names only for action terms that need it (not the passthrough "actions" term).
    if "joint" in kind or kind in ("stiffness_targets", "damping_targets"):
        entry["joint_names"] = joint_names

    # For stiffness/damping, optionally fall back to YAML constants.
    if kind == "stiffness_targets" and not use_onnx_for_gains:
        entry["key"] = None  # Use YAML values instead of ONNX.
        entry["stiffness"] = stiffness

    if kind == "damping_targets" and not use_onnx_for_gains:
        entry["key"] = None  # Use YAML values instead of ONNX.
        entry["damping"] = damping

    return entry


def _generate_yaml_content(
    input_shapes: Dict[str, list],
    output_shapes: Dict[str, list],
    onnx_in_names: list,
    onnx_out_names: list,
    joint_names: list,
    body_names: list,
    stiffness: list,
    damping: list,
    anchor_body: str = "pelvis",
) -> Dict[str, Any]:
    """Generate the complete YAML content for isaac-deploy."""
    # Build policy inputs.
    policy_inputs = []
    for onnx_name in onnx_in_names:
        entry = _build_policy_input(
            onnx_name, input_shapes, joint_names, body_names, anchor_body
        )
        if entry:
            policy_inputs.append(entry)

    # Build policy outputs.
    policy_outputs = []
    for onnx_name in onnx_out_names:
        entry = _build_policy_output(
            onnx_name, output_shapes, joint_names, stiffness, damping
        )
        if entry:
            policy_outputs.append(entry)

    return {
        "type": "unified_pipeline",
        "joint_names": joint_names,
        "body_names": body_names,
        "stiffness": stiffness,
        "damping": damping,
        "policy_inputs": policy_inputs,
        "policy_outputs": policy_outputs,
    }


def export_unified_pipeline(
    observation_configs: Dict[str, Any],
    sample_context: Dict[str, Any],
    policy_module: torch.nn.Module,
    policy_in_keys: list,
    policy_action_key: str,
    action_processing_module: "ActionProcessingModule",
    path: str,
    device: torch.device,
    robot_config: Any,
    passthrough_obs: Optional[Dict[str, torch.Tensor]] = None,
    validate: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Export the complete pipeline (context -> actions) as a single ONNX model.

    Creates a unified model that:
    1. Computes observations from context
    2. Runs the policy to get raw actions (using both computed obs and passthrough inputs)
    3. Applies action processing (clamp + PD/torque transform)

    Outputs four tensors:
    - actions: Raw actions from policy (used by env.step())
    - joint_pos_targets: Actions after clamping and PD/torque transform
    - stiffness_targets: Per-DOF stiffness values (constant)
    - damping_targets: Per-DOF damping values (constant)

    Also generates a YAML configuration file for isaac-deploy.

    Args:
        observation_configs: Dict of observation component configurations.
        sample_context: Sample context dict for tracing.
        policy_module: Policy model (actor) that takes observations.
        policy_in_keys: List of observation keys the policy expects.
        policy_action_key: Key for action output from policy.
        action_processing_module: ActionProcessingModule instance.
        path: Path to save the ONNX model.
        device: Device for tensor operations.
        robot_config: Robot configuration with kinematic_info and control.
        passthrough_obs: Dict of additional observations passed directly to policy
            (e.g., historical observations). Keys are obs names, values are sample tensors.
        validate: If True, validates with onnxruntime.
        meta: Optional metadata to include.

    Returns:
        Path to the exported ONNX model.
    """
    import logging
    import yaml

    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("Exporting Unified Pipeline (Context -> Actions)")
    log.info("=" * 60)

    passthrough_obs = passthrough_obs or {}

    # Extract robot metadata.
    joint_names = robot_config.kinematic_info.dof_names
    body_names = robot_config.kinematic_info.body_names
    stiffness = [
        float(robot_config.control.control_info[j].stiffness) for j in joint_names
    ]
    damping = [
        float(robot_config.control.control_info[j].damping) for j in joint_names
    ]
    # Get anchor body name (defaults to first body if not specified).
    anchor_body = (
        robot_config.anchor_body_name
        if robot_config.anchor_body_name
        else body_names[0]
    )

    log.info(f"Joint names: {joint_names}")
    log.info(f"Body names: {body_names}")
    log.info(f"Anchor body: {anchor_body}")

    # Step 1: Create observation module.
    obs_module = ObservationExportModule(observation_configs, sample_context, device)
    obs_module.eval()

    obs_input_keys = obs_module.get_input_keys()
    obs_output_keys = obs_module.get_output_keys()
    passthrough_keys = list(passthrough_obs.keys())

    log.info(f"Context input keys (for obs): {obs_input_keys}")
    log.info(f"Observation output keys: {obs_output_keys}")
    log.info(f"Passthrough keys (direct to policy): {passthrough_keys}")
    log.info(f"Policy input keys: {policy_in_keys}")

    # Check coverage: obs outputs + passthrough should cover all policy inputs.
    available = set(obs_output_keys) | set(passthrough_keys)
    missing = set(policy_in_keys) - available
    if missing:
        raise ValueError(
            f"Policy requires inputs not available: {missing}. "
            f"Available from obs: {obs_output_keys}, passthrough: {passthrough_keys}"
        )

    # Step 2: Create unified module with passthrough.
    # Move everything to CPU for ONNX export.
    unified_module = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=policy_module.cpu(),
        action_processing_module=action_processing_module.cpu(),
        policy_in_keys=policy_in_keys,
        policy_action_key=policy_action_key,
        passthrough_keys=passthrough_keys,
    )
    unified_module.cpu()
    unified_module.eval()

    # All input keys: observation context + passthrough.
    all_input_keys = unified_module.get_all_input_keys()

    # Build sample inputs from context (for obs) and passthrough_obs.
    # Move all to CPU for ONNX export.
    sample_inputs = []
    input_shapes = {}

    # Observation context inputs.
    for key in obs_input_keys:
        value = eval(key, {"__builtins__": {}}, sample_context)
        if isinstance(value, torch.Tensor):
            sample_inputs.append(value.cpu())
            input_shapes[key] = list(value.shape)
        else:
            raise ValueError(f"Input '{key}' is not a tensor: {type(value)}")

    # Passthrough inputs.
    for key in passthrough_keys:
        value = passthrough_obs[key]
        sample_inputs.append(value.cpu())
        input_shapes[key] = list(value.shape)

    # Run forward pass to get output shapes.
    with torch.no_grad():
        actions, joint_pos_targets, stiffness_targets, damping_targets = (
            unified_module(*sample_inputs)
        )

    log.info(f"Actions shape: {list(actions.shape)}")
    log.info(f"Joint pos targets shape: {list(joint_pos_targets.shape)}")
    log.info(f"Stiffness targets shape: {list(stiffness_targets.shape)}")
    log.info(f"Damping targets shape: {list(damping_targets.shape)}")

    # Create ONNX names.
    def sanitize_name(name: str) -> str:
        return (
            name.replace(".", "_").replace("[", "_").replace("]", "_").replace(":", "_")
        )

    onnx_input_names = [sanitize_name(k) for k in all_input_keys]
    onnx_output_names = [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]

    # Export to ONNX.
    log.info(f"Exporting unified pipeline to {path}...")
    torch.onnx.export(
        unified_module,
        tuple(sample_inputs),
        path,
        input_names=onnx_input_names,
        output_names=onnx_output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in onnx_input_names},
            **{name: {0: "batch_size"} for name in onnx_output_names},
        },
        dynamo=False,
    )

    # Load to get actual ONNX names.
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    actual_onnx_in_names = [inp.name for inp in session.get_inputs()]
    actual_onnx_out_names = [out.name for out in session.get_outputs()]

    # Build mapping from ONNX name to semantic key.
    # Match by sanitized name similarity (ONNX may reorder inputs!).
    onnx_name_to_in_key = {}
    sanitized_to_semantic = {sanitize_name(k): k for k in all_input_keys}

    for onnx_name in actual_onnx_in_names:
        matched = False

        # Try exact match with sanitized names.
        if onnx_name in sanitized_to_semantic:
            onnx_name_to_in_key[onnx_name] = sanitized_to_semantic[onnx_name]
            matched = True
        else:
            # Try stripping ONNX suffixes (.1, .2, etc.).
            base_name = onnx_name
            for suffix in [".1", ".2", ".3", "_1", "_2", "_3"]:
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break

            if base_name in sanitized_to_semantic:
                onnx_name_to_in_key[onnx_name] = sanitized_to_semantic[base_name]
                matched = True

        if not matched:
            log.warning(f"Could not match ONNX input '{onnx_name}' to any semantic key")

    # Build output shapes dict.
    output_shapes = {
        "actions": list(actions.shape),
        "joint_pos_targets": list(joint_pos_targets.shape),
        "stiffness_targets": list(stiffness_targets.shape),
        "damping_targets": list(damping_targets.shape),
    }

    # Generate YAML content.
    yaml_content = _generate_yaml_content(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        onnx_in_names=actual_onnx_in_names,
        onnx_out_names=actual_onnx_out_names,
        joint_names=joint_names,
        body_names=body_names,
        stiffness=stiffness,
        damping=damping,
        anchor_body=anchor_body,
    )

    # Add runtime metadata for visualization/testing (not used by isaac-deploy).
    yaml_content["_runtime"] = {
        "onnx_in_names": actual_onnx_in_names,
        "onnx_out_names": actual_onnx_out_names,
        "onnx_name_to_in_key": onnx_name_to_in_key,
        "passthrough_keys": passthrough_keys,
        "obs_context_keys": obs_input_keys,
    }

    # Add metadata if provided.
    if meta:
        yaml_content["metadata"] = meta

    # Save YAML.
    yaml_path = path.replace(".onnx", ".yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)

    log.info(f"✓ Unified pipeline exported to {path}")
    log.info(f"✓ YAML configuration saved to {yaml_path}")

    # Validate with onnxruntime.
    if validate:
        try:
            import numpy as np

            log.info("Validating with onnxruntime...")

            # Build input dict.
            input_key_to_value = {
                key: inp for key, inp in zip(all_input_keys, sample_inputs)
            }
            onnx_inputs = {}
            for onnx_name in actual_onnx_in_names:
                if onnx_name in onnx_name_to_in_key:
                    semantic_key = onnx_name_to_in_key[onnx_name]
                    onnx_inputs[onnx_name] = (
                        input_key_to_value[semantic_key].detach().cpu().numpy()
                    )

            onnx_outputs = session.run(actual_onnx_out_names, onnx_inputs)

            # Compare with PyTorch outputs.
            pytorch_outputs = [
                actions.detach().cpu().numpy(),
                joint_pos_targets.detach().cpu().numpy(),
                stiffness_targets.detach().cpu().numpy(),
                damping_targets.detach().cpu().numpy(),
            ]

            for i, (onnx_name, pytorch_out) in enumerate(
                zip(onnx_output_names, pytorch_outputs)
            ):
                diff = np.abs(onnx_outputs[i] - pytorch_out).max()
                log.info(f"  {onnx_name}: max_diff = {diff:.6e}")
                if diff > 1e-4:
                    log.warning(f"  ⚠ Large difference detected for {onnx_name}")

            log.info("✓ Validation passed")

        except ImportError:
            log.warning("onnxruntime not installed, skipping validation")
        except Exception as e:
            log.error(f"Validation failed: {e}")
            raise

    return path


###############################################################################
# Observation Export Utilities
###############################################################################


class ObservationExportModule(torch.nn.Module):
    """Module that wraps observation functions for ONNX export.
    
    This module takes raw context tensors as inputs and computes observations
    by calling the configured observation functions. It's designed to be
    exported to ONNX for deployment.
    
    The module resolves all variable mappings at construction time and stores
    them as a mapping from input tensor names to function argument names.
    
    Args:
        observation_configs: Dict of observation component configurations.
        sample_context: Sample context dict to determine input shapes and resolve mappings.
        device: Device for tensor operations.
    
    Example:
        >>> from protomotions.envs.obs.mimic_obs_functions import mimic_target_poses_max_coords_factory
        >>> configs = {"mimic_target_poses": mimic_target_poses_max_coords_factory()}
        >>> context = env._get_global_context()
        >>> module = ObservationExportModule(configs, context, device)
        >>> # Export to ONNX
        >>> export_observations(module, context, "observations.onnx")
    """
    
    def __init__(
        self,
        observation_configs: Dict[str, Any],
        sample_context: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        
        # Store observation functions and their input mappings
        self._obs_functions = {}
        self._obs_input_mappings = {}  # obs_name -> {arg_name: context_key}
        self._obs_constants = {}  # obs_name -> {arg_name: value}
        self._input_keys = set()  # All unique context keys needed
        self._output_keys = []  # Ordered list of output observation names
        
        for obs_name, component in observation_configs.items():
            if component.function is None:
                continue
            
            self._obs_functions[obs_name] = component.function
            self._output_keys.append(obs_name)
            
            input_mapping = {}
            constants = {}
            
            for arg_name, var_value in component.variables.items():
                if isinstance(var_value, str):
                    # This is a context reference (flat keys like "current_state_rigid_body_pos")
                    input_mapping[arg_name] = var_value
                    self._input_keys.add(var_value)
                else:
                    # This is a constant value (int, float, bool, etc.)
                    constants[arg_name] = var_value
            
            self._obs_input_mappings[obs_name] = input_mapping
            self._obs_constants[obs_name] = constants
        
        # Convert to ordered list for consistent ONNX input ordering
        self._input_keys = sorted(list(self._input_keys))
        
        # Pre-resolve constant tensors that might be in context (like hinge_axes_map)
        self._resolved_constants = {}
        for obs_name, mapping in self._obs_input_mappings.items():
            self._resolved_constants[obs_name] = {}
            for arg_name, var_expr in list(mapping.items()):
                try:
                    # Try to resolve from context - if it's not a tensor, treat as constant
                    value = eval(var_expr, {"__builtins__": {}}, sample_context)
                    if not isinstance(value, torch.Tensor):
                        # Move from input_mapping to constants
                        self._obs_constants[obs_name][arg_name] = value
                        del self._obs_input_mappings[obs_name][arg_name]
                        self._resolved_constants[obs_name][arg_name] = value
                except Exception:
                    pass
        
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


def export_observations(
    observation_configs: Dict[str, Any],
    sample_context: Dict[str, Any],
    path: str,
    device: torch.device,
    validate: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Export observation computation to ONNX format.
    
    Creates an ObservationExportModule from the observation configs and exports
    it to ONNX. The exported model takes raw context tensors as inputs and
    produces observation tensors as outputs.
    
    Args:
        observation_configs: Dict of observation component configurations.
        sample_context: Sample context dict for tracing and shape inference.
        path: Path to save the ONNX model.
        device: Device for tensor operations.
        validate: If True, validates with onnxruntime.
        meta: Optional metadata to include in the JSON sidecar.
    
    Returns:
        Path to the exported ONNX model.
    
    Example:
        >>> configs = env.config.observation_components
        >>> context = env._get_global_context()
        >>> export_observations(configs, context, "observations.onnx", device)
    """
    import logging
    log = logging.getLogger(__name__)
    
    # Create the export module
    module = ObservationExportModule(observation_configs, sample_context, device)
    module.eval()
    
    input_keys = module.get_input_keys()
    output_keys = module.get_output_keys()
    
    log.info(f"Observation export - Input keys: {input_keys}")
    log.info(f"Observation export - Output keys: {output_keys}")
    
    # Build sample inputs from context
    sample_inputs = []
    input_shapes = {}
    for key in input_keys:
        value = eval(key, {"__builtins__": {}}, sample_context)
        if isinstance(value, torch.Tensor):
            sample_inputs.append(value)
            input_shapes[key] = list(value.shape)
        else:
            raise ValueError(f"Input '{key}' is not a tensor: {type(value)}")
    
    # Run forward pass to get output shapes
    with torch.no_grad():
        sample_outputs = module(*sample_inputs)
    
    output_shapes = {
        key: list(out.shape) for key, out in zip(output_keys, sample_outputs)
    }
    
    log.info(f"Observation export - Input shapes: {input_shapes}")
    log.info(f"Observation export - Output shapes: {output_shapes}")
    
    # Create ONNX input/output names (sanitize for ONNX compatibility)
    def sanitize_name(name: str) -> str:
        return name.replace(".", "_").replace("[", "_").replace("]", "_").replace(":", "_")
    
    onnx_input_names = [sanitize_name(k) for k in input_keys]
    onnx_output_names = [sanitize_name(k) for k in output_keys]
    
    # Export to ONNX
    log.info(f"Exporting observations to {path}...")
    torch.onnx.export(
        module,
        tuple(sample_inputs),
        path,
        input_names=onnx_input_names,
        output_names=onnx_output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in onnx_input_names},
            **{name: {0: "batch_size"} for name in onnx_output_names},
        },
        dynamo=False,
    )
    
    # Load the exported model to get ACTUAL input/output names
    # ONNX may rename inputs if there are graph-level issues
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    actual_onnx_in_names = [inp.name for inp in session.get_inputs()]
    actual_onnx_out_names = [out.name for out in session.get_outputs()]
    
    log.info(f"Requested ONNX input names: {onnx_input_names}")
    log.info(f"Actual ONNX input names: {actual_onnx_in_names}")
    
    if len(actual_onnx_in_names) != len(onnx_input_names):
        log.warning(f"ONNX has {len(actual_onnx_in_names)} inputs but we expected {len(onnx_input_names)}!")
    
    # Build mapping from actual ONNX input name to semantic key
    # This handles cases where ONNX adds suffixes like .1, .2
    # Strategy: match by stripping suffixes and finding the original key
    onnx_name_to_in_key = {}
    in_key_to_onnx_names = {key: [] for key in input_keys}  # One key may map to multiple ONNX inputs
    
    for onnx_name in actual_onnx_in_names:
        # Try to find the semantic key that matches this ONNX name
        matched = False
        # First try exact match with sanitized names
        for i, expected_name in enumerate(onnx_input_names):
            if onnx_name == expected_name:
                semantic_key = input_keys[i]
                onnx_name_to_in_key[onnx_name] = semantic_key
                in_key_to_onnx_names[semantic_key].append(onnx_name)
                matched = True
                break
        
        if not matched:
            # Try matching by stripping .1, .2, etc. suffixes
            base_name = onnx_name.rsplit(".", 1)[0] if "." in onnx_name else onnx_name
            # Also try removing trailing numbers after underscore (e.g., previous_actions_1)
            base_name_alt = base_name.rsplit("_", 1)[0] if base_name[-1].isdigit() else base_name
            
            for i, expected_name in enumerate(onnx_input_names):
                if base_name == expected_name or base_name_alt == expected_name:
                    semantic_key = input_keys[i]
                    onnx_name_to_in_key[onnx_name] = semantic_key
                    in_key_to_onnx_names[semantic_key].append(onnx_name)
                    matched = True
                    break
        
        if not matched:
            log.warning(f"Could not match ONNX input '{onnx_name}' to any semantic key")
    
    log.info(f"ONNX name to semantic key mapping: {onnx_name_to_in_key}")
    
    # Save metadata with ACTUAL ONNX names
    metadata = {
        "type": "observations",
        "in_keys": input_keys,
        "out_keys": output_keys,
        "onnx_in_names": actual_onnx_in_names,  # Use actual names
        "onnx_out_names": actual_onnx_out_names,  # Use actual names
        "onnx_name_to_in_key": onnx_name_to_in_key,  # Reverse mapping for inference
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
    }
    if meta:
        metadata.update(meta)
    
    meta_path = path.replace(".onnx", ".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"✓ Observations exported to {path}")
    log.info(f"✓ Metadata saved to {meta_path}")
    
    # Validate with onnxruntime
    if validate:
        try:
            import numpy as np
            
            log.info("Validating with onnxruntime...")
            
            # Build input dict: for each ONNX input, find the semantic key and get the value
            input_key_to_value = {key: inp for key, inp in zip(input_keys, sample_inputs)}
            onnx_inputs = {}
            for onnx_name in actual_onnx_in_names:
                if onnx_name in onnx_name_to_in_key:
                    semantic_key = onnx_name_to_in_key[onnx_name]
                    onnx_inputs[onnx_name] = input_key_to_value[semantic_key].detach().cpu().numpy()
                else:
                    log.warning(f"No value for ONNX input '{onnx_name}'")
            
            onnx_outputs = session.run(actual_onnx_out_names, onnx_inputs)
            
            # Compare with PyTorch outputs
            for i, (key, onnx_out) in enumerate(zip(output_keys, onnx_outputs)):
                pytorch_out = sample_outputs[i].detach().cpu().numpy()
                max_diff = np.abs(onnx_out - pytorch_out).max()
                log.info(f"  {key}: max_diff = {max_diff:.6e}")
                if max_diff > 1e-4:
                    log.warning(f"  ⚠ Large difference detected for {key}")
            
            log.info("✓ Validation passed")
        except ImportError:
            log.warning("onnxruntime not installed, skipping validation")
        except Exception as e:
            log.error(f"Validation failed: {e}")
            raise
    
    return path
