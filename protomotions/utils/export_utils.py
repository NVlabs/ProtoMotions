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
using torch.onnx.dynamo_export. The exported models can be used for deployment
and inference in production environments.

Key Functions:
    - export_onnx: Export a TensorDictModule to ONNX format
    - export_ppo_model: Export a trained PPO model to ONNX
"""

import torch
import json
from pathlib import Path
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from typing import Optional, Dict, Any


class ONNXExportWrapper(torch.nn.Module):
    """Wrapper for TensorDictModule that accepts ``**kwargs`` for ONNX export.

    TensorDictModules expect a TensorDict argument, but torch.onnx.dynamo_export
    unpacks inputs as kwargs. This wrapper bridges the gap.
    """

    def __init__(self, module: TensorDictModuleBase, in_keys: list):
        super().__init__()
        self.module = module
        self.in_keys = in_keys

    def forward(self, **kwargs):
        """Forward that reconstructs TensorDict from kwargs."""
        # Reconstruct TensorDict from kwargs
        batch_size = kwargs[self.in_keys[0]].shape[0]
        td = TensorDict(kwargs, batch_size=batch_size)

        # Forward through original module
        output_td = self.module(td)

        # Return tuple of outputs (ONNX expects tuple, not dict)
        return tuple(output_td[key] for key in self.module.out_keys)


@torch.inference_mode()
def export_onnx(
    module: TensorDictModuleBase,
    td: TensorDict,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    validate: bool = True,
):
    """Export a TensorDictModule to ONNX format.

    Uses torch.onnx.dynamo_export to export the module. Creates a wrapper that
    converts between TensorDict and **kwargs for ONNX compatibility.

    Args:
        module: TensorDictModule to export.
        td: Sample TensorDict input (used for tracing).
        path: Path to save the ONNX model (must end with .onnx).
        meta: Optional additional metadata to save.
        validate: If True, validates the exported model with onnxruntime.

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

    print(f"Exporting model to ONNX (PyTorch {torch.__version__})...")
    print(f"  Input keys: {module.in_keys}")
    print(f"  Output keys: {module.out_keys}")

    # Create wrapper that accepts **kwargs instead of TensorDict
    wrapper = ONNXExportWrapper(module, module.in_keys)
    wrapper.eval()

    # Export using dynamo with unpacked dict
    onnx_program = torch.onnx.dynamo_export(wrapper, **td.to_dict())
    onnx_program.save(path)
    print(f"✓ Exported ONNX model to {path}")

    # Get actual ONNX input/output names
    import onnxruntime as ort

    ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onnx_input_names = [inp.name for inp in ort_session.get_inputs()]
    onnx_output_names = [out.name for out in ort_session.get_outputs()]

    # Save metadata
    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = module.in_keys
    meta["out_keys"] = module.out_keys
    meta["in_shapes"] = [list(td[k].shape) for k in module.in_keys]

    meta["onnx_input_names"] = onnx_input_names
    meta["onnx_output_names"] = onnx_output_names
    meta["input_mapping"] = {
        onnx_name: semantic_name
        for onnx_name, semantic_name in zip(onnx_input_names, module.in_keys)
    }
    meta["output_mapping"] = {
        onnx_name: semantic_name
        for onnx_name, semantic_name in zip(onnx_output_names, module.out_keys)
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

            onnx_input = tuple(td[k] for k in module.in_keys)
            onnxruntime_input = {
                inp.name: to_numpy(v)
                for inp, v in zip(ort_session.get_inputs(), onnx_input)
            }

            ort_output = ort_session.run(None, onnxruntime_input)
            assert len(ort_output) == len(
                module.out_keys
            ), f"Output length mismatch: {len(ort_output)} vs {len(module.out_keys)}"

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

    # Export the mu network (policy mean)
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
