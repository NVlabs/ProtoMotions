# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure functions for action processing.

Action functions transform raw policy outputs into simulator-ready actions.
They take the action tensor and control parameters, returning processed actions
with control gains (stiffness, damping).

Basic Usage
-----------

In experiment configs, use the helper to create an action config:

    from protomotions.envs.action import make_pd_action_config

    action_config = make_pd_action_config(robot_cfg)

The helpers live in ``protomotions.envs.action.config`` and return an
``ActionConfig`` (fn plus the bound parameters), processed by the ActionStage.

Adding New ONNX-Compilable Action Functions
--------------------------------------------

If you need custom action processing (e.g., learned gains, compliance control),
create a new function following this pattern:

1. **Function Signature** - Must take `action` first, then other parameters:

    def my_custom_action(
        action: Tensor,                    # Raw policy output [num_envs, action_dim]
        pd_action_offset: Tensor,          # Control parameters [action_dim]
        my_custom_param: float,            # Your custom parameters
        ...
    ) -> Dict[str, Tensor]:

2. **Return Dict** - Must return these exact keys for ONNX export:

    return {
        "processed_action": joint_pos_targets,   # [num_envs, num_actions]
        "stiffness_targets": stiffness_values,   # [num_envs, num_actions]
        "damping_targets": damping_values,       # [num_envs, num_actions]
    }

3. **ONNX Compatibility** - Keep it tensor-only:
    - All inputs/outputs must be Tensors (no lists, dicts in logic)
    - Avoid control flow that depends on tensor values (if/while on tensor data)
    - Use torch operations throughout

4. **Configuration** - Create your config dict:

    action_config = {
        "fn": my_custom_action,
        "pd_action_offset": compute_offset(...),
        "my_custom_param": 0.5,
        ...
    }

5. **ONNX Export** - The ActionExportModule in export_utils.py will:
    - Extract "fn" and call it with action + all other dict keys as kwargs
    - Register tensor parameters as buffers
    - Store scalar/list parameters as constants
    - Export the entire processing graph to ONNX

Example: Compliance Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def compliance_pd_action(
        action: Tensor,                    # [num_envs, action_dim]
        pd_action_offset: Tensor,          # [action_dim]
        base_stiffness: Tensor,            # [action_dim]
        base_damping: Tensor,              # [action_dim]
        compliance_scale: float = 0.5,     # How much to reduce gains
    ) -> Dict[str, Tensor]:
        '''Compliant PD action with reduced gains for safer interaction.'''

        batch_size = action.shape[0]

        # Tanh normalize
        action = torch.tanh(action)

        # Compute position targets
        processed_action = pd_action_offset + action

        # Reduce gains for compliance
        stiffness = (base_stiffness * compliance_scale).unsqueeze(0).expand(batch_size, -1)
        damping = (base_damping * compliance_scale).unsqueeze(0).expand(batch_size, -1)

        return {
            "processed_action": processed_action.clone(),
            "stiffness_targets": stiffness.clone(),
            "damping_targets": damping.clone(),
        }

    # Use in config:
    action_config = {
        "fn": compliance_pd_action,
        "pd_action_offset": ...,
        "base_stiffness": ...,
        "base_damping": ...,
        "compliance_scale": 0.3,
    }

Why Action Processing is Different from Obs/Rew/Term
------------------------------------------------------

Unlike observations, rewards, and terminations:

1. **No MdpComponent** - Action doesn't bind to context paths (it transforms
   policy output, not environment state)

2. **Single function** - Unlike obs/rew/term which have multiple components,
   action is always singular (one action processor per environment)

3. **Direct parameters** - Control parameters (stiffness, damping, offsets)
   are passed directly from robot config, not from environment context
"""

from typing import Dict, Literal
import numpy as np
import torch
from torch import Tensor

ActionTransform = Literal["clamp", "tanh", None]


# =============================================================================
# Action Processing Functions
# =============================================================================


def normalized_pd_fixed_gains_action(
    action: Tensor,
    pd_action_offset: Tensor,
    pd_action_scale: Tensor,
    stiffness: Tensor,
    damping: Tensor,
    action_transform: ActionTransform = "tanh",
    clamp_value: float = 1.0,
) -> Dict[str, Tensor]:
    """Transform normalized action to PD position targets with fixed gains.

    Uses constant stiffness/damping values from robot config (not learned or dynamic).

    Action transform options:
        - "tanh": Apply tanh to bound actions to [-1, 1] (default, smooth gradients)
        - "clamp": Hard clamp to [-clamp_value, clamp_value]
        - None: No transform (raw policy output)

    Args:
        action: Raw action tensor from policy [num_envs, num_actions]
        pd_action_offset: Per-joint offset (joint default positions) [num_actions]
        pd_action_scale: Per-joint scale (action range) [num_actions]
        stiffness: Fixed per-joint stiffness gains [num_actions]
        damping: Fixed per-joint damping gains [num_actions]
        action_transform: How to bound actions - "tanh", "clamp", or None.
        clamp_value: Max absolute action value for clamp mode. Default 1.0.

    Returns:
        Dict with:
            - processed_action: Joint position targets [num_envs, num_actions]
            - stiffness_targets: Fixed stiffness values [num_envs, num_actions]
            - damping_targets: Fixed damping values [num_envs, num_actions]
    """
    if action_transform == "tanh":
        action = torch.tanh(action)
    elif action_transform == "clamp":
        action = torch.clamp(action, -clamp_value, clamp_value)

    batch_size = action.shape[0]
    processed_action = pd_action_offset + pd_action_scale * action

    # Clone outputs for CUDA graphs compatibility
    return {
        "processed_action": processed_action.clone(),
        "stiffness_targets": stiffness.unsqueeze(0).expand(batch_size, -1).clone(),
        "damping_targets": damping.unsqueeze(0).expand(batch_size, -1).clone(),
    }


def passthrough_pd_action(
    action: Tensor,
    stiffness: Tensor,
    damping: Tensor,
) -> Dict[str, Tensor]:
    """Pass raw policy output through as joint position targets without any transformation.

    No tanh, clamp, offset, or scale is applied. Useful when the policy already
    outputs joint position targets directly (e.g., from a motion library or
    external planner).

    Args:
        action: Raw action tensor from policy [num_envs, num_actions].
            Interpreted directly as joint position targets.
        stiffness: Fixed per-joint stiffness gains [num_actions]
        damping: Fixed per-joint damping gains [num_actions]

    Returns:
        Dict with:
            - processed_action: Joint position targets (unchanged) [num_envs, num_actions]
            - stiffness_targets: Fixed stiffness values [num_envs, num_actions]
            - damping_targets: Fixed damping values [num_envs, num_actions]
    """
    batch_size = action.shape[0]

    # Clone outputs for CUDA graphs compatibility
    return {
        "processed_action": action.clone(),
        "stiffness_targets": stiffness.unsqueeze(0).expand(batch_size, -1).clone(),
        "damping_targets": damping.unsqueeze(0).expand(batch_size, -1).clone(),
    }


def build_pd_action_offset_scale(
    hinge_axes_map, dof_limits_lower, dof_limits_upper, action_scale, device
):
    sorted_body_ids = list(hinge_axes_map.keys())
    sorted_body_ids.sort()

    # Clone tensors to avoid modifying the original kinematic_info limits
    lim_low = dof_limits_lower.clone().cpu().numpy()
    lim_high = dof_limits_upper.clone().cpu().numpy()

    dof_offset = 0

    for body_id in sorted_body_ids:
        dof_size = len(hinge_axes_map[body_id])

        if dof_size == 3:
            curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
            curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
            curr_low = np.max(np.abs(curr_low))
            curr_high = np.max(np.abs(curr_high))
            curr_scale = max([curr_low, curr_high])
            curr_scale = 2 * action_scale * curr_scale
            curr_scale = min([curr_scale, np.pi])

            lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
            lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

        elif dof_size == 1:
            curr_low = lim_low[dof_offset]
            curr_high = lim_high[dof_offset]
            curr_mid = 0.5 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = action_scale * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[dof_offset] = curr_low
            lim_high[dof_offset] = curr_high

        else:
            raise ValueError(f"Invalid dof size: {dof_size}")

        dof_offset += dof_size

    pd_action_offset = 0.5 * (lim_high + lim_low)
    pd_action_scale = 0.5 * (lim_high - lim_low)
    pd_action_offset = torch.tensor(pd_action_offset, device=device)
    pd_action_scale = torch.tensor(pd_action_scale, device=device)

    return pd_action_offset, pd_action_scale


def bm_pd_action(
    action: Tensor,
    pd_action_offset: Tensor,
    action_scale: Tensor,
    stiffness: Tensor,
    damping: Tensor,
) -> Dict[str, Tensor]:
    """BeyondMimic-style action scaling for implicit PD control.

    Scales raw policy output by ``0.25 * effort_limit / stiffness`` per joint,
    so ~O(1) policy outputs map to ~25% of the maximum achievable deflection.
    No tanh or clamp is applied — the policy learns its own output magnitude.

    Args:
        action: Raw action tensor from policy [num_envs, num_actions]
        pd_action_offset: Per-joint default positions [num_actions]
        action_scale: Per-joint scale ``0.25 * effort_limit / stiffness`` [num_actions]
        stiffness: Fixed per-joint stiffness gains [num_actions]
        damping: Fixed per-joint damping gains [num_actions]

    Returns:
        Dict with:
            - processed_action: Joint position targets [num_envs, num_actions]
            - stiffness_targets: Fixed stiffness values [num_envs, num_actions]
            - damping_targets: Fixed damping values [num_envs, num_actions]
    """
    batch_size = action.shape[0]
    processed_action = pd_action_offset + action_scale * action

    # Clone outputs for CUDA graphs compatibility
    return {
        "processed_action": processed_action.clone(),
        "stiffness_targets": stiffness.unsqueeze(0).expand(batch_size, -1).clone(),
        "damping_targets": damping.unsqueeze(0).expand(batch_size, -1).clone(),
    }
