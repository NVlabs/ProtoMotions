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
"""Export trained model to ONNX format using real environment observations.

This script loads a trained checkpoint, creates the environment, gets real observations,
and exports the model (or submodule) to ONNX format for deployment.

Usage:
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/my_experiment/last.ckpt \\
        --simulator isaaclab \\
        [--module-name _actor] \\
        [--export-name actor] \\
        [--output custom_dir/] \\
        [--visualize] \\
        [--motion-file path/to/motion.pt]

Examples:
    # Export full model (default: saves to checkpoint_dir/compiled_models/)
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab

    # Export only actor submodule
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --module-name _actor

    # Export and visualize agent running
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --visualize

    # Export critic with custom name and directory
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --module-name _critic \\
        --export-name value_network \\
        --output custom_export_dir/

    # Export with custom motion file
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --motion-file data/motions/walk.pt

    # Export with baked-in action processing (clamp + PD/torque transform)
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --with-action-processing \\
        --visualize

    # Export observations as a separate ONNX model
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --export-observations

    # Export everything: policy, action processing, and observations
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --with-action-processing \\
        --export-observations

    # Export unified model and visualize inference
    # - Raw model predictions are stored in history buffer for observations
    # - Processed actions (clamped + PD transform) are used for simulation
    python scripts/export_model_to_onnx.py \\
        --checkpoint results/tracker/last.ckpt \\
        --simulator isaaclab \\
        --export-unified \\
        --visualize
"""

import argparse
import logging
import json
from pathlib import Path
from dataclasses import asdict

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def create_parser():
    """Create and configure the argument parser for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export trained model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file to export",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for ONNX models (default: checkpoint_dir/compiled_models/)",
    )
    parser.add_argument(
        "--module-name",
        type=str,
        default=None,
        help="Name of submodule to export (e.g., '_actor', '_critic'). If not set, exports full model.",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default=None,
        help="Name for exported model file (default: uses module-name or 'model')",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate exported models with onnxruntime",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip validation with onnxruntime",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments (affects batch size)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Run visualization loop after export (non-headless)",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for export. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--with-action-processing",
        action="store_true",
        default=False,
        help="Bake action processing (clamp + PD/torque transform) into the ONNX model. "
        "Output will be PD targets or torques instead of raw actions.",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default=None,
        help="Key for action output from model (default: auto-detect 'mean_action' or 'action'). "
        "Only used with --with-action-processing.",
    )
    parser.add_argument(
        "--export-observations",
        action="store_true",
        default=False,
        help="Also export observation computation as a separate ONNX model. "
        "This allows observations to be computed outside of Python.",
    )
    parser.add_argument(
        "--export-unified",
        action="store_true",
        default=False,
        help="Export a single unified ONNX model (context -> actions). "
        "Combines observations, policy, and action processing into one model. "
        "Outputs: (actions, post_processed_actions).",
    )
    return parser


# Parse arguments
parser = create_parser()
args = parser.parse_args()

# Import simulator before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import torch and other modules
import torch  # noqa: E402
from lightning.fabric import Fabric  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402

log.info(f"Exporting model from checkpoint: {args.checkpoint}")
log.info(f"Output directory: {args.output}")

# Load configs from resolved_configs_inference.pt (like inference_agent.py)
# This contains eval overrides applied (e.g., domain randomization disabled)
checkpoint_path = Path(args.checkpoint)
resolved_configs_path = checkpoint_path.parent / "resolved_configs_inference.pt"

if not resolved_configs_path.exists():
    log.warning(
        f"resolved_configs_inference.pt not found, falling back to resolved_configs.pt. "
        "Domain randomization (observation noise, push) may still be active!"
    )
    resolved_configs_path = checkpoint_path.parent / "resolved_configs.pt"

if not resolved_configs_path.exists():
    log.error(
        f"Could not find resolved_configs.pt or resolved_configs_inference.pt in {checkpoint_path.parent}"
    )
    exit(1)

log.info(f"Loading configs from {resolved_configs_path}")
resolved_configs = torch.load(
    resolved_configs_path, map_location="cpu", weights_only=False
)

robot_config = resolved_configs["robot"]
simulator_config = resolved_configs["simulator"]
env_config = resolved_configs["env"]
agent_config = resolved_configs["agent"]
terrain_config = resolved_configs["terrain"]
scene_lib_config = resolved_configs["scene_lib"]
motion_lib_config = resolved_configs["motion_lib"]

# Check if we need to switch simulators
# Extract simulator name from current config's _target_
current_simulator = simulator_config._target_.split(".")[
    -3
]  # e.g., "isaacgym" from "protomotions.simulator.isaacgym.simulator.IsaacGymSimulator"

if args.simulator != current_simulator:
    log.info(
        f"Switching simulator from '{current_simulator}' (training) to '{args.simulator}' (export)"
    )
    from protomotions.simulator.factory import update_simulator_config_for_test

    simulator_config = update_simulator_config_for_test(
        current_simulator_config=simulator_config,
        new_simulator=args.simulator,
        robot_config=robot_config,
    )

# Override simulator and num_envs
log.info(f"Overriding num_envs: {args.num_envs}")
env_config.num_envs = args.num_envs
simulator_config.num_envs = args.num_envs  # Also set simulator num_envs
simulator_config.headless = not args.visualize  # Headless unless visualizing

if args.motion_file is not None:
    log.info(f"CLI override: motion_file = {args.motion_file}")
    motion_lib_config.motion_file = args.motion_file

# Create fabric (minimal setup for export)
fabric_config = FabricConfig(
    devices=1,
    num_nodes=1,
    loggers=[],
    callbacks=[],
)
fabric: Fabric = Fabric(**asdict(fabric_config))
fabric.launch()

# Setup IsaacLab simulation_app if using IsaacLab simulator
simulator_extra_params = {}
if args.simulator == "isaaclab":
    app_launcher_flags = {
        "headless": simulator_config.headless,
        "device": str(fabric.device),
    }
    app_launcher = AppLauncher(app_launcher_flags)
    simulator_extra_params["simulation_app"] = app_launcher.app

# Convert friction settings for simulator compatibility
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator  # noqa: E402

terrain_config, simulator_config = convert_friction_for_simulator(
    terrain_config, simulator_config
)

# Create components
from protomotions.utils.component_builder import build_all_components  # noqa: E402

save_dir_for_weights = (
    getattr(env_config, "save_dir", None) if hasattr(env_config, "save_dir") else None
)
components = build_all_components(
    terrain_config=terrain_config,
    scene_lib_config=scene_lib_config,
    motion_lib_config=motion_lib_config,
    simulator_config=simulator_config,
    robot_config=robot_config,
    device=fabric.device,
    save_dir=save_dir_for_weights,
    **simulator_extra_params,
)
terrain = components["terrain"]
scene_lib = components["scene_lib"]
motion_lib = components["motion_lib"]
simulator = components["simulator"]

# Create environment
log.info("Creating environment...")
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402

EnvClass = get_class(env_config._target_)
env: BaseEnv = EnvClass(
    config=env_config,
    robot_config=robot_config,
    device=fabric.device,
    terrain=terrain,
    scene_lib=scene_lib,
    motion_lib=motion_lib,
    simulator=simulator,
)

# Create agent
log.info("Creating agent...")
checkpoint_path = Path(args.checkpoint)
agent_kwargs = {"root_dir": checkpoint_path.parent}

from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402

AgentClass = get_class(agent_config._target_)
agent: BaseAgent = AgentClass(
    config=agent_config, env=env, fabric=fabric, **agent_kwargs
)

# Setup and load checkpoint
log.info("Loading checkpoint...")
agent.setup()
agent.load(args.checkpoint, load_env=False)
agent.eval()  # Set to evaluation mode

# Get real observations from environment
log.info("Getting real observations from environment...")
env.reset()
obs = env.get_obs()
obs = agent.add_agent_info_to_obs(obs)
obs_dict_td = agent.obs_dict_to_tensordict(obs)

log.info(f"Observation keys: {list(obs_dict_td.keys())}")
for key, val in obs_dict_td.items():
    log.info(f"  {key}: shape={val.shape}, dtype={val.dtype}")

# Determine output directory (default: checkpoint_dir/compiled_models/)
if args.output is None:
    output_dir = checkpoint_path.parent / "compiled_models"
    log.info(f"No output directory specified, using default: {output_dir}")
else:
    output_dir = Path(args.output)

output_dir.mkdir(parents=True, exist_ok=True)
log.info(f"Output directory: {output_dir}")

# Unwrap model from DDP/Fabric wrappers for export
# ONNX can't handle distributed operations
unwrapped_model = agent.model
if hasattr(unwrapped_model, "module"):
    # Unwrap from Fabric wrapper
    unwrapped_model = unwrapped_model.module

log.info(f"Unwrapped model type: {type(unwrapped_model).__name__}")

# Determine which module to export
if args.module_name:
    # Export specific submodule
    log.info(f"Exporting submodule: {args.module_name}")
    try:
        submodule = getattr(unwrapped_model, args.module_name)
        if not isinstance(submodule, torch.nn.Module):
            raise ValueError(f"{args.module_name} is not a module")
    except AttributeError:
        log.error(f"✗ Model does not have attribute '{args.module_name}'")
        log.error(
            f"Available attributes: {[attr for attr in dir(unwrapped_model) if not attr.startswith('__')]}"
        )
        exit(1)

    module_to_export = submodule
    export_name = args.export_name or args.module_name.lstrip("_")
else:
    # Export full model
    log.info("Exporting full model")
    module_to_export = unwrapped_model
    export_name = args.export_name or "model"

# Import export utilities
from protomotions.utils.export_utils import (  # noqa: E402
    export_onnx,
    export_with_action_processing,
    export_observations,
    export_unified_pipeline,
    ActionProcessingModule,
)

# Export separate models (skip if unified is requested)
if not args.export_unified:
    log.info("=" * 60)
    log.info(f"Exporting: {export_name}")
    if args.with_action_processing:
        log.info("  (with baked-in action processing)")
    log.info("=" * 60)

    try:
        export_path = str(output_dir / f"{export_name}.onnx")

        if args.with_action_processing:
            # Determine action key (auto-detect if not specified)
            if args.action_key:
                action_key = args.action_key
            else:
                # Auto-detect: prefer mean_action for PPO actors, fall back to action
                if hasattr(module_to_export, "out_keys"):
                    out_keys = list(module_to_export.out_keys)
                    if "mean_action" in out_keys:
                        action_key = "mean_action"
                    elif "action" in out_keys:
                        action_key = "action"
                    else:
                        raise ValueError(
                            f"Could not auto-detect action key. Available: {out_keys}. "
                            "Use --action-key to specify."
                        )
                else:
                    action_key = "action"  # Default fallback

            log.info(f"  Action key: {action_key}")
            log.info(f"  Control type: {robot_config.control.control_type.name}")
            log.info(f"  Clamp value: {robot_config.control.clamp_actions}")

            # Get action processing parameters from simulator
            pd_action_offset = simulator._common_pd_action_offset
            pd_action_scale = simulator._common_pd_action_scale
            torque_limits = simulator._torque_limits_common

            # Export TWO separate models:
            # 1. Policy model (obs -> raw actions)
            # 2. Action processing model (raw actions -> PD targets/torques)

            policy_path = str(output_dir / f"{export_name}_policy.onnx")
            action_proc_path = str(output_dir / f"{export_name}_action_processing.onnx")

            log.info("\n--- Exporting Policy Model ---")
            export_onnx(
                module=module_to_export,
                td=obs_dict_td,
                path=policy_path,
                meta={
                    "module_name": args.module_name or "full_model",
                    "checkpoint": str(args.checkpoint),
                    "action_key": action_key,
                },
                validate=args.validate,
            )

            log.info("\n--- Exporting Action Processing Model ---")
            from protomotions.utils.export_utils import export_action_processing
            export_action_processing(
                control_type=robot_config.control.control_type,
                clamp_value=robot_config.control.clamp_actions,
                pd_action_offset=pd_action_offset,
                pd_action_scale=pd_action_scale,
                torque_limits=torque_limits,
                num_actions=robot_config.number_of_actions,
                path=action_proc_path,
                batch_size=obs_dict_td.batch_size[0],
                meta={
                    "checkpoint": str(args.checkpoint),
                },
                validate=args.validate,
            )

            log.info(f"\n✓ Policy model exported to {policy_path}")
            log.info(f"✓ Action processing model exported to {action_proc_path}")

        else:
            export_onnx(
                module=module_to_export,
                td=obs_dict_td,
                path=export_path,
                meta={
                    "module_name": args.module_name or "full_model",
                    "checkpoint": str(args.checkpoint),
                },
                validate=args.validate,
            )
            log.info(f"✓ Model exported successfully to {export_path}")
            log.info(f"✓ Metadata saved to {export_path.replace('.onnx', '.json')}")
    except Exception as e:
        log.error(f"✗ Export failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

# Export observations if requested
if args.export_observations:
    log.info("\n" + "=" * 60)
    log.info("Exporting Observation Computation")
    log.info("=" * 60)
    
    try:
        # Get the global context from the environment
        context = env._get_global_context()
        
        # Get observation configs
        observation_configs = env.config.observation_components
        
        if not observation_configs:
            log.warning("No observation components configured, skipping observation export")
        else:
            obs_export_path = str(output_dir / "observations.onnx")
            
            log.info(f"Observation components to export: {list(observation_configs.keys())}")
            
            export_observations(
                observation_configs=observation_configs,
                sample_context=context,
                path=obs_export_path,
                device=fabric.device,
                validate=args.validate,
                meta={
                    "checkpoint": str(args.checkpoint),
                    "observation_components": list(observation_configs.keys()),
                },
            )
            
            log.info(f"✓ Observations exported to {obs_export_path}")
    except Exception as e:
        log.error(f"✗ Observation export failed: {e}")
        import traceback
        traceback.print_exc()

# Export unified pipeline if requested
unified_path = None
if args.export_unified:
    log.info("\n" + "=" * 60)
    log.info("Exporting Unified Pipeline (Context -> Actions)")
    log.info("=" * 60)
    
    try:
        # Get context and observation configs
        context = env._get_global_context()
        observation_configs = env.config.observation_components
        
        if not observation_configs:
            log.error("No observation components configured, cannot create unified model")
        else:
            # Get policy module - models with _actor (PPO) vs models without (MaskedMimic)
            if hasattr(unwrapped_model, "_actor"):
                policy_module = unwrapped_model._actor
                policy_in_keys = list(policy_module.in_keys)
            elif hasattr(unwrapped_model, "forward_inference"):
                # MaskedMimic: use inference-only forward (prior path, no encoder)
                from protomotions.agents.masked_mimic.model import MaskedMimicModel
                
                # Create a wrapper that uses forward_inference
                class InferenceWrapper(torch.nn.Module):
                    def __init__(self, model: MaskedMimicModel):
                        super().__init__()
                        self.model = model
                        self.in_keys = model.get_inference_in_keys()
                        self.out_keys = ["action"]
                    
                    def forward(self, tensordict):
                        return self.model.forward_inference(tensordict)
                
                policy_module = InferenceWrapper(unwrapped_model)
                policy_in_keys = list(policy_module.in_keys)
                log.info("Using MaskedMimic inference path (prior only, no encoder)")
            else:
                # Other models use the model itself as policy
                policy_module = unwrapped_model
                policy_in_keys = list(policy_module.in_keys)
            
            # Filter observation configs to only those needed by the policy
            # This excludes encoder-only observations for MaskedMimic inference
            obs_output_keys = list(observation_configs.keys())
            needed_obs_keys = set(policy_in_keys) & set(obs_output_keys)
            filtered_observation_configs = {
                k: v for k, v in observation_configs.items() if k in needed_obs_keys
            }
            
            log.info(f"Policy needs {len(needed_obs_keys)} observations from {len(observation_configs)} total")
            excluded = set(obs_output_keys) - needed_obs_keys
            if excluded:
                log.info(f"Excluding encoder-only observations: {sorted(excluded)}")
            
            missing = set(policy_in_keys) - set(obs_output_keys)
            
            # Get sample observations to find passthrough tensors
            sample_obs = env.get_obs()
            sample_obs = agent.add_agent_info_to_obs(sample_obs)
            
            # Collect passthrough observations (inputs needed by policy but not from obs configs)
            passthrough_obs = {}
            if missing:
                log.info(f"Policy requires additional inputs not in observation configs:")
                for key in missing:
                    if key in sample_obs:
                        log.info(f"  - {key} (will be passthrough input)")
                        passthrough_obs[key] = sample_obs[key]
                    else:
                        log.error(f"  - {key} (NOT AVAILABLE - cannot export)")
                
                # Check if all missing are available as passthrough
                still_missing = missing - set(passthrough_obs.keys())
                if still_missing:
                    log.error(f"Cannot create unified model - missing inputs: {still_missing}")
                    raise ValueError(f"Missing inputs: {still_missing}")
            
            # Determine action key
            if hasattr(policy_module, "out_keys"):
                out_keys = list(policy_module.out_keys)
                if "mean_action" in out_keys:
                    action_key = "mean_action"
                elif "action" in out_keys:
                    action_key = "action"
                else:
                    action_key = out_keys[0]
            else:
                action_key = "action"
            
            # Build stiffness and damping tensors from robot config.
            joint_names = robot_config.kinematic_info.dof_names
            stiffness = torch.tensor(
                [
                    robot_config.control.control_info[j].stiffness
                    for j in joint_names
                ],
                dtype=torch.float32,
            )
            damping = torch.tensor(
                [
                    robot_config.control.control_info[j].damping
                    for j in joint_names
                ],
                dtype=torch.float32,
            )

            # Create action processing module with stiffness/damping.
            action_proc = ActionProcessingModule(
                control_type=robot_config.control.control_type,
                clamp_value=robot_config.control.clamp_actions,
                pd_action_offset=simulator._common_pd_action_offset,
                pd_action_scale=simulator._common_pd_action_scale,
                torque_limits=simulator._torque_limits_common,
                stiffness=stiffness,
                damping=damping,
            )

            unified_path = str(output_dir / "unified_pipeline.onnx")

            export_unified_pipeline(
                observation_configs=filtered_observation_configs,
                sample_context=context,
                policy_module=policy_module,
                policy_in_keys=policy_in_keys,
                policy_action_key=action_key,
                action_processing_module=action_proc,
                path=unified_path,
                device=fabric.device,
                robot_config=robot_config,
                passthrough_obs=passthrough_obs,
                validate=args.validate,
                meta={
                    "checkpoint": str(args.checkpoint),
                    "control_type": robot_config.control.control_type.name,
                },
            )
            
            log.info(f"✓ Unified pipeline exported to {unified_path}")
    except Exception as e:
        log.error(f"✗ Unified pipeline export failed: {e}")
        import traceback
        traceback.print_exc()

log.info("\n" + "=" * 60)
log.info("Export Complete!")
log.info("=" * 60)
if args.export_unified:
    if unified_path:
        log.info(f"Unified pipeline exported to: {unified_path}")
else:
    log.info(f"Model exported to: {output_dir / export_name}.onnx")
if args.export_observations:
    log.info(f"Observations exported to: {output_dir / 'observations.onnx'}")

# Visualization loop with validation (if requested)
if args.visualize:
    # Disable env-level action processing for proper inference
    # ONNX models handle action processing externally, so we:
    # 1. Disable action clamping (ONNX handles it)
    # 2. Make PD offset/scale identity (ONNX provides pre-computed PD targets)
    original_clamp = robot_config.control.clamp_actions
    robot_config.control.clamp_actions = None

    original_pd_offset = simulator._common_pd_action_offset.clone()
    original_pd_scale = simulator._common_pd_action_scale.clone()

    simulator._common_pd_action_offset = torch.zeros_like(simulator._common_pd_action_offset)
    simulator._common_pd_action_scale = torch.ones_like(simulator._common_pd_action_scale)

    log.info("\n" + "=" * 60)
    log.info("Starting Visualization Loop")
    log.info("Press Ctrl+C to exit")
    log.info("=" * 60)
    log.info("Inference mode: env-level action processing disabled")
    log.info(f"  Clamping: {original_clamp} -> None")
    log.info(f"  PD offset: {original_pd_offset.mean().item():.4f} -> 0.0")
    log.info(f"  PD scale: {original_pd_scale.mean().item():.4f} -> 1.0")

    import onnxruntime as ort
    import numpy as np

    # Determine which mode to use (priority: unified > separate models)
    unified_session = None
    unified_metadata = None
    
    if args.export_unified and unified_path:
        # Use unified pipeline model.
        unified_yaml_path = unified_path.replace(".onnx", ".yaml")
        log.info(f"Loading unified pipeline from: {unified_path}")
        unified_session = ort.InferenceSession(
            unified_path, providers=["CPUExecutionProvider"]
        )

        import yaml

        with open(unified_yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        # Extract runtime metadata for visualization.
        runtime = yaml_content.get("_runtime", {})
        unified_metadata = {
            "onnx_in_names": runtime.get("onnx_in_names", []),
            "onnx_out_names": runtime.get("onnx_out_names", []),
            "onnx_name_to_in_key": runtime.get("onnx_name_to_in_key", {}),
            "passthrough_keys": runtime.get("passthrough_keys", []),
            "obs_context_keys": runtime.get("obs_context_keys", []),
        }

        log.info(f"Unified input keys: {unified_metadata['onnx_in_names']}")
        log.info(f"Unified output keys: {unified_metadata['onnx_out_names']}")
        log.info("Mode: UNIFIED (Context -> Actions in one model)")
    else:
        log.info("Mode: SEPARATE MODELS (not implemented in unified loop)")
        log.info("Use --export-unified flag for the simplified visualization loop")
        unified_session = None

    if unified_session is not None:
        # Simple unified loop
        max_action_diff = 0.0
        step_count = 0
        
        # Get passthrough keys from metadata
        passthrough_keys = set(unified_metadata.get("passthrough_keys", []))
        obs_context_keys = set(unified_metadata.get("obs_context_keys", []))
        
        # Log inference pipeline info before starting
        log.info("\n✓ Inference pipeline:")
        log.info("    - Raw model predictions stored in history buffer")
        log.info("    - Processed actions (PD targets) used for simulation")
        if passthrough_keys:
            log.info(f"\nNote: Passthrough inputs (from Python env.get_obs()):")
            for key in passthrough_keys:
                log.info(f"    - {key}")
        
        try:
            while True:
                # Get context from environment (for observation computation inputs)
                context = env._get_global_context()
                
                # Get observations (for passthrough inputs like historical obs)
                python_obs = env.get_obs()
                python_obs = agent.add_agent_info_to_obs(python_obs)
                
                # Build ONNX inputs
                onnx_name_to_key = unified_metadata.get("onnx_name_to_in_key", {})
                onnx_inputs = {}
                for onnx_name in unified_metadata["onnx_in_names"]:
                    if onnx_name in onnx_name_to_key:
                        semantic_key = onnx_name_to_key[onnx_name]
                        
                        if semantic_key in passthrough_keys:
                            # Get from observations (passthrough)
                            value = python_obs[semantic_key]
                        else:
                            # Get from context (for obs computation)
                            value = eval(semantic_key, {"__builtins__": {}}, context)
                        
                        onnx_inputs[onnx_name] = value.detach().cpu().numpy()
                
                # Run unified ONNX model.
                onnx_out_names = unified_metadata["onnx_out_names"]
                onnx_outputs = unified_session.run(onnx_out_names, onnx_inputs)

                # Get actions from ONNX outputs:
                # Output 0: raw actions (model predictions)
                # Output 1: joint_pos_targets (clamped + PD transform)
                # Output 2: stiffness_targets (constant)
                # Output 3: damping_targets (constant)
                raw_actions_onnx = onnx_outputs[0]
                post_processed_onnx = onnx_outputs[1]
                # stiffness_targets = onnx_outputs[2]  # Available if needed.
                # damping_targets = onnx_outputs[3]    # Available if needed.
                
                # Get Python reference for validation
                obs_dict_td = agent.obs_dict_to_tensordict(python_obs)
                
                # Use inference forward for MaskedMimic, regular forward for others
                with torch.no_grad():
                    if hasattr(unwrapped_model, "forward_inference"):
                        pytorch_output = unwrapped_model.forward_inference(obs_dict_td.cpu())
                    elif hasattr(unwrapped_model, "_actor"):
                        pytorch_output = unwrapped_model._actor(obs_dict_td.cpu())
                    else:
                        pytorch_output = unwrapped_model(obs_dict_td.cpu())
                
                # Find action key
                if "mean_action" in pytorch_output.keys():
                    pytorch_actions = pytorch_output["mean_action"].numpy()
                else:
                    pytorch_actions = pytorch_output["action"].numpy()
                
                # Validate raw actions match PyTorch
                action_diff = np.abs(raw_actions_onnx - pytorch_actions).max()
                if action_diff > max_action_diff:
                    max_action_diff = action_diff
                    if action_diff > 1e-5:
                        log.warning(f"Step {step_count}: Action diff = {action_diff:.6e}")
                
                # Use processed actions (PD targets) for simulation
                # Raw actions go into history buffer for observations
                step_actions = torch.from_numpy(post_processed_onnx).to(fabric.device)
                raw_actions_tensor = torch.from_numpy(raw_actions_onnx).to(fabric.device)
                
                # Step environment with processed actions
                _, _, dones, _, _ = env.step(step_actions)
                
                # Override historical actions with raw model predictions
                # This ensures observations see raw actions, not processed ones
                if env.state_history is not None:
                    env.state_history.actions[:, 0] = raw_actions_tensor
                
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(done_indices) > 0:
                    log.info(f"Step {step_count}: Resetting {len(done_indices)} env(s)")
                env.reset(done_indices)
                step_count += 1
                
                if step_count % 100 == 0:
                    log.info(f"Step {step_count}: max_action_diff = {max_action_diff:.6e}")
        
        except KeyboardInterrupt:
            log.info(f"\nVisualization interrupted.")
            log.info(f"  Steps completed: {step_count}")
            log.info(f"  Max action diff: {max_action_diff:.6e}")

# Cleanup
if args.simulator == "isaaclab" and "simulation_app" in simulator_extra_params:
    simulator_extra_params["simulation_app"].close()

log.info("\nYou can now use these ONNX models for deployment!")
