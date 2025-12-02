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
        [--visualize]

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
"""

import argparse
import logging
import json
from pathlib import Path

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

# Load configs from resolved_configs.pt (like eval_agent.py)
checkpoint_path = Path(args.checkpoint)
resolved_configs_path = checkpoint_path.parent / "resolved_configs_eval.pt"

if not resolved_configs_path.exists():
    resolved_configs_path = checkpoint_path.parent / "resolved_configs.pt"

if not resolved_configs_path.exists():
    log.error(
        f"Could not find resolved_configs.pt or resolved_configs_eval.pt in {checkpoint_path.parent}"
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
    from protomotions.simulator.factory import update_simulator_config_for_eval

    simulator_config = update_simulator_config_for_eval(
        current_simulator_config=simulator_config,
        new_simulator=args.simulator,
        robot_config=robot_config,
    )

# Override simulator and num_envs
log.info(f"Overriding num_envs: {args.num_envs}")
env_config.num_envs = args.num_envs
simulator_config.num_envs = args.num_envs  # Also set simulator num_envs
simulator_config.headless = not args.visualize  # Headless unless visualizing

# Create fabric (minimal setup for export)
fabric_config = FabricConfig(
    devices=1,
    num_nodes=1,
    loggers=[],
    callbacks=[],
)
fabric: Fabric = Fabric(**fabric_config.to_dict())
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
from protomotions.utils.export_utils import export_onnx  # noqa: E402

# Export
log.info("=" * 60)
log.info(f"Exporting: {export_name}")
log.info("=" * 60)
try:
    export_path = str(output_dir / f"{export_name}.onnx")
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

log.info("\n" + "=" * 60)
log.info("Export Complete!")
log.info("=" * 60)
log.info(f"Model exported to: {output_dir / export_name}.onnx")

# Visualization loop (if requested)
if args.visualize:
    log.info("\n" + "=" * 60)
    log.info("Starting Visualization Loop with ONNX Model")
    log.info("Press Ctrl+C to exit")
    log.info("=" * 60)

    # Load the exported ONNX model
    import onnxruntime as ort

    onnx_path = str(output_dir / f"{export_name}.onnx")
    meta_path = str(output_dir / f"{export_name}.json")

    log.info(f"Loading ONNX model from: {onnx_path}")
    ort_session = ort.InferenceSession(onnx_path)

    # Load metadata for mappings
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Get input/output names from ONNX
    onnx_input_names = [inp.name for inp in ort_session.get_inputs()]
    onnx_output_names = [out.name for out in ort_session.get_outputs()]

    # Get semantic names from metadata
    semantic_in_keys = metadata.get("in_keys", [])
    semantic_out_keys = metadata.get("out_keys", [])

    log.info(f"Semantic input keys: {semantic_in_keys}")
    log.info(f"ONNX input names: {onnx_input_names}")
    log.info(f"Semantic output keys: {semantic_out_keys}")
    log.info(f"ONNX output names: {onnx_output_names}")

    if "output_mapping" in metadata:
        log.info(f"Output mapping: {metadata['output_mapping']}")

    try:
        step_count = 0
        while True:
            # Get observations
            obs = env.get_obs()
            obs = agent.add_agent_info_to_obs(obs)
            obs_dict_td = agent.obs_dict_to_tensordict(obs)

            # Prepare input for ONNX model (use semantic keys to map to ONNX input names)
            onnx_input = {}
            for onnx_name, semantic_key in zip(onnx_input_names, semantic_in_keys):
                onnx_input[onnx_name] = obs_dict_td[semantic_key].detach().cpu().numpy()

            # Run ONNX model inference
            onnx_output = ort_session.run(onnx_output_names, onnx_input)

            # Convert output back to tensor
            # The first output is typically the action (outputs are in order of semantic_out_keys)
            if "mean_action" in semantic_out_keys:
                action = onnx_output[semantic_out_keys.index("mean_action")]
            else:
                action = onnx_output[semantic_out_keys.index("action")]
            action = torch.from_numpy(action).to(fabric.device)

            # Step environment
            _, _, dones, _, _ = env.step(action)
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_indices) > 0:
                log.info(
                    f"Step {step_count}: Resetting {len(done_indices)} environment(s)"
                )
            env.reset(done_indices)
            step_count += 1

    except KeyboardInterrupt:
        log.info("\nVisualization interrupted by user")

# Cleanup
if args.simulator == "isaaclab" and "simulation_app" in simulator_extra_params:
    simulator_extra_params["simulation_app"].close()

log.info("\nYou can now use these ONNX models for deployment!")
