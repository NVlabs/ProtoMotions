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
"""
Tutorial 7: DeepMimic Agent

This tutorial demonstrates how to build a PPO agent for imitation learning.
Building on Tutorial 6 (Mimic environment), we now show:
1. How to configure a PPO agent for imitation learning
2. How to set up actor and critic networks with proper observation inputs
3. How to configure training parameters (learning rates, batch size, etc.)
4. How to run a simple training loop with the agent
5. Understanding the complete pipeline: Environment + Agent

IsaacLab and IsaacGym must be imported before torch is imported.

As many modules may import torch internally, it is best practice to simply detect the selected simulator
at the top and import it right away.
"""

# Parse arguments first (argparse is safe, doesn't import torch)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulator",
    type=str,
    required=True,
    help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
)
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from protomotions.simulator.base_simulator.config import SimulatorConfig  # noqa: E402
from protomotions.envs.mimic.env import Mimic  # noqa: E402
from protomotions.envs.mimic.config import (  # noqa: E402
    MimicEnvConfig,
    MimicMotionManagerConfig,
    MimicObsConfig,
    MimicEarlyTerminationEntry,
)
from protomotions.envs.obs.config import (  # noqa: E402
    HumanoidObsConfig,
    MaxCoordsSelfObsConfig,
    MimicTargetPoseConfig,
    MimicPhaseObsConfig,
    MimicTimeLeftObsConfig,
)
from protomotions.envs.base_env.config import RewardComponentConfig  # noqa: E402
from protomotions.envs.utils.rewards import (  # noqa: E402
    mean_squared_error_exp,
    rotation_error_exp,
)
from protomotions.components.motion_lib import MotionLibConfig  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.agents.ppo.config import (  # noqa: E402
    PPOAgentConfig,
    PPOActorConfig,
    PPOModelConfig,
)
from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig  # noqa: E402
from protomotions.agents.base_agent.config import OptimizerConfig  # noqa: E402
from protomotions.agents.evaluators.config import MimicEvaluatorConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import factory functions
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402

robot_cfg = robot_config("smpl")

print("\n=== Robot Configuration ===")
print("Robot type: smpl")
print(f"Robot config class: {type(robot_cfg).__name__}")
print(f"Number of actions: {robot_cfg.number_of_actions}")
print(f"Number of DOFs: {robot_cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {robot_cfg.kinematic_info.num_bodies}")
print(f"Contact bodies: {robot_cfg.contact_bodies}")

# Extra simulator parameters
extra_simulator_params = {}
if args.simulator == "isaaclab":
    app_launcher_flags = {"headless": False, "device": str(device)}
    app_launcher = AppLauncher(app_launcher_flags)
    simulation_app = app_launcher.app
    extra_simulator_params["simulation_app"] = simulation_app

# Create simulator configuration
simulator_cfg: SimulatorConfig = simulator_config(
    args.simulator,
    robot_cfg,
    headless=False,
    num_envs=4,
    experiment_name="deepmimic_tutorial",
)

print("\n=== Simulator Configuration ===")
print(f"Simulator type: {args.simulator}")
print(f"Simulator class: {get_class(simulator_cfg._target_).__name__}")
print(f"Number of environments: {simulator_cfg.num_envs}")
print(f"Device: {device}")
print(f"Headless: {simulator_cfg.headless}")

# Motion file for imitation learning
motion_file = "examples/data/smpl_humanoid_sit_armchair.motion"

print("\n=== DeepMimic Configuration ===")
print(f"Motion file: {motion_file}")
print("This tutorial shows how to train a policy to imitate this motion")

# Configure mimic observations - this is key for imitation learning
print("\n=== Mimic Observations Configuration ===")
print("Configuring comprehensive mimic observations:")
print("  → Phase observations: sin/cos of motion progress (cyclical)")
print("  → Time left observations: remaining time in current motion")
print("  → Target pose observations: future reference poses for policy")

from protomotions.components.scene_lib import (  # noqa: E402
    ObjectOptions,
    MeshSceneObject,
    Scene,
    SceneLibConfig,
    SceneLib,
)

# Define object physics properties
chair_options = ObjectOptions(
    density=1000,
    fix_base_link=True,
    angular_damping=0.01,
    linear_damping=0.01,
    max_angular_velocity=100.0,
    vhacd_enabled=True,
    vhacd_params={
        "max_convex_hulls": 10,
        "max_num_vertices_per_ch": 64,
        "resolution": 300000,
    },
)

# Create scene with chair
chair = MeshSceneObject(
    object_path="examples/data/armchair.usda"
    if args.simulator == "isaaclab"
    else "examples/data/armchair.urdf",
    options=chair_options,
    translation=(0.0, 0.9, 0.0),
    rotation=(0.0, 0.0, 0.0, 1.0),
)

scene = Scene(objects=[chair], humanoid_motion_id=0)

# Create environment configuration (similar to mimic_environment tutorial)
env_config = MimicEnvConfig(
    max_episode_length=300,
    # Uses reference motion resets automatically (motion_lib is set)
    sync_motion=False,  # Policy training mode (not kinematic)
    humanoid_obs=HumanoidObsConfig(
        max_coords_obs=MaxCoordsSelfObsConfig(
            enabled=True,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
            num_historical_steps=2,
        ),
    ),
    mimic_obs=MimicObsConfig(
        enabled=True,  # Enable mimic observations
        mimic_phase_obs=MimicPhaseObsConfig(
            enabled=False  # phase observations (sin/cos of motion progress)
        ),
        mimic_time_left_obs=MimicTimeLeftObsConfig(
            enabled=False  # time left observations are similar to phase, but in absolute time left and not percentage of the motion
        ),
        mimic_target_pose=MimicTargetPoseConfig(
            enabled=True,  # Enable target pose observations
            future_steps=1,  # Look 1 step ahead
            with_time=True,  # Include time information
            with_contacts=False,  # This demo motion does not specify contact information
            with_velocities=True,  # Include velocity information
        ),
    ),
    motion_manager=MimicMotionManagerConfig(
        init_start_prob=1.0,  # Always start from beginning for consistent training
        resample_on_reset=False,  # When failing to track, do not resample the motion. Instead, reset the character to the correct position and orientation at the current motion time.
    ),
    # Reward configuration using dictionary format with reward functions
    reward_config={
        # Mimic tracking rewards - matching body positions and rotations
        "gt_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_pos",
                "ref_x": "ref_state.rigid_body_pos",
                "coefficient": -100.0,
            },
            weight=0.5,
        ),
        "gr_rew": RewardComponentConfig(
            function=rotation_error_exp,
            variables={
                "q": "current_state.rigid_body_rot",
                "ref_q": "ref_state.rigid_body_rot",
                "coefficient": -5.0,
            },
            weight=0.3,
        ),
        "gv_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_vel",
                "ref_x": "ref_state.rigid_body_vel",
                "coefficient": -0.5,
            },
            weight=0.1,
        ),
        "gav_rew": RewardComponentConfig(
            function=mean_squared_error_exp,
            variables={
                "x": "current_state.rigid_body_ang_vel",
                "ref_x": "ref_state.rigid_body_ang_vel",
                "coefficient": -0.1,
            },
            weight=0.1,
        ),
    },
    mimic_early_termination=[
        MimicEarlyTerminationEntry(
            mimic_early_termination_key="max_joint_err",  # Early terminate based on the max_joint_err, the maximal error of any joint from the reference motion.
            mimic_early_termination_thresh=0.5,  # The threshold is set to 0.5 meters.
            less_than=False,  # The termination will trigger when the max_joint_err is greater than the threshold (not less than).
        )
    ],
)

print("\n=== Environment Configuration ===")
print("Environment type: Mimic (for imitation learning)")
print(f"Sync motion: {env_config.sync_motion} (policy training mode)")
print(f"Episode length: {env_config.max_episode_length}")
print("Motion sampling: Always start from beginning for consistent training")
print(f"Mimic observations: {env_config.mimic_obs.enabled}")
print(f"Phase observations: {env_config.mimic_obs.mimic_phase_obs.enabled}")
print(f"Time left observations: {env_config.mimic_obs.mimic_time_left_obs.enabled}")
print(f"Target pose observations: {env_config.mimic_obs.mimic_target_pose.enabled}")
print(f"Resample on reset: {env_config.motion_manager.resample_on_reset}")


# Create terrain with simple flat configuration
from protomotions.components.terrains.terrain import Terrain  # noqa: E402

terrain_config = TerrainConfig()  # Simple flat terrain for this demo (default is flat)
terrain = Terrain(config=terrain_config, num_envs=simulator_cfg.num_envs, device=device)

scene_lib_config = SceneLibConfig(scene_file=None)
scene_lib = SceneLib(
    config=scene_lib_config,
    num_envs=simulator_cfg.num_envs,
    scenes=[scene],
    device=device,
    terrain=terrain,
)

motion_lib_config = MotionLibConfig(motion_file=motion_file)
from protomotions.components.motion_lib import MotionLib  # noqa: E402

motion_lib = MotionLib(config=motion_lib_config, device=device)

from protomotions.utils.hydra_replacement import get_class  # noqa: E402

SimulatorClass = get_class(simulator_cfg._target_)
simulator = SimulatorClass(
    config=simulator_cfg,
    robot_config=robot_cfg,
    terrain=terrain,
    scene_lib=scene_lib,
    device=device,
    **extra_simulator_params,
)

# Create the environment
env = Mimic(
    config=env_config,
    robot_config=robot_cfg,
    device=device,
    simulator=simulator,
    motion_lib=motion_lib,
    terrain=terrain,
    scene_lib=scene_lib,
    **extra_simulator_params,
)

print("\n=== Environment Initialization ===")
print("Mimic environment created successfully")
print(f"Motion library loaded: {env.motion_lib is not None}")
print(f"Motion manager type: {type(env.motion_manager).__name__}")
print(f"Mimic observations callback: {hasattr(env, 'mimic_obs_cb')}")

# Reset environment and get initial observations
print("\n=== Environment Reset ===")
env.reset()
print("Environment reset completed")

obs = env.get_obs()
print(f"Observations: {obs is not None}")
print(f"Motion IDs: {env.motion_manager.motion_ids}")
print(f"Motion times: {env.motion_manager.motion_times}")

# Analyze observation structure
print("\n=== Observation Structure from Reset ===")
print(f"Observation keys from reset: {list(obs.keys())}")
for key, value in obs.items():
    print(
        f"  '{key}': shape {value.shape}, range [{value.min().item():.3f}, {value.max().item():.3f}]"
    )

# Now create the PPO agent configuration
print("\n=== PPO Agent Configuration ===")

# Define observation keys used by both actor and critic
obs_keys = ["max_coords_obs", "mimic_target_poses"]

# Actor configuration - maps observations to actions
# Uses MLPWithConcatConfig: concatenates all observation keys and processes through MLP
actor_config = PPOActorConfig(
    num_out=robot_cfg.kinematic_info.num_dofs,
    actor_logstd=-2.9,  # Initial log standard deviation for action noise
    in_keys=obs_keys,  # Observation keys to process
    mu_key="actor_trunk_out",  # Output key for the mean action
    mu_model=MLPWithConcatConfig(
        in_keys=obs_keys,  # Same observation keys
        out_keys=["actor_trunk_out"],  # Output key
        normalize_obs=True,  # Normalize observations
        norm_clamp_value=5,  # Clamp normalized values
        num_out=robot_cfg.number_of_actions,  # Output size (robot DOFs)
        layers=[  # Network architecture
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
        output_activation="tanh",  # Bound actions to [-1, 1]
    ),
)

# Critic configuration - maps observations to value estimates
# Uses same MLPWithConcatConfig pattern as actor
critic_config = MLPWithConcatConfig(
    in_keys=obs_keys,  # Same observation keys as actor
    out_keys=["value"],  # Output key for value estimate
    normalize_obs=True,  # Normalize observations
    norm_clamp_value=5,  # Clamp normalized values
    num_out=1,  # Single value output
    layers=[  # Slightly smaller network for critic
        MLPLayerConfig(units=512, activation="relu"),
        MLPLayerConfig(units=256, activation="relu"),
    ],
)

print("Actor configuration:")
print(f"  - Output size: {actor_config.num_out} (robot DOFs)")
print(f"  - Log std: {actor_config.actor_logstd}")
print(f"  - Input keys: {actor_config.in_keys}")
print(f"  - Network layers: {len(actor_config.mu_model.layers)}")

print("Critic configuration:")
print(f"  - Output size: {critic_config.num_out} (value estimate)")
print(f"  - Input keys: {critic_config.in_keys}")
print(f"  - Network layers: {len(critic_config.layers)}")

# Create PPO agent configuration
agent_config = PPOAgentConfig(
    model=PPOModelConfig(
        in_keys=obs_keys,  # Observation keys for the model
        out_keys=["action", "mean_action", "neglogp", "value"],  # Output keys
        actor=actor_config,
        critic=critic_config,
        actor_optimizer=OptimizerConfig(
            _target_="torch.optim.Adam",
            lr=2e-5,  # Learning rate for actor
        ),
        critic_optimizer=OptimizerConfig(
            _target_="torch.optim.Adam",
            lr=1e-4,  # Learning rate for critic (higher than actor)
        ),
    ),
    batch_size=128,  # Appropriate for 4 envs * 32 steps = 128 samples per rollout
    training_max_steps=2560,  # 20 epochs * 4 envs * 32 steps = 2560 total steps
    num_steps=32,  # Steps per rollout
    num_mini_epochs=4,  # Mini epochs per update
    gradient_clip_val=50.0,  # Gradient clipping for stability
    clip_critic_loss=True,  # Clip critic loss for stability
    evaluator=MimicEvaluatorConfig(
        eval_metric_keys=["gt_err", "gr_err"]  # Key metrics to track
    ),
)

print("\n=== PPO Agent Configuration ===")
print("Agent type: PPO (Proximal Policy Optimization)")
print(f"Batch size: {agent_config.batch_size} (4 envs × 32 steps = 128 samples)")
print(f"Training max steps: {agent_config.training_max_steps} (≈20 epochs)")
print(f"Steps per rollout: {agent_config.num_steps}")
print(f"Mini epochs per update: {agent_config.num_mini_epochs}")
print(f"Actor learning rate: {agent_config.model.actor_optimizer.lr}")
print(f"Critic learning rate: {agent_config.model.critic_optimizer.lr}")
print(f"Gradient clipping: {agent_config.gradient_clip_val}")
print(f"Evaluation metrics: {agent_config.evaluator.eval_metric_keys}")

print("\nAgent configuration finalized:")
print(f"  - Model input keys: {agent_config.model.in_keys}")
print(f"  - Model output keys: {agent_config.model.out_keys}")
print(f"  - Actor output size: {agent_config.model.actor.num_out}")
print(f"  - Critic output size: {agent_config.model.critic.num_out}")

# Create Fabric for distributed training (even for single GPU)
from lightning.fabric import Fabric  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402

fabric_config = FabricConfig(
    devices=1,
    num_nodes=1,
    loggers=[],
    callbacks=[],
)

fabric: Fabric = Fabric(**fabric_config.to_dict())
fabric.launch()

print("\n=== Fabric Configuration ===")
print(f"Fabric accelerator: {fabric_config.accelerator}")
print(f"Fabric device: {fabric.device}")
print(f"Fabric precision: {fabric_config.precision}")

# Create the agent with Fabric
from protomotions.agents.ppo.agent import PPO  # noqa: E402

agent = PPO(
    fabric=fabric,
    env=env,
    config=agent_config,
    root_dir=Path("./tutorial_7_output"),  # Directory for saving checkpoints
)
agent.setup()

print("\n=== Agent Initialization ===")
print("PPO agent created successfully")
print(f"Agent device: {agent.device}")
print(f"Max epochs calculated: {agent.max_epochs}")
print(f"Agent has actor: {hasattr(agent, 'actor')}")
print(f"Agent has critic: {hasattr(agent, 'critic')}")
print(f"Training will run for {agent.max_epochs} epochs")

# Reset environment and get initial observations
print("\n=== Environment and Agent Reset ===")
env.reset()
obs = env.get_obs()

print("Environment reset completed")
print(f"Observation keys from reset: {list(obs.keys())}")

# Show how agent processes observations
print("\n=== Agent Action Generation ===")
with torch.no_grad():  # No gradients needed for inference
    agent_outs = agent.model(obs)

print("Agent processed observations:")
print(f"  - Input observation keys: {list(obs.keys())}")
print(f"  - Generated actions shape: {agent_outs['action'].shape}")
print(f"  - Action log probabilities shape: {agent_outs['neglogp'].shape}")
print(f"  - Value estimates shape: {agent_outs['value'].shape}")
print(f"  - Actions range: [{agent_outs['action'].min().item():.3f}, {agent_outs['action'].max().item():.3f}]")
print(f"  - Values range: [{agent_outs['value'].min().item():.3f}, {agent_outs['value'].max().item():.3f}]")

# Run actual training using the agent's fit function
print("\n=== Starting Agent Training ===")
print("Running actual PPO training for 20 epochs:")
print("  - Agent will learn to imitate the reference motion")
print("  - Training progress will be displayed")
print("  - You can watch the robot improve over time")
print("Camera controls during training:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    # Use the agent's fit function for training
    # This runs the complete PPO training loop
    # Training parameters are set in the agent_config above
    agent.fit()

    print("\nTraining completed successfully!")
    print("Agent has learned to imitate the reference motion")

except KeyboardInterrupt:
    print("\nTraining stopped by user")
finally:
    env.close()

print("\n=== Tutorial Summary ===")
print("This tutorial demonstrated:")
print("1. How to configure a PPO agent for imitation learning:")
print("   - Actor network: Maps observations to actions using MLPWithConcatConfig")
print("   - Critic network: Maps observations to value estimates")
print("   - Simple concatenation of observation keys with normalization")
print("2. How to set up training parameters:")
print("   - Learning rates (actor vs critic)")
print("   - Batch size and training steps")
print("   - Gradient clipping for stability")
print("3. How agent-environment interaction works:")
print("   - Agent.get_action_and_value(obs) → actions, log_probs, values")
print("   - env.step(actions) → new_obs, rewards, dones, infos")
print("4. How to collect rollout data for training")
print("5. How mimic rewards encourage motion tracking")
print("\nKey DeepMimic Concepts:")
print("- Policy Learning: Neural network learns to map observations to actions")
print("- Value Estimation: Critic estimates future rewards for policy optimization")
print("- Observation Keys: in_keys specify which observations to use")
print("- Imitation Learning: Policy learns to reproduce reference motion")
print("\nBuilds on Tutorial 6: Adds agent training to mimic environment")
print("This completes the full pipeline: Simulator → Environment → Agent!")
