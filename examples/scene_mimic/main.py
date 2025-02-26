"""
    IsaacLab and app launcher must be setup before all other imports.
"""

from isaaclab.app import AppLauncher

headless = False
app_launcher = AppLauncher({"headless": headless})
simulation_app = app_launcher.app

import torch
from omegaconf import OmegaConf
from protomotions.utils.config_utils import *
from examples.scene_mimic.env import SceneMimic

# Define configuration paths
base_motion_manager_config_path = "protomotions/config/motion_manager/base_manager.yaml"
mimic_motion_manager_config_path = "protomotions/config/motion_manager/mimic_manager.yaml"
base_env_config_path = "protomotions/config/env/base_env.yaml"
env_config_path = "protomotions/config/env/mimic.yaml"
base_robot_config_path = "protomotions/config/robot/base.yaml"
robot_config_path = "protomotions/config/robot/smpl.yaml"
base_simulator_config_path = "protomotions/config/simulator/base.yaml"
physx_simulator_config_path = "protomotions/config/simulator/physx.yaml"
physx_isaaclab_simulator_config_path = "protomotions/config/simulator/physx_isaaclab.yaml"
isaaclab_simulator_config_path = "protomotions/config/simulator/isaaclab.yaml"
terrain_config_path = "protomotions/config/terrain/flat.yaml"
motion_file = "data/motions/smpl_humanoid_sit_armchair.npy"

# Load and merge configurations
base_motion_manager_config = OmegaConf.load(base_motion_manager_config_path)
mimic_motion_manager_config = OmegaConf.load(mimic_motion_manager_config_path)
base_env_config = OmegaConf.load(base_env_config_path)
env_config = OmegaConf.load(env_config_path)
base_robot_config = OmegaConf.load(base_robot_config_path)
robot_config = OmegaConf.load(robot_config_path)
base_simulator_config = OmegaConf.load(base_simulator_config_path)
physx_simulator_config = OmegaConf.load(physx_simulator_config_path)
physx_isaaclab_simulator_config = OmegaConf.load(physx_isaaclab_simulator_config_path)
isaaclab_simulator_config = OmegaConf.load(isaaclab_simulator_config_path)
terrain_config = OmegaConf.load(terrain_config_path)

# Create a merged configuration
full_config = OmegaConf.merge(
    {"motion_manager": base_motion_manager_config.motion_manager},
    {"motion_manager": mimic_motion_manager_config.motion_manager},
    {"env": base_env_config.env},
    {"env": env_config.env},
    {"robot": base_robot_config.robot},
    {"robot": robot_config.robot},
    {"motion_lib": base_robot_config.motion_lib},
    {"simulator": base_simulator_config.simulator},
    {"simulator": physx_simulator_config.simulator},
    {"simulator": physx_isaaclab_simulator_config.simulator},
    {"simulator": isaaclab_simulator_config.simulator},
    {"terrain": terrain_config.terrain},
    {"motion_file": motion_file},
    {"headless": headless},
    {"ref_respawn_offset": 0},
    {"num_envs": 4},
    {"sync_motion": True},
    {"experiment_name": "scene_mimic"}
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolve the configuration
full_config = OmegaConf.to_container(full_config, resolve=True)
full_config = OmegaConf.create(full_config)

# Create and initialize the mimic environment
env = SceneMimic(
    config=full_config.env.config,
    device=device,
    simulation_app=simulation_app
)

# Reset the environment
env.reset()

# Run the simulation loop
try:
    while True:
        """
            Camera controls in IsaacLab and IsaacGym:
            1. L - start/stop recording. Once stopped it will save the video.
            2. ; - cancel recording and delete the video.
            3. O - toggle camera target. This will cycle through the available camera targets, such as humanoids and objects in the scene.
            4. Q - close the simulator.
        """
        # In mimic environment with motion_sync=True, the actions are ignored
        # as the robot follows the motion library
        actions = torch.zeros((env.num_envs, env.simulator.get_num_act()), device=device)
        obs, rewards, dones, infos = env.step(actions)
            
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    env.close()
