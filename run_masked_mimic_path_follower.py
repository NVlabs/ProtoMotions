#!/usr/bin/env python3
"""
Script to run pretrained masked mimic agent in path_follower environment.
This allows you to see how the pretrained agent performs on path following tasks.
"""

import os
import sys
import torch
import time
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from lightning.fabric import Fabric

# Add the protomotions directory to the path
sys.path.insert(0, str(Path(__file__).parent / "protomotions"))

from protomotions.agents.masked_mimic.agent import MaskedMimic
from protomotions.agents.masked_mimic.model import VaeDeterministicOutputModel
from protomotions.envs.path_follower.env import PathFollowing
from protomotions.agents.common.common import weight_init

def create_path_follower_env(config, device):
    """Create a path follower environment."""
    env_config = config.env
    env_config.config.num_envs = 1  # Single environment for evaluation
    env_config.config.headless = False  # Show visualization
    
    env = instantiate(env_config, device=device)
    return env

def load_masked_mimic_agent(checkpoint_path, device):
    """Load a pretrained masked mimic agent."""
    # Load the expert model configuration
    expert_config_path = Path(checkpoint_path) / "config.yaml"
    expert_config = OmegaConf.load(expert_config_path)
    
    # Create the model
    model: VaeDeterministicOutputModel = instantiate(expert_config.agent.config.model)
    model.apply(weight_init)
    
    # Load the checkpoint
    checkpoint_file = Path(checkpoint_path) / "last.ckpt"
    if not checkpoint_file.exists():
        checkpoint_file = Path(checkpoint_path) / "score_based.ckpt"
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model, expert_config

def run_evaluation(model, env, device, num_episodes=5):
    """Run evaluation of the masked mimic agent in path follower environment."""
    
    print(f"Starting evaluation for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        
        episode_reward = 0.0
        step_count = 0
        max_steps = env.config.max_episode_length
        
        while step_count < max_steps:
            # Add VAE noise for masked_mimic model
            if "vae_noise" not in obs:
                vae_latent_dim = model.config.vae_latent_dim
                obs["vae_noise"] = torch.randn(1, vae_latent_dim, device=device)
            
            # Get action from model
            with torch.no_grad():
                action = model.act(obs)
            
            # Step environment
            obs, rewards, dones, terminated, extras = env.step(action)
            
            episode_reward += rewards.item()
            step_count += 1
            
            # Check if episode is done
            if dones.item() or terminated.item():
                break
            
            # Small delay for visualization
            time.sleep(0.01)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Average Reward per Step: {episode_reward/step_count:.4f}")

def main():
    # Configuration
    checkpoint_path = "data/pretrained_models/masked_mimic/smpl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Loading masked mimic model from: {checkpoint_path}")
    
    # Load the masked mimic model
    model, expert_config = load_masked_mimic_agent(checkpoint_path, device)
    model = model.to(device)
    
    # Create path follower environment configuration
    config = OmegaConf.create({
        "env": {
            "_target_": "protomotions.envs.path_follower.env.PathFollowing",
            "config": {
                "num_envs": 1,
                "headless": False,
                "max_episode_length": 300,
                "robot": {
                    "self_obs_size": 108,  # SMPL robot observation size
                    "number_of_actions": 45,  # SMPL robot action size
                    "head_body_name": "head"
                },
                "path_follower_params": {
                    "num_traj_samples": 10,
                    "fail_dist": 4.0,
                    "fail_height_dist": 0.5,
                    "traj_sample_timestep": 0.5,
                    "path_generator": {
                        "num_verts": 101,
                        "dtheta_max": 2.0,
                        "sharp_turn_prob": 0.02,
                        "accel_max": 2.0,
                        "speed_max": 5.0,
                        "speed_min": 0.0,
                        "fixed_path": False,
                        "slow": False,
                        "height_conditioned": True,
                        "start_speed_max": 3.0,
                        "speed_z_max": 0.5,
                        "accel_z_max": 0.2,
                        "head_height_max": 1.5,
                        "head_height_min": 0.4,
                        "use_naive_path_generator": False
                    },
                    "path_obs_size": 30,  # 3 coords * 10 samples
                    "enable_path_termination": True,
                    "height_conditioned": True,
                    "num_path_obs_per_point": 3
                },
                "humanoid_obs": {
                    "num_historical_steps": 8
                },
                "enable_height_termination": False,
                "termination_height": 0.15,
                "head_termination_height": 0.3,
                "shield_termination_height": 0.32
            }
        }
    })
    
    # Create environment
    env = create_path_follower_env(config, device)
    
    # Run evaluation
    run_evaluation(model, env, device, num_episodes=5)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main() 