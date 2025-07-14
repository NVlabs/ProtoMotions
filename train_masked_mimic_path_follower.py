#!/usr/bin/env python3
"""
Training script for Masked Mimic Agent in Path Follower Environment

This script trains a masked mimic agent to perform path following tasks using a pretrained
expert model as demonstration data. The agent learns to follow paths while maintaining
natural motion quality.

Usage:
    python train_masked_mimic_path_follower.py [options]

Options:
    --experiment_name: Name for the experiment (default: masked_mimic_path_follower_training)
    --training_max_steps: Maximum training steps (default: 10000000)
    --num_envs: Number of parallel environments (default: 4)
    --headless: Run in headless mode (default: True)
    --motion_file: Path to motion file (default: data/motions/smpl_humanoid_walk.npy)
    --expert_model_path: Path to pretrained expert model (default: data/pretrained_models/masked_mimic/smpl)
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description="Train Masked Mimic Agent in Path Follower Environment")
    parser.add_argument("--experiment_name", type=str, default="masked_mimic_path_follower_training",
                       help="Name for the experiment")
    parser.add_argument("--training_max_steps", type=int, default=10000000,
                       help="Maximum training steps")
    parser.add_argument("--num_envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", default=True,
                       help="Run in headless mode")
    parser.add_argument("--motion_file", type=str, default="data/motions/smpl_humanoid_walk.npy",
                       help="Path to motion file")
    parser.add_argument("--expert_model_path", type=str, default="data/pretrained_models/masked_mimic/smpl",
                       help="Path to pretrained expert model")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.motion_file).exists():
        print(f"Error: Motion file not found: {args.motion_file}")
        sys.exit(1)
    
    if not Path(args.expert_model_path).exists():
        print(f"Error: Expert model path not found: {args.expert_model_path}")
        sys.exit(1)
    
    # Construct training command
    cmd = [
        "PYTHONPATH=/home/rover2/OrcaRL/ProtoMotions",
        "python", "protomotions/train_agent.py",
        "+exp=masked_mimic/path_follower_training",
        "+robot=smpl",
        "+simulator=isaaclab",
        f"experiment_name={args.experiment_name}",
        f"training_max_steps={args.training_max_steps}",
        f"num_envs={args.num_envs}",
        f"headless={str(args.headless).lower()}",
        f"motion_file={args.motion_file}",
        f"agent.config.expert_model_path={args.expert_model_path}"
    ]
    
    # Print the command
    print("Training command:")
    print(" ".join(cmd))
    print("\nStarting training...")
    
    # Execute the command
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main() 