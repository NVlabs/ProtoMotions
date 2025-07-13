#!/usr/bin/env python3
"""
Simple script to run masked mimic agent in path follower environment.
Uses the existing evaluation infrastructure.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_masked_mimic_path_follower(checkpoint_path, robot="smpl", simulator="isaaclab"):
    """
    Run masked mimic agent in path follower environment.
    
    Args:
        checkpoint_path: Path to the masked mimic checkpoint
        robot: Robot type (default: smpl)
        simulator: Simulator type (default: isaaclab)
    """
    
    # Change to the ProtoMotions directory
    os.chdir("/home/rover2/OrcaRL/ProtoMotions")
    
    # Build the command
    cmd = [
        "PYTHON_PATH", 
        "protomotions/eval_agent.py",
        "+exp=masked_mimic_path_follower_eval",
        f"+robot={robot}",
        f"+simulator={simulator}",
        f"checkpoint={checkpoint_path}"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("Evaluation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error code: {e.returncode}")
        return False
    
    return True

def main():
    """Main function to run the evaluation."""
    
    # Configuration
    checkpoint_path = "data/pretrained_models/masked_mimic/smpl"
    robot = "smpl"
    simulator = "isaaclab"
    
    print("Running Masked Mimic Agent in Path Follower Environment")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Robot: {robot}")
    print(f"Simulator: {simulator}")
    print()
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        print("Please make sure the masked mimic model is trained and available.")
        return
    
    # Run the evaluation
    success = run_masked_mimic_path_follower(checkpoint_path, robot, simulator)
    
    if success:
        print("\nEvaluation completed successfully!")
    else:
        print("\nEvaluation failed!")

if __name__ == "__main__":
    main() 