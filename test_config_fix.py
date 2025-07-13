#!/usr/bin/env python3
"""
Test script to verify the path follower configuration loads correctly with discriminator.
"""

import os
import sys
from pathlib import Path

# Add the protomotions directory to the path
sys.path.insert(0, str(Path(__file__).parent / "protomotions"))

import hydra
from omegaconf import OmegaConf

def test_config_loading():
    """Test that the path follower configuration loads correctly with discriminator."""
    
    # Set up Hydra
    os.chdir("/home/rover2/OrcaRL/ProtoMotions")
    
    # Test the configuration
    config = hydra.compose(
        config_name="base",
        overrides=[
            "+exp=path_follower_amp_mlp",
            "+robot=smpl", 
            "+simulator=isaaclab",
            "agent.config.expert_model_path=data/pretrained_models/masked_mimic/smpl"
        ]
    )
    
    print("Configuration loaded successfully!")
    print(f"Agent target: {config.agent._target_}")
    print(f"Expert model path: {config.agent.config.expert_model_path}")
    
    # Check if discriminator configuration exists
    if hasattr(config.agent.config.model.config, 'discriminator'):
        print("✓ Discriminator configuration found")
        print(f"  Discriminator target: {config.agent.config.model.config.discriminator._target_}")
        print(f"  Discriminator input size: {config.agent.config.model.config.discriminator.num_in}")
        print(f"  Discriminator output size: {config.agent.config.model.config.discriminator.num_out}")
    else:
        print("✗ Discriminator configuration missing")
        return False
    
    # Check if discriminator optimizer exists
    if hasattr(config.agent.config.model.config, 'discriminator_optimizer'):
        print("✓ Discriminator optimizer configuration found")
    else:
        print("✗ Discriminator optimizer configuration missing")
        return False
    
    # Check if historical_self_obs is in extra_inputs
    if hasattr(config.agent.config, 'extra_inputs') and 'historical_self_obs' in config.agent.config.extra_inputs:
        print("✓ historical_self_obs in extra_inputs")
    else:
        print("✗ historical_self_obs missing from extra_inputs")
        return False
    
    # Check if humanoid_obs configuration exists
    if hasattr(config.env.config, 'humanoid_obs') and hasattr(config.env.config.humanoid_obs, 'num_historical_steps'):
        print(f"✓ humanoid_obs.num_historical_steps: {config.env.config.humanoid_obs.num_historical_steps}")
    else:
        print("✗ humanoid_obs configuration missing")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing path follower configuration with discriminator...")
    print("=" * 60)
    
    try:
        success = test_config_loading()
        if success:
            print("\n✓ All tests passed! Configuration is ready for training.")
        else:
            print("\n✗ Configuration test failed.")
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc() 