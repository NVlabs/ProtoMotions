#!/usr/bin/env python3
"""
Test script to verify path follower configuration with expert model loading.
"""

import os
import sys
from pathlib import Path

# Add the protomotions directory to the path
sys.path.insert(0, str(Path(__file__).parent / "protomotions"))

import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

def test_config_loading():
    """Test that the path follower configuration loads correctly with expert model path."""
    
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
    print(f"Task reward weight: {config.agent.config.task_reward_w}")
    print(f"Discriminator reward weight: {config.agent.config.discriminator_reward_w}")
    
    # Check if expert model path exists
    expert_path = Path(config.agent.config.expert_model_path)
    if expert_path.exists():
        print(f"✓ Expert model path exists: {expert_path}")
        
        # Check for required files
        config_file = expert_path / "config.yaml"
        checkpoint_file = expert_path / "last.ckpt"
        
        if config_file.exists():
            print(f"✓ Expert config file exists: {config_file}")
        else:
            print(f"✗ Expert config file missing: {config_file}")
            
        if checkpoint_file.exists():
            print(f"✓ Expert checkpoint file exists: {checkpoint_file}")
        else:
            print(f"✗ Expert checkpoint file missing: {checkpoint_file}")
    else:
        print(f"✗ Expert model path does not exist: {expert_path}")
    
    return config

def test_agent_import():
    """Test that the AMPWithExpert agent can be imported."""
    try:
        from protomotions.agents.amp.agent_with_expert import AMPWithExpert
        print("✓ AMPWithExpert agent imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import AMPWithExpert agent: {e}")
        return False

if __name__ == "__main__":
    print("Testing path follower setup with expert model loading...")
    print("=" * 60)
    
    # Test agent import
    agent_import_success = test_agent_import()
    
    if agent_import_success:
        # Test configuration loading
        try:
            config = test_config_loading()
            print("\n✓ All tests passed! Configuration is ready for training.")
        except Exception as e:
            print(f"\n✗ Configuration test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✗ Agent import failed, cannot proceed with configuration test.") 