# Path Follower with Pretrained Masked Mimic Models

This document describes the modifications made to ProtoMotions to enable training path following agents using pretrained masked mimic models as expert demonstrations.

## Overview

The goal was to modify the original ProtoMotions code to train an agent to follow paths while inheriting motion quality from pretrained masked mimic models. This combines the path following capabilities of AMP (Adversarial Motion Priors) with the expert demonstration learning from masked mimic models.

## Modifications Made

### 1. New Agent: AMPWithExpert

**File**: `protomotions/agents/amp/agent_with_expert.py`

Created a new agent class that inherits from AMP but adds expert model loading capabilities:

- **Expert Model Loading**: Can load pretrained masked mimic models as expert demonstrations
- **Expert Action Collection**: Collects expert actions during training for behavioral cloning
- **AMP Functionality**: Maintains all AMP discriminator functionality for motion quality

Key features:
- Loads expert models from checkpoint paths
- Collects expert actions during training loop
- Registers expert actions in experience buffer
- Maintains compatibility with AMP discriminator training

### 2. New Configuration: agent_with_expert.yaml

**File**: `protomotions/config/agent/amp/agent_with_expert.yaml`

Created a configuration for the new AMPWithExpert agent that:
- Inherits from PPO base configuration
- Uses AMP model structure
- Adds expert_model_path parameter
- Maintains AMP discriminator parameters

### 3. Modified Path Follower Configuration

**File**: `protomotions/config/exp/path_follower_amp_mlp.yaml`

Updated the path follower configuration to:
- Use the new AMPWithExpert agent instead of inheriting from masked_mimic
- Add expert_model_path parameter for pretrained model loading
- Maintain path following functionality with path observations
- Keep AMP discriminator functionality

## Usage

### Training Command

```bash
PYTHON_PATH protomotions/train_agent.py \
  +exp=path_follower_amp_mlp \
  +robot=smpl \
  +simulator=isaaclab \
  +experiment_name=smpl_amp_path \
  agent.config.expert_model_path=data/pretrained_models/masked_mimic/smpl
```

### Parameters

- `+exp=path_follower_amp_mlp`: Uses the modified path follower configuration
- `+robot=smpl`: Specifies SMPL humanoid robot
- `+simulator=isaaclab`: Uses Isaac Lab simulator
- `+experiment_name=smpl_amp_path`: Sets experiment name
- `agent.config.expert_model_path`: Path to pretrained masked mimic model

### Expert Model Requirements

The expert model path should contain:
- `config.yaml`: Configuration file for the expert model
- `last.ckpt` or `score_based.ckpt`: Checkpoint file with trained model weights

## How It Works

1. **Expert Model Loading**: The AMPWithExpert agent loads a pretrained masked mimic model during setup
2. **Training Loop**: During training, the agent collects both its own actions and expert actions
3. **Behavioral Cloning**: The agent learns to mimic the expert's actions while following paths
4. **AMP Discriminator**: The discriminator ensures motion quality similar to the expert
5. **Path Following**: The agent learns to follow paths while maintaining natural motion

## Benefits

- **Motion Quality**: Inherits high-quality motion from pretrained masked mimic models
- **Path Following**: Learns to follow paths while maintaining natural movement
- **Flexibility**: Can use any pretrained masked mimic model as expert
- **Compatibility**: Maintains compatibility with existing AMP infrastructure

## Testing

A test script `test_path_follower_setup.py` is provided to verify:
- Agent import functionality
- Configuration loading
- Expert model path validation
- Required file existence checks

Run the test script before training to ensure everything is set up correctly.

## Files Modified/Created

### New Files
- `protomotions/agents/amp/agent_with_expert.py`
- `protomotions/config/agent/amp/agent_with_expert.yaml`
- `test_path_follower_setup.py`
- `PATH_FOLLOWER_MODIFICATIONS.md`

### Modified Files
- `protomotions/config/exp/path_follower_amp_mlp.yaml`

## Future Enhancements

Potential improvements could include:
- Support for multiple expert models
- Dynamic expert model switching
- Expert action filtering/weighting
- Integration with other motion priors 