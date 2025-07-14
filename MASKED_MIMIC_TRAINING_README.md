# Training Masked Mimic Agents in New Environments

This guide explains how to train masked mimic agents to perform new tasks (steering and path following) while inheriting motion quality from pretrained models.

## Overview

The masked mimic training process uses a pretrained expert model to provide demonstration data. The agent learns to:
1. **Follow task-specific commands** (steering directions, path following)
2. **Maintain natural motion quality** from the pretrained expert
3. **Adapt to new environments** while preserving learned motion patterns

## Prerequisites

Before training, ensure you have:
1. **Pretrained masked mimic model** at `data/pretrained_models/masked_mimic/smpl/`
2. **Motion file** at `data/motions/smpl_humanoid_walk.npy`
3. **IsaacLab simulator** properly configured

## Training Configurations

### 1. Steering Environment Training

**Configuration**: `protomotions/config/exp/masked_mimic/steering_training.yaml`

This configuration trains the agent to:
- Follow directional commands (cyan arrows)
- Maintain target speeds
- Handle random direction changes
- Preserve natural motion quality

**Key Features**:
- Uses steering environment with complex terrain
- Loads pretrained expert model for demonstrations
- Trains with VAE noise for exploration
- Evaluates steering performance metrics

### 2. Path Following Environment Training

**Configuration**: `protomotions/config/exp/masked_mimic/path_follower_training.yaml`

This configuration trains the agent to:
- Follow generated paths accurately
- Navigate through complex terrain
- Maintain motion quality while path following
- Handle path termination conditions

**Key Features**:
- Uses path follower environment with complex terrain
- Loads pretrained expert model for demonstrations
- Configurable path complexity and parameters
- Evaluates path following accuracy

## Training Scripts

### Quick Start Commands

#### Train Steering Agent
```bash
cd /home/rover2/OrcaRL/ProtoMotions

# Using the training script (recommended)
python train_masked_mimic_steering.py

# Or using direct command
PYTHONPATH=/home/rover2/OrcaRL/ProtoMotions python protomotions/train_agent.py \
  +exp=masked_mimic/steering_training \
  +robot=smpl \
  +simulator=isaaclab \
  experiment_name=masked_mimic_steering_training \
  training_max_steps=10000000 \
  num_envs=4 \
  headless=true \
  motion_file=data/motions/smpl_humanoid_walk.npy \
  agent.config.expert_model_path=data/pretrained_models/masked_mimic/smpl
```

#### Train Path Following Agent
```bash
cd /home/rover2/OrcaRL/ProtoMotions

# Using the training script (recommended)
python train_masked_mimic_path_follower.py

# Or using direct command
PYTHONPATH=/home/rover2/OrcaRL/ProtoMotions python protomotions/train_agent.py \
  +exp=masked_mimic/path_follower_training \
  +robot=smpl \
  +simulator=isaaclab \
  experiment_name=masked_mimic_path_follower_training \
  training_max_steps=10000000 \
  num_envs=4 \
  headless=true \
  motion_file=data/motions/smpl_humanoid_walk.npy \
  agent.config.expert_model_path=data/pretrained_models/masked_mimic/smpl
```

### Script Options

Both training scripts support the following options:

```bash
# Custom experiment name
python train_masked_mimic_steering.py --experiment_name my_steering_experiment

# Custom training parameters
python train_masked_mimic_steering.py \
  --training_max_steps 5000000 \
  --num_envs 8 \
  --headless

# Custom paths
python train_masked_mimic_steering.py \
  --motion_file data/motions/custom_motion.npy \
  --expert_model_path data/pretrained_models/custom_expert
```

## Training Process

### Phase 1: Expert Model Loading
The training process starts by loading the pretrained masked mimic model as an expert:
- Loads model weights from the specified path
- Validates configuration compatibility
- Sets up expert action collection

### Phase 2: Behavioral Cloning
During training, the agent:
- Collects expert actions for each state
- Learns to mimic expert behavior
- Maintains motion quality through VAE encoding

### Phase 3: Task Adaptation
The agent gradually adapts to:
- **Steering**: Following directional commands and speed targets
- **Path Following**: Navigating along generated paths
- **Environment**: Complex terrain and obstacles

## Training Parameters

### Key Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `expert_model_path` | Path to pretrained expert model | `data/pretrained_models/masked_mimic/smpl` |
| `training_max_steps` | Maximum training steps | `10000000` |
| `num_envs` | Number of parallel environments | `4` |
| `batch_size` | Training batch size | `1024` |
| `num_mini_epochs` | PPO mini-epochs per update | `6` |
| `eval_metrics_every` | Evaluation frequency | `500` |

### VAE Parameters

| Parameter | Description | Training | Evaluation |
|-----------|-------------|----------|------------|
| `noise_type` | VAE noise type | `"normal"` | `"zeros"` |
| `latent_dim` | VAE latent dimension | `64` | `64` |
| `kld_schedule` | KL divergence schedule | Enabled | Disabled |

## Monitoring Training

### Logging
Training progress is logged to:
- **TensorBoard**: `results/{experiment_name}/lightning_logs/`
- **WandB**: If configured (optional)
- **Console**: Real-time training metrics

### Key Metrics to Monitor

#### Steering Training
- `model/bc_loss`: Behavioral cloning loss
- `model/vae_kld_loss`: VAE KL divergence loss
- `model/kld_coeff`: KL divergence coefficient
- `rewards/steering_reward`: Steering task reward
- `rewards/motion_quality`: Motion quality reward

#### Path Following Training
- `model/bc_loss`: Behavioral cloning loss
- `model/vae_kld_loss`: VAE KL divergence loss
- `rewards/path_following_reward`: Path following accuracy
- `rewards/motion_quality`: Motion quality reward

### Evaluation
The agent is evaluated every 500 epochs to:
- Measure task performance
- Assess motion quality
- Save best models automatically

## Expected Training Behavior

### Early Training (0-1000 epochs)
- High behavioral cloning loss
- Poor task performance
- Good motion quality (inherited from expert)

### Mid Training (1000-5000 epochs)
- Decreasing behavioral cloning loss
- Improving task performance
- Maintaining motion quality

### Late Training (5000+ epochs)
- Low behavioral cloning loss
- Good task performance
- Preserved motion quality

## Troubleshooting

### Common Issues

1. **Expert Model Not Found**
   ```
   Error: Expert model path not found: data/pretrained_models/masked_mimic/smpl
   ```
   **Solution**: Ensure the pretrained model exists at the specified path

2. **Motion File Not Found**
   ```
   Error: Motion file not found: data/motions/smpl_humanoid_walk.npy
   ```
   **Solution**: Check that the motion file exists and is accessible

3. **Configuration Mismatch**
   ```
   AssertionError: Configuration mismatch between expert and current model
   ```
   **Solution**: Ensure the expert model was trained with compatible settings

4. **Training Instability**
   - Reduce learning rate: `agent.config.model.config.optimizer.lr: 1e-5`
   - Increase batch size: `agent.config.batch_size: 2048`
   - Adjust VAE parameters: `agent.config.vae.kld_schedule`

### Performance Tips

1. **Use Multiple GPUs**: Increase `ngpu` parameter for faster training
2. **Increase Environments**: Use more `num_envs` for better data collection
3. **Monitor Memory**: Reduce batch size if out of memory
4. **Checkpoint Regularly**: Models are saved every 10 epochs

## Results and Evaluation

### Model Checkpoints
Trained models are saved to:
- `results/{experiment_name}/last.ckpt` (latest checkpoint)
- `results/{experiment_name}/score_based.ckpt` (best performing)

### Evaluation Scripts
Use the provided evaluation scripts to test trained models:
- `run_masked_mimic_steering.py` (for steering models)
- `run_masked_mimic_path_follower.py` (for path following models)

### Expected Performance
After successful training, you should see:
- **Steering**: Agent follows directional commands accurately
- **Path Following**: Agent navigates paths with minimal deviation
- **Motion Quality**: Natural, human-like movement patterns
- **Task Completion**: High success rates on target tasks

## Next Steps

After training, you can:
1. **Evaluate** the trained model using evaluation scripts
2. **Fine-tune** parameters for better performance
3. **Transfer** to other environments or tasks
4. **Compare** with other training methods

For more information, see the individual README files:
- `MASKED_MIMIC_STEERING_README.md`
- `MASKED_MIMIC_PATH_FOLLOWER_README.md` 