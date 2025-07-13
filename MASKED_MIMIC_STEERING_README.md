# Running Masked Mimic Agent in Steering Environment

This guide explains how to run a pretrained masked mimic agent in the steering environment to see how it performs on steering tasks.

## Overview

The steering environment requires the agent to follow directional commands and speed targets. The agent receives:
- **Direction commands**: Target direction to move in
- **Speed commands**: Target speed to maintain
- **Visual markers**: Cyan arrows showing the target direction

## Quick Start

### Method 1: Using the Script (Recommended)

```bash
cd /home/rover2/OrcaRL/ProtoMotions
python run_masked_mimic_steering.py
```

### Method 2: Direct Command

```bash
cd /home/rover2/OrcaRL/ProtoMotions
PYTHON_PATH protomotions/eval_agent.py \
  +exp=masked_mimic_steering_eval \
  +robot=smpl \
  +simulator=isaaclab \
  checkpoint=data/pretrained_models/masked_mimic/smpl/last.ckpt
```

## What You'll See

### Visual Elements
1. **Humanoid Character**: The SMPL humanoid agent
2. **Cyan Arrows**: Target direction markers showing where the agent should move
3. **Movement**: The agent attempting to follow the steering commands

### Steering Behavior
The environment will:
- **Change directions**: Randomly change target direction every 40-150 steps
- **Adjust speeds**: Vary target speed between 1.2-6.0 units
- **Random stops**: Occasionally set speed to 0 (5% probability)
- **Smooth transitions**: Gradually change heading and speed

## Expected Behavior

Since the masked mimic agent was trained for motion tracking (not steering), you might observe:

### Motion Quality
- **Natural Movement**: Realistic humanoid motion
- **Smooth Transitions**: Fluid action sequences
- **Balance**: Good balance maintenance

### Steering Performance
- **Poor Direction Following**: May not follow target directions accurately
- **Random Movement**: Might move in random directions
- **Speed Issues**: May not maintain target speeds
- **No Steering Awareness**: May ignore steering commands entirely

## Configuration Details

The `masked_mimic_steering_eval.yaml` configuration includes:

### Steering Parameters
```yaml
steering_params:
  heading_change_steps_min: 40    # Min steps between direction changes
  heading_change_steps_max: 150   # Max steps between direction changes
  random_heading_probability: 0.2 # Probability of random direction change
  standard_heading_change: 1.57   # Standard heading change (radians)
  tar_speed_min: 1.2              # Minimum target speed
  tar_speed_max: 6.0              # Maximum target speed
  standard_speed_change: 0.3      # Standard speed change
  stop_probability: 0.05          # Probability of stopping
  obs_size: 3                     # Steering observation size
```

### Agent Settings
- **VAE Noise**: Set to "zeros" for deterministic behavior
- **Expert Model**: Disabled (not needed for evaluation)
- **Motion File**: Uses `smpl_humanoid_walk.npy`

## Customization

### Changing Checkpoint
Edit the `checkpoint_path` in `run_masked_mimic_steering.py`:
```python
checkpoint_path = "path/to/your/checkpoint.ckpt"
```

### Modifying Steering Parameters
Edit `protomotions/config/exp/masked_mimic_steering_eval.yaml`:
```yaml
env:
  config:
    steering_params:
      tar_speed_min: 0.5    # Slower minimum speed
      tar_speed_max: 8.0    # Faster maximum speed
      stop_probability: 0.1  # More frequent stops
```

### Running with Different Robots
```bash
python run_masked_mimic_steering.py
# Edit the robot variable in the script
```

## Comparison with Path Following

### Steering vs Path Following
- **Steering**: Continuous direction/speed commands
- **Path Following**: Discrete path points to follow
- **Complexity**: Steering is simpler, path following is more complex

### Expected Performance Differences
- **Steering**: May perform slightly better due to simpler task
- **Path Following**: Likely worse performance due to task complexity
- **Motion Quality**: Should be similar in both environments

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
   ```
   Error: Checkpoint path does not exist
   ```
   - Verify the checkpoint file exists
   - Check the path in the script

2. **Import Errors**
   ```
   ImportError: cannot import name 'xxx'
   ```
   - Ensure you're in the `env_isaaclab` environment
   - Check all dependencies are installed

3. **Simulator Issues**
   ```
   Error with Isaac Lab simulator
   ```
   - Verify Isaac Lab installation
   - Check GPU availability

### Getting Help

If you encounter issues:
1. Check console output for error messages
2. Verify checkpoint path exists
3. Ensure environment is properly set up
4. Run with `HYDRA_FULL_ERROR=1` for detailed errors

## Next Steps

After evaluating the masked mimic agent in steering:

1. **Compare Environments**: Compare steering vs path following performance
2. **Train Steering Agent**: Create a steering agent with expert demonstrations
3. **Analyze Motion Quality**: Assess naturalness of generated motions
4. **Optimize Parameters**: Adjust steering parameters for better performance

## Files Created

- `protomotions/config/exp/masked_mimic_steering_eval.yaml`: Configuration for steering evaluation
- `run_masked_mimic_steering.py`: Script to run steering evaluation
- `MASKED_MIMIC_STEERING_README.md`: This documentation 