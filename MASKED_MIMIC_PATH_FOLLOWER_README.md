# Running Masked Mimic Agent in Path Follower Environment

This guide explains how to run a pretrained masked mimic agent in the path follower environment to see how it performs on path following tasks.

## Overview

The masked mimic agent was trained on motion tracking tasks, but you can evaluate how it performs when placed in a path following environment. This will show you:

1. How well the pretrained agent can follow paths
2. The quality of motion it produces
3. Whether it can adapt to path following tasks without retraining

## Methods

### Method 1: Using the Simple Script (Recommended)

Use the provided script that leverages the existing evaluation infrastructure:

```bash
cd /home/rover2/OrcaRL/ProtoMotions
python run_masked_mimic_path_follower_simple.py
```

This script will:
- Load the pretrained masked mimic model from `data/pretrained_models/masked_mimic/smpl`
- Run it in the path follower environment
- Show visualization of the agent following paths
- Display performance metrics

### Method 2: Direct Command

You can also run the evaluation directly using the eval_agent.py:

```bash
cd /home/rover2/OrcaRL/ProtoMotions
PYTHON_PATH protomotions/eval_agent.py \
  +exp=masked_mimic_path_follower_eval \
  +robot=smpl \
  +simulator=isaaclab \
  checkpoint=data/pretrained_models/masked_mimic/smpl
```

### Method 3: Custom Script

For more control, use the custom evaluation script:

```bash
cd /home/rover2/OrcaRL/ProtoMotions
python run_masked_mimic_path_follower.py
```

This script provides more detailed control over the evaluation process and can be modified for specific needs.

## Configuration

The evaluation uses the `masked_mimic_path_follower_eval.yaml` configuration which:

- **Agent**: Uses the masked mimic agent with VAE noise set to "zeros" for deterministic behavior
- **Environment**: Path follower environment with visualization enabled
- **Evaluation**: Single environment, non-headless mode for visualization
- **Termination**: Disabled path termination to allow full episode evaluation

## Expected Behavior

When you run the evaluation, you should see:

1. **Visualization**: The Isaac Lab simulator window showing the humanoid character
2. **Path Markers**: Red spheres indicating the path the agent should follow
3. **Character Movement**: The masked mimic agent attempting to follow the path
4. **Performance Metrics**: Console output showing episode rewards and statistics

## What to Look For

### Motion Quality
- **Natural Movement**: The agent should produce realistic humanoid motion
- **Smooth Transitions**: Actions should be smooth and not jerky
- **Balance**: The character should maintain balance while moving

### Path Following
- **Path Adherence**: How closely the agent follows the target path
- **Speed**: Whether the agent moves at appropriate speeds
- **Direction Changes**: How well it handles turns and direction changes

### Limitations
Since the masked mimic agent was trained for motion tracking (not path following), you might observe:
- **Poor Path Following**: The agent may not follow paths accurately
- **Random Movement**: It might move in random directions
- **No Path Awareness**: It may ignore the path entirely

This is expected and demonstrates why training with expert demonstrations (like in our path_follower_amp_mlp configuration) is beneficial.

## Customization

### Changing Checkpoint
To use a different masked mimic checkpoint:

```bash
python run_masked_mimic_path_follower_simple.py
# Edit the checkpoint_path variable in the script
```

### Modifying Environment
To change path following parameters, edit `protomotions/config/exp/masked_mimic_path_follower_eval.yaml`:

```yaml
env:
  config:
    path_follower_params:
      num_traj_samples: 10  # Number of path points to follow
      fail_dist: 4.0        # Distance threshold for failure
      path_generator:
        speed_max: 5.0      # Maximum path speed
        accel_max: 2.0      # Maximum path acceleration
```

### Running Multiple Episodes
The evaluation will run multiple episodes automatically. You can modify the number of episodes in the configuration or scripts.

## Troubleshooting

### Common Issues

1. **Checkpoint Not Found**
   ```
   Error: Checkpoint path does not exist
   ```
   - Make sure the masked mimic model is trained
   - Check the path in the script

2. **Import Errors**
   ```
   ImportError: cannot import name 'xxx'
   ```
   - Make sure you're in the correct environment (env_isaaclab)
   - Check that all dependencies are installed

3. **Simulator Issues**
   ```
   Error with Isaac Lab simulator
   ```
   - Make sure Isaac Lab is properly installed
   - Check GPU availability and drivers

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify the checkpoint path exists
3. Ensure the environment is properly set up
4. Try running with `HYDRA_FULL_ERROR=1` for detailed error information

## Next Steps

After evaluating the masked mimic agent, you can:

1. **Train Path Following Agent**: Use the `path_follower_amp_mlp` configuration we created
2. **Compare Performance**: Compare the masked mimic vs trained path follower
3. **Analyze Results**: Look at the motion quality and path following accuracy
4. **Modify Training**: Adjust the training configuration based on observations 