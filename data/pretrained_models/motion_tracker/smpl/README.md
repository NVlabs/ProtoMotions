# Motion Tracker -- SMPL Humanoid

<div float="center">
    <img src="assets/breakdance.gif" width="300"/>
    <img src="assets/monkey_walk_backflip.gif" width="300"/>
</div>

# What is this?

- Pre-trained motion tracker for the SMPL (no fingers) humanoid.
- The goal of this model is to reproduce kinematic recordings within simulation.
- It observes:
  - Current pose
  - 15 future poses
  - Projected surrounding heightmap
- It predicts:
  - Next action (PD target for each joint)


- Trained in IsaacLab. The model may not perform as well in IsaacGym/Genesis.

# Evaluating the model
To evaluate the model run the following command:

```
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +motion_file=<path to motion file> +checkpoint=data/pretrained_models/motion_tracker/smpl/last.ckpt
```

- You should pick which `motion_file` to load.
- The model was trained and performs best in IsaacLab. Simulator can selected using the `simulator` flag --- performance may vary.
- For faster loading times use a flat terrain (config defaults to random heightmap) `+terrain=flat`.