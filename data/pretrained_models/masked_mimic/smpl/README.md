# Masked Mimic -- SMPL Humanoid

<div float="center">
    <img src="assets/inbetweening.gif" width="300"/>
</div>

# What is this?

- Pre-trained Masked Mimic agent for the SMPL (no fingers) humanoid.
- The goal of this model is to generate novel motions from partial constraints.
- It observes:
  - Current pose
  - 15 future poses
  - Projected surrounding heightmap
- It can be constrained using:
  - Any-joint-any-time. Any number of future states (defined via time). For each state, any subset of joints. Each joint constraint supports translation and/or rotation constraints.
- It predicts:
  - Next action (PD target for each joint)


- Trained in IsaacLab. The model may not perform as well in IsaacGym/Genesis.

# Evaluating the model
To evaluate the model run the following command:

```
PYTHON_PATH protomotions/eval_agent.py +robot=smpl +simulator=isaaclab +motion_file=<path to motion file> +checkpoint=data/pretrained_models/masked_mimic/smpl/last.ckpt
```

- You should pick which `motion_file` to load.
- The model was trained and performs best in IsaacLab. Simulator can selected using the `simulator` flag --- performance may vary.
- For faster loading times use a flat terrain (config defaults to random heightmap) `+terrain=flat`.

For easy evaluation of a target-pose inbetween objective add the following flags
```
+opt=masked_mimic/constraints/no_constraint env.config.masked_mimic.masked_mimic_masking.target_pose_visible_prob=1
```
The `no_constraint` yaml file turns off all constraints. Then we only enable the target_pose visibility.