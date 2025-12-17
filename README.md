<div align="center">

# ProtoMotions 3

**A GPU-Accelerated Framework for Simulated Humanoids**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://protomotions.github.io/)

</div>

---

## Overview

**ProtoMotions3** is a GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots. Our mission is to provide a **fast prototyping platform** for various simulated humanoid learning tasks and environments—for researchers and practitioners in **animation**, **robotics**, and **reinforcement learning**—bridging efforts across domains.

**Modularity**, **extensibility**, and **scalability** are at the core of ProtoMotions3. It is permissively licensed under the [Apache-2.0 license](LICENSE.md).

Also check out [**MimicKit**](https://github.com/xbpeng/MimicKit/tree/main), our sibling repository for a lightweight framework for motion imitation learning.

---

## What You Can Do with ProtoMotions3

### 🏃 Large-Scale Motion Learning

Train your fully physically simulated character to learn motion skills from the entire public [**AMASS**](https://amass.is.tue.mpg.de/) human animation dataset (**40+ hours**) within **12 hours** on 4 A100s.

<p align="center">
  <img src="data/static/smpl-flat-gym.gif" alt="SMPL training" height="280">
</p>

### 📈 Scalable Multi-GPU Training

Scale training to even larger datasets with each GPU handling a subset of motions. For example, we have trained with **24 A100s** with **13K motions** on each GPU.

<p align="center">
  <img src="data/static/rigv1-flat-gym.gif" alt="RigV1 training" height="280">
</p>

### 🔄 One-Command Retargeting

Transfer (retarget) the entire [AMASS](https://amass.is.tue.mpg.de/) dataset to your favorite robot with the built-in [**PyRoki**](https://github.com/chungmin99/pyroki)-based optimizer—in one command.

<p align="center">
  <img src="data/static/retargeting-g1.gif" alt="G1 retargeting" height="280">
</p>

### 🤖 Train Any Robot

Train your robot to perform AMASS motor skills in **12 hours**, by just changing one command argument:  
`--robot-name=smpl` → `--robot-name=h1_2` and preparing retargeted motions (see [here](https://protomotions.github.io/tutorials/workflows/retargeting_pyroki.html))

<p align="center">
  <img src="data/static/h1_2_gym.gif" alt="H1_2 AMASS training" height="280">
</p>

### 🔬 Sim2Sim Testing

One-click test (`--simulator=isaacgym` → `--simulator=newton`) of robot control policies on **H1_2** or **G1** in different physics engines (newly released [**NVIDIA Newton**](https://github.com/newton-physics/newton), built upon MuJoCo Warp). Policies shown below only use observations you could actually get from real hardwares.

<p align="center">
  <img src="data/static/h12-g1-newton-sim2sim.gif" alt="H1_2/G1 sim2sim" height="280">
</p>

### 🎨 High-Fidelity Rendering

Test your policy in [**IsaacSim 5.0+**](https://developer.nvidia.com/isaac-sim), which allows you to load beautifully rendered Gaussian splatting backgrounds (with [**Omniverse NuRec**](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/) — this rendered scene is not physically interact-able yet).

<p align="center">
  <img src="data/static/g1-neurc.gif" alt="G1 NeuRec" height="280">
</p>

### 🎬 Motion Authoring Integration

With a motion authoring model *(not included in ProtoMotions)*, generate any motion from a text prompt, and author a scene in ProtoMotions to go along with this motion—for both the animation character and the G1 robot to perform this stunt.

<p align="center">
  <img src="data/static/aibm-vaulting.gif" alt="Vaulting" height="280">
</p>

> *Image Credit: [NVIDIA Human Motion Modeling Research](https://research.nvidia.com/labs/sil/human_motion_modeling/)*

### 🏗️ Procedural Scene Generation

Procedurally generate many scenes for scalable **Synthetic Data Generation (SDG)**: start from a seed motion set, use RL to adapt motions to augmented scenes.

<p align="center">
  <img src="data/static/augmented_combined.gif" alt="Augmented Scenes and Motions" height="280">
</p>

### 🎭 Generative Policies

Train a generative policy (e.g., [**MaskedMimic**](https://research.nvidia.com/labs/par/maskedmimic/)) that can autonomously choose its "move" to finish the task.

<p align="center">
  <img src="data/static/smpl_masked_mimic.gif" alt="MaskedMimic SMPL" height="280">
</p>

### ⛰️ Terrain Navigation

Train your robot to hike challenging terrains!

<p align="center">
  <img src="data/static/smpl_terrain.gif" alt="SMPL Terrain" height="280">
</p>

### 🎯 Custom Environments

Have a new task? Implement your own custom [environment](protomotions/envs/steering/env.py):

<p align="center">
  <img src="data/static/g1_steering.gif" alt="G1 Steering" height="280">
</p>


### 🧪 New RL Algorithms

Want to try a new RL algorithm? Implement algorithms like **ADD** in ProtoMotions in ~50 lines of code, utilizing our modularized design:

📄 [`protomotions/agents/mimic/agent_add.py`](protomotions/agents/mimic/agent_add.py)

### 🔧 Custom Simulators

Would like to use your own simulator? Implement these APIs interfacing among different simulators:

📄 [`protomotions/simulator/base_simulator/`](protomotions/simulator/base_simulator/)

Refer to this community-contributed example:

📄 [`protomotions/simulator/genesis/`](protomotions/simulator/genesis/)

### 🤖 Add Your Own Robot

Want to add your own robot? Follow these steps:

1. Add your `.xml` MuJoCo spec file to [`protomotions/data/robots/`](protomotions/data/robots/)
2. Fill in config fields (see examples like [`protomotions/robot_configs/g1.py`](protomotions/robot_configs/g1.py))
3. Register in [`protomotions/robot_configs/factory.py`](protomotions/robot_configs/factory.py)

And you're good to go!

---

## Documentation

📚 **[Full Documentation](https://protomotions.github.io/)**

- [Installation Guide](https://protomotions.github.io/getting_started/installation.html)
- [Quick Start](https://protomotions.github.io/getting_started/quickstart.html)
- [AMASS Data Preparation](https://protomotions.github.io/getting_started/amass_preparation.html)
- [PHUMA Data Preparation](https://protomotions.github.io/getting_started/phuma_preparation.html)
- [Tutorials](https://protomotions.github.io/tutorials/)
- [API Reference](https://protomotions.github.io/api_reference/)

---

## Contributing

We welcome contributions! Please read our [**Contributing Guide**](CONTRIBUTING.md) before submitting pull requests.

## License

ProtoMotions3 is released under the [**Apache-2.0 License**](LICENSE.md).

---

## Citation

If you use ProtoMotions3 in your research, please cite:

```bibtex
@misc{ProtoMotions,
  title = {ProtoMotions3: An Open-source Framework for Humanoid Simulation and Control},
  author = {Tessler*, Chen and Jiang*, Yifeng and Peng, Xue Bin and Coumans, Erwin and Shi, Yi and Zhang, Haotian and Rempe, Davis and Chechik†, Gal and Fidler†, Sanja},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVLabs/ProtoMotions/}},
}
```
