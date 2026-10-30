<div align="center">

# ProtoMotions 3

**A GPU-Accelerated Framework for Simulated Humanoids**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![Documentation](https://img.shields.io/badge/docs-online-green.svg)](https://nvlabs.github.io/ProtoMotions/)

[![Newton](https://img.shields.io/badge/Newton-1.0.0-brightgreen.svg)](https://pypi.org/project/newton/1.0.0/)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-3.0-blue.svg)](https://github.com/isaac-sim/IsaacLab/commit/4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8)
[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview_4-blue.svg)](https://developer.nvidia.com/isaac-gym)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.0+-orange.svg)](https://github.com/google-deepmind/mujoco)

</div>

---

## 📰 News

- **Oct 28, 2026: Metatron is released.** Metatron is our generative whole-body policy for the Unitree G1: one network that handles motion tracking, steering, path following, reaching for sparse body goals, scene interaction and getting back up after a fall. Read the [**paper**](https://research.nvidia.com/labs/par/projects/metatron/assets/metatron-paper.pdf?v=de2411a21822) and the [**project page**](https://research.nvidia.com/labs/par/projects/metatron/), try it in your browser with the [**interactive demo**](https://nvlabs.github.io/ProtoMotions/demos/metatron/), race six of them in [**G1 Racer**](https://nvlabs.github.io/ProtoMotions/demos/g1-racer/), download the weights and the native package from [**Hugging Face**](https://huggingface.co/nvidia/Metatron-v1) with `python scripts/download_metatron.py`, or train your own with the [**Metatron G1 student guide**](https://nvlabs.github.io/ProtoMotions/user_guide/metatron_student.html).
- **Oct 28, 2026: `main` received a major refactor.** The environment is now a short list of explicit step phases over modular state owners and task control components, with checked boundaries between them, so a change in one place cannot silently break another. Experiments use a cleaner [**configuration system**](https://nvlabs.github.io/ProtoMotions/user_guide/configuration.html) and a declarative way of [**authoring observations, rewards and terminations**](https://nvlabs.github.io/ProtoMotions/concepts/environment_context.html) as pure tensor kernels bound to named context paths. Sensors are declared once per channel, each with [**measurement noise and a state-estimation stage**](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/domain_randomization.html#privileged-clean-and-noisy-sources), which gives domain randomization and sim-to-real transfer a principled footing. Saved configurations from earlier versions must be regenerated, and the Genesis backend has been removed; the [**changelog**](https://nvlabs.github.io/ProtoMotions/changelog.html) has the full list.

---

## Overview

**ProtoMotions3** is a GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots. Our mission is to provide a **fast prototyping platform** for various simulated humanoid learning tasks and environments—for researchers and practitioners in **animation**, **robotics**, and **reinforcement learning**—bridging efforts across communities.

**Modularity**, **extensibility**, and **scalability** are at the core of ProtoMotions3. It is **community-driven** and permissively licensed under the [Apache-2.0 license](LICENSE.md).

Also check out **[MimicKit](https://github.com/xbpeng/MimicKit/tree/main)**, our sibling repository for a lightweight framework for motion imitation learning.

---

## What You Can Do with ProtoMotions3

### 🎭 Generative Policies

Train a generative policy that chooses its own "move" to finish a task instead of following one reference clip. [**MaskedMimic**](https://research.nvidia.com/labs/par/maskedmimic/) learns a single policy that completes a motion from partial goals such as a few target bodies, a path or objects in the scene. The GPC recipe trains a reusable discrete latent prior and then adapts it to new tasks with small PEFT adapters; see the [**GPC and PEFT guide**](https://nvlabs.github.io/ProtoMotions/user_guide/gpc.html). Our released Unitree G1 policy, [**Metatron**](https://research.nvidia.com/labs/par/projects/metatron/), is trained this way: a MaskedMimic-style student distilled from a motion-tracking teacher, whose learned prior turns tracking, steering, body-goal, scene and recovery goals into whole-body motion. Run it in the [**browser demo**](https://nvlabs.github.io/ProtoMotions/demos/metatron/), race it in [**G1 Racer**](https://nvlabs.github.io/ProtoMotions/demos/g1-racer/), download the weights from [**Hugging Face**](https://huggingface.co/nvidia/Metatron-v1), or follow the [**Metatron G1 student guide**](https://nvlabs.github.io/ProtoMotions/user_guide/metatron_student.html) to train your own.

<table align="center">
<tr>
<td align="center"><img src="data/static/gpc_parkour.gif" alt="GPC human-scene interaction" height="200"/></td>
<td align="center"><img src="data/static/racer.gif" alt="G1 Racer" height="200"/></td>
</tr>
<tr>
<td align="center"><sub>GPC: human-scene interaction</sub></td>
<td align="center"><sub><a href="https://nvlabs.github.io/ProtoMotions/demos/g1-racer/">G1 Racer</a>: six robots, one batched Metatron policy, in your browser</sub></td>
</tr>
<tr>
<td align="center" colspan="2"><img src="data/static/metatron.gif" alt="Metatron on the real G1" width="640"/></td>
</tr>
<tr>
<td align="center" colspan="2"><sub><a href="https://nvlabs.github.io/ProtoMotions/demos/metatron/">Metatron</a> on the real G1: sitting and standing, whole-body teleoperation, recovery from the ground</sub></td>
</tr>
</table>

### 🏃 Large-Scale Motion Learning

Train your fully physically simulated character to learn motion skills from the entire public [**AMASS**](https://amass.is.tue.mpg.de/) human animation dataset (**40+ hours**) within **12 hours** on 4 A100s.

<p align="center">
  <img src="data/static/smpl_mlp_094132.gif" alt="SMPL motion 1" height="180">
  <img src="data/static/smpl_mlp_094428.gif" alt="SMPL motion 2" height="180">
  <img src="data/static/smpl_mlp_095344.gif" alt="SMPL motion 3" height="180">
  <img src="data/static/smpl_mlp_095848.gif" alt="SMPL motion 4" height="180">
</p>

### 📈 Scalable Multi-GPU Training

Scale training to even larger datasets with each GPU handling a subset of motions. For example, we have trained with **24 A100s** with **13K motions** on each GPU with the [**BONES**](https://huggingface.co/datasets/bones-studio/seed) dataset in [**SOMA**](https://github.com/NVlabs/SOMA-X) skeleton format. Check out [Quick Start](https://nvlabs.github.io/ProtoMotions/getting_started/quickstart.html) and [SEED BVH Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/seed_bvh_preparation.html) to play around with the dataset and pre-trained models today.

<p align="center">
  <img src="data/static/soma_regen_markers.gif" height="180">
  <img src="data/static/soma_regen_2.gif" height="180">
  <img src="data/static/soma_regen_3.gif" height="180">
  <img src="data/static/soma_regen_5.gif" height="180">
</p>

### 🔄 One-Command Retargeting

Transfer (retarget) the entire [AMASS](https://amass.is.tue.mpg.de/) dataset to your favorite robot with the built-in [**PyRoki**](https://github.com/chungmin99/pyroki)-based optimizer—in one command.

> **Note:** As of v3, we use [PyRoki](https://github.com/chungmin99/pyroki) for retargeting. Earlier versions used [Mink](https://github.com/kevinzakka/mink).

<p align="center">
  <img src="data/static/retargeting-g1.gif" alt="G1 retargeting" height="280">
</p>

### 🤖 Train Any Robot

Train your robot to perform AMASS motor skills in **12 hours**, by just changing one command argument:  
`--robot-name=smpl` → `--robot-name=h1_2` and preparing retargeted motions (see [here](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/retargeting_pyroki.html))

<p align="center">
  <img src="data/static/h1_2_gym.gif" alt="H1_2 AMASS training" height="280">
</p>

### 🔬 Sim2Sim Testing

One-click test (`--simulator=isaacgym` → `--simulator=newton` → `--simulator=mujoco`) of robot control policies on **H1_2** or **G1** in different physics engines (NVIDIA Newton, MuJoCo CPU). Policies shown below only use observations you could actually get from real hardware.

<p align="center">
  <img src="data/static/h12-g1-newton-sim2sim.gif" alt="H1_2/G1 sim2sim" height="280">
</p>

### 🤖 From Sim to Real

Train in simulation, deploy on real hardware. ProtoMotions trains one General Tracking Policy on entire [**BONES-SEED**](https://huggingface.co/datasets/bones-studio/seed) dataset (~142K motions) and transfers directly to the Unitree G1 humanoid robot zero-shot.

<p align="center">
  <img src="data/static/g1_deploy_1.gif" alt="G1 deployment 1" height="240">
  <img src="data/static/g1_deploy_2.gif" alt="G1 deployment 2" height="240">
  <img src="data/static/real_robot_2.gif" alt="G1 real robot" height="240">
</p>

<p align="center">
  <img src="data/static/sit-sim2real.gif" alt="G1 sitting on furniture from scene boxes and body goals, in MuJoCo and on the real robot" width="400">
  <img src="data/static/run-sim2real.gif" alt="Real G1 steering in different directions" width="400">
</p>

Our deployment pipeline exports a single ONNX model (with observation computation baked in), so deployment frameworks only need to provide raw sensor signals — no need to rewrite obs functions or match training internals. We tested on the Unitree G1 via the brilliant [**RoboJuDo**](https://github.com/HansZ8/RoboJuDo) framework, adding just one policy file with no mandatory changes to RoboJuDo core.

📖 [**Full Deployment Tutorial**](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/g1_deployment.html) — from data preparation to real robot, fully reproducible.

Large generative policies such as Metatron are a different deployment problem from a small tracking network: a transformer with several hundred megabytes of weights has to return a command within its policy period, every step, so latency has to be engineered end to end instead of left to a general-purpose Python stack. For that we also prototype our own deployment runtime: a standalone C++ runtime with MuJoCo and Unitree G1 backends, a safety supervisor, a policy mapping workflow, and a TensorRT build and qualification pipeline. See [**Deployment runtime (experimental)**](deployment/README.md). That folder is prototyping reference code for people building their own deployment path, not a supported release artifact.

### 🎨 High-Fidelity Rendering

Test your policy in [**IsaacSim 5.0+**](https://developer.nvidia.com/isaac-sim), which allows you to load beautifully rendered Gaussian splatting backgrounds (with [**Omniverse NuRec**](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/) — this rendered scene is not physically interact-able yet).

<p align="center">
  <img src="data/static/g1-neurc.gif" alt="G1 NeuRec" height="280">
</p>

### 🎬 Motion Authoring with Kimodo

With [**Kimodo**](https://research.nvidia.com/labs/sil/projects/kimodo/) (NVIDIA's text-to-motion generation model), generate any motion from a text prompt and use ProtoMotions to train a physics-based policy that performs the motion — for both the SOMA animation character and the Unitree G1 robot. Policies trained this way can be deployed directly on real hardware.

See [Kimodo Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/kimodo_preparation.html) for how to convert Kimodo outputs to ProtoMotions format.

<p align="center">
  <img src="data/static/aibm-vaulting.gif" alt="Vaulting" height="240">
  <img src="data/static/g1_robot_walking.gif" alt="G1 robot walking" height="240">
</p>

> *Image Credit: [NVIDIA Human Motion Modeling Research](https://research.nvidia.com/labs/sil/human_motion_modeling/)*



### 🏗️ Procedural Scene Generation

Procedurally generate many scenes for scalable **Synthetic Data Generation (SDG)**: start from a seed motion set, use RL to adapt motions to augmented scenes.

<p align="center">
  <img src="data/static/augmented_combined.gif" alt="Augmented Scenes and Motions" height="280">
</p>

### ⛰️ Terrain Navigation

Train your robot to hike challenging terrains!

<p align="center">
  <img src="data/static/smpl_terrain.gif" alt="SMPL Terrain" height="280">
</p>

### 🎯 Custom Environments

Have a new task? Build it from modular components — no monolithic env class needed. Here's how the **steering** task is composed:

| Layer | File | What it does |
|-------|------|-------------|
| **Control** | [`steering_control.py`](protomotions/envs/control/steering_control.py) | Manages task state (target direction, speed, facing). Periodically samples new heading targets. |
| **Observation** | [`obs/steering.py`](protomotions/envs/obs/steering.py) | Pure tensor kernel — transforms targets to robot-local frame → 5D feature vector. |
| **Reward** | [`rewards/task.py`](protomotions/envs/rewards/task.py) | `compute_heading_velocity_rew` — blends direction-matching (0.7) and facing-matching (0.3) rewards. |
| **Experiment** | [`steering/mlp.py`](examples/experiments/steering/mlp.py) | Wires components together as `MdpComponent` instances via context paths. |

Each piece is a standalone function or class — the experiment config binds them into a complete task using [`MdpComponent`](protomotions/envs/mdp_component.py) and [`FieldPath`](protomotions/envs/context_views.py) descriptors.

<p align="center">
  <img src="data/static/g1_steering.gif" alt="G1 Steering" height="280">
</p>


### 🧪 New RL Algorithms

Want to try a new RL algorithm? Implement algorithms like **ADD** in ProtoMotions in ~50 lines of code, utilizing our modularized design:

📄 [`protomotions/agents/mimic/agent_add.py`](protomotions/agents/mimic/agent_add.py)

### 🔧 Custom Simulators

Would like to use your own simulator? Implement these APIs interfacing among different simulators:

📄 [`protomotions/simulator/base_simulator/`](protomotions/simulator/base_simulator/)

Refer to the MuJoCo backend as the smallest reference implementation:

📄 [`protomotions/simulator/mujoco/`](protomotions/simulator/mujoco/)

### 🤖 Add Your Own Robot

Want to add your own robot? Follow these steps:

1. Add your `.xml` MuJoCo spec file to [`protomotions/data/assets/mjcf/`](protomotions/data/assets/mjcf/)
2. Create `protomotions/robot_configs/<your_robot>.py` defining exactly one class whose direct base is `RobotConfig` (copy [`protomotions/robot_configs/g1.py`](protomotions/robot_configs/g1.py) and fill in the fields)
3. That is the registration: [`protomotions/robot_configs/factory.py`](protomotions/robot_configs/factory.py) scans that directory, so the file's stem is the name you pass to `--robot-name`. There is nothing to edit in the factory. If the robot goes into this repository and you run the test suite, also add its name to `_ROBOTS` in [`protomotions/tests/test_robot_facts_d1.py`](protomotions/tests/test_robot_facts_d1.py), which pins the registered list.

And you're good to go!

---

## Documentation

📚 **[Full Documentation](https://nvlabs.github.io/ProtoMotions/)**

- [Installation Guide](https://nvlabs.github.io/ProtoMotions/getting_started/installation.html)
- [Quick Start](https://nvlabs.github.io/ProtoMotions/getting_started/quickstart.html)
- [GPC and PEFT](https://nvlabs.github.io/ProtoMotions/user_guide/gpc.html)
- [AMASS Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/amass_preparation.html)
- [PHUMA Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/phuma_preparation.html)
- [SEED BVH Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/seed_bvh_preparation.html)
- [SEED G1 CSV Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/seed_g1_csv_preparation.html)
- [Kimodo Data Preparation](https://nvlabs.github.io/ProtoMotions/getting_started/kimodo_preparation.html)
- [Tutorials](https://nvlabs.github.io/ProtoMotions/tutorials/)
- [API Reference](https://nvlabs.github.io/ProtoMotions/api_reference/)
- [G1 Deployment: Data to Real Robot](https://nvlabs.github.io/ProtoMotions/tutorials/workflows/g1_deployment.html)

---

## Contributing

We welcome contributions! Please read our [**Contributing Guide**](CONTRIBUTING.md) before submitting pull requests.

## License

ProtoMotions3 source code is released under the [**Apache-2.0 License**](LICENSE.md).

NVIDIA-provided pretrained models are released under the **NVIDIA Open Model
License**, unless their model card explicitly states otherwise. Each model
directory includes its model card, license and attribution notice. See
[model licensing and redistribution](legal/MODEL-LICENSING.md).

Third-party software and bundled asset notices are listed in [legal/](legal/), including Unitree, BeyondMimic, Isaac Lab, and SMPL/SMPL-H attribution and license notices.

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
