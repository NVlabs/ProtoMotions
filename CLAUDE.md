# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProtoMotions3 is a GPU-accelerated simulation and RL framework for training physically simulated digital humans and humanoid robots. It supports multiple physics simulators (IsaacGym, IsaacLab, Newton, Genesis, MuJoCo) and RL algorithms (PPO, AMP, ASE, MaskedMimic). Written in Python 3.8+, Apache-2.0 licensed.

## Common Commands

### Setup
```bash
pip install -e .
pip install -r requirements_isaacgym.txt  # or requirements_isaaclab.txt, requirements_newton.txt, requirements_genesis.txt, requirements_mujoco.txt
```

**MuJoCo CPU Backend**: For CPU-only testing, use `requirements_mujoco.txt`:
```bash
# Install PyTorch CPU version (lighter, no CUDA needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install -r requirements_mujoco.txt
```

Note: MuJoCo backend is CPU-only and supports `num_envs=1` only.

### Training
```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_experiment \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 4096 \
    --batch-size 16384

# With config overrides (these become PERMANENT in resolved_configs.pt)
python protomotions/train_agent.py ... --overrides agent.config.learning_rate=0.0001 env.max_episode_length=1000
```

### Inference
```bash
# G1 pretrained model
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --simulator isaacgym --num-envs 16

# SOMA pretrained model
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
    --motion-file data/motion_for_trackers/soma23_bones_seed_mini.pt \
    --simulator isaacgym --num-envs 16

# Sim2sim: train in isaacgym, test in newton
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --simulator newton --num-envs 16

# CPU-only inference with MuJoCo (single env)
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --simulator mujoco --num-envs 1
```

### Testing
```bash
pytest protomotions/tests/
pytest protomotions/tests/test_newton_simulator_fk.py  # single test file
```

### Linting and Formatting
```bash
# IMPORTANT: Do NOT use `pre-commit run --all-files` — many repo files don't conform yet,
# causing 100+ unrelated modifications. Instead, target specific files:
pre-commit run --files <file1> <file2> ...   # explicit file list
pre-commit run                                # runs only on staged files
pre-commit run ruff --files <file1> ...       # lint only, specific files
```

### ONNX Export (BeyondMimic Trackers)
```bash
# Reference script for BM tracker configs (auto-detects actor obs keys from checkpoint)
python deployment/export_bm_tracker_onnx.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

# For non-BM configs, copy and adapt this script to match your observation keys
```

## Key Dependencies

torch, lightning (Fabric), tensordict, wandb

**Newton simulator**: Tested against Newton repo commit `e7a737c`. Newton is installed from source (not PyPI) — see `requirements_newton.txt`.

**MuJoCo simulator**: CPU-only backend using the `mujoco` Python package (>=3.0). Supports single environment (`num_envs=1`) for lightweight testing and debugging. Useful for quick policy validation without GPU.

## Important Files

- `examples/experiments/format.py` — documents experiment config function signatures
- `protomotions/train_agent.py` — main entry point, documents config system in detail
- `protomotions/envs/mdp_component.py` — MdpComponent design docs
- `protomotions/envs/context_views.py` — FieldPath/context system
- `protomotions/utils/simulator_imports.py` — simulator import order handling

## Architecture

### Agent Hierarchy

```
BaseAgent (abstract) — training loop, checkpoints, Lightning Fabric
├── PPO — actor-critic, GAE advantages, clipped surrogate
│   ├── AMP — adds discriminator, replay buffer, style rewards
│   │   └── ASE — adds MI encoder, latent skills, diversity loss
│   └── Mimic/ADD — adds pose tracking diff (extends AMP)
└── MaskedMimic — expert distillation (behavioral cloning, not RL)
```

All models are `TensorDictModuleBase` subclasses. Forward passes read from and write to a shared `TensorDict`. Models use `nn.LazyLinear` extensively — input shapes are inferred on first forward pass.

Key methods each algorithm implements: `create_model()`, `perform_optimization_step()`, `record_rollout_step()`, `register_algorithm_experience_buffer_keys()`.

Training loop (`BaseAgent.fit()`): collect rollout (no_grad) → normalize rewards → compute advantages → optimize in minibatches → evaluate periodically.

### Configuration System

Configs are built from experiment Python files (not YAML), following this pipeline:

1. `robot_factory()` and `simulator_factory()` create base configs
2. `configure_robot_and_simulator()` customizes them for the experiment
3. `env_config()` builds the environment config
4. `agent_config()` builds the agent config
5. CLI `--overrides` are applied (these are saved permanently)
6. Everything is pickled to `resolved_configs.pt` for exact reproducibility

On **resume**, configs are loaded directly from pickle — the experiment file is NOT re-executed. On **inference**, `resolved_configs_inference.pt` is loaded, then `apply_inference_overrides()` runs, then CLI overrides.

Experiment files live in `examples/experiments/` and `examples/experiments/format.py` documents the required function signatures.

All configs use `_target_` strings for dynamic class instantiation (e.g., `_target_: "protomotions.agents.ppo.agent.PPO"`).

### MdpComponent System (`protomotions/envs/mdp_component.py`)

The core abstraction for observations, rewards, and terminations. An `MdpComponent` binds a pure tensor function to context paths, keeping compute separate from environment state:

```python
MdpComponent(
    compute_func=pure_tensor_function,
    dynamic_vars={"dof_pos": EnvContext.current.dof_pos},  # resolved at runtime
    static_params={"scale": 1.0}                           # fixed at creation
)
```

Three compute levels: Level 1 (pure tensor), Level 2 (aggregated, ONNX-exportable), Level 3 (with side effects). Components are managed by `ComponentManager`.

### Context Path System (`protomotions/envs/context_views.py`, `context_paths.py`)

`FieldPath` descriptors provide dual access — class-level access returns a path string, instance-level returns the actual tensor. This enables type-safe bindings between MdpComponents and environment data without copying.

Key views: `CurrentStateView` (current robot state), `HistoricalView` (state history buffer), `EnvContext` (namespace for all views). Control components populate their own views (e.g., `ctx.mimic`, `ctx.steering`).

### Environment Step Flow

`BaseEnv.step(actions)`:
1. Action processing (PD control, clamping, scaling) via `ComponentManager`
2. `simulator.step(actions)` — physics substeps with decimation
3. `post_physics_step()` — get new robot state, update context
4. Control components step (motion tracking, steering targets, etc.)
5. Observations computed via `ComponentManager` (MdpComponents)
6. Rewards computed via `ComponentManager` (MdpComponents)
7. Terminations computed via `ComponentManager` (MdpComponents)
8. Reset done environments

### Multi-Simulator Abstraction (`protomotions/simulator/`)

`Simulator` is the abstract base class with ~17 abstract methods. State exchange uses `RobotState`/`ObjectState` dataclasses with `StateConversion` (COMMON vs SIMULATOR format).

**Quaternion convention**: Common format uses xyzw. IsaacGym/IsaacLab use wxyz internally (converted automatically). Newton/Genesis use xyzw natively.

**Body/DOF ordering**: Each simulator has its own ordering. Conversion tensors (`body_convert_to_common`, `dof_convert_to_sim`) are computed once in `_finalize_setup()` and reused via tensor indexing.

**Two-phase initialization**: Constructor creates shell; `_initialize_with_markers()` allocates GPU memory after env provides visualization markers.

**Friction combine modes**: PhysX (IsaacGym/IsaacLab) uses AVERAGE, Newton uses MAX. `convert_friction_for_simulator()` handles conversion between them.

**Control modes**: `BUILT_IN_PD` (simulator-native), `PROPORTIONAL` (custom PD with action scaling), `TORQUE` (direct torque). Action noise domain randomization applied in `_apply_control()`.

### Robot Configuration (`protomotions/robot_configs/`)

Each robot has a config file defining assets, control parameters, and body mappings. `KinematicInfo` is extracted from MJCF at `__post_init__` time via `pose_lib.extract_kinematic_info()`.

Key fields: `common_naming_to_robot_body_names` (semantic body mapping — values must be **lists**), `control_info` (per-DOF stiffness/damping/effort from MJCF), `simulation_params` (per-simulator physics parameters).

### Components

**MotionLib** (`components/motion_lib.py`): Loads motion clips from .pt/.motion/.yaml files. Stores concatenated tensors (gts, grs, gvs, gavs, dps, dvs) with `length_starts` for O(1) motion indexing. SLERP interpolation for quaternions. Distributed loading via `.slurmrank.pt` per-rank files.

**PoseLib** (`components/pose_lib.py`): Batched forward kinematics from MJCF. Multi-horizon minimum velocity estimation (filters mocap noise). Automatic region weight discovery from kinematic tree (finds end effectors as leaf nodes, traces paths to root for limb regions).

**SceneLib** (`components/scene_lib.py`): Object management with mesh/box/sphere/cylinder primitives. Pointcloud sampling via trimesh for collision.

**Terrain** (`components/terrains/`): Procedural height field generation (slopes, stairs, stepping stones, etc.) with curriculum levels. Separate flat "object playground" region for scene objects.

### Key Directories

- `protomotions/envs/obs/` — observation compute kernels (humanoid, humanoid_historical, target_poses, masked_mimic, steering, path)
- `protomotions/envs/rewards/` — reward functions (tracking.py, regularization.py, task.py)
- `protomotions/envs/terminations/` — termination conditions (tracking.py, base.py)
- `protomotions/envs/action/` — action processing and PD control (action_functions.py)
- `protomotions/envs/control/` — control components (mimic, steering, path_follower, masked_mimic, kinematic_replay)
- `protomotions/envs/component_factories.py` — factory functions building MdpComponents from experiment configs
- `protomotions/agents/common/` — shared NN modules (MLPWithConcat, ModuleContainer — TensorDictModuleBase subclasses)
- `protomotions/utils/component_builder.py` — builds terrain, scene_lib, motion_lib, simulator from configs

## Gotchas

- **Import order matters**: Simulators (isaacgym/isaaclab) must be imported before torch. See `utils/simulator_imports.py`.
- **`resolved_configs.pt` is pickle, not torch tensors** — use `weights_only=False` when loading with `torch.load()`.
- **Resume mode ignores CLI `--overrides`** — configs are loaded from pickle, the experiment file is NOT re-executed.
- **Pre-existing F822 errors** in `component_factories.py` `__all__` — these are known, don't fix as part of unrelated work.
- **Robot body name mappings** (`common_naming_to_robot_body_names`) — values must be **lists**, not strings.

## Code Standards

- Pre-commit hooks enforce: Ruff linting/formatting, Apache-2.0 license headers on all .py files (except setup.py), typos spell checking
- Commits require sign-off (`git commit -s`) per DCO
- Use OOP (`nn.Module` subclasses) for model architectures; functional style for data processing pipelines
- All new `.py` files must have the full Apache-2.0 license header (not abbreviated):
```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```
