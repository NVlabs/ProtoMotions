# SOMA BONES-SEED Motion Tracker

## Overview

This is a continuous-action general motion tracker for the SOMA 23-body
humanoid on flat terrain.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker |
| Robot | SOMA 23-body humanoid, 66 actions |
| Motion corpus | BONES-SEED |
| Primary checkpoint | `last.ckpt` |

## Intended Use

- Track BONES-SEED motion represented with the SOMA skeleton.
- Provide a continuous-action baseline for comparison with FSQ trackers.
- Run the bundled small SOMA BONES-SEED motion subset in IsaacLab.

## Training

- `last.ckpt` was fine-tuned in **IsaacLab**.
- Training method: PPO motion tracking with maximum-coordinate self state,
  previous actions, and future target poses.
- Training data: BONES-SEED motion represented with the SOMA skeleton.
- Training environment: flat terrain and the SOMA PD-control contract.

## Inputs and Outputs

The policy consumes current SOMA state, the previous action, and future tracking
targets. It outputs 66 joint actions for the configured SOMA PD controller.

## Artifacts

- `last.ckpt`: primary IsaacLab-fine-tuned tracker checkpoint.
- `experiment_config.py`: tracker experiment wiring.
- `resolved_configs.pt`: serialized runtime configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `compiled_models/unified_pipeline.onnx`: legacy compiled policy export.
- `compiled_models/kinematic_info.pt`: SOMA kinematic metadata for the compiled
  pipeline.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Use `data/motion_for_trackers/soma23_bones_seed_mini.pt` for the bundled
  IsaacLab example.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- The saved configuration preserves a legacy MuJoCo target and is not evidence
  of MuJoCo training or support. Use the documented IsaacLab invocation.
- The compiled ONNX file is a legacy export and is not the support contract for
  the current PyTorch checkpoint.
- The model requires the SOMA skeleton, joint ordering, PD gains, observation
  processing, and control rate saved with the checkpoint.
- The policy tracks reference motion and does not generate motion on its own.

## Provenance

This card was curated from the shipped checkpoint and compiled-model inventory,
serialized configurations, the bundled experiment definition, BONES-SEED
documentation, and the reproduced artifact limitation in issue #230.
Machine-specific paths and internal experiment identifiers are intentionally
omitted.
