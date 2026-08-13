# SOMA BONES-SEED Motion Tracker

## Overview

This is a continuous-action general motion tracker for the SOMA 23-body
humanoid on flat terrain.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker |
| Robot | SOMA 23-body humanoid, 66 actions |
| Motion corpus | BONES-SEED |
| Recommended checkpoint | `last_lab.ckpt` (IsaacLab) |
| Legacy checkpoint | `last.ckpt` |

## Intended Use

- Track BONES-SEED motion represented with the SOMA skeleton.
- Provide a continuous-action baseline for comparison with FSQ trackers.
- Run the bundled small SOMA BONES-SEED motion subset in IsaacLab with
  `last_lab.ckpt`.

## Training

- `last_lab.ckpt` was fine-tuned in **IsaacLab**.
- The training simulator for the legacy `last.ckpt` cannot be recovered from
  its generated public config.
- Training method: PPO motion tracking with maximum-coordinate self state,
  previous actions, and future target poses.
- Training data: BONES-SEED motion represented with the SOMA skeleton.
- Training environment: flat terrain and the SOMA PD-control contract.

## Inputs and Outputs

The policy consumes current SOMA state, the previous action, and future tracking
targets. It outputs 66 joint actions for the configured SOMA PD controller.

The serialized configs describe the legacy checkpoint. The IsaacLab fine-tune
uses the same model interface, but no separate resolved config is shipped for
it.

## Artifacts

- `last_lab.ckpt`: recommended IsaacLab-fine-tuned tracker checkpoint.
- `last.ckpt`: legacy base tracker checkpoint.
- `experiment_config.py`: tracker experiment wiring.
- `resolved_configs.pt`: serialized runtime configuration generated for the
  legacy checkpoint.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `compiled_models/unified_pipeline.onnx`: compiled export of the legacy
  `last.ckpt` policy.
- `compiled_models/kinematic_info.pt`: SOMA kinematic metadata for the compiled
  pipeline.

## Runtime Support

- Recommended runtime: **IsaacLab** with `last_lab.ckpt`.
- No runtime support claim is made for the legacy `last.ckpt`.
- Simulator expectation: **Training simulator only**.
- Use `data/motion_for_trackers/soma23_bones_seed_mini.pt` for the bundled
  IsaacLab example with `last_lab.ckpt`.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- The generated config beside `last.ckpt` targets MuJoCo but does not establish
  the simulator used to train that legacy checkpoint. MuJoCo support is not
  claimed.
- The model requires the SOMA skeleton, joint ordering, PD gains, observation
  processing, and control rate saved with the checkpoint.
- The policy tracks reference motion and does not generate motion on its own.
- The two checkpoints have different provenance. Do not substitute the legacy
  checkpoint for the IsaacLab-fine-tuned checkpoint.

## Provenance

This card was curated from the shipped checkpoint and compiled-model inventory,
serialized configurations, the bundled experiment definition, BONES-SEED
documentation, and the reproduced artifact limitation in issue #230.
Machine-specific paths and internal experiment identifiers are intentionally
omitted.
