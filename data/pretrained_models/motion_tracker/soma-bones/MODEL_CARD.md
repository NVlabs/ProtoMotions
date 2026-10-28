# SOMA BONES-SEED Motion Tracker

## License

Model license: NVIDIA Open Model License

The model weights, checkpoints, serialized model configurations and exported
models are licensed under the [NVIDIA Open Model License](LICENSE.pdf).
See [NOTICE.txt](NOTICE.txt) for attribution. Include both files when
redistributing these artifacts.

ProtoMotions source code remains Apache-2.0. Training datasets and robot assets
have separate terms; this model license does not grant rights to those materials.

## Overview

This is a continuous-action general motion tracker for the SOMA 23-body
humanoid on flat terrain.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker |
| Robot | SOMA 23-body humanoid, 66 actions |
| Motion corpus | BONES-SEED |
| Recommended checkpoint | `inference_last_lab.ckpt` (IsaacLab) |
| Legacy checkpoint | `inference_last.ckpt` |

## Intended Use

- Track BONES-SEED motion represented with the SOMA skeleton.
- Provide a continuous-action baseline for comparison with FSQ trackers.
- Run the bundled small SOMA BONES-SEED motion subset in IsaacLab with
  `inference_last_lab.ckpt`.

## Training

- `inference_last_lab.ckpt` was fine-tuned in **IsaacLab**.
- The training simulator for the legacy `inference_last.ckpt` cannot be recovered from
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

- `inference_last_lab.ckpt`: recommended IsaacLab-fine-tuned tracker checkpoint.
- `inference_last.ckpt`: legacy base tracker checkpoint.
- `experiment_config.py`: tracker experiment wiring.
- `config.yaml`: the portable launcher arguments that regenerate the bundle.
- `resolved_configs.json`: the self-contained v2 runtime configuration bundle,
  shared by both checkpoints.
- `provenance.json`, `provenance_lab.json`: per-checkpoint provenance (sha256,
  byte count, tensor count; the IsaacLab variant records `trained_on`).

## Runtime Support

- Recommended runtime: **IsaacLab** with `inference_last_lab.ckpt`.
- No runtime support claim is made for the legacy `inference_last.ckpt`.
- Simulator expectation: **Training simulator only**.
- Use `data/motion_for_trackers/soma23_bones_seed_mini.pt` for the bundled
  IsaacLab example with `inference_last_lab.ckpt`.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- The shipped `resolved_configs.json` targets IsaacLab but does not by itself
  establish the simulator used to train the legacy `inference_last.ckpt`.
- The model requires the SOMA skeleton, joint ordering, PD gains, observation
  processing, and control rate saved with the checkpoint.
- The policy tracks reference motion and does not generate motion on its own.
- The two checkpoints have different provenance. Do not substitute the legacy
  checkpoint for the IsaacLab-fine-tuned checkpoint.

## Provenance

This card was curated from the shipped checkpoint inventory, serialized
configurations, the bundled experiment definition, BONES-SEED documentation, and
the reproduced artifact limitation in issue #230.
Machine-specific paths and internal experiment identifiers are intentionally
omitted.
