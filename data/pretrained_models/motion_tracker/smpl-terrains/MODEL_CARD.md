# SMPL AMASS Terrain Tracker

## Overview

This is a general SMPL motion tracker trained on a mixture of procedural terrain
types. It follows an SMPL reference motion while the environment places the
character on non-flat ground.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker |
| Robot | SMPL humanoid, 24 bodies and 69 actions |
| Motion corpus | AMASS |
| Primary checkpoint | `last.ckpt` |

## Intended Use

- Track AMASS-derived SMPL motion on the repository's procedural terrains.
- Compare flat-only and mixed-terrain motion-tracking policies.
- Evaluate terrain-aware reset, termination, and tracking behavior.

## Training

- Training simulator: **IsaacLab**.
- Training method: PPO motion tracking with a mixed procedural-terrain
  curriculum.
- Training data: AMASS motion represented with the repository's SMPL humanoid.
- Terrain mixture includes slopes, stairs, rough terrain, and flat ground as
  configured in the serialized artifact.

## Inputs and Outputs

The policy consumes the current SMPL state, previous action, and future tracking
targets in the terrain-aware environment. It outputs 69 joint actions for the
configured SMPL PD controller.

Exact terrain proportions, target horizons, normalization statistics, control
rate, and dimensions are stored in `resolved_configs.pt`.

## Artifacts

- `last.ckpt`: full terrain tracker checkpoint.
- `experiment_config.py`: mixed-terrain tracker experiment wiring.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `resolved_configs.yaml` and `resolved_configs_inference.yaml`: readable
  configuration sidecars.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Supply an SMPL MotionLib prepared through the AMASS workflow and use the
  checkpoint's terrain configuration.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- No AMASS MotionLib is bundled with this checkpoint.
- The policy is tied to the saved terrain-generation and SMPL control contract.
- SMPL spherical-joint representations are not portable across all simulator
  backends.
- Terrain training does not imply support for arbitrary meshes or scene assets.

## Provenance

This card was curated from the shipped checkpoint inventory, serialized
configurations, the bundled experiment definition, and AMASS documentation.
Machine-specific paths and internal experiment identifiers are intentionally
omitted.
