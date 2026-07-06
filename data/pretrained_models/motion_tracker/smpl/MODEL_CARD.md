# SMPL AMASS Flat-Terrain Tracker

## Overview

This is a general motion-tracking policy for the SMPL humanoid on flat terrain.
It follows reference motion supplied by an SMPL MotionLib.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker |
| Robot | SMPL humanoid, 24 bodies and 69 actions |
| Motion corpus | AMASS |
| Primary checkpoint | `last.ckpt` |

## Intended Use

- Track AMASS-derived SMPL motion on flat ground.
- Provide a baseline continuous-action motion tracker.
- Evaluate custom SMPL motions prepared with the AMASS conversion workflow.

## Training

- Training simulator: **IsaacLab**.
- Training method: PPO motion tracking with pose, velocity, and tracking
  termination objectives.
- Training data: AMASS motion represented with the repository's SMPL humanoid.
- Training environment: flat terrain with maximum-coordinate self state,
  previous actions, and future target poses.

## Inputs and Outputs

The policy consumes the current SMPL state, previous action, and future tracking
targets. It outputs 69 joint actions for the configured SMPL PD controller.

Exact target horizons, normalization statistics, control rate, and dimensions
are stored in `resolved_configs.pt`.

## Artifacts

- `last.ckpt`: full tracker checkpoint.
- `experiment_config.py`: flat-terrain tracker experiment wiring.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `assets/`: qualitative preview media for this model.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Supply an SMPL MotionLib prepared through the AMASS workflow.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- No AMASS MotionLib is bundled with this checkpoint.
- The model is trained for flat terrain; use the terrain tracker for mixed
  procedural terrain.
- SMPL spherical-joint representations are not portable across all simulator
  backends.
- The policy tracks reference motion and does not generate motion on its own.

## Provenance

This card was curated from the shipped checkpoint inventory, serialized
configurations, the bundled experiment definition, and AMASS documentation.
Machine-specific paths and internal experiment identifiers are intentionally
omitted.
