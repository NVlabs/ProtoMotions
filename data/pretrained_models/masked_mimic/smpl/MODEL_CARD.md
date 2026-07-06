# SMPL MaskedMimic

## Overview

This is a MaskedMimic controller for the SMPL humanoid. The policy produces
physically simulated motion while conditioning on sparse or masked future body
targets.

| Field | Value |
| --- | --- |
| Family | MaskedMimic |
| Robot | SMPL humanoid, 24 bodies and 69 actions |
| Motion corpus | AMASS-derived SMPL motion |
| Primary checkpoint | `last.ckpt` |

## Intended Use

- Reconstruct or inpaint motion from partial future-pose constraints.
- Evaluate sparse body, time, and pose conditioning on an SMPL character.
- Serve as a reference implementation for MaskedMimic inference and research.

This checkpoint is not a text-to-motion model and is not compatible with G1 or
SOMA motion data.

## Training

- Training simulator: **IsaacLab**.
- Training method: supervised MaskedMimic training over masked future motion
  targets.
- Training data: AMASS-derived motion represented with the SMPL humanoid.
- Training environment: flat terrain with current and historical character
  state, previous actions, future target times, and body/pose masks.

## Inputs and Outputs

The policy consumes current and historical SMPL state, the previous action, and
masked future-pose targets with body and time masks. It outputs 69 joint actions
for the SMPL controller.

The exact mask layout, history length, normalization statistics, and model
dimensions are stored in `resolved_configs.pt`.

## Artifacts

- `last.ckpt`: full MaskedMimic policy checkpoint.
- `experiment_config.py`: experiment wiring captured with this model.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `resolved_configs.yaml` and `resolved_configs_inference.yaml`: readable
  configuration sidecars.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Supply an SMPL MotionLib prepared through the AMASS workflow.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- No matching SMPL MotionLib is bundled with this checkpoint.
- The public checkpoint supports masked motion constraints, not text control.
- No G1 checkpoint is included in this model directory.
- SMPL spherical-joint representations are not portable across all simulator
  backends.

## Provenance

This card was curated from the shipped checkpoint inventory, serialized
configurations, the bundled experiment definition, and public MaskedMimic and
AMASS documentation. Machine-specific paths and internal experiment identifiers
are intentionally omitted.
