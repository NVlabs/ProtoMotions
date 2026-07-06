# SOMA BONES-SEED FSQ Motion Tracker

## Overview

This is a SOMA motion tracker with a finite-scalar-quantization (FSQ) bottleneck.
It maps tracking observations to discrete latent codes and decodes those codes
into low-level actions. This tracker supplies the motion vocabulary used by the
SOMA GPC prior.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker with FSQ |
| Robot | SOMA 23-body humanoid, 66 actions |
| Motion corpus | BONES-SEED |
| Primary inference artifact | `inference_last.ckpt` |

## Intended Use

- Track BONES-SEED motion with a discrete FSQ action bottleneck.
- Encode target actions into discrete labels for GPC prior training.
- Supply the decoder embedded into SOMA GPC priors and PEFT workflows.

## Training

- Training simulator: **IsaacLab**.
- Training method: PPO motion tracking with an FSQ actor bottleneck.
- Training data: BONES-SEED motion represented with the SOMA skeleton.
- Training environment: flat terrain with current state, previous actions, and
  future tracking targets.

## Inputs and Outputs

The tracker consumes the current SOMA state, previous action, and future target
poses. Its actor quantizes a latent representation into FSQ codes and decodes
them into 66 joint actions.

Exact codebook levels, scalar grouping, normalization statistics, target
horizons, and dimensions are stored in `resolved_configs.pt`.

## Artifacts

- `inference_last.ckpt`: inference-oriented tracker artifact used by the public
  GPC configuration.
- `last.ckpt`: full training/resume checkpoint.
- `config.yaml`: concise artifact metadata.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `resolved_configs.yaml` and `resolved_configs_inference.yaml`: readable
  configuration sidecars.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Use the SOMA robot, target-pose horizon, and FSQ decoder contract saved with
  this checkpoint.

This model was not trained with the full transfer-oriented combination of
friction, joint-state, observation-noise, and push randomization. Cross-simulator
transfer should not be assumed.

## Limitations

- The FSQ latent layout is part of the checkpoint contract and cannot be mixed
  with a differently configured prior or tracker.
- The model is a tracker and GPC dependency, not an unconditional motion prior.
- No cross-simulator performance guarantee is provided.
- Performance depends on matching the SOMA asset, control rate, target horizon,
  and normalization state.

## Provenance

This card was curated from the shipped full and inference checkpoint inventory,
serialized configurations, public FSQ experiment definitions, BONES-SEED
documentation, and the public GPC guide. Machine-specific paths and internal
experiment identifiers are intentionally omitted.
