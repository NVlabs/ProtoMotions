# SOMA BONES-SEED FSQ AMP Tracker

## Overview

This is a SOMA motion tracker that combines an FSQ actor bottleneck with
adversarial motion-prior (AMP) training. It also uses historical and
nearest-surface observations in its saved policy contract.

| Field | Value |
| --- | --- |
| Family | AMP motion tracker with FSQ |
| Robot | SOMA 23-body humanoid, 66 actions |
| Motion corpus | BONES-SEED |
| Primary inference artifact | `inference_last.ckpt` |

## Intended Use

- Track BONES-SEED motion with a discrete FSQ bottleneck and AMP regularization.
- Compare PPO-only and adversarially regularized FSQ trackers.
- Supply an alternative discrete tracker for representation-learning research.

## Training

- Training simulator: **IsaacLab**.
- Training method: PPO plus AMP discriminator training with an FSQ actor and a
  Muon-based actor optimizer.
- Training data: BONES-SEED motion represented with the SOMA skeleton.
- Training context: current and historical maximum-coordinate state,
  nearest-surface observations, and future tracking targets.

## Inputs and Outputs

The tracker consumes current and historical SOMA state, nearest-surface context,
and future target poses. Its actor quantizes a latent representation into FSQ
codes and decodes them into 66 joint actions. The discriminator is training-only
state and is not an additional inference input.

Exact codebook levels, history length, normalization statistics, target
horizons, and dimensions are stored in `resolved_configs.pt`.

## Artifacts

- `inference_last.ckpt`: inference-oriented tracker artifact.
- `last.ckpt`: full training/resume checkpoint, including training state.
- `config.yaml`: concise artifact metadata.
- `experiment_config.py`: tracker experiment wiring.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Use the SOMA robot and the historical and nearest-surface observation
  components saved with this checkpoint.

This model does not use the full G1 transfer-oriented combination of friction,
joint-state, observation-noise, and push randomization. Cross-simulator transfer
should not be assumed.

## Limitations

- AMP regularization improves the training objective but does not establish
  simulator transfer.
- The FSQ layout and nearest-surface observation contract must match exactly.
- The model is a tracker, not an unconditional GPC prior.
- No cross-simulator performance guarantee is provided.

## Provenance

This card was curated from the shipped full and inference checkpoint inventory,
serialized configurations, the bundled experiment definition, public FSQ and
AMP code, and BONES-SEED documentation. Machine-specific paths and internal
experiment identifiers are intentionally omitted.
