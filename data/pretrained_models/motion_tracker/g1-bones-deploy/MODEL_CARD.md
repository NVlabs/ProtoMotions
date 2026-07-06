# G1 BONES-SEED Deployment Tracker

## Overview

This is a general motion-tracking policy for the Unitree G1 humanoid. It is the
deployment-oriented tracker shipped with both a PyTorch checkpoint and an ONNX
pipeline.

| Field | Value |
| --- | --- |
| Family | Motion tracker with AMP |
| Robot | Unitree G1, 29 actions |
| Motion corpus | BONES-SEED retargeted to G1 |
| Primary checkpoint | `last.ckpt` |
| Deployment export | `compiled_models/unified_pipeline.onnx` |

## Intended Use

- Track G1-compatible reference motion in simulation.
- Evaluate sim-to-sim transfer with the same robot, control, and observation
  contract.
- Export or deploy the unified ONNX policy pipeline using the G1 deployment
  workflow.

Real-robot use requires independent safety engineering, supervision, joint and
torque limits, and an emergency-stop path.

## Training

- Training simulator: **IsaacLab**.
- Training method: PPO motion tracking with adversarial motion-prior (AMP)
  regularization.
- Training data: BONES-SEED motion retargeted to the Unitree G1.
- Transfer-oriented randomization: friction, joint-state variation, sensor and
  observation noise, and external pushes.
- Policy contract: noisy reduced-coordinate observations for the actor, clean
  observations for training losses, target poses, and processed-action history.

## Inputs and Outputs

The policy consumes reduced-coordinate G1 proprioception, future tracking
targets, and action history. It outputs 29 joint targets for the configured G1
PD controller. The unified ONNX pipeline includes the observation processing
needed by the deployment interface.

Exact joint ordering, normalization statistics, history length, control rate,
and gains are stored in `resolved_configs.pt` and the compiled-model metadata.

## Artifacts

- `last.ckpt`: full PyTorch tracker checkpoint.
- `experiment_config.py`: deployment-oriented experiment wiring.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration.
- `compiled_models/unified_pipeline.onnx`: unified deployment policy.
- `compiled_models/unified_pipeline.yaml`: ONNX input/output and deployment
  metadata.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Expected to transfer**.
- The full transfer-oriented domain-randomization recipe is intended to make
  this G1 policy robust to simulator and hardware differences.

Expected transfer is not a claim that every simulator version is validated.
The target backend must preserve the G1 joint order, observation preprocessing,
control rate, PD gains, and action scaling.

## Limitations

- Transfer quality depends on matching the trained robot and control contract.
- The policy is a motion tracker; it does not generate reference motion.
- The ONNX export targets the documented deployment interface and is not a
  generic drop-in policy for arbitrary G1 stacks.
- The checkpoint is not safety certified for autonomous real-robot operation.

## Provenance

This card was curated from the shipped checkpoint and ONNX inventory,
serialized configurations, the bundled deployment experiment, BONES-SEED
documentation, and the public G1 deployment guide. Machine-specific paths and
internal experiment identifiers are intentionally omitted.
