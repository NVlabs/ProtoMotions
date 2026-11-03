# Anatomical SOMA BONES-SEED FSQ Motion Tracker

## License

Model license: NVIDIA Open Model License

The model weights, checkpoints, serialized model configurations and exported
models are licensed under the [NVIDIA Open Model License](LICENSE.pdf).
See [NOTICE.txt](NOTICE.txt) for attribution. Include both files when
redistributing these artifacts.

ProtoMotions source code remains Apache-2.0. Training datasets and robot assets
have separate terms; this model license does not grant rights to those materials.

## Overview

This is a motion tracker for the anatomical SOMA humanoid (`anatomic_soma`): the
SOMA23 skeleton with its joints expanded into serial hinges with engineering
joint limits (15 joints get three hinges, the knees, elbows and toes one, Neck2
a single yaw: 52 hinges on 53 bodies, 30 of them massless-in-effect carrier
links), driven by G1-class motors. The policy has a
finite-scalar-quantization (FSQ) bottleneck: it maps tracking observations to
discrete latent codes and decodes them into low-level actions, so it can supply
the motion vocabulary for a GPC prior on this robot.

| Field | Value |
| --- | --- |
| Family | PPO motion tracker with FSQ |
| Robot | Anatomical SOMA, 53 bodies, 52 actions |
| Motion corpus | BONES-SEED, retargeted to the anatomical skeleton |
| Primary inference artifact | `inference_last.ckpt` |

## Intended Use

- Track BONES-SEED motion retargeted to `anatomic_soma` with a discrete FSQ
  action bottleneck.
- Serve as the frozen tracker for a GPC prior on the anatomical SOMA robot.
- Act as a sim-to-sim reference policy for the anatomical robot (see Runtime
  Support).

## Training

- Training simulator: **IsaacLab**.
- Training recipe: `examples/experiments/mimic/anatomical_soma_fsq_mimic.py`,
  shipped here as `experiment_config.py`: the SOMA23 FSQ/PPO tracker with
  G1-class actuators.
- Actions: BeyondMimic scaling (a unit action moves the joint target far enough
  that the PD spring alone would push with a quarter of the joint's effort limit;
  no tanh projection onto the joint range).
- Rewards: the G1 BeyondMimic tracker's Gaussian body-tracking terms summed raw, plus the
  SOMA foot-contact match, a soft joint-limit penalty and an action-rate penalty.
- Randomization: friction, torso centre of mass, hand mass (+/-20% of the hand's
  own mass), pushes, action noise and reset noise; no observation noise.
- Exploration: the run learned a per-clip action std (one row per training
  clip, initialized at log-std -1.51 and ending near -0.94). The shipped recipe
  trains with a fixed log-std of -1.51 instead.
- Training data: the full BONES-SEED corpus retargeted to the anatomical skeleton
  with `data/scripts/from_soma_to_anatomical_soma/retarget_motions.py`.
  `config.yaml` names the small public mini pack only because it regenerates the
  configuration; the corpus itself is not shipped.
- Termination during training: an episode ends when any body's position error
  exceeds 0.5 m (the SOMA tracker's rule). Evaluation success below uses the
  mean body-position error staying under 0.5 m.

## Inputs and Outputs

The tracker consumes the current anatomical SOMA maximum-coordinate state
(observations are expressed relative to the `Chest` anchor) and future target
poses for all 53 bodies, positions, rotations and velocities, at the saved
future-step horizon. Its actor quantizes a latent representation into FSQ codes
and decodes them into 52 joint actions.

Exact codebook levels, scalar grouping, normalization statistics, target
horizons and dimensions are stored in `resolved_configs.json`.

## Artifacts

- `inference_last.ckpt`: the model weights plus light run metadata (epoch,
  step count, best evaluated score); no optimizer or replay state, not
  resumable. The run's per-clip exploration log-std table is averaged over
  clips to a single row (about -0.94 per joint) so the tensor matches the fixed
  std actor the recipe builds; the value is kept for shape compatibility only.
  Inference uses the mean action and never reads it, and it is not the -1.51
  the recipe starts fresh training from.
- `experiment_config.py`: the tracker experiment wiring.
- `config.yaml`: the portable launcher arguments that regenerate the bundle.
- `resolved_configs.json`: the self-contained v2 runtime configuration bundle.
- `provenance.json`: checkpoint provenance (sha256, byte count, tensor count).

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Expected to transfer**.
- Use the `anatomic_soma` robot, target-pose horizon and FSQ decoder contract
  saved with this checkpoint, and motions retargeted to the anatomical skeleton.

Trained with physical randomization and evaluated with randomization off and
fixed friction (the recipe's inference override). On a 256-clip BONES-SEED
subset the tracker completes every clip in IsaacLab and in MuJoCo with a mean
body-position error of about 0.05 m in both; on the shipped ten-clip mini pack
it completes every clip in both simulators with about 0.045 m. Newton and
IsaacGym were not evaluated.

## Limitations

- Body-level quantities run over all 53 bodies, carriers included: each
  three-hinge joint's carriers sit on their semantic body, so those joints count
  three times in the target observation, the relative-body and velocity rewards,
  the termination and the evaluation error, while single-hinge joints count
  once. The policy was trained and evaluated this way; the error numbers above
  are therefore not comparable to SOMA23 tracker numbers.
- SOMA23 motions, checkpoints and GPC priors are not interchangeable with this
  robot: the joint coordinates, body count and action space differ.
- The FSQ latent layout is part of the checkpoint contract and cannot be mixed
  with a differently configured prior or tracker.
- The model is a tracker, not an unconditional motion prior; no GPC prior for
  this robot is shipped yet.
- Evaluation covers BONES-SEED clips retargeted to this skeleton; behaviour on
  other motion sources or on hardware is not characterized.

## Provenance

This card was curated from the shipped inference checkpoint, its serialized
configuration, the public recipe and robot definitions, the conversion tools and
the evaluation runs described above. Machine-specific paths and internal
experiment identifiers are intentionally omitted.
