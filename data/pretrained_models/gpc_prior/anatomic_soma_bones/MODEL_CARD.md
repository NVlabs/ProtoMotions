# Anatomical SOMA BONES-SEED GPC Prior

## License

Model license: NVIDIA Open Model License

The model weights, checkpoints, serialized model configurations and exported
models are licensed under the [NVIDIA Open Model License](LICENSE.pdf).
See [NOTICE.txt](NOTICE.txt) for attribution. Include both files when
redistributing these artifacts.

ProtoMotions source code remains Apache-2.0. Training datasets and robot assets
have separate terms; this model license does not grant rights to those materials.

## Overview

This checkpoint is an autoregressive Generative Pre-trained Controller (GPC)
prior for the anatomical SOMA humanoid (`anatomic_soma`: the SOMA23 skeleton
with its joints expanded into serial hinges, 52 hinges on 53 bodies, driven by
G1-class motors). It models the discrete FSQ motion codes of the packaged
anatomical SOMA BONES-SEED tracker and decodes sampled codes into low-level
joint actions through that tracker's frozen decoder.

| Field | Value |
| --- | --- |
| Family | Discrete autoregressive GPC prior |
| Robot | Anatomical SOMA, 53 bodies, 52 actions |
| Motion corpus | BONES-SEED, retargeted to the anatomical skeleton |
| Primary inference artifact | `inference_last.ckpt` |

## Intended Use

- Sample unconditional motion behavior from the anatomical SOMA FSQ vocabulary.
- Serve as the frozen base prior for GPC PEFT task adaptation on the
  anatomical robot.
- Provide a self-contained latent decoder at inference time.

## Training

- Training simulator: **IsaacLab**.
- Training recipe: `examples/experiments/gpc/prior_anatomic_soma.py`, shipped
  here as `experiment_config.py`: the GPC prior of `examples/experiments/gpc/prior.py`
  rolled out in the anatomical tracker's own world (BeyondMimic quarter-effort
  actions, self collisions, friction 1.0, the tracker's physical randomization).
- Training method: supervised autoregressive prediction of frozen tracker FSQ
  codes collected from expert rollouts of the shipped anatomical tracker.
- Training data: BONES-SEED motion retargeted to the anatomical SOMA skeleton.
- Prior context: the current robot state encoded by `max_coords_obs`.
- Prior network: a 6-layer transformer, 4 heads, width 1024, over the token
  sequence.
- Latent layout: 40 FSQ scalars with 9 levels each, grouped into 8 prior tokens
  of 5 scalars with a vocabulary of 59,049 values per token.
- The shipped weights are the state after 6343 training epochs.

The exact FSQ tracker used for this prior is shipped at
`data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq/`; the decoder
and encoder embedded in this checkpoint are tensor-for-tensor identical to that
bundle's.

## Inputs and Outputs

The prior consumes only `max_coords_obs`. It samples eight categorical latent
tokens, expands them to the tracker FSQ representation, and uses the embedded
frozen decoder to produce 52 joint actions.

Target poses are used only by the frozen encoder while generating supervised
training labels. They are not an inference input to the prior. Previous actions
and nearest-surface observations are not part of this prior's input contract.

## Artifacts

- `inference_last.ckpt`: model-only inference checkpoint (no optimizer or training
  state); the frozen decoder's weights load with it.
- `config.yaml`: the portable launcher arguments that regenerate the bundle.
- `experiment_config.py`: the public experiment source used to regenerate configs.
- `resolved_configs.json`: the self-contained v2 runtime configuration bundle, with
  the frozen decoder's construction config embedded. Inference derives its view at
  load, so there is no separate inference bundle.
- `provenance.json`: checkpoint provenance (sha256, byte count, tensor count).

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**. The tracker it decodes
  through was verified in MuJoCo; the prior's own rollouts were not.
- Use the `anatomic_soma` robot and the `resolved_configs.json` bundle shipped
  with the checkpoint.

## Limitations

- This is a motion prior, not a reference-motion tracker or task-specific
  controller.
- Motion quality depends on remaining within the state distribution covered by
  the training corpus.
- The FSQ token layout and decoder are part of the checkpoint contract and must
  not be mixed with a different tracker vocabulary; SOMA23 priors and trackers
  are not interchangeable with this robot (different joint coordinates, body
  count and action space).
- Cross-simulator performance is not guaranteed.

## Provenance

This card was curated from the shipped checkpoints, public experiment source,
regenerated public configuration artifacts, and the documented GPC data flow.
Machine-specific paths and private experiment identifiers are intentionally
omitted.
