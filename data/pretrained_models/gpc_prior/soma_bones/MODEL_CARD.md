# SOMA BONES-SEED GPC Prior

## Overview

This checkpoint is an autoregressive Generative Pre-trained Controller (GPC)
prior for the SOMA 23-body humanoid. It models the discrete FSQ motion codes of
the packaged SOMA BONES-SEED tracker and decodes sampled codes into low-level
joint actions.

| Field | Value |
| --- | --- |
| Family | Discrete autoregressive GPC prior |
| Robot | SOMA 23-body humanoid, 66 actions |
| Motion corpus | BONES-SEED |
| Primary inference artifact | `inference_last.ckpt` |

## Intended Use

- Sample unconditional motion behavior from the SOMA FSQ vocabulary.
- Serve as the frozen base prior for GPC PEFT task adaptation.
- Provide a self-contained latent decoder at inference time.

## Training

- Training simulator: **IsaacLab**.
- Training method: supervised autoregressive prediction of frozen tracker FSQ
  codes collected from expert rollouts.
- Training data: BONES-SEED motion represented with the SOMA skeleton.
- Prior context: the current robot state encoded by `max_coords_obs`.
- Latent layout: 40 FSQ scalars grouped into 8 prior tokens with a vocabulary
  of 59,049 values per token.

The exact FSQ tracker used for this prior is shipped at
`data/pretrained_models/motion_tracker/soma_bones_fsq/`.

## Inputs and Outputs

The prior consumes only `max_coords_obs`. It samples eight categorical latent
tokens, expands them to the tracker FSQ representation, and uses the embedded
frozen decoder to produce 66 joint actions.

Target poses are used only by the frozen encoder while generating supervised
training labels. They are not an inference input to the prior. Previous actions
and nearest-surface observations are not part of this prior's input contract.

## Artifacts

- `inference_last.ckpt`: inference-oriented checkpoint without optimizer state.
- `last.ckpt`: full training/resume checkpoint.
- `config.yaml`: concise public artifact metadata.
- `experiment_config.py`: public experiment source used to regenerate configs.
- `resolved_configs.pt`: serialized training configuration.
- `resolved_configs_inference.pt`: serialized inference configuration with the
  frozen decoder construction config embedded.
- `resolved_configs.yaml` and `resolved_configs_inference.yaml`: readable
  configuration sidecars.

## Runtime Support

- Training simulator: **IsaacLab**.
- Simulator expectation: **Training simulator only**.
- Use the SOMA robot and the configuration sidecars shipped with the selected
  checkpoint.

## Limitations

- This is a motion prior, not a reference-motion tracker or task-specific
  controller.
- Motion quality depends on remaining within the state distribution covered by
  the training corpus.
- The FSQ token layout and decoder are part of the checkpoint contract and must
  not be mixed with a different tracker vocabulary.
- Cross-simulator performance is not guaranteed.

## Provenance

This card was curated from the shipped checkpoints, public experiment source,
regenerated public configuration artifacts, and the documented GPC data flow.
Machine-specific paths and private experiment identifiers are intentionally
omitted.
