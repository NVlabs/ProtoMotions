Release Notes
=============

This release expands ProtoMotions with a public Generative Pretrained Controllers
(GPC) workflow, new parameter-efficient fine-tuning examples, refreshed
pretrained tracker artifacts, and updated documentation for running and extending
the framework.

GPC and PEFT
------------

* Add the GPC training stack for discrete-token motor-control pretraining. The
  workflow trains an FSQ-bottleneck motion tracker, learns an autoregressive
  prior over grouped latent tokens, and adapts that prior to downstream tasks.
* Add PEFT agents and model components for adapting frozen discrete priors with
  SFT, RLFT, and RLFT+AMP. Public examples cover target reaching and steering,
  each with explicit ``--prior-checkpoint`` arguments so users can train or
  provide the prior they want to adapt.
* Add reusable FSQ, discrete-latent, autoregressive, frozen-decoder, and adapter
  utilities under the common agent and PEFT modules.
* Document the staged GPC workflow in :doc:`user_guide/gpc`, including tracker
  training, prior training, SFT bootstrapping, and PEFT fine-tuning.
* Keep the release example set focused: FSQ tracker training, GPC prior
  training, SFT target adaptation, target RLFT/RLFT+AMP, and steering
  RLFT/RLFT+AMP.
* Include the SOMA FSQ tracker used for GPC prior training, and mark pretrained
  GPC prior and PEFT skill checkpoints as forthcoming.
* Update checkpoint sidecars so the shipped tracker artifacts run from the
  packaged inference commands without private paths or stale observation
  bindings.

Framework Generalization
------------------------

* Generalize supervised imitation and distillation into a reusable agent path so
  model-specific workflows such as MaskedMimic and GPC SFT can share the same
  rollout, expert, loss, and checkpoint structure.
* Add a common fine-tuning lifecycle for agents that need frozen pretrained
  modules before model construction, keeping trackers, priors, and PEFT adapters
  on the same explicit pretrained-module contract.
* Refactor AMP discriminator training into a reusable component so AMP-style
  reward shaping can compose with fine-tuning agents instead of requiring a
  separate one-off training loop.
* Move task behavior into environment components and experiment wiring, so
  target, steering, mimic, GPC, and deployment examples now follow the same
  configuration patterns.

Task Control and Inference
--------------------------

* Add keyboard-controlled target commands for interactive target-reaching
  inference, alongside random target sampling for training.
* Improve quickstart and workflow docs for running pretrained models, using G1
  deployment assets, and exporting BeyondMimic-style tracker policies.
