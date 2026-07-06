GPC and PEFT
============

GPC experiments train a reusable discrete latent prior, then adapt that prior to
task skills with PEFT. The structure is intentionally the same as the rest of
ProtoMotions: the experiment file owns the data flow, agent configs describe the
model, and observation keys stay explicit.

Training Stages
---------------

Tracker
~~~~~~~

A motion tracker is trained first. GPC uses the tracker's actor module as a
frozen latent autoencoder:

* the tracker encoder maps target poses to FSQ codes;
* the tracker quantizer converts codes to discrete token indices;
* the tracker decoder maps generated codes back to robot actions.

.. note::

   The tracker used for GPC must expose an FSQ bottleneck: its encoder maps
   target poses into finite scalar quantized codes, and its decoder maps
   generated FSQ codes back to robot actions. For a concrete training setup,
   see ``examples/experiments/mimic/fsq.py``. That experiment trains the
   motion tracker whose tokens and decoder are reused by the GPC prior.

Prior
~~~~~

``examples/experiments/gpc/prior.py`` trains the autoregressive prior. It uses
``DiscreteAutoregressiveLatentSupervisedAgentConfig`` with expert rollouts from the
frozen tracker. The model learns to predict tracker FSQ tokens from prior context
observations such as ``max_coords_obs``.

The saved prior checkpoint is a full
``DiscreteAutoregressiveLatentPriorModel``. PEFT configs should load this whole
model through ``pretrained_modules["prior"]``. Do not point PEFT at an old
``actor.mu`` submodule path; PEFT needs the prior transformer, latent grouping,
frozen decoder, and SFT target encoder.

SFT
~~~

``examples/experiments/gpc/sft_target_prior_peft.py`` bootstraps a task adapter
with supervised fine-tuning. The tracker provides target FSQ tokens from
``mimic_target_poses``. The PEFT actor receives ``task_obs`` from the same target
observation factory used by RLFT, but the target comes from a future root XY
point on the reference clip plus jitter. This keeps the SFT data path close to
the later task-learning path.

SFT uses ``DiscretePriorPEFTSFTAgentConfig`` and no critic. The main loss is the
configured supervised loss between generated latent logits and frozen
encoder target tokens. SFT and RLFT both load the frozen prior through the
shared ``pretrained_modules`` lifecycle, so changing the prior checkpoint path
uses the same config shape in both training loops.

RLFT
~~~~

``examples/experiments/gpc/task_target_prior_peft.py`` fine-tunes the adapter
with PPO on task rewards. The actor config stays compatible with the SFT config,
so an SFT checkpoint can warm-start RLFT. The environment usually swaps from the
SFT mimic-target source to a task control source such as random target reaching.

``examples/experiments/gpc/task_target_prior_peft_amp.py`` adds AMP rewards to
the same PEFT actor and task critic when dense style reward is useful.

PEFT Config Contract
--------------------

The public discrete-prior PEFT actor shape is:

.. code-block:: python

   DiscretePriorPEFTActorConfig(
       in_keys=["task_obs"],
       out_keys=["action", "mean_action", "neglogp", "prior_tokens"],
       peft=DiscretePriorPEFTConfig(
           model=ModuleContainerConfig(
               in_keys=["task_obs"],
               out_keys=["task_cond"],
               models=[...],
           ),
           condition_key="task_cond",
           ...
       ),
   )

``actor.in_keys`` declares the task observations needed to build PEFT
conditioning. ``actor.peft.model`` is a TensorDict module that consumes those
keys and writes ``actor.peft.condition_key``. That condition key is the only
task-conditioning tensor produced by the public config. ``DiscretePriorWithPEFT`` then
combines it with the frozen prior context keys discovered from the checkpoint.

The frozen prior's own context keys are discovered from the loaded prior
checkpoint and appended by ``DiscretePriorPEFTActor`` at runtime. Experiment configs do
not need legacy routing fields such as ``task_conditioning_keys``,
``terrain_key``, ``conditioning_model``, or actor-level target/terrain context
keys on ``DiscretePriorPEFTActorConfig`` or ``DiscretePriorPEFTConfig``. A concrete
``actor.peft.model`` may still have its own module-specific fields, such as a
terrain encoder's input key.

If ``actor.peft.model`` is omitted, ``DiscretePriorPEFTConfig`` builds a small default
``ObsProcessorConfig`` that normalizes and concatenates ``actor.in_keys`` into
``condition_key``. Use an explicit ``actor.peft.model`` when the task needs a
real conditioning network or extra preprocessing.

KL and Prior-Constraint Sampling
--------------------------------

During RLFT, ``DiscretePriorPEFTRLFTAgent`` pins an anchor from the checkpoint-loaded
PEFT-wrapped prior at fit start. When ``kl_coeff > 0``, the KL term compares the
active adapter logits against that anchor on the same batch. With
``sampling_mode="prior_constraint"``, generation samples from the active adapter
while constraining support to the anchor prior's top-p nucleus.

Resume loads the latest checkpoint state first, then pins the anchor from that
loaded state. Warm-starting a new RLFT experiment from an SFT checkpoint pins the
SFT adapter; resuming an RLFT run pins the resumed RLFT adapter.

Common Commands
---------------

The examples use the packaged SOMA crouch motion and FSQ tracker:
``data/motion_for_trackers/crouch_soma23.pt`` and
``data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt``.
A packaged GPC prior is releasing soon. Until then, train the prior with the
first command and use that run's ``last.ckpt`` for SFT and RLFT.

Train the discrete GPC prior:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --experiment-path examples/experiments/gpc/prior.py \
       --tracker-checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt \
       --num-envs 1024 \
       --batch-size 1024 \
       --experiment-name prior_gpc_soma23

Bootstrap a target-reaching adapter with SFT:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --experiment-path examples/experiments/gpc/sft_target_prior_peft.py \
       --prior-checkpoint results/prior_gpc_soma23/last.ckpt \
       --tracker-checkpoint data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt \
       --num-envs 1024 \
       --batch-size 1024 \
       --training-max-steps 50000000 \
       --experiment-name sft_target_peft_crouch_soma

Run RLFT from the SFT checkpoint:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --experiment-path examples/experiments/gpc/task_target_prior_peft.py \
       --prior-checkpoint results/prior_gpc_soma23/last.ckpt \
       --checkpoint results/sft_target_peft_crouch_soma/last.ckpt \
       --num-envs 512 \
       --batch-size 512 \
       --experiment-name rlft_target_peft_crouch_soma

Use ``--peft-sampling-mode nucleus`` to sample from the student's nucleus and
regularize toward the prior with KL. The default ``prior_constraint`` mode uses
the frozen-prior nucleus as the rollout constraint.

Checkpoint Roles During Training
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Artifact
     - Used For
     - Notes
   * - Tracker checkpoint
     - Prior training and SFT target timing
     - Prior training embeds the tracker decoder and target encoder into the
       saved prior artifact. SFT reads the tracker config only to match the
       reference target lookahead timing.
   * - Prior checkpoint
     - Frozen base prior for SFT/RLFT
     - Use either a full ``last.ckpt`` or a prior ``inference_last.ckpt``. The
       PEFT agent loads the whole prior model, not an ``actor.mu`` submodule.
   * - SFT/RLFT ``last.ckpt``
     - Resume and warm-start
     - RLFT should warm-start from the SFT ``last.ckpt`` so optimizer/training
       state and the full PEFT model are available.
   * - SFT/RLFT ``inference_last.ckpt``
     - Inference and sharing
     - This is the small PEFT-only artifact. Do not use it to resume training.

For the packaged SOMA assets, the tracker path is
``data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt``.
The GPC prior is releasing soon; until then, use the ``last.ckpt`` produced by
the prior-training command above.

Inference
---------

The PEFT checkpoint contract keeps two artifacts:

* ``last.ckpt`` is the full training/resume checkpoint. It contains the full
  PEFT model state plus optimizer/training state. Use this when resuming SFT
  or RLFT.
* ``inference_last.ckpt`` is the slim shareable checkpoint. It contains only
  trainable PEFT/task state selected by ``actor.adapter_state_dict()``
  (``actor_peft_model.*``, PEFT adapter ``lora`` / ``gamma`` / ``beta`` /
  ``m`` entries, and PEFT conditioning-normalizer state). It does not duplicate
  the frozen base prior, critic, optimizer, or tracker decoder.

``DiscretePriorPEFTRLFTAgentConfig`` enables inference checkpoint saves by default, so
normal SFT/RLFT checkpoint writes produce both files. The size difference is
intentional: the full checkpoint is for training continuity, while the slim
checkpoint is the artifact to move between machines or use for deployment.

Run inference directly from the PEFT run's slim checkpoint:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --checkpoint results/rlft_target_peft_crouch_soma/inference_last.ckpt \
       --num-envs 16

At inference, the PEFT run's ``resolved_configs_inference.pt`` builds the PEFT
agent and points it at the frozen prior checkpoint. The slim PEFT checkpoint is
loaded onto that prior as adapter/task state. Updated prior checkpoints embed
their own ``latent_decoder`` config (originally sourced from the tracker at
prior-train time), so a separate tracker file is not required at PEFT inference
time.

If you move the PEFT run to another machine, keep the PEFT
``resolved_configs_inference.pt`` next to ``inference_last.ckpt`` and override
only the prior path:

.. code-block:: bash

   --overrides agent.pretrained_modules.prior.checkpoint_path=/path/to/prior/last.ckpt

The discrete-prior PEFT inference artifact is self-describing: ``--checkpoint``
should point at the PEFT run's ``inference_last.ckpt``.

Key Files
---------

* ``examples/experiments/gpc/prior.py`` - train the discrete latent prior.
* ``examples/experiments/gpc/sft_target_prior_peft.py`` - supervised PEFT
  bootstrap for target reaching.
* ``examples/experiments/gpc/task_target_prior_peft.py`` - PPO RLFT target
  reaching.
* ``examples/experiments/gpc/task_target_prior_peft_amp.py`` - RLFT with AMP
  rewards.
* ``protomotions/agents/supervised/latent_prior_model.py`` - frozen tracker
  decoder plus trainable autoregressive prior.
* ``protomotions/agents/peft/`` - discrete-prior PEFT actor, agent, adapter, and AMP
  variants.
