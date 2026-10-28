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

``examples/experiments/gpc/target_prior_peft.py`` fine-tunes the adapter
with PPO on task rewards. The actor config stays compatible with the SFT config,
so an SFT checkpoint can warm-start RLFT. The environment usually swaps from the
SFT mimic-target source to a task control source such as random target reaching.

SFT is optional. RLFT can start straight from the trained prior: leave out
``--checkpoint`` and the adapters start freshly initialized. Their output weights
start near zero, so the first rollouts behave almost exactly like the prior.

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
``actor.peft.model`` may read those context keys but must not write them: the frozen
prior, and the RLFT reference policy, read them straight from the observations, and the
actor refuses a model whose ``out_keys`` include one.

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

During RLFT, ``DiscretePriorPEFTRLFTAgent`` compares the policy it is training
(the student) against a frozen anchor, the reference policy. The reference is a
frozen copy of the policy training starts from: its adapters, its FiLM input
normalizer and its conditioning network (``actor.peft.model``). When
``kl_coeff > 0``, the KL term compares the student's logits against the
reference's on the same batch. With ``sampling_mode="prior_constraint"``,
generation samples from the student while limiting each token to the
reference's top-p nucleus.

The reference is pinned explicitly at training start, after checkpoint
loading. A new RLFT run that needs one, fresh or warm-started from any
checkpoint, pins the policy it just loaded (including SFT adapters), in either
sampling mode, and ignores any reference stored in that checkpoint.
``--overrides agent.model.actor.peft.reference_checkpoint=<file>`` pins the
trained policy saved in that full or adapter checkpoint instead. A resume
keeps the reference saved in its checkpoint. ``clear_peft`` clears the student
only right after a new run pins its reference; it never applies on resume or
during inference. Setup materializes shapes without sampling, and sampling
never creates a reference implicitly.

With ``sampling_mode="nucleus"`` and ``kl_coeff=0`` nothing reads a reference,
so the run holds none and drops a loaded one, and a new run skips a configured
``clear_peft`` with a warning rather than discarding the loaded policy. The
student's FiLM input normalizer and the conditioning network's own observation
normalizer keep updating during training. The frozen prior's normalizers are
frozen at setup, and only the reference's own copies are frozen. Because the
reference reads its condition from its own frozen copy of the conditioning
network, its logits do not drift while the student trains.

RLFT files saved before the reference kept that copy store the reference
without ``reference_peft_model.*`` keys. Loading one is refused with a message
naming ``scripts/migrate_peft_reference_conditioning.py``; the script copies the
student conditioning network saved in the same file, the network that reference
read when the file was saved, under the reference prefix and writes a new file.
Load that file instead; nothing fills the copy in on load.

Checkpoint loading ignores saved critic, discriminator and discriminator-critic
weights only when the destination has no corresponding network. AMP permits
new discriminator networks to keep their initialization, and PEFT permits a
new value critic when starting RLFT from SFT. When a network is present on both
sides, partial or incompatible weights fail validation. Actor weights are
never optional. Student and reference keys and shapes are checked before any
checkpoint weights are copied.

``last.ckpt`` is the full training checkpoint, including optimizer state.
``inference_last.ckpt`` is the adapter checkpoint: the student's adapter and
task tensors and, when the run holds a reference, the reference's adapter
tensors, FiLM input normalizer and conditioning network. Frozen base weights
come from the configured prior. Both files restore the same policy and
reference.

Files saved before the reference kept its own conditioning network are
refused until ``scripts/migrate_peft_reference_conditioning.py`` has written
a migrated copy (see above). The compatibility only goes one way: older
ProtoMotions code cannot load a newer file that holds a reference.

Missing-reference behavior depends on the operation, not the checkpoint format:

* Ordinary nucleus inference does not need a reference and loads silently.
* Prior-constrained RLFT inference fails if the checkpoint has no saved
  reference. The error suggests loading the run's ``last.ckpt`` or overriding
  ``agent.model.actor.peft.sampling_mode=nucleus``.
* SFT inference never needs a reference, because the SFT model replays its
  target encoder's tokens instead of sampling, so it loads either SFT file.
* Resume fails without a saved reference when KL or constrained sampling needs it.
* A new training run never uses a saved reference, so it can warm-start from a
  file with or without one; a file whose reference lacks its conditioning
  network is still migrated first, since the loaders refuse it on every path.
  At training start it pins the policy it loaded, or the trained policy in
  ``agent.model.actor.peft.reference_checkpoint`` if that is set. An adapter
  checkpoint is not a full optimizer resume.

The inference entry point passes ``for_inference=True`` to ``agent.load`` so
missing-reference requirements are checked before weights change. Programmatic
inference should do the same, with ``load_training_state=False``. Warm starts
use ``load_training_state=False`` without the inference flag; ``fit()`` prepares
the reference. An older RLFT adapter file missing its reference cannot reproduce
constrained sampling: use a checkpoint that actually contains that reference.

Common Commands
---------------

The examples use the packaged SOMA crouch motion and FSQ tracker:
``data/motion_for_trackers/crouch_soma23.pt`` and
``data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt``.
A packaged GPC prior is available at
``data/pretrained_models/gpc_prior/soma_bones/inference_last.ckpt``. Use it for
SFT and RLFT, or train a prior with the first command below.

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
       --experiment-path examples/experiments/gpc/target_prior_peft.py \
       --prior-checkpoint results/prior_gpc_soma23/last.ckpt \
       --checkpoint results/sft_target_peft_crouch_soma/last.ckpt \
       --num-envs 512 \
       --batch-size 512 \
       --experiment-name rlft_target_peft_crouch_soma

Or skip SFT and run RLFT straight from the prior. The command is the same
without ``--checkpoint``:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --experiment-path examples/experiments/gpc/target_prior_peft.py \
       --prior-checkpoint results/prior_gpc_soma23/last.ckpt \
       --num-envs 512 \
       --batch-size 512 \
       --experiment-name rlft_target_peft_from_prior_crouch_soma

Use ``--peft-sampling-mode nucleus`` to sample from the student's nucleus and
regularize it toward the reference policy with KL. The default
``prior_constraint`` mode instead limits each sampled token to the reference
policy's top-p set. Either way the reference is the policy a new run starts
from: the loaded SFT policy when RLFT starts from SFT, the loaded RLFT policy
when it warm-starts from another RLFT run, or the frozen prior with its freshly
initialized adapters when it starts from the prior. A resume keeps the
reference the run started with.

To anchor a new run to a different policy, point
``agent.model.actor.peft.reference_checkpoint`` at the checkpoint of the run
that trained it. Either file works, but the adapter file is much smaller to
load. For example, this command starts a new run from the RLFT run above and
keeps training its policy, while keeping the SFT policy as the reference. It
is the RLFT command with a different ``--checkpoint``, a new experiment name
and the override:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --experiment-path examples/experiments/gpc/target_prior_peft.py \
       --prior-checkpoint results/prior_gpc_soma23/last.ckpt \
       --checkpoint results/rlft_target_peft_crouch_soma/last.ckpt \
       --num-envs 512 \
       --batch-size 512 \
       --experiment-name rlft_target_peft_crouch_soma_sft_anchor \
       --overrides agent.model.actor.peft.reference_checkpoint=results/sft_target_peft_crouch_soma/inference_last.ckpt

Warm-starting from a run's own ``last.ckpt`` under a new experiment name is a
new run, so it re-anchors to the trained policy. To keep the original anchor
when you move a run, resume it instead: copy the whole
``results/<experiment-name>/`` directory and rerun with the same experiment
name. Or set ``reference_checkpoint`` to the file the run originally started
from.

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
     - A resume restores the weights, optimizer and training state. When RLFT
       follows SFT, warm-start it from the SFT ``last.ckpt``; a warm start
       loads the full PEFT model's weights only, and the optimizer and
       training state start fresh. RLFT straight from the prior needs no SFT
       checkpoint.
   * - SFT/RLFT ``inference_last.ckpt``
     - Inference and sharing
     - This is the small PEFT-only artifact. Do not use it to resume training.

For the packaged SOMA assets, the corresponding paths are
``data/pretrained_models/motion_tracker/soma_bones_fsq/inference_last.ckpt`` and
``data/pretrained_models/gpc_prior/soma_bones/inference_last.ckpt``. Both the
tracker and prior provide inference-oriented checkpoints; use their full
``last.ckpt`` artifacts only for training or resume.

Inference
---------

The PEFT checkpoint contract keeps two artifacts:

* ``last.ckpt`` is the full training/resume checkpoint. It contains the full
  PEFT model state plus optimizer/training state. Use this when resuming SFT
  or RLFT.
* ``inference_last.ckpt`` is the small, shareable adapter checkpoint. It contains
  only trainable PEFT/task state selected by ``actor.adapter_state_dict()``:
  the conditioning network (``actor_peft_model.*``), PEFT adapter ``lora`` /
  ``gamma`` / ``beta`` / ``m`` entries, and the FiLM input normalizer
  (``prior_with_peft.film_input_norm.*``). When the run holds a reference
  policy, it also stores the reference's own adapter tensors
  (``prior_with_peft.reference_prior.*``), FiLM input normalizer
  (``prior_with_peft.reference_film_input_norm.*``) and conditioning network
  (``reference_peft_model.*``), selected by ``actor.reference_state_dict()``,
  so loading it gives the same reference as loading ``last.ckpt``. This
  roughly doubles its size. RLFT runs pin a reference at training start when
  they sample with ``prior_constraint`` or set ``kl_coeff > 0``. RLFT runs with
  nucleus sampling and no KL penalty never hold one, and neither do SFT runs,
  which drop any reference they load. The file never duplicates the frozen
  base prior, critic, optimizer, or tracker decoder.

``DiscretePriorPEFTRLFTAgentConfig`` enables inference checkpoint saves by default, so
normal SFT/RLFT checkpoint writes produce both files. The size difference is
intentional: the full checkpoint is for training continuity, while the adapter
checkpoint is the artifact to move between machines or use for deployment.

Run inference directly from the PEFT run's adapter checkpoint:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --robot-name soma23 \
       --simulator isaaclab \
       --motion-file data/motion_for_trackers/crouch_soma23.pt \
       --checkpoint results/rlft_target_peft_crouch_soma/inference_last.ckpt \
       --num-envs 16

At inference, the PEFT run's ``resolved_configs.json`` builds the PEFT
agent and points it at the frozen prior checkpoint. The adapter PEFT checkpoint is
loaded onto that prior as adapter/task state. Updated prior checkpoints embed
their own ``latent_decoder`` config (originally sourced from the tracker at
prior-train time), so a separate tracker file is not required at PEFT inference
time.

If you move the PEFT run to another machine, keep the PEFT
``resolved_configs.json`` and ``experiment_config.py`` next to
``inference_last.ckpt`` and override only the prior path:

.. code-block:: bash

   --overrides agent.pretrained_modules.prior.checkpoint_path=data/pretrained_models/gpc_prior/soma_bones/inference_last.ckpt

The discrete-prior PEFT inference artifact is self-describing: ``--checkpoint``
should point at the PEFT run's ``inference_last.ckpt``.

Key Files
---------

* ``examples/experiments/gpc/prior.py`` - train the discrete latent prior.
* ``examples/experiments/gpc/sft_target_prior_peft.py`` - supervised PEFT
  bootstrap for target reaching.
* ``examples/experiments/gpc/target_prior_peft.py`` - PPO RLFT target
  reaching.
* ``examples/experiments/gpc/task_target_prior_peft_amp.py`` - RLFT with AMP
  rewards.
* ``protomotions/agents/supervised/latent_prior_model.py`` - frozen tracker
  decoder plus trainable autoregressive prior.
* ``protomotions/agents/peft/`` - discrete-prior PEFT actor, agent, adapter, and AMP
  variants.
