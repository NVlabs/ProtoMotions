Metatron G1 Student
=======================

``examples/experiments/metatron/g1_student.py`` is one explicit
supervised recipe, not a collection of research variants. Train a five-body G1
tracker with ``examples/experiments/metatron/g1_teacher.py`` first.
The student requires its checkpoint and resolved configs:

.. code-block:: bash

   python protomotions/train_agent.py --robot-name g1 --simulator isaaclab \
     --experiment-path examples/experiments/metatron/g1_student.py \
     --expert-model-path results/tracker/last.ckpt --motion-file data/motions.pt \
     --num-envs 2048 --batch-size 4096 --experiment-name metatron_g1

Inputs and Architecture
-----------------------

The prior sees sensor-derived FK and reduced state, the last raw action,
previous-only history at lags 8/4/3/2/1, and global, local, unordered style, and
velocity goals. Each modality uses a shared per-frame MLP; the style stream is
masked-mean pooled. A six-layer, four-head, 1024-wide Transformer produces the
prior mean and log variance.

The privileged encoder sees maximal state and full future reference poses.
Its mean is added to the prior mean to form the posterior mean; its log
variance defines the posterior variance. The decoder combines a
64-dimensional spherical latent with current sensor-derived state and the
last raw action. Prior features are not extra decoder inputs.

Goal inputs use the standard ``ObsProcessor`` and ``MLPWithConcat`` modules.
The observation kernels already zero hidden body/property values. Each input
is reshaped to ``[batch, frames, features]``; pose and style-velocity values are
normalized, while mask bits and time-to-go stay unnormalized. A shared MLP
encodes each frame. There is no separate goal-encoder class or second copy of
the observation mask layout inside the model.

All frames of a modality share normalization statistics and MLP weights.
Hidden entries contribute zeros to the statistics. Normalization and MLP
biases can produce nonzero embeddings for absent frames; the existing
``MaskedMean`` and Transformer frame masks exclude these from pooling and
attention. Permuting style frames therefore leaves the pooled style token
unchanged. This replaces the earlier frame-specific global/local statistics
and custom parameter layout; use fresh configs and training, not old checkpoints.

Training Contract
-----------------

Student and teacher joint/body ordering must agree exactly. Action mapping,
controller gains, control limits, simulator timing, reset noise, and domain
randomization come from the teacher configs. The teacher remains privileged:
its simulator-truth observations do not become noisy student observations.
Only observations needed by the teacher actor are added to the student env.

Ten percent of episodes use teacher actions; the others use posterior student
actions. Teacher action labels are collected in both cases. Adam uses a
2e-5 learning rate, two mini-epochs, and gradient clipping at 50. KL ramps from
1e-4 to 1e-2 between epochs 500 and 2000, then repeats cyclically; the
prior-mean unit-norm penalty is 0.001. Adjacent-action and solution-feature
losses have weights 0.05 and 0.001. Temporal pairs exclude resets and rollout
edges before minibatch shuffling.

Inference
---------

.. code-block:: bash

   python protomotions/inference_agent.py --simulator mujoco \
     --checkpoint results/metatron_g1/last.ckpt --motion-file data/motions.pt \
     --num-envs 1

The derived inference view removes the teacher dependency and privileged
observations. Setup initializes the prior path; checkpoint loading supplies
the unused posterior weights without requiring posterior observations.
Reference-derived prompts remain the default. CPU tests cover architecture,
materialization, gradients, checkpoint loading, and mask isolation; they do
not establish learned locomotion quality.
