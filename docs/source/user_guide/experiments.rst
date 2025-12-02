Experiments
===========

ProtoMotions implements several state-of-the-art algorithms for physics-based character animation.
This page provides a high-level overview of each approach and when to use them.

Mimic
-----

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/smpl_mimic.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**Paper**: `DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills <https://arxiv.org/abs/1804.02717>`__ (Peng et al., 2018)

Mimic trains agents to imitate reference motion clips through reinforcement learning.
The agent receives rewards for matching the reference pose at each timestep, learning to
reproduce the motion while maintaining physical plausibility.

**Experiment Variants**:

* ``mlp.py`` - MLP policy for flat terrain
* ``mlp_complex_terrain.py`` - MLP policy for complex terrains (stairs, slopes, etc.)
* ``mlp_domain_rand.py`` - Reduced coordinate observations with domain randomization for sim2real
* ``transformer.py``, ``transformer_complex_terrain.py`` - Transformer variants with more future observation frames

**Example command**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/mimic/mlp.py \
       --robot-name smpl \
       --simulator isaacgym \
       --motion-file <path_to_motion_file> \
       --experiment-name smpl_mimic

ADD (Adversarial Differential Discriminators)
---------------------------------------------

**Paper**: `ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators <https://add-moo.github.io/>`__

ADD is an adversarial approach to motion tracking that automatically balances multiple
tracking objectives without manual reward weight tuning. It uses a discriminator to learn
how to combine tracking errors dynamically.

**Example command**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/add/mlp.py \
       --robot-name smpl \
       --simulator isaacgym \
       --motion-file <path_to_motion_file> \
       --experiment-name smpl_add

MaskedMimic
-----------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/smpl_masked_mimic.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**Paper**: `MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting <https://research.nvidia.com/labs/par/maskedmimic/>`__ (Tessler et al., SIGGRAPH Asia 2024)

MaskedMimic formulates character control as a motion inpainting problem. A single unified
controller learns to synthesize full-body motions from partial observations (masked keyframes,
text descriptions, or scene information).

.. note::

   MaskedMimic requires a pre-trained Mimic expert model. First, train Mimic on multiple motions, then provide
   the checkpoint path via ``--overrides``.

**Example command**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/masked_mimic/transformer.py \
       --robot-name smpl \
       --simulator isaacgym \
       --motion-file <path_to_motions> \
       --overrides "agent.config.expert_model_path='<path_to_expert_model>/last.ckpt'" \
       --experiment-name smpl_masked_mimic

AMP (Adversarial Motion Priors)
-------------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/g1_amp.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**Paper**: `AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control <https://arxiv.org/abs/2104.02180>`__ (Peng et al., 2021)

AMP uses a discriminator to learn a motion prior from reference data. Instead of tracking
specific poses, the agent learns to move in a style similar to the reference motions while
accomplishing task objectives.

**Example command**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/amp/mlp.py \
       --robot-name g1 \
       --simulator isaacgym \
       --motion-file <path_to_motion> \
       --experiment-name g1_amp

Steering
--------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/g1_steering.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

Steering combines AMP-style motion priors with directional control objectives. The agent
learns to walk/run in user-specified directions while maintaining natural motion style.

**Example command**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/steering/mlp.py \
       --robot-name g1 \
       --simulator isaacgym \
       --motion-file <path_to_motion> \
       --experiment-name g1_steering

ASE (Adversarial Skill Embeddings)
----------------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/smpl_ase.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**Paper**: `ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters <https://arxiv.org/abs/2205.01906>`__ (Peng et al., 2022)

ASE extends AMP by learning a latent skill space. The policy is conditioned on latent codes,
allowing a single controller to perform many different skills by varying the latent input.

.. note::

   ASE requires a **diverse motion dataset** with many different motion types. It does not
   work with a single motion clip - the skill embedding relies on having varied behaviors
   to learn meaningful latent codes.

**Example command**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/ase/mlp.py \
       --robot-name smpl \
       --simulator isaacgym \
       --motion-file <path_to_motions> \
       --experiment-name smpl_ase

See Also
--------

* :doc:`configuration` - Configuration system
* :doc:`../tutorials/code_tutorials` - Step-by-step tutorials
* :doc:`../getting_started/quickstart` - Quick start guide


