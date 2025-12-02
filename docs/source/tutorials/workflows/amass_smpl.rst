SMPL Training on AMASS
======================

This workflow covers training an SMPL humanoid to imitate motions from the AMASS dataset.

Prerequisites
-------------

* AMASS data converted to ProtoMotions format (see :doc:`../../getting_started/amass_preparation`)
* Packaged MotionLib ``.pt`` file

Training
--------

Basic Training Command
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name smpl_amass_flat \
       --motion-file /path/to/amass_train.pt \
       --num-envs 8192 \
       --batch-size 8192 \
       --ngpu 4

This trains an MLP policy on flat terrain.

Training on Complex Terrain
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For robust locomotion on uneven terrain:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp_complex_terrain.py \
       --experiment-name smpl_amass_terrain \
       --motion-file /path/to/amass_train.pt \
       --num-envs 8192 \
       --batch-size 8192 \
       --ngpu 4


Expected Training Time
~~~~~~~~~~~~~~~~~~~~~~

On 4x A100 GPUs with full AMASS (40+ hours of motion):
~2 hours to 90% success rate.
~12 hours to 99% success rate, more training can improve success rate and rewards further.

Key Metrics to Monitor
----------------------

With ``--use-wandb``, track these metrics:

* **Eval/gt_err**: Position tracking error (lower is better). This is unbiased - 
  evaluates all motions equally.
* **Eval/success_rate**: Fraction of motions completed without falling.
* **Train/episode_reward**: Training reward. May fluctuate due to prioritized 
  sampling focusing on harder motions.
* **Train/clip_frac**: Fraction of policy updates clipped by PPO. Keep this under 
  ~0.3 for stable training. If consistently higher, consider lowering the learning rate.
* **Train/actor_grad_norm** and **Train/critic_grad_norm**: Monitor these to ensure 
  gradients are not exploding. Sudden spikes may indicate issues with your changes 
  or reward configuration.

.. note::

   If ``Train/episode_reward`` drops, it can mean that the evaluator 
   re-weighted motions (ref. mimic_evaluator.py) and training is now focusing on harder cases. Check 
   ``Eval/gt_err`` for unbiased metric, where each motion is evaluated fully once.

.. tip::

   Weights & Biases has many useful features beyond basic metric plots. You can search 
   and filter runs by any config parameter, compare runs side-by-side, and create custom 
   dashboards. Spend some time exploring the UI to get the most out of experiment tracking.

Experiment Configurations
-------------------------

The ``mlp.py`` experiment defines:

**Environment Config:**

* 1000 step episodes
* Early termination on large tracking error (>0.5 rad max joint error)
* Bootstrap on episode end for value estimation

**Reward Components:**

* ``gt_rew``: Global body position tracking
* ``gr_rew``: Global body rotation tracking
* ``gv_rew``, ``gav_rew``: Velocity tracking
* ``rh_rew``: Root height tracking
* ``pow_rew``: Power consumption penalty
* ``contact_match_rew``: Foot contact matching
* ``action_smoothness``: Action smoothness penalty

**Network:**

* 6-layer MLP with 1024 units
* Separate actor and critic networks
* Running mean/std observation normalization

Customizing Training Examples
-----------------------------

Adjust Mini-Epochs
~~~~~~~~~~~~~~~~~~

More mini-epochs can improve sample efficiency:

.. code-block:: bash

   --overrides "agent.num_mini_epochs=4"

Disable Contact Rewards
~~~~~~~~~~~~~~~~~~~~~~~

For purely motion imitation (DeepMimic) rewards:

.. code-block:: bash

   --overrides "env.reward_config.contact_match_rew.weight=0.0" \
               "env.reward_config.contact_force_change_rew.weight=0.0"

Visualizing Motions
-------------------

Before training or for debugging, you can visualize the packaged MotionLib using 
the motion visualizer:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/amass_train.pt \
       --robot smpl \
       --simulator isaacgym

The visualizer supports comparing multiple MotionLibs side-by-side, which is useful 
for comparing source motions with retargeted or predicted motions:

.. code-block:: bash

   python examples/motion_libs_visualizer.py \
       --motion_files /path/to/amass_train.pt /path/to/predicted_motions.pt \
       --robot smpl \
       --simulator isaacgym

**Controls:**

* **R**: Switch to next motion
* **1/2**: Increase/decrease playback speed  
* **3/4**: Adjust smoothness threshold for jitter highlighting

Evaluation
----------

Run inference on trained model:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/smpl_amass_flat/last.ckpt \
       --simulator isaacgym

Full evaluation over all motions:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/smpl_amass_flat/last.ckpt \
       --simulator isaacgym \
       --num-envs 1024 \
       --full-eval

This assigns motion 0 to env 0, motion 1 to env 1, etc., and reports aggregate metrics.

.. note::

   For full-eval, set ``--num-envs`` to a large value (e.g., 1024 or more) to evaluate many motions 
   in parallel. The default is 1, which would make full evaluation very slow. --headless might be needed to save memory.

Next Steps
----------

* :doc:`retargeting_pyroki` - Retarget these motions to robots like G1
* :doc:`domain_randomization` - Add domain randomization for sim2sim
* :doc:`../../concepts/abstractions` - Understand the underlying architecture

