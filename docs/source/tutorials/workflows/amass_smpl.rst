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
       --experiment-path examples/experiments/mimic/mlp_no_contact.py \
       --experiment-name smpl_amass_flat \
       --motion-file /path/to/amass_train.pt \
       --num-envs 8192 \
       --batch-size 8192 \
       --ngpu 4

This trains an MLP policy on flat terrain. The command uses
``mimic/mlp_no_contact.py`` rather than ``mimic/mlp.py`` because IsaacGym's GPU
pipeline reports one combined force per body, so it cannot run a recipe that
declares contact bodies. The two recipes are otherwise identical (see
`Disable Contact Rewards`_ below). On IsaacLab, run ``mimic/mlp.py`` with
``--simulator isaaclab`` to keep the foot-contact reward.

Training on Complex Terrain
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For robust locomotion on uneven terrain:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp_complex_terrain_no_contact.py \
       --experiment-name smpl_amass_terrain \
       --motion-file /path/to/amass_train.pt \
       --num-envs 8192 \
       --batch-size 8192 \
       --ngpu 4

As above, the ``_no_contact`` recipe is the one IsaacGym can run;
``mimic/mlp_complex_terrain.py`` is the same recipe with the foot-contact
reward, for IsaacLab.

Expected Training Time
~~~~~~~~~~~~~~~~~~~~~~

On 4x A100 GPUs with full AMASS (40+ hours of motion):
~2 hours to 90% success rate.
~12 hours to 99% success rate, more training can improve success rate and rewards further.

Key Metrics to Monitor
----------------------

With ``--use-wandb``, track these metrics:

* ``eval/gt_error/mean``: Position tracking error (lower is better).
  The evaluator averages each evaluated trial's scored frames, then averages
  those trial means equally.
* ``eval/success_rate``: Fraction of evaluated trials passing the configured
  failure rules. The mimic MLP recipe uses a position-error threshold.
* ``info/episode_reward``: Training reward, affected by prioritized sampling.
* ``actor/clip_frac``: Fraction of samples whose PPO probability ratio is clipped.
* ``actor/grad_norm_before_clip`` and ``critic/grad_norm_before_clip``: Gradient
  magnitudes before clipping; sudden spikes can help diagnose unstable updates.

.. note::

   Evaluation metrics cover only scored frames. Trials may stop scoring after
   an any-frame failure; a lower tracking error alone does not imply a better
   policy. Compare it with ``eval/success_rate`` and ``eval/num_evaluated``.

.. tip::

   Weights & Biases has many useful features beyond basic metric plots. You can search 
   and filter runs by any config parameter, compare runs side-by-side, and create custom 
   dashboards. Spend some time exploring the UI to get the most out of experiment tracking.

Experiment Configurations
-------------------------

The ``mlp.py`` experiment defines the following (``mlp_no_contact.py`` is the
same recipe without the ``contact_match_rew`` reward and with no contact bodies
declared on the robot):

**Environment Config:**

* 1000 step episodes
* Early termination when any body is more than 0.5 m from its reference position
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

For purely motion imitation (DeepMimic) rewards, run the recipe without the
foot-contact reward. ``examples/experiments/mimic/mlp_no_contact.py`` is
``mlp.py`` with the ``contact_match_rew`` entry deleted from
``reward_components`` and no contact bodies declared on the robot;
``mlp_complex_terrain_no_contact.py`` does the same for the terrain recipe. To
do this to a recipe of your own, copy it and delete the entry. That is the
supported route.

A reward cannot be switched off from the command line. A weight of zero is
refused (``RewardMeta.weight must be a finite nonzero float; delete the
component instead of weighting it 0``), and the weight is not a
``static_params`` key, so ``...static_params.weight=0.0`` is refused as an
unknown path before the environment is built. What ``--overrides`` can do is
re-weight a reward through its ``meta`` block with a nonzero value. This
example raises the body-position tracking weight from 0.5 to 0.6 and works
with every recipe on this page:

.. code-block:: bash

   --overrides "env.reward_components.gt_rew.meta.weight=0.6"

The path must name a reward the recipe actually has. ``contact_match_rew``
exists only in ``mlp.py`` and ``mlp_complex_terrain.py``, so re-weighting it
under a ``_no_contact`` recipe is refused as an unknown key.

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
