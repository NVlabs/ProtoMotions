Training and Evaluation Scripts
================================

These are the main entry point scripts for training and evaluating agents.
They are documented by hand here because the training and inference entry
points are thin shims: their option tables live in
``protomotions/launch/training_cli.py`` and ``protomotions/launch/inference.py``,
and the SLURM launcher declares only its own cluster flags.

train_agent.py
--------------

**Main training script for ProtoMotions agents.**

Handles configuration loading, distributed training setup, agent initialization, 
and checkpoint management. Builds configs from Python experiment files.

**Usage:**

.. code-block:: bash

   python protomotions/train_agent.py \
       --experiment-path examples/experiments/steering/mlp.py \
       --robot-name h1_2 \
       --simulator isaacgym \
       --motion-file data/motion_for_trackers/h1_2_random_subset_tiny.pt \
       --num-envs 4096 \
       --batch-size 16384 \
       --experiment-name h1_2_steering

Every flag in the example is required. ``--robot-name`` takes the stem of a
file under ``protomotions/robot_configs/`` (for example ``g1``, ``h1_2``,
``smpl`` or ``soma23``).

**Key Features:**

* Automatic configuration saving for reproducibility
* Distributed training with Lightning Fabric
* Checkpoint auto-resume
* Weights & Biases integration
* Multi-simulator support (IsaacGym, IsaacLab, Newton, MuJoCo)

**Configuration System:**

All configurations are saved to ``results/<experiment_name>/``:

* ``config.yaml`` - CLI arguments
* ``resolved_configs.json`` - The config bundle, a tagged JSON tree (**source of truth**)
* ``experiment_config.py`` - Copy of experiment file
* ``last.ckpt`` - Model checkpoint

inference_agent.py
------------------

**Evaluation and visualization script for trained agents.**

Loads trained checkpoints and runs agents in the simulation environment for
evaluation, visualization, and analysis. Supports interactive controls and
video recording when the selected simulator backend provides a viewer.

**Usage:**

.. code-block:: bash

   python protomotions/inference_agent.py \
       --simulator isaacgym \
       --checkpoint results/h1_2_steering/last.ckpt

**Motion Playback:**

For kinematic motion playback without physics training state, ``examples/env_kinematic_playback.py``
builds the environment of the experiment you pass and drops the agent. The
motion file must belong to the robot; this one is the SOMA-23 subset:

.. code-block:: bash

   python examples/env_kinematic_playback.py \
       --experiment-path examples/experiments/mimic/mlp_no_contact.py \
       --robot-name soma23 \
       --simulator isaacgym \
       --num-envs 16 \
       --motion-file data/motion_for_trackers/soma23_bones_seed_mini.pt

The example uses the ``mlp_no_contact`` mimic recipe because IsaacGym's GPU
pipeline reports one combined force per body, so it cannot run a recipe that
declares contact bodies. On IsaacLab, ``examples/experiments/mimic/mlp.py``
works as it is.

**Keyboard Controls:**

Click inside the simulation viewer to give it keyboard focus, then use the
task's key bindings.

* **W/A/S/D** - Move keyboard-controlled task targets when enabled
* **R** - Reset environments
* **Q** - Quit

**MuJoCo only:** press **.** to toggle between ProtoMotions key bindings
(default) and MuJoCo's native viewer key bindings. Only the selected set is
active. Switching back to ProtoMotions key bindings restores the initial
visualization settings. Mouse orbit and zoom work in either mode.

train_slurm.py
--------------

**SLURM cluster training script.**

Wrapper around train_agent.py for distributed training on SLURM-based HPC clusters.
Handles job submission, node coordination, and auto-resume on preemption.

**Usage:**

.. code-block:: bash

   python protomotions/train_slurm.py \
       --user <cluster_username> \
       --experiment-path examples/experiments/steering/mlp.py \
       --robot-name h1_2 \
       --simulator isaacgym \
       --motion-file data/motion_for_trackers/h1_2_random_subset_tiny.pt \
       --num-envs 4096 \
       --batch-size 16384 \
       --nodes 2 \
       --ngpu 4 \
       --experiment-name slurm_experiment

``--user`` is the launcher's own flag and is required. Everything else is the
training command line, forwarded to ``train_agent.py`` (with ``--use-slurm``
prepended) and checked against its parser before anything is submitted, so the
same required flags apply here.

**Features:**

* Multi-node distributed training
* Auto-resume on preemption
* SLURM job management
* Log aggregation

See Also
--------

* :doc:`../getting_started/quickstart` - Getting started guide
* :doc:`../user_guide/configuration` - Configuration system
