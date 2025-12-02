Training and Evaluation Scripts
================================

These are the main entry point scripts for training and evaluating agents.
Since these scripts execute code at module level, they are documented manually here.

train_agent.py
--------------

**Main training script for ProtoMotions agents.**

Handles configuration loading, distributed training setup, agent initialization, 
and checkpoint management. Supports Hydra-based configuration composition.

**Usage:**

.. code-block:: bash

   python protomotions/train_agent.py \\
       --experiment-path examples/experiments/steering_mlp.py \\
       --robot-name h1 \\
       --simulator isaacgym \\
       --experiment-name h1_steering

**Key Features:**

* Automatic configuration saving for reproducibility
* Distributed training with Lightning Fabric
* Checkpoint auto-resume
* Weights & Biases integration
* Multi-simulator support (IsaacGym, IsaacLab, Genesis)

**Configuration System:**

All configurations are saved to ``results/<experiment_name>/``:

* ``config.yaml`` - CLI arguments
* ``resolved_configs.pt`` - Full config objects (pickled, **source of truth**)
* ``resolved_configs.yaml`` - Human-readable configs (**do not modify**)
* ``experiment_config.py`` - Copy of experiment file
* ``last.ckpt`` - Model checkpoint

eval_agent.py
-------------

**Evaluation and visualization script for trained agents.**

Loads trained checkpoints and runs agents in the simulation environment for
evaluation, visualization, and analysis. Supports interactive controls and
video recording.

**Usage:**

.. code-block:: bash

   python protomotions/eval_agent.py \\
       --robot-name h1 \\
       --simulator isaacgym \\
       --checkpoint results/h1_steering/last.ckpt

**Motion Playback:**

For kinematic motion playback without physics:

.. code-block:: bash

   python protomotions/eval_agent.py \\
       --config-name play_motion \\
       --robot-name smpl \\
       --simulator isaacgym \\
       +--motion-file data/motions/walk.motion

**Keyboard Controls:**

* **J** - Apply random forces (robustness test)
* **R** - Reset environments
* **O** - Toggle camera view
* **L** - Start/stop video recording
* **Q** - Quit

train_slurm.py
--------------

**SLURM cluster training script.**

Wrapper around train_agent.py for distributed training on SLURM-based HPC clusters.
Handles job submission, node coordination, and auto-resume on preemption.

**Usage:**

.. code-block:: bash

   python protomotions/train_slurm.py \\
       --experiment-path examples/experiments/steering_mlp.py \\
       --robot-name h1 \\
       --simulator isaacgym \\
       --nodes 2 \\
       --ngpu 4 \\
       --experiment-name slurm_experiment

**Features:**

* Multi-node distributed training
* Auto-resume on preemption
* SLURM job management
* Log aggregation

See Also
--------

* :doc:`../getting_started/quickstart` - Getting started guide
* :doc:`../user_guide/configuration` - Configuration system

