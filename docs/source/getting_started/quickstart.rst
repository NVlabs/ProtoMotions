Quick Start
===========

This guide helps you run pre-trained models and start training your own agents.

Run Pre-trained Models
----------------------

We provide pre-trained checkpoints for various robots and tasks. Download them and run inference to see the results.

**Available Pre-trained Models:**

The first four models below are **General Motion Trackers** - DeepMimic-style policies capable of tracking a wide variety of human motions, trained on large motion datasets (AMASS or BONES-SEED).

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Model
     - Description
     - Checkpoint Path
   * - SMPL AMASS (flat)
     - General motion tracker: SMPL humanoid on flat terrain
     - ``data/pretrained_models/motion_tracker/smpl/last.ckpt``
   * - SMPL AMASS (terrain)
     - General motion tracker: SMPL humanoid on complex terrain
     - ``data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt``
   * - G1 BONES-SEED
     - General motion tracker: Unitree G1 on BONES-SEED retargeted motions
     - ``data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt``
   * - SOMA BONES-SEED
     - General motion tracker: SOMA 23-body humanoid on BONES-SEED motions
     - ``data/pretrained_models/motion_tracker/soma-bones/last.ckpt``
   * - Vaulting
     - DeepMimic policy for a vaulting motion
     - *Coming soon*
   * - MaskedMimic SMPL
     - MaskedMimic policy for SMPL
     - ``data/pretrained_models/masked_mimic/smpl/last.ckpt``
   * - MaskedMimic G1
     - MaskedMimic policy for G1 trained on AMASS
     - ``data/pretrained_models/masked_mimic/g1/last.ckpt``

**Example Motion Data:**

We provide small example motion files for testing with robot models:

* ``data/motion_for_trackers/g1_random_subset_tiny.pt`` - Small subset of retargeted AMASS for G1
* ``data/motion_for_trackers/g1_bones_seed_mini.pt`` - Small subset of BONES-SEED retargeted motions for G1
* ``data/motion_for_trackers/soma23_bones_seed_mini.pt`` - Small subset of BONES-SEED motions for SOMA 23-body humanoid
* ``data/motion_for_trackers/h1_2_random_subset_tiny.pt`` - Small subset of retargeted AMASS for H1-2

For SMPL motion data, see :doc:`amass_preparation` to generate your own MotionLib from AMASS.
There is a simple script ``scripts/subset_motion_lib.py`` to subset the motion lib into a smaller size,
if your local GPU memory is not enough to load the entire motion lib of AMASS.

**Run Inference:**

.. code-block:: bash

   # Run G1 on BONES-SEED motions
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
       --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
       --simulator isaacgym

   # Run SOMA 23-body humanoid on BONES-SEED motions
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
       --motion-file data/motion_for_trackers/soma23_bones_seed_mini.pt \
       --simulator isaacgym

   # Run SMPL on flat terrain (requires AMASS MotionLib, see amass_preparation)
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
       --motion-file path/to/your/amass_motionlib.pt \
       --simulator isaacgym

   # Run SMPL on complex terrain
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt \
       --motion-file path/to/your/amass_motionlib.pt \
       --simulator isaacgym

   # Test sim2sim transfer - run IsaacLab-trained policy in MuJoco
   # CPU-only testing (no GPU needed, single env)
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
       --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
       --simulator mujoco \
       --num-envs 1

.. note::

   Sim2sim transfer works for robots with hinge (revolute) joints (G1, H1, etc.)
   but not yet for robots with spherical joints (SMPL, SMPL-X) due to differing
   spherical joint representations across simulators.  See
   :doc:`../tutorials/workflows/domain_randomization` for details.

Train Your First Agent
----------------------

Motion Imitation Training With DeepMimic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train a motion imitation agent using an MLP policy:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name smpl_mimic_example \
       --motion-file path/to/your/motion_lib.pt \
       --num-envs 4096 \
       --batch-size 16384 \
       --ngpu 1

For motion data preparation, see :doc:`amass_preparation`.

Selecting Simulator and Robot
-----------------------------

Simulator Selection
~~~~~~~~~~~~~~~~~~~

Use the ``--simulator`` argument:

* ``isaacgym`` - NVIDIA IsaacGym (recommended for training)
* ``isaaclab`` - NVIDIA IsaacLab/IsaacSim
* ``newton`` - NVIDIA Newton (built on MuJoCo Warp, currently beta)
* ``genesis`` - Genesis simulator
* ``mujoco`` - MuJoCo CPU-only (single env, for quick testing/debugging)

Robot Selection
~~~~~~~~~~~~~~~

Use the ``--robot-name`` argument:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Robot
     - Description
   * - ``smpl``
     - SMPL humanoid (digital human)
   * - ``smplx``
     - SMPL-X humanoid with hands
   * - ``g1``
     - Unitree G1 humanoid robot
   * - ``h1_2``
     - Unitree H1 humanoid robot (version 2)
   * - ``amp``
     - AMP humanoid
   * - ``soma23``
     - SOMA 23-body humanoid (digital human)
   * - ``rigv1``
     - Custom rigged character

See :doc:`../tutorials/workflows/custom_robot` for adding your own robot.

Experiment Management
---------------------

The ``--experiment-name`` determines where results are saved. When training with an 
existing experiment name, training automatically resumes from the last checkpoint.

Results are saved to:

.. code-block:: text

   results/<experiment_name>/
   ├── config.yaml                      # CLI arguments and wandb ID
   ├── resolved_configs.pt              # Full config objects (for exact reproducibility)
   ├── resolved_configs.yaml            # Human-readable configs
   ├── resolved_configs_inference.pt    # Inference-time configs (largely same as training configs)
   ├── resolved_configs_inference.yaml  # Human-readable inference configs
   ├── experiment_config.py             # Copy of experiment file
   ├── last.ckpt                        # Latest model checkpoint
   ├── score_based.ckpt                 # Best-performing checkpoint (by eval score)
   ├── epoch_100.ckpt                   # Intermediate checkpoints (if configured)
   └── env_<task_id>.ckpt               # Environment state for exact resume

.. note::

   Resume (if experiment name is the same) uses exact saved configs - CLI overrides are ignored during resume. This design helps automatic resume with many-gpu runs on clusters.

   For config changes, use a new experiment name. When training on cloud/cluster, you can also copy the source code to a new directory and train there with any experiment name.

.. warning::

   **Do NOT modify resolved_configs.yaml files.** They are for human readability 
   only—the source of truth is the ``.pt`` file. For config changes, use 
   ``--overrides`` (small changes) or ``--create-config-only`` and copy the new 
   ``.pt`` to your checkpoint directory (large changes). See :doc:`/user_guide/configuration`.

Training Configuration
----------------------

Common configuration options:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name smpl \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name my_experiment \
       --motion-file path/to/motions.pt \
       --num-envs 4096 \
       --batch-size 16384 \
       --ngpu 1 \
       --training-max-steps 10000000

Config Overrides
~~~~~~~~~~~~~~~~

Use ``--overrides`` to modify config values at runtime:

.. code-block:: bash

   --overrides "agent.num_mini_epochs=4" "env.max_episode_length=500"

**Supported override format:** ``config_type.field.subfield=value``

Supported config types: ``env``, ``simulator``, ``robot``, ``agent``, ``terrain``, ``motion_lib``, ``scene_lib``

Supported value types: ``int``, ``float``, ``bool``, ``str``, ``None``

**Limitations:** Overrides only support simple scalar values. Complex types like lists, nested objects, or dataclass instances cannot be overridden via CLI. For such changes, create a new experiment file - this is also good practice for managing and tracking different experiment configurations.

See :doc:`../user_guide/configuration` for more details on the configuration system.

Logging with Weights & Biases
-----------------------------

First, set up wandb authentication:

.. code-block:: bash

   wandb login

Then enable experiment tracking:

.. code-block:: bash

   python protomotions/train_agent.py \
       ... \
       --use-wandb

Key metrics to monitor:

* ``Eval/gt_err`` - Position tracking error (unbiased, evaluates all motions equally)
* ``Eval/success_rate`` - Motion completion rate (unbiased)
* ``Train/episode_reward`` - Training reward (may fluctuate due to prioritized sampling)
* ``Train/clip_frac`` - Keep under ~0.3 for stable training (lower lr if consistently higher)
* ``Train/actor_grad_norm`` / ``Train/critic_grad_norm`` - Watch for gradient explosions

.. tip::

   Weights & Biases has many useful features beyond basic metric plots. You can search 
   and filter runs by any config parameter, compare runs side-by-side, and create custom 
   dashboards. Spend some time exploring the UI to get the most out of experiment tracking.

Evaluation
----------

Evaluate a trained agent:

.. code-block:: bash

   # Evaluate G1 pretrained model
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
       --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
       --simulator isaacgym

   # Evaluate SOMA pretrained model
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/soma-bones/last.ckpt \
       --motion-file data/motion_for_trackers/soma23_bones_seed_mini.pt \
       --simulator isaacgym

   # Evaluate SMPL pretrained model (flat terrain)
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
       --motion-file path/to/your/amass_motionlib.pt \
       --simulator isaacgym

   # Evaluate SMPL pretrained model (complex terrain)
   python protomotions/inference_agent.py \
       --checkpoint data/pretrained_models/motion_tracker/smpl-terrains/last.ckpt \
       --motion-file path/to/your/amass_motionlib.pt \
       --simulator isaacgym

   # Or evaluate your own trained model
   python protomotions/inference_agent.py \
       --checkpoint results/my_experiment/last.ckpt \
       --motion-file data/motion_for_trackers/g1_random_subset_tiny.pt \
       --simulator isaacgym

Keyboard Controls
~~~~~~~~~~~~~~~~~

During visualization:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Key
     - Description
   * - ``J``
     - Apply physical force to all robots (test robustness)
   * - ``R``
     - Reset the task
   * - ``O``
     - Toggle camera (cycles through entities)
   * - ``L``
     - Toggle video recording
   * - ``Q``
     - Quit

Next Steps
----------

* :doc:`amass_preparation` - Prepare AMASS motion data
* :doc:`../tutorials/index` - End-to-end workflow tutorials
* :doc:`../concepts/index` - Understand core abstractions
* :doc:`../user_guide/configuration` - Configuration system deep dive
