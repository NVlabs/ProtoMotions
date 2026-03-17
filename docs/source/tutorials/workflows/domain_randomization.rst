Domain Randomization & Sim2Sim
==============================

This workflow covers training with domain randomization for robust policies that
transfer across simulators (sim2sim) or to real robots (sim2real).

For a complete end-to-end pipeline from data to deployment, see
:doc:`g1_deployment`.

Why Domain Randomization?
-------------------------

Policies trained in simulation often fail when deployed to different physics engines
or real hardware due to the "reality gap". Domain randomization addresses this by:

1. **Randomizing physics parameters** (friction, center of mass)
2. **Adding action noise** (motor imprecision)
3. **Adding observation noise** (sensor noise for IMU, encoders)
4. **Applying external perturbations** (pushes, velocity impulses)
5. **Forcing the policy** to be robust to parameter variations

Training with Domain Randomization
-----------------------------------

The BeyondMimic L2C2 experiment config (``examples/experiments/mimic/mlp_bm_l2c2.py``)
is a good starting point for DR training:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaaclab \
       --experiment-path examples/experiments/mimic/mlp_bm_l2c2.py \
       --experiment-name g1_bm_dr \
       --motion-file /path/to/bones_seed_g1_motions.pt \
       --num-envs 4096 \
       --batch-size 16384

You can also create your own experiment configs with DR settings.  The randomization types
described below are general and can be mixed into any experiment config.


Domain Randomization Types
--------------------------

ProtoMotions supports several randomization types via ``DomainRandomizationConfig``.
Below are the settings used in the pre-trained G1 BeyondMimic tracker
(``data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py``).  You can
adjust ranges to suit your robot and deployment conditions.

**Action Noise:**

.. code-block:: python

   ActionNoiseDomainRandomizationConfig(
       action_noise_range=(-0.025, 0.025),  # +/-2.5% noise on PD targets
       dof_names=[".*"],  # Apply to all joints
   )

**Friction Randomization:**

.. code-block:: python

   FrictionDomainRandomizationConfig(
       num_buckets=64,
       static_friction_range=(0.3, 1.6),
       dynamic_friction_range=(0.3, 1.2),
       restitution_range=(0.0, 0.5),
       body_names=[".*"],  # Apply to all bodies
   )

.. note::

   **Default Values:** ProtoMotions assumes a default friction of **1.0** and
   restitution of **0.0** for all entities (robot bodies and terrain) when
   values are not explicitly set.  This ensures consistent behavior across
   simulators.

**Center of Mass Randomization:**

.. code-block:: python

   CenterOfMassDomainRandomizationConfig(
       com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
       body_names=["torso_link"],  # Apply to torso
   )

**Observation Noise:**

Adds Gaussian noise to observations to improve robustness to real-world sensor
imperfections.  The noise levels are set empirically to make the policy robust
-- they are not calibrated against specific sensor datasheets.

The BM tracker config uses per-component noise via ``RobotNoiseConfig``:

.. code-block:: python

   RobotNoiseConfig(
       dof_pos_noise=0.01,         # Joint encoder noise (radians)
       dof_vel_noise=0.5,          # Joint velocity noise (rad/s)
       anchor_rot_noise=0.05,      # Torso IMU orientation noise (quat components)
       anchor_ang_vel_noise=0.2,   # Pelvis IMU gyroscope noise (rad/s)
   )

At inference time (and in the exported ONNX model), all noise is disabled --
the policy sees clean sensor data but has learned to be robust to the noise
levels encountered during training.

When observation noise is configured, the environment provides both clean and
noisy versions of state variables in the context.  Observation components can
request noisy inputs (for actor) or clean inputs (for critic) via the
``use_noisy`` parameter:

.. code-block:: python

   # Noisy observations for actor (helps sim-to-real transfer)
   "noisy_obs": reduced_coords_obs_factory(use_noisy=True),

   # Clean observations for critic (asymmetric actor-critic)
   "clean_obs": reduced_coords_obs_factory(use_noisy=False),

**Push/Perturbation Randomization:**

.. code-block:: python

   PushDomainRandomizationConfig(
       push_interval_range=(1.0, 3.0),         # Seconds between pushes
       max_linear_velocity=(0.5, 0.5, 0.2),    # Max push velocity (x, y, z) m/s
       max_angular_velocity=(0.52, 0.52, 0.78), # Max angular impulse (rad/s)
   )

This helps policies learn to recover from unexpected perturbations, improving
robustness on real hardware where the robot may be bumped or jostled.

**Reset Noise (initial state perturbation):**

.. code-block:: python

   RobotNoiseConfig(
       dof_pos_noise=0.1,
       root_pos_noise=[0.05, 0.05, 0.01],
       root_rot_noise=[0.1, 0.1, 0.2],
       root_vel_noise=[0.1, 0.1, 0.05],
       root_ang_vel_noise=[0.1, 0.1, 0.1],
   )

At the start of each episode, the robot's initial state is perturbed around the
reference motion's first frame.  This prevents the policy from relying on a
perfect starting pose.


Automatic Friction Combine Mode Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different physics engines use different friction combine modes:

* **IsaacGym / IsaacLab (PhysX)**: Uses ``AVERAGE`` -- effective friction is
  the average of robot and terrain friction: ``(robot + terrain) / 2``
* **MuJoCo**: Uses ``MAX`` -- effective friction is the maximum of the two:
  ``max(robot, terrain)``

ProtoMotions **automatically converts** friction settings when switching
simulators via ``convert_friction_for_simulator()``.  You can configure friction
using any combine mode, and the system ensures equivalent effective friction
behavior across simulators.

**How it works:**

When running on MuJoCo with an ``AVERAGE`` mode config:

1. Robot shape friction is set to a minimum value (0.01)
2. Terrain friction is set to the computed effective value
3. Result: ``max(0.01, effective) = effective`` -- matching the intended behavior

When domain randomization is configured:

1. Terrain friction is set to minimum (0.01)
2. Robot friction randomization range is converted to effective values
3. Result: ``max(robot_dr_value, 0.01) = robot_dr_value`` -- matching intended range

This conversion is transparent -- you don't need to change your config for
different simulators.  The same experiment config works on IsaacGym, IsaacLab,
and MuJoCo with equivalent friction behavior.


L2C2 Smoothness Regularization
------------------------------

The BM tracker config additionally uses **L2C2** (Locally Lipschitz Continuous
Constraint, Kobayashi 2022) to encourage smooth policy outputs despite noisy
observations:

* Both noisy and clean versions of each actor observation are computed
* The policy runs on both, producing ``mu_noisy`` and ``mu_clean``
* An auxiliary loss penalizes ``MSE(mu_noisy, mu_clean)``
* This encourages the policy to produce similar actions regardless of sensor noise

At inference time, only the clean (noise-free) observations are used and the
L2C2 components are removed.


Sim2Sim Testing
---------------

After training with DR, test on MuJoCo to verify transfer before real-robot
deployment:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/g1_bm_dr/last.ckpt \
       --simulator mujoco \
       --motion-file /path/to/motion.motion

MuJoCo uses different contact dynamics, solver, and friction combine mode than
the training simulator (IsaacLab).  A policy that works across both has learned
robust dynamics rather than overfitting to one physics engine.

.. warning::

   **Spherical joint limitation:** Sim2sim transfer currently only works for
   robots with hinge (revolute) joints, such as the G1 and H1.  Robots that
   use spherical (ball) joints — like SMPL and SMPL-X — have different
   spherical joint representations across simulators (IsaacGym/IsaacLab vs
   Newton/MuJoCo), and cross-simulator transfer is not yet supported for
   these morphologies.

For a more thorough deployment validation, see the standalone MuJoCo test script
described in :doc:`g1_deployment` (Step 4), which runs the exported ONNX model
independently of the ProtoMotions training framework.


Training Tips
-------------

**Start without DR**: Train a baseline without domain randomization first. This
confirms your motion data and rewards are working.  In our experience, DR does
not make training significantly harder.

**Observation history**: Some configs use observation history to help the
policy infer physics parameters from recent state transitions.  The BM tracker
config uses previous processed actions as a form of single-step history.

**Noise levels**: The noise values above are empirically set to produce robust
policies.  If your real-robot sensors are particularly noisy or clean, adjust
the noise ranges accordingly.  When in doubt, err on the side of more noise --
it's better to over-regularize than to have a fragile policy.


Next Steps
----------

* :doc:`g1_deployment` - Full pipeline from data to real robot deployment
* :doc:`custom_robot` - Add your robot for DR training
* :doc:`../../concepts/abstractions` - Understand simulator abstraction
