Domain Randomization & Sim2Sim
==============================

This workflow covers training with domain randomization for robust policies that 
transfer across simulators (sim2sim) or to real robots (sim2real).

Why Domain Randomization?
-------------------------

Policies trained in simulation often fail when deployed to different physics engines 
or real hardware due to the "reality gap". Domain randomization addresses this by:

1. **Randomizing physics parameters** (friction, center of mass)
2. **Adding action noise** (motor imprecision)
3. **Adding observation noise** (sensor noise)
4. **Applying external perturbations** (pushes, bumps)
5. **Forcing the policy** to be robust to parameter variations

Training with Domain Randomization
----------------------------------

Use the ``mlp_domain_rand.py`` experiment config:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp_domain_rand.py \
       --experiment-name g1_amass_dr \
       --motion-file /path/to/amass_g1.pt \
       --num-envs 8192 \
       --batch-size 8192 \
       --ngpu 4

Domain Randomization Parameters
-------------------------------

ProtoMotions supports several randomization types via ``DomainRandomizationConfig``:

* **Action Noise** — Motor imprecision
* **Friction** — Ground and body friction coefficients
* **Center of Mass** — Body mass distribution shifts
* **Observation Noise** — Sensor noise simulation
* **Push Perturbations** — External disturbances

The ``mlp_domain_rand.py`` example config demonstrates common settings:

**Action Noise:**

.. code-block:: python

   ActionNoiseDomainRandomizationConfig(
       action_noise_range=(-0.02, 0.02),  # ±2% noise on actions
       dof_names=[".*"],  # Apply to all joints
   )

**Friction Randomization:**

.. code-block:: python

   FrictionDomainRandomizationConfig(
      num_buckets=64,  # Number of friction groups
      static_friction_range=(0.6, 3.0),
      dynamic_friction_range=(0.6, 3.0),
      restitution_range=(0.0, 1.0),
      body_names=[".*"],  # Apply to all bodies
   )

.. note::

   **Default Values:** ProtoMotions assumes a default friction of **1.0** and restitution of 
   **0.0** for all entities (robot bodies and terrain) when values are not explicitly set. 
   This ensures consistent behavior across simulators.

.. note::

   **Friction Combine Mode:** In physics simulators, friction between two surfaces is 
   computed from both materials. The ``mlp_domain_rand.py`` config sets the floor friction 
   to near-zero (0.01) with ``CombineMode.AVERAGE``. This means the effective friction 
   is approximately half of the robot body's friction value.
   
   With robot friction randomized in the range (0.6, 3.0) and floor at 0.01:
   
   * **Effective friction range:** ~(0.3, 1.5)
   
   This approach lets you control the full friction range through robot body randomization 
   while keeping the floor constant.

Automatic Friction Combine Mode Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different physics engines use different friction combine modes:

* **IsaacGym (PhysX)**: Only supports ``AVERAGE`` — effective friction is the average 
  of robot and terrain friction: ``(robot + terrain) / 2``
* **Newton (MuJoCo)**: Only supports ``MAX`` — effective friction is the maximum of 
  the two: ``max(robot, terrain)``

ProtoMotions **automatically converts** friction settings when switching simulators. 
You can configure friction using any combine mode, and the system ensures equivalent 
effective friction behavior across simulators.

**How it works:**

When running on Newton with an ``AVERAGE`` mode config:

1. Robot shape friction is set to a minimum value (0.01)
2. Terrain friction is set to the computed effective value
3. Result: ``max(0.01, effective) = effective`` — matching the intended behavior

When domain randomization is configured:

1. Terrain friction is set to minimum (0.01)
2. Robot friction randomization range is converted to effective values
3. Result: ``max(robot_dr_value, 0.01) = robot_dr_value`` — matching intended range

This conversion is transparent — you don't need to change your config for different 
simulators. The same ``mlp_domain_rand.py`` config works on both IsaacGym and Newton 
with equivalent friction behavior.

**Center of Mass Randomization:**

.. code-block:: python

   CenterOfMassDomainRandomizationConfig(
       com_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
       body_names=["torso_link"],  # Apply to torso
   )

**Observation Noise:**

Adds Gaussian noise to observations to simulate sensor noise for sim-to-real transfer.
Noise is applied hierarchically to different state components:

.. code-block:: python

   ObservationNoiseDomainRandomizationConfig(
       # DOF noise (joint encoders)
       dof_pos_noise=0.01,       # Joint position noise (radians)
       dof_vel_noise=0.05,       # Joint velocity noise (rad/s)
       
       # Root body noise (IMU)
       root_rot_noise=0.02,      # Root orientation noise (radians)
       root_ang_vel_noise=0.1,   # Root angular velocity noise (rad/s)
       
       # Anchor body noise (pelvis IMU if different from root)
       anchor_rot_noise=0.02,
       anchor_ang_vel_noise=0.1,
       
       # Whole-body noise (motion capture noise)
       rigid_body_pos_noise=0.01,    # Position noise (meters)
       rigid_body_rot_noise=0.02,    # Rotation noise (radians)
       rigid_body_vel_noise=0.05,    # Linear velocity noise (m/s)
       rigid_body_ang_vel_noise=0.1, # Angular velocity noise (rad/s)
       
       # Environment noise
       ground_height_noise=0.02,     # Terrain height estimation noise
   )

When observation noise is configured, the environment provides both clean and noisy 
versions of state variables in the context. Observation components can request noisy 
inputs (for actor) or clean inputs (for critic) via the ``observation_noise`` parameter:

.. code-block:: python

   # Noisy observations for actor (helps sim-to-real transfer)
   "noisy_obs": reduced_coords_obs_factory(observation_noise=True),
   
   # Clean observations for critic (asymmetric actor-critic)
   "clean_obs": reduced_coords_obs_factory(observation_noise=False),

**Push/Perturbation Randomization:**

Applies random velocity impulses to simulate external disturbances (bumps, pushes):

.. code-block:: python

   PushDomainRandomizationConfig(
       push_interval_range=(1.0, 3.0),         # Seconds between pushes
       max_linear_velocity=(0.5, 0.5, 0.0),    # Max push velocity (x, y, z) m/s
       max_angular_velocity=(0.0, 0.0, 0.3),   # Max angular impulse (rad/s)
   )

This helps policies learn to recover from unexpected perturbations, improving 
robustness on real hardware where the robot may be bumped or jostled.

Sim2Sim Testing
---------------

After training with DR, test on different simulators to verify transfer:

**Test on Newton (MuJoCo-based):**

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/g1_amass_dr/last.ckpt \
       --simulator newton

.. note::

   Newton is currently in beta. You may observe physics artifacts as we have not yet 
   spent significant time tuning its solver parameters. Community contributions to 
   improve Newton's physics fidelity are welcome!

If the policy works across simulators, it has learned robust dynamics rather than 
overfitting to IsaacGym's specific physics.

ONNX Export for Deployment
--------------------------

Export trained policy to ONNX for deployment:

.. code-block:: bash

   python scripts/export_model_to_onnx.py \
       --checkpoint results/g1_amass_dr/last.ckpt \
       --output-path g1_policy.onnx

The ONNX model can be loaded in C++ or other frameworks for robot deployment.

Training Tips
-------------

**Start without DR**: Train a baseline without domain randomization first. This 
confirms your motion data and rewards are working. 
We did not find training becomes harder with DR in our experiments though.

**Observation history**: DR configs often use observation history to help the 
policy infer physics parameters:

.. code-block:: python

   max_coords_obs=MaxCoordsSelfObsConfig(
       enabled=True,
       num_historical_steps=3,  # 3 steps of history
   )

Full Pipeline: Train → DR → Sim2Sim
------------------------------------

1. **Baseline training** (no DR):

   .. code-block:: bash
   
      python protomotions/train_agent.py \
          --experiment-path examples/experiments/mimic/mlp.py \
          --experiment-name g1_baseline \
          ...

2. **DR training**:

   .. code-block:: bash
   
      python protomotions/train_agent.py \
          --experiment-path examples/experiments/mimic/mlp_domain_rand.py \
          --experiment-name g1_dr \
          ...

3. **Sim2sim test**:

   .. code-block:: bash
   
      # Test both policies on Newton
      python protomotions/inference_agent.py \
          --checkpoint results/g1_baseline/last.ckpt \
          --simulator newton
      
      python protomotions/inference_agent.py \
          --checkpoint results/g1_dr/last.ckpt \
          --simulator newton

4. **Compare**: The DR policy should perform better on Newton than the baseline.

Next Steps
----------

* :doc:`custom_robot` - Add your robot for DR training
* :doc:`../../user_guide/configuration` - More on config overrides
* :doc:`../../concepts/abstractions` - Understand simulator abstraction

