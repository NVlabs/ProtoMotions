Domain Randomization & Sim2Sim
==============================

This workflow covers training with domain randomization for robust policies that 
transfer across simulators (sim2sim) or to real robots (sim2real).

Why Domain Randomization?
-------------------------

Policies trained in simulation often fail when deployed to different physics engines 
or real hardware due to the "reality gap". Domain randomization addresses this by:

1. **Randomizing physics parameters** during training (friction, mass, etc.)
2. **Adding noise** to actions and observations
3. **Forcing the policy** to be robust to parameter variations

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

The ``mlp_domain_rand.py`` config enables several randomization types:

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

   **Friction Combine Mode:** In physics simulators, friction between two surfaces is 
   computed from both materials. The ``mlp_domain_rand.py`` config sets the floor friction 
   to near-zero (0.01) with ``CombineMode.AVERAGE``. This means the effective friction 
   is approximately half of the robot body's friction value.
   
   With robot friction randomized in the range (0.6, 3.0) and floor at 0.01:
   
   * **Effective friction range:** ~(0.3, 1.5)
   
   This approach lets you control the full friction range through robot body randomization 
   while keeping the floor constant.

**Center of Mass Randomization:**

.. code-block:: python

   CenterOfMassDomainRandomizationConfig(
       com_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
       body_names=["torso_link"],  # Apply to torso
   )

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

