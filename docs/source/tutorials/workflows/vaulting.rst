Scene Workflow (Vaulting)
=========================

This workflow demonstrates a complete pipeline for training scene interaction tasks,
using vaulting over obstacles as an example. It covers motion preparation, scene
creation, retargeting, and training.

Overview
--------

The vaulting example shows a character vaulting over obstacles. This requires:

1. Motion data with the vaulting animation
2. Scene data with obstacle placement matching the motion
3. Training the agent to reproduce the motion while interacting with the scene

.. code-block:: text

   Motion data ────────┐
                       ├──► Training ──► Vaulting policy
   Scene data ─────────┘

.. note::
   
   All intermediate files from each step are pre-computed and included in
   ``examples/data/rigv1-vaulting/``. You can skip any step and use the provided
   outputs, or run through the workflow to verify each step produces matching results.
   
   **Provided files:**
   
   * ``generated_vault_over_small_obstacle.pkl`` - Source motion
   * ``proto/`` - Step 1 output (proto format motion)
   * ``proto.pt`` - Step 1 output (packaged MotionLib)
   * ``keypoints-for-retarget/`` - Step 1 output (keypoints for retargeting)
   * ``rigv1-obstacle.pt`` - Step 2 output (rigv1 scene)
   * ``pyroki-retargeted/`` - Step 5a output (retargeted motion)
   * ``proto-g1/`` - Step 5b output (G1 proto format)
   * ``proto-g1.pt`` - Step 5c output (G1 MotionLib)
   * ``g1-obstacle.pt`` - Step 5d output (G1 scene)
   
   Pre-trained models for both rigv1 and G1 will be available in the pre-trained
   models collection (not yet uploaded).

Step 1: Motion Preparation
--------------------------

Convert source motion to ProtoMotions format:

.. code-block:: bash

   python data/scripts/convert_rigv1_pkl_to_proto.py \
       examples/data/rigv1-vaulting/ \
       examples/data/rigv1-vaulting/proto \
       --input-fps 30 --output-fps 30 \
       --yaml-output-name rigv1-vaulting.yaml \
       --ignore-motion-filter \
       --force-remake \
       --extract-keypoints \
       --keypoints-output-path examples/data/rigv1-vaulting/keypoints-for-retarget

Package into MotionLib:

.. code-block:: bash

   python ./protomotions/components/motion_lib.py \
       --motion-path examples/data/rigv1-vaulting/proto \
       --output-file examples/data/rigv1-vaulting/proto.pt \
       --device cpu

**Output:**

.. code-block:: text

   Loading 1/1 motion files: examples/data/rigv1-vaulting/proto/generated_vault_over_small_obstacle.motion
   Loaded 1 motions with a total length of 2.633s.
   Motion library saved to examples/data/rigv1-vaulting/proto.pt

Step 2: Scene Creation
----------------------

Create scene data matching the motion. The scene script reads motion metadata and
places obstacles at the correct positions:

.. code-block:: bash

   python examples/data/rigv1-vaulting/rigv1-obstacle.py \
       --yaml examples/data/rigv1-vaulting/proto/rigv1-vaulting.yaml \
       --output examples/data/rigv1-vaulting/rigv1-obstacle.pt

This creates a ``.pt`` file containing obstacle positions, dimensions, and per-motion
scene assignments.

How Scene Scripts Work
~~~~~~~~~~~~~~~~~~~~~~

Scene scripts use the SceneLib API to define objects:

.. code-block:: python

   from protomotions.components.scene_lib import SceneLib, Scene, SceneObject
   
   # Create a box obstacle
   obstacle = SceneObject(
       object_type="box",
       position=[1.0, 0.0, 0.3],  # x, y, z
       dimensions=[0.5, 0.8, 0.6],  # width, depth, height
       fix_base_link=True,  # Static object
   )
   
   scene = Scene(objects=[obstacle])
   scene_lib = SceneLib(scenes=[scene])

Step 3: Verify with Kinematic Playback
--------------------------------------

Before training, verify motion and scene alignment:

.. code-block:: bash

   python examples/env_kinematic_playback.py \
       --experiment-path=examples/experiments/mimic/mlp.py \
       --motion-file=examples/data/rigv1-vaulting/proto.pt \
       --robot-name=rigv1 \
       --simulator=isaacgym \
       --num-envs=5 \
       --scenes-file=examples/data/rigv1-vaulting/rigv1-obstacle.pt

This plays the motion kinematically with the scene objects rendered. Check that:

* The character's hands contact the obstacle at the right time
* The character clears the obstacle during the vault
* Scene objects are correctly positioned

Step 4: Train on Original Character
-----------------------------------

Train the original character (rigv1):

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name rigv1 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name rigv1_vaulting \
       --motion-file examples/data/rigv1-vaulting/proto.pt \
       --scenes-file examples/data/rigv1-vaulting/rigv1-obstacle.pt \
       --num-envs 4096 \
       --batch-size 16384

Note: For vaulting, you may want to disable contact matching rewards since
hand-obstacle contacts are part of the task:

.. code-block:: bash

   --overrides "env.reward_config.contact_match_rew.weight=0.0"

(yes, such string-based overrides will work with the factory method rewards)

Step 5: Retarget to Robot (G1)
------------------------------

To transfer the vaulting skill to a robot like G1:

**5a. Retarget motion:**

.. code-block:: bash

   conda activate pyroki
   
   python pyroki/batch_retarget_to_g1_from_keypoints.py \
       --keypoints-folder-path examples/data/rigv1-vaulting/keypoints-for-retarget \
       --output-dir examples/data/rigv1-vaulting/pyroki-retargeted \
       --no-visualize \
       --skip-existing \
       --subsample-factor 1

**5b. Convert to proto format:**

.. code-block:: bash

   python data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
       --retargeted-motion-dir examples/data/rigv1-vaulting/pyroki-retargeted/ \
       --output-dir examples/data/rigv1-vaulting/proto-g1 \
       --robot-type g1 \
       --input-fps 30 --output-fps 30 \
       --force-remake

**5c. Package motion:**

.. code-block:: bash

   python ./protomotions/components/motion_lib.py \
       --motion-path examples/data/rigv1-vaulting/proto-g1 \
       --output-file examples/data/rigv1-vaulting/proto-g1.pt \
       --device cpu

**5d. Create G1 scene (obstacle heights may differ):**

.. code-block:: bash

   python examples/data/rigv1-vaulting/g1-obstacle.py \
       --yaml examples/data/rigv1-vaulting/proto/rigv1-vaulting.yaml \
       --output examples/data/rigv1-vaulting/g1-obstacle.pt

**5e. Verify retargeted motion:**

.. code-block:: bash

   python examples/env_kinematic_playback.py \
       --experiment-path=examples/experiments/mimic/mlp.py \
       --motion-file=examples/data/rigv1-vaulting/proto-g1.pt \
       --robot-name=g1 \
       --simulator=isaacgym \
       --num-envs=5 \
       --scenes-file=examples/data/rigv1-vaulting/g1-obstacle.pt

**5f. Train G1:**

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaacgym \
       --experiment-path examples/experiments/mimic/mlp.py \
       --experiment-name g1_vaulting \
       --motion-file examples/data/rigv1-vaulting/proto-g1.pt \
       --scenes-file examples/data/rigv1-vaulting/g1-obstacle.pt \
       --num-envs 4096 \
       --batch-size 16384

Creating Your Own Scenes
------------------------

To create custom scenes:

1. Create a Python script that generates scene data
2. Use SceneLib API to define objects
3. Save to ``.pt`` file
4. Reference in training with ``--scenes-file``

Example script structure:

.. code-block:: python

   import torch
   from protomotions.components.scene_lib import SceneLib, Scene, SceneObject
   
   def create_corridor_scene():
       """Create a corridor with walls on both sides."""
       left_wall = SceneObject(
           object_type="box",
           position=[-1.0, 0.0, 1.0],
           dimensions=[0.1, 10.0, 2.0],
           fix_base_link=True,
       )
       right_wall = SceneObject(
           object_type="box",
           position=[1.0, 0.0, 1.0],
           dimensions=[0.1, 10.0, 2.0],
           fix_base_link=True,
       )
       return Scene(objects=[left_wall, right_wall])
   
   if __name__ == "__main__":
       scene = create_corridor_scene()
       scene_lib = SceneLib(scenes=[scene])
       torch.save(scene_lib.to_dict(), "corridor_scene.pt")


Next Steps
----------

* :doc:`domain_randomization` - Prepare for sim2sim transfer
* :doc:`custom_robot` - Add your own robot for vaulting
* :doc:`../../concepts/abstractions` - Understand SceneLib in depth

