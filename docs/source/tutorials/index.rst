Tutorials
=========

These tutorials walk through complete end-to-end workflows for common tasks.

Deployment Workflows
--------------------

* :doc:`workflows/g1_deployment` - G1 whole-body tracker: from data to real robot (full pipeline)

Training Workflows
------------------

* :doc:`workflows/amass_smpl` - Train SMPL humanoid on AMASS motion data
* :doc:`workflows/retargeting_pyroki` - Retarget AMASS/SMPL motions to robots
* :doc:`workflows/vaulting` - Scene interaction with vaulting motions
* :doc:`workflows/domain_randomization` - Sim2sim transfer techniques

Robot & Environment Setup
-------------------------

* :doc:`workflows/custom_robot` - Add your own robot morphology

Code Tutorials
--------------

For learning ProtoMotions internals from the ground up, see :doc:`code_tutorials` - 
8 progressive Python scripts in ``examples/tutorial/`` that teach core concepts
like simulators, terrain, robots, scenes, and environments.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Workflow
     - Description
     - Prerequisites
   * - :doc:`workflows/g1_deployment`
     - Full pipeline: data to real robot (G1 tracker)
     - Pre-trained model or BONES-SEED data
   * - :doc:`workflows/amass_smpl`
     - Train SMPL on AMASS motions
     - AMASS data prepared
   * - :doc:`workflows/retargeting_pyroki`
     - Retarget AMASS/SMPL to robots
     - Packaged AMASS .pt
   * - :doc:`workflows/vaulting`
     - Full scene interaction workflow
     - Motion + scene data
   * - :doc:`workflows/domain_randomization`
     - Sim2sim transfer
     - Trained policy
   * - :doc:`workflows/custom_robot`
     - Add new robot morphology
     - MJCF file

Coming Soon
-----------

* **MJCF to USD Conversion** - Convert MuJoCo assets to USD for IsaacLab
* **USD GS Backgrounds** - Load Gaussian Splatting scenes in IsaacSim 5.0+
* **Pure RL Training** - Train without motion data (steering, locomotion)
* **MaskedMimic / AMP / ASE** - Generative policy training
