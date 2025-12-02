Key Concepts
============

This section covers the core abstractions and design principles of ProtoMotions.

Overview
--------

ProtoMotions is designed to support many combinations:

* **Simulators**: IsaacGym, IsaacLab, Newton, Genesis
* **Robots**: SMPL, G1, H1, custom morphologies
* **Algorithms**: PPO, AMP, ASE, MaskedMimic
* **Environments**: Mimic, Steering, PathFollower

This necessitates modular abstractions that allow mixing and matching components
without code changes.

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Concept
     - Purpose
     - Location
   * - :doc:`architecture`
     - High-level system design
     - Overview
   * - :doc:`abstractions`
     - Core component classes
     - ``protomotions/components/``, ``envs/``, ``simulator/``
   * - :doc:`pose_lib`
     - MJCF parsing, FK/IK utilities
     - ``protomotions/components/pose_lib.py``
   * - :doc:`simulator_state`
     - Robot and object state representation
     - ``protomotions/simulator/base_simulator/simulator_state.py``

