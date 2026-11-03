Robots
======

Choose a robot with ``--robot-name`` when building a training configuration.
These pages describe the default models registered in
``protomotions/robot_configs/factory.py``, not every asset variant or physical
robot that can be integrated into ProtoMotions.

.. list-table:: Default Models
   :header-rows: 1
   :widths: 20 13 13 14 20 20

   * - Model / CLI name
     - Bodies
     - Joint DOFs
     - Forward
     - Observation anchor
     - Typical input
   * - :doc:`g1` / ``g1``
     - 33
     - 29
     - +X
     - ``torso_link``
     - G1-retargeted motion
   * - :doc:`h1_2` / ``h1_2``
     - 29
     - 27
     - +X
     - ``pelvis`` (root)
     - H1-2-retargeted motion
   * - :doc:`soma23` / ``soma23``
     - 23
     - 66
     - -Y
     - ``Hips`` (root)
     - Converted SOMA motion
   * - :doc:`anatomic_soma` / ``anatomic_soma``
     - 53
     - 52
     - +X
     - ``Chest``
     - Retargeted SOMA motion
   * - :doc:`smpl` / ``smpl``
     - 24
     - 69
     - +X
     - ``Pelvis`` (root)
     - Converted SMPL motion
   * - :doc:`smplx` / ``smplx``
     - 52
     - 153
     - +X
     - ``Pelvis`` (root)
     - Converted SMPL-X motion
   * - :doc:`amp` / ``amp``
     - 15
     - 36
     - +X
     - ``pelvis`` (root)
     - Motion matching the AMP humanoid

Counts refer to the configured MJCF, exclude the world body and floating-base
DOFs, and include fixed bodies and intermediate hinge carriers. Anatomical
SOMA has 23 semantic parts plus 30 carriers. All models use Z up; positions
are in meters and scalar joint angles are in radians. A root anchor means
``anchor_body_name=None``, which resolves to body index zero.

.. toctree::
   :maxdepth: 1

   g1
   h1_2
   soma23
   anatomic_soma
   smpl
   smplx
   amp

Configuration And Compatibility
-------------------------------

A robot config supplies the asset, semantic body mappings, control parameters
and backend-specific simulation settings. An experiment may override these
defaults before saving ``resolved_configs.json``. Resume uses that saved config;
inference starts from the saved config and applies inference overrides. The
pages here do not override the contract of an existing checkpoint.

Use motion files and checkpoints prepared for the same articulation, joint
order, body frames and observation layout. Matching DOF counts alone does not
establish compatibility. The observation anchor is distinct from the physical
root and from the bodies selected for target tracking.

PD gains and effort limits are resolved from the MJCF and the config's
``control.override_control_info``. The action-to-target mapping belongs to the
experiment. Changing gains, action scaling or simulation frequency can change
an existing policy's behavior; a backend config is not evidence of sim-to-sim
transfer quality.

Assets And Workflows
--------------------

Asset paths on each page are relative to ``protomotions/data/assets/``. Use a
source checkout for the linked example experiments and data tools. Git LFS
payloads must be materialized where required. SMPL-family assets and body
models have separate licensing and installation requirements; see their pages.

* :doc:`../getting_started/installation`: installation and asset availability.
* :doc:`../getting_started/pretrained_models`: available policies and matching data.
* :doc:`../user_guide/configuration`: experiment and saved-config lifecycle.
* :doc:`../tutorials/workflows/custom_robot`: integrate a different robot.
* :doc:`../api_reference/protomotions.robot_configs`: configuration API.
