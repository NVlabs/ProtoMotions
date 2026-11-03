SOMA23
======

Select ``--robot-name soma23`` for the 23-body SOMA humanoid. It drops seven
unactuated leaf bodies from SOMASkeleton30 and retains 66 joint coordinates.
The default asset is ``mjcf/soma23_humanoid.xml``.

Default Contract
----------------

* Config: ``protomotions/robot_configs/soma23.py`` (``Soma23RobotConfig``).
* Structure: 23 bodies, 66 joint DOFs; -Y forward, Z up.
* Observation anchor: the root body ``Hips`` (no override).
* Target bodies: ``Hips``, ``Head``, ``LeftFoot``, ``RightFoot``,
  ``LeftHand``, ``RightHand``.

PD gains and armatures come from the MJCF. The config overrides effort limits
to 300 N m for torso/head and legs, 150 N m for arms/hands, and joint velocity
limits to 100 rad/s. Experiments may override these values and the action map.
The default MuJoCo settings use ``implicitfast`` at 960 physics steps/s and
decimation 32, preserving a 30 Hz control rate.

These defaults are the observation contract of the shipped SOMA23 policies and
stay as they are. Anatomical SOMA anchors on ``Chest`` instead; see its page.

Motion Preparation
------------------

Use the SOMA conversion pipeline for this model. Do not substitute anatomical
SOMA coordinates: its hinge structure, forward direction and body count differ.
See :doc:`anatomic_soma` for the comparison and explicit retargeting procedure.

* :doc:`../getting_started/seed_bvh_preparation`: SOMA BVH data conversion.
* :doc:`../getting_started/kimodo_preparation`: Kimodo motion preparation.
* :doc:`../getting_started/pretrained_models`: SOMA trackers and priors.
