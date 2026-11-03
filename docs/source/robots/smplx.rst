SMPL-X Humanoid
===============

Select ``--robot-name smplx`` for the simulated SMPL-X articulation with
articulated fingers. Its configured asset is ``mjcf/smplx_humanoid.xml``.
The counts below describe this simulation asset, not every joint or landmark
in the full statistical SMPL-X body model.

Default Contract
----------------

* Config: ``protomotions/robot_configs/smplx.py`` (``SMPLXRobotConfig``).
* Structure: 52 bodies, 153 joint DOFs; +X forward, Z up.
* Observation anchor: root body ``Pelvis`` (no explicit anchor override).
* Target bodies: all bodies by default, inherited from ``RobotConfig``.

Hand semantic groups include the wrists and finger segments. Finger contact
bodies and smaller visualization markers are configured explicitly. The config
uses lower gains and effort limits for fingers than for the torso and limbs;
preserve those distinctions when adapting a controller.

Assets And Motion Preparation
-----------------------------

SMPL-family MJCF assets are excluded from the pip wheel. Obtain the required
source assets and body models under their own license terms. Use the SMPL-X
conversion path; a 24-body SMPL motion does not supply this model's finger
coordinates or body layout.

* :doc:`../getting_started/installation`: source checkout and asset availability.
* :doc:`../getting_started/amass_preparation`: SMPL-X model setup and conversion
  with ``--humanoid-type smplx``.
* :doc:`../user_guide/configuration`: choose a smaller task-specific tracking subset.
