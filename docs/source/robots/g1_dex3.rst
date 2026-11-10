Unitree G1 With Dex3-1 Hands
============================

Select ``--robot-name g1_dex3`` for the G1 body with Unitree's Dex3-1 hands.
The asset is ``mjcf/g1_dex3.xml``: the ``g1`` model with the three-finger hands
grafted onto the wrists in place of the rubber-hand markers. Every body joint,
gain and body frame is the ``g1`` robot's; the fingers are extra joints.

Default Contract
----------------

* Config: ``protomotions/robot_configs/g1_dex3.py`` (``G1Dex3RobotConfig``).
* Structure: 47 bodies, 43 joint DOFs (29 body joints plus 7 finger joints per
  hand); +X forward, Z up.
* Observation anchor: ``torso_link``; physical root: ``pelvis``.
* Target bodies: the ``g1`` set, with the hands tracked at
  ``left_wrist_yaw_link`` and ``right_wrist_yaw_link``, so a policy that tracks
  the body transfers between the two robots.

Finger joints use the GEAR-SONIC HOI experts' Dex3 motor settings (stiffness
0.5, thumb root 2.0, damping 0.1, armature 0.001) and rest open in the default
pose. The body uses the ``g1`` standing pose and PD gains.

Motion And Deployment
---------------------

Use G1-retargeted motion; finger targets come from the experiment, not from the
body motion file. A checkpoint trained on ``g1`` does not load on ``g1_dex3``
unchanged: the action and observation sizes include the 14 finger joints.

* :doc:`g1`: the shared body contract.
* :doc:`../tutorials/workflows/g1_deployment`: export and deployment workflow.
