AMP Humanoid
============

Select ``--robot-name amp`` for the compact simulated humanoid in
``mjcf/amp_humanoid.xml``. The robot name does not select the AMP learning
algorithm: the agent and rewards are chosen independently by the experiment.

Default Contract
----------------

* Config: ``protomotions/robot_configs/amp.py`` (``AMPRobotConfig``).
* Structure: 15 bodies, 36 joint DOFs; +X forward, Z up.
* Observation anchor: root body ``pelvis`` (no explicit anchor override).
* Target bodies: ``torso``, ``head``, ``left_foot``, ``right_foot``,
  ``left_hand``, ``right_hand``.

PD gains, armatures and effort limits come from the MJCF; the config selects
built-in PD without per-joint overrides. Its hands are fixed bodies rather than
the articulated fingers of SMPL-X. The default control rate is 30 Hz.

Motion And Experiments
----------------------

Use reference motion prepared for this asset's body names and joint order.
Do not select ``amp`` merely because a dataset or an experiment uses an AMP
discriminator; that discriminator can also be used with other robot models.

* :doc:`../user_guide/experiments`: experiment and agent choices.
* :doc:`../concepts/pose_lib`: articulation and motion coordinate conventions.
* :doc:`../tutorials/workflows/custom_robot`: preparing compatible assets and mappings.
