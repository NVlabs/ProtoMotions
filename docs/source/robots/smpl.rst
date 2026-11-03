SMPL Humanoid
=============

Select ``--robot-name smpl`` for the 24-body simulated SMPL humanoid. Its
configured asset is ``mjcf/smpl_humanoid.xml``. This articulation is distinct
from the SMPL statistical body-model files used during data conversion.

Default Contract
----------------

* Config: ``protomotions/robot_configs/smpl.py`` (``SmplRobotConfig``).
* Structure: 24 bodies, 69 joint DOFs; +X forward, Z up.
* Observation anchor: root body ``Pelvis`` (no explicit anchor override).
* Target bodies: ``Pelvis``, ``L_Ankle``, ``R_Ankle``, ``L_Hand``,
  ``R_Hand``, ``Head``.

The config overrides PD gains and effort/velocity limits by joint family.
It uses built-in PD control by default; the experiment determines how actions
become PD targets. SMPL-X and SOMA motions are not interchangeable with this
model's joint/body arrays.

Assets And Motion Preparation
-----------------------------

SMPL-derived assets are excluded from the pip wheel. Follow the source-checkout
installation instructions and the applicable model licenses. AMASS conversion
also requires separately obtained SMPL body-model files; framework installation
does not grant access to those models or to AMASS data.

* :doc:`../getting_started/installation`: source assets and package limitations.
* :doc:`../getting_started/amass_preparation`: body-model setup and conversion
  with ``--humanoid-type smpl``.
* :doc:`../tutorials/workflows/amass_smpl`: motion-tracking training workflow.
