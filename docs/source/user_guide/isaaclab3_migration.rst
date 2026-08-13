IsaacLab 3 Migration Pin
========================

ProtoMotions' IsaacLab backend targets the public IsaacLab ``develop`` line that
includes the nested rigid-body contact-sensor fix from issue `#5085`_ (merged in
PR `#6259`_, merge commit ``17136c6``).

Pinned revision
---------------

* Public IsaacLab commit: ``4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8``
* Resolved: 2026-07-16
* Guarantees: newer than ``17136c6`` / PR `#6259`

Humanoid assets
---------------

Humanoid robot configs declare only MJCF via ``RobotAssetConfig.asset_file_name``.
At IsaacLab scene construction, ``convert_robot_mjcf_to_usd()`` invokes IsaacLab
3 ``MjcfConverterCfg`` / ``MjcfConverter`` and spawns through ``UsdFileCfg``.
Contact-sensor prim paths are resolved from the converted USD hierarchy
(flat or nested). Non-IsaacLab simulators continue to load the same MJCF files.

Temporary MJCF importer compatibility
--------------------------------------

The IsaacLab 3 converter currently collapses multiple single-axis MJCF joints
between the same bodies into a PhysX D6 joint. ProtoMotions installs a narrow
compatibility repair at first conversion that preserves the shared source
frame, restores MuJoCo spring gains, and locks unused D6 axes. IsaacLab still
owns the conversion; the repair is removed once the upstream importer includes
the corresponding fix. The converter cache includes the repair version, so a
USD generated before an update is not reused silently.

For SOMA and SMPL, IsaacLab exposes the collapsed axes with ``:0``, ``:1``, and
``:2`` suffixes. ProtoMotions maps those backend names back to the semantic MJCF
DOF names before configuring actuators and simulator state.

Kit-free dry run
----------------

Unit tests inject a converter factory or set
``PROTOMOTIONS_ISAACLAB_MJCF_DRY_RUN=1`` to exercise path/config/caching without
starting Kit.

Headless smoke (requires IsaacLab 3 + Kit)
------------------------------------------

Run in an IsaacLab 3 environment checked out at the pin above::

   python protomotions/inference_agent.py \\
     --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \\
     --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \\
     --simulator isaaclab --num-envs 1 --headless

That path converts the robot MJCF, builds the articulation/contact sensors, and
steps physics. In the local verification, Kit reached scene setup and policy
materialization before a ten-minute bound expired during cold Torch warm-up.
The direct simulator tutorial (``examples/tutorial/0_create_simulator.py``)
using the same pinned runtime initialized, reset, stepped, and read state for
more than 3,800 action/physics cycles before its 180-second guard expired. Allow
more time on a cold machine for the first policy compilation.

.. _#5085: https://github.com/isaac-sim/IsaacLab/issues/5085
.. _#6259: https://github.com/isaac-sim/IsaacLab/pull/6259
