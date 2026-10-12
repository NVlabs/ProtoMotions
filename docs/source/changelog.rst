ProtoMotions Changelog
======================

Unreleased
----------

Record changes for the next release in this section. Move completed entries
into a dated release section when the release is published.

* Upgrade Newton from 1.0.0 to 1.6.0 (``warp-lang`` 1.12.0 -> 1.17.0, ``mujoco``/
  ``mujoco-warp`` 3.5.0 -> 3.12.0; the ``newton-actuators`` package is folded
  into ``newton.actuators``). Follows upstream renames/removals: domain
  randomization now notifies via ``newton.ModelFlags`` (``SolverNotifyFlags``
  was removed in Newton 1.5), contact sensors use the ``total_force`` /
  ``sensing_indices`` attributes (``net_force`` / ``sensing_objs`` were removed
  in Newton 1.3), the built-in PD controller writes ``joint_target_q`` instead
  of the removed ``joint_target_pos`` alias, and ``SolverMuJoCo`` no longer
  takes an ``ls_parallel`` argument (``NewtonSimParams.ls_parallel`` is kept as
  a no-op field only so old ``resolved_configs.yaml`` checkpoints still load).
  Also zeroes ``Model.joint_damping`` directly (MJCF passive damping moved off
  ``Model.mujoco.dof_passive_damping``, which was removed in Newton 1.6, onto
  this core attribute) — affects ``smpl``/``smplx``/``amp``/``h1_2`` and the
  ``g1_bm*`` assets, which author nonzero MJCF joint damping. Verified that
  Newton 1.6.0's default ``newton.use_coord_layout_targets = True`` is a
  behavior-preserving no-op for ProtoMotions (all joint-target reads/writes go
  through ``ArticulationView``, never raw array indexing), so no pin is needed.
* Replace the post-construction ``solver.mjw_model.geom_margin``/``geom_gap``
  override in ``_setup_sim`` — which never actually affected simulated contact
  behavior under the default ``use_mujoco_contacts=True`` path on either
  Newton 1.0.0 or 1.6.0 (native MuJoCo contact resolution reads margin/gap
  from the compiled ``mujoco.MjModel``, a separate structure from
  ``mjw_model``) — with ``ModelBuilder.rigid_gap = 0.01``, set on both the
  per-robot and replicated builders before ``finalize()``. Confirmed none of
  the bundled MJCF/URDF assets author an explicit per-shape ``gap``, so this
  applies uniformly (``Model.shape_gap`` is a single value across every shape
  for both ``g1`` and ``smpl``).
* Add ``NewtonSimParams.use_mujoco_contacts`` (default ``True``) to let
  ``SolverMuJoCo`` use Newton's own contact pipeline instead of MuJoCo's.
  Setting it to ``False`` requires contacts to be generated externally each
  substep (``newton.CollisionPipeline.collide()``) and fed into
  ``solver.step()``; ``_create_contacts`` now builds an explicit
  ``CollisionPipeline`` (replacing the deprecated ``Model.contacts()``) and
  ``_simulate`` calls ``collision_pipeline.collide()`` once per substep,
  gated on ``not use_mujoco_contacts`` so the default path doesn't pay for an
  unused collision pass. Without the ``collide()`` call, setting
  ``use_mujoco_contacts=False`` silently fed the solver an always-empty
  contacts buffer: no exception or warning, just a robot falling through the
  ground floor every time.
* Bump ``dm-control`` 1.0.37 -> 1.0.45. Newton 1.6.0's ``mujoco`` 3.12.0 removed
  ``MjData.qM``, which ``dm_control`` 1.0.37 (and ``requirements_newton.txt``'s
  prior pin) still accessed, breaking ``pose_lib.py``'s standalone
  ``dm_control``-based MJCF FK self-check
  (``python -m protomotions.components.pose_lib``) with
  ``AttributeError: 'MjData' object has no attribute 'qM'``. 1.0.45 fixes it;
  verified all of ``test_pose_lib_helpers.py`` passes again.

August 12, 2026
---------------

The following changes are included in this release:

Task Control and Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Add configurable target commands, semantic heading axes, recovery-pose
  resets, and a minimum motion-weight floor for reliable task sampling.
* Add soft joint-limit and end-effector position rewards, with safe termination
  when joint state diverges.
* Add baked-FK target observations, odometry corruption/ablation paths, and
  optional reference-travel feedforward for deployment-oriented tracking.

Simulator and Physics
~~~~~~~~~~~~~~~~~~~~~

* Switch to Newton 1.0.0 at public commit ``e7a737c`` and IsaacLab 3.0 /
  Isaac Sim 6.0 at IsaacLab commit
  ``4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8``
  (``v3.0.0-beta-712-g4ecd0b``).
* Make MJCF the single source of truth for robot assets. IsaacLab converts
  packaged MJCF at load time and preserves multi-axis joint structure and
  metadata.
* Fix friction randomization and combine-mode conversion across Newton,
  IsaacGym, and IsaacLab, including partial ranges and shared bucketing.
* Improve Newton distributed-device selection, joint-name diagnostics, and
  evaluation handling; align character friction defaults across backends.
* Improve scene-object state handling and quaternion, reset, and single-env
  simulator state behavior.

Training and Packaging
~~~~~~~~~~~~~~~~~~~~~~

* Add configurable Weights & Biases project names, training iteration limits,
  and sharded MotionLib loading with runtime shard switching for multi-GPU
  datasets.
* Add PyPI and UV installation support while preserving preconfigured Torch
  builds.
* Improve IsaacLab throughput by collapsing actuator configuration and reducing
  terrain, motion-reference, and subset-reset hot-path overhead.
* Improve quickstart and workflow documentation for pretrained models, G1
  deployment, and SLURM training.

Models and Deployment
~~~~~~~~~~~~~~~~~~~~~

* Refresh G1 BONES-SEED, SMPL, SOMA, and MaskedMimic pretrained artifacts with
  authoritative ``resolved_configs.pt`` sidecars.
* Improve ONNX deployment metadata and the MuJoCo tracker harness, including
  motion-index selection and exported effort limits.
* Improve pretrained-policy inference compatibility between MuJoCo and Newton.
