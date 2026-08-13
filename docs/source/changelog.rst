ProtoMotions Changelog
======================

Unreleased
----------

Record changes for the next release in this section. Move completed entries
into a dated release section when the release is published.

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
