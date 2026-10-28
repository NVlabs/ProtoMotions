ProtoMotions Changelog
======================

Unreleased
----------

Record changes for the next release in this section. Move completed entries
into a dated release section when the release is published.

October 28, 2026
----------------

The following changes are included in this release:

Simulators and Physics
~~~~~~~~~~~~~~~~~~~~~~

* Remove the Genesis backend, its ``genesis`` package extra and
  ``requirements_genesis.txt``; IsaacGym, IsaacLab, Newton and MuJoCo remain.
* Clear Newton/MuJoCo-Warp solver warm starts only for reset environments,
  and snapshot/restore them across temporary evaluation. This prevents a new
  episode or restored training state from inheriting evaluation's solver history.
* Add the IsaacLab ``gpu_collision_stack_size`` setting for dense contact
  scenes.
* Unify privileged contact reporting behind one contact matrix with selectable
  bodies and partners, which self-contact features read as well. See
  :doc:`concepts/contact_matrix` and :doc:`concepts/self_contacts`.

Environment and Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Robot files declare one control rate (``control_hz``), per-backend
  simulation parameter deltas and motor gain classes from a shared registry.
* Make ``BaseEnv.step()`` and ``reset()`` explicit phase lists over owner
  objects that publish a flat context map; observation, reward, termination
  and metric components bind from their kernels and the component factory
  module is removed, so recipes that imported it must be ported.
* Replace observation noise with sensors declared once per channel, each with
  measurement noise and a state-estimation stage; the simulator config's
  ``observation_noise`` fields are removed.
* Terminations and evaluation failures share one rule type, with new foot
  tracking and low-pose rules.
* Replace the pickled resolved configs and their YAML sidecars with one JSON
  config bundle; inference derives its view from it at load rather than from a
  separate saved inference bundle. Regenerate saved configs.
* Remove the inference ``--command-source`` flag in favour of a
  ``env.control_components.<name>.command_source`` override, and add
  ``--inference-config`` for a Python file whose ``apply_inference_overrides()``
  applies to the derived inference view. See :doc:`user_guide/native_goals`.
* Recipes list observation keys once and PPO, AMP and ASE models derive the
  input union; Ruff now enforces F401 in ``examples/experiments``.

Training and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

* Replace PPO's ``learnable_std`` with ``std_mode`` (default fixed), bounded
  log-std, a std-aware KL term, a deterministic mean path for teachers and
  export, and an opt-in shared per-motion exploration table.
* Separate evaluator measurements from threshold rules. Experiments
  now use ``metric_components``, ``failure_components`` and ``score_key``;
  thresholds support any-frame, sum, mean, min, max and final-frame reductions.
  Regenerate old resolved configs; there is no automatic compatibility path.
  See :doc:`user_guide/evaluation` for the Python API migration.
* Keep offline evaluation from updating motion sampling; training updates only
  evaluated motions. Handle empty and heterogeneous evaluation ranks, export only
  recorded trajectory frames and restore rollout state on evaluation errors.
* Share one evaluator rollout and metric manager. MaskedMimic uses posterior
  actions for training evaluation and prior-only actions for inference. Its
  training score is a different benchmark from prior-based evaluation.
* Stop collecting a trial after evaluation failure and reset its environment
  before another action. End each batch when no scored trials remain, without
  skipping later motion batches. Diagnostic averages now cover frames up to
  failure instead of the full window; configured failure thresholds are unchanged.
* Add the Metatron G1 and SOMA teacher and student recipes with native
  interactive goals, and a deployable G1 observation set for MaskedMimic.
  See :doc:`user_guide/metatron_student` and :doc:`user_guide/native_goals`.

Models and Deployment
~~~~~~~~~~~~~~~~~~~~~

* Move pretrained bundles to a self-contained layout: a model-only
  ``inference_last.ckpt`` with ``resolved_configs.json``, ``config.yaml``,
  ``experiment_config.py`` and ``provenance.json``; the full ``last.ckpt`` and
  pickled configs are removed and the bundles no longer resume training.
  See :doc:`getting_started/pretrained_models`.
* Remove the SMPL MaskedMimic, SMPL flat and terrain trackers, the SOMA FSQ
  AMP tracker and the SOMA BONES compiled ONNX; add the Metatron v1 G1 package
  as a verified Hugging Face download via ``scripts/download_metatron.py``.
* Re-export the G1 BONES deployment ONNX through one exporter,
  ``deployment/export_to_onnx.py``, which replaces the two tracker-specific
  export scripts; add the experimental ``deployment/native/`` standalone
  runtime.
* Add the compiled Metatron browser demo and G1 Racer under ``demos/`` with
  installer scripts that fetch their weights from Hugging Face. See
  :doc:`user_guide/browser_demo` and :doc:`user_guide/g1_racer`.
* Ship the compiled browser demo's policy as a small graph plus external
  weights hosted on Hugging Face (the same FP32 bytes as the single file, in
  a separate file), so phones no longer run out of memory while the model
  loads: the browser copies the weights once, prefers WebGPU on phones with a
  single-thread WASM fallback, and stops after two memory-reload cycles
  instead of downloading again. Desktop runtime selection and speed are
  unchanged. See :doc:`user_guide/browser_demo`.
* Native deployment: reject motion indices that would overflow the motion
  library and refuse a symlinked Pico input lock file.

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
  authoritative resolved-config sidecars.
* Improve ONNX deployment metadata and the MuJoCo tracker harness, including
  motion-index selection and exported effort limits.
* Improve pretrained-policy inference compatibility between MuJoCo and Newton.
