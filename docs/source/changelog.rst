ProtoMotions Changelog
======================

Unreleased
----------

Record changes for the next release in this section. Move completed entries
into a dated release section when the release is published.

October 31, 2026
----------------

The following changes are included in this release:

Demos and Documentation
~~~~~~~~~~~~~~~~~~~~~~~

* **G1 Football** is released: five penalty shots against a simulated Unitree
  G1 goalkeeper driven by Metatron, with MuJoCo physics, the keeper's rendered
  RGB-D camera and the supplied ball detector, tracker and commit running in
  the browser (``demos/g1-football/``, installed beside the documentation by
  ``scripts/install_g1_football.py``; user guide page ``user_guide/g1_football``).
  The keeper sees only its camera. Wide low or mid shots (a camera-derived
  crossing at least 1.2 m to the side) trigger a goalkeeper dive that shuffles
  and drops toward the ball line as early as the body allows, holds there until
  the ball arrives and hands over to the native recovery; nearer shots keep the
  timed save. The dive follows a reference clip derived from one BONES-SEED
  motion, credited inside ``keeper-dive-clip.json`` and in ``NOTICE.txt``.
  Over 24 paired kicks (8 kick types, 3 camera-noise seeds, identical seeds
  for both keepers) the dive saved 21 where the timed save made 16. High
  corners are not dived at (no jumping reference); getting up after a dive is
  slower than after the timed save.
* The football page keeps the sensor's head mount and pose for the replay but
  opens the lens to 100 degrees so the keeper's own arms are visible; the
  keeper-vision panel keeps the calibrated projection the detector saw. After a
  decided shot, camera frames past the replayable window are not analysed
  (the supplied detector took seconds per frame with the ball hidden behind
  the keeper's body), and a request-scoped watchdog reports a camera frame
  that never returns instead of freezing.
* Phone layout: the portrait canvas margins, the start panel and the keeper
  panel no longer overlap the pitch or the thumb controls; a charge pressed
  while the keeper camera is still preparing the next shot is accepted and its
  release is queued until the game is ready. Chromium touch emulation passes in
  both orientations; physical iPhone/Safari devices are not qualified.
* The Racer browser check records bounded failure diagnostics and waits for
  acknowledged respawn frames before judging a run.

October 29, 2026
----------------

The following changes are included in this release:

Deployment
~~~~~~~~~~

* ``matrix_to_quaternion`` is ONNX-static: it gathers on the argmax and selects
  with ``where`` instead of boolean-mask writes, so the FK subgraph a native host
  compiles fits a static TensorRT plan.

* Scene boxes are an exportable input stream. A scene-conditioned policy (the
  Metatron G1 release) binds the eight ``scene_boxes.*`` names: box centres and
  extents in the object frame, object poses, per-box validity, per-object
  visibility and the observed-pose bias. ``SceneBoxesControl``
  (``protomotions/envs/control/scene_boxes_control.py``) publishes them in the
  simulator as an all-padding scene of a fixed capacity, so one saved
  configuration both runs and exports; it reads nothing from the scene
  library (scene-box training is not supported, and a scene library with
  objects is refused). The exporter labels them with the new ``scene``
  provider, and the recipe helper ``deployable_scene_box_observations``
  (``examples/experiments/masked_mimic/deployable_g1_observations.py``) binds
  the two observations that compute the box corners in the root heading frame
  inside the graph, with the root pose from the recipe's FK kernel, so a
  configuration that binds it exports the eight ``scene_boxes_*`` inputs and
  no body pose. ``configure_observations(robot_cfg, env_cfg, scene_boxes=True)``
  binds the pair to the recipe's one FK kernel, and the pinned Metatron
  package does.
* The pinned Metatron G1 package (``data/pretrained_models/metatron/
  huggingface.json``) moves to the Hub revision whose saved configuration
  binds the FK goal observation to the raw joint readings and the anchor pose
  and reads the scene from the host: ``SceneBoxesControl`` publishes the empty
  scene in the simulator and the two scene observations replace the dummy
  ones. ``deployment/export_to_onnx.py`` exports the downloaded package with
  37 inputs, the eight ``scene_boxes_*`` inputs among them and no body pose
  (the dummy-sized ``current_dof_pos`` is gone), and
  ``deployment/export_fk_to_onnx.py`` exports its ``fk.onnx``. The weights
  are unchanged; the new graph matches the maintainer's 39-input graph on
  scene cases to ``1.1e-5``. The dummy scene functions in
  ``examples/inference/metatron.py`` are removed.
  ``scripts/download_metatron.py`` does not overwrite a different local
  package: move an older ``G1/`` directory aside before downloading.
* Move the forward kinematics of the robot's own body into the exported
  graph. The deployable Metatron goal observation
  (``fk_masked_mimic_target_poses``) binds a ``BakedAnchorFK`` kernel to the
  joint readings and the anchor pose instead of reading reconstructed body
  poses; the policy graph keeps its four control outputs (``actions``,
  ``joint_pos_targets``, ``stiffness_targets``, ``damping_targets``) and
  every output is a command. The ``body_pos``/``body_rot`` host input kinds
  and the ``derived_state_provider`` sensor metadata are removed: a robot host
  feeds raw readings, its past actions and goals only.
* Add ``deployment/export_fk_to_onnx.py``, which exports the same FK kernel
  the policy traces as a second small graph, ``fk.onnx`` with an ``fk.yaml``
  sidecar, for a host that needs the body poses outside the policy call (the
  native Metatron runtime runs FK on the current measurement before every
  call). Its inputs carry the policy graph's names for the same paths
  (``noisy_dof_pos``, ``noisy_anchor_pos``, ``noisy_anchor_rot``) with the
  batch fixed to one robot; it returns ``body_pos`` and ``body_rot`` in robot
  body order and is checked against the kernel with ONNX Runtime at export.
  The previous standalone FK exporter under ``deployment/native/tools``, which
  re-ran pose_lib's forward kinematics with its own quaternion conversion, is
  deleted, so there is one FK implementation and the two graphs agree by
  construction. ``build_metatron_fk_tensorrt.py`` compiles ``fk.onnx`` to the
  TensorRT plan and, given the policy manifest, checks the sidecar's body
  order, joint order and anchor against it; the C++ engine binds joints plus
  the torso anchor pose.
* ``deployment/export_to_onnx.py --host robot|simulation`` names who feeds the
  exported graph. The default robot host is the sensor definition, the
  policy's past actions and the goal streams, as before; a simulation host (a
  game engine, a browser MuJoCo, a sim-to-sim check) may also feed the
  simulator's own body state, so ``current.rigid_body_pos``/``rot``/``vel``/
  ``ang_vel`` and ``current.ground_heights`` now carry the ``simulated_*``
  host kinds and ``mimic.future_vel``/``future_ang_vel`` the
  ``reference_motion_body_vel``/``_ang_vel`` kinds. A robot export refuses a
  simulation-only binding, naming the kind and the host that could feed it;
  a simulation sidecar records ``host: simulation`` (omitted means robot, so
  robot sidecars are unchanged). The shipped SOMA tracker, whose observations
  read the simulator's body state, exports only with ``--host simulation``.
* ``deployment/export_to_onnx.py --leapp-graph-history`` writes a second LEAPP
  bundle in which the graph owns the declared sensor and action histories as
  LEAPP feedback state, so every LEAPP runtime runs the same history update
  instead of replaying ``InputHistory`` on the host. The history inputs leave
  the interface; per tick the host supplies the measurements and goals as
  before, plus ``applied_actions`` (the raw action actually applied on the
  previous tick) and ``command_accepted`` (false when that command was
  dropped). The export needs ``PYTORCH_JIT=0``; the bundle has batch size 1
  and no ``deployment_contract.yaml``. The existing ``--leapp`` bundle is
  unchanged. See ``deployment/native/docs/leapp.md``.

Training and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

* Fix per-motion exploration (``std_mode=per_motion``): the agent injected
  ``global_motion_id`` into the observations and the model also declared it as a
  stored output, so the experience buffer registered the key twice and the first
  rollout of every per-motion run failed. Keys already registered from the
  observations are dropped from the model-output list.
* ``load_experiment_module`` prefers the running tree over the project root
  found beside a recipe copy, so a run directory saved inside another checkout
  no longer imports that checkout's code.

Examples, Tutorials and Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Every documented command was run or checked against the tree, and the ones
  that could not run as written now can: tutorial 0 declares the G1 joint
  limits its hand-built robot config needs, tutorial 7 drops the PPO keys the
  agent refuses and creates its output directory, and
  ``random_pose_visualizer.py`` accepts every registered robot.
* Add ``examples/experiments/mimic/mlp_no_contact.py`` and
  ``mlp_complex_terrain_no_contact.py``: the mimic recipes without contact
  bodies, for IsaacGym (one combined force per body) and motion files without
  contact labels.
* Remove the dead ``examples/benchmark`` scripts, the stale SMPL viser viewer
  (``examples/visualize_smpl_motion.py``) and the IsaacLab 2 duplicate of the
  MJCF-to-USD converter under ``data/scripts``.
* Drop the unused ``openmesh`` pin: nothing imports it, and it had no wheel for
  Python 3.10 or newer.

Demos and Documentation
~~~~~~~~~~~~~~~~~~~~~~~

* G1 Racer: a service worker adds the cross-origin isolation headers on static
  hosts so the CPU path gets its WASM threads (one bounded reload per tab;
  ``?threads=1`` opts out). WebGPU selection is unchanged.
* README: a News section, Generative Policies first with new clips (GPC
  parkour, G1 Racer, Metatron on the real G1 and on furniture), the Metatron
  paper and project links, and every documentation link pointing at the
  published site.

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
