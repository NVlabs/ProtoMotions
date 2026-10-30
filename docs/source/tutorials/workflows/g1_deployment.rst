G1 Whole-Body Tracker: From Data to Real Robot
================================================

ProtoMotions provides a complete, fully reproducible pipeline from motion data
to real-robot deployment for the Unitree G1 humanoid.  Every step -- data
preparation, retargeting, RL training, ONNX export, MuJoCo validation, and
real-robot deployment -- uses fully open-sourced code and data with minimal
dependencies.

The design philosophy:

* **Modular and general** -- each stage is a self-contained script with clear
  inputs and outputs.
* **Minimal and readable** -- deployment code avoids heavy frameworks.  The
  reference MuJoCo test script (``deployment/test_tracker_mujoco.py``) needs
  ``mujoco``, ``onnxruntime``, ``numpy``, ``pyyaml`` and ``torch``, plus the
  ``protomotions`` package on the import path for its sensor and context
  helpers. It needs no simulator backend and no GPU.
* **Reproducible** -- the pre-trained general motion tracker was trained on the full
  `BONES-SEED <https://huggingface.co/datasets/bones-studio/seed>`_ dataset (~142K retargeted G1
  motions, see :doc:`../../getting_started/seed_g1_csv_preparation`) using
  24 A100 GPUs.  However, training with significantly fewer GPUs and a
  smaller subset of motions still produces a reasonable general whole-body tracker
  for many common motions (walking, turning, gestures, etc.).

.. note::

   A pre-trained checkpoint and its exported ONNX model are provided in
   ``data/pretrained_models/motion_tracker/g1-bones-deploy/`` so you can skip
   straight to deployment without training. The example G1 motions live
   elsewhere in the repository: single clips under
   ``data/g1-kimodo-generated/proto/`` (``output_walk.motion``,
   ``output_wave.motion``, ``output_dance.motion``, ``output_high5.motion``
   and ``output_jumpjack.motion``) and the small library
   ``data/motion_for_trackers/g1_bones_seed_mini.pt``, which the test script
   indexes with ``--motion-index``.

Pipeline Overview
-----------------

.. code-block:: text

   BONES-SEED CSVs (open-source, ~142K retargeted G1 motions)
        |
        v  data/scripts/convert_g1_csv_to_proto.py
   ProtoMotions .motion files (30 fps)
        |
        v  protomotions/train_agent.py (IsaacGym/IsaacLab, multi-GPU)
   Trained checkpoint (last.ckpt + resolved_configs.json + experiment_config.py)
        |
        v  deployment/export_to_onnx.py (CPU, no simulator)
   Unified ONNX model + YAML metadata
        |
        +---> deployment/test_tracker_mujoco.py  (reference MuJoCo test)
        |
        +---> robojudo/ integration              (MuJoCo sim + real G1)

Design Notes
----------------

**Unified ONNX model**.  The exported ONNX bundles observation computation,
the actor network, and action processing (tanh + PD offset/scale) into a
single model.  Deployment frameworks provide joint positions/velocities, the
torso orientation, the pelvis angular velocity, and future motion reference
frames -- without having to rewrite observation functions.  The torso
orientation is not an IMU reading: the default deploy path reads only the
pelvis IMU, and computes the torso orientation from it and the joint encoders
by forward kinematics (see `Sensor Requirements`_).  This minimises the chance
of discrepancy between training and deployment.

**MuJoCo-first validation**.  Before touching real hardware, we validate the
ONNX model in a standalone MuJoCo script (``deployment/test_tracker_mujoco.py``)
that uses no simulator backend and only a few small ProtoMotions modules (the
sensor and context vocabularies behind ``deployment/tracker_inputs.py`` and
``deployment/contract.py``).  This script serves as the
**deployment contract**: any framework that reproduces its behaviour will drive
the policy correctly.

**Cached 50 fps motion**.  Reference motions are resampled from their native
FPS (typically 30) to the control rate (50 Hz) using the exact same SLERP/lerp
interpolation as training, then cached to a ``.pt`` file.  Subsequent runs use
pure NumPy array indexing with no PyTorch computation.

**Heading alignment**.  The policy computes a yaw-only heading offset between
the robot's actual torso heading and the motion's heading at frame 0.  This
is always-on and handles both simulation (where the offset is near-identity)
and real hardware (where the robot may face any direction at power-on).

**RoboJuDo integration**.  We use `RoboJuDo <https://github.com/HansZ8/RoboJuDo>`_
as the deployment framework for the G1 because it is clean, readable, and
lightweight -- with a modular Policy / Environment / Controller architecture
and no heavy dependencies.  The ProtoMotions tracker integrates as a single
Policy class with no changes to RoboJuDo core.


Step 1: Prepare Motion Data
----------------------------

Follow :doc:`../../getting_started/seed_g1_csv_preparation` to convert
BONES-SEED G1 CSVs into ProtoMotions ``.motion`` files and package them into
a MotionLib ``.pt`` file.

Alternatively, use the example G1 motions already in the repository:
``data/g1-kimodo-generated/proto/output_walk.motion`` (and its siblings) or
``data/motion_for_trackers/g1_bones_seed_mini.pt``.


Step 2: Train (or use pre-trained)
-----------------------------------

**Using pre-trained checkpoint** (recommended to start):

The pre-trained model is at ``data/pretrained_models/motion_tracker/g1-bones-deploy/``.
It includes the checkpoint, resolved configs, and a pre-exported ONNX model.
Skip to Step 3.

**Training from scratch**:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaaclab \
       --experiment-path data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py \
       --experiment-name g1_bm_tracker \
       --motion-file /path/to/bones_seed_g1_motions.pt \
       --num-envs 4096 \
       --batch-size 16384

The experiment config
(``data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py``) uses
BeyondMimic-style settings with our small modifications:

* **Observations**: reduced-coords proprioception (noisy) + multi-horizon
  reduced-coords target poses (steps [1, 2, 4, 8]) + previous processed
  actions.  Clean (noise-free) counterparts are added for L2C2 smoothness
  regularization.
* **Rewards**: BeyondMimic heading-invariant relative body tracking
  (position + orientation + linear/angular velocity, region-weighted) plus
  global anchor orientation, action rate penalty, and soft joint limit
  penalty.
* **Action processing**: BeyondMimic PD action config
  (``make_bm_pd_action_config``).
* **Domain randomization**: observation noise, friction randomization,
  center-of-mass randomization, push perturbations.


Step 3: Export ONNX
--------------------

Export the policy to a unified ONNX model.  This needs PyTorch, the
ProtoMotions package and ``onnxruntime`` -- no simulator or GPU.
``onnxruntime`` is not optional: right after writing the graph, the script
opens it to read back the input names for the YAML sidecar, so without it the
run stops after the ``.onnx`` is written and no YAML is produced
(``--no-validate`` only skips the later validation pass).

.. code-block:: bash

   python deployment/export_to_onnx.py \
       --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/inference_last.ckpt \
       --output /path/to/export_dir

This produces, in the ``--output`` directory:

* ``unified_pipeline.onnx`` -- the ONNX model
* ``unified_pipeline.yaml`` -- rich metadata documenting every
  input/output, timing, conventions, and frame requirements

Without ``--output`` the files land next to the checkpoint, in
``<checkpoint_dir>/compiled_models/``. For the shipped bundle that is
``data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/``,
which would overwrite the pre-exported pair described below, so pass
``--output`` to keep them.

The default ``--host robot`` is right for this tracker: its observations read
only the joint encoders, the IMU, its own past processed actions and the
reference motion, which a robot can feed. A policy whose observations read the simulator's own body state (the
shipped SOMA tracker, for example) is refused with the default and exports
with ``--host simulation``, for a game engine, a browser MuJoCo or a
sim-to-sim check; the sidecar then records ``host: simulation``.

For a model you trained yourself, point ``--checkpoint`` at
``results/<experiment>/last.ckpt``; it exports fine. The slimmer
``inference_last.ckpt`` (weights only, no optimizer state) is written only
when the recipe sets ``save_inference_checkpoint=True`` in its agent config,
which the shipped deploy recipe does not.

The YAML metadata acts as a machine-readable deployment contract.  It includes
per-input documentation of expected shapes, source expressions, and -- critically
-- the reference frame convention for angular velocity inputs (see
:ref:`g1-deploy-frame-convention` below).

.. note::

   A pre-exported ONNX model is already included at
   ``data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx``.


Step 4: Test in MuJoCo
------------------------

The reference MuJoCo test script validates the ONNX model with minimal
dependencies.  It is the **deployment contract** -- any framework that matches
its behaviour will drive the policy correctly.

.. code-block:: bash

   # First run: resample motion to 50fps and cache it next to the clip
   python deployment/test_tracker_mujoco.py \
       --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
       --motion data/g1-kimodo-generated/proto/output_walk.motion \
       --cache-motion --render

   # Subsequent runs use the cached motion (skips the SLERP resampling step)
   python deployment/test_tracker_mujoco.py \
       --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
       --motion data/g1-kimodo-generated/proto/output_walk.50fps.pt \
       --render

Additional test mode:

.. code-block:: bash

   # Headless benchmark
   python deployment/test_tracker_mujoco.py \
       --onnx ... --motion ... --no-realtime

The script handles: MJCF loading/patching, PD configuration, heading alignment,
acceleration clamping, and real-time pacing. There is no EMA filter: the
sidecar's ``action_ema_alpha`` is always ``1.0`` (no filtering) and the
exporter cannot set anything else.


Step 5: Deploy via RoboJuDo (Simulation)
-----------------------------------------

`RoboJuDo <https://github.com/HansZ8/RoboJuDo>`_ provides a plug-and-play
deploy framework for humanoid robots with MuJoCo simulation and Unitree real
robot support.  ProtoMotions BM tracker support is upstreamed -- no patch is
required.

**Setup**:

.. code-block:: bash

   git clone https://github.com/HansZ8/RoboJuDo.git
   cd RoboJuDo
   git checkout release   # contains the ProtoMotions tracker integration
   pip install -e .
   git lfs pull           # download mesh assets

**Run the tracker in MuJoCo simulation**:

The tracker has its own launcher. The generic ``scripts/run_pipeline.py``
accepts only ``-c``; ``scripts/run_tracker_pipeline.py`` adds the ONNX and
motion flags.

.. code-block:: bash

   python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker \
       --onnx-path /path/to/unified_pipeline.onnx \
       --motion-path /path/to/motion.motion

The upstream RoboJuDo integration (``release`` branch) provides:

* ``ProtoMotionsTrackerPolicy`` -- loads ONNX + cached 50fps motion,
  provides PD targets with action history feedback, heading alignment,
  and motion fade-in/fade-out support.
* Virtual gantry -- spring-damper safety harness for real-robot deployment
  transitions.
* Blend-in/blend-out -- smooth transitions between idle pose and policy
  control.
* ``scripts/run_tracker_pipeline.py`` with ``-c``, ``--onnx-path``,
  ``--motion-path`` and ``--motion-index`` (an index into a multi-motion
  ``.pt`` library). The ``release`` branch has no other tracker flags.

.. note::

   The ``release`` branch's tracker policy feeds the graph only the older
   ``historical.processed_actions`` input and silently skips any graph input
   it does not know. The ONNX shipped in ``g1-bones-deploy`` was re-exported
   from the current contract and requires ``processed_action_history`` (its
   YAML says ``output_key: joint_pos_targets``, ``steps_back: [1]`` and
   ``fill_before_available: zeros``), so passing it with ``--onnx-path``
   fails in onnxruntime with a missing required input. Until the integration
   feeds that input, run the shipped model with
   ``deployment/test_tracker_mujoco.py``, or run RoboJuDo with its own
   bundled copy of the tracker (``assets/models/g1/protomotions_tracker/``),
   an older export whose YAML still names ``historical.processed_actions``
   with ``output_key: robot_action``. One known deviation there: RoboJuDo
   records the fed-back targets after its acceleration clamp rather than as
   returned (contract item 2 below), so expect its joint targets to be
   somewhat jumpier than ``deployment/test_tracker_mujoco.py``'s; tracking
   itself is barely affected.

**Bringing your own deployment framework**: The key contract is:

1. Provide the ONNX inputs documented in the YAML sidecar (joint
   positions/velocities, torso orientation, pelvis local angular velocity,
   previous processed actions, future motion references).
2. Feed each action-history input the graph's own output from earlier calls,
   recorded as returned, before any clamp or smoothing: ``joint_pos_targets``
   for ``processed_action_history`` (or the older
   ``historical.processed_actions``), ``actions`` for ``raw_action_history``.
   The YAML entry names that output (``output_key``), says how far back each
   slot reaches (``steps_back``: ``[1]`` is the previous call, ``[2]`` the one
   before) and what to feed until that many calls have run since a reset
   (``fill_before_available``). The bundled G1 YAML carries all three
   (``output_key: joint_pos_targets``, ``steps_back: [1]``,
   ``fill_before_available: zeros``). Only older exports, such as RoboJuDo's
   own bundled copy, say ``output_key: robot_action``, which means
   ``joint_pos_targets``. Without ``steps_back``, use ``[1]`` for the two
   current inputs and ``[2]`` for ``historical.processed_actions``, with a zero
   fill. ``ActionFeedback`` in ``deployment/tracker_inputs.py`` does all of
   this.
3. Apply the ONNX outputs (PD position targets) to your robot's actuators.
4. Apply the acceleration clamp documented in the YAML ``control`` section
   (``pd_target_max_accel``; ``null`` means none). ``action_ema_alpha`` is
   always ``1.0`` there, so there is no EMA to apply.
5. Verify that MuJoCo behaviour matches between your framework and
   ``deployment/test_tracker_mujoco.py``.


Step 6: Deploy on Real G1 Robot
--------------------------------

.. warning::

   Before deploying on the real robot, read the safety disclaimers in the
   `RoboJuDo README <https://github.com/HansZ8/RoboJuDo#alert--disclaimer-%EF%B8%8F%EF%B8%8F%EF%B8%8F>`_.
   Always verify that the emergency stop works.  Policies can cause violent
   motions when losing balance.

For general real-robot setup (Ethernet connection, firewall configuration,
Unitree SDK installation), refer to the
`RoboJuDo real robot guide <https://github.com/HansZ8/RoboJuDo#run-robojudo-on-real-robot->`_.

Once the robot is connected, run:

.. code-block:: bash

   cd RoboJuDo
   python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker_real \
       --onnx-path /path/to/unified_pipeline.onnx \
       --motion-path /path/to/motion.motion

The ``g1_protomotions_tracker_real`` config extends the simulation config
with the Unitree real robot environment (``UnitreeCppEnv``) and safety checks
enabled; set its ``net_if`` to your network interface.

Safety
^^^^^^

Press the **A button** on the Unitree controller at any time to
**emergency stop** the robot.  Always keep the controller in hand and verify
that the emergency stop works before starting a deployment session.

Deployment Phases
^^^^^^^^^^^^^^^^^

Real-robot deployment uses four phases to safely transition between idle
and policy control.  Each phase is necessary -- skipping or rushing them
risks sudden joint movements that can damage the robot or the environment.

**Phase 1: Ramp-up.**
The joints slowly interpolate from the robot's current pose to the first
frame of the motion clip.  The robot should be held in a gantry during this
phase.  Wait for the transition to complete before proceeding.

**Phase 2: Blend-in.**
The policy is activated but receives only the first frame as its target,
telling it to hold that pose autonomously.  Control is passed linearly from
the static pose to the policy over several seconds.  During this phase the
operator should slowly lower the robot in the gantry until it is standing
on the ground under its own balance.  By the end of Phase 2 the robot is
fully autonomous and the gantry can be released.

**Phase 3: Tracking.**
The policy receives the full motion and begins reconstructing it.  The robot
will perform the motion clip from start to finish.

**Phase 4: Blend-out.**
When the motion ends or the operator triggers a fade-out from the Unitree
controller, the process reverses: PD targets blend from policy output back
to the init pose.  The robot smoothly returns to a stable standing pose.
The operator should raise the gantry back into supporting position during
this phase so the robot is fully supported by the time blend-out completes.
After blend-out the pipeline can loop back to blend-in for the next motion,
or the session can be ended.

Sensor Requirements
^^^^^^^^^^^^^^^^^^^

The ProtoMotions tracker policy is compatible with the real G1 robot.
All ONNX inputs map to sensors available on the physical G1:

* ``dof_pos`` / ``dof_vel`` -- from joint encoders
* ``anchor_rot`` (torso) -- computed via forward kinematics from pelvis IMU
  + joint encoders
* ``root_local_ang_vel`` (pelvis) -- directly from pelvis IMU gyroscope
  (already in body-local frame)
* Future motion references -- from the cached motion file

Domain randomization during training (observation noise, friction
randomization, push perturbations) provides robustness to real-world sensor
noise and dynamics differences.


.. _g1-deploy-frame-convention:

Important: Angular Velocity Frame Convention
---------------------------------------------

The ONNX model expects ``root_local_ang_vel`` in the **pelvis body's local
frame**.  Different sources provide angular velocity in different frames:

.. list-table::
   :header-rows: 1

   * - Source
     - Frame
     - What to do
   * - MuJoCo ``data.cvel[body+1, 0:3]``
     - World
     - Apply ``quat_rotate_inverse(pelvis_rot, ang_vel)``
   * - MuJoCo ``data.qvel[3:6]`` (free joint)
     - Local
     - Use directly -- no rotation
   * - Real robot IMU gyroscope
     - Local
     - Use directly -- no rotation