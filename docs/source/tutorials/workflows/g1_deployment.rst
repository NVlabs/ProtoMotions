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
  reference MuJoCo test script (``deployment/test_tracker_mujoco.py``) runs
  with only ``mujoco``, ``onnxruntime``, ``numpy``, and ``pyyaml``.
* **Reproducible** -- the pre-trained general motion tracker was trained on the full
  `BONES-SEED <https://huggingface.co/datasets/bones-studio/seed>`_ dataset (~142K retargeted G1
  motions, see :doc:`../../getting_started/seed_g1_csv_preparation`) using
  24 A100 GPUs.  However, training with significantly fewer GPUs and a
  smaller subset of motions still produces a reasonable general whole-body tracker
  for many common motions (walking, turning, gestures, etc.).

.. note::

   Pre-trained checkpoints, ONNX models, and example motions are provided in
   ``data/pretrained_models/motion_tracker/g1-bones-deploy/`` so you can skip straight to
   deployment without training.

Pipeline Overview
-----------------

.. code-block:: text

   BONES-SEED CSVs (open-source, ~142K retargeted G1 motions)
        |
        v  data/scripts/convert_g1_csv_to_proto.py
   ProtoMotions .motion files (30 fps)
        |
        v  protomotions/train_agent.py (IsaacGym/IsaacLab, multi-GPU)
   Trained checkpoint (last.ckpt + resolved_configs_inference.pt)
        |
        v  deployment/export_bm_tracker_onnx.py (CPU, no simulator)
   Unified ONNX model + YAML metadata
        |
        +---> deployment/test_tracker_mujoco.py  (reference MuJoCo test)
        |
        +---> robojudo/ integration              (MuJoCo sim + real G1)

Design Notes
----------------

**Unified ONNX model**.  The exported ONNX bundles observation computation,
the actor network, and action processing (tanh + PD offset/scale) into a
single model.  Deployment frameworks provide raw sensor signals -- joint
positions/velocities, torso IMU orientation, pelvis angular velocity, and
future motion reference frames -- without having to rewrite observation
functions.  This minimises the chance of discrepancy between training and
deployment.

**MuJoCo-first validation**.  Before touching real hardware, we validate the
ONNX model in a standalone MuJoCo script (``deployment/test_tracker_mujoco.py``)
that has near-zero dependency on ProtoMotions.  This script serves as the
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
Policy class (~230 lines) with no changes to RoboJuDo core.


Step 1: Prepare Motion Data
----------------------------

Follow :doc:`../../getting_started/seed_g1_csv_preparation` to convert
BONES-SEED G1 CSVs into ProtoMotions ``.motion`` files and package them into
a MotionLib ``.pt`` file.

Alternatively, use the example motions provided with the pre-trained model.


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

Export the policy to a unified ONNX model.  This requires only PyTorch and
the ProtoMotions package -- no simulator or GPU needed.

.. code-block:: bash

   python deployment/export_bm_tracker_onnx.py \
       --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt

This produces:

* ``compiled_models/unified_pipeline.onnx`` -- the ONNX model
* ``compiled_models/unified_pipeline.yaml`` -- rich metadata documenting every
  input/output, timing, conventions, and frame requirements

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

   # First run: resample motion to 50fps and cache
   python deployment/test_tracker_mujoco.py \
       --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
       --motion /path/to/walk.motion \
       --cache-motion --render

   # Subsequent runs use the cached motion (no protomotions import needed)
   python deployment/test_tracker_mujoco.py \
       --onnx data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
       --motion /path/to/walk.50fps.pt \
       --render

Additional test modes:

.. code-block:: bash

   # Test heading alignment (robot starts at random yaw)
   python deployment/test_tracker_mujoco.py \
       --onnx ... --motion ... --render --random-heading

   # Test explicit PD (matches real robot motor loop)
   python deployment/test_tracker_mujoco.py \
       --onnx ... --motion ... --render --explicit-pd

   # Headless benchmark
   python deployment/test_tracker_mujoco.py \
       --onnx ... --motion ... --no-realtime

The script handles: MJCF loading/patching, implicit or explicit PD
configuration, heading alignment, acceleration clamping, EMA action filtering,
and real-time pacing.


Step 5: Deploy via RoboJuDo (Simulation)
-----------------------------------------

`RoboJuDo <https://github.com/HansZ8/RoboJuDo>`_ provides a plug-and-play
deploy framework for humanoid robots with MuJoCo simulation and Unitree real
robot support.

.. note::

   We provide a git patch that adds ProtoMotions BM tracker support to
   RoboJuDo.  This is a temporary solution -- we are working on integrating
   these changes as a proper upstream PR.  The patch will be removed once
   that PR is merged.

**Setup**:

.. code-block:: bash

   # 1. Clone RoboJuDo at the compatible commit
   git clone https://github.com/hansz8/robojudo.git
   cd robojudo
   git checkout 1199579188964bf82a90f0f320b4ff781907684b

   # 2. Apply the ProtoMotions patch
   git am /path/to/protomotions/g1_deploy/robojudo_patch/protomotions-bm-tracker.patch

   # 3. Install RoboJuDo
   pip install -e .
   git lfs pull   # download mesh assets

If ``git am`` fails (e.g. due to commit signing), use ``git apply`` instead
(applies changes without creating commits):

.. code-block:: bash

   git apply /path/to/protomotions/g1_deploy/robojudo_patch/protomotions-bm-tracker.patch

**Run the BM tracker in MuJoCo simulation**:

.. code-block:: bash

   cd robojudo
   python scripts/run_pipeline.py -c g1_protomotions_bm_tracker \
       --onnx-path /path/to/unified_pipeline.onnx \
       --motion-path /path/to/motion.motion

The patch adds the following to RoboJuDo:

* ``ProtoMotionsBMTrackerPolicy`` -- loads ONNX + cached 50fps motion,
  provides PD targets with action history feedback, heading alignment,
  and motion fade-in/fade-out support.
* Virtual gantry -- spring-damper safety harness for real-robot deployment
  transitions.
* Blend-in/blend-out -- smooth transitions between idle pose and policy
  control.
* CLI enhancements: ``--onnx-path``, ``--motion-path``, ``--motion-index``,
  ``--simulate-deploy``, ``--hold-seconds``.

See ``g1_deploy/robojudo_patch/README.md`` for the full list of changes.

**Bringing your own deployment framework**: The key contract is:

1. Provide the ONNX inputs documented in the YAML sidecar (joint
   positions/velocities, torso orientation, pelvis local angular velocity,
   previous processed actions, future motion references).
2. Apply the ONNX outputs (PD position targets) to your robot's actuators.
3. Apply action post-processing (acceleration clamp + EMA) as documented
   in the YAML ``control`` section.
4. Verify that MuJoCo behaviour matches between your framework and
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

   cd robojudo
   python scripts/run_pipeline.py -c g1_protomotions_bm_tracker_real \
       --onnx-path /path/to/unified_pipeline.onnx \
       --motion-path /path/to/motion.motion

The ``g1_protomotions_bm_tracker_real`` config extends the simulation config
with the Unitree real robot environment (``UnitreeCppEnv``) and safety checks
enabled.

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