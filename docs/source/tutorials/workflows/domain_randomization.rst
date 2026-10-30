Domain Randomization & Sim2Sim
==============================

This workflow covers training with domain randomization for robust policies that
transfer across simulators (sim2sim) or to real robots (sim2real).

For a complete end-to-end pipeline from data to deployment, see
:doc:`g1_deployment`.

Why Domain Randomization?
-------------------------

Policies trained in simulation often fail when deployed to different physics engines
or real hardware due to the "reality gap". Domain randomization addresses this by:

1. **Randomizing physics parameters** (friction, center of mass)
2. **Adding action noise** (motor imprecision)
3. **Adding observation noise** (sensor noise for IMU, encoders)
4. **Applying external perturbations** (pushes, velocity impulses)
5. **Randomizing action latency** (holding the previous action for a sampled delay)
6. **Forcing the policy** to be robust to parameter variations

Training with Domain Randomization
-----------------------------------

The BeyondMimic L2C2 experiment config (``examples/experiments/mimic/mlp_bm_l2c2.py``)
is a good starting point for DR training:

.. code-block:: bash

   python protomotions/train_agent.py \
       --robot-name g1 \
       --simulator isaaclab \
       --experiment-path examples/experiments/mimic/mlp_bm_l2c2.py \
       --experiment-name g1_bm_dr \
       --motion-file /path/to/bones_seed_g1_motions.pt \
       --num-envs 4096 \
       --batch-size 16384

You can also create your own experiment configs with DR settings.  The randomization types
described below are general and can be mixed into any experiment config.


Domain Randomization Types
--------------------------

ProtoMotions supports several randomization types via ``DomainRandomizationConfig``.
The action-noise, friction, center-of-mass, push and reset-noise blocks below
are the settings the pre-trained G1 BeyondMimic tracker was trained with
(``data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py``).
The action-latency, scene-object and body-mass blocks illustrate the remaining
types; the shipped tracker does not use them. You can adjust ranges to suit
your robot and deployment conditions.

**Action Noise:**

.. code-block:: python

   ActionNoiseDomainRandomizationConfig(
       action_noise_range=(-0.025, 0.025),  # +/-2.5% noise on PD targets
       dof_names=[".*"],  # Apply to all joints
   )

**Action Latency** (not used by the shipped tracker):

Action latency samples a delay independently for each environment at every control
step, uniformly between zero and ``max_latency_ms`` milliseconds. The previous
action is held until that delay expires. If the arrival time falls between two
physics-substep boundaries, that one substep uses the time-average of the previous
and new actions; later substeps use the new action. The first command after an
environment reset is never delayed. The maximum delay must be no greater than one
control interval, which is ``1000 / control_hz`` milliseconds (20 ms for the 50 Hz
humanoids); the simulator refuses a larger value at construction. A value of
``0.0`` disables latency randomization, which is the default.

.. code-block:: python

   DomainRandomizationConfig(
       latency=LatencyDomainRandomizationConfig(max_latency_ms=3.0),
   )

The setting applies to all supported simulator backends: IsaacGym, IsaacLab,
Newton, and MuJoCo. On Newton, enabling action latency disables CUDA
graph execution because control targets must change between physics substeps;
expect lower simulation throughput while it is enabled.

**Friction Randomization:**

.. code-block:: python

   FrictionDomainRandomizationConfig(
       num_buckets=64,
       static_friction_range=(0.3, 1.6),
       dynamic_friction_range=(0.3, 1.2),
       restitution_range=(0.0, 0.5),
       body_names=[".*"],  # Apply to all bodies
   )

.. note::

   **Default Values:** ProtoMotions assumes a default friction of **1.0** and
   restitution of **0.0** for all entities (robot bodies and terrain) when
   values are not explicitly set.  This ensures consistent behavior across
   simulators.

**Scene Object Asset Randomization** (not used by the shipped tracker):

Scene objects can define deterministic base properties through
``ObjectOptions``.  Mesh and primitive objects can set either ``mass`` or
``density`` (not both), plus ``static_friction``, ``dynamic_friction``, and
``restitution`` in the scene library.  ``ObjectAssetDomainRandomizationConfig``
samples absolute override values for those properties; the sampled values
replace the scene-lib base values for the selected asset bucket.  They are not
additive or multiplicative offsets.  The ranges are global for all scene object
assets; per-asset range configuration is not currently supported.

.. code-block:: python

   DomainRandomizationConfig(
       object_assets=ObjectAssetDomainRandomizationConfig(
           num_buckets=32,
           static_friction_range=(0.4, 1.4),
           dynamic_friction_range=(0.3, 1.0),
           restitution_range=(0.0, 0.2),
           mass_range=(0.5, 3.0),
           center_of_mass_range={
               "x": (-0.05, 0.05),
               "y": (-0.02, 0.02),
               "z": (0.00, 0.10),
           },
       )
   )

``mass_range`` and ``density_range`` are mutually exclusive, matching
``ObjectOptions``.  ``center_of_mass_range`` is only available through domain
randomization; it is an absolute local center-of-mass value, and omitted axes
default to ``0.0``.  IsaacLab and IsaacGym apply these scene object asset
properties to both primitive and mesh objects.  Newton currently does not spawn
scene-lib objects, so object asset randomization has no effect there.

**Robot Body Mass Randomization** (not used by the shipped tracker):

Robot body mass randomization samples absolute masses and scales inertia by the
same ratio. It is supported by IsaacGym, IsaacLab, and Newton. Body-mass domain
randomization is not supported yet in MuJoCo; that backend fails
immediately when it is configured instead of silently ignoring it.

.. code-block:: python

   BodyMassDomainRandomizationConfig(
       num_buckets=64,
       mass_range=(0.001, 1.0),
       log_uniform=True,
       body_names=["left_rubber_hand", "right_rubber_hand"],  # G1's hand bodies
   )

``body_names`` are regular expressions matched against the whole body name, so
they must name bodies the robot actually has. G1's hands are
``left_rubber_hand`` and ``right_rubber_hand`` (``[".*_hand"]`` matches both).
A pattern that matches nothing stops the build with ``body mass randomization
matched no robot bodies``.

**Center of Mass Randomization:**

.. code-block:: python

   CenterOfMassDomainRandomizationConfig(
       com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
       body_names=["torso_link"],  # Apply to torso
   )

**Sensor Noise:**

Declare physical sensors on ``env.sensors``, one channel at a time. Joint
encoders and the IMU (orientation and gyro) are mandatory fields; localization,
base linear velocity and height are optional fields that are absent when left
``None``. Each channel carries its own noise levels. The robot config declares
the IMU body and mounting quaternion separately from the observation anchor.
This is the block the L2C2 recipe uses:

.. code-block:: python

   from protomotions.envs.sensors.config import (
       IMUSensor, JointEncoderSensor, SensorConfig,
   )

   env_cfg.sensors = SensorConfig(
       joint_encoders=JointEncoderSensor(
           pos_noise=0.01,   # Uniform half-width, radians
           vel_noise=0.5,    # Uniform half-width, rad/s
       ),
       imu=IMUSensor(
           quat_noise=0.05,  # Quaternion-component perturbation, then normalize
           gyro_noise=0.2,   # Uniform half-width, rad/s in sensor axes
       ),
       # localization, base_velocity and height stay None: not on this robot.
   )

A channel that is present but read exactly takes ``NoNoise("reason")`` in
place of a half-width, for example
``IMUSensor(quat_noise=NoNoise("calibrated mount"), gyro_noise=0.2)``. A sensor
the robot does not have is left ``None``, not given a ``NoNoise`` entry: there
is no suite object and no noise slot for a channel that does not exist.

These are empirical robustness settings, not a hardware calibration. Each
physical channel is sampled once per step. Body positions, rotations and
velocities are reconstructed from those readings using forward kinematics, so
noise remains correlated across the body. Both ideal and noisy readings use the
same estimator. Reset perturbations remain separate in ``RobotNoiseConfig``.

.. note::

   On G1 the IMU is on the pelvis (``imu_body_name``) and the observation
   anchor is ``torso_link`` (``anchor_body_name``).  The anchor is not measured
   directly: its orientation and angular velocity are computed from the pelvis
   IMU and the waist joint encoders, so IMU and joint-encoder noise both reach
   it.  The default deploy path computes the torso orientation the same way
   (see :doc:`g1_deployment`).

   To anchor on the pelvis instead, call
   ``robot_cfg.update_fields(anchor_body_name=None)``. ``anchor_body_index``
   is derived from the name (0 for ``None``) and cannot be set directly;
   ``update_fields(anchor_body_index=...)`` is refused.

Observation kernels bind the privileged names (``current.*``) by default;
``sensed()`` repoints a dict of components at one source and returns the
rebound copies:

.. code-block:: python

   from protomotions.envs.context_sources import sensed
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.obs.humanoid import (
       compute_humanoid_reduced_coords_observations,
   )

   reduced = {
       "self_obs": MdpComponent.from_kernel(
           compute_humanoid_reduced_coords_observations
       )
   }
   actor_obs = sensed(reduced, "noisy")
   ideal_sensor_obs = sensed(reduced, "clean")
   critic_obs = reduced  # privileged: the kernel's own defaults

.. _sensor-observation-sources:

Privileged, clean and noisy sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These names specify both **where the values come from** and **which information
is available**. They are not three interchangeable noise levels.

.. list-table:: Observation sources
   :header-rows: 1
   :widths: 15 45 40

   * - Source
     - Construction
     - Typical use
   * - ``privileged``
     - Read simulator state directly, including available body positions,
       velocities, contacts and forces. This is the default source.
     - Ground-truth rewards/evaluation and privileged critic or teacher inputs.
   * - ``clean``
     - Measure only the declared sensors without added noise or drift, then
       reconstruct body state using the kinematic model.
     - A noise-free sensor baseline, including the clean side of L2C2.
   * - ``noisy``
     - Perturb those same sensor readings, then run the same reconstruction.
     - Policy observations during sensor-noise training.

For example, without a localization channel, the simulator still knows the
robot's world position, but neither ``clean`` nor ``noisy`` exposes it. Without
a base-velocity channel, neither estimate exposes world linear body velocities.
Turning off noise does not add missing measurements. The sensor estimates also
do not acquire simulator contact forces simply because those forces are known
to the simulator.

Why keep ``clean`` as well as ``privileged``? A clean/noisy pair lets us isolate
the effect of sensor corruption while keeping the sensor suite, estimator,
feature layout and coordinate conventions the same. Replacing the clean side
with privileged state could instead change the available information or its
derivation. The BM L2C2 recipe uses this pair for its actor consistency loss;
the critic can independently use privileged observations. Experiments that do
not use a clean baseline need not bind any observations to it; the current
runtime nevertheless reconstructs and stores both estimates when a suite is
configured.

With the required sensors present and an exact kinematic model, corresponding
clean values should agree with simulator truth to numerical tolerance. This
does not make the sources interchangeable: unmeasured fields remain absent,
and derived quantities must use matching conventions. When localization and the
height channel are declared, sensor ground height is inferred from root clearance,
whereas the privileged view queries terrain under the observation anchor; these
may differ on uneven ground. Without those channels, sensor ground height is
absent, not a substitute terrain query.

Without a configured suite, neither sensor source exists. A component that
binds an undeclared channel fails when its bindings are finalized, naming the
component and path. For example, enabling root velocity observations requires
the suite's base-velocity channel.

The current root-height observation takes world root position minus ground
height, so it requires both localization and the height channel. A height
reading alone does not provide world positions; the estimator never invents
world XY coordinates to satisfy a position binding.

For noise-free simulation inference nothing has to be replaced:
``inference_agent.py`` switches every declared channel to an exact read
(``SensorConfig.for_inference()``) before any override or recipe hook runs,
and the channels and source bindings stay as trained. The manual equivalent,
if a script needs it, is ``exact_channel()`` in
``protomotions/envs/sensors/config.py``. Exported policies do not sample
noise. Their contract
describes the physical sensor contract and their host must supply those
measurements. ``deployment/export_to_onnx.py`` reads the sensor suite
from the checkpoint and records the channels the actor binds in the sidecar.
In a robot export, body poses are never graph inputs: a recipe that needs the
current body poses binds the ``BakedAnchorFK`` kernel to the joint readings and the anchor pose,
and the exported graph computes them inside the policy. The policy graph keeps
its four control outputs (``actions``, ``joint_pos_targets``,
``stiffness_targets``, ``damping_targets``); every output is a command. A host
that needs the body poses outside the policy call runs a second, small graph,
``fk.onnx``, which ``deployment/export_fk_to_onnx.py`` exports from the same
kernel object the policy traces, rebuilt from the same config bundle beside
the checkpoint. Its inputs carry the policy graph's names for the same sensor
paths, so a host feeds one tensor under one name to both graphs, and the two
graphs agree by construction because there is one FK implementation.
The exporter is also told who will feed the graph: the default,
``--host robot``, refuses a binding to the simulator's own body state, while
``--host simulation`` (a game engine, a browser MuJoCo, a sim-to-sim check)
accepts the privileged body poses, velocities and ground heights under
``current.*`` and records ``host: simulation`` in the sidecar. A policy meant
for a robot is exported with the default even while it is tested in a
simulator.
The current standalone tracker host requires a root-body IMU with identity
mount. Other placements need host-side conversion before using that exporter.

**Implementation and history:**

* ``envs/sensors/measurement.py`` reads the physical channels.
* ``noise.py`` applies channel noise and shared odometry corruption math.
* ``estimation.py`` reconstructs body state in world SI units using xyzw quaternions.
* ``runtime.py`` owns the random generator and episode-start odometry records.

After each physical step, ideal and noisy readings are reconstructed together
in one FK call. A single history buffer stores privileged, clean and noisy
states, with commands shared between the sources. A reset repeats the measured
post-reset state into history and zeros its commands; it does not insert earlier
poses from the reference motion.

Raw actions are policy outputs before action processing. Processed actions are
the action pipeline's simulator commands. Command-history observations start
with the command that produced the current state (one decision ago).
Previous-state history excludes the current state and carries each older
state's incoming command (starting two decisions ago).
The action-smoothness reward reads ``previous_action``, so the history ring
keeps one past slot for it; nothing has to be configured.

**Migration from observation-level noise:**

Saved configs containing the retired ``observation_noise`` or environment
odometry fields are rejected at config loading. Regenerate the
``resolved_configs.json`` bundle from an experiment with an explicit sensor
suite and source selectors. Retrain policies:
correlated sensor noise, reset history and action-history timing change the
observation contract. Renaming fields in an old checkpoint does not preserve
its training semantics.

IsaacLab and IsaacGym root and body linear velocities in this update refer to
the link origin, not the center of mass. Both engines report center-of-mass
velocities natively, and the backends convert them; MuJoCo already reported
link-origin velocities. For G1 the difference reaches a few tenths of a meter
per second when the robot turns quickly. MuJoCo refreshes kinematics after
stepping.

IsaacGym needs two more corrections, both checked against native Gym on a GPU:

- Gym's body tensors only update when physics steps, so right after a reset they
  still hold the previous episode's poses. The backend reconstructs post-reset
  body kinematics before observations, sensor readings and history are
  captured.
- Gym keeps only the last root-state write of each kind before physics runs.
  The backend now merges every write in a step (resets, pushes, projectiles,
  markers) into one. Before this update, a push was silently lost whenever a
  reset or marker update followed it in the same step. With thousands of
  environments some environment resets almost every step, so most pushes never
  reached the physics engine, and recipes tuned on IsaacGym before this update
  were in effect trained with few pushes.

**Push/Perturbation Randomization:**

.. code-block:: python

   PushDomainRandomizationConfig(
       push_interval_range=(1.0, 3.0),         # Seconds between pushes
       max_linear_velocity=(0.5, 0.5, 0.2),    # Max push velocity (x, y, z) m/s
       max_angular_velocity=(0.52, 0.52, 0.78), # Max angular impulse (rad/s)
   )

This helps policies learn to recover from unexpected perturbations, improving
robustness on real hardware where the robot may be bumped or jostled.

**Reset Noise (initial state perturbation):**

.. code-block:: python

   RobotNoiseConfig(
       dof_pos_noise=0.1,
       root_pos_noise=[0.05, 0.05, 0.01],
       root_rot_noise=[0.1, 0.1, 0.2],
       root_vel_noise=[0.1, 0.1, 0.05],
       root_ang_vel_noise=[0.1, 0.1, 0.1],
   )

At the start of each episode, the robot's initial state is perturbed around the
reference motion's first frame.  This prevents the policy from relying on a
perfect starting pose.


Automatic Friction Combine Mode Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physics engines combine robot and terrain friction differently. Two backends
have a fixed combine mode, and those are the two ProtoMotions converts for:

* **IsaacGym**: fixed ``AVERAGE`` -- effective friction is
  ``(robot + terrain) / 2``. Converted.
* **Newton**: fixed ``MAX`` -- effective friction is ``max(robot, terrain)``.
  Converted.
* **IsaacLab**: applies the combine mode written in the terrain config, so
  there is nothing to convert.
* **MuJoCo**: no conversion. The backend sets the robot's own geoms to
  ``simulator.default_robot_friction`` and does not touch the floor.

``convert_friction_for_simulator()`` runs once, when the components are built.
For IsaacGym and Newton it rewrites the terrain friction, and the friction
domain-randomization ranges if any, so that the configured effective friction
is preserved under that backend's combine mode. For IsaacLab and MuJoCo it
returns the configs unchanged.

**How it works** when the target backend combines with ``MAX`` (Newton):

1. Character shape friction is set from ``simulator.default_robot_friction``
2. Terrain friction is adjusted to preserve the configured effective value
3. Domain-randomized character friction overrides the baseline for selected bodies

This keeps the configured character friction explicit even when domain
randomization is disabled. Between IsaacGym, IsaacLab and Newton the same
experiment config therefore gives the same effective friction without edits.
MuJoCo is the exception: the floor keeps whatever friction the MJCF gives it,
which is one of the differences a sim2sim check exercises.


L2C2 Smoothness Regularization
------------------------------

The BM tracker config additionally uses **L2C2** (Locally Lipschitz Continuous
Constraint, Kobayashi 2022) to encourage smooth policy outputs despite noisy
observations:

* Both noisy and clean versions of each actor observation are computed
* The policy runs on both, producing ``mu_noisy`` and ``mu_clean``
* An auxiliary loss penalizes ``MSE(mu_noisy, mu_clean)``
* This encourages the policy to produce similar actions regardless of sensor noise

For simulation inference, the recipe disables injected sensor noise and removes
the extra L2C2 observations/loss. It keeps the actor's sensor-source bindings;
it does not replace them with privileged simulator state. On hardware, the host
supplies real measurements, which may themselves be noisy. Disabling simulated
noise does not make real sensors perfect.


Sim2Sim Testing
---------------

After training with DR, test on MuJoCo to verify transfer before real-robot
deployment:

.. code-block:: bash

   python protomotions/inference_agent.py \
       --checkpoint results/g1_bm_dr/last.ckpt \
       --simulator mujoco \
       --motion-file /path/to/motion.motion

MuJoCo uses different contact dynamics, solver, and friction combine mode than
the training simulator (IsaacLab).  A policy that works across both has learned
robust dynamics rather than overfitting to one physics engine.

.. warning::

   **Spherical joint limitation:** Sim2sim transfer currently only works for
   robots with hinge (revolute) joints, such as the G1 and H1.  Robots that
   use spherical (ball) joints — like SMPL and SMPL-X — have different
   spherical joint representations across simulators (IsaacGym/IsaacLab vs
   Newton/MuJoCo), and cross-simulator transfer is not yet supported for
   these morphologies.

For a more thorough deployment validation, see the standalone MuJoCo test script
described in :doc:`g1_deployment` (Step 4), which runs the exported ONNX model
independently of the ProtoMotions training framework.


Training Tips
-------------

**Start without DR**: Train a baseline without domain randomization first. This
confirms your motion data and rewards are working.  In our experience, DR does
not make training significantly harder.

**Observation history**: Some configs use observation history to help the
policy infer physics parameters from recent state transitions.  The BM tracker
config uses previous processed actions as a form of single-step history.

**Noise levels**: The noise values above are empirically set to produce robust
policies.  If your real-robot sensors are particularly noisy or clean, adjust
the noise ranges accordingly.  When in doubt, err on the side of more noise --
it's better to over-regularize than to have a fragile policy.


Next Steps
----------

* :doc:`g1_deployment` - Full pipeline from data to real robot deployment
* :doc:`custom_robot` - Add your robot for DR training
* :doc:`../../concepts/abstractions` - Understand simulator abstraction
