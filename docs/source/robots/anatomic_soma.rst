Anatomical SOMA
===============

Two Complementary Models
------------------------

Anatomical SOMA is an additional robot, not a replacement for SOMA23.
Both remain selectable through ``--robot-name``:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Contract
     - ``soma23``
     - ``anatomic_soma``
   * - Structure
     - 23 bodies, 66 joint coordinates
     - 23 semantic parts plus 30 hinge carriers, 52 joint coordinates
   * - Forward direction
     - -Y (Z up)
     - +X (Z up)
   * - Intended use
     - Existing SOMA23 motion libraries and policies
     - Reduced anatomical articulation with ordered serial hinges
   * - Motion inputs
     - Existing SOMA23 coordinates
     - Retargeted motion and consistently transformed scene poses

Anatomical SOMA uses ``Chest`` as its observation anchor. Its default
trackable subset (``Chest``, ``Head``, both feet and both hands) is what
partial-target tasks select targets from; the shipped tracker observes and
tracks all 53 bodies. SOMA23 keeps its root (``Hips``) anchor and target set,
which is the observation contract of the shipped SOMA23 policies. Experiments may select a different subset. The SOMA23 asset,
controller and observation defaults are unchanged. These models do not share an
interchangeable action space or full-body observation layout. Use motions and
checkpoints built for the selected model.

The chest anchor is specific to this model: a torso anchor is steadier than the
pelvis through the whole-body rolls and inversions in its corpus. Saved resolved
configurations retain their recorded selections; changing the anchor for an
existing policy requires a matching observation contract, not just loading
weights into the new defaults.

Training The Tracker
--------------------

``examples/experiments/mimic/anatomical_soma_fsq_mimic.py`` is the recipe
that trained the released anatomical SOMA tracker: the SOMA23 FSQ/PPO tracker
with G1-class actuators (BeyondMimic quarter-effort actions, the G1 tracker's
raw tracking rewards, G1-style physical randomization) with a wide fixed
exploration std. ``anatomic_soma_bones_seed_mini.pt`` holds
ten BONES-SEED clips retargeted with the tools below::

    python protomotions/train_agent.py --robot-name anatomic_soma --simulator isaaclab \
      --experiment-path examples/experiments/mimic/anatomical_soma_fsq_mimic.py \
      --motion-file data/motion_for_trackers/anatomic_soma_bones_seed_mini.pt \
      --num-envs 4096 --batch-size 4096 --experiment-name anatomical_soma_tracker

A ``slurmrank`` motion-file pattern cycles shards across ranks after each
evaluation, so a multi-shard corpus trains on fewer GPUs than shards.

The released tracker trained with this recipe ships at
``data/pretrained_models/motion_tracker/anatomic_soma_bones_fsq`` (see its
model card). It tracks every clip of the mini pack in IsaacLab and in MuJoCo.

Motor Design
------------

The robot configuration replaces the MJCF's controller defaults with G1-style
motor classes. Each selected class has 1.25 times the reference G1 motor class's
torque capacity and armature, and 1.5 times its speed limit. The speed
headroom exists because the corpus is retargeted under a 20 rad/s joint-speed
budget (see below): without it the 20 rad/s class on the knees and the hip roll
hinges would sit exactly at the reference peak and the hard velocity clamp would leave the
policy no room to catch up after an error. This is not an
axis-by-axis mapping: all torso axes use the 88 Nm class (G1 waist yaw), while
all wrist axes deliberately use the low-effort 5 Nm class (G1 wrist pitch/yaw),
including hand twist. G1 instead uses 50 Nm waist roll/pitch and 25 Nm wrist
roll. These are simulated actuator design
choices, not measured human properties. SOMA23's controller is unchanged.

Gains follow ``Kp = armature * (2*pi*10)^2`` and
``Kd = 2 * 2 * armature * (2*pi*10)``. The nominal 10 Hz frequency and damping
ratio of 2 refer to rotor inertia alone, not the pose-dependent inertia of the
whole articulated limb. Body masses, inertias, geometry and joint limits are
unchanged.

.. list-table:: Resolved motor parameters (rounded)
   :header-rows: 1

   * - Joints
     - Kp (Nm/rad)
     - Kd (Nm s/rad)
     - Armature (kg m^2)
     - Effort (Nm)
     - Speed (rad/s)
   * - Spine, chest, hip pitch and twist (``Leg_x``, ``Leg_z``)
     - 50.22
     - 3.20
     - 0.012722
     - 110
     - 48
   * - Hip roll (``Leg_y``), knees
     - 123.87
     - 7.89
     - 0.031377
     - 173.75
     - 30
   * - Ankles
     - 35.63
     - 2.27
     - 0.009024
     - 62.5
     - 55.5
   * - Neck1, clavicles, arms, forearms, toes
     - 17.81
     - 1.13
     - 0.004512
     - 31.25
     - 55.5
   * - Neck2, head, hands
     - 20.97
     - 1.34
     - 0.005313
     - 6.25
     - 33

Controller stability does not require the character to balance under zero or
random targets. A random-target test allows falls, then holds a fixed target
to check for persistent oscillation. Finite, settling behavior is necessary
but does not establish learned tracking performance or sim-to-sim transfer.

IsaacLab uses 240 Hz physics with decimation 8, eight position iterations and
four velocity iterations. The 30 Hz decision rate is unchanged. The shipped
check is ``test_random_targets_at_30_hz_remain_finite_and_settle`` in
``protomotions/tests/test_anatomic_soma.py``: one robot in MuJoCo, random
direct joint targets, then a held target, must stay finite and settle. A
one-off IsaacLab run of the same protocol during development (eight robots, 30 s
of random targets, 20 s holding) also settled, with 0.011 rad of residual joint
motion over the last five seconds; that script is not shipped. Neither check
uses a learned policy, and contact jitter remains, so this is not a claim of
motionless contact equilibrium. Other backend settings remain the same as SOMA23.

The default joint-angle offset is zero (the asset's T-pose), not a balancing
standing pose. With BeyondMimic action scaling, a zero action requests that pose.

MuJoCo Stability
----------------

Anatomical SOMA inherits SOMA23's ``implicitfast`` integrator, 960 Hz physics
and decimation 32 (30 Hz control). Keep MuJoCo's built-in implicit PD enabled
(``use_implicit_pd=True``): ``implicitfast`` alone does not implicitly integrate
damping supplied as externally computed torques. The MuJoCo backend normalizes motor gearing
before applying the resolved PD controller. These settings address integration
and actuator-unit problems independently of the motor-class overrides.
Euler integration can oscillate with stiff actuator damping even when the
target is constant. Do not compensate for that numerical problem by silently
changing a trained policy's gains.

Saved checkpoints retain their saved gains and simulator settings. Check those settings
when testing an older policy; changing the robot defaults does not update a
saved configuration. A stable constant-target PD test is not a substitute for
closed-loop tracking tests with contacts on both simulators.

Anatomical Structure
--------------------

``--robot-name anatomic_soma`` selects 52 scalar hinges with 23 named anatomical
parts. Fifteen of SOMA23's 22 joints become three serial hinges each: pitch,
roll and twist. The knees (``Shin_x``), elbows (``ForeArm_z``) and toes
(``ToeBase_x``) are single hinges, and Neck2 keeps one yaw hinge limited to
+/-30 degrees, tilted with the neck. Intermediate carrier links serialize the
triples, so the simulator sees single-axis joints rather than a multi-axis joint
group; each carrier sits on its semantic body and carries no collision geometry.
The other joint limits are engineering ranges, not medically validated human
limits. The asset uses +X forward, +Y left and +Z up, in meters.

Hinge names keep SOMA's original axis letters, which do not name the asset axis
they rotate about. In the rest pose:

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Suffix
     - Asset axis
     - Anatomical meaning
   * - ``_x``
     - +/-Y (lateral)
     - Pitch: flexion/extension (hips, knees, toes, shoulders, spine, head)
   * - ``_y``
     - -X (forward)
     - Roll: abduction/adduction, lateral bending
   * - ``_z``
     - +Z (vertical)
     - Twist about the limb or spine; elbow flexion for the T-posed forearm

So ``LeftLeg_x`` is hip flexion and ``LeftLeg_y`` is hip abduction; the 20 rad/s
motor class sits on ``Leg_y`` and the knees, as on G1's hip roll and knee.

Carrier bodies are named by position in the part's hinge order, not by the
hinge they hold: ``<Part>_x_link`` carries the first hinge, ``<Part>_y_link``
the second and the part body itself the third. For the torso that means
``Spine1_x_link`` carries ``Spine1_z``, and ``LeftArm_y_link`` carries
``LeftArm_z``. Read per-body logs with the table above, not the link name.

Every body-level quantity (target observations, relative-body and velocity
rewards, the any-body termination, the evaluator's body error) runs over all 53
bodies, carriers included, so a three-hinge joint counts three times where a
single-hinge joint counts once. The shipped tracker was trained and evaluated
that way; its error numbers are not comparable to SOMA23's.

Joint orders are defined per body in
``data/scripts/from_soma_to_anatomical_soma/anatomical_soma_joint_structure.py``
and realized as the asset's hinge order. In original SOMA axis names, torso
and neck triples use ZYX, wrists YXZ, and hips, ankles and clavicles XYZ.
Upper arms use XZY. Because
SOMA retains its T-pose frames, upper-arm XZY is not the same literal axis
sequence as the pitch-roll-yaw chain of an arms-down robot. In the canonical
frame its signed axis sequence is YZX; torso ZYX becomes ZXY, wrists YXZ
becomes XYZ, and hips XYZ becomes YXZ.

Each named axis retains its signed range and dynamics when reordered. These
are scalar engineering limits: the coupled reachable orientation envelope
changes with ordering and is not an exact conversion of native rotation-vector
limits. Both left and right axis signs are retained. Source motions must be
retargeted again; existing hinge coordinates cannot be permuted into new poses.

Regenerate the ordering from a geometry template with::

    python data/scripts/from_soma_to_anatomical_soma/build_soma_joint_structure.py \
        --source protomotions/data/assets/mjcf/anatomic_soma.xml \
        --output /tmp/anatomic_soma_rebuilt.xml --serial

On the shipped asset this reproduces the file byte for byte, which doubles as
a consistency check after editing it. For a template with three joints on each
semantic body, omit ``--serial``. The builder preserves body frames, collision
geometry, inertias and each named axis's limits; it also orders actuators to
match the resulting DOF order. The retargeter's hinge initialization reads the
mounting quaternions from the asset.

The collision template has 39 primitives: horizontal torso capsules, limb
capsules, a box head, and five rounded longitudinal rails on each foot/toe
segment. Intermediate hinge carriers have no collision geometry. Mass and
inertia are explicit; overlapping sole rails do not multiply the body mass.
Neck1 and shoulder capsule radii are reduced to 29 mm and 33 mm respectively
to remove non-excluded rest-pose overlap without disabling self-collision.
This asset needs new policy training. Original SOMA checkpoints do not have
the same action or body-observation dimensions.

Retargeting Existing SOMA Motions
---------------------------------

Run the packed-data driver in the ProtoMotions environment and provide the
Python executable from a PyRoki environment (set up as in
:doc:`../tutorials/workflows/retargeting_pyroki`)::

    python data/scripts/from_soma_to_anatomical_soma/retarget_motions.py \
        --input source.pt --output anatomic.pt \
        --pyroki-python /path/to/pyroki/python \
        --cache-dir /path/to/jax-cache

Use ``--clip-ids`` for an explicit pilot subset; without it every clip is
processed. The driver preserves motion names, frame counts, timestamps,
weights and contact labels, and recomputes joint/body velocities from
the fitted trajectory. It writes a packed MotionLib, per-clip audit JSONL, and
run metadata. Existing outputs are never overwritten. Scene alignment must
be checked separately when retargeting scene-interaction data.

Source SOMA faces -Y. The driver applies one fixed +90-degree rotation about Z
to positions and conjugates body rotations into the new coordinate basis.
Root trajectories and headings receive that same transform, with no per-clip
reheading, time warp or trimming. Convert each associated serialized SceneLib
using the same basis before using scene-interaction data::

    python data/scripts/from_soma_to_anatomical_soma/convert_soma_scene.py \
        --input source_scenes.pt --output anatomic_scenes.pt

This rotates world object poses and scene offsets, retains local collision/mesh
geometry, scales and motion IDs, and resolves mesh dependencies against the
source scene directory. For unchanged object-local meshes the orientation
transform is ``C @ R``, not the robot body's ``C @ R @ C.T``. Mesh files remain
external dependencies recorded in the scene metadata. Existing per-frame goal masks are copied unchanged;
they contain no geometric coordinates.

The bounded hinge projection is only an initialization. PyRoki fits the
original body positions and orientations over each complete clip, with bounded
joint coordinates, joint velocity/acceleration regularization and increased
position weight for source contact bodies. It does not clip the optimized
output or remove difficult frames. Non-finite coordinates, limit violations
and coordinate jumps over pi cause failure. Exhausting the iteration budget
without a convergence criterion also fails. The inverse-tanh initialization
moves exact-bound seeds slightly inward; output coordinates remain bounded
throughout the solve, not clipped afterward. Initial LM damping is 1.0 to
avoid aggressive steps saturating bounded variables on difficult clips.
The maximum LM damping is 1e12, allowing stiff temporal penalties to reject
an aggressive step without exhausting the damping range. The iteration cap
defaults to 500; convergence can stop a solve earlier.
The audit reports position and
rotation error even when the solver succeeds: success is not a certificate
of good contacts, collision-free motion or physical trackability.

Synthetic recovery data may start outside the anatomical pose envelope.
For these packs, ``--allow-motion-budget-exceedance`` makes velocity and
acceleration acceptance-budget exceedances diagnostic warnings. It does not
change the objective, joint bounds, convergence or finite-value checks.

Goal priorities follow the G1 retargeter's approach rather than assigning
equal importance to every body. Hands, feet/toes, head and chest receive
full position weight; intermediate bodies receive one quarter. Their
orientation weight is reduced further so intermediate orientations cannot
dominate end-effector placement. Limb-direction terms preserve the bend
direction, and persistent source contacts penalize world-space displacement.
The audit includes position and orientation error for each named body,
both before and after fitting.

This is an adaptation, not an identical G1 objective. SOMA-to-SOMA retains
the same rest dimensions and fixes the original root position and rotation;
it does not scale root height or horizontal travel. Source orientations
provide hand, head and foot orientation goals directly, without G1-specific
auxiliary keypoints. Contact orientation follows the source rather than
forcing every foot flat on a horizontal floor, which would damage climbing
and non-upright motions. Different-sized embodiments require a separate
proportion-aware target/root conversion; this script does not perform it.

Joint continuity has a separate limit-violation loss on physical scalar
coordinates: velocity is ``(q[t+1]-q[t])/dt`` and acceleration is
``(q[t+1]-2*q[t]+q[t-1])/dt**2``. Angles are not wrapped; wrapping could hide
a full-turn discontinuity. Penalties begin at 90% of the configured budgets,
and the completed solve is rejected if it exceeds a budget (apart from a
1e-5 relative numerical tolerance). Defaults are 20 rad/s and 300 rad/s²;
these are explicit retargeting budgets, not medically validated or
actuator-certified limits. Set ``--max-joint-velocity`` and
``--max-joint-acceleration`` for the intended robot/control regime.
No frames are removed and no output trajectory is clipped to satisfy a cap.
Collision, contact feasibility and torque-limited dynamic tracking remain
separate simulation checks before treating converted data as training-ready.

Verification
------------

Run the asset, kinematics, motor and conversion tests in the ProtoMotions
environment::

    pytest protomotions/tests/test_anatomic_soma.py protomotions/tests/test_soma_joint_structure.py

Run the solver's own tests in the PyRoki environment::

    JAX_PLATFORMS=cpu python -m pytest pyroki/from_soma_to_anatomical_soma/ -q

Solver internals (analytic kinematics, the banded linear algebra, padding and
compilation caching) are described in
``pyroki/from_soma_to_anatomical_soma/README.md``.
