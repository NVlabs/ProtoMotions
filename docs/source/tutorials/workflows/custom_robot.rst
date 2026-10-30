Adding a Custom Robot
=====================

This guide walks through adding a new robot to ProtoMotions, using the H1_2 
robot as a reference. We focus on the **areas of change** rather than 
comprehensive details.

Overview: What Needs to Change
------------------------------

Adding a robot requires changes in these areas:

1. **MJCF file** - Robot physics definition
2. **Robot config** - ProtoMotions configuration
3. **Registration** - Automatic: the file name under
   ``protomotions/robot_configs/`` is the robot name
4. **Retargeting script** (optional) - For motion transfer
5. **Motion conversion** (optional) - For retargeted motions

We'll use H1_2 (Unitree H1 v2) and G1 as references throughout.

Step 1: MJCF File
-----------------

Create your robot's MJCF file in ``protomotions/data/assets/mjcf/``.

**Key requirements:**

* Root body with ``<freejoint/>`` for floating base
* All joints with proper limits and axes
* Collision geometries for physics

**Reference:** Compare ``h1_2_no_mesh_box_feet.xml`` and ``g1_bm_no_mesh_box_feet.xml``
to see naming conventions.

.. note::

   If you have URDF, convert to MJCF first. The MJCF serves as the ground-truth 
   robot specification.

Step 2: Robot Config
--------------------

Create ``protomotions/robot_configs/<your_robot>.py``. Key fields to define:

**Body Name Mappings:**

Map common names to your robot's body names for contact detection, observations, etc.

.. code-block:: python

   # From H1_2 config
   common_naming_to_robot_body_names: Dict[str, List[str]] = field(
       default_factory=lambda: {
           "all_left_foot_bodies": ["left_ankle_roll_link"],  # Your foot link
           "all_right_foot_bodies": ["right_ankle_roll_link"],
           "all_left_hand_bodies": ["left_wrist_yaw_link"],   # Your hand link
           "all_right_hand_bodies": ["right_wrist_yaw_link"],
           "head_body_name": ["head_aux"],                     # Your head link
           "torso_body_name": ["torso_link"],                  # Your torso link
       }
   )

**Compare G1 vs H1_2:**

* G1 uses ``left_rubber_hand`` for hands, H1_2 uses ``left_wrist_yaw_link``
* G1 uses ``head`` for head, H1_2 uses ``head_aux``
* Different joint names but same structure

**Trackable Bodies:**

Bodies used for MaskedMimic:

.. code-block:: python

   trackable_bodies_subset: List[str] = field(
       default_factory=lambda: [
           "torso_link", "head_aux",
           "right_ankle_roll_link", "left_ankle_roll_link",
           "left_wrist_yaw_link", "right_wrist_yaw_link",
       ]
   )

**Asset Configuration:**

.. code-block:: python

   asset: RobotAssetConfig = field(
       default_factory=lambda: RobotAssetConfig(
           asset_file_name="mjcf/your_robot.xml",  # MJCF is the source of truth; IsaacLab converts to USD
           self_collisions=False,  # Enable if needed
       )
   )

**Default Root Height:**

Standing height for reset/initialization:

.. code-block:: python

   default_root_height: float = 1.03  # H1_2 is taller than G1 (0.8)

**PD Control Parameters:**

Gains are not typed per joint. Each physical actuator is a ``MotorClass`` in
``protomotions/robot_configs/motor_classes.py`` (its rotor armature plus
torque and speed limits), and its ``control_info()`` derives the PD gains
from that: a 10 Hz natural frequency with a damping ratio of 2, so the
joints are over-damped rather than critically damped. Add one class per
motor rating your robot uses, then map joint-name patterns to them:

.. code-block:: python

   # protomotions/robot_configs/motor_classes.py
   H1_2_300 = MotorClass(armature=0.040, effort_limit=300, velocity_limit=50)
   H1_2_200 = MotorClass(armature=0.030, effort_limit=200, velocity_limit=50)
   H1_2_60 = MotorClass(armature=0.010, effort_limit=60, velocity_limit=50)
   H1_2_40 = MotorClass(armature=0.005, effort_limit=40, velocity_limit=50)

   # protomotions/robot_configs/h1_2.py
   control: ControlConfig = field(
       default_factory=lambda: ControlConfig(
           control_type=ControlType.BUILT_IN_PD,
           override_control_info={
               # Hip joints: the 200 N·m motor
               ".*_hip_(yaw|pitch|roll)_joint": H1_2_200.control_info(),
               # Knee joints: the 300 N·m motor
               ".*_knee_joint": H1_2_300.control_info(),
               # Two motors in parallel on one joint: scale the gains
               ".*_ankle_pitch_joint": H1_2_60.control_info(scale=2),
               # A joint whose actuator is weaker than its gain class
               ".*_elbow_joint": H1_2_40.control_info(effort_limit=18),
               # ... more joints
           },
       )
   )

``ControlInfo(stiffness=..., damping=..., effort_limit=..., velocity_limit=...)``
with literal numbers also works if you would rather not derive the gains.

**Control Rate and Simulator-Specific Parameters:**

The policy rate is ``control_hz``, and each backend's physics rate follows from
it. H1_2 and G1 run at 50 Hz with the default humanoid physics (IsaacGym 100,
IsaacLab and Newton 200, MuJoCo 1000 fps). There is no ``decimation`` field;
the simulator derives it as ``fps // control_hz``. To tune one backend, pass
its block to ``SimulatorParams.humanoid``. A passed block replaces that
backend's template wholesale, so set every field you care about:

.. code-block:: python

   control_hz: int = 50

   simulation_params: SimulatorParams = field(
       default_factory=lambda: SimulatorParams.humanoid(
           50,
           isaacgym=IsaacGymSimParams(
               fps=100,
               substeps=2,
               physx=IsaacGymPhysXParams(
                   num_position_iterations=8,
                   num_velocity_iterations=4,
               ),
           ),
           newton=NewtonSimParams(fps=200),
       )
   )

Step 3: Registration
--------------------

There is nothing to edit in ``protomotions/robot_configs/factory.py``. The
registry is the directory: the file you created in Step 2 is registered under
its file stem, so ``protomotions/robot_configs/your_robot.py`` is
``--robot-name your_robot``. Two rules apply:

* The module must define exactly one class whose direct base is
  ``RobotConfig``; zero or several such classes make ``robot_config()`` raise
  ``TypeError``.
* The stem must not be one of the infrastructure modules (``base``,
  ``factory``, ``motor_classes``).

To confirm the robot is picked up:

.. code-block:: bash

   python -c "from protomotions.robot_configs.factory import registered_robot_names; print(registered_robot_names())"

Step 4: Test with Random Pose Visualizer
----------------------------------------

Before training, verify your robot loads correctly:

.. code-block:: bash

   python examples/random_pose_visualizer.py \
       --robot your_robot \
       --simulator isaacgym

The viewer accepts every registered robot name. It draws a random joint pose
with **zero gravity and zero torque**, holds it, and draws a new one every two
seconds; there is no key to press. For a robot that is not in the viewer's own
table of marker bodies it falls back to neutral defaults, so a new robot needs
no edit to the script. The robot should hold each pose without drifting or
flying apart; if it does not, check the MJCF joint limits, the body order, the
self-collision settings and the asset itself (meshes and inertias).


Step 5: Retargeting Script (Optional)
-------------------------------------

To retarget motions to your robot, create a retargeting script.

Refer to :doc:`retargeting_pyroki` for details.

Multi-Simulator (IsaacLab) Considerations
-----------------------------------------

IsaacLab needs no USD asset from you. At scene construction the backend
converts the robot's MJCF to USD itself and caches the result under
``~/.cache/protomotions/isaaclab_mjcf_usd/``. The cache key is a fingerprint of
the MJCF and every mesh it references, so an edited robot is converted again on
the next launch; there is nothing to re-export. ``RobotAssetConfig`` has no USD
field and accepts only an ``.xml`` path. See
:doc:`../../user_guide/isaaclab3_migration` for the converter details.

After changing the MJCF, check that all simulators see the same robot:

1. Test on each simulator with ``random_pose_visualizer.py``
2. Verify joint order and limits match

Next Steps
----------

* :doc:`retargeting_pyroki` - Retarget motions to your robot
* :doc:`../../concepts/pose_lib` - Understand FK/IK utilities
* :doc:`../../concepts/abstractions` - Robot config architecture

