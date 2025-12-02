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
3. **Factory registration** - Make robot available by name
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
           asset_file_name="mjcf/your_robot.xml",
           usd_asset_file_name="usd/your_robot/your_robot.usda",  # For IsaacLab
           self_collisions=False,  # Enable if needed
       )
   )

**Default Root Height:**

Standing height for reset/initialization:

.. code-block:: python

   default_root_height: float = 1.03  # H1_2 is taller than G1 (0.8)

**PD Control Parameters:**

Define stiffness/damping per joint using regex patterns:

.. code-block:: python

   control: ControlConfig = field(
       default_factory=lambda: ControlConfig(
           control_type=ControlType.BUILT_IN_PD,
           override_control_info={
               # Hip joints - high torque
               ".*_hip_(yaw|pitch|roll)_joint": ControlInfo(
                   stiffness=STIFFNESS_200,
                   damping=DAMPING_200,
                   effort_limit=200,
                   velocity_limit=50,
               ),
               # Knee joints - highest torque
               ".*_knee_joint": ControlInfo(
                   stiffness=STIFFNESS_300,
                   damping=DAMPING_300,
                   effort_limit=300,
               ),
               # ... more joints
           },
       )
   )

**Simulator-Specific Parameters:**

.. code-block:: python

   simulation_params: SimulatorParams = field(
       default_factory=lambda: SimulatorParams(
           isaacgym=IsaacGymSimParams(
               fps=100, decimation=2, substeps=2,
               physx=IsaacGymPhysXParams(
                   num_position_iterations=8,
                   num_velocity_iterations=4,
               ),
           ),
           newton=NewtonSimParams(fps=200, decimation=4),
       )
   )

Step 3: Register in Factory
---------------------------

Add to ``protomotions/robot_configs/factory.py``:

.. code-block:: python

   elif robot_name == "your_robot":
       from protomotions.robot_configs.your_robot import YourRobotConfig
       config = YourRobotConfig()

Step 4: Test with Random Pose Visualizer
----------------------------------------

Before training, verify your robot loads correctly:

.. code-block:: bash

   python examples/random_pose_visualizer.py \
       --robot your_robot \
       --simulator isaacgym

This sets the robot to random poses with **zero gravity and zero torque**. 
The robot should be able to hold its reset pose (press R key to reset).


Step 5: Retargeting Script (Optional)
-------------------------------------

To retarget motions to your robot, create a retargeting script.

Refer to :doc:`retargeting_pyroki` for details.

Multi-Simulator (IsaacLab) Considerations
-----------------------------------------

For IsaacLab, you also need a USD asset. The MJCF→USD conversion is covered in 
a separate tutorial.

After changing MJCF, ensure all simulators see the same robot:

1. Re-export USD from updated MJCF
2. Test on each simulator with ``random_pose_visualizer.py``
3. Verify joint order and limits match

Next Steps
----------

* :doc:`retargeting_pyroki` - Retarget motions to your robot
* :doc:`../../concepts/pose_lib` - Understand FK/IK utilities
* :doc:`../../concepts/abstractions` - Robot config architecture

