PoseLib Toolkit
===============

**Location:** ``protomotions/components/pose_lib.py``

PoseLib is a suite of utilities that bridges MJCF robot definitions to simulation
state. It's called throughout the codebase for FK/IK, motion processing, and
coordinate conversions.

What PoseLib Does
-----------------

1. **Parse MJCF → KinematicInfo**: Extract robot structure from MJCF files
2. **Coordinate conversions**: Transform between reduced (qpos) and maximal coordinates
3. **Forward/Inverse Kinematics**: Compute body poses from joint angles and vice versa
4. **Velocity computation**: Finite-difference velocities from pose sequences

MJCF Parsing
------------

``extract_kinematic_info()`` parses an MJCF file and returns a ``KinematicInfo`` 
dataclass containing the robot's kinematic structure:

.. code-block:: python

   from protomotions.components.pose_lib import extract_kinematic_info
   
   kinematic_info = extract_kinematic_info(
       "protomotions/data/assets/mjcf/g1_bm_no_mesh_box_feet.xml"
   )

**KinematicInfo fields:**

.. code-block:: python

   @dataclass
   class KinematicInfo:
       body_names: List[str]          # All body names in order
       dof_names: List[str]           # Joint names (excluding root)
       parent_indices: List[int]      # Parent body index for each body
       local_pos: Tensor              # Local position offsets
       local_rot_ref_mat: Tensor      # Reference local rotations
       hinge_axes_map: Dict           # Hinge joint axes
       nq: int                        # Dimension of qpos
       nv: int                        # Dimension of qvel
       num_bodies: int
       num_dofs: int
       dof_limits_lower: Tensor
       dof_limits_upper: Tensor

This is automatically populated when you create a robot config.

Coordinate Systems
------------------

Motion data and simulation use different coordinate representations:

**Reduced Coordinates (qpos):**

MuJoCo-style generalized coordinates:

.. code-block:: text

   qpos = [root_pos(3), root_quat_wxyz(4), joint_angles(...)]

Compact representation - one value per DOF.

**Maximal Coordinates:**

World-space position and rotation for each rigid body:

.. code-block:: python

   rigid_body_pos: Tensor[num_envs, num_bodies, 3]
   rigid_body_rot: Tensor[num_envs, num_bodies, 4]

Easier for reward computation (distances in Cartesian space).

Coordinate Conversions
----------------------

**Reduced → Maximal (Forward Kinematics):**

.. code-block:: python

   from protomotions.components.pose_lib import (
       extract_transforms_from_qpos,
       compute_forward_kinematics_from_transforms
   )
   
   # qpos → local transforms
   root_pos, local_rot_mats = extract_transforms_from_qpos(
       kinematic_info, qpos
   )
   
   # local transforms → world poses
   world_pos, world_rot_mat = compute_forward_kinematics_from_transforms(
       kinematic_info, root_pos, local_rot_mats
   )

**Maximal → Reduced (Inverse):**

.. code-block:: python

   from protomotions.components.pose_lib import extract_qpos_from_transforms
   
   qpos = extract_qpos_from_transforms(
       kinematic_info, root_pos, local_rot_mats,
       multi_dof_decomposition_method="exp_map"
   )

Velocity Computation
--------------------

Compute velocities from pose sequences via finite differences:

.. code-block:: python

   from protomotions.components.pose_lib import (
       compute_cartesian_velocity,
       compute_angular_velocity
   )
   
   # Linear velocity from position sequence
   linear_vel = compute_cartesian_velocity(positions, fps=30)
   
   # Angular velocity from rotation sequence
   angular_vel = compute_angular_velocity(rotation_mats, fps=30)

High-Level Functions
--------------------

For motion processing, use the high-level wrapper:

.. code-block:: python

   from protomotions.components.pose_lib import fk_from_transforms_with_velocities
   
   # Compute full state (positions, rotations, velocities)
   state = fk_from_transforms_with_velocities(
       kinematic_info=kinematic_info,
       root_pos=root_positions,
       joint_rot_mats=local_rotations,
       fps=30,
       compute_velocities=True
   )
   # Returns RobotState with all fields populated

ControlInfo
-----------

PoseLib also defines ``ControlInfo`` for PD control parameters:

.. code-block:: python

   @dataclass
   class ControlInfo:
       stiffness: float      # P gain
       damping: float        # D gain
       armature: float       # Motor inertia
       friction: float       # Joint friction
       effort_limit: float   # Max torque
       velocity_limit: float # Max velocity

Used in robot configs to specify per-joint control properties.

Testing
-------

Verify FK/IK against MuJoCo:

.. code-block:: python

   from protomotions.components.pose_lib import test_fk_against_mujoco
   
   test_fk_against_mujoco("path/to/robot.xml")

This compares PoseLib's FK output to MuJoCo's internal FK.

Next Steps
----------

* :doc:`simulator_state` - How states are represented in simulation
* :doc:`abstractions` - Where PoseLib fits in the architecture

