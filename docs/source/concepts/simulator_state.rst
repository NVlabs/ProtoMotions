Simulator State
===============

**Location:** ``protomotions/simulator/base_simulator/simulator_state.py``

The simulator state classes provide a unified representation of simulation state
across all physics backends.

RobotState
----------

``RobotState`` contains the complete state of the robot:

.. code-block:: python

   @dataclass
   class RobotState:
       # Maximal coordinates (world-space)
       rigid_body_pos: Tensor[num_envs, num_bodies, 3]
       # quaternion xyzw ordering for the common state shared by all simulators
       # see conversion to different sim backends discussed later.
       rigid_body_rot: Tensor[num_envs, num_bodies, 4]
       rigid_body_vel: Tensor[num_envs, num_bodies, 3]
       rigid_body_ang_vel: Tensor[num_envs, num_bodies, 3]
       
       # Reduced coordinates (joint-space)
       dof_pos: Tensor[num_envs, num_dofs]
       dof_vel: Tensor[num_envs, num_dofs]
       
       # Forces
       dof_forces: Tensor[num_envs, num_dofs]  # Applied torques
       
       # Contacts
       rigid_body_contacts: Tensor[num_envs, num_bodies]  # Binary contact

**Why both maximal and reduced?**

* **Maximal** (rigid_body_*): Used for reward computation, observations that need
  world positions (e.g., distance to target, body heights)
* **Reduced** (dof_*): Used for action space, some observation types
  (real robot might only have reduced observations)

Accessing State Fields
~~~~~~~~~~~~~~~~~~~~~~

Common patterns:

.. code-block:: python

   # Root position (body 0)
   root_pos = robot_state.rigid_body_pos[:, 0, :]  # [num_envs, 3]
   
   # Root height
   root_height = robot_state.rigid_body_pos[:, 0, 2]  # [num_envs]
   
   # Foot positions
   left_foot_pos = robot_state.rigid_body_pos[:, left_foot_idx, :]
   
   # All body velocities
   body_vels = robot_state.rigid_body_vel  # [num_envs, num_bodies, 3]

ObjectState
-----------

``ObjectState`` contains the state of scene objects:

.. code-block:: python

   @dataclass
   class ObjectState:
       object_pos: Tensor[num_envs, num_objects, 3]
       object_rot: Tensor[num_envs, num_objects, 4]
       object_vel: Tensor[num_envs, num_objects, 3]
       object_ang_vel: Tensor[num_envs, num_objects, 3]

Used when training with interactive objects (vaulting, manipulation).

StateConversion
---------------

Utility for converting between common (base-simulator) and simulator-specific state representations, since different simulators may have different quaternion conventions and/or body/dof ordering.

Common Operations
-----------------

**Cloning state:**

.. code-block:: python

   state_copy = robot_state.clone()

**Converting to dict (for saving):**

.. code-block:: python

   state_dict = robot_state.to_dict()
   torch.save(state_dict, "state.motion")
   
   # Loading
   loaded_dict = torch.load("state.motion")
   robot_state = RobotState.from_dict(loaded_dict)


Motion Library State
--------------------

MotionLib stores reference states using the same structure:

.. code-block:: python

   # Query reference state at specific time
   ref_state: RobotState = motion_lib.get_motion_state(motion_ids, motion_times)
   
   # Compare to current state for rewards
   pos_error = current_state.rigid_body_pos - ref_state.rigid_body_pos

This allows direct comparison between simulation state and reference motion.

Next Steps
----------

* :doc:`abstractions` - Full system overview

