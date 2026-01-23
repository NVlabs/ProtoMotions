Environment Context
===================

The environment context is a dictionary that connects all components in ProtoMotions.
It contains the current state of the simulation and task-specific variables that
observations, rewards, and terminations need to compute their outputs.

Overview
--------

Each timestep, the environment builds a context dictionary containing:

1. **Current state** from the simulator
2. **Historical state** from the history buffer
3. **Task context** from control components
4. **Environment parameters** (dt, limits, body indices)

Components then access this context to compute their outputs:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    Environment Context                      │
   ├─────────────────────────────────────────────────────────────┤
   │  Current State         │  Historical State                  │
   │  ─────────────         │  ────────────────                  │
   │  current_state_*       │  historical_*                      │
   │  noisy_current_state_* │  noisy_historical_*                │
   ├─────────────────────────────────────────────────────────────┤
   │  Control Context       │  Environment Parameters            │
   │  ───────────────       │  ──────────────────────            │
   │  ref_state (mimic)     │  dt                                │
   │  tar_dir (steering)    │  soft_dof_limits_*                 │
   │  tar_pos (path)        │  contact_body_ids                  │
   └─────────────────────────────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  Observations   │    │     Rewards     │    │  Terminations   │
   └─────────────────┘    └─────────────────┘    └─────────────────┘

Context Categories
------------------

Current State Variables
~~~~~~~~~~~~~~~~~~~~~~~

Ground-truth state from the simulator. Use for critic observations and rewards.

.. code-block:: python

   # Rigid body state [num_envs, num_bodies, ...]
   "current_state_rigid_body_pos"       # [envs, bodies, 3]
   "current_state_rigid_body_rot"       # [envs, bodies, 4] quaternion
   "current_state_rigid_body_vel"       # [envs, bodies, 3]
   "current_state_rigid_body_ang_vel"   # [envs, bodies, 3]
   "current_state_rigid_body_contacts"  # [envs, bodies] contact flags
   
   # DOF state [num_envs, num_dofs]
   "current_state_dof_pos"              # Joint positions
   "current_state_dof_vel"              # Joint velocities
   "current_state_dof_forces"           # Joint forces (if available)
   
   # Root convenience aliases
   "current_state_root_pos"             # [envs, 3] = rigid_body_pos[:, 0]
   "current_state_root_rot"             # [envs, 4]
   "current_state_root_ang_vel"         # [envs, 3]
   "current_state_root_local_ang_vel"   # [envs, 3] in local frame
   "current_state_root_height"          # [envs] z coordinate
   
   # Anchor body (typically pelvis for humanoids)
   "current_state_anchor_pos"           # [envs, 3]
   "current_state_anchor_rot"           # [envs, 4]

Noisy State Variables
~~~~~~~~~~~~~~~~~~~~~

State with observation noise applied. Use for actor observations during training
to improve sim-to-real transfer.

.. code-block:: python

   # Same structure as current_state_*, but with noisy_ prefix
   "noisy_current_state_rigid_body_pos"
   "noisy_current_state_dof_pos"
   "noisy_current_state_root_rot"
   # ... etc

Historical State Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~

Past states from the history buffer. Use for temporal observations.

.. code-block:: python

   # Historical rigid body state [envs, history_steps-1, bodies, ...]
   "historical_rigid_body_pos"
   "historical_rigid_body_rot"
   "historical_dof_pos"
   "historical_dof_vel"
   
   # Historical actions [envs, history_steps-1, num_actions]
   "historical_actions"
   
   # Noisy versions for actor
   "noisy_historical_rigid_body_pos"
   "noisy_historical_dof_pos"

Action Variables
~~~~~~~~~~~~~~~~

.. code-block:: python

   "current_actions"     # Actions being applied this step
   "previous_actions"    # Actions from previous step

Environment Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   "dt"                              # Timestep duration
   "soft_dof_limits_lower"           # Joint limits (lower)
   "soft_dof_limits_upper"           # Joint limits (upper)
   "contact_body_ids"                # Body indices for contact detection
   "ground_heights_beneath_root"     # Terrain height at root position
   "body_contacts"                   # Contact state per body

Control Component Context
-------------------------

Control components add task-specific variables to the context via ``get_context()``.

MimicControl
~~~~~~~~~~~~

Motion tracking context for imitation learning.

.. code-block:: python

   # Reference state from motion library
   "ref_state"                       # RobotState at current motion time
   "mimic_ref_rigid_body_pos"        # [envs, bodies, 3]
   "mimic_ref_rigid_body_rot"        # [envs, bodies, 4]
   "mimic_ref_rigid_body_vel"        # [envs, bodies, 3]
   "mimic_ref_dof_pos"               # [envs, dofs]
   "mimic_ref_dof_vel"               # [envs, dofs]
   "mimic_ref_contacts"              # [envs, bodies]
   
   # Motion tracking state
   "motion_ids"                      # Current motion ID per env
   "motion_times"                    # Current time in motion

SteeringControl
~~~~~~~~~~~~~~~

Locomotion command context.

.. code-block:: python

   "tar_dir"        # [envs, 2] target heading direction (x, y)
   "tar_speed"      # [envs] target speed
   "tar_face_dir"   # [envs, 2] target facing direction
   "prev_root_pos"  # [envs, 3] root position from previous step

PathFollowerControl
~~~~~~~~~~~~~~~~~~~

Path following context.

.. code-block:: python

   "tar_pos"            # [envs, 3] current target position
   "head_pos"           # [envs, 3] head position
   "traj_samples"       # [envs, num_samples, 3] future path points
   "height_conditioned" # [envs] whether height tracking is enabled

Using Context in Components
---------------------------

Components access context variables through their ``variables`` mapping:

.. code-block:: python

   from protomotions.envs.rewards import gt_rew_factory
   
   # This reward compares current_state to ref_state from mimic control
   reward_components = {
       "gt_rew": gt_rew_factory(weight=0.5, coefficient=-100.0)
   }

The factory creates a config that maps function arguments to context keys:

.. code-block:: python

   RewardComponentConfig(
       function=body_position_tracking_reward,
       variables={
           "current_pos": "current_state_rigid_body_pos",  # From base context
           "target_pos": "mimic_ref_rigid_body_pos",       # From MimicControl
       },
       weight=0.5,
   )

At evaluation time, the manager resolves these keys from the context dictionary
and calls the function with the resolved tensors.

Adding Custom Context
---------------------

To add custom variables to the context, create a control component:

.. code-block:: python

   class MyControlComponent(ControlComponent):
       def get_context(self) -> Dict[str, Any]:
           return {
               "my_target": self._compute_target(),
               "my_state": self._current_state,
           }

These variables become available to all observation, reward, and termination
components.

Next Steps
----------

* :doc:`abstractions` - Component system details
* :doc:`simulator_state` - SimulatorState representation

