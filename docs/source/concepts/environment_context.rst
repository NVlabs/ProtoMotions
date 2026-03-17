Environment Context
===================

The environment context provides typed access to all simulation state and task variables
that observations, rewards, and terminations need. ProtoMotions uses a descriptor-based
pattern for type-safe context paths with IDE autocomplete.

Overview
--------

Each timestep, the environment builds an ``EnvContext`` instance containing:

1. **Current state** - Ground-truth and noisy robot state views
2. **Historical state** - Past states from the history buffer
3. **Task context** - Task-specific variables from control components
4. **Environment parameters** - dt, contact tracking, etc.

Components bind their kernels to context paths using ``ContextRouter``:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                      EnvContext                             │
   ├─────────────────────────────────────────────────────────────┤
   │  Current State Views   │  Historical State Views            │
   │  ───────────────────   │  ──────────────────────            │
   │  current: CurrentState │  historical: HistoricalView        │
   │  noisy: CurrentState   │  noisy_historical: NoisyHistView   │
   ├─────────────────────────────────────────────────────────────┤
   │  Control Context       │  Environment Parameters            │
   │  ───────────────────   │  ──────────────────────            │
   │  mimic: MimicContext      │  dt: float                         │
   │  steering: SteeringContext│  ground_heights: Tensor            │
   │  path: PathContext        │  contact_body_ids: Tensor          │
   └─────────────────────────────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  Observations   │    │     Rewards     │    │  Terminations   │
   │  (ContextRouter)│    │  (ContextRouter)│    │  (ContextRouter)│
   └─────────────────┘    └─────────────────┘    └─────────────────┘

ContextRouter Pattern
---------------------

Components use ``ContextRouter`` to bind pure tensor kernels to context paths:

.. code-block:: python

   from protomotions.envs.context_views import EnvContext
   from protomotions.envs.context_router import ContextRouter
   from protomotions.envs.rewards import compute_gt_rew
   
   reward_components = {
       "gt_rew": ContextRouter(
           kernel=compute_gt_rew,                      # Pure tensor function
           dynamic_bindings={                                  # Map params to context paths
               "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
               "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
           },
           static_params={"weight": 0.5, "coefficient": -100.0},  # Static parameters
       ),
   }

**Key benefits:**

- **Type-safe**: IDE autocomplete for context paths (``EnvContext.current.rigid_body_pos``)
- **Explicit**: Bindings show exactly what data each kernel needs
- **Testable**: Kernels are pure functions that can be tested independently
- **Exportable**: ONNX export uses ``get_bindings_dict()`` for input mapping

Context Views
-------------

EnvContext provides typed views into simulation state:

CurrentStateView
~~~~~~~~~~~~~~~~

Current robot state (ground-truth for critic/rewards, noisy for actor):

.. code-block:: python

   # Access via EnvContext.current or EnvContext.noisy
   
   # Rigid body state
   .rigid_body_pos        # [num_envs, num_bodies, 3]
   .rigid_body_rot        # [num_envs, num_bodies, 4] quaternion (w-last)
   .rigid_body_vel        # [num_envs, num_bodies, 3]
   .rigid_body_ang_vel    # [num_envs, num_bodies, 3]
   .rigid_body_contacts   # [num_envs, num_bodies] boolean
   
   # DOF state
   .dof_pos               # [num_envs, num_dofs]
   .dof_vel               # [num_envs, num_dofs]
   .dof_forces            # [num_envs, num_dofs]
   
   # Root properties (precomputed)
   .root_pos              # [num_envs, 3]
   .root_rot              # [num_envs, 4]
   .root_vel              # [num_envs, 3]
   .root_ang_vel          # [num_envs, 3]
   .root_height           # [num_envs]
   .root_local_ang_vel    # [num_envs, 3] in local frame
   
   # Anchor properties (typically pelvis, precomputed)
   .anchor_pos            # [num_envs, 3]
   .anchor_rot            # [num_envs, 4]
   .anchor_vel            # [num_envs, 3]
   .anchor_ang_vel        # [num_envs, 3]
   .anchor_local_ang_vel  # [num_envs, 3]

HistoricalView
~~~~~~~~~~~~~~

Past states for temporal observations:

.. code-block:: python

   # Access via EnvContext.historical or EnvContext.noisy_historical
   
   .rigid_body_pos        # [num_envs, history_steps, num_bodies, 3]
   .rigid_body_rot        # [num_envs, history_steps, num_bodies, 4]
   .dof_pos               # [num_envs, history_steps, num_dofs]
   .dof_vel               # [num_envs, history_steps, num_dofs]
   .actions               # [num_envs, history_steps, action_dim]
   .processed_actions     # [num_envs, history_steps, action_dim]
   .ground_heights        # [num_envs, history_steps]
   .body_contacts         # [num_envs, history_steps, num_contact_bodies]

Control Component Views
-----------------------

Control components populate task-specific views in the context.

MimicContext
~~~~~~~~~~~~

Motion tracking context for imitation learning:

.. code-block:: python

   # Access via EnvContext.mimic
   
   # Reference state (at current time)
   .ref_state             # RobotState with rigid_body_pos, dof_pos, etc.
   .ref_root_height       # [num_envs] precomputed
   .ref_anchor_pos        # [num_envs, 3] precomputed
   .ref_anchor_rot        # [num_envs, 4] precomputed
   
   # Future target poses (multi-step)
   .future_pos            # [num_envs, future_steps, num_bodies, 3]
   .future_rot            # [num_envs, future_steps, num_bodies, 4]
   .future_vel            # [num_envs, future_steps, num_bodies, 3]
   .future_ang_vel        # [num_envs, future_steps, num_bodies, 3]
   .future_dof_pos        # [num_envs, future_steps, num_dofs]
   .future_dof_vel        # [num_envs, future_steps, num_dofs]
   
   # Convenience properties (precomputed)
   .future_root_pos       # [num_envs, future_steps, 3]
   .future_anchor_pos     # [num_envs, future_steps, 3]

SteeringContext
~~~~~~~~~~~~~~~

Locomotion command context:

.. code-block:: python

   # Access via EnvContext.steering
   
   .tar_dir               # [num_envs, 2] target heading direction
   .tar_dir_theta         # [num_envs] target direction as angle
   .tar_speed             # [num_envs] target speed
   .tar_face_dir          # [num_envs, 2] target facing direction
   .prev_root_pos         # [num_envs, 3] root position from previous step

PathContext
~~~~~~~~~~~

Path following context:

.. code-block:: python

   # Access via EnvContext.path
   
   .tar_pos               # [num_envs, 3] current target position
   .head_pos              # [num_envs, 3] head body position
   .traj_samples          # [num_envs, num_samples, 3] future waypoints
   .height_conditioned    # bool - whether height tracking is enabled
   .head_body_id          # int - index of head body

Environment Parameters
~~~~~~~~~~~~~~~~~~~~~~

Direct fields on EnvContext:

.. code-block:: python

   .dt                                # float - simulation timestep
   .ground_heights                    # [num_envs] terrain height
   .noisy_ground_heights              # [num_envs] noisy terrain height
   .body_contacts                     # [num_envs, num_contact_bodies]
   .contact_body_ids                  # [num_contact_bodies] tracked body indices
   .current_processed_action          # [num_envs, action_dim]
   .previous_action                   # [num_envs, action_dim]
   .previous_processed_action         # [num_envs, action_dim]
   .current_contact_force_magnitudes  # [num_envs, num_bodies]
   .prev_contact_force_magnitudes     # [num_envs, num_bodies]

Using Context in ContextRouter
-------------------------------

**Class access** (for configuration):

.. code-block:: python

   # Returns FieldPath objects with .path property
   EnvContext.current.rigid_body_pos           # FieldPath("current.rigid_body_pos")
   EnvContext.mimic.future_pos                 # FieldPath("mimic.future_pos")
   EnvContext.ground_heights                   # FieldPath("ground_heights")

**Instance access** (at runtime):

.. code-block:: python

   # Returns actual tensor values
   ctx = env.context                           # EnvContext instance
   ctx.current.rigid_body_pos                  # Tensor [num_envs, num_bodies, 3]
   ctx.mimic.future_pos                        # Tensor [num_envs, future_steps, ...]

**In experiment configs**:

.. code-block:: python

   from protomotions.envs.context_views import EnvContext
   from protomotions.envs.context_router import ContextRouter
   from protomotions.envs.obs import compute_humanoid_max_coords_observations
   
   observation_components = {
       "max_coords_obs": ContextRouter(
           kernel=compute_humanoid_max_coords_observations,
           dynamic_bindings={
               "body_pos": EnvContext.current.rigid_body_pos,     # Type-safe!
               "body_rot": EnvContext.current.rigid_body_rot,
               "body_vel": EnvContext.current.rigid_body_vel,
               "body_ang_vel": EnvContext.current.rigid_body_ang_vel,
               "ground_height": EnvContext.ground_heights,
               "body_contacts": EnvContext.body_contacts,
           },
           static_params={"local_obs": True, "root_height_obs": True, "w_last": True},
       ),
   }

Adding Custom Context Views
----------------------------

To add custom variables to the context, create a control component that
populates the EnvContext with a custom view:

.. code-block:: python

   from protomotions.envs.context_paths import FieldPath, NestedField
   
   class MyCustomView:
       """Custom view for my task."""
       
       # Define fields as FieldPath descriptors
       target_pos: Tensor = FieldPath()
       target_vel: Tensor = FieldPath()
       
       def __init__(self, target_pos, target_vel):
           self.target_pos = target_pos
           self.target_vel = target_vel
   
   class MyControlComponent(ControlComponent):
       def populate_context(self, ctx: EnvContext) -> None:
           # Add your custom view to the context
           ctx.my_custom = MyCustomView(
               target_pos=self._compute_target_pos(),
               target_vel=self._compute_target_vel(),
           )

Then use it in ContextRouter bindings:

.. code-block:: python

   observation_components = {
       "custom_obs": ContextRouter(
           kernel=compute_custom_obs,
           dynamic_bindings={
               "target_pos": EnvContext.my_custom.target_pos,  # Type-safe!
           },
       ),
   }

Next Steps
----------

* :doc:`abstractions` - Component system details
* :doc:`simulator_state` - SimulatorState representation
