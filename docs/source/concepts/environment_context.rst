Environment Context
===================

The environment context gives observations, rewards and terminations one place
to read simulation state and task variables from. ``env.context`` is a flat,
read-only map from a path string (``"current.rigid_body_pos"``) to a value;
``EnvContext`` in ``context_views.py`` is the vocabulary of those paths, so
bindings can spell them with IDE autocomplete and the environment can check
them at build time.

Overview
--------

Each timestep, the environment builds one context: a read-only dictionary
built once per step from the simulator state, the history buffer and the
control components. It holds:

1. **Current state** - Ground-truth and sensor-estimated robot state
2. **Historical state** - Past states from the history buffer
3. **Task context** - Task-specific variables from control components
4. **Environment parameters** - dt, contact tracking, etc.

Components bind their kernels to context paths using ``MdpComponent``:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │            env.context: one flat, read-only map             │
   ├─────────────────────────────────────────────────────────────┤
   │  Current state         │  Historical state                  │
   │  ───────────────────   │  ──────────────────────            │
   │  current.*             │  historical.*                      │
   │  clean.*               │  clean_historical.*                │
   │  noisy.*               │  noisy_historical.*                │
   ├─────────────────────────────────────────────────────────────┤
   │  Control namespaces    │  Top-level names                   │
   │  ───────────────────   │  ──────────────────────            │
   │  mimic.*  steering.*   │  dt, env_ids, progress_buf         │
   │  masked_mimic.*        │  ground_heights, contact_body_ids  │
   │  path.*  target.*      │  contact_forces, current_action    │
   │  scene_boxes.*         │                                    │
   └─────────────────────────────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  Observations   │    │     Rewards     │    │  Terminations   │
   │  (MdpComponent) │    │  (MdpComponent) │    │  (MdpComponent) │
   └─────────────────┘    └─────────────────┘    └─────────────────┘

MdpComponent Pattern
---------------------

Components use ``MdpComponent`` to bind pure tensor kernels to context paths:

.. code-block:: python

   from protomotions.envs.context_views import EnvContext
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.rewards.tracking import compute_gt_rew
   
   reward_components = {
       "gt_rew": MdpComponent(
           compute_func=compute_gt_rew,                      # Pure tensor function
           dynamic_vars={                                  # Map params to context paths
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

Context Namespaces
------------------

The vocabulary groups the published names by namespace. Each namespace
class below only declares names; the values come from the simulator state,
the history buffer and the control components
(``protomotions/envs/base_env/publishers.py`` and the control components'
``publish``).

Current state: PrivilegedStateView and SensorStateView
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current robot state has three explicit sources:

* ``EnvContext.current`` is privileged simulator truth.
* ``EnvContext.clean`` is reconstructed from the declared sensors with no added
  measurement noise or drift. It is not unrestricted simulator truth.
* ``EnvContext.noisy`` uses the same sensor suite and reconstruction after
  perturbing the readings.

Clean and noisy exist only when a sensor suite is configured. Both omit names
that the suite cannot estimate; for example, neither exposes world body positions
without localization. The names below are the declared vocabulary, not a promise
that every source provides every one: a name that needs something says so in
``context_views.py`` (``FieldPath(requires="localization")`` for the sensor
sources' body positions, ``requires="contacts"`` for the privileged contact
flags). A binding to a missing name fails before anything runs, with the
reason it is missing.

Privileged state is useful for rewards and critics. Clean/noisy pairs isolate
sensor corruption for consistency losses such as L2C2 without giving the clean
side additional simulator information. See :ref:`sensor-observation-sources`
for the construction, uses and limits of each source.

.. code-block:: python

   # Access via EnvContext.current, EnvContext.clean or EnvContext.noisy
   
   # Rigid body state
   .rigid_body_pos        # [num_envs, num_bodies, 3]     sensors: needs localization
   .rigid_body_rot        # [num_envs, num_bodies, 4] quaternion (w-last)
   .rigid_body_vel        # [num_envs, num_bodies, 3]     sensors: needs base velocity
   .rigid_body_ang_vel    # [num_envs, num_bodies, 3]
   .rigid_body_contacts   # [num_envs, num_contact_bodies] boolean; current only, needs contact bodies
   
   # DOF state
   .dof_pos               # [num_envs, num_dofs]
   .dof_vel               # [num_envs, num_dofs]
   .dof_forces            # [num_envs, num_dofs]           current only
   
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

Past state: PrivilegedHistoricalView and SensorHistoricalView
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Past states for temporal observations, present when a bound observation,
reward or termination reads a past state (the history ring sizes itself from
the bindings' step selections):

.. code-block:: python

   # Access via EnvContext.historical or EnvContext.noisy_historical
   
   .rigid_body_pos        # [num_envs, history_steps, num_bodies, 3]
   .rigid_body_rot        # [num_envs, history_steps, num_bodies, 4]
   .dof_pos               # [num_envs, history_steps, num_dofs]
   .dof_vel               # [num_envs, history_steps, num_dofs]
   .raw_actions           # [num_envs, history_steps, action_dim]
   .processed_actions     # [num_envs, history_steps, action_dim]
   .ground_heights        # [num_envs, history_steps]
   .body_contacts         # [num_envs, history_steps, num_contact_bodies]; historical only

Control Component Namespaces
----------------------------

Each control component publishes the names of its own namespace from
``publish(env_ids)``.

MimicContext
~~~~~~~~~~~~

Motion tracking context for imitation learning:

.. code-block:: python

   # Access via EnvContext.mimic
   
   # Reference state (at current time)
   .ref_state             # .ref_state.rigid_body_pos, .ref_state.dof_pos, ...
   .ref_anchor_pos        # [num_envs, 3] precomputed
   .anchor_idx            # int - index of the anchor body

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
   .env_ids                           # rows this context covers (None = every env)
   .progress_buf                      # [num_envs] steps since the episode began
   .ground_heights                    # [num_envs] terrain height under the anchor
   .contact_forces                    # [num_envs, robot_body, partner, 3], world N
   .contact_body_index                # body name -> matrix row (static)
   .contact_partner_index             # partner name -> matrix column (static)
   .contact_body_ids                  # [num_contact_bodies] tracked body indices
   .current_action                    # [num_envs, action_dim]
   .current_processed_action          # [num_envs, action_dim]
   .previous_action                   # [num_envs, action_dim]
   .previous_processed_action         # [num_envs, action_dim]
   .prev_contact_force_magnitudes     # [num_envs, num_contact_bodies]

Using Context in MdpComponent
-------------------------------

**Class access** (for configuration):

.. code-block:: python

   # Returns FieldPath objects with .path property
   EnvContext.current.rigid_body_pos           # FieldPath("current.rigid_body_pos")
   EnvContext.mimic.future_pos                 # FieldPath("mimic.future_pos")
   EnvContext.ground_heights                   # FieldPath("ground_heights")

**Runtime access** (the flat map):

.. code-block:: python

   # The context is a read-only mapping keyed by the same path strings
   ctx = env.context                           # a FrozenContext, built once per step
   ctx["current.rigid_body_pos"]               # Tensor [num_envs, num_bodies, 3]
   ctx["mimic.future_pos"]                     # Tensor [num_envs, future_steps, ...]
   "noisy.dof_pos" in ctx                      # False without a sensor suite

**In experiment configs**:

.. code-block:: python

   from protomotions.envs.context_views import EnvContext
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
   
   observation_components = {
       "max_coords_obs": MdpComponent(
           compute_func=compute_humanoid_max_coords_observations,
           dynamic_vars={
               "body_pos": EnvContext.current.rigid_body_pos,     # Type-safe!
               "body_rot": EnvContext.current.rigid_body_rot,
               "body_vel": EnvContext.current.rigid_body_vel,
               "body_ang_vel": EnvContext.current.rigid_body_ang_vel,
               "ground_height": EnvContext.ground_heights,
               "body_contacts": EnvContext.current.rigid_body_contacts,
           },
           static_params={"local_obs": True, "root_height_obs": True, "w_last": True},
       ),
   }

Checking bindings before anything is built
------------------------------------------

``train_agent.py`` and ``inference_agent.py`` check your bindings against
the config before the simulator starts
(``protomotions.envs.context_check.run_config_binding_check``). A binding
your configuration can never satisfy fails in seconds with the reason: a
sensor name without a sensor suite ("this config has no sensor suite"), a
body position from a suite without localization ("the sensor suite has no
localization body"), a history name bound without a step selection,
a contact name for a robot without contact bodies, a task namespace with no
control component for it, or a masked-mimic modality whose config switch is
off. Names that depend on the motion file (the reference motion's contact
labels and local rotations) are checked when the environment is built, and
the entry points log them as such. What each name needs is the
``requires=`` tag on its declaration in ``context_views.py``; how a
configuration satisfies a tag is the table in
``protomotions/envs/context_requirements.py``. Every example experiment that
builds offline is run through the check in the test suite.

Adding Custom Context Names
----------------------------

To add custom variables to the context, declare their names in the
vocabulary and publish them from a control component, which lists the
namespaces it publishes in ``namespaces``. A name that only some
configurations of your component publish carries ``requires="<switch>"``,
where the switch is a field of your config; the checks read
``config.<switch>`` to decide whether the name is expected, and refuse a
binding to it when the switch is off, naming the switch. The environment
checks the declarations against each other when it is built. A namespace a
component publishes without declaring it (or declares without publishing)
is an error naming the component. A published name nobody declared, and a
tagged name published while its switch is off, are errors naming the
publisher or control component that published the name and, for the tagged
name, every tag that applies and the fact that is missing. A declared name
without a tag that nobody published is listed by namespace.

.. code-block:: python

   # In protomotions/envs/context_views.py
   from protomotions.envs.context_paths import FieldPath

   class MyCustomContext:
       """Names of my task."""

       target_pos: Tensor = FieldPath()
       target_vel: Tensor = FieldPath()
       # Only when MyControlConfig.track_heading is on.
       target_heading: Tensor = FieldPath(requires="track_heading")

   class EnvContext:
       ...
       my_custom: MyCustomContext = control_namespace(MyCustomContext)

   # In the control component
   class MyControlComponent(ControlComponent):
       namespaces = ("my_custom",)

       def publish(self, env_ids: Optional[Tensor]) -> Dict[str, Any]:
           rows = slice(None) if env_ids is None else env_ids
           names = {
               "my_custom.target_pos": self._target_pos[rows],
               "my_custom.target_vel": self._target_vel[rows],
           }
           if self.config.track_heading:
               names["my_custom.target_heading"] = self._target_heading[rows]
           return names

Then use it in MdpComponent bindings:

.. code-block:: python

   observation_components = {
       "custom_obs": MdpComponent(
           compute_func=compute_custom_obs,
           dynamic_vars={
               "target_pos": EnvContext.my_custom.target_pos,  # Type-safe!
           },
       ),
   }

Deployable G1 observation layout
-------------------------------------

``deployable_proprioception_observations`` packs encoder/IMU-derived FK tokens,
reduced sensor state and incoming raw actions. History lags are 8, 4, 3, 2, 1;
each token carries its own incoming action and age. It consumes the shared
sensor history and does not sample noise or maintain another sensor cache.
Privileged teacher/posterior observations remain separate.

The G1 composition helper lives in
``examples.experiments.masked_mimic.deployable_g1_observations``. Select the
five-body subset before building goals, declare a physical sensor suite and
noise levels in the experiment, then add these observations. Global goals
compute the current body poses themselves: the recipe binds one
``BakedAnchorFK`` kernel (passed as ``fk=``, as the history observation
already takes it) to ``noisy.dof_pos``, ``noisy.anchor_pos`` and
``noisy.anchor_rot``. Its world poses equal the ``noisy.rigid_body_pos`` and
``noisy.rigid_body_rot`` the sensor runtime publishes, because it is the same
FK composed on the same anchor pose. Body poses are therefore never graph
inputs of a robot export: the policy graph takes the raw readings, computes
the body poses inside and returns only its four control outputs (a simulation
host may feed them; next paragraph). A host that needs
the body poses outside the policy call exports the same kernel as a second
small graph, ``fk.onnx``, with ``deployment/export_fk_to_onnx.py``; it is
rebuilt from the config bundle beside the checkpoint, so there is one FK
implementation and the two graphs agree by construction. Binding validation
still rejects suites without localization, which the anchor position needs.

A host kind belongs to a host. The sensor views carry only what a robot has
(joint readings, IMU, localization, past actions, goals); the privileged view's
body state (``current.rigid_body_*``, ``current.ground_heights``) carries the
``simulated_*`` kinds, which only ``export_to_onnx.py --host simulation``
accepts, for a game engine, a browser MuJoCo or a sim-to-sim check. A recipe
such as the SOMA tracker, whose observations read the simulator's body state,
exports only that way; the default robot export refuses it, naming the kind
and the host that could feed it.

Height-free goals ground current and reference body origins independently at
their own minimum Z. XY localization error remains. This is a flat-ground
observation contract, not terrain-relative foot height. At inference retain
the same bindings and use ``SensorNoise.exact("inference")`` when desired.
Regenerate resolved configs after changing the observation layout; existing
policy weights cannot adopt different input widths.

Next Steps
----------

* :doc:`abstractions` - Component system details
* :doc:`simulator_state` - ``RobotState``, ``ResetState`` and ``ObjectState``
