Partner-Resolved Contacts
=========================

Net contact force cannot tell whether a hand touched the torso, furniture, or
terrain. Contact reporting uses one partner-resolved matrix. In the experiment's
``configure_robot_and_simulator()``, set ``robot_cfg.contact_bodies`` to a nonempty
list of link names (or ``"all"``) to enable it. ``None`` or ``[]`` disables it.
There is no separate matrix enable flag or net-force reporting API.

The environment exposes one **current-only, privileged** matrix. It is simulator
ground truth, not a measured or estimated real-robot signal. Neither the
``clean`` (ideal sensor estimate) nor ``noisy`` sensor source contains it.
Using it as actor input requires a separate deployment measurement contract;
turning sensor noise off does not make the matrix deployable.

.. code-block:: text

   ctx["contact_forces"][environment, robot_body, partner, xyz]
   ctx["contact_body_index"]     # robot body name -> row
   ctx["contact_partner_index"]  # partner name -> column

``robot_cfg.contact_bodies`` selects the rows. ``robot_cfg.contact_partners``
selects the **contact against** columns, defaulting to ``["terrain"]``. The two
index maps are static axis metadata exposed through context bindings, not
policy observations or historical state. Available column names are:

* ``terrain``: the terrain mesh.
* ``robot/<body_name>``: each robot link, in common order.
* ``object/<slot>``: each per-environment SceneLib object slot.
* ``projectile/<slot>``: each configured projectile.

Select exact names or glob patterns; unknown names and patterns matching nothing
fail at simulator construction. Selected axes are compact and retain canonical
order, regardless of selector order. Always resolve indices through the maps,
including the terrain index. For example, only two foot rows against terrain
and one object, regardless of how many objects the scene contains:

.. code-block:: python

   robot_cfg.update_fields(
       contact_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
       contact_partners=["terrain", "object/7"],
   )
   # Matrix shape: [num_envs, 2, 2, 3].

``contact_partners=["robot/*"]`` selects only self-contact counterparts.
``["terrain"]`` selects only terrain. ``["object/*"]`` selects all scene slots.
A partner robot link need not also be selected as a source row. Backend sensors
allocate the selected rows and columns, not a full matrix sliced afterwards.
Selection affects reporting only: unselected pairs still collide physically.

An object slot is not an asset name: ``object/0`` can be a chair in one
environment and a box in another. SceneLib's ``get_per_object_valid_mask()``
distinguishes real objects from padding slots. Objects baked into the terrain
mesh belong to ``terrain``, not to a separate object column.

Values are **normal force vectors on the source body**, expressed in world axes
and newtons. Each cell has exactly three components (x, y, z), **not a 6D wrench**.
Tangential friction and torque are not included. A future wrench API would need
an explicit torque reference point. Including tangential friction would still
produce a 3D force, not require a wrench: total force is normal plus friction.
The normal-only contract here matches the filtered PhysX force API used by this
implementation; it is not a claim that friction is undesirable. Total-force
reporting requires consistent collection and validation across backends and is
outside this API's current scope. There is no aggregate ``everything`` column.
Multiple collision points for the same body/partner are summed into that cell.
This is not a matrix of individual collision points, nor an object-to-object
contact matrix.
Self-pairs and collision-disabled pairs remain zero; collision filtering is
unchanged.

Precisely, for source body ``b`` and partner ``p`` at the last physics substep:

.. code-block:: text

   F[b, p] = sum(f_normal[k] * n_on_b_world[k], contacts k between b and p)

``n_on_b_world[k]`` is a unit contact normal pointing in the direction of the
normal force on body ``b``; ``f_normal[k]`` is its magnitude in newtons.
``F[b, p]`` is the resultant **force vector**, not a unit normal and not a
scalar sum of magnitudes. Two upward contacts of 10 N and 15 N produce
``[0, 0, 25]`` N. Contacts on different faces of the same pair may partially or
fully cancel: ``[10, 0, 0] + [-10, 0, 0]`` gives zero even though two contacts
exist. The 3D aggregate cannot recover individual normals, contact locations,
contact counts, or moments. Forces against different partners remain separate
cells and cannot cancel until a consumer explicitly sums those columns.

Timing And Ownership
--------------------

The simulator refreshes the matrix once after each control step, using the
last physics substep. It does not accumulate impacts across decimation. After
reset, only the reset environments' entries are cleared; they remain zero until
the next step. ``simulator.get_contact_forces(env_ids=None)`` returns an owned
copy and preserves requested environment order. Static maps are available on
``simulator.contact_layout``.

The matrix is **not** stored in ``StateHistoryBuffer`` or under the
``current.*`` / ``historical.*`` namespaces. Observations, rewards, and
terminations bind to the same current ``EnvContext.contact_forces`` and
perform their own selection and reduction. Existing privileged history stores only compact, derived selected-partner
Boolean flags. Sensor-estimated histories do not invent contact measurements.
The impact reward retains one previous per-body force magnitude, not a previous
matrix. Evaluation snapshots preserve the current matrix along with simulator
runtime state, so restoring training also restores current contact observations.

Reporting is opt-in because storage grows with body and partner counts.
Disabled reporting leaves these context fields absent. Existing binding
validation rejects components that require them at environment construction.
Factories declare required entries in mapping-valued bindings using
``MdpComponent.required_keys``. The existing component binding check verifies
these entries at environment construction, before compilation or rollout.
A missing row or partner raises an error naming the component and missing key.
Requirements can be exact names or glob patterns: ``robot/*`` requires at least
one robot partner, not a full robot-by-robot matrix.
Contact-dependent components must enable reporting explicitly; disabling it
does not silently replace their inputs with zeros or turn contact-based fall
termination into height-only termination.

Foot-To-Terrain Matching
------------------------

Use this reward when the reference contact annotation specifically means
foot-to-floor contact:

.. code-block:: python

   from protomotions.envs.body_selection import BodySelection
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.rewards.meta import RewardMeta
   from protomotions.envs.rewards.regularization import (
       compute_terrain_foot_contact_match_rew,
   )

   feet = ["left_ankle_roll_link", "right_ankle_roll_link"]  # G1 example

   # In configure_robot_and_simulator():
   robot_cfg.update_fields(contact_bodies=feet, contact_partners=["terrain"])

   # In env_config(), after constructing cfg:
   cfg.ref_contact_smooth_window = 7  # Optional reference-label smoothing.
   cfg.reward_components["foot_floor_match"] = MdpComponent.from_kernel(
       compute_terrain_foot_contact_match_rew,
       meta=RewardMeta(weight=-0.1, zero_during_grace_period=True),
       required_keys={"contact_body_index": feet, "contact_partner_index": ["terrain"]},
       contact_body_ids=BodySelection.named(*feet),
       body_names=feet,
       force_threshold=0.01,  # N
   )

``body_names`` selects the feet to compare. ``contact_body_ids`` maps them to
the motion-library contact labels, which use the full common body order: the
build resolves the ``BodySelection`` to indices once and the saved configuration
keeps the names. Neither enables contact reporting for additional bodies. Use
actual link names for the robot. A name the robot lacks is refused at build; a
repeated name where ``required_keys`` is declared. Those foot rows and the
terrain column must be included in the simulator selection; a missing requested
name fails rather than pretending there was no contact. This check also catches
CLI overrides that remove a required row or column after the experiment builds
its components.

The kernel's defaults bind the configured matrix and the **current kinematic
reference's** ``mimic.ref_state.rigid_body_contacts``. MimicControl samples
these labels from MotionLib at the environment's current motion time, not a
future target time. Matrix rows are compact, while the reference uses the full
common body order, so the kernel addresses the two by name and by index:

.. code-block:: text

   foot_rows = [contact_body_index[name] for name in foot_names]
   reference_ids = contact_body_ids  # the resolved selection, in foot_names order
   foot_forces = forces[:, foot_rows, contact_partner_index["terrain"], :]
   measured_contact = norm(foot_forces, xyz) > threshold
   mismatch = sum(abs(measured_contact - reference_contacts[:, reference_ids]), feet)

The reward manager applies the weight and reset grace period. Smoothed,
fractional reference labels are supported. The motion data must provide contact
annotations in that common body ordering; the reward does not infer missing
labels or substitute simulated contacts. A foot touching a hand, a projectile,
or furniture cannot activate this reward's terrain-contact flag.
``compute_contact_match_rew`` uses all configured partner columns;
scene-support tasks should choose semantics matching their annotations.

Example Recipe Wiring
---------------------

``mimic/mlp.py``, ``mimic/fsq.py``, and ``mimic/mlp_complex_terrain.py`` explicitly
request only the robot's named left/right foot bodies against ``terrain`` and bind
``compute_terrain_foot_contact_match_rew``. Their tracking-error termination is
unchanged; they do not acquire an unrelated contact-based fall termination.
Recipes without contact observations, rewards, or terminations leave contact
reporting disabled. A task needs more rows or partners only when a component
actually consumes them.

For contact-based fall detection, name the forbidden bodies and partners:

.. code-block:: python

   from protomotions.envs.body_selection import BodySelection
   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.terminations.base import named_contact_fall_termination
   from protomotions.envs.terminations.rule import TerminationRule

   # Here only a low torso touching terrain constitutes a fall.
   robot_cfg.update_fields(contact_bodies=["torso_link"], contact_partners=["terrain"])
   cfg.termination_components["fall"] = TerminationRule(
       MdpComponent.from_kernel(
           named_contact_fall_termination,
           required_keys={
               "contact_body_index": ["torso_link"],
               "contact_partner_index": ["terrain"],
           },
           body_ids=BodySelection.named("torso_link"),
           body_names=["torso_link"],
           partner_names=["terrain"],
           termination_height=0.15,  # metres above ground
       )
   )

Name every forbidden non-foot body when the task needs whole-body fall detection.
``required_keys`` requires those rows; configuring only feet raises an error rather
than silently making torso falls invisible. The rule remains unwanted contact
AND a low forbidden body, with the existing two-step reset grace.

``TargetControlConfig.enable_fall_termination`` uses the same kernel: every body
outside ``robot_cfg.non_termination_contact_bodies`` is a forbidden body, and
only terrain contact counts. Register these rows and ``contact_partners=["terrain"]``
when enabling it. The control validates this coverage at construction, so a
feet-only matrix cannot silently stand in for whole-body fall detection.

Other Components
----------------

Resolve names to indices when building a component, and pass those indices to
its tensor kernel. For example, a hand-to-torso reward selects the hand row and
``robot/<torso_name>`` column. A task rewarding support on furniture selects
``object/<slot>``. Components decide whether to sum vectors, take the largest
force magnitude, average magnitudes, or threshold individual pairs. These
operations are deliberately not baked into the simulator API.

For an "any selected partner" flag, threshold each pair's force magnitude:

.. code-block:: python

   selected = forces[:, body_ids, :, :][:, :, partner_ids, :]
   pair_magnitudes = selected.norm(dim=-1)  # [env, selected body, selected partner]
   has_contact = (pair_magnitudes > threshold).any(dim=-1)

Summing ``pair_magnitudes`` before thresholding instead measures aggregate
strength across the selected partners. Summing force vectors first measures net
force, so opposing forces can cancel. The matrix already sums collision points
within each pair and excludes friction; these reductions are not equivalent to
counting raw collision points or reproducing a net-force contact flag.

Migration And Changed Semantics
-------------------------------

All current contact observations, matching rewards, impact rewards, and fall
terminations read the matrix. Their shared ``contact_flags`` helper thresholds
the **maximum pair magnitude** per body (default 0.01 N), not a vector sum.
The impact penalty compares this maximum magnitude against the previous step;
it no longer includes tangential friction. Fall detection retains its contact
AND height rule and reset grace period. Contact detection covers only the
configured rows and partners. Register each forbidden body requested by the fall
factory; selecting only feet cannot detect a torso contact.
Review observation sizes when changing row selection, as the default contact
observation contains one flag per selected row.

``get_bodies_contact_buf()``, ``get_binary_body_contacts()``, the object net-force
getter, and ``RobotState.rigid_body_contact_forces`` are removed. Rebuild the
saved configs with the updated experiment code: the serialized
components now include ``required_keys``, and contact bindings/factory signatures
have changed. A stale ``resolved_configs.json`` bundle is not migrated;
regenerate it. There are no compatibility aliases. ``RobotState.rigid_body_contacts`` remains a motion annotation, not a
simulator measurement. Recording and predicted evaluation motions derive
any-partner annotations from the matrix, including furniture contacts, only
when all bodies and all available partners are selected. Partial or disabled
reporting omits these annotations rather than labeling unmeasured contacts false.
Reward matching can still use selected rows against the full reference labels.

The sensor-only deployment exporters do not supply contact measurements. A
policy depending on simulated contact observations requires an explicit host
measurement contract; it must not be exported with fabricated zero contacts.

Backend Scope
--------------

* IsaacLab/PhysX reports robot contacts with terrain, robot links, scene slots,
  and projectiles. One native PhysX view uses exact per-environment rigid-body
  paths, in common body order. Exact paths avoid confusing a link with a
  same-named visual child. This adds no sensor history or per-link sensor objects.
* MuJoCo supports robot links, horizontal world ground planes, and projectiles,
  following its existing single-environment, flat-terrain restriction. It does
  not support SceneLib objects. Unregistered world-attached obstacles and wall
  planes are not silently classified as terrain.
* Newton reduces per-contact forces and normals using physical shape IDs,
  mapping them to the selected robot rows and terrain/robot/projectile columns.
  It projects out friction before summing each pair. SceneLib objects are not
  supported by this backend.
* IsaacGym contact reporting is unsupported: its GPU tensor API does not
  identify collision partners. Construction asserts if ``contact_bodies`` is
  nonempty. Use a supported backend, or disable reporting and remove
  contact-dependent components. Physics collision handling is unaffected.

This API changes contact reporting, not collision physics, policy architecture,
or learned contact prediction.

Validation
----------

Run ``python scripts/check_contact_matrix.py --scene`` in the IsaacLab Python
environment with the G1 mesh assets installed. It exercises real terrain,
robot-to-robot, scene-object, and projectile contacts, replica isolation,
environment selection, and partial reset. No policy or motion dataset is needed.
Add ``--subset`` to verify only two foot rows and the terrain/object columns are
allocated, while unselected projectile contacts remain physically active.

For a matched overhead measurement, run each command in a fresh process:

.. code-block:: bash

   python scripts/check_contact_matrix.py --benchmark --num-envs 4096
   python scripts/check_contact_matrix.py --benchmark --num-envs 4096 --no-matrix

The benchmark reports initialization time and 100 control steps after ten
warmup steps. The timed path includes physics and the current-context matrix
copy, but not a policy, rewards, or training. Both runs use the same increased
PhysX contact-patch capacity for the simultaneous robot drop. The matrix alone
uses ``num_envs * num_bodies * num_partners * 3 * 4`` bytes in float32, excluding
the backend's buffers and owned context copies.
