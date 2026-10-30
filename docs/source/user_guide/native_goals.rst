Native MaskedMimic Goals
============================

The native adapter supplies the same timed world-pose and velocity modalities
used in training. It does not change actions, PD gains, simulator state or the
checkpoint's network. It requires a checkpoint trained with anchor submode
observations; steering additionally requires the velocity modality.

The task is chosen inside an inference config, not on the command line. The
shipped preset ``examples/inference/masked_mimic.py`` starts in ``idle`` with
get-up recovery enabled, which needs a student trained with local get-up
conditioning:

.. code-block:: bash

   python protomotions/inference_agent.py --simulator mujoco \
     --checkpoint results/student/last.ckpt --motion-file data/motions.pt \
     --num-envs 1 --inference-config examples/inference/masked_mimic.py

``--inference-config`` loads a Python file defining
``apply_inference_overrides(configs, args)``, where ``configs`` is the
``ExperimentConfigs`` view (``configs.env``, ``configs.robot``, ...). The hook
edits the derived inference view in place, after the framework defaults, the
CLI ``--overrides`` and the recipe's own hook have been applied. It does not
rebuild the network or reload the training experiment. Only load trusted
Python files. Errors in the hook stop startup.

Task parameters cannot be set with ``--overrides``. The overrides are applied
to the saved training config before the hook runs, and the training-time
``MaskedMimicControlConfig`` has no ``task``, ``command_source``, ``getup`` or
``reach_body_name`` field, so such an override is refused as an undeclared
field; a hook that changes a value given in ``--overrides`` is refused as well.
To pick a task, copy the preset and edit its ``configure_masked_mimic_task``
call. The controller config that call installs is the one with those fields,
so anything else is set on it right after the call:

.. code-block:: python

   from protomotions.envs.control.masked_mimic_tasks.controller import (
       GetupRecoveryConfig,
       configure_masked_mimic_task,
   )


   def apply_inference_overrides(configs, args):
       if args.full_eval:
           raise ValueError("Interactive goals cannot run full evaluation")
       # Steering from the keyboard, for a student trained with get-up goals.
       # Pass getup=None for a student without get-up conditioning.
       configure_masked_mimic_task(
           configs.env, configs.robot, "steering", getup=GetupRecoveryConfig()
       )
       configs.env.control_components["masked_mimic"].command_source = "keyboard"

New policy demos belong in their own inference config, not in the shared
runner's argument parser. The example, rather than the runner, rejects
tracking evaluation for these interactive goals.

The motion file still supplies initialization and the ordinary reference
context. Interactive task goals do not follow that reference, and clip endings
do not reset the robot. Native tasks cannot be combined with ``--full-eval``:
tracking a dataset clip is a different evaluation protocol.

Tasks
-----

* ``idle`` holds the anchor's initial XY location and heading.
* ``location`` asks for a world XY target and facing heading at a fixed deadline.
  The default target is two meters along world +X from the starting anchor.
* ``reaching`` holds the anchor XY/heading and targets one conditionable body's
  world XYZ position. Name the body in the inference config, after
  ``configure_masked_mimic_task(configs.env, configs.robot, "reaching")``, with
  ``configs.env.control_components["masked_mimic"].reach_body_name = "<body_name>"``
  using its exact robot body name.
* ``steering`` supplies world XY velocity in m/s (``heading`` and ``speed``), a
  heading-only facing goal in the first frame (``facing``, default the movement
  heading) and anchor-height goals above the terrain at the robot's default root
  height in the next frames (``height_target_times_seconds``). The facing goal
  is aimed with the robot's ``semantic_forward_axis_xy``. Its deadline is fixed
  at each command as the turn time at ``turn_rate_degrees_per_second``, shown
  with at least ``min_target_time_seconds`` remaining. The last frame stays
  free for point goals, so steering needs more frames than height goals plus
  one. It does not continuously move a position target ahead of the robot.

Heights follow the sensor-only convention of the trained policy: target features
are relative to the target pose's lowest body, because a deployed robot has no
terrain input. Goal heights are therefore stored above the terrain under the
goal, and hidden reference bodies are placed on that terrain to stand in for the
missing support foot. A reaching height is relative to the ground the anchor goal
stands on, not to the terrain under the hand. Two approximations are accepted:
the stand-in plane sits at the surface while a real support foot's origin is a
few centimetres above it, and heights jump when the anchor crosses a step edge.
World positions and deadlines remain fixed until a new command. Time-to-go
counts down and clamps at zero; an overdue goal is held rather than silently
extended. All tasks clear unused local and style channels; switching away from steering clears
its velocity command. Resetting a subset does not alter another environment.

Keyboard and Script Commands
----------------------------

Keyboard mode uses the simulator's shared key registry and selected environment.
X cycles idle → location → steering → reaching → idle. Steering is available
only with velocity commands; reaching only with a configured ``reach_body_name``.
Unavailable modes are skipped. Switching captures the current pose without
resetting the robot; steering starts stationary, facing its current direction.
Idle ignores movement keys until X selects another mode.
In steering, W/S rotate movement direction counter-clockwise/clockwise by 10°.
A/D independently rotate facing direction counter-clockwise/clockwise by 10°.
Neither pair changes speed.
C/Z increase/decrease speed by 0.1 m/s in steering and location modes. Steering
speed is clamped to 0–5 m/s. Location speed (default 1.5 m/s, range 0.1–5 m/s)
sets the arrival deadline to remaining distance divided by speed; changing it
keeps the target in place. Each selected robot has its own speed setting.
Scripted steering commands use the same cap. A scripted location deadline is
extended when necessary so the initial distance divided by duration is at most
5 m/s; this limits the command, not the robot's measured velocity.
For location/reaching, W/S move the target along world X and A/D along world Y,
in 0.2 m steps. U/I raise/lower a reaching target by 0.1 m and do nothing in
other modes. Q remains the simulator quit key; camera and
recording controls are unchanged. Keyboard mode requires a visible simulator.

Green spheres show point targets. In steering, a red arrow shows movement
direction and a blue arrow shows facing direction, even at zero speed. The
arrows share the same root-height origin and overlap when their directions agree.

The controller reports errors for the selected robot every 0.5 simulation seconds:
velocity and facing-heading error in steering, location and speed error in
location/idle, hand-position error in reaching, and torso-height deficit during
recovery. In script mode it reports robot zero. Set
``status_print_interval_seconds=0`` on the controller to silence diagnostics.

Script mode is the default. From a running environment, retrieve the native
controller from its control manager, then issue batched commands:

Switching into steering requires the same velocity input, goal-canvas capacity,
and timing settings as starting in steering. Invalid scripted switches are
rejected before changing the current command; smaller canvases remain valid
for the other modes.

.. code-block:: python

   control = env.control_manager.components["masked_mimic"]
   control.hold(env_ids)
   control.set_location(env_ids, world_xy, seconds=2.0)   # [E, 2]
   control.set_reaching(env_ids, world_xyz, seconds=1.0) # [E, 3]
   control.set_steering(env_ids, speed=1.0, heading=0.0)  # +X world heading
   control.set_steering(env_ids, 1.0, heading=0.0, facing=1.57)  # strafe

The implementation has two modules under ``protomotions/envs/control/masked_mimic_tasks/``:
``goals.py`` defines ``BodyGoal`` and ``pack_goals``; ``controller.py`` owns the
concrete task controller, its configs and the inference setup helper. There is
no task-class hierarchy. Regenerate saved task configs that reference the old
``masked_mimic_tasks.base`` module; there is no compatibility alias.

For custom prompt builders, import ``BodyGoal`` and ``pack_goals`` from
``masked_mimic_tasks.goals``. Body IDs are mapped into the checkpoint's conditionable-body
order, not assumed to be canvas indices. Quaternions are xyzw. Bodies sharing
a frame must share its deadline. The packer emits masks, anchor submodes,
time offsets and explicitly empty unused modalities.

The controller's ``getup`` field defaults to ``None``; the example inference
preset explicitly enables ``GetupRecoveryConfig()``. Use recovery only for
checkpoints trained with it: for a student without get-up conditioning, call
``configure_masked_mimic_task`` with ``getup=None`` in your copy of the preset.
Every control mode latches into recovery when
its torso is below ``enter_height_fraction`` of the default standing height
above its lowest body and leaves above ``exit_height_fraction``. While latched,
all task goals and velocity commands are hidden and only local torso-height
goals are shown, repeated ``time_gap_seconds`` apart with a first deadline that
grows with the height deficit. The selected mode and commands are preserved and
resume when the robot stands up. This adapter does not include path following. Tests verify
goal geometry, modality isolation, deadlines, reset/snapshot behavior and UI
integration. Learned success depends on the trained policy; these tests do not
qualify a released policy artifact.
