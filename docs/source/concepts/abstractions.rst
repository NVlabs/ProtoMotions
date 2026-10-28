Core Abstractions
=================

Each layer has one job: the simulator advances physics, the environment defines
the task, the model maps observations to outputs, and the agent collects data
and optimizes the model. Experiment files connect these layers.

Motion, Terrain and Scenes
--------------------------

**MotionLib** (``protomotions/components/motion_lib.py``) packs clips into
contiguous tensors, with per-motion frame offsets, lengths and durations.
``get_motion_state(motion_ids, motion_times)`` interpolates batched reference
states without a Python loop over environments. The environment's motion
manager owns which clip and time each environment follows.

**Terrain** (``protomotions/components/terrains/``) supplies ground geometry
and height queries. A flat ground is sufficient for tasks that do not need
procedural terrain.

**SceneLib** (``protomotions/components/scene_lib.py``) manages scene objects,
their poses and geometry. An empty scene library keeps the same environment
interface for tasks without objects.

Robot and Simulator
-------------------

**RobotConfig** (``protomotions/robot_configs/``) describes the asset,
semantic body mappings, control parameters and per-simulator physics settings.
Semantic mappings use lists of body names, even for a single torso or head.
PoseLib extracts the body hierarchy and joint information into
``kinematic_info``; see :doc:`pose_lib`.

**Simulator** (``protomotions/simulator/base_simulator/``) hides backend
indexing and state conversions behind a common interface. Its main operations
are ``step(actions)``, ``get_robot_state(env_ids)``, and
``reset_envs(new_states, new_object_states=None, env_ids=None)``.
The reset input is a ``ResetState``, not an arbitrary observation tensor.

IsaacGym, IsaacLab, Newton and MuJoCo implement this interface. Common
state types make task code reusable; they do not imply numerically identical
physics across backends. See :doc:`simulator_state` and :doc:`contact_matrix`.

Environment and Controls
------------------------

**BaseEnv** (``protomotions/envs/base_env/env.py``) advances physics, updates
task state and computes observations, rewards and training terminations. Tasks
are composed through ``EnvConfig``, rather than a separate environment subclass
for every task.

**ControlComponent** (``protomotions/envs/control/base.py``) owns mutable task
state. It is constructed from its config and a ``TaskInputs`` value (the batch,
the robot, the read-only episode clock, the placed reference, the ground-height
query, its random generator) and never holds the environment. Every per-env
tensor is allocated through ``self.state.add(name, shape, dtype, reset_value)``,
so the environment's row reset, snapshot and restore reach it. Its hooks are:

.. code-block:: python

   reset_rows(env_ids, state)        # Seed task state for these environments after the physical reset.
   advance(state)                    # Move the task one step on the robot state the env read; return logging.
   resample(env_ids, state)          # A schedule change inside an episode.
   publish(env_ids)                  # Return the task tensors as context names (pure).
   terminations(config, robot)       # Classmethod: the task's episode-ending checks as TerminationRule entries.
   save_state() / load_state(snap)   # The registry's snapshot and in-place restore.

For example, ``MimicControl`` exposes reference poses from the motion manager,
``SteeringControl`` maintains speed/heading targets, and ``PathFollowerControl``
maintains paths. ``MaskedMimicControl`` samples sparse conditioning;
``MaskedMimicTaskControl`` supplies interactive goals during inference.

Sensors and History
-------------------

The sensor runtime measures simulator state, applies configured noise and
reconstructs clean/noisy body estimates. The environment stores these alongside
privileged simulator state in its history buffer. A component binding selects
the source it needs; observation kernels do not draw sensor noise themselves.

At reset, history repeats the first post-reset reading. Evaluation snapshots
preserve sensor RNG, episode references and history instead of drawing fresh
noise during restoration. See :doc:`environment_context` for source selection
and observation layouts.

Observations, Rewards and Terminations
--------------------------------------

All three use ``MdpComponent``:

* ``compute_func`` is a pure tensor function.
* ``dynamic_vars`` binds its arguments to typed ``EnvContext`` paths.
* ``static_params`` supplies constants and component settings, such as weights.

The functions live under ``protomotions/envs/obs/``,
``protomotions/envs/rewards/``, and ``protomotions/envs/terminations/``.
Every kernel carries its context bindings as parameter defaults and is bound
with ``MdpComponent.from_kernel``; a recipe writes only what it chooses (a
weight, a coefficient, a body selection) and repoints an actor's state
bindings at a sensor source with ``sensed(components, "noisy")``:

.. code-block:: python

   from protomotions.envs.mdp_component import MdpComponent
   from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
   from protomotions.envs.obs.target_poses import build_max_coords_target_poses
   from protomotions.envs.rewards.meta import RewardMeta
   from protomotions.envs.rewards.regularization import compute_action_smoothness
   from protomotions.envs.rewards.tracking import compute_gt_rew

   observation_components = {
       "max_coords_obs": MdpComponent.from_kernel(
           compute_humanoid_max_coords_observations
       ),
       "target_poses": MdpComponent.from_kernel(build_max_coords_target_poses),
   }
   reward_components = {
       "gt_rew": MdpComponent.from_kernel(
           compute_gt_rew, meta=RewardMeta(weight=0.5), coefficient=-100.0
       ),
       "action_smoothness": MdpComponent.from_kernel(
           compute_action_smoothness, meta=RewardMeta(weight=-0.02)
       ),
   }

``ComponentManager`` resolves the bindings and evaluates the kernels.
``BaseEnv`` combines reward terms, applies reset grace rules, and combines
termination components with control-component decisions and episode timeouts.
Benchmark failures are configured separately in the evaluator, not inferred
from training terminations. See :doc:`../user_guide/evaluation`.

Models and Agents
-----------------

**Models** consume and produce TensorDict keys. Their ``in_keys`` and
``out_keys`` make dependencies explicit. The common module contract owns
rollout context, additional experience-buffer keys, training targets and
model-specific auxiliary losses.

**BaseAgent** owns rollout collection, optimization scheduling, checkpointing
and evaluation. Algorithm-specific agents specialize this loop:

* ``PPO``: actor-critic policy optimization.
* ``AMP``: PPO with discriminator-based motion-prior rewards.
* ``ASE``: AMP with latent skill embeddings.
* ``SupervisedAgent``: rollout imitation with configured labels and losses.

MaskedMimic uses ``SupervisedAgent`` with a ``MaskedMimicModel``. The model owns
its prior/posterior, KL schedule and temporal losses; the agent need not duplicate
those computations. External teacher observations and labels are training
dependencies, not deployable policy inputs.

See :doc:`../user_guide/configuration` for configuration lifecycle and
:doc:`../user_guide/metatron_student` for a complete student recipe.
