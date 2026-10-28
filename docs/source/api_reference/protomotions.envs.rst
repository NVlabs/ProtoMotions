protomotions.envs package
============================

Tasks share ``BaseEnv`` and differ through configuration. Motion tracking,
steering and path following use control components, not separate environment
subclasses. Their experiment files wire controls to observation, reward,
termination and action functions.

BaseEnv
-------

``protomotions.envs.base_env.env.BaseEnv`` receives an ``EnvConfig``, a robot
config, device, terrain, simulator, scene library and motion library. The
training entry point constructs these dependencies; the environment initializes
the simulator. Scene and motion libraries can be empty.

The runtime interface is:

.. code-block:: python

   obs, info = env.reset()
   obs, rewards, dones, terminated, extras = env.step(actions)

``obs`` is a dictionary of named tensors. The agent converts it to a TensorDict
for the model. Actions have shape ``[num_envs, num_actions]``. ``dones`` includes
episode timeouts; ``terminated`` identifies terminal states for value
bootstrapping. Call ``reset(env_ids)`` for environments whose episodes ended;
the agent owns this rollout/reset sequence. ``get_obs()`` returns the current
observations without stepping the simulation.

Component ownership
-------------------

* ``control_components`` own mutable task state: reference poses, commands,
  goals and timers.
* ``observation_components`` map context tensors to named model inputs.
* ``reward_components`` map context tensors to reward terms.
* ``termination_components`` define training episode failure conditions.
* ``action_config`` maps policy outputs to simulator commands.

Observations, rewards and terminations use ``MdpComponent`` to bind pure tensor
functions to ``EnvContext`` paths. Sensor estimates and clean simulator state
are separate sources; each binding selects which it needs. Do not add noise
inside a reward or observation function to duplicate sensor behavior.

Examples
--------

.. list-table:: Task wiring
   :header-rows: 1

   * - Task
     - Stateful control
     - Experiment
   * - Motion tracking
     - ``MimicControl``
     - ``examples/experiments/mimic/mlp.py``
   * - Steering
     - ``SteeringControl``
     - ``examples/experiments/steering/mlp.py``
   * - Path following
     - ``PathFollowerControl``
     - ``examples/experiments/path_follower/mlp.py``

See :doc:`../user_guide/configuration` for component bindings and config
lifecycle, and :doc:`../user_guide/evaluation` for benchmark scoring and
temporary environment snapshots. Evaluation failure criteria are configured
separately from training terminations.
