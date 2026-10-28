Architecture Overview
=====================

ProtoMotions is designed around modular components that can be mixed and matched
to create different training scenarios.

High-Level Data Flow
--------------------

.. code-block:: text

   ┌─────────────┐     ┌─────────────┐
   │   Terrain   │     │ Scene Lib   │
   └──────┬──────┘     └──────┬──────┘
          │                   │
          └────────┬──────────┘
                   ▼
   ┌─────────────────────────────┐
   │        Simulator            │
   │(IsaacGym/Lab/Newton/Gen/MJ)   │◄─── Robot Config
   └──────────────┬──────────────┘
                  │
                  ▼ RobotState
   ┌─────────────────────────────┐
   │      RL Environment         │◄─── Motion Library
   │  (observations, rewards)    │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │          Agent              │
   │   (policy, value, update)   │
   └─────────────────────────────┘

Component Responsibilities
--------------------------

**Terrain**: Procedural heightfield generation. Provides ground surface for simulation.

**SceneLib**: Object spawning and management. Creates obstacles, props, and 
interactive objects in the scene.

**Robot Config**: Defines "what character to simulate" - body mappings, PD gains, physics 
settings. The MJCF file is the ground truth; config adds simulator-specific details.

**Simulator**: Physics engine abstraction. All backends (IsaacGym, IsaacLab, 
Newton, MuJoCo) implement the same interface. This is "how to simulate"

**Motion Library**: Stores reference motions in packed tensors for efficient 
parallel access.

**RL Environment**: Advances physics and task state through modular components:

* **Control Components**: Stateful task managers (mimic, steering, path following)
* **Observation Components**: Stateless functions that compute observations
* **Reward Components**: Stateless functions that compute rewards
* **Termination Components**: Stateless functions that check termination conditions

The stateless components bind the context names they read
(``EnvContext.current.dof_pos`` ...), and the names are checked twice: from
the configs alone, before a simulator exists (``train_agent.py`` and
``inference_agent.py`` call ``context_check.run_config_binding_check`` right
after building the configs; it raises on a binding this configuration can
never satisfy and logs the names that depend on the motion file, which are
checked when the environment is built), and again against the first context
the built environment publishes. A binding this configuration can never
satisfy fails in seconds with the reason; see :doc:`environment_context`.

**Agent**: RL algorithm (PPO, AMP, etc.). Collects experience, updates policy.

Why This Design?
----------------

**Problem**: We need to support many combinations:

* 5 simulators × N robots × M algorithms × K environments

Without abstractions, this would require N×M×K separate implementations.

**Solution**: Clear interfaces between components:

* Simulators exchange ``RobotState``, ``ResetState`` and ``ObjectState`` records
  in a common body and joint ordering
* Environments produce dictionaries of observation tensors
* Agents convert observations to TensorDicts, collect rollouts and optimize models
* Robot configs describe shared morphology and per-simulator physics settings


Next Steps
----------

* :doc:`abstractions` - Deep dive into each component
* :doc:`../user_guide/configuration` - Config system details
