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
   │  (IsaacGym/Lab/Newton/Gen)  │◄─── Robot Config
   └──────────────┬──────────────┘
                  │
                  ▼ SimulatorState
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
Newton, Genesis) implement the same interface. This is "how to simulate"

**Motion Library**: Stores reference motions in packed tensors for efficient 
parallel access.

**RL Environment**: Orchestrates the training loop via modular components:

* **Control Components**: Stateful task managers (mimic, steering, path following)
* **Observation Components**: Stateless functions that compute observations
* **Reward Components**: Stateless functions that compute rewards
* **Termination Components**: Stateless functions that check termination conditions

**Agent**: RL algorithm (PPO, AMP, etc.). Collects experience, updates policy.

Why This Design?
----------------

**Problem**: We need to support many combinations:

* 4 simulators × N robots × M algorithms × K environments

Without abstractions, this would require N×M×K separate implementations.

**Solution**: Clear interfaces between components:

* Simulators share ``SimulatorState`` representation
* Environments produce ``TensorDict`` observations
* Agents consume observations, produce actions
* Robot configs are simulator-agnostic


Next Steps
----------

* :doc:`abstractions` - Deep dive into each component
* :doc:`../user_guide/configuration` - Config system details

