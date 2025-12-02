protomotions.envs package
=========================

Environments for reinforcement learning tasks. All environments extend BaseEnv
and implement task-specific reward functions and observations.

BaseEnv
-------

.. py:class:: BaseEnv(config, robot_config, simulator_config, device, terrain=None, scene_lib=None, motion_lib=None)

   Base class for all reinforcement learning environments.
   
   Provides core functionality including simulator integration, terrain management,
   scene and object handling, motion library integration, observation computation,
   and episode management.
   
   :param config: Environment configuration
   :param robot_config: Robot morphology configuration  
   :param simulator_config: Simulator backend configuration
   :param device: PyTorch device for computations
   :param terrain: Optional terrain instance
   :param scene_lib: Optional scene library
   :param motion_lib: Optional motion library
   
   **Key Methods**:
   
   - ``reset()`` - Reset environments
   - ``step(actions)`` - Step simulation with actions
   - ``compute_observations()`` - Compute observations
   - ``compute_reward()`` - Compute rewards
   
   **Example**::
   
       config = SteeringEnvConfig()
       env = Steering(config, robot_config, simulator_config, device)
       obs, _ = env.reset()
       next_obs, rewards, dones, info = env.step(actions)

Mimic
-----

.. py:class:: Mimic(config, robot_config, simulator_config, device)

   Motion imitation environment for full-body tracking.
   
   Trains agents to imitate reference motions from a motion library with detailed
   tracking rewards for joint positions, velocities, and end-effector locations.
   
   **Key Features**:
   
   - Full-body pose tracking with per-body-part weights
   - Contact-aware foot placement rewards
   - Sync/async motion playback modes
   - Early termination on tracking errors
   - Terrain-aware tracking
   
   :param config: MimicEnvConfig with reward weights
   :param robot_config: Robot configuration
   :param simulator_config: Simulator configuration
   :param device: PyTorch device
   
   **Example**::
   
       config = MimicEnvConfig()
       env = Mimic(config, robot_config, simulator_config, device)
       obs, _ = env.reset()

Steering
--------

.. py:class:: Steering(config, robot_config, simulator_config, device)

   Steering task environment for humanoid locomotion.
   
   Trains agents to walk in a target direction at a target speed. Target direction
   and speed change periodically to encourage versatile locomotion.
   
   **Key Features**:
   
   - Variable target speeds (including stopping)
   - Periodic heading changes with random variations
   - Visual markers for target direction
   - Rewards for velocity and heading matching
   
   :param config: SteeringEnvConfig
   :param robot_config: Robot configuration
   :param simulator_config: Simulator configuration
   :param device: PyTorch device

PathFollowing
-------------

.. py:class:: PathFollowing(config, robot_config, simulator_config, device)

   Path following task environment for navigation.
   
   Trains agents to follow predefined paths with observations of future waypoints.
   Rewards staying close to the path and making forward progress.
   
   **Key Features**:
   
   - Multiple future waypoint observations
   - Path generation (lines, curves, complex trajectories)
   - Visual path markers
   - Distance and progress rewards
   
   :param config: PathFollowerEnvConfig
   :param robot_config: Robot configuration
   :param simulator_config: Simulator configuration
   :param device: PyTorch device

