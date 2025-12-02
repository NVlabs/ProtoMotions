Code Tutorials (Progressive Series)
====================================

Learn ProtoMotions through 8 progressive Python tutorials in ``examples/tutorial/``.

.. warning::

   **GPU Required**: These simulators (IsaacGym, IsaacLab, Genesis, Newton) are designed for GPU acceleration.
   While ``--cpu-only`` is available, it is **highly experimental** and not recommended for most use cases.

Overview
--------

These tutorials teach you to build ProtoMotions systems from scratch. Each tutorial
is a complete, runnable Python script that builds on previous concepts.

**How to use**:

1. Read the tutorial documentation below
2. Run the corresponding Python file
3. Examine the code to understand implementation
4. Modify and experiment

**Prerequisites**: ProtoMotions installed with a simulator (isaacgym, isaaclab, genesis, or newton)

Tutorial 0: Create Simulator
-----------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_0.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**File**: ``examples/tutorial/0_create_simulator.py``

Learn the foundation of ProtoMotions - creating a physics simulator with a G1 robot.

**What you'll learn**:

* Import simulator before torch (required for IsaacGym/IsaacLab)
* Configure robot with simulation parameters per backend
* Create terrain and simulator instances
* Run a basic simulation loop with random actions

**Run it**:

.. code-block:: bash

   python examples/tutorial/0_create_simulator.py --simulator isaacgym

**Code highlights**:

.. code-block:: python

   # Robot configuration with per-simulator params
   robot_cfg = RobotConfig(
       asset=RobotAssetConfig(asset_file_name="mjcf/g1_bm.xml", ...),
       simulation_params=SimulatorParams(
           isaacgym=IsaacGymSimParams(fps=100, decimation=2, substeps=2),
           isaaclab=IsaacLabSimParams(fps=200, decimation=4),
           ...
       ),
   )
   
   # Create simulator via factory
   simulator_cfg = simulator_config(args.simulator, robot_cfg, headless=False, num_envs=4)
   SimulatorClass = get_class(simulator_cfg._target_)
   simulator = SimulatorClass(config=simulator_cfg, robot_config=robot_cfg, ...)

Tutorial 1: Add Terrain
------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_1.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**File**: ``examples/tutorial/1_add_terrain.py``

Learn to create complex terrains for robust locomotion training.

**What you'll learn**:

* Generate procedural terrains with ``ComplexTerrainConfig``
* Configure terrain proportions (slopes, stairs, stepping stones, poles)
* Sample valid spawn locations on terrain
* Query terrain heights during simulation

**Run it**:

.. code-block:: bash

   python examples/tutorial/1_add_terrain.py --simulator isaacgym

**Code highlights**:

.. code-block:: python

   # Terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]
   terrain_config = ComplexTerrainConfig(
       terrain_proportions=[0.2, 0.1, 0.1, 0.1, 0.05, 0.2, 0.3, 0.1],
   )
   TerrainClass = get_class(terrain_config._target_)
   terrain = TerrainClass(config=terrain_config, num_envs=num_envs, device=device)

Tutorial 2: Load Robot
-----------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_2.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**File**: ``examples/tutorial/2_load_robot.py``

Learn to load different robots using the robot factory.

**What you'll learn**:

* Use ``robot_config()`` factory to load robots by name
* Compare different robot configurations (DOFs, bodies, actions)
* Access robot state (positions, velocities, joint info)

**Run it**:

.. code-block:: bash

   # Load G1 humanoid
   python examples/tutorial/2_load_robot.py --simulator isaacgym --robot g1
   
   # Load SMPL humanoid
   python examples/tutorial/2_load_robot.py --simulator isaacgym --robot smpl

**Code highlights**:

.. code-block:: python

   from protomotions.robot_configs.factory import robot_config
   
   robot_cfg = robot_config(args.robot)  # "g1", "smpl", "smplx", etc.
   print(f"Robot has {robot_cfg.number_of_actions} actions, {robot_cfg.kinematic_info.num_dofs} DOFs")

Tutorial 3: Scene Creation
---------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_3.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**File**: ``examples/tutorial/3_scene_creation.py``

Learn to add objects and create scenes for robot interaction.

**What you'll learn**:

* Create ``MeshSceneObject`` and ``BoxSceneObject`` with physics properties
* Configure object options (density, damping, VHACD collision)
* Compose scenes with multiple objects
* Access object state during simulation

**Run it**:

.. code-block:: bash

   python examples/tutorial/3_scene_creation.py --simulator isaacgym --robot smpl

**Code highlights**:

.. code-block:: python

   elephant = MeshSceneObject(
       object_path="examples/data/elephant.urdf",
       options=ObjectOptions(fix_base_link=False, density=1000, vhacd_enabled=True),
       translation=(0.0, 0.0, 1.5),
   )
   table = BoxSceneObject(width=1.0, depth=1.0, height=0.1, ...)
   
   scene = Scene(objects=[elephant, table], humanoid_motion_id=0)
   scene_lib = SceneLib(config=scene_lib_config, scenes=[scene], ...)

Tutorial 4: Basic Environment
------------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_4.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**File**: ``examples/tutorial/4_basic_environment.py``

Learn to create a complete RL environment using ``BaseEnv``.

**What you'll learn**:

* Configure ``BaseEnv`` with ``EnvConfig`` and observation settings
* Use the standard RL interface: ``reset()``, ``step()``, ``get_obs()``
* Access structured observations (humanoid state, terrain)
* Handle episode termination and automatic resets

**Run it**:

.. code-block:: bash

   python examples/tutorial/4_basic_environment.py --simulator isaacgym --robot smpl

**Code highlights**:

.. code-block:: python

   env_config = EnvConfig(
       max_episode_length=1000,
       humanoid_obs=HumanoidObsConfig(
           max_coords_obs=MaxCoordsSelfObsConfig(enabled=True, local_obs=True),
       ),
   )
   
   env = BaseEnv(config=env_config, robot_config=robot_cfg, simulator=simulator, ...)
   obs, rewards, dones, terminated, extras = env.step(actions)

Tutorial 5: Motion Manager
---------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_5.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

**File**: ``examples/tutorial/5_motion_manager.py``

Learn to work with motion libraries for reference motion playback.

**What you'll learn**:

* Load motion data from ``.motion`` files (torch format)
* Load object trajectories from numpy files
* Configure motion manager parameters (``init_start_prob``)
* Track motion progress (IDs, times) during simulation

**Run it**:

.. code-block:: bash

   python examples/tutorial/5_motion_manager.py --simulator isaacgym

.. note::

   This tutorial uses a hard-coded SMPLX robot (52 bodies with hand articulation)
   to match the teapot pour motion data.

**Code highlights**:

.. code-block:: python

   motion_lib_config = MotionLibConfig(motion_file="examples/data/grab_teapot_pour/s1_teapot_pour_1.motion")
   motion_lib = MotionLib(config=motion_lib_config, device=device)
   
   # Motion manager controls sampling
   motion_manager = MimicMotionManagerConfig(init_start_prob=1.0)  # Always start from t=0

Tutorial 6: Mimic Environment
------------------------------

.. raw:: html

   <video width="100%" controls>
     <source src="../_static/tutorial_6.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

*The red spheres indicate the target motion pose, while the robot is simulated with random actions.*

**File**: ``examples/tutorial/6_mimic_environment.py``

Learn to create motion imitation environments using ``Mimic``.

**What you'll learn**:

* Configure mimic-specific observations (phase, time left, target poses)
* Understand ``sync_motion`` modes: kinematic playback vs. policy training
* Set up reference state initialization (RSI) with ``init_start_prob``

**Run it**:

.. code-block:: bash

   python examples/tutorial/6_mimic_environment.py --simulator isaacgym

.. note::

   This tutorial uses a hard-coded SMPL humanoid to match the sitting on chair motion data.

**Code highlights**:

.. code-block:: python

   env_config = MimicEnvConfig(
       sync_motion=False,  # False = policy training, True = kinematic playback
       mimic_obs=MimicObsConfig(
           mimic_phase_obs=MimicPhaseObsConfig(enabled=True),
           mimic_target_pose=MimicTargetPoseConfig(enabled=True, future_steps=1),
       ),
       motion_manager=MimicMotionManagerConfig(init_start_prob=0.5),  # 50% RSI
   )
   
   env = Mimic(config=env_config, motion_lib=motion_lib, ...)

Tutorial 7: DeepMimic Agent
----------------------------

**File**: ``examples/tutorial/7_deepmimic.py``

Learn to train a complete motion tracking agent with PPO.

**What you'll learn**:

* Configure PPO actor-critic networks with ``MLPWithConcatConfig``
* Set up imitation learning rewards (position, rotation, velocity tracking)
* Configure early termination based on tracking error
* Run training with the agent's ``fit()`` method

**Run it**:

.. code-block:: bash

   python examples/tutorial/7_deepmimic.py --simulator isaacgym

.. note::

   This tutorial uses a hard-coded SMPL humanoid to match the sitting on chair motion data.

**Code highlights**:

.. code-block:: python

   # Reward configuration for motion tracking
   reward_config = {
       "gt_rew": RewardComponentConfig(
           function=mean_squared_error_exp,
           variables={"x": "current_state.rigid_body_pos", "ref_x": "ref_state.rigid_body_pos", "coefficient": -100.0},
           weight=0.5,
       ),
       ...
   }
   
   # PPO agent with actor-critic
   agent_config = PPOAgentConfig(
       model=PPOModelConfig(
           actor=PPOActorConfig(mu_model=MLPWithConcatConfig(in_keys=obs_keys, ...)),
           critic=MLPWithConcatConfig(in_keys=obs_keys, num_out=1, ...),
       ),
       batch_size=128,
       num_steps=32,
   )
   
   agent = PPO(fabric=fabric, env=env, config=agent_config, ...)
   agent.fit()  # Run training

**This is a complete training example!**
