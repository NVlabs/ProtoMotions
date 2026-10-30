Code Tutorials (Progressive Series)
====================================

Learn ProtoMotions through 8 progressive Python tutorials in ``examples/tutorial/``.

.. warning::

   **GPU Required**: IsaacGym, IsaacLab and Newton are designed for GPU acceleration.
   While ``--cpu-only`` is available, it is **highly experimental** and not recommended for most use cases.
   The exception is MuJoCo, a CPU-only backend that builds a single environment: it
   always needs ``--cpu-only``, and it only runs tutorials 0, 1 and 2 because it
   refuses to build scenes.

Overview
--------

These tutorials teach you to build ProtoMotions systems from scratch. Each tutorial
is a complete, runnable Python script that builds on previous concepts.

**How to use**:

1. Read the tutorial documentation below
2. Run the corresponding Python file
3. Examine the code to understand implementation
4. Modify and experiment

**Prerequisites**: ProtoMotions installed with a simulator (isaacgym, isaaclab, newton, or mujoco)

Which backend runs which tutorial
---------------------------------

Every script takes ``--simulator``, but the backends differ in what they can build,
so not every tutorial runs on every backend:

* **Tutorials 0, 1 and 2** (simulator, terrain, robot) run on all four backends.
  On MuJoCo add ``--cpu-only``: the backend only accepts a CPU device, and the
  scripts default the device to ``cuda:0``. The scripts build one environment on
  MuJoCo and four everywhere else.
* **Tutorials 3 to 7** build a scene, so MuJoCo cannot run them: its simulator
  refuses any scene library that holds a scene. Their ``--simulator`` help text says so.
* **Newton** does not create scene objects. Tutorial 3 crashes on it right after the
  reset, because the object state it prints is empty. Tutorials 4 to 6 run, but the
  chair, the teapot and the table are silently absent. Use IsaacGym or IsaacLab to
  see the objects. Tutorial 7 is documented on IsaacGym and IsaacLab.
* **Tutorial 5** runs on IsaacLab, not IsaacGym. Its hard-coded SMPL-X robot config
  lists the 30 finger bodies as contact bodies by default, and IsaacGym refuses a
  robot config with contact bodies: its GPU pipeline reports one combined force per
  body, so it cannot tell what each body is touching.

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
* Build a ``RobotConfig`` by hand: the MJCF asset, the semantic body names, the
  per-joint PD gains and limits from the G1 motor classes, and the control rate
* Create terrain and simulator instances
* Run a basic simulation loop with random actions

**Run it**:

.. code-block:: bash

   python examples/tutorial/0_create_simulator.py --simulator isaacgym

   # CPU only, one environment
   python examples/tutorial/0_create_simulator.py --simulator mujoco --cpu-only

**Code highlights**:

.. code-block:: python

   # Robot configuration, built by hand for the G1
   robot_cfg = RobotConfig(
       semantic_forward_axis_xy=(1.0, 0.0),
       asset=RobotAssetConfig(asset_file_name="mjcf/g1_bm.xml"),
       common_naming_to_robot_body_names={
           "all_left_foot_bodies": ["left_ankle_roll_link"],
           "all_right_foot_bodies": ["right_ankle_roll_link"],
           ...
           "torso_body_name": ["torso_link"],
       },
       # Every joint needs its PD gains and effort/velocity limits declared; the
       # robot config refuses a joint no pattern covers. These follow the G1's motors.
       control=ControlConfig(
           override_control_info={
               ".*_hip_(pitch|yaw)_joint": G1_7520_14.control_info(),
               ".*_knee_joint": G1_7520_22.control_info(),
               ".*_ankle_.*": G1_5020.control_info(scale=2, effort_limit=50),
               ...
           },
       ),
       # 50 Hz policy; simulation_params defaults to the humanoid physics for 50 Hz.
       control_hz=50,
   )

   # Create simulator via factory
   simulator_cfg = simulator_config(
       args.simulator,
       robot_cfg,
       headless=args.headless,
       num_envs=1 if args.simulator == "mujoco" else 4,
       experiment_name="smpl_humanoid_isaaclab_example",
   )
   SimulatorClass = get_class(simulator_cfg._target_)
   simulator = SimulatorClass(config=simulator_cfg, robot_config=robot_cfg, ...)

The per-backend physics parameters are not written out here. ``simulation_params``
is left unset, so ``RobotConfig`` fills it with the humanoid physics for the declared
``control_hz``, and each simulator derives how many physics steps it takes per
control step from its own physics rate and that control rate.

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
* Configure object options (mass/density, damping, material, VHACD collision)
* Compose scenes with multiple objects
* Access object state during simulation

**Run it**:

.. code-block:: bash

   python examples/tutorial/3_scene_creation.py --simulator isaacgym --robot smpl

.. note::

   Run this tutorial on IsaacGym or IsaacLab. On Newton the script crashes right
   after the reset: the backend creates no scene objects and returns an empty object
   state, so the print of the object positions fails. MuJoCo refuses to build a scene
   at all.

**Code highlights**:

.. code-block:: python

   elephant = MeshSceneObject(
       object_path="examples/data/elephant.urdf",
       options=ObjectOptions(
           fix_base_link=False,
           density=1000,  # Use mass=... instead for explicit kg.
           static_friction=0.8,
           dynamic_friction=0.6,
           restitution=0.0,
           vhacd_enabled=True,
       ),
       translation=(0.0, 0.0, 1.5),
   )
   table = BoxSceneObject(width=1.0, depth=1.0, height=0.1, ...)
   
   scene = Scene(objects=[elephant, table], humanoid_motion_id=0)
   scene_lib = SceneLib(config=scene_lib_config, scenes=[scene], ...)

One detail to know before you reuse the script's ``SceneLibConfig``: it passes
``replicate_method="random"`` and ``subset_method="random"`` as strings. That goes
unnoticed only because the script builds one environment with one scene, so
``SceneLib`` never replicates or subsets. With more environments than scenes,
``SceneLib`` compares those fields against the enums and rejects the strings, so
pass ``ReplicationMethod.RANDOM`` and ``SubsetMethod.RANDOM`` from
``protomotions.components.scene_lib`` instead.

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
* Read the done flags and reset finished environments yourself

**Run it**:

.. code-block:: bash

   python examples/tutorial/4_basic_environment.py --simulator isaacgym --robot smpl

.. note::

   On Newton the script runs, but the chair is not created, so the robot stands next
   to nothing. MuJoCo refuses to build the scene.

**Code highlights**:

.. code-block:: python

   env_config = EnvConfig(
       max_episode_length=1000,
       observation_components={
           "max_coords_obs": MdpComponent.from_kernel(
               compute_humanoid_max_coords_observations
           ),
           "terrain": MdpComponent.from_kernel(compute_terrain_height_map),
       },
   )

   env = BaseEnv(
       config=env_config,
       robot_config=robot_cfg,
       device=device,
       simulator=simulator,
       terrain=terrain,
       scene_lib=scene_lib,
       motion_lib=motion_lib,  # Always provided (empty if no motions)
   )
   obs, rewards, dones, terminated, extras = env.step(actions)

``step()`` never resets an environment. It returns the done flags, and the caller
resets the rows that are done. The script's loop does not do that: it resets every
environment once before the loop and then only calls ``step()``. The script
configures no termination components, so an environment is done when it reaches
``max_episode_length``, and from then on it keeps simulating under the random actions, never
respawned, while its done flag stays set. The "Environments reset this step" line in the script's output, and its
closing summary about automatic resets, are therefore misleading: the number is the
count of done flags, and nothing was respawned. To respawn finished environments,
reset the done rows at the top of each iteration, as tutorial 6 does:

.. code-block:: python

   done_indices = None
   while env.is_simulation_running():
       env.reset(done_indices)  # None resets every environment on the first pass
       actions.normal_()
       obs, rewards, dones, terminated, extras = env.step(actions)
       done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

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

   python examples/tutorial/5_motion_manager.py --simulator isaaclab

.. note::

   This tutorial uses a hard-coded SMPL-X robot (52 bodies with hand articulation)
   to match the teapot pour motion data. Run it on IsaacLab: the SMPL-X config lists
   its 30 finger bodies as contact bodies by default, and IsaacGym refuses a robot
   config with contact bodies because its GPU pipeline reports one combined force per
   body. On Newton the script runs, but the teapot and the table are not created.

**Code highlights**:

.. code-block:: python

   motion_lib_config = MotionLibConfig(motion_file="examples/data/grab_teapot_pour/s1_teapot_pour_1.motion")
   motion_lib = MotionLib(config=motion_lib_config, device=device)
   
   env_config = EnvConfig(
       max_episode_length=1000,
       # Kinematic playback: the robot is set to the reference pose, actions are ignored
       control_components={"kinematic_replay": KinematicReplayControlConfig()},
       observation_components={
           "max_coords_obs": MdpComponent.from_kernel(
               compute_humanoid_max_coords_observations
           ),
       },
       # Motion manager controls sampling
       motion_manager=MimicMotionManagerConfig(init_start_prob=1.0),  # Always start from t=0
   )

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

* Configure mimic-specific observations (target poses, previous actions)
* Understand the two control components: ``MimicControl`` for policy training
  (this tutorial) and ``KinematicReplayControl`` for kinematic playback (tutorial 5)
* Set up reference state initialization (RSI) with ``init_start_prob``

**Run it**:

.. code-block:: bash

   python examples/tutorial/6_mimic_environment.py --simulator isaacgym

.. note::

   This tutorial uses a hard-coded SMPL humanoid to match the sitting on chair motion data.
   On Newton the script runs, but the armchair is not created, so the motion plays
   without the chair. MuJoCo refuses to build the scene.

**Code highlights**:

.. code-block:: python

   control_components = {
       "mimic": MimicControlConfig(bootstrap_on_episode_end=True),
   }
   observation_components = {
       "max_coords_obs": MdpComponent.from_kernel(
           compute_humanoid_max_coords_observations
       ),
       "previous_actions": MdpComponent.from_kernel(
           compute_historical_actions_from_state, history_steps=1
       ),
       "mimic_target_poses": MdpComponent.from_kernel(
           build_max_coords_target_poses, with_velocities=True
       ),
   }
   reward_components = {
       "action_smoothness": MdpComponent.from_kernel(
           compute_action_smoothness, meta=RewardMeta(weight=-0.02)
       ),
       "gt_rew": MdpComponent.from_kernel(compute_gt_rew, meta=RewardMeta(weight=0.5)),
       "gr_rew": MdpComponent.from_kernel(compute_gr_rew, meta=RewardMeta(weight=0.3)),
       "gv_rew": MdpComponent.from_kernel(compute_gv_rew, meta=RewardMeta(weight=0.1)),
       "gav_rew": MdpComponent.from_kernel(
           compute_gav_rew, meta=RewardMeta(weight=0.1)
       ),
   }

   env_config = EnvConfig(
       max_episode_length=300,
       control_components=control_components,
       observation_components=observation_components,
       reward_components=reward_components,
       motion_manager=MimicMotionManagerConfig(init_start_prob=0.5),
   )

   env = BaseEnv(
       config=env_config,
       robot_config=robot_cfg,
       device=device,
       simulator=simulator,
       motion_lib=motion_lib,
       terrain=terrain,
       scene_lib=scene_lib,
   )

The three observation keys above are the whole set this environment produces. The
script's observation breakdown has print branches for ``mimic_phase`` and
``mimic_time_left``, but no kernel produces those observations, so those branches
never fire. Playback versus training is decided by the control component alone:
``MimicControl`` publishes the reference state for the rewards and the target-pose
observation while the policy's actions drive the robot, whereas
``KinematicReplayControl`` (tutorial 5) sets the robot to the reference pose and
ignores the actions. The simulation loop also shows the reset pattern tutorial 4
omits: it resets the done rows at the top of each iteration.

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
   Run it on IsaacGym or IsaacLab; MuJoCo refuses to build the scene.

**Code highlights**:

.. code-block:: python

   reward_components = {
       "action_smoothness": MdpComponent.from_kernel(
           compute_action_smoothness, meta=RewardMeta(weight=-0.02)
       ),
       "gt_rew": MdpComponent.from_kernel(compute_gt_rew, meta=RewardMeta(weight=0.5)),
       "gr_rew": MdpComponent.from_kernel(compute_gr_rew, meta=RewardMeta(weight=0.3)),
       "gv_rew": MdpComponent.from_kernel(compute_gv_rew, meta=RewardMeta(weight=0.1)),
       "gav_rew": MdpComponent.from_kernel(
           compute_gav_rew, meta=RewardMeta(weight=0.1)
       ),
   }
   termination_components = {
       "tracking_error": TerminationRule(
           MdpComponent.from_kernel(max_body_pos_error), 0.5
       ),
   }

   env_config = EnvConfig(
       max_episode_length=300,
       control_components=control_components,
       observation_components=observation_components,
       reward_components=reward_components,
       termination_components=termination_components,
       action_config=make_pd_action_config(robot_cfg),
       motion_manager=MimicMotionManagerConfig(
           init_start_prob=1.0,  # Always start from the beginning
           resample_on_reset=False,  # Keep the current motion time on reset
       ),
   )

   obs_keys = ["max_coords_obs", "mimic_target_poses"]
   actor_config = PPOActorConfig(
       num_out=robot_cfg.kinematic_info.num_dofs,
       actor_logstd=-2.9,
       in_keys=obs_keys,
       mu_key="actor_trunk_out",  # The mean network's output key
       mu_model=MLPWithConcatConfig(
           in_keys=obs_keys,
           out_keys=["actor_trunk_out"],
           num_out=robot_cfg.number_of_actions,
           ...
       ),
   )
   critic_config = MLPWithConcatConfig(
       in_keys=obs_keys,
       out_keys=["value"],
       num_out=1,
       ...
   )
   # The model derives its in_keys and out_keys from the actor and the critic;
   # passing them is a construction error.
   agent_config = PPOAgentConfig(
       model=PPOModelConfig(actor=actor_config, critic=critic_config, ...),
       batch_size=128,  # 4 envs x 32 steps per rollout
       training_max_steps=2560,  # About 20 epochs
       num_steps=32,
       ...
   )

   output_dir = Path("./tutorial_7_output")  # Directory for saving checkpoints
   output_dir.mkdir(parents=True, exist_ok=True)
   agent = PPO(fabric=fabric, env=env, config=agent_config, root_dir=output_dir)
   agent.setup()  # Builds the model and the optimizers; fit() needs it first
   agent.fit()

``mu_key`` and ``training_max_steps`` have no defaults, so both must be given.
``root_dir`` is required here because the tutorial's Fabric has no logger the agent
could borrow a directory from, and the script creates the directory before handing
it over. ``setup()`` is a separate step: ``fit()`` expects the model and the
optimizers to exist already.

**This is a complete training example!**
