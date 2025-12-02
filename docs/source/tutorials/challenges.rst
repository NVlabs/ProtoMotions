Challenges
==========

These open-ended challenges help you learn the codebase by implementing new
features. Each challenge has hints to get you started.

Challenge 1: Add Projectiles API
--------------------------------

**Goal:** Add an API to the simulator for spawning and shooting projectiles 
at the robot (for robustness testing).

**Difficulty:** Medium

**What you'll learn:**

* Simulator abstraction layer
* Dynamic object spawning
* Setting object states

**Hints:**

1. Look at how objects are added in each simulator. We don't need SceneLib for projectiles
2. Consider both IsaacGym and Newton implementations

**Starting point:**

Add a method like:

.. code-block:: python

   class BaseSimulator:
       def spawn_projectile(self, position, velocity, mass=1.0):
           """Spawn a projectile at position with initial velocity."""
           pass
       
       def update_projectiles(self):
           """Step projectile physics and check for collisions."""
           pass

Challenge 2: Corridor Scene
---------------------------

**Goal:** Create a corridor environment using SceneLib boxes around a walking 
motion, forcing the character to navigate through a narrow passage.

**Difficulty:** Easy-Medium

**What you'll learn:**

* SceneLib API
* Scene-motion alignment
* Environment customization

**Hints:**

1. See ``examples/data/rigv1-vaulting/rigv1-obstacle.py`` for scene creation
2. Use ``SceneObject`` with ``object_type="box"`` for walls
3. Test with ``env_kinematic_playback.py`` before training

**Example scene structure:**

.. code-block:: python

   left_wall = SceneObject(
       object_type="box",
       position=[-1.0, 0.0, 1.0],
       dimensions=[0.1, 10.0, 2.0],  # thin, long, tall
       fix_base_link=True,
   )
   right_wall = SceneObject(
       object_type="box",
       position=[1.0, 0.0, 1.0],
       dimensions=[0.1, 10.0, 2.0],
       fix_base_link=True,
   )

Challenge 3: Add T1 Robot Support
---------------------------------

**Goal:** Add Booster T1 humanoid robot following the custom robot guide.

**Difficulty:** Medium-Hard

**What you'll learn:**

* Robot configuration system
* MJCF file requirements
* Retargeting pipeline

**Hints:**

1. Follow the custom robot guide and retargeting guide

Challenge 4: Crouching Reward for Steering
------------------------------------------

**Goal:** Modify the pure RL steering task to include a crouching reward,
teaching the robot to walk while staying low.

**Difficulty:** Easy

**What you'll learn:**

* Reward configuration
* Environment customization
* Reward function design

**Hints:**

1. Check ``examples/experiments/steering/mlp.py`` for steering config
2. Add a reward component that penalizes high root height
3. Consider target height as a parameter

**Example reward:**

.. code-block:: python

   "crouch_rew": RewardComponentConfig(
       function=mean_squared_error_exp,
       variables={
           "x": "current_state.rigid_body_pos[:, 0, 2]",  # Root Z
           "ref_x": "0.6",  # Target height
           "coefficient": "-10.0",
       },
       weight=1.0,
   )

Challenge 5: Agent Class Extension
----------------------------------

**Goal:** Implement a new RL algorithm as a custom agent.

**Difficulty:** Hard

**What you'll learn:**

* Agent abstraction
* Training loop design
* Algorithm implementation

**Starting point:**

1. Study how ADD is implemented in ``protomotions/agents/add/agent.py``
2. Look at how other agents extend BaseAgent

Challenge 6: OMOMO Dataset Loader
---------------------------------

**Goal:** Create a data loader for the `OMOMO dataset <https://omomo.stanford.edu/>`_ 
that pairs SMPL/AMASS format human motions with corresponding object shapes and motions,
generating SceneLib scenes that match each motion clip.

**Difficulty:** Medium-Hard

**What you'll learn:**

* AMASS/SMPL motion data format
* SceneLib API for meshes and moving objects
* Motion-scene synchronization
* Data pipeline design

**Background:**

OMOMO contains human-object interaction data with:

* Human motion in SMPL format (compatible with AMASS pipeline)
* Object meshes (OBJ files)
* Object motion trajectories (6-DoF poses over time)

**Hints:**

1. Start with the existing AMASS workflow in :doc:`workflows/amass_smpl`
2. Use ``MeshSceneObject`` for object shapes from OBJ files
3. SceneLib supports moving objects - provide translation/rotation as sequences
4. Match object motion FPS with humanoid motion FPS
5. See ``examples/data/rigv1-vaulting/rigv1-obstacle.py`` for scene creation patterns

**Example scene structure:**

.. code-block:: python

   from protomotions.components.scene_lib import (
       Scene, MeshSceneObject, ObjectOptions, SceneLib
   )
   
   def create_omomo_scene(motion_id, obj_mesh_path, obj_translations, obj_rotations, fps):
       """Create a scene pairing humanoid motion with object motion."""
       
       options = ObjectOptions(
           density=500,
           fix_base_link=False,  # Object moves
       )
       
       obj = MeshSceneObject(
           mesh_file=obj_mesh_path,
           translation=obj_translations,  # (N, 3) array for N frames
           rotation=obj_rotations,         # (N, 4) array, quaternion xyzw
           options=options,
           fps=fps,
       )
       
       return Scene(objects=[obj], humanoid_motion_id=motion_id)
   
   # Build scenes for all OMOMO clips
   scenes = []
   for clip in omomo_clips:
       scene = create_omomo_scene(
           motion_id=clip.motion_id,
           obj_mesh_path=clip.object_mesh,
           obj_translations=clip.object_positions,
           obj_rotations=clip.object_orientations,
           fps=clip.fps,
       )
       scenes.append(scene)
   
   SceneLib.save_scenes_to_file(scenes, "omomo_scenes.pt")

**Validation steps:**

1. Visualize with ``env_kinematic_playback.py`` to verify alignment
2. Check object motion matches human contact timing
3. Ensure consistent coordinate frames between human and object data

