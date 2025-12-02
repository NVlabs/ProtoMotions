Core Abstractions
=================

This document covers the main components of ProtoMotions in detail.

MotionLib
---------

**Location:** ``protomotions/components/motion_lib.py``

The MotionLib stores reference motion data for imitation learning. 

**Why pack motions into tensors?**

When running 4096 parallel environments, each needs to query a different motion
at a different time. Naive per-motion storage would require 4096 separate lookups.

Instead, we concatenate all motions into contiguous tensors:

.. code-block:: python

   gts: Tensor[total_frames, num_bodies, 3]   # positions
   grs: Tensor[total_frames, num_bodies, 4]   # rotations (quaternion)
   gvs: Tensor[total_frames, num_bodies, 3]   # linear velocities
   gavs: Tensor[total_frames, num_bodies, 3]  # angular velocities
   dps: Tensor[total_frames, num_dofs]        # DOF positions
   dvs: Tensor[total_frames, num_dofs]        # DOF velocities
   contacts: Tensor[total_frames, num_bodies] # contact labels

Frame boundaries are tracked via:

.. code-block:: python

   length_starts: Tensor[num_motions]      # Start index of each motion
   motion_num_frames: Tensor[num_motions]  # Number of frames per motion
   motion_lengths: Tensor[num_motions]     # Duration in seconds

**Querying motions:**

.. code-block:: python

   # Get state for multiple envs at once
   state = motion_lib.get_motion_state(
       motion_ids=torch.tensor([0, 1, 2, ...]),  # Which motion per env
       motion_times=torch.tensor([0.5, 1.2, ...])  # Time in seconds
   )
   # Returns RobotState with interpolated values

Frame interpolation provides smooth motion even at fractional times.

Terrain
-------

**Location:** ``protomotions/components/terrains/``

Terrain generates procedural heightfields for training robustness.

**Configuration:**

.. code-block:: python

   TerrainConfig(
       sim_config=TerrainSimConfig(
           static_friction=1.0,
           dynamic_friction=1.0,
       ),
       # Subterrain types, proportions, etc.
   )

Complex terrain is optional - we set to flat terrain for basic training.

SceneLib
--------

**Location:** ``protomotions/components/scene_lib.py``

SceneLib manages objects in the environment for interaction tasks.

**Object types:**

* **Box**: Simple box primitive
* **Sphere**: Sphere primitive
* **Mesh**: Custom mesh from file

**Example:**

.. code-block:: python

   scene = Scene(objects=[
       SceneObject(
           object_type="box",
           position=[1.0, 0.0, 0.5],
           dimensions=[0.5, 0.5, 1.0],
           fix_base_link=True,  # Static object
       )
   ])

**Point cloud generation:** SceneLib can generate point clouds of objects for
perception-based policies.

Robot Config
------------

**Location:** ``protomotions/robot_configs/``

Robot configs define "what to simulate" beyond the MJCF file.

**Why not just use MJCF?**

MJCF defines the robot's physical structure, but we need additional information:

1. **Semantic body mappings**: Which bodies are feet, hands, head?
2. **Simulator-specific settings**: Different solvers need different iterations
3. **Control parameters**: PD gains, action scaling
4. **Asset paths**: MJCF for physics, USD for IsaacLab

**Key fields:**

.. code-block:: python

   @dataclass
   class RobotConfig:
       # Semantic mappings (used for contact detection, observations)
       common_naming_to_robot_body_names: Dict[str, List[str]]
       
       # Physics assets
       asset: RobotAssetConfig
       
       # Control (PD gains, action scaling)
       control: ControlConfig
       
       # Per-simulator physics settings
       simulation_params: SimulatorParams
       
       # Populated from MJCF (auto-extracted)
       kinematic_info: KinematicInfo  # Body hierarchy, joint info

**Relationship to PoseLib:**

``pose_lib.extract_kinematic_info()`` parses the MJCF and populates ``kinematic_info``.
This provides the body hierarchy needed for FK/IK and observation computation.

Simulator
---------

**Location:** ``protomotions/simulator/base_simulator/``

The simulator abstraction wraps different physics backends.

**Interface:**

.. code-block:: python

   class BaseSimulator:
       def step(self, actions: Tensor) -> None: ...
       def get_state(self) -> SimulatorState: ...
       def set_state(self, state: SimulatorState) -> None: ...
       def reset_envs(self, env_ids: Tensor) -> None: ...

All backends (IsaacGym, IsaacLab, Newton, Genesis) implement this interface,
allowing environment code to be simulator-agnostic.

**SimulatorState:**

This is the central data structure shared among data prep, simulator, and env.
See :doc:`simulator_state` for more details.

Environment
-----------

**Location:** ``protomotions/envs/``

Environments compute observations, rewards, and manage episodes.

**Structure:**

* ``BaseEnv``: Common functionality (reset, step interface)
* ``MimicEnv``: Motion imitation with tracking rewards
* ``SteeringEnv``: Pure RL locomotion control

Agent
-----

**Location:** ``protomotions/agents/``

Agents implement RL algorithms.

**Structure:**

* ``BaseAgent``: Training loop, checkpointing
* ``PPOAgent``: Proximal Policy Optimization
* ``AMPAgent``: Adversarial Motion Priors
* ``ASEAgent``: Adversarial Skill Embeddings
* ``MaskedMimicAgent``: Masked motion imitation with tracking rewards

Next Steps
----------

* :doc:`pose_lib` - MJCF parsing and FK/IK utilities
* :doc:`simulator_state` - State representation details

