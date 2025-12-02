protomotions.simulator package
===============================

Physics simulator interfaces providing unified API across different engines.

Simulator (Base Class)
----------------------

.. py:class:: Simulator(config, robot_config, terrain, device, scene_lib, visualization_markers)

   Base class for all physics simulators.
   
   Provides unified interface for IsaacGym, IsaacLab, Genesis, and Newton engines.
   Handles robot spawning, environment setup, scene management, terrain integration,
   and state management.
   
   **Key Responsibilities**:
   
   - Environment setup and robot spawning
   - State management with unified RobotState format
   - PD control and direct torque control
   - Visualization markers
   - Data conversion between simulator orderings
   
   :param config: SimulatorConfig (num_envs, physics params)
   :param robot_config: Robot morphology and control
   :param terrain: Optional terrain for complex surfaces
   :param device: PyTorch device
   :param scene_lib: Optional scene library
   :param visualization_markers: Optional markers
   
   **Key Methods**:
   
   - ``reset()`` - Reset simulation
   - ``step(actions)`` - Step physics with actions
   - ``get_robot_state()`` - Get current robot state
   - ``set_robot_state(state)`` - Set robot state
   
   **Example**::
   
       sim = IsaacGymSimulator(config, robot_config, device=device)
       sim.reset()
       for _ in range(1000):
           actions = policy(sim.robot_state)
           sim.step(actions)

IsaacGym Simulator
------------------

.. py:class:: IsaacGymSimulator(config, robot_config, terrain, device, scene_lib, visualization_markers)

   Simulator implementation using NVIDIA IsaacGym.
   
   Provides GPU-accelerated physics simulation optimized for RL training.
   Fastest simulator option for parallel environment simulation.
   
   :param config: Simulator configuration
   :param robot_config: Robot configuration
   :param terrain: Optional terrain
   :param device: PyTorch device
   :param scene_lib: Optional scene library
   :param visualization_markers: Optional markers

IsaacLab Simulator
------------------

.. py:class:: IsaacLabSimulator(config, robot_config, terrain, device, scene_lib, visualization_markers)

   Simulator implementation using NVIDIA IsaacLab (Isaac Sim).
   
   Provides GPU-accelerated physics with photorealistic rendering.
   Supports USD assets, advanced materials, and ray tracing.
   
   :param config: Simulator configuration
   :param robot_config: Robot configuration
   :param terrain: Optional terrain
   :param device: PyTorch device
   :param scene_lib: Optional scene library
   :param visualization_markers: Optional markers
   
   **Features**:
   
   - Photorealistic rendering
   - USD asset pipeline
   - Advanced materials and lighting
   - Camera sensors

Genesis Simulator
-----------------

.. py:class:: GenesisSimulator(config, robot_config, terrain, device, scene_lib, visualization_markers)

   Simulator implementation using Genesis physics engine.
   
   Modern physics engine for robotics and animation.
   
   :param config: Simulator configuration
   :param robot_config: Robot configuration
   :param terrain: Optional terrain
   :param device: PyTorch device
   :param scene_lib: Optional scene library
   :param visualization_markers: Optional markers

Newton Simulator
----------------

.. py:class:: NewtonSimulator(config, robot_config, terrain, device, scene_lib, visualization_markers)

   Simulator implementation using Newton physics engine.
   
   High-performance physics engine built on NVIDIA Warp for fast parallel simulation.
   
   :param config: Simulator configuration
   :param robot_config: Robot configuration
   :param terrain: Optional terrain
   :param device: PyTorch device
   :param scene_lib: Optional scene library
   :param visualization_markers: Optional markers
   
   **Features**:
   
   - Very fast simulation performance
   - Built on NVIDIA Warp for GPU acceleration
   - Efficient parallel environment execution
   - Modern Python-based architecture

